import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from trafficpred.models.transport import TransportHead
from trafficpred.sinkhorn import split_flow

def log_flow(x, count_scale=0.0, count_offset=0.0):
    """The log-flow input feature. count_scale = 0: log1p of the scaled data, as first trained (on
    0-1 scaled data that is nearly linear: log1p(0.01) = 0.00995). count_scale = span, count_offset
    = lo: log1p of the raw counts, which compresses the heavy-tailed flow as intended."""
    x = x.clamp_min(0)
    return torch.log1p((x * count_scale + count_offset).clamp_min(0) if count_scale else x)


RESIDUAL_FLOOR = 1e-6   # keeps log P finite where the last frame had no traffic
EPS = 1e-6


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout),
        )
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        return F.gelu(self.net(x) + self.skip(x))


class FiLM(nn.Module):
    def __init__(self, cond_dim, ch):
        super().__init__()
        self.lin = nn.Linear(cond_dim, 2 * ch)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, h, cond):
        gamma, beta = self.lin(cond).chunk(2, dim=-1)
        return h * (1 + gamma[..., None, None]) + beta[..., None, None]


class MultiscaleEncoder(nn.Module):
    def __init__(self, cin, width, cond_dim):
        super().__init__()
        w = width
        self.e1 = ConvBlock(cin, w)
        self.e2 = ConvBlock(w, 2 * w)
        self.e3 = ConvBlock(2 * w, 4 * w)
        self.mid = nn.Sequential(*[
            nn.Sequential(nn.Conv2d(4 * w, 4 * w, 3, padding=d, dilation=d), nn.GELU())
            for d in (1, 2, 4)
        ])
        self.film = FiLM(cond_dim, 4 * w)
        self.d2 = ConvBlock(4 * w + 2 * w, 2 * w)
        self.d1 = ConvBlock(2 * w + w, w)
        self.out_ch, self.mid_ch = w, 4 * w

    def forward(self, x, cond):
        s1 = self.e1(x)
        s2 = self.e2(F.avg_pool2d(s1, 2))
        s3 = self.e3(F.avg_pool2d(s2, 2))
        bottleneck = self.film(s3 + self.mid(s3), cond)
        u = self.d2(torch.cat([F.interpolate(bottleneck, scale_factor=2), s2], 1))
        u = self.d1(torch.cat([F.interpolate(u, scale_factor=2), s1], 1))
        return u, bottleneck


class FrameEncoder(nn.Module):
    """--frame-encoder: encode every input frame on its own before the U-Net, then combine them over time.
    A shared conv stack maps each frame's maps (shape, log-flow, aux) to `width` features per cell; a learned
    embedding per frame position (lag, or a day / week back) is added; `layers` transformer layers attend
    across the frames at each cell, with one learned target token whose output is the cell's feature map.
    Replaces the channel stack (all frames' maps concatenated, i.e. time mixed in by the first conv)."""

    def __init__(self, cin_frame, T, width, layers=2, heads=4):
        super().__init__()
        self.conv = nn.Sequential(ConvBlock(cin_frame, width), nn.GELU(), ConvBlock(width, width))
        self.frame = nn.Parameter(torch.zeros(1, T + 1, width))
        self.target = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.trunc_normal_(self.frame, std=0.02)
        nn.init.trunc_normal_(self.target, std=0.02)
        layer = nn.TransformerEncoderLayer(width, heads, 2 * width, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)

    def forward(self, frames):
        """frames: (B, T, cin_frame, H, W) -> (B, width, H, W)."""
        B, T, Cf, H, W = frames.shape
        h = self.conv(frames.reshape(B * T, Cf, H, W))                       # (B*T, w, H, W)
        w = h.shape[1]
        seq = h.reshape(B, T, w, H * W).permute(0, 3, 1, 2).reshape(B * H * W, T, w)
        seq = torch.cat([seq, self.target.expand(B * H * W, -1, -1)], 1) + self.frame
        out = self.norm(self.temporal(seq)[:, -1])                           # the target token
        return out.reshape(B, H * W, w).transpose(1, 2).reshape(B, w, H, W)


def log_last_distribution(p_in):
    """log P of the last input frame, (B, 1, C, H*W): the persistence prior for residual logits."""
    B, _, C, H, W = p_in.shape
    return torch.log(p_in[:, -1:].reshape(B, 1, C, H * W) + RESIDUAL_FLOOR)


EXT_EMBED = 8   # --ext-input: features per frame after the shared embedding


def ext_maps(embed, ext, T, T_out, H, W):
    """--ext-input: every frame's external features (B, T + T_out, D: the input frames', then the targets')
    through the shared embedding, as constant maps (B, (T + T_out) * EXT_EMBED, H, W) to stack with the
    flow channels; also returned per frame (B, T + T_out, EXT_EMBED)."""
    if ext is None:
        raise ValueError("built with ext_input; pass ext")
    if ext.shape[1] != T + T_out:
        raise ValueError(f"ext has {ext.shape[1]} rows for {T} input frames + {T_out} targets")
    e = embed(ext)                                                                 # (B, T + T_out, E)
    B = e.shape[0]
    return e.reshape(B, -1, 1, 1).expand(-1, -1, H, W), e


class ODTransportHead(nn.Module):
    """End(t+1) gets the arrivals of the trips under way at t, moved by a transport plan.

    r_i = trips under way from origin i at the last input frame (the first --od map; known at t).
    plan_ij = r_i * softmax_j(log K_ij + D_ij + g_j): every row carries exactly r_i (the known mass is
    conserved); K = the train split's all-day destination kernel (fixed prior), D = a learned
    correction (weight decay pulls it back to the prior), g = a destination potential predicted per
    sample from the encoder features: the dual variable of a singly constrained transport / gravity
    model, which lets the plan follow the time of day and conditions. arrivals_j = sum_i plan_ij.
    D and g start at 0, so the head starts as the historical kernel. A (S, N, N) kernel is one per
    time-of-day slot (--od-slot): each sample starts from its own slot's kernel, D stays shared."""

    def __init__(self, kernel, width, H, W):
        """width: features to read the potential from with a 1x1 conv (DistVol); None when the caller
        predicts the potential itself and passes it as g (DistVolFormer's distribution head)."""
        super().__init__()
        N = H * W
        k = torch.as_tensor(kernel, dtype=torch.float32)
        k = k.reshape(-1, N, N) if k.ndim == 3 else k.reshape(N, N)
        self.register_buffer("log_prior", torch.log(k.clamp_min(1e-8)))
        self.correction = nn.Parameter(torch.zeros(N, N))
        self.potential = None
        if width:
            self.potential = nn.Conv2d(width, 1, 1)
            nn.init.zeros_(self.potential.weight)
            nn.init.zeros_(self.potential.bias)
        self.H, self.W = H, W

    def forward(self, u, in_progress, g=None, slot=None):
        """u: (B, width, >=H, >=W) features (or None with g: (B, H, W) potentials); in_progress:
        (B, H, W); slot: (B,) time-of-day slots, with a per-slot kernel -> arrivals (B, H, W)."""
        B = in_progress.shape[0]
        if g is None:
            g = self.potential(u)[:, 0]
        g = g[:, :self.H, :self.W].reshape(B, 1, -1)
        prior = self.log_prior
        if prior.ndim == 3:
            if slot is None:
                raise ValueError("a per-slot kernel needs each sample's slot (--od-slot)")
            prior = prior[slot]
        plan = (prior + self.correction + g).softmax(-1)                       # (B, N, N), rows sum to 1
        arrivals = torch.bmm(in_progress.reshape(B, 1, -1).clamp_min(0), plan)  # (B, 1, N)
        return arrivals.reshape(B, self.H, self.W)


class DistVolNet(nn.Module):
    def __init__(self, T_in=4, T_out=4, C=2, H=32, W=32, width=64, dist_residual=False,
                 transport=0, n_periodic=0, count_scale=0.0, count_offset=0.0, aux=0, od_head=None,
                 od_slot_scale=0.0, frame_encoder=False, ext_dim=0, ext_input=False, ext_volume=False):
        """aux: extra input maps per frame after the C flow channels (--od). They go in as log-flow maps
        next to the flow's, and their log totals next to the flow's totals; they are not predicted.
        od_slot_scale > 0 (--od-slot): the last of them is the frame's time-of-day slot, scaled by
        1 / od_slot_scale, read only by the od head; aux counts the maps the encoder gets."""
        super().__init__()
        self.od_slot_scale = od_slot_scale
        self.count_scale, self.count_offset = count_scale, count_offset
        self.T_in, self.T_out, self.C, self.H, self.W = T_in, T_out, C, H, W
        self.aux = aux
        self.dist_residual = dist_residual
        k_in, k_out = T_in * C, T_out * C
        self.vol_enc = nn.Sequential(nn.Linear(k_in + T_in * aux, 128), nn.GELU(), nn.Linear(128, 128))
        self.ext = None   # --external: the target steps' features, added to the volume code (zero at the start)
        self.ext_embed = None   # --ext-input: every frame's features as encoder input maps instead
        self.ext_volume = bool(ext_dim and ext_volume)   # --ext-volume: the ext term goes to the volume head only
        if ext_input and ext_volume:
            raise ValueError("--ext-input and --ext-volume are alternatives")
        if ext_dim and ext_input:
            if frame_encoder:
                raise ValueError("--ext-input isn't wired into the frame encoder")
            self.ext_embed = nn.Linear(ext_dim, EXT_EMBED)
        elif ext_dim:
            self.ext = nn.Sequential(nn.Linear(T_out * ext_dim, 64), nn.GELU(), nn.Linear(64, 128))
            nn.init.zeros_(self.ext[-1].weight)
            nn.init.zeros_(self.ext[-1].bias)
        e_in = (T_in + T_out) * EXT_EMBED if self.ext_embed is not None else 0
        self.frame_encoder = FrameEncoder(2 * C + aux, T_in, width) if frame_encoder else None
        self.encoder = MultiscaleEncoder(width if frame_encoder else 2 * k_in + T_in * aux + e_in, width, cond_dim=128)
        self.transport = TransportHead(T_out, C, transport, n_periodic) if transport else None
        self.dist_head = nn.Conv2d(width, self.transport.channels if transport else k_out, 1)
        self.od_head = None
        if od_head is not None:   # the --od-head kernel file; End = within-slot part + transported trips
            if aux < 1 or T_out != 1 or C != 2:
                raise ValueError("od_head needs --od, --t-out 1 and the Start / End channels")
            self.od_head = ODTransportHead(np.load(od_head), width, H, W)
        if transport:
            self.transport.init_head(self.dist_head.weight, self.dist_head.bias)
        elif dist_residual:   # starts as "the last frame's distribution", learns the change
            nn.init.zeros_(self.dist_head.weight)
            nn.init.zeros_(self.dist_head.bias)
        self.vol_head = nn.Sequential(
            nn.Linear(self.encoder.mid_ch + 128, 256), nn.GELU(), nn.Linear(256, k_out)
        )
        nn.init.zeros_(self.vol_head[-1].weight)
        nn.init.zeros_(self.vol_head[-1].bias)

    def forward(self, x, return_parts=False, ext=None):
        B, T, _, H, W = x.shape
        C = self.C
        slot = None
        if self.od_slot_scale:
            slot = (x[:, -1, -1, 0, 0] * self.od_slot_scale).round().long()
            x = x[:, :, :-1]
        x, extra = x[:, :, :C], x[:, :, C:]
        p_in, n_in = split_flow(x.clamp_min(0))
        per_frame = [p_in * H * W, log_flow(x, self.count_scale, self.count_offset)]   # (B, T, C, H, W) each
        log_totals = [torch.log(n_in + EPS).reshape(B, T * C)]
        if self.aux:
            per_frame.append(log_flow(extra, self.count_scale, self.count_offset))
            log_totals.append(torch.log(extra.clamp_min(0).sum((-2, -1)) + EPS).reshape(B, T * self.aux))
        per_frame = torch.cat(per_frame, dim=2)                                       # (B, T, 2C + aux, H, W)
        if self.frame_encoder is not None:
            features = self.frame_encoder(per_frame)
        else:   # the channel stack, in the original order (all frames' shapes, then log-flows, then aux)
            features = torch.cat([per_frame[:, :, :C].reshape(B, T * C, H, W),
                                  per_frame[:, :, C:2 * C].reshape(B, T * C, H, W)]
                                 + ([per_frame[:, :, 2 * C:].reshape(B, T * self.aux, H, W)] if self.aux else []), 1)
        if self.ext_embed is not None:
            features = torch.cat([features, ext_maps(self.ext_embed, ext, T, self.T_out, H, W)[0]], 1)
        cond = self.vol_enc(torch.cat(log_totals, -1))
        vol_cond = cond
        if self.ext is not None:
            if ext is None:
                raise ValueError("DistVolNet was built with ext_dim > 0; pass ext")
            vol_cond = cond + self.ext(ext[:, -self.T_out:].reshape(B, -1))
            if not self.ext_volume:   # conditioning: the encoder's FiLM sees it too
                cond = vol_cond
        features = F.pad(features, (0, -W % 4, 0, -H % 4))
        u, bottleneck = self.encoder(features, cond)

        if self.transport is not None:
            p_hat = self.transport(self.dist_head(u)[..., :H, :W], p_in, H, W)
        else:
            logits = self.dist_head(u)[..., :H, :W].reshape(B, self.T_out, C, H * W)
            if self.dist_residual:
                logits = logits + log_last_distribution(p_in)
            p_hat = logits.softmax(dim=-1).reshape(B, self.T_out, C, H, W)

        pooled = bottleneck.mean(dim=(-2, -1))
        delta = self.vol_head(torch.cat([pooled, vol_cond], dim=-1)).reshape(B, self.T_out, C)
        n_hat = n_in[:, -1:, :] * torch.exp(delta.clamp(-5, 5))

        x_hat = p_hat * n_hat[..., None, None]
        if self.od_head is not None:   # End(t+1) = within-slot part (above) + arrivals of trips under way
            arrivals = self.od_head(u, extra[:, -1, 0], slot=slot)
            x_hat = torch.stack([x_hat[:, :, 0], x_hat[:, :, 1] + arrivals[:, None]], 2)
            if return_parts:
                return (x_hat, *split_flow(x_hat))
        if return_parts:
            return x_hat, p_hat, n_hat
        return x_hat
