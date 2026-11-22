import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F

from trafficpred.models.distvol import EPS, ConvBlock, ODTransportHead, log_flow, log_last_distribution
from trafficpred.models.transport import TransportHead
from trafficpred.sinkhorn import split_flow


class VolumeUNet(nn.Module):
    def __init__(self, cin, width, code_dim):
        super().__init__()
        w = width
        self.e1 = ConvBlock(cin, w)
        self.e2 = ConvBlock(w, 2 * w)
        self.e3 = ConvBlock(2 * w, 4 * w)
        self.mid = nn.Sequential(*[
            nn.Sequential(nn.Conv2d(4 * w, 4 * w, 3, padding=d, dilation=d), nn.GELU())
            for d in (1, 2, 4)
        ])
        self.d2 = ConvBlock(4 * w + 2 * w, 2 * w)
        self.d1 = ConvBlock(2 * w + w, w)
        self.proj = nn.Sequential(nn.Linear(4 * w + 2 * w, code_dim), nn.GELU())

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(F.avg_pool2d(s1, 2))
        s3 = self.e3(F.avg_pool2d(s2, 2))
        bottleneck = s3 + self.mid(s3)
        u = self.d2(torch.cat([F.interpolate(bottleneck, scale_factor=2), s2], 1))
        u = self.d1(torch.cat([F.interpolate(u, scale_factor=2), s1], 1))
        pooled = torch.cat([bottleneck.mean(dim=(-2, -1)), u.mean(dim=(-2, -1)), u.amax(dim=(-2, -1))], -1)
        return self.proj(pooled)


def sinkhorn_attention(attn, x, iters):
    """Self-attention with nn.MultiheadAttention `attn`'s weights, the softmax replaced by `iters`
    Sinkhorn normalisations in log space (rows, columns, rows, ...; 1 = the usual softmax), so the
    attention matrix is close to doubly stochastic (Sinkformers, Sander et al., AISTATS 2022)."""
    B, N, D = x.shape
    h = attn.num_heads
    q, k, v = F.linear(x, attn.in_proj_weight, attn.in_proj_bias).chunk(3, dim=-1)
    q, k, v = (t.reshape(B, N, h, D // h).transpose(1, 2) for t in (q, k, v))
    logits = q @ k.transpose(-2, -1) / (D // h) ** 0.5
    for i in range(iters):
        logits = logits - logits.logsumexp(dim=-1 if i % 2 == 0 else -2, keepdim=True)
    weights = F.dropout(logits.exp(), attn.dropout, attn.training)
    out = (weights @ v).transpose(1, 2).reshape(B, N, D)
    return attn.out_proj(out)


class AdaLNBlock(nn.Module):
    grad_checkpoint = False   # train.py --grad-checkpoint: recompute the block in the backward pass

    def __init__(self, dim, heads, mlp_dim, cond_dim, drop, sinkformer=0):
        super().__init__()
        self.sinkformer = sinkformer
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(mlp_dim, dim), nn.Dropout(drop))
        self.ada = nn.Linear(cond_dim, 6 * dim)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x, cond):
        if self.grad_checkpoint and self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(self._forward, x, cond, use_reentrant=False)
        return self._forward(x, cond)

    def _forward(self, x, cond):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(cond)[:, None].chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale1) + shift1
        if self.sinkformer:
            a = sinkhorn_attention(self.attn, h, self.sinkformer)
        else:
            a = self.attn(h, h, h, need_weights=False)[0]
        x = x + (1 + gate1) * a
        h = self.norm2(x) * (1 + scale2) + shift2
        return x + (1 + gate2) * self.mlp(h)


class DistTransformer(nn.Module):
    def __init__(self, cin, cout, grid, patch, dim, depth, heads, mlp_dim, cond_dim, drop, sinkformer=0):
        super().__init__()
        self.patch, self.cout = patch, cout
        self.embed = nn.Conv2d(cin, dim, patch, stride=patch)
        tokens = (grid[0] // patch) * (grid[1] // patch)
        self.pos = nn.Parameter(torch.zeros(1, tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(AdaLNBlock(dim, heads, mlp_dim, cond_dim, drop, sinkformer) for _ in range(depth))
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.final_ada = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.final_ada.weight)
        nn.init.zeros_(self.final_ada.bias)
        self.head = nn.Linear(dim, patch * patch * cout)

    def forward(self, x, cond):
        B, _, H, W = x.shape
        h, w = H // self.patch, W // self.patch
        tokens = self.embed(x).flatten(2).transpose(1, 2) + self.pos
        for block in self.blocks:
            tokens = block(tokens, cond)
        shift, scale = self.final_ada(cond)[:, None].chunk(2, dim=-1)
        out = self.head(self.norm(tokens) * (1 + scale) + shift)
        out = out.transpose(1, 2).reshape(B, -1, h, w)
        return F.pixel_shuffle(out, self.patch)


class SpaceTimeTransformer(nn.Module):
    """The distribution transformer with time as tokens: every input frame is its own set of patch
    tokens (one shared patch embedding, a learned position per patch and a learned embedding per
    frame, i.e. per lag: t-1 .. t-k, a day back, a week back, ...), followed by T_out learned frames
    of target tokens. Blocks alternate attention across frames at each patch (temporal: a target
    token picks which earlier frames to draw on, like STDN's periodically shifted attention) and
    across patches within each frame (spatial). The head reads the target tokens only."""

    def __init__(self, cin_frame, cout_step, T_in, T_out, grid, patch, dim, depth, heads, mlp_dim,
                 cond_dim, drop, sinkformer=0):
        super().__init__()
        self.patch, self.T_in, self.T_out, self.cout_step = patch, T_in, T_out, cout_step
        self.embed = nn.Conv2d(cin_frame, dim, patch, stride=patch)
        tokens = (grid[0] // patch) * (grid[1] // patch)
        self.pos = nn.Parameter(torch.zeros(1, 1, tokens, dim))
        self.frame = nn.Parameter(torch.zeros(1, T_in + T_out, 1, dim))   # lag / target-step embedding
        self.target = nn.Parameter(torch.zeros(1, T_out, 1, dim))
        for p in (self.pos, self.frame, self.target):
            nn.init.trunc_normal_(p, std=0.02)
        self.blocks = nn.ModuleList(AdaLNBlock(dim, heads, mlp_dim, cond_dim, drop, sinkformer) for _ in range(depth))
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.final_ada = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.final_ada.weight)
        nn.init.zeros_(self.final_ada.bias)
        self.head = nn.Linear(dim, patch * patch * cout_step)

    def forward(self, x, cond):
        """x: (B, T_in, cin_frame, H, W) -> (B, T_out * cout_step, H, W), step-major like DistTransformer."""
        B, T, Cf, H, W = x.shape
        h, w = H // self.patch, W // self.patch
        tokens = self.embed(x.reshape(B * T, Cf, H, W)).flatten(2).transpose(1, 2).reshape(B, T, h * w, -1)
        target = self.target.expand(B, -1, h * w, -1)
        z = torch.cat([tokens, target], dim=1) + self.pos + self.frame          # (B, F, N, D)
        F_, N, D = z.shape[1:]
        for i, block in enumerate(self.blocks):
            if i % 2 == 0:   # temporal: each patch's sequence of frames
                seq = z.transpose(1, 2).reshape(B * N, F_, D)
                z = block(seq, cond.repeat_interleave(N, dim=0)).reshape(B, N, F_, D).transpose(1, 2)
            else:            # spatial: each frame's patches
                z = block(z.reshape(B * F_, N, D), cond.repeat_interleave(F_, dim=0)).reshape(B, F_, N, D)
        out = z[:, T:]                                                           # target frames
        shift, scale = self.final_ada(cond)[:, None, None].chunk(2, dim=-1)
        out = self.head(self.norm(out) * (1 + scale) + shift)                    # (B, T_out, N, p*p*c)
        out = out.transpose(2, 3).reshape(B * self.T_out, -1, h, w)
        return F.pixel_shuffle(out, self.patch).reshape(B, self.T_out * self.cout_step, H, W)


class VolumeTransformer(nn.Module):
    """Volume head as a transformer over time: one token per input frame (a small conv encoder of the
    frame's distribution / log-flow maps, plus its log totals), a learned embedding per lag (t-1 ..
    t-k, a day back, a week back, ...), then T_out learned target tokens. The target tokens read the
    change in log volume per step and channel; every token's mean is the conditioning code for the
    distribution head."""

    def __init__(self, cin_frame, C, T_in, T_out, width, dim, depth, heads, code_dim, drop):
        super().__init__()
        self.T_out, self.C = T_out, C
        self.frame_enc = nn.Sequential(
            nn.Conv2d(cin_frame, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1), nn.GELU())
        self.frame_proj = nn.Linear(2 * width + C, dim)
        self.lag = nn.Parameter(torch.zeros(1, T_in + T_out, dim))
        self.target = nn.Parameter(torch.zeros(1, T_out, dim))
        for p in (self.lag, self.target):
            nn.init.trunc_normal_(p, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, drop, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.delta = nn.Linear(dim, C)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        self.code = nn.Sequential(nn.Linear(dim, code_dim), nn.GELU())

    def forward(self, frames, log_n):
        """frames: (B, T, cin_frame, H, W); log_n: (B, T, C) -> delta (B, T_out, C), code (B, code_dim)."""
        B, T = frames.shape[:2]
        f = self.frame_enc(frames.flatten(0, 1))
        f = torch.cat([f.mean(dim=(-2, -1)), f.amax(dim=(-2, -1))], -1).reshape(B, T, -1)
        tokens = self.frame_proj(torch.cat([f, log_n], -1))
        z = torch.cat([tokens, self.target.expand(B, -1, -1)], 1) + self.lag
        z = self.norm(self.encoder(z))
        return self.delta(z[:, T:]), self.code(z.mean(1))


def grid_hops(H, W):
    """Hop distance between cells on the 8-neighbour grid graph (Chebyshev distance), (HW, HW)."""
    i, j = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    i, j = i.flatten(), j.flatten()
    return torch.maximum((i[:, None] - i[None]).abs(), (j[:, None] - j[None]).abs())


class GraphBlock(nn.Module):
    """AdaLN transformer block over graph nodes; attention carries a learned per-head bias for each
    hop distance (Graphormer's spatial encoding) and, with a radius, only reaches nodes that close."""

    def __init__(self, dim, heads, mlp_dim, cond_dim, drop, buckets):
        super().__init__()
        self.heads, self.drop = heads, drop
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.hop_bias = nn.Parameter(torch.zeros(heads, buckets))
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(mlp_dim, dim), nn.Dropout(drop))
        self.ada = nn.Linear(cond_dim, 6 * dim)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x, cond, hops, mask):
        B, N, D = x.shape
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(cond)[:, None].chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale1) + shift1
        q, k, v = (t.reshape(B, N, self.heads, -1).transpose(1, 2) for t in self.qkv(h).chunk(3, -1))
        bias = (self.hop_bias[:, hops] + mask)[None].to(q.dtype)                # (1, heads, N, N)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=self.drop if self.training else 0.0)
        x = x + (1 + gate1) * self.out(a.transpose(1, 2).reshape(B, N, D))
        h = self.norm2(x) * (1 + scale2) + shift2
        return x + (1 + gate2) * self.mlp(h)


class GraphTransformer(nn.Module):
    """Distribution head as a graph transformer: every grid cell is a node (no patching) with its
    stacked input features, a learned node embedding, and attention biased by hop distance on the
    8-neighbour grid graph (radius R > 0: attention only within R hops). Outputs per-node logits."""

    def __init__(self, cin, cout, H, W, dim, depth, heads, mlp_dim, cond_dim, drop, radius=0, max_hops=8):
        super().__init__()
        hops = grid_hops(H, W)
        self.register_buffer("hops", hops.clamp_max(max_hops), persistent=False)
        mask = torch.zeros(hops.shape)
        if radius:
            mask[hops > radius] = float("-inf")
        self.register_buffer("mask", mask, persistent=False)
        self.H, self.W = H, W
        self.embed = nn.Linear(cin, dim)
        self.node = nn.Parameter(torch.zeros(1, H * W, dim))
        nn.init.trunc_normal_(self.node, std=0.02)
        self.blocks = nn.ModuleList(GraphBlock(dim, heads, mlp_dim, cond_dim, drop, max_hops + 1)
                                    for _ in range(depth))
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.final_ada = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.final_ada.weight)
        nn.init.zeros_(self.final_ada.bias)
        self.head = nn.Linear(dim, cout)

    def forward(self, x, cond):
        """x: (B, cin, H, W) -> (B, cout, H, W)."""
        B = x.shape[0]
        z = self.embed(x[..., :self.H, :self.W].flatten(2).transpose(1, 2)) + self.node
        for block in self.blocks:
            z = block(z, cond, self.hops, self.mask)
        shift, scale = self.final_ada(cond)[:, None].chunk(2, dim=-1)
        out = self.head(self.norm(z) * (1 + scale) + shift)                    # (B, N, cout)
        return out.transpose(1, 2).reshape(B, -1, self.H, self.W)


class DistVolFormer(nn.Module):
    def __init__(self, T_in=4, T_out=4, C=2, H=32, W=32, width=64, patch=2, dim=192, depth=6,
                 heads=6, mlp_dim=768, drop=0.1, code_dim=128, dist_residual=False,
                 transport=0, n_periodic=0, sinkformer=0, time_tokens=False,
                 vol_transformer=False, vol_dim=128, vol_depth=3, graph=False, graph_radius=0,
                 count_scale=0.0, count_offset=0.0, direct=False, aux=0, od_head=None, od_slot_scale=0.0):
        """aux: extra input maps per frame after the C flow channels (--od), fed as log-flow maps next to
        the flow's (in every layout: stacked, time tokens, graph, volume transformer); not predicted.

        od_head: the --od-head kernel file. End(t+1) = the factorised part + the trips under way at t moved
        by a transport plan (distvol.ODTransportHead); the destination potential is one more output
        channel of the distribution head, zero at the start. od_slot_scale > 0 (--od-slot): the last
        extra map is the frame's time-of-day slot (scaled by 1 / od_slot_scale), read only by the od head
        for its per-slot prior; aux counts the maps the encoders get.

        direct: the ablation without the factorisation -- the same inputs and backbone (the volume
        encoder still conditions the distribution transformer), but the head outputs the flow itself
        instead of softmax logits, and there is no total-volume head."""
        super().__init__()
        self.count_scale, self.count_offset = count_scale, count_offset
        self.od_slot_scale = od_slot_scale
        if direct and (transport or dist_residual):
            raise ValueError("direct has no distribution to transport / add a residual to")
        self.direct = direct
        if time_tokens and transport:
            raise ValueError("time_tokens doesn't support the transport head")
        if graph and (time_tokens or transport or sinkformer):
            raise ValueError("the graph distribution head doesn't combine with time_tokens / transport / sinkformer")
        self.time_tokens, self.vol_transformer, self.graph = time_tokens, vol_transformer, graph
        if od_head is not None and (aux < 1 or T_out != 1 or C != 2 or direct or transport):
            raise ValueError("od_head needs --od, --t-out 1, the Start / End channels, and no direct / transport")
        g_out = 1 if od_head is not None else 0   # the potential channel, after the logits
        self.T_in, self.T_out, self.C, self.H, self.W = T_in, T_out, C, H, W
        self.aux = aux
        k_in, k_out = T_in * C, T_out * C
        a_in = T_in * aux
        self.pad_h, self.pad_w = -H % 4, -W % 4
        self.pad_h += -(H + self.pad_h) % patch
        self.pad_w += -(W + self.pad_w) % patch
        grid = (H + self.pad_h, W + self.pad_w)
        if vol_transformer:
            self.vol_enc = VolumeTransformer(3 * C + aux, C, T_in, T_out, width, vol_dim, vol_depth,
                                             max(1, vol_dim // 32), code_dim, drop)
        else:
            self.vol_enc = VolumeUNet(3 * k_in + a_in, width, code_dim)
        self.transport = TransportHead(T_out, C, transport, n_periodic) if transport else None
        if graph:
            self.dist_enc = GraphTransformer(2 * k_in + a_in, k_out + g_out, H, W, dim, depth, heads, mlp_dim,
                                             code_dim, drop, graph_radius)
        elif time_tokens:
            self.dist_enc = SpaceTimeTransformer(2 * C + aux, C + g_out, T_in, T_out, grid, patch, dim, depth,
                                                 heads, mlp_dim, code_dim, drop, sinkformer)
        else:
            self.dist_enc = DistTransformer(2 * k_in + a_in, self.transport.channels if transport else k_out + g_out,
                                            grid, patch,
                                            dim, depth, heads, mlp_dim, code_dim, drop, sinkformer)
        self.dist_residual = dist_residual
        if transport:
            p2 = patch * patch   # head outputs (patch*patch*cout) are pixel-shuffled: channel-major
            self.transport.init_head(self.dist_enc.head.weight, self.dist_enc.head.bias.view(-1, p2)[:, 0])
            with torch.no_grad():
                self.dist_enc.head.bias.view(-1, p2)[:] = self.dist_enc.head.bias.view(-1, p2)[:, :1]
        elif dist_residual:   # starts as "the last frame's distribution", learns the change
            nn.init.zeros_(self.dist_enc.head.weight)
            nn.init.zeros_(self.dist_enc.head.bias)
        self.od_head = None
        if od_head is not None:
            p2 = 1 if graph else patch * patch   # pixel-shuffled heads: channel c is rows c*p2 .. (c+1)*p2
            with torch.no_grad():
                self.dist_enc.head.weight[k_out * p2:(k_out + 1) * p2].zero_()
                self.dist_enc.head.bias[k_out * p2:(k_out + 1) * p2].zero_()
            self.od_head = ODTransportHead(np.load(od_head), None, H, W)
        self.vol_head = None if (vol_transformer or direct) else nn.Sequential(nn.Linear(code_dim + k_in, 256), nn.GELU(), nn.Linear(256, k_out))
        if self.vol_head is not None:
            nn.init.zeros_(self.vol_head[-1].weight)
            nn.init.zeros_(self.vol_head[-1].bias)

    def forward(self, x, return_parts=False):
        B, T, _, H, W = x.shape
        C, A = self.C, self.aux
        slot = None
        if self.od_slot_scale:
            slot = (x[:, -1, -1, 0, 0] * self.od_slot_scale).round().long()
            x = x[:, :, :-1]
        x, extra_raw = x[:, :, :C], x[:, :, C:]
        p_in, n_in = split_flow(x.clamp_min(0))
        log_n = torch.log(n_in + EPS).reshape(B, T * C)
        features = torch.cat([
            (p_in * H * W).reshape(B, T * C, H, W),
            log_flow(x, self.count_scale, self.count_offset).reshape(B, T * C, H, W),
        ], dim=1)
        features = F.pad(features, (0, self.pad_w, 0, self.pad_h))
        Hp, Wp = features.shape[-2:]
        extra = F.pad(log_flow(extra_raw, self.count_scale, self.count_offset).reshape(B, T * A, H, W),
                      (0, self.pad_w, 0, self.pad_h)) if A else None
        volume_maps = log_n[..., None, None].expand(-1, -1, Hp, Wp)
        if self.vol_transformer:   # per frame: distribution, log flow, log totals (, extra) as (3C + A, H, W)
            frames = torch.cat([features, volume_maps], 1).reshape(B, 3, T, C, Hp, Wp).transpose(1, 2)
            frames = frames.reshape(B, T, 3 * C, Hp, Wp)
            if A:
                frames = torch.cat([frames, extra.reshape(B, T, A, Hp, Wp)], 2)
            delta, code = self.vol_enc(frames, log_n.reshape(B, T, C))
        else:
            code = self.vol_enc(torch.cat([features, volume_maps] + ([extra] if A else []), dim=1))
        if self.time_tokens:   # the same features, one (2C + A, H, W) stack per frame
            features = features.reshape(B, 2, T, C, Hp, Wp).transpose(1, 2).reshape(B, T, 2 * C, Hp, Wp)
            if A:
                features = torch.cat([features, extra.reshape(B, T, A, Hp, Wp)], 2)
        elif A:
            features = torch.cat([features, extra], 1)

        if self.direct:
            x_hat = self.dist_enc(features, code)[..., :H, :W].reshape(B, self.T_out, C, H, W)
            return (x_hat, *split_flow(x_hat)) if return_parts else x_hat
        if self.transport is not None:
            p_hat = self.transport(self.dist_enc(features, code)[..., :H, :W], p_in, H, W)
        else:
            out = self.dist_enc(features, code)[..., :H, :W]
            if self.od_head is not None:
                out, g = out[:, :self.T_out * C], out[:, self.T_out * C]
            logits = out.reshape(B, self.T_out, C, H * W)
            if self.dist_residual:
                logits = logits + log_last_distribution(p_in)
            p_hat = logits.softmax(dim=-1).reshape(B, self.T_out, C, H, W)

        if not self.vol_transformer:
            delta = self.vol_head(torch.cat([code, log_n], dim=-1)).reshape(B, self.T_out, C)
        n_hat = n_in[:, -1:, :] * torch.exp(delta.clamp(-5, 5))

        x_hat = p_hat * n_hat[..., None, None]
        if self.od_head is not None:   # End(t+1) = factorised part (above) + arrivals of trips under way
            arrivals = self.od_head(None, extra_raw[:, -1, 0], g=g, slot=slot)
            x_hat = torch.stack([x_hat[:, :, 0], x_hat[:, :, 1] + arrivals[:, None]], 2)
            if return_parts:
                return (x_hat, *split_flow(x_hat))
        if return_parts:
            return x_hat, p_hat, n_hat
        return x_hat
