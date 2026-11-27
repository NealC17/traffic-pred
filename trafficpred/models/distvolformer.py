import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F

from trafficpred.models.distvol import EPS, EXT_EMBED, ConvBlock, ODTransportHead, ext_maps, log_flow, log_last_distribution
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


class ModeMixer(nn.Module):
    """Low-rank spatial mixing in place of self-attention (--spatial-modes r): the N tokens are read into r
    learned spatial modes, Z = V^T h (r x dim), and written back, U (Z W) (N x dim), so every token reaches
    every other one through r shared global patterns instead of an N x N attention matrix. The modes are
    fixed after training (experiment A of the mode study).

    gate (--mode-gate, experiment B): each mode's contents are scaled by g = 1 + tanh(G(Z)) before they are
    written back, G = a linear map within each mode (across features) + one across modes (r x r), so the
    input turns modes up or down and modes act on each other. G starts at zero: B starts as A."""

    def __init__(self, tokens, dim, rank, drop, gate=False):
        super().__init__()
        self.V = nn.Parameter(torch.randn(tokens, rank) / tokens ** 0.5)
        self.U = nn.Parameter(torch.randn(tokens, rank) / rank ** 0.5)
        self.mix = nn.Linear(dim, dim)
        self.drop = nn.Dropout(drop)
        self.gate_feat = self.gate_mode = None
        if gate:
            self.gate_norm = nn.LayerNorm(dim, elementwise_affine=False)
            self.gate_feat = nn.Linear(dim, dim)
            self.gate_mode = nn.Linear(rank, rank, bias=False)
            for p in (self.gate_feat.weight, self.gate_feat.bias, self.gate_mode.weight):
                nn.init.zeros_(p)

    def forward(self, h):                       # (B, N, dim)
        z = torch.einsum("nr,bnd->brd", self.V, h)
        if self.gate_feat is not None:
            zn = self.gate_norm(z)
            g = 1 + torch.tanh(self.gate_feat(zn) + self.gate_mode(zn.transpose(1, 2)).transpose(1, 2))
            z = g * z
        return self.drop(torch.einsum("nr,brd->bnd", self.U, self.mix(z)))


class LatentTransport(nn.Module):
    """Spatial mixing by entropic optimal transport with a learned cost (--latent-transport K), in place of
    self-attention. Every token i is a unit of mass at phi(z_i) in a learned "now" space; every token j asks
    for one unit at psi(z_j) in a learned "next" space. The cost of moving i to j has two learned parts, and
    no hand-made ground cost:
      - state: ||phi(z_i) - psi(z_j)||, the unsquared distance. (A squared distance would reduce to the dot
        product under Sinkhorn's row / column scalings, i.e. Sinkformer attention; this one doesn't.)
      - topology: softplus(w) ||e_i - e_j||, e_i a learned position of token i (a sensor or cell) in a
        latent space, so the model finds how far apart the sensors are for moving mass between them.
    The plan is K log-domain Sinkhorn steps on -cost / eps (eps learned per head, balanced uniform marginals),
    ending on the destinations, and token j receives the barycentric transport of the values: sum_i P_ij v_i.

    n_src (--time-transport, the temporal blocks): only the first n_src tokens send mass (the input frames),
    every token receives it (input and future frames), so the future tokens are made of transported input
    tokens; the topology is then over the lags / target steps. Each destination gets 1 unit, so each source
    sends M / n_src: every input frame must be shipped somewhere, which attention needn't do."""

    def __init__(self, tokens, dim, heads, drop, iters, topo_dim=16, n_src=None, topology=True, unbalanced=0.0,
                 learned_mass=False, groups=1):
        """Variants (the temporal --tt-* flags; the defaults are the original layer):
          topology=False (--tt-no-topology): the cost is the state distance alone.
          unbalanced rho > 0 (--tt-unbalanced): the source marginal is a KL penalty of strength rho (learned per
            head, from this start) instead of a hard constraint; each column step moves only a fraction
            rho / (rho + eps) of the way, so a source may send more or less than its share. Rows stay exact.
          learned_mass (--tt-learned-mass): each source's mass comes from its own token (a softmax over the
            sources, per head) instead of being equal; the total is still N.
          groups G > 1 (--tt-sensor-topology): G topologies, token sequence b uses topology b % G (the temporal
            blocks see the sensors as B * N sequences, sensor-minor: one lag geometry per sensor)."""
        super().__init__()
        self.heads, self.iters, self.drop, self.n_src = heads, iters, drop, n_src
        self.groups = groups
        self.src, self.dst, self.val, self.out = (nn.Linear(dim, dim) for _ in range(4))
        if topology:
            shape = (heads, tokens, topo_dim) if groups == 1 else (groups, heads, tokens, topo_dim)
            self.topo = nn.Parameter(0.1 * torch.randn(shape))
            self.topo_w = nn.Parameter(torch.zeros(heads))
        else:
            self.topo = self.topo_w = None
        self.log_eps = nn.Parameter(torch.zeros(heads))
        self.log_rho = nn.Parameter(torch.full((heads,), float(np.log(unbalanced)))) if unbalanced > 0 else None
        self.mass = nn.Linear(dim, heads) if learned_mass else None
        if self.mass is not None:   # equal masses at the start
            nn.init.zeros_(self.mass.weight)
            nn.init.zeros_(self.mass.bias)

    @staticmethod
    def _dist(a, b):   # (..., N, d), (..., M, d) -> (..., M, N) Euclidean distances, safe gradient at 0
        sq = b.pow(2).sum(-1, keepdim=True) + a.pow(2).sum(-1)[..., None, :] - 2 * b @ a.transpose(-2, -1)
        return (sq.clamp_min(0) + 1e-4).sqrt()

    def _topology_cost(self, S, B):
        w = F.softplus(self.topo_w)[:, None, None]
        if self.groups == 1:
            return w * self._dist(self.topo[:, :S], self.topo)                    # (h, N, S)
        if B % self.groups:
            raise ValueError(f"{B} sequences aren't a multiple of the {self.groups} topology groups")
        cost = w * self._dist(self.topo[:, :, :S], self.topo)                      # (G, h, N, S)
        return cost.repeat(B // self.groups, 1, 1, 1)                              # sequence b -> group b % G

    def forward(self, x):
        B, N, D = x.shape
        S = self.n_src or N
        split = lambda t: t.reshape(B, t.shape[1], self.heads, -1).transpose(1, 2)   # (B, h, tokens, d)
        a, b, v = split(self.src(x[:, :S])), split(self.dst(x)), split(self.val(x[:, :S]))
        cost = self._dist(a, b)                                                    # (B, h, N dest, S sources)
        if self.topo is not None:
            cost = cost + self._topology_cost(S, B)
        eps = self.log_eps.exp()[:, None, None]
        logits = -cost / eps
        # log of what each source sends: N / S each (0 when every token is a source), or learned shares of N
        if self.mass is not None:
            col = float(np.log(N)) + self.mass(x[:, :S]).transpose(1, 2).log_softmax(-1)[:, :, None, :]
        else:
            col = float(np.log(N / S))
        frac = None
        if self.log_rho is not None:   # unbalanced: a damped column step
            rho = self.log_rho.exp()[:, None, None]
            frac = rho / (rho + eps)
        for i in range(self.iters):   # odd count ends on the rows: each destination's weights sum to 1
            if i % 2 == 0:
                logits = logits - logits.logsumexp(dim=-1, keepdim=True)
            else:
                step = col - logits.logsumexp(dim=-2, keepdim=True)
                logits = logits + (step if frac is None else frac * step)
        plan = F.dropout(logits.exp(), self.drop, self.training)
        self.last_plan = plan.detach() if getattr(self, "keep_plan", False) else None   # tools/plot_tt.py
        return self.out((plan @ v).transpose(1, 2).reshape(B, N, D))


class AdaLNBlock(nn.Module):
    grad_checkpoint = False   # train.py --grad-checkpoint: recompute the block in the backward pass

    def __init__(self, dim, heads, mlp_dim, cond_dim, drop, sinkformer=0, modes=None, transport=None):
        """modes: (tokens, rank) to mix tokens with a ModeMixer instead of self-attention.
        transport: (tokens, Sinkhorn steps[, sources[, options]]) to mix them with a LatentTransport instead;
        options are LatentTransport's keyword arguments, plus hybrid=True (--tt-hybrid): keep the self-attention
        too and add the two (a = attention(h) + transport(h))."""
        super().__init__()
        self.sinkformer = sinkformer
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.modes = ModeMixer(modes[0], dim, modes[1], drop, *modes[2:]) if modes else None
        hybrid = False
        if transport:
            opts = dict(transport[3]) if len(transport) > 3 else {}
            hybrid = opts.pop("hybrid", False)
            self.modes = LatentTransport(transport[0], dim, heads, drop, transport[1],
                                         n_src=transport[2] if len(transport) > 2 else None, **opts)
        self.hybrid = hybrid
        self.attn = None if (self.modes is not None and not hybrid) else \
            nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
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
        if self.hybrid:
            a = self.modes(h) + self.attn(h, h, h, need_weights=False)[0]
        elif self.modes is not None:
            a = self.modes(h)
        elif self.sinkformer:
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
                 cond_dim, drop, sinkformer=0, spatial_modes=0, mode_gate=False, latent_transport=0,
                 time_transport=0, adaptive_embed=False, tt_opts=None):
        super().__init__()
        self.patch, self.T_in, self.T_out, self.cout_step = patch, T_in, T_out, cout_step
        self.embed = nn.Conv2d(cin_frame, dim, patch, stride=patch)
        tokens = (grid[0] // patch) * (grid[1] // patch)
        self.target = nn.Parameter(torch.zeros(1, T_out, 1, dim))
        if adaptive_embed:   # STAEformer's spatio-temporal adaptive embedding: one free vector per (frame, patch)
            self.adaptive = nn.Parameter(torch.zeros(1, T_in + T_out, tokens, dim))
            params = (self.adaptive, self.target)
        else:                # separable: a position per patch + an embedding per frame (lag / target step)
            self.adaptive = None
            self.pos = nn.Parameter(torch.zeros(1, 1, tokens, dim))
            self.frame = nn.Parameter(torch.zeros(1, T_in + T_out, 1, dim))
            params = (self.pos, self.frame, self.target)
        for p in params:
            nn.init.trunc_normal_(p, std=0.02)
        modes = (tokens, spatial_modes, mode_gate) if spatial_modes else None   # spatial blocks only; temporal keep attention
        transport = (tokens, latent_transport) if latent_transport else None   # spatial blocks
        # temporal blocks: the T_in input frames send, all T_in + T_out frames receive
        tt_opts = dict(tt_opts or {})
        if tt_opts.pop("sensor_topology", False):   # one lag topology per patch (sensor)
            tt_opts["groups"] = tokens
        t_transport = (T_in + T_out, time_transport, T_in, tt_opts) if time_transport else None
        self.blocks = nn.ModuleList(AdaLNBlock(dim, heads, mlp_dim, cond_dim, drop, sinkformer,
                                               modes if i % 2 == 1 else None,
                                               transport if i % 2 == 1 else t_transport) for i in range(depth))
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.final_ada = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.final_ada.weight)
        nn.init.zeros_(self.final_ada.bias)
        self.head = nn.Linear(dim, patch * patch * cout_step)

    def forward(self, x, cond, target_add=None):
        """x: (B, T_in, cin_frame, H, W) -> (B, T_out * cout_step, H, W), step-major like DistTransformer.
        target_add: optional (B, T_out, dim) added to the target tokens (--ext-input: the targets' features)."""
        B, T, Cf, H, W = x.shape
        h, w = H // self.patch, W // self.patch
        tokens = self.embed(x.reshape(B * T, Cf, H, W)).flatten(2).transpose(1, 2).reshape(B, T, h * w, -1)
        target = self.target.expand(B, -1, h * w, -1)
        if target_add is not None:
            target = target + target_add[:, :, None]
        z = torch.cat([tokens, target], dim=1)                                    # (B, F, N, D)
        z = z + (self.adaptive if self.adaptive is not None else self.pos + self.frame)
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
                 count_scale=0.0, count_offset=0.0, direct=False, aux=0, od_head=None, od_slot_scale=0.0,
                 ext_dim=0, ext_input=False, ext_volume=False, spatial_modes=0, mode_head=0, mode_init="random", mode_gate=False,
                 crop_pad=False, latent_transport=0, time_transport=0, adaptive_embed=False, zscore=None,
                 no_vol_encoder=False, tt_opts=None):
        """aux: extra input maps per frame after the C flow channels (--od), fed as log-flow maps next to
        the flow's (in every layout: stacked, time tokens, graph, volume transformer); not predicted.

        od_head: the --od-head kernel file. End(t+1) = the factorised part + the trips under way at t moved
        by a transport plan (distvol.ODTransportHead); the destination potential is one more output
        channel of the distribution head, zero at the start. od_slot_scale > 0 (--od-slot): the last
        extra map is the frame's time-of-day slot (scaled by 1 / od_slot_scale), read only by the od head
        for its per-slot prior; aux counts the maps the encoders get.

        direct: the ablation without the factorisation -- the same inputs and backbone (the volume
        encoder still conditions the distribution transformer), but the head outputs the flow itself
        instead of softmax logits, and there is no total-volume head.

        spatial_modes r (--spatial-modes, with time_tokens): the spatial blocks mix tokens through r learned
        modes (ModeMixer) instead of self-attention; mode_gate (--mode-gate) gates the modes from the input. mode_head r (--mode-head): the logits get a low-rank
        term Phi a, Phi (C, H*W, r) learned spatial modes, a = a linear map of the volume code (zero at the
        start); mode_init "pca" starts Phi from the train split's change modes (init_modes).

        crop_pad (time_tokens, patch 1; set for sensor data, H = 1): the grid is padded to a multiple of 4 for
        the volume U-Net only; the transformer gets the H x W cells themselves (padding H = 1 to 4 would
        quadruple the tokens with empty rows)."""
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
        self.ext = None   # --external: the target steps' features, added to the volume code (zero at the start)
        self.ext_embed = None   # --ext-input: every frame's features as input maps to both encoders instead
        self.ext_volume = bool(ext_dim and ext_volume)   # --ext-volume: the ext term goes to the volume head only
        if ext_input and ext_volume:
            raise ValueError("--ext-input and --ext-volume are alternatives")
        if ext_dim and ext_input:
            if vol_transformer:
                raise ValueError("--ext-input isn't wired into the volume transformer")
            self.ext_embed = nn.Linear(ext_dim, EXT_EMBED)
        elif ext_dim:
            if vol_transformer:
                raise ValueError("--external isn't wired into the volume transformer's own volume head")
            self.ext = nn.Sequential(nn.Linear(T_out * ext_dim, 64), nn.GELU(), nn.Linear(64, code_dim))
            nn.init.zeros_(self.ext[-1].weight)
            nn.init.zeros_(self.ext[-1].bias)
        if od_head is not None and (aux < 1 or T_out != 1 or C != 2 or direct or transport):
            raise ValueError("od_head needs --od, --t-out 1, the Start / End channels, and no direct / transport")
        g_out = 1 if od_head is not None else 0   # the potential channel, after the logits
        self.T_in, self.T_out, self.C, self.H, self.W = T_in, T_out, C, H, W
        self.aux = aux
        k_in, k_out = T_in * C, T_out * C
        if mode_gate and not spatial_modes:
            raise ValueError("--mode-gate gates the --spatial-modes mixer: give --spatial-modes")
        if spatial_modes and not time_tokens:
            raise ValueError("--spatial-modes replaces the time-token transformer's spatial attention: add --time-tokens")
        if latent_transport and (not time_tokens or spatial_modes or sinkformer):
            raise ValueError("--latent-transport replaces the time-token transformer's spatial attention: add "
                             "--time-tokens, and no --spatial-modes / --sinkformer")
        if adaptive_embed and not time_tokens:
            raise ValueError("--adaptive-embed is a per-(frame, patch) embedding of the time-token transformer: add --time-tokens")
        # --zscore: (mean, std) of the train flow in the 0-1 data scale; the direct head predicts
        # (x - mean) / std, the papers' z-scored target, and is mapped back here (the loss is scaled in train.py)
        self.zscore = tuple(zscore) if zscore else None
        if time_transport and (not time_tokens or sinkformer):
            raise ValueError("--time-transport replaces the time-token transformer's temporal attention: add "
                             "--time-tokens, and no --sinkformer")
        if tt_opts and not time_transport:
            raise ValueError("the --tt-* options change --time-transport: give --time-transport K")
        if mode_head and (direct or transport):
            raise ValueError("--mode-head adds to softmax logits: no --direct / --transport")
        self.mode_init = mode_init
        self.phi = nn.Parameter(torch.randn(C, H * W, mode_head) / (H * W) ** 0.5) if mode_head else None
        if mode_head:
            self.mode_coef = nn.Linear(code_dim, T_out * C * mode_head)
            nn.init.zeros_(self.mode_coef.weight)
            nn.init.zeros_(self.mode_coef.bias)
        a_in = T_in * aux
        e_in = (T_in + T_out) * EXT_EMBED if self.ext_embed is not None else 0   # stacked layouts
        self.pad_h, self.pad_w = -H % 4, -W % 4
        self.pad_h += -(H + self.pad_h) % patch
        self.pad_w += -(W + self.pad_w) % patch
        grid = (H + self.pad_h, W + self.pad_w)
        if crop_pad and not (time_tokens and patch == 1):
            raise ValueError("crop_pad goes with time_tokens and patch 1")
        self.crop_pad = crop_pad
        if no_vol_encoder and (not direct or vol_transformer or mode_head or ext_volume):
            raise ValueError("--no-vol-encoder: only with distvolformer_direct (no volume head to feed), "
                             "no --vol-transformer / --mode-head / --ext-volume")
        self.code_dim = code_dim
        if no_vol_encoder:   # the blocks are conditioned on a zero code (+ the --external term, if any)
            self.vol_enc = None
        elif vol_transformer:
            self.vol_enc = VolumeTransformer(3 * C + aux, C, T_in, T_out, width, vol_dim, vol_depth,
                                             max(1, vol_dim // 32), code_dim, drop)
        else:
            self.vol_enc = VolumeUNet(3 * k_in + a_in + e_in, width, code_dim)
        self.transport = TransportHead(T_out, C, transport, n_periodic) if transport else None
        if graph:
            self.dist_enc = GraphTransformer(2 * k_in + a_in + e_in, k_out + g_out, H, W, dim, depth, heads, mlp_dim,
                                             code_dim, drop, graph_radius)
        elif time_tokens:
            self.dist_enc = SpaceTimeTransformer(2 * C + aux + (EXT_EMBED if self.ext_embed is not None else 0), C + g_out, T_in, T_out,
                                                 (H, W) if crop_pad else grid, patch, dim, depth,
                                                 heads, mlp_dim, code_dim, drop, sinkformer, spatial_modes, mode_gate,
                                                 latent_transport, time_transport, adaptive_embed, tt_opts)
        else:
            self.dist_enc = DistTransformer(2 * k_in + a_in + e_in, self.transport.channels if transport else k_out + g_out,
                                            grid, patch,
                                            dim, depth, heads, mlp_dim, code_dim, drop, sinkformer)
        self.ext_target = nn.Linear(EXT_EMBED, dim) if (self.ext_embed is not None and time_tokens) else None
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

    def forward(self, x, return_parts=False, ext=None):
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
        e_maps = e_frames = None
        if self.ext_embed is not None:   # (B, (T + T_out) * E, Hp, Wp), (B, T + T_out, E)
            e_maps, e_frames = ext_maps(self.ext_embed, ext, T, self.T_out, Hp, Wp)
        if self.vol_transformer:   # per frame: distribution, log flow, log totals (, extra) as (3C + A, H, W)
            frames = torch.cat([features, volume_maps], 1).reshape(B, 3, T, C, Hp, Wp).transpose(1, 2)
            frames = frames.reshape(B, T, 3 * C, Hp, Wp)
            if A:
                frames = torch.cat([frames, extra.reshape(B, T, A, Hp, Wp)], 2)
            delta, code = self.vol_enc(frames, log_n.reshape(B, T, C))
        elif self.vol_enc is None:
            code = features.new_zeros(B, self.code_dim)
        else:
            code = self.vol_enc(torch.cat([features, volume_maps] + ([extra] if A else [])
                                          + ([e_maps] if e_maps is not None else []), dim=1))
        vol_code = code
        if self.ext is not None:
            if ext is None:
                raise ValueError("DistVolFormer was built with ext_dim > 0; pass ext")
            vol_code = code + self.ext(ext[:, -self.T_out:].reshape(B, -1))
            if not self.ext_volume:   # conditioning: every AdaLN block sees it too
                code = vol_code
        if self.time_tokens:   # the same features, one (2C + A, H, W) stack per frame
            features = features.reshape(B, 2, T, C, Hp, Wp).transpose(1, 2).reshape(B, T, 2 * C, Hp, Wp)
            if A:
                features = torch.cat([features, extra.reshape(B, T, A, Hp, Wp)], 2)
            if e_frames is not None:   # each input frame its own features; the targets' go to the target tokens
                features = torch.cat([features, e_frames[:, :T, :, None, None].expand(-1, -1, -1, Hp, Wp)], 2)
            if self.crop_pad:
                features = features[..., :H, :W]
        else:
            if A:
                features = torch.cat([features, extra], 1)
            if e_maps is not None:
                features = torch.cat([features, e_maps], 1)
        dist_kw = {"target_add": self.ext_target(e_frames[:, T:])} if self.ext_target is not None else {}

        if self.direct:
            x_hat = self.dist_enc(features, code, **dist_kw)[..., :H, :W].reshape(B, self.T_out, C, H, W)
            if self.zscore:
                x_hat = self.zscore[0] + self.zscore[1] * x_hat
            return (x_hat, *split_flow(x_hat)) if return_parts else x_hat
        if self.transport is not None:
            p_hat = self.transport(self.dist_enc(features, code, **dist_kw)[..., :H, :W], p_in, H, W)
        else:
            out = self.dist_enc(features, code, **dist_kw)[..., :H, :W]
            if self.od_head is not None:
                out, g = out[:, :self.T_out * C], out[:, self.T_out * C]
            logits = out.reshape(B, self.T_out, C, H * W)
            if self.dist_residual:
                logits = logits + log_last_distribution(p_in)
            if self.phi is not None:   # + Phi a: r global modes per channel, activations from the volume code
                a = self.mode_coef(code).reshape(B, self.T_out, C, -1)
                logits = logits + torch.einsum("cnr,btcr->btcn", self.phi, a)
            p_hat = logits.softmax(dim=-1).reshape(B, self.T_out, C, H, W)

        if not self.vol_transformer:
            delta = self.vol_head(torch.cat([vol_code, log_n], dim=-1)).reshape(B, self.T_out, C)
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

    @torch.no_grad()
    def init_modes(self, basis):
        """Phi from (C, H*W, r) modes, e.g. tools / train.py's pca of the train split's change maps."""
        self.phi.copy_(torch.as_tensor(basis, dtype=self.phi.dtype))


def change_modes(train, C, rank, samples=4000):
    """(C, H*W, rank) top principal directions of the train windows' change in log distribution, log p(target)
    - log p(last input frame), per channel: the data's own modes of how the map moves one step."""
    from trafficpred.models.distvol import RESIDUAL_FLOOR
    idx = np.linspace(0, len(train) - 1, min(len(train), samples)).astype(int)
    d = []
    for i in idx:
        item = train[i]
        x, y = item[0][-1, :C].double().clamp_min(0), item[1][0, :C].double().clamp_min(0)
        lp = lambda m: torch.log(m.flatten(1) / (m.flatten(1).sum(-1, keepdim=True) + 1e-9) + RESIDUAL_FLOOR)
        d.append((lp(y) - lp(x)).numpy())
    d = np.stack(d)                                     # (S, C, N)
    out = []
    for c in range(C):
        A = d[:, c] - d[:, c].mean(0)
        out.append(np.linalg.svd(A, full_matrices=False)[2][:rank].T)
    return np.stack(out).astype(np.float32)

