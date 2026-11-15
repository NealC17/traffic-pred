import torch
import torch.nn as nn
import torch.nn.functional as F

from trafficpred.models.distvol import EPS, ConvBlock, log_last_distribution
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


class DistVolFormer(nn.Module):
    def __init__(self, T_in=4, T_out=4, C=2, H=32, W=32, width=64, patch=2, dim=192, depth=6,
                 heads=6, mlp_dim=768, drop=0.1, code_dim=128, dist_residual=False,
                 transport=0, n_periodic=0, sinkformer=0):
        super().__init__()
        self.T_in, self.T_out, self.C, self.H, self.W = T_in, T_out, C, H, W
        k_in, k_out = T_in * C, T_out * C
        self.pad_h, self.pad_w = -H % 4, -W % 4
        self.pad_h += -(H + self.pad_h) % patch
        self.pad_w += -(W + self.pad_w) % patch
        grid = (H + self.pad_h, W + self.pad_w)
        self.vol_enc = VolumeUNet(3 * k_in, width, code_dim)
        self.transport = TransportHead(T_out, C, transport, n_periodic) if transport else None
        self.dist_enc = DistTransformer(2 * k_in, self.transport.channels if transport else k_out, grid, patch, dim, depth, heads, mlp_dim,
                                        code_dim, drop, sinkformer)
        self.dist_residual = dist_residual
        if transport:
            p2 = patch * patch   # head outputs (patch*patch*cout) are pixel-shuffled: channel-major
            self.transport.init_head(self.dist_enc.head.weight, self.dist_enc.head.bias.view(-1, p2)[:, 0])
            with torch.no_grad():
                self.dist_enc.head.bias.view(-1, p2)[:] = self.dist_enc.head.bias.view(-1, p2)[:, :1]
        elif dist_residual:   # starts as "the last frame's distribution", learns the change
            nn.init.zeros_(self.dist_enc.head.weight)
            nn.init.zeros_(self.dist_enc.head.bias)
        self.vol_head = nn.Sequential(nn.Linear(code_dim + k_in, 256), nn.GELU(), nn.Linear(256, k_out))
        nn.init.zeros_(self.vol_head[-1].weight)
        nn.init.zeros_(self.vol_head[-1].bias)

    def forward(self, x, return_parts=False):
        B, T, C, H, W = x.shape
        p_in, n_in = split_flow(x.clamp_min(0))
        log_n = torch.log(n_in + EPS).reshape(B, T * C)
        features = torch.cat([
            (p_in * H * W).reshape(B, T * C, H, W),
            torch.log1p(x.clamp_min(0)).reshape(B, T * C, H, W),
        ], dim=1)
        features = F.pad(features, (0, self.pad_w, 0, self.pad_h))
        volume_maps = log_n[..., None, None].expand(-1, -1, *features.shape[-2:])
        code = self.vol_enc(torch.cat([features, volume_maps], dim=1))

        if self.transport is not None:
            p_hat = self.transport(self.dist_enc(features, code)[..., :H, :W], p_in, H, W)
        else:
            logits = self.dist_enc(features, code)[..., :H, :W].reshape(B, self.T_out, C, H * W)
            if self.dist_residual:
                logits = logits + log_last_distribution(p_in)
            p_hat = logits.softmax(dim=-1).reshape(B, self.T_out, C, H, W)

        delta = self.vol_head(torch.cat([code, log_n], dim=-1)).reshape(B, self.T_out, C)
        n_hat = n_in[:, -1:, :] * torch.exp(delta.clamp(-5, 5))

        x_hat = p_hat * n_hat[..., None, None]
        if return_parts:
            return x_hat, p_hat, n_hat
        return x_hat
