"""
DistVolNet: forecast WHERE traffic goes (a spatial distribution) separately from
HOW MUCH traffic there is (total volume), then multiply them back together.

    X_{t-L:t}  ->  split into  p (normalised inflow/outflow maps)  and  N (totals)
               ->  multiscale CNN encoder (+ FiLM conditioning on the totals)
               ->  distribution head: spatial softmax over H*W  ->  p_hat
               ->  volume head:       positive prediction       ->  N_hat
    X_hat = p_hat * N_hat

Interface matches PredFormer and the CNN baselines:
    forward(x) with x (B, T_in, C, H, W)  ->  (B, T_out, C, H, W)
forward(x, return_parts=True) also returns p_hat and N_hat, which the loss needs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


def split_flow(x):
    """x (..., H, W) non-negative -> (p, N): p sums to 1 over (H, W), N = total."""
    N = x.sum(dim=(-2, -1))
    p = x / (N[..., None, None] + EPS)
    return p, N


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
    """Scale and shift feature maps from a conditioning vector (the volume history)."""
    def __init__(self, cond_dim, ch):
        super().__init__()
        self.lin = nn.Linear(cond_dim, 2 * ch)
        nn.init.zeros_(self.lin.weight); nn.init.zeros_(self.lin.bias)  # starts as identity

    def forward(self, h, c):
        g, b = self.lin(c).chunk(2, dim=-1)
        return h * (1 + g[..., None, None]) + b[..., None, None]


class MultiscaleEncoder(nn.Module):
    """
    U-Net over the grid, with the time axis folded into channels (ST-ResNet style).
    Three scales: 32x32 (local streets), 16x16 (districts), 8x8 (city-wide),
    plus dilated convs at the bottleneck for even longer range.
    """
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
        s1 = self.e1(x)                        # (B, w,  H,   W)
        s2 = self.e2(F.avg_pool2d(s1, 2))      # (B, 2w, H/2, W/2)
        s3 = self.e3(F.avg_pool2d(s2, 2))      # (B, 4w, H/4, W/4)
        m = self.film(s3 + self.mid(s3), cond)
        u = self.d2(torch.cat([F.interpolate(m, scale_factor=2), s2], 1))
        u = self.d1(torch.cat([F.interpolate(u, scale_factor=2), s1], 1))
        return u, m                            # full-res features, bottleneck features


class DistVolNet(nn.Module):
    def __init__(self, T_in=4, T_out=4, C=2, H=32, W=32, width=64):
        super().__init__()
        self.T_in, self.T_out, self.C, self.H, self.W = T_in, T_out, C, H, W
        k_in, k_out = T_in * C, T_out * C

        # Volume history -> conditioning vector (log scale: totals span orders of magnitude).
        self.vol_enc = nn.Sequential(nn.Linear(k_in, 128), nn.GELU(), nn.Linear(128, 128))

        # Encoder sees the normalised maps (scaled by H*W so a uniform map is all-ones)
        # plus log(1 + x) of the raw counts, so it still knows absolute busyness per cell.
        self.encoder = MultiscaleEncoder(2 * k_in, width, cond_dim=128)

        # Distribution head: one logit map per (t_out, channel); softmax over H*W.
        self.dist_head = nn.Conv2d(width, k_out, 1)

        # Volume head: pooled bottleneck features + volume history -> log-ratio vs. last total.
        self.vol_head = nn.Sequential(
            nn.Linear(self.encoder.mid_ch + 128, 256), nn.GELU(), nn.Linear(256, k_out)
        )
        nn.init.zeros_(self.vol_head[-1].weight); nn.init.zeros_(self.vol_head[-1].bias)

    def forward(self, x, return_parts=False):
        B, T, C, H, W = x.shape
        p_in, N_in = split_flow(x.clamp_min(0))                    # (B,T,C,H,W), (B,T,C)

        feats = torch.cat([
            (p_in * H * W).reshape(B, T * C, H, W),
            torch.log1p(x.clamp_min(0)).reshape(B, T * C, H, W),
        ], dim=1)
        logN_in = torch.log(N_in + EPS).reshape(B, T * C)
        cond = self.vol_enc(logN_in)

        u, m = self.encoder(feats, cond)

        # Distribution head: spatial softmax.
        logits = self.dist_head(u).reshape(B, self.T_out, C, H * W)
        p_hat = logits.softmax(dim=-1).reshape(B, self.T_out, C, H, W)

        # Volume head: N_hat = N_last * exp(delta). Always positive; delta = 0 at init,
        # so training starts from the "same as last frame" baseline.
        pooled = m.mean(dim=(-2, -1))
        delta = self.vol_head(torch.cat([pooled, cond], dim=-1)).reshape(B, self.T_out, C)
        N_last = N_in[:, -1:, :]                                   # (B,1,C)
        N_hat = N_last * torch.exp(delta.clamp(-5, 5))

        x_hat = p_hat * N_hat[..., None, None]
        if return_parts:
            return x_hat, p_hat, N_hat
        return x_hat


class DistVolLoss(nn.Module):
    """
    L = l1 * ||X_hat - X||_1  +  l2 * S_eps(p_hat, p)  +  l3 * |N_hat - N|

    Each term is scaled to be per-cell so the lambdas are comparable:
      - L1 term: mean over cells.
      - Volume term: |N_hat - N| / (H*W), i.e. the error in mean cell value.
      - Sinkhorn term: in squared grid-fraction units (see sinkhorn.py); its natural
        scale is small, so l2 defaults higher.
    """
    def __init__(self, l1=1.0, l2=10.0, l3=1.0, eps=0.01, iters=30):
        super().__init__()
        from sinkhorn import sinkhorn_divergence
        self.S = sinkhorn_divergence
        self.l1, self.l2, self.l3, self.eps, self.iters = l1, l2, l3, eps, iters

    def forward(self, x_hat, p_hat, N_hat, y):
        B, T, C, H, W = y.shape
        p, N = split_flow(y.clamp_min(0))
        l_rec = (x_hat - y).abs().mean()
        l_ot = torch.zeros((), device=y.device)
        if self.l2 > 0:
            l_ot = self.S(p_hat.reshape(-1, H, W), p.reshape(-1, H, W),
                          eps=self.eps, iters=self.iters).mean()
        l_vol = (N_hat - N).abs().mean() / (H * W)
        total = self.l1 * l_rec + self.l2 * l_ot + self.l3 * l_vol
        # Detached 0-d tensors, not .item(): reading a value back every step makes the CPU
        # wait for the GPU. float(t) / f"{t:.5f}" still work where a number is needed.
        return total, {"l1": l_rec.detach(), "sinkhorn": l_ot.detach(), "volume": l_vol.detach()}


