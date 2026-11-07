import torch
import torch.nn as nn
import torch.nn.functional as F

from trafficpred.sinkhorn import split_flow

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


class DistVolNet(nn.Module):
    def __init__(self, T_in=4, T_out=4, C=2, H=32, W=32, width=64):
        super().__init__()
        self.T_in, self.T_out, self.C, self.H, self.W = T_in, T_out, C, H, W
        k_in, k_out = T_in * C, T_out * C
        self.vol_enc = nn.Sequential(nn.Linear(k_in, 128), nn.GELU(), nn.Linear(128, 128))
        self.encoder = MultiscaleEncoder(2 * k_in, width, cond_dim=128)
        self.dist_head = nn.Conv2d(width, k_out, 1)
        self.vol_head = nn.Sequential(
            nn.Linear(self.encoder.mid_ch + 128, 256), nn.GELU(), nn.Linear(256, k_out)
        )
        nn.init.zeros_(self.vol_head[-1].weight)
        nn.init.zeros_(self.vol_head[-1].bias)

    def forward(self, x, return_parts=False):
        B, T, C, H, W = x.shape
        p_in, n_in = split_flow(x.clamp_min(0))
        features = torch.cat([
            (p_in * H * W).reshape(B, T * C, H, W),
            torch.log1p(x.clamp_min(0)).reshape(B, T * C, H, W),
        ], dim=1)
        cond = self.vol_enc(torch.log(n_in + EPS).reshape(B, T * C))
        features = F.pad(features, (0, -W % 4, 0, -H % 4))
        u, bottleneck = self.encoder(features, cond)

        logits = self.dist_head(u)[..., :H, :W].reshape(B, self.T_out, C, H * W)
        p_hat = logits.softmax(dim=-1).reshape(B, self.T_out, C, H, W)

        pooled = bottleneck.mean(dim=(-2, -1))
        delta = self.vol_head(torch.cat([pooled, cond], dim=-1)).reshape(B, self.T_out, C)
        n_hat = n_in[:, -1:, :] * torch.exp(delta.clamp(-5, 5))

        x_hat = p_hat * n_hat[..., None, None]
        if return_parts:
            return x_hat, p_hat, n_hat
        return x_hat
