import torch.nn as nn
import torch.nn.functional as F

from trafficpred.sinkhorn import split_flow


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(8, ch), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(8, ch),
        )

    def forward(self, x):
        return F.gelu(x + self.net(x))


class CNN(nn.Module):
    def __init__(self, T_in=4, T_out=4, C=2, H=32, W=32, width=64, blocks=6, aux=0, ext_dim=0):
        """aux: extra input channels per frame after the C flow ones (--od), not predicted. ext_dim > 0
        (--external): the target steps' external features (calendar, last known weather) through a small MLP
        to a per-channel shift of the stem's features (zero at the start)."""
        super().__init__()
        self.ext = None
        if ext_dim:
            self.ext = nn.Sequential(nn.Linear(T_out * ext_dim, 64), nn.GELU(), nn.Linear(64, width))
            nn.init.zeros_(self.ext[-1].weight)
            nn.init.zeros_(self.ext[-1].bias)
        self.T_in, self.T_out, self.C, self.H, self.W = T_in, T_out, C, H, W
        self.stem = nn.Sequential(nn.Conv2d(T_in * (C + aux), width, 3, padding=1), nn.GELU())
        self.body = nn.Sequential(*[ResBlock(width) for _ in range(blocks)])
        self.head = nn.Conv2d(width, T_out * C, 1)

    def forward(self, x, return_parts=False, ext=None):
        B, T, C, H, W = x.shape
        h = self.stem(x.reshape(B, T * C, H, W))
        if self.ext is not None:
            if ext is None:
                raise ValueError("CNN was built with ext_dim > 0; pass ext")
            h = h + self.ext(ext[:, -self.T_out:].reshape(B, -1))[..., None, None]
        features = self.body(h)
        x_hat = F.softplus(self.head(features)).reshape(B, self.T_out, self.C, H, W)
        if return_parts:
            return (x_hat, *split_flow(x_hat))
        return x_hat
