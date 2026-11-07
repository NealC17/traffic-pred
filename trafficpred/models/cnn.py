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
    def __init__(self, T_in=4, T_out=4, C=2, H=32, W=32, width=64, blocks=6):
        super().__init__()
        self.T_in, self.T_out, self.C, self.H, self.W = T_in, T_out, C, H, W
        self.stem = nn.Sequential(nn.Conv2d(T_in * C, width, 3, padding=1), nn.GELU())
        self.body = nn.Sequential(*[ResBlock(width) for _ in range(blocks)])
        self.head = nn.Conv2d(width, T_out * C, 1)

    def forward(self, x, return_parts=False):
        B, T, C, H, W = x.shape
        features = self.body(self.stem(x.reshape(B, T * C, H, W)))
        x_hat = F.softplus(self.head(features)).reshape(B, self.T_out, C, H, W)
        if return_parts:
            return (x_hat, *split_flow(x_hat))
        return x_hat
