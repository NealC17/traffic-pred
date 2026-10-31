"""
Baseline CNN trained with the same loss as DistVolNet (L1 + Sinkhorn + volume).

This is the ablation for DistVolNet: same loss, but none of the architecture.
    - No distribution / volume split: the network outputs counts directly.
    - No multiscale encoder: a stack of 3x3 residual conv blocks at full 32x32
      resolution (receptive field grows ~2 cells per conv).
    - No FiLM / volume conditioning.

Time is folded into channels (ST-ResNet style), so both inflow and outflow at every
future step come out of one conv as T_out * C output channels.

Interface matches DistVolNet:
    forward(x) with x (B, T_in, C, H, W)  ->  (B, T_out, C, H, W)
    forward(x, return_parts=True) -> (x_hat, p_hat, N_hat), where p_hat and N_hat are
    computed FROM x_hat (normalise each map, sum each map), so DistVolLoss applies as-is:
    the Sinkhorn term compares the shape of the predicted map with the true one, and the
    volume term compares their totals.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from distvol import split_flow


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(8, ch), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(8, ch),
        )

    def forward(self, x):
        return F.gelu(x + self.net(x))


class BaselineCNN(nn.Module):
    def __init__(self, T_in=4, T_out=4, C=2, H=32, W=32, width=64, blocks=6):
        super().__init__()
        self.T_in, self.T_out, self.C, self.H, self.W = T_in, T_out, C, H, W
        self.stem = nn.Sequential(nn.Conv2d(T_in * C, width, 3, padding=1), nn.GELU())
        self.body = nn.Sequential(*[ResBlock(width) for _ in range(blocks)])
        self.head = nn.Conv2d(width, T_out * C, 1)

    def forward(self, x, return_parts=False):
        B, T, C, H, W = x.shape
        h = self.body(self.stem(x.reshape(B, T * C, H, W)))
        # Softplus keeps counts positive with a non-zero gradient everywhere, which the
        # Sinkhorn term needs (it normalises x_hat into a distribution).
        x_hat = F.softplus(self.head(h)).reshape(B, self.T_out, C, H, W)
        if return_parts:
            p_hat, N_hat = split_flow(x_hat)
            return x_hat, p_hat, N_hat
        return x_hat