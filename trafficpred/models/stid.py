"""STID (Shao et al., CIKM'22, "Spatial-Temporal Identity"), following the official BasicTS
implementation (GestaltCogTeam/BasicTS, baselines/STID/arch/stid_arch.py): the whole input history of a
sensor embedded by one 1x1 conv, concatenated with a learned node embedding and time-of-day / day-of-week
embeddings of the last input step, then residual MLP layers and a 1x1 conv to all output steps at once.
A simple, strong PEMS baseline, here to check that our pipeline reproduces the published numbers.

Sensors are the H * W cells (PEMS: H = 1). The time of day and day of week come from --external's
calendar features (dow one-hot 7, day off, sin / cos time of day) of the last input step."""
import math

import torch
import torch.nn as nn


class _MLP(nn.Module):
    def __init__(self, dim, drop):
        super().__init__()
        self.fc1 = nn.Conv2d(dim, dim, 1)
        self.fc2 = nn.Conv2d(dim, dim, 1)
        self.act, self.drop = nn.ReLU(), nn.Dropout(drop)

    def forward(self, x):
        return x + self.fc2(self.drop(self.act(self.fc1(x))))


class STID(nn.Module):
    def __init__(self, T_in=12, T_out=12, C=1, H=1, W=307, ext_dim=0, steps_per_day=288, embed_dim=32,
                 node_dim=32, tod_dim=32, dow_dim=32, layers=3, drop=0.15):
        super().__init__()
        self.T_in, self.T_out, self.C, self.H, self.W = T_in, T_out, C, H, W
        self.steps_per_day = steps_per_day
        self.ext = ext_dim or None     # train / eval pass ext= when this is set
        n = H * W
        self.ts_emb = nn.Conv2d(T_in * C, embed_dim, 1)
        self.node_emb = nn.Parameter(torch.empty(n, node_dim))
        nn.init.xavier_uniform_(self.node_emb)
        hidden = embed_dim + node_dim
        if ext_dim:
            self.tod_emb = nn.Parameter(torch.empty(steps_per_day, tod_dim))
            self.dow_emb = nn.Parameter(torch.empty(7, dow_dim))
            nn.init.xavier_uniform_(self.tod_emb)
            nn.init.xavier_uniform_(self.dow_emb)
            hidden += tod_dim + dow_dim
        self.encoder = nn.Sequential(*[_MLP(hidden, drop) for _ in range(layers)])
        self.out = nn.Conv2d(hidden, T_out * C, 1)

    def forward(self, x, ext=None):
        B, T = x.shape[:2]
        n = self.H * self.W
        hist = x[:, :, :self.C].reshape(B, T * self.C, n, 1)                 # (B, T C, N, 1)
        parts = [self.ts_emb(hist), self.node_emb.T[None, :, :, None].expand(B, -1, -1, -1)]
        if self.ext:
            if ext is None:
                raise ValueError("STID was built with ext_dim > 0; pass ext")
            last = ext[:, T - 1]                                              # the last input step
            dow = last[:, :7].argmax(-1)
            angle = torch.atan2(last[:, 8], last[:, 9]) % (2 * math.pi)
            tod = (angle / (2 * math.pi) * self.steps_per_day).round().long() % self.steps_per_day
            for table, idx in ((self.tod_emb, tod), (self.dow_emb, dow)):
                parts.append(table[idx][:, :, None, None].expand(-1, -1, n, 1))
        h = self.encoder(torch.cat(parts, 1))
        return self.out(h).reshape(B, self.T_out, self.C, self.H, self.W)
