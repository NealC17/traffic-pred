"""STAEformer (Liu et al., CIKM'23, "Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for
Traffic Forecasting"), following the official code (github.com/XDZhelheim/STAEformer, model/STAEformer.py):

  input  [flow (z-scored), time of day (fraction of the day), day of week (0-6)] -> Linear(3, 24)
  + time-of-day embedding (288 x 24) + day-of-week embedding (7 x 24) + adaptive embedding (T_in x N x 80),
  concatenated to 152 per (step, sensor);
  3 temporal self-attention layers (across the T_in steps of each sensor), then 3 spatial ones (across the
  sensors of each step): post-norm (x = LN(x + drop(attn(x))), x = LN(x + drop(FFN(x)))), 4 heads, FFN 256, ReLU;
  output: each sensor's T_in x 152 features flattened, Linear(T_in * 152, T_out) ("mixed projection").

The flow is z-scored with the train mean / std (the official StandardScaler on the train inputs) and the output
is mapped back (inverse_transform), here into this repo's 0-1 scale. Sensors are the H * W cells (PEMS: H = 1).
The time of day / day of week come from --external's calendar features (dow one-hot 7, day off, sin / cos time
of day) of each input step, like the STID port. Train it with --loss huber_raw (the official nn.HuberLoss(),
delta 1 vehicle on the raw counts) and its recipe (Adam, wd 1.5e-3, lr 1e-3 cut x0.1 at epochs 25 / 45 / 65,
batch 16, up to 300 epochs, early stopping after 30, no clipping): the model's train defaults here."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from trafficpred.kernels import padded_sdpa


def calendar_slots(ext, steps_per_day):
    """(B, T, E) calendar features (dow one-hot 7, day off, sin / cos time of day) -> (B, T) time-of-day slot
    and day of week of each step."""
    dow = ext[..., :7].argmax(-1)
    angle = torch.atan2(ext[..., 8], ext[..., 9]) % (2 * math.pi)
    tod = (angle / (2 * math.pi) * steps_per_day).round().long() % steps_per_day
    return tod, dow


class _Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.q, self.k, self.v, self.out = (nn.Linear(dim, dim) for _ in range(4))

    def forward(self, x):                                   # (..., L, dim)
        split = lambda t: torch.cat(torch.split(t, self.head_dim, dim=-1), dim=0)   # heads onto the batch
        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        # softmax(q k^T / sqrt(d)) v as the official code, fused: memory grows with L, not L^2 (BjTT: 1260 roads)
        out = padded_sdpa(q, k, v)   # head size 38: padded to 40 so the fused kernels apply
        out = torch.cat(torch.split(out, x.shape[0], dim=0), dim=-1)
        return self.out(out)


class _SelfAttentionLayer(nn.Module):
    def __init__(self, dim, ff_dim, heads, drop):
        super().__init__()
        self.attn = _Attention(dim, heads)
        self.ff = nn.Sequential(nn.Linear(dim, ff_dim), nn.ReLU(inplace=True), nn.Linear(ff_dim, dim))
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.drop1, self.drop2 = nn.Dropout(drop), nn.Dropout(drop)

    def forward(self, x, dim):
        x = x.transpose(dim, -2)
        x = self.ln1(x + self.drop1(self.attn(x)))
        x = self.ln2(x + self.drop2(self.ff(x)))
        return x.transpose(dim, -2)


class STAEformer(nn.Module):
    def __init__(self, T_in=12, T_out=12, C=1, H=1, W=170, ext_dim=0, steps_per_day=288, zscore=(0.0, 1.0),
                 input_dim=24, tod_dim=24, dow_dim=24, adaptive_dim=80, ff_dim=256, heads=4, layers=3, drop=0.1):
        super().__init__()
        if C != 1:
            raise ValueError("STAEformer here forecasts one flow channel (PEMS)")
        if not ext_dim:
            raise ValueError("STAEformer needs the time of day / day of week: add --external")
        self.T_in, self.T_out, self.C, self.H, self.W = T_in, T_out, C, H, W
        self.ext = ext_dim             # train / eval pass ext= when this is set
        self.steps_per_day = steps_per_day
        self.mean, self.std = (float(v) for v in zscore)   # train flow mean / std in the 0-1 data scale
        n = H * W
        self.dim = input_dim + tod_dim + dow_dim + adaptive_dim
        self.input_proj = nn.Linear(3, input_dim)
        self.tod_embedding = nn.Embedding(steps_per_day, tod_dim)
        self.dow_embedding = nn.Embedding(7, dow_dim)
        self.adaptive = nn.init.xavier_uniform_(nn.Parameter(torch.empty(T_in, n, adaptive_dim)))
        self.layers_t = nn.ModuleList(_SelfAttentionLayer(self.dim, ff_dim, heads, drop) for _ in range(layers))
        self.layers_s = nn.ModuleList(_SelfAttentionLayer(self.dim, ff_dim, heads, drop) for _ in range(layers))
        self.output_proj = nn.Linear(T_in * self.dim, T_out)

    def _calendar(self, ext, T):
        """(B, T) time-of-day slot and day of week of each input step, from the calendar features."""
        return calendar_slots(ext[:, :T], self.steps_per_day)

    def forward(self, x, ext=None):
        if ext is None:
            raise ValueError("STAEformer was built with ext_dim > 0; pass ext")
        B, T = x.shape[:2]
        n = self.H * self.W
        flow = (x[:, :, 0].reshape(B, T, n) - self.mean) / self.std              # z-scored, (B, T, N)
        tod, dow = self._calendar(ext, T)
        tod_n, dow_n = tod[:, :, None].expand(-1, -1, n), dow[:, :, None].expand(-1, -1, n)
        raw = torch.stack([flow, tod_n.float() / self.steps_per_day, dow_n.float()], -1)   # (B, T, N, 3)
        h = torch.cat([self.input_proj(raw), self.tod_embedding(tod_n), self.dow_embedding(dow_n),
                       self.adaptive.expand(B, -1, -1, -1)], -1)                  # (B, T, N, 152)
        for layer in self.layers_t:
            h = layer(h, dim=1)
        for layer in self.layers_s:
            h = layer(h, dim=2)
        out = self.output_proj(h.transpose(1, 2).reshape(B, n, T * self.dim))   # (B, N, T_out), z units
        out = self.mean + self.std * out.transpose(1, 2)                          # back to the 0-1 scale
        return out.reshape(B, self.T_out, 1, self.H, self.W)
