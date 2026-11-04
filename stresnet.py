"""
ST-ResNet (Zhang, Zheng, Qi, AAAI 2017: "Deep Spatio-Temporal Residual Networks for
Citywide Crowd Flows Prediction"), the original TaxiBJ model.

    closeness frames  -> Conv -> L residual units -> Conv -> X_c --+
    period frames     -> Conv -> L residual units -> Conv -> X_p --+-- W_c*X_c + W_p*X_p + W_q*X_q
    trend frames      -> Conv -> L residual units -> Conv -> X_q --+   (+ external branch)
                                                                        -> squash -> X_hat

Each branch stacks its frames along channels (time folded into channels). A residual unit
is X + F(X), F = [BN ->] ReLU -> Conv3x3 -> [BN ->] ReLU -> Conv3x3 (pre-activation, as in
the paper; BN is optional there, --st-bn here). Fusion uses learnable per-cell weights
W of shape (C_out, H, W) (Hadamard product), so each region can weight the branches
differently. External factors (weather, holidays, time of day) go through
Linear(ext, 10) -> ReLU -> Linear(10, C_out*H*W) -> ReLU and are added before the squash.

Paper configuration for TaxiBJ: closeness 3 frames, period 1 (same time one day earlier),
trend 1 (one week earlier), 64 filters, L = 12 residual units, external factors on,
one-step prediction, data scaled to [-1, 1] with a tanh output.

Differences here, so it fits this codebase:
  - Multi-step: predicts all T_out frames at once (T_out * C output channels) instead of
    one step. With T_out = 1 it's the paper's setup.
  - Output squash: sigmoid instead of tanh, because data.py scales to [0, 1].
    sigmoid(2z) is tanh(z) mapped from [-1, 1] to [0, 1], so this is the same model. It
    also keeps predictions non-negative, which the Sinkhorn loss needs.
  - Period, trend and external inputs are optional. taxibj_splits() windows only hold
    consecutive frames, so with the standard pipeline this is ST-ResNet with the
    closeness branch only. Pass x_period / x_trend / ext to forward() (and set the
    lengths / ext_dim) if a data pipeline provides them.

Interface matches the other models:
    forward(x) with x (B, T_in, C, H, W)  ->  (B, T_out, C, H, W)
    forward(x, return_parts=True)         ->  (x_hat, p_hat, N_hat)
"""
import torch
import torch.nn as nn

from distvol import split_flow


class ResUnit(nn.Module):
    """X + F(X), F = [BN] ReLU Conv [BN] ReLU Conv (pre-activation)."""

    def __init__(self, ch, bn=False):
        super().__init__()
        norm = (lambda: nn.BatchNorm2d(ch)) if bn else nn.Identity
        self.f = nn.Sequential(
            norm(), nn.ReLU(), nn.Conv2d(ch, ch, 3, padding=1),
            norm(), nn.ReLU(), nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.f(x)


class Branch(nn.Module):
    """One temporal component: Conv1 -> L residual units -> (ReLU) -> Conv2."""

    def __init__(self, cin, cout, filters, units, bn):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, filters, 3, padding=1),
            *[ResUnit(filters, bn) for _ in range(units)],
            nn.ReLU(),
            nn.Conv2d(filters, cout, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class STResNet(nn.Module):
    def __init__(self, T_in=4, T_out=4, C=2, H=32, W=32, filters=64, units=12, bn=False,
                 len_period=0, len_trend=0, ext_dim=0):
        """T_in: closeness frames. len_period / len_trend: frames in those branches
        (0 = branch off). ext_dim: size of the external-factor vector (0 = off)."""
        super().__init__()
        self.T_in, self.T_out, self.C, self.H, self.W = T_in, T_out, C, H, W
        cout = T_out * C
        self.lens = {"closeness": T_in, "period": len_period, "trend": len_trend}
        self.branches = nn.ModuleDict({
            k: Branch(n * C, cout, filters, units, bn) for k, n in self.lens.items() if n > 0
        })
        # Parametric-matrix fusion: one weight per (output channel, cell) per branch.
        self.fuse = nn.ParameterDict({
            k: nn.Parameter(torch.ones(cout, H, W) / len(self.branches)) for k in self.branches
        })
        self.ext = None
        if ext_dim > 0:
            self.ext = nn.Sequential(nn.Linear(ext_dim, 10), nn.ReLU(),
                                     nn.Linear(10, cout * H * W), nn.ReLU())

    def forward(self, x, return_parts=False, x_period=None, x_trend=None, ext=None):
        B = x.shape[0]
        inputs = {"closeness": x, "period": x_period, "trend": x_trend}
        out = 0
        for k, branch in self.branches.items():
            xi = inputs[k]
            if xi is None:
                raise ValueError(f"STResNet was built with a {k} branch; pass x_{k}")
            out = out + self.fuse[k] * branch(xi.reshape(B, -1, self.H, self.W))
        if self.ext is not None:
            if ext is None:
                raise ValueError("STResNet was built with ext_dim > 0; pass ext")
            out = out + self.ext(ext).reshape(B, -1, self.H, self.W)
        x_hat = torch.sigmoid(2 * out).reshape(B, self.T_out, self.C, self.H, self.W)
        if return_parts:
            p_hat, N_hat = split_flow(x_hat)
            return x_hat, p_hat, N_hat
        return x_hat
