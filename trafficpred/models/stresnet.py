import torch
import torch.nn as nn

from trafficpred.sinkhorn import split_flow


class ResUnit(nn.Module):
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
                 len_period=0, len_trend=0, ext_dim=0, init_mean=None):
        super().__init__()
        self.T_in, self.T_out, self.C, self.H, self.W = T_in, T_out, C, H, W
        cout = T_out * C
        self.lens = {"closeness": T_in, "period": len_period, "trend": len_trend}
        self.branches = nn.ModuleDict({
            k: Branch(n * C, cout, filters, units, bn) for k, n in self.lens.items() if n > 0
        })
        self.fuse = nn.ParameterDict({
            k: nn.Parameter(torch.ones(cout, H, W) / len(self.branches)) for k in self.branches
        })
        if init_mean is not None:
            mean = float(min(max(init_mean, 1e-4), 1 - 1e-4))
            bias = 0.5 * torch.logit(torch.tensor(mean)).item()
            for branch in self.branches.values():
                nn.init.constant_(branch.net[-1].bias, bias)
        self.ext = None
        if ext_dim > 0:
            self.ext = nn.Sequential(nn.Linear(ext_dim, 10), nn.ReLU(),
                                     nn.Linear(10, cout * H * W), nn.ReLU())

    def forward(self, x, return_parts=False, x_period=None, x_trend=None, ext=None):
        B = x.shape[0]
        inputs = {"closeness": x, "period": x_period, "trend": x_trend}
        out = 0
        for k, branch in self.branches.items():
            if inputs[k] is None:
                raise ValueError(f"STResNet was built with a {k} branch; pass x_{k}")
            out = out + self.fuse[k] * branch(inputs[k].reshape(B, -1, self.H, self.W))
        if self.ext is not None:
            if ext is None:
                raise ValueError("STResNet was built with ext_dim > 0; pass ext")
            out = out + self.ext(ext).reshape(B, -1, self.H, self.W)
        x_hat = torch.sigmoid(2 * out).reshape(B, self.T_out, self.C, self.H, self.W)
        if return_parts:
            return (x_hat, *split_flow(x_hat))
        return x_hat
