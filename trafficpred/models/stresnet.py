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
                 len_period=0, len_trend=0, ext_dim=0, init_mean=None, period_idx=None, trend_idx=None,
                 aux=0):
        """period_idx / trend_idx: which of the T_in input frames feed the period and trend branches
        (e.g. the --periods frames a day / a week back, as in the ST-ResNet paper); the other frames
        are the closeness branch's. Without them every frame goes to closeness, unless x_period /
        x_trend are passed to forward (len_period / len_trend frames each)."""
        super().__init__()
        self.T_in, self.T_out, self.C, self.H, self.W = T_in, T_out, C, H, W
        cout = T_out * C
        self.routes = None
        if period_idx or trend_idx:
            if len_period or len_trend:
                raise ValueError("give period_idx / trend_idx or len_period / len_trend, not both")
            taken = set(period_idx or ()) | set(trend_idx or ())
            if len(taken) != len(period_idx or ()) + len(trend_idx or ()) or not taken <= set(range(T_in)):
                raise ValueError(f"period_idx / trend_idx must be distinct frame indices below T_in={T_in}")
            self.routes = {"closeness": [i for i in range(T_in) if i not in taken],
                           "period": list(period_idx or ()), "trend": list(trend_idx or ())}
            for k, idx in self.routes.items():
                self.register_buffer(f"idx_{k}", torch.tensor(idx, dtype=torch.long), persistent=False)
            self.lens = {k: len(v) for k, v in self.routes.items()}
        else:
            self.lens = {"closeness": T_in, "period": len_period, "trend": len_trend}
        self.branches = nn.ModuleDict({
            k: Branch(n * (C + aux), cout, filters, units, bn) for k, n in self.lens.items() if n > 0
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
            self.ext = nn.Sequential(nn.Linear(T_out * ext_dim, 10), nn.ReLU(),
                                     nn.Linear(10, cout * H * W), nn.ReLU())

    def forward(self, x, return_parts=False, x_period=None, x_trend=None, ext=None):
        B = x.shape[0]
        if self.routes is not None:
            inputs = {k: x.index_select(1, getattr(self, f"idx_{k}")) for k in self.routes}
        else:
            inputs = {"closeness": x, "period": x_period, "trend": x_trend}
        out = 0
        for k, branch in self.branches.items():
            if inputs[k] is None:
                raise ValueError(f"STResNet was built with a {k} branch; pass x_{k}")
            out = out + self.fuse[k] * branch(inputs[k].reshape(B, -1, self.H, self.W))
        if self.ext is not None:
            if ext is None:
                raise ValueError("STResNet was built with ext_dim > 0; pass ext")
            out = out + self.ext(ext[:, self.T_in:].reshape(B, -1)).reshape(B, -1, self.H, self.W)
        x_hat = torch.sigmoid(2 * out).reshape(B, self.T_out, self.C, self.H, self.W)
        if return_parts:
            return (x_hat, *split_flow(x_hat))
        return x_hat
