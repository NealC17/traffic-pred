"""Training losses, independent of the models.

Pointwise losses compare the prediction with the target cell by cell and work with every model.
The distribution / volume losses need a model that returns its parts (x_hat, p_hat, n_hat):
p_hat is where the flow goes (each map sums to 1), n_hat how much of it there is (the map's total).
Their terms are

    reconstruction  ||x_hat - x||_1
    transport       S_eps(p_hat, p)        debiased Sinkhorn divergence between the shapes
    volume          |n_hat - n| / (H W)

and they differ only in which terms are trained and how the terms are combined:

    sinkhorn         1 * reconstruction + 10 * transport + 1 * volume   (fixed weights)
    sinkhorn_only    transport alone
    sinkhorn_vol     10 * transport + 1 * volume
    sinkhorn_vol_uw  the vector (transport, volume), combined by learned uncertainty weights
                     (Kendall, Gal & Cipolla 2018): sum_i exp(-s_i) * L_i + s_i

Every term is logged whether or not it is trained.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from trafficpred.sinkhorn import sinkhorn_divergence, split_flow


def masked_l1_loss(pred, y):
    mask = (y != 0).float()
    return ((pred - y).abs() * mask).sum() / mask.sum().clamp_min(1.0)


POINTWISE = {"mse": F.mse_loss, "mae": F.l1_loss, "masked_mae": masked_l1_loss}
WEIGHTED = {  # name -> default (l1, l2, l3): weights on reconstruction, transport, volume
    "sinkhorn": (1.0, 10.0, 1.0),
    "sinkhorn_only": (0.0, 10.0, 0.0),
    "sinkhorn_vol": (0.0, 10.0, 1.0),
}
UNCERTAINTY = ("sinkhorn_vol_uw",)
SINKHORN_DEFAULTS = dict(eps=0.01, sinkhorn_iters=30)

PARTS_LOSSES = (*WEIGHTED, *UNCERTAINTY)
LOSSES = (*POINTWISE, *PARTS_LOSSES)


class PointwiseLoss(nn.Module):
    needs_parts = False

    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.fn = POINTWISE[kind]

    def forward(self, pred, y):
        loss = self.fn(pred.float(), y)
        return loss, {self.kind: loss.detach()}


class DistVolTerms(nn.Module):
    needs_parts = True

    def __init__(self, eps=0.01, iters=30):
        super().__init__()
        self.eps, self.iters = eps, iters

    def terms(self, parts, y, transport=True):
        x_hat, p_hat, n_hat = (t.float() for t in parts)
        H, W = y.shape[-2:]
        p, n = split_flow(y.clamp_min(0))
        out = {"l1": (x_hat - y).abs().mean(),
               "volume": (n_hat - n).abs().mean() / (H * W)}
        if transport:
            out["sinkhorn"] = sinkhorn_divergence(p_hat.reshape(-1, H, W), p.reshape(-1, H, W),
                                                  eps=self.eps, iters=self.iters).mean()
        return out


class DistVolLoss(DistVolTerms):
    """Fixed-weight sum l1 * reconstruction + l2 * transport + l3 * volume."""

    def __init__(self, l1=1.0, l2=10.0, l3=1.0, eps=0.01, iters=30):
        super().__init__(eps, iters)
        self.weights = {"l1": l1, "sinkhorn": l2, "volume": l3}

    def forward(self, parts, y):
        terms = self.terms(parts, y, transport=self.weights["sinkhorn"] > 0)
        total = sum(self.weights[k] * v for k, v in terms.items() if self.weights[k] > 0)
        return total, {k: v.detach() for k, v in terms.items()}


class UncertaintyLoss(DistVolTerms):
    """The loss vector (transport, volume), combined as sum_i exp(-s_i) * L_i + s_i.

    s_i = log sigma_i^2 is learned with the model (no weight decay). At its optimum
    exp(s_i) = L_i, so each term's weight tracks 1 / L_i and the two terms carry gradients of
    similar size, however different their scales. s_i starts at log L_i of the first batch, so
    training starts balanced instead of spending epochs moving s from 0 to about log(1e-3).
    """

    VECTOR = ("sinkhorn", "volume")

    def __init__(self, eps=0.01, iters=30):
        super().__init__(eps, iters)
        self.log_vars = nn.Parameter(torch.zeros(len(self.VECTOR)))
        self.register_buffer("initialized", torch.tensor(False))

    def forward(self, parts, y):
        terms = self.terms(parts, y)
        vector = torch.stack([terms[k] for k in self.VECTOR])
        if not self.initialized:
            with torch.no_grad():
                self.log_vars.copy_(vector.detach().clamp_min(1e-12).log())
                self.initialized.fill_(True)
        total = (torch.exp(-self.log_vars) * vector + self.log_vars).sum()
        logged = {k: v.detach() for k, v in terms.items()}
        for k, s in zip(self.VECTOR, self.log_vars.detach()):
            logged[f"weight_{k}"] = torch.exp(-s)
        return total, logged


def needs_parts(name):
    return name in PARTS_LOSSES


def default_options(name):
    """The options a loss is built with when nothing is overridden (stored with each run)."""
    if name in WEIGHTED:
        l1, l2, l3 = WEIGHTED[name]
        return dict(l1=l1, l2=l2, l3=l3, **SINKHORN_DEFAULTS)
    if name in UNCERTAINTY:
        return dict(SINKHORN_DEFAULTS)
    return {}


def make_loss(name, **options):
    if name in POINTWISE:
        return PointwiseLoss(name)
    options = {**default_options(name), **options}
    iters = options.pop("sinkhorn_iters")
    if name in WEIGHTED:
        return DistVolLoss(iters=iters, **options)
    if name in UNCERTAINTY:
        return UncertaintyLoss(iters=iters, **options)
    raise ValueError(f"unknown loss {name!r} (losses: {', '.join(LOSSES)})")
