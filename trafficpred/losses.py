"""Training losses, independent of the models.

Pointwise losses compare the prediction with the target cell by cell and work with every model.
The distribution / volume losses need a model that returns its parts (x_hat, p_hat, n_hat):
p_hat is where the flow goes (each map sums to 1), n_hat how much of it there is (the map's total).
Their terms are

    reconstruction  ||x_hat - x||_1
    transport       S_eps(p_hat, p)        debiased Sinkhorn divergence between the shapes
    volume          |n_hat - n| / (H W)
    kl              KL(p || p_hat)         sum_cells p log(p / p_hat): the shapes compared cell by
                                           cell, with no notion of distance between cells
    change          S_eps(d_hat+, d+) + S_eps(d_hat-, d-), d = flow minus the last input frame:
                    transport between where the flow grows (and where it shrinks) in the prediction
                    and in truth, each normalized to a distribution, maps weighted by how much the
                    true flow changes. The whole-city shape hardly moves between frames and is
                    dominated by static background, so `transport` barely sees the moving part;
                    this term sees only it, at full scale

and they differ only in which terms are trained and how the terms are combined:

    sinkhorn         1 * reconstruction + 10 * transport + 1 * volume   (fixed weights)
    sinkhorn_only    transport alone
    sinkhorn_vol     10 * transport + 1 * volume
    kl               1 * reconstruction + 0.1 * kl + 1 * volume         (KL in place of Sinkhorn)
    sinkhorn_kl      1 * reconstruction + 10 * transport + 1 * volume + 0.1 * kl
    sinkhorn_change  1 * reconstruction + 1 * volume + 0.3 * change, with sharp, converged Sinkhorn
                     (eps 0.002 = blur ~1.4 cells, 60 epsilon-scaled iterations)
    sinkhorn_vol_uw  the vector (transport, volume), combined by learned uncertainty weights
                     (Kendall, Gal & Cipolla 2018): sum_i exp(-s_i) * L_i + s_i
    sinkhorn_adaptive  all three terms, weighted so each one's gradient on the model's parameters
                     is as large as the reconstruction term's (times l1 / l2 / l3, default 1 each),
                     from Adam-style running averages of the squared gradient norms

--l1 / --l2 / --l3 / --l4 / --l5 weigh reconstruction / transport / volume / kl / change. Every
term is logged whether or not it is trained, except that the Sinkhorn terms (transport, change)
are only computed when their weight is above 0. --sinkhorn-anneal turns on epsilon scaling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from trafficpred.sinkhorn import sinkhorn_divergence, split_flow


def masked_l1_loss(pred, y):
    mask = (y != 0).float()
    return ((pred - y).abs() * mask).sum() / mask.sum().clamp_min(1.0)


POINTWISE = {"mse": F.mse_loss, "mae": F.l1_loss, "masked_mae": masked_l1_loss}
CHANGE_WEIGHT = 0.3
CHANGE_FLOOR = 1e-2
WEIGHTED = {  # name -> default weights: l1 reconstruction, l2 transport, l3 volume, l4 kl
    "sinkhorn": dict(l1=1.0, l2=10.0, l3=1.0),
    "sinkhorn_only": dict(l1=0.0, l2=10.0, l3=0.0),
    "sinkhorn_vol": dict(l1=0.0, l2=10.0, l3=1.0),
    "kl": dict(l1=1.0, l2=0.0, l3=1.0, l4=0.1),
    "sinkhorn_kl": dict(l1=1.0, l2=10.0, l3=1.0, l4=0.1),
    "sinkhorn_change": dict(l1=1.0, l2=0.0, l3=1.0, l5=CHANGE_WEIGHT),
}
SHARP_SINKHORN = dict(eps=0.002, sinkhorn_iters=60, sinkhorn_anneal=True)   # sinkhorn_change's
KL_FLOOR = 1e-12
UNCERTAINTY = ("sinkhorn_vol_uw",)
ADAPTIVE = ("sinkhorn_adaptive",)
SINKHORN_DEFAULTS = dict(eps=0.01, sinkhorn_iters=30)
ADAPTIVE_DEFAULTS = dict(l1=1.0, l2=1.0, l3=1.0, adapt_beta=0.95, adapt_every=10)

PARTS_LOSSES = (*WEIGHTED, *UNCERTAINTY, *ADAPTIVE)
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


def kl_divergence(p, p_hat):
    """Mean over maps of KL(p || p_hat) = sum_cells p log(p / p_hat), in nats (0 log 0 = 0)."""
    log_ratio = torch.log(p.clamp_min(KL_FLOOR)) - torch.log(p_hat.clamp_min(KL_FLOOR))
    return torch.where(p > 0, p * log_ratio, torch.zeros_like(p)).sum(dim=(-2, -1)).mean()


def change_transport(x_hat, y, x_last, eps, iters, anneal):
    """Sinkhorn between where the flow grows (and shrinks) relative to the last input frame in
    the prediction and in truth, each normalized to a distribution; maps weighted by their true
    change. A floor of CHANGE_FLOOR * the mean true change keeps a map with almost no predicted
    growth from blowing up the normalization's gradient."""
    H, W = y.shape[-2:]
    d_hat = (x_hat - x_last).reshape(-1, H, W)
    d = (y - x_last).reshape(-1, H, W)
    a = torch.cat([d_hat.clamp_min(0), (-d_hat).clamp_min(0)])
    b = torch.cat([d.clamp_min(0), (-d).clamp_min(0)])
    mass = b.sum(dim=(-2, -1))
    floor = CHANGE_FLOOR * mass.mean().detach() / (H * W)
    s = sinkhorn_divergence(a + floor, b + floor, eps=eps, iters=iters, anneal=anneal)
    return (s * mass).sum() / mass.sum().clamp_min(1e-12)


class DistVolTerms(nn.Module):
    needs_parts = True
    anneal = False   # epsilon scaling in the Sinkhorn terms (set by make_loss)

    def __init__(self, eps=0.01, iters=30):
        super().__init__()
        self.eps, self.iters = eps, iters

    def terms(self, parts, y, transport=True, x=None, change=False):
        x_hat, p_hat, n_hat = (t.float() for t in parts)
        H, W = y.shape[-2:]
        p, n = split_flow(y.clamp_min(0))
        out = {"l1": (x_hat - y).abs().mean(),
               "volume": (n_hat - n).abs().mean() / (H * W),
               "kl": kl_divergence(p, p_hat)}
        if transport:
            out["sinkhorn"] = sinkhorn_divergence(p_hat.reshape(-1, H, W), p.reshape(-1, H, W),
                                                  eps=self.eps, iters=self.iters,
                                                  anneal=self.anneal).mean()
        if change:
            if x is None:
                raise RuntimeError("the change term needs the input frames: loss(parts, y, x=x)")
            out["change"] = change_transport(x_hat, y, x[:, -1:].float(), self.eps, self.iters, self.anneal)
        return out


class DistVolLoss(DistVolTerms):
    """Fixed-weight sum l1 * reconstruction + l2 * transport + l3 * volume + l4 * kl + l5 * change."""

    def __init__(self, l1=1.0, l2=10.0, l3=1.0, l4=0.0, l5=0.0, eps=0.01, iters=30):
        super().__init__(eps, iters)
        self.weights = {"l1": l1, "sinkhorn": l2, "volume": l3, "kl": l4, "change": l5}
        self.needs_input = l5 > 0

    def forward(self, parts, y, x=None):
        terms = self.terms(parts, y, transport=self.weights["sinkhorn"] > 0, x=x,
                           change=self.weights["change"] > 0)
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


class AdaptiveLoss(DistVolTerms):
    """l1 * w_1 * reconstruction + l2 * w_2 * transport + l3 * w_3 * volume, with w_i set like Adam.

    Every `every` training steps, each term's gradient on the model's parameters is measured
    (one extra backward pass per term) and its squared norm goes into a running average
    v_i <- beta * v_i + (1 - beta) * ||grad L_i||^2, bias-corrected as in Adam. The weights are
    w_i = sqrt(v_1 / v_i): every term pushes the parameters as hard as the reconstruction term
    (scaled by the alphas l1 / l2 / l3), so the total gradient keeps the size it has with the plain
    L1 loss and gradient clipping behaves as before. The weights are constants in the backward pass.
    """

    TERMS = ("l1", "sinkhorn", "volume")

    def __init__(self, l1=1.0, l2=1.0, l3=1.0, eps=0.01, iters=30, beta=0.95, every=10):
        super().__init__(eps, iters)
        self.alphas = {"l1": l1, "sinkhorn": l2, "volume": l3}
        self.beta, self.every = beta, every
        self.params = []
        self.register_buffer("sq_norms", torch.zeros(len(self.TERMS), dtype=torch.float64))
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("calls", torch.zeros((), dtype=torch.long))

    def bind(self, parameters):
        self.params = [p for p in parameters if p.requires_grad]

    def _measure(self, terms):
        norms = []
        for k in self.TERMS:
            if k not in terms or self.alphas[k] == 0:
                norms.append(0.0)
                continue
            grads = torch.autograd.grad(terms[k], self.params, retain_graph=True, allow_unused=True)
            norms.append(sum(float(g.double().pow(2).sum()) for g in grads if g is not None))
        new = torch.tensor(norms, dtype=torch.float64, device=self.sq_norms.device)
        self.sq_norms.mul_(self.beta).add_((1 - self.beta) * new)
        self.updates += 1

    def weights(self):
        if int(self.updates) == 0:
            return {k: 1.0 for k in self.TERMS}
        v = self.sq_norms / (1 - self.beta ** int(self.updates))
        ref = float(v[0])
        return {k: (ref / float(v[i])) ** 0.5 if float(v[i]) > 0 else 0.0 for i, k in enumerate(self.TERMS)}

    def forward(self, parts, y):
        if not self.params:
            raise RuntimeError("sinkhorn_adaptive needs the model's parameters: call loss.bind(model.parameters())")
        terms = self.terms(parts, y, transport=self.alphas["sinkhorn"] > 0)
        if torch.is_grad_enabled() and int(self.calls) % self.every == 0:
            self._measure(terms)
        self.calls += 1
        w = self.weights()
        total = sum(self.alphas[k] * w[k] * v for k, v in terms.items() if self.alphas.get(k, 0) > 0)
        logged = {k: v.detach() for k, v in terms.items()}
        for k in self.TERMS:
            if self.alphas[k] > 0:
                logged[f"weight_{k}"] = torch.tensor(self.alphas[k] * w[k])
        return total, logged


def needs_parts(name):
    return name in PARTS_LOSSES


def default_options(name):
    """The options a loss is built with when nothing is overridden (stored with each run)."""
    if name in WEIGHTED:
        return dict(WEIGHTED[name], **(SHARP_SINKHORN if "l5" in WEIGHTED[name] else SINKHORN_DEFAULTS))
    if name in UNCERTAINTY:
        return dict(SINKHORN_DEFAULTS)
    if name in ADAPTIVE:
        return dict(ADAPTIVE_DEFAULTS, **SINKHORN_DEFAULTS)
    return {}


def make_loss(name, **options):
    if name in POINTWISE:
        return PointwiseLoss(name)
    options = {**default_options(name), **options}
    iters = options.pop("sinkhorn_iters")
    anneal = bool(options.pop("sinkhorn_anneal", False))
    if name in WEIGHTED:
        loss = DistVolLoss(iters=iters, **options)
    elif name in UNCERTAINTY:
        loss = UncertaintyLoss(iters=iters, **options)
    elif name in ADAPTIVE:
        loss = AdaptiveLoss(iters=iters, beta=options.pop("adapt_beta"), every=options.pop("adapt_every"),
                            **options)
    else:
        loss = None
    if loss is not None:
        loss.anneal = anneal
        return loss
    raise ValueError(f"unknown loss {name!r} (losses: {', '.join(LOSSES)})")
