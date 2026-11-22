import contextlib
import os

import torch

MAX_EXPONENT = 80.0


def split_flow(x, eps=1e-6):
    total = x.sum(dim=(-2, -1))
    return x / (total[..., None, None] + eps), total


def sinkhorn_divergence(a, b, eps=0.01, iters=50, mass_floor=1e-8, anneal=False, reach=None):
    """Debiased Sinkhorn divergence between maps a and b (B, H, W), each normalized to sum 1.

    reach=R makes it unbalanced (Wasserstein-Fisher-Rao type, Sejourne et al. 2019): the marginal
    constraints become penalties rho * KL(pi 1 | a) + rho * KL(pi^T 1 | b) with rho = R^2, so mass
    may also be created or destroyed. a and b are then NOT normalized (pass them on a common scale):
    a mismatch closer than ~R is paid as transport, a farther one as local excess / shortfall. R is
    in the same units as the blur (grid side = 1). reach=None is the balanced divergence.

    The cost is |x - y|^2 / 2 with the grid scaled to the unit square, so the blur sqrt(eps) is
    in units of the grid side (eps 0.01 -> 0.1 -> ~3 cells on 32 x 32). anneal=True is epsilon
    scaling: the first half of the iterations lower eps geometrically from the grid's largest
    cost to eps, the rest run at eps. Small eps (sharp transport) then converges in ~100
    iterations, where a fixed eps of 0.001 is still ~50% off after 30.
    """
    with _no_autocast(a.device):
        return _divergence(a, b, eps, iters, mass_floor, anneal, None if reach is None else reach ** 2)


def _grid_cost(n, device, dtype):
    x = (torch.arange(n, device=device, dtype=dtype) + 0.5) / n
    return (x[:, None] - x[None, :]) ** 2 / 2


def _lse_grid(h, scaled_y, scaled_x):
    t = torch.logsumexp(h[:, :, None, :] - scaled_x[None, None], dim=-1)
    return torch.logsumexp(t[:, None, :, :] - scaled_y[None, :, :, None], dim=2)


_compiled_lse = None   # None: not tried yet; False: unavailable, use eager


def _lse(h, scaled_y, scaled_x):
    """The log-domain softmin's two logsumexps. On CUDA it is torch.compile'd: fused, it skips
    the (B, H, W, W) intermediates and runs ~12x faster, which small-eps Sinkhorn needs.
    TRAFFICPRED_NO_COMPILE=1 turns that off."""
    global _compiled_lse
    if h.is_cuda and _compiled_lse is not False and not os.environ.get("TRAFFICPRED_NO_COMPILE"):
        try:
            if _compiled_lse is None:
                import torch._functorch.config as functorch_config
                # backward(retain_graph=True) must keep working (sinkhorn_adaptive measures each
                # term's gradient that way), which donated buffers don't allow
                functorch_config.donated_buffer = False
                _compiled_lse = torch.compile(_lse_grid)
            return _compiled_lse(h, scaled_y, scaled_x)
        except Exception as e:  # no triton, unsupported GPU, ...
            print(f"[sinkhorn] torch.compile unavailable ({type(e).__name__}: {e}); using eager logsumexp")
            _compiled_lse = False
    return _lse_grid(h, scaled_y, scaled_x)


def _softmin_logsumexp(eps, cost_y, cost_x, h):
    return -eps * _lse(h, cost_y / eps, cost_x / eps)


def _softmin_matmul(eps, kernel_y, kernel_x, h):
    m = h.detach().amax(dim=-1, keepdim=True)
    t = torch.matmul(torch.exp(h - m), kernel_x).log_() + m
    m = t.detach().amax(dim=1, keepdim=True)
    s = torch.matmul(kernel_y, torch.exp(t - m)).log_() + m
    return -eps * s


def _no_autocast(device):
    try:
        return torch.autocast(device_type=device.type, enabled=False)
    except Exception:
        return contextlib.nullcontext()


def _eps_schedule(eps, iters, anneal, max_cost):
    if not anneal or iters < 2:
        return [eps] * iters
    n = iters // 2
    start = max(max_cost, eps)
    ratio = (eps / start) ** (1 / max(n - 1, 1))
    return [start * ratio ** k for k in range(n)] + [eps] * (iters - n)


def _divergence(a, b, eps, iters, mass_floor, anneal=False, rho=None):
    B, H, W = a.shape
    dtype = torch.float32 if a.dtype in (torch.float16, torch.bfloat16) else a.dtype
    a, b = a.to(dtype), b.to(dtype)
    cost_y, cost_x = _grid_cost(H, a.device, dtype), _grid_cost(W, a.device, dtype)
    axis_max = max(cost_y.max().item(), cost_x.max().item())
    kernels = {}

    def softmin_at(e):
        if axis_max / e < MAX_EXPONENT:
            if e not in kernels:
                kernels[e] = torch.exp(-cost_y / e), torch.exp(-cost_x / e)
            kernel_y, kernel_x = kernels[e]
            return lambda h: _softmin_matmul(e, kernel_y, kernel_x, h)
        return lambda h: _softmin_logsumexp(e, cost_y, cost_x, h)

    if rho is None:
        a = (a + mass_floor) / (a + mass_floor).sum(dim=(1, 2), keepdim=True)
        b = (b + mass_floor) / (b + mass_floor).sum(dim=(1, 2), keepdim=True)
    else:
        a, b = a.clamp_min(0) + mass_floor, b.clamp_min(0) + mass_floor
    log_weights = torch.cat([b.log(), a.log(), a.log(), b.log()])

    def update(potentials, e):
        f_ab, g_ba, f_aa, g_bb = potentials.chunk(4)
        out = softmin_at(e)(log_weights + torch.cat([g_ba, f_ab, f_aa, g_bb]) / e)
        return out if rho is None else out * (rho / (rho + e))   # KL marginal penalty: damped update

    with torch.no_grad():
        potentials = torch.zeros(4 * B, H, W, device=a.device, dtype=dtype)
        for e in _eps_schedule(eps, iters, anneal, cost_y.max().item() + cost_x.max().item()):
            potentials = 0.5 * (potentials + update(potentials, e))

    if rho is not None:
        return _unbalanced_value(a, b, potentials, softmin_at(eps), eps, rho)
    f_ab, g_ba, f_aa, g_bb = update(potentials.detach(), eps).chunk(4)
    return (a * (f_ab - f_aa)).sum(dim=(1, 2)) + (b * (g_ba - g_bb)).sum(dim=(1, 2))


def _unbalanced_value(a, b, potentials, softmin, eps, rho):
    """The debiased unbalanced divergence (Sejourne et al. 2019)
        S = OT(a, b) - OT(a, a) / 2 - OT(b, b) / 2 + eps / 2 * (m(a) - m(b))^2,
    each OT its dual objective at the converged (detached) potentials f, g:
        <a, rho (1 - e^{-f / rho})> + <b, rho (1 - e^{-g / rho})> - eps <a x b, e^{(f + g - C) / eps} - 1>.
    The potentials maximize it, so (envelope theorem) its autograd gradient in a and b is exact."""
    f_ab, g_ba, f_aa, g_bb = potentials.detach().chunk(4)
    total = lambda m: m.sum(dim=(1, 2))

    def ot(x, y, f, g):
        # sum_j y_j e^{(g_j - C_ij) / eps} = e^{-softmin_i / eps}, so the plan's mass is sum_i x_i e^{(f_i - softmin_i) / eps}
        plan_mass = total(x * torch.exp((f - softmin(y.log() + g / eps)) / eps))
        return (total(x * rho * (1 - torch.exp(-f / rho))) + total(y * rho * (1 - torch.exp(-g / rho)))
                - eps * (plan_mass - total(x) * total(y)))

    return (ot(a, b, f_ab, g_ba) - 0.5 * ot(a, a, f_aa, f_aa) - 0.5 * ot(b, b, g_bb, g_bb)
            + eps / 2 * (total(a) - total(b)) ** 2)
