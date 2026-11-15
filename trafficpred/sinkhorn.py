import contextlib
import os

import torch

MAX_EXPONENT = 80.0


def split_flow(x, eps=1e-6):
    total = x.sum(dim=(-2, -1))
    return x / (total[..., None, None] + eps), total


def sinkhorn_divergence(a, b, eps=0.01, iters=50, mass_floor=1e-8, anneal=False):
    """Debiased Sinkhorn divergence between maps a and b (B, H, W), each normalized to sum 1.

    The cost is |x - y|^2 / 2 with the grid scaled to the unit square, so the blur sqrt(eps) is
    in units of the grid side (eps 0.01 -> 0.1 -> ~3 cells on 32 x 32). anneal=True is epsilon
    scaling: the first half of the iterations lower eps geometrically from the grid's largest
    cost to eps, the rest run at eps. Small eps (sharp transport) then converges in ~100
    iterations, where a fixed eps of 0.001 is still ~50% off after 30.
    """
    with _no_autocast(a.device):
        return _divergence(a, b, eps, iters, mass_floor, anneal)


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


def _divergence(a, b, eps, iters, mass_floor, anneal=False):
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

    a = (a + mass_floor) / (a + mass_floor).sum(dim=(1, 2), keepdim=True)
    b = (b + mass_floor) / (b + mass_floor).sum(dim=(1, 2), keepdim=True)
    log_weights = torch.cat([b.log(), a.log(), a.log(), b.log()])

    def update(potentials, e):
        f_ab, g_ba, f_aa, g_bb = potentials.chunk(4)
        return softmin_at(e)(log_weights + torch.cat([g_ba, f_ab, f_aa, g_bb]) / e)

    with torch.no_grad():
        potentials = torch.zeros(4 * B, H, W, device=a.device, dtype=dtype)
        for e in _eps_schedule(eps, iters, anneal, cost_y.max().item() + cost_x.max().item()):
            potentials = 0.5 * (potentials + update(potentials, e))

    f_ab, g_ba, f_aa, g_bb = update(potentials.detach(), eps).chunk(4)
    return (a * (f_ab - f_aa)).sum(dim=(1, 2)) + (b * (g_ba - g_bb)).sum(dim=(1, 2))
