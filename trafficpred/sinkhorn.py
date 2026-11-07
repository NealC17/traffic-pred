import contextlib

import torch

MAX_EXPONENT = 80.0


def split_flow(x, eps=1e-6):
    total = x.sum(dim=(-2, -1))
    return x / (total[..., None, None] + eps), total


def sinkhorn_divergence(a, b, eps=0.01, iters=50, mass_floor=1e-8):
    with _no_autocast(a.device):
        return _divergence(a, b, eps, iters, mass_floor)


def _grid_cost(n, device, dtype):
    x = (torch.arange(n, device=device, dtype=dtype) + 0.5) / n
    return (x[:, None] - x[None, :]) ** 2 / 2


def _softmin_logsumexp(eps, cost_y, cost_x, h):
    t = torch.logsumexp(h[:, :, None, :] - cost_x[None, None] / eps, dim=-1)
    t = torch.logsumexp(t[:, None, :, :] - cost_y[None, :, :, None] / eps, dim=2)
    return -eps * t


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


def _divergence(a, b, eps, iters, mass_floor):
    B, H, W = a.shape
    dtype = torch.float32 if a.dtype in (torch.float16, torch.bfloat16) else a.dtype
    a, b = a.to(dtype), b.to(dtype)
    cost_y, cost_x = _grid_cost(H, a.device, dtype), _grid_cost(W, a.device, dtype)

    if max(cost_y.max().item(), cost_x.max().item()) / eps < MAX_EXPONENT:
        kernel_y, kernel_x = torch.exp(-cost_y / eps), torch.exp(-cost_x / eps)
        softmin = lambda h: _softmin_matmul(eps, kernel_y, kernel_x, h)
    else:
        softmin = lambda h: _softmin_logsumexp(eps, cost_y, cost_x, h)

    a = (a + mass_floor) / (a + mass_floor).sum(dim=(1, 2), keepdim=True)
    b = (b + mass_floor) / (b + mass_floor).sum(dim=(1, 2), keepdim=True)
    log_weights = torch.cat([b.log(), a.log(), a.log(), b.log()])

    def update(potentials):
        f_ab, g_ba, f_aa, g_bb = potentials.chunk(4)
        return softmin(log_weights + torch.cat([g_ba, f_ab, f_aa, g_bb]) / eps)

    with torch.no_grad():
        potentials = torch.zeros(4 * B, H, W, device=a.device, dtype=dtype)
        for _ in range(iters):
            potentials = 0.5 * (potentials + update(potentials))

    f_ab, g_ba, f_aa, g_bb = update(potentials.detach()).chunk(4)
    return (a * (f_ab - f_aa)).sum(dim=(1, 2)) + (b * (g_ba - g_bb)).sum(dim=(1, 2))
