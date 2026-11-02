"""
Debiased Sinkhorn divergence S_eps(a, b) between distributions on a 2-D grid.

    OT_eps(a, b)  = entropic optimal transport cost
    S_eps(a, b)   = OT_eps(a, b) - 1/2 OT_eps(a, a) - 1/2 OT_eps(b, b)

S_eps is >= 0, equals 0 only when a == b, and goes to the Wasserstein-2 cost
(||x - y||^2 / 2) as eps -> 0.

Separable cost: on a regular grid the squared-Euclidean cost splits into a row part
plus a column part, C[(i,j),(k,l)] = (Cy[i,k] + Cx[j,l]). So the reduction over all H*W
source cells is done one axis at a time: O(H*W*(H+W)) per distribution instead of
O((H*W)^2).

Each 1-D reduction is a matrix multiply against the Gibbs kernel K = exp(-C / eps),
after subtracting a running max (a "max-shifted" log-sum-exp):
    lse_l(h[l] - C[j,l]/eps) = m + log( sum_l exp(h[l] - m) * K[j,l] ),   m = max_l h[l]
The largest term is exp(0) * K[j, l*] >= exp(-max(C) / eps), so nothing underflows in
float32 as long as max(C) / eps < ~80 (eps > ~0.006 on the unit grid). Below that it falls
back to plain logsumexp. The matmul version avoids the (B, H, W, W) temporaries of the
plain version and runs as a few batched GEMMs, which is what makes it fast on a GPU.

All four transport problems (a->b, b->a, a->a, b->b) are stacked into one batch, so
each Sinkhorn iteration is a single call instead of four. Sinkhorn always runs in
float32 with autocast disabled (bf16 would wreck the potentials).

Gradients: the Sinkhorn loop runs without autograd, then one last update is done with
autograd on. By the envelope theorem, that gives the correct gradient of OT_eps with
respect to a and b (the same scheme GeomLoss uses).
"""
import contextlib

import torch

_MAX_EXPONENT = 80.0     # exp(-80) ~ 1.8e-35, still a normal float32


def _grid_cost(n, device, dtype):
    # Cell centres in [0, 1]; cost is |x - y|^2 / 2 along one axis.
    x = (torch.arange(n, device=device, dtype=dtype) + 0.5) / n
    return (x[:, None] - x[None, :]) ** 2 / 2


def _softmin_lse(eps, Cy, Cx, h):
    """Reference version: -eps * log sum_{k,l} exp(h[k,l] - C[(i,j),(k,l)] / eps)."""
    t = torch.logsumexp(h[:, :, None, :] - Cx[None, None] / eps, dim=-1)       # over l
    t = torch.logsumexp(t[:, None, :, :] - Cy[None, :, :, None] / eps, dim=2)  # over k
    return -eps * t


def _softmin_mm(eps, Ky, Kx, h):
    """Same as _softmin_lse, as two max-shifted matmuls with K = exp(-C / eps)."""
    # m is only a numerical shift (its exact gradient is zero), so it's detached: otherwise
    # backprop computes that zero as 1 - sum(weights), and float32 rounding lands on the max cell.
    m = h.detach().amax(dim=-1, keepdim=True)              # (B, H, 1)
    t = torch.matmul(torch.exp(h - m), Kx).log_() + m      # over l   -> (B, H(k), W(j))
    m = t.detach().amax(dim=1, keepdim=True)               # (B, 1, W)
    s = torch.matmul(Ky, torch.exp(t - m)).log_() + m      # over k   -> (B, H(i), W(j))
    return -eps * s


def _no_autocast(device):
    try:
        return torch.autocast(device_type=device.type, enabled=False)
    except Exception:                                      # device type without autocast
        return contextlib.nullcontext()


def sinkhorn_divergence(a, b, eps=0.01, iters=50, mass_floor=1e-8):
    """
    a, b : (B, H, W) non-negative, each sums to 1 over (H, W).
    eps  : entropic blur, in squared grid units of a [0, 1] square. eps = 0.01 means a
           blur of about 0.1 of the grid width (~3 cells on a 32x32 grid).
    Returns: (B,) divergence per distribution.
    """
    with _no_autocast(a.device):
        return _sinkhorn_divergence(a, b, eps, iters, mass_floor)


def _sinkhorn_divergence(a, b, eps, iters, mass_floor):
    B, H, W = a.shape
    dt = torch.float32 if a.dtype in (torch.float16, torch.bfloat16) else a.dtype
    a, b = a.to(dt), b.to(dt)
    Cy, Cx = _grid_cost(H, a.device, dt), _grid_cost(W, a.device, dt)

    if max(Cy.max().item(), Cx.max().item()) / eps < _MAX_EXPONENT:
        Ky, Kx = torch.exp(-Cy / eps), torch.exp(-Cx / eps)   # symmetric
        softmin = lambda h: _softmin_mm(eps, Ky, Kx, h)
    else:                                                     # very small eps
        softmin = lambda h: _softmin_lse(eps, Cy, Cx, h)

    # Cells with exactly zero mass give log(0) = -inf; floor and renormalise.
    a = (a + mass_floor) / (a + mass_floor).sum(dim=(1, 2), keepdim=True)
    b = (b + mass_floor) / (b + mass_floor).sum(dim=(1, 2), keepdim=True)
    la, lb = a.log(), b.log()

    # Stack the four problems: rows [0:B) f_ab, [B:2B) g_ba, [2B:3B) f_aa, [3B:4B) g_bb.
    # f_ab is the softmin of g_ba weighted by b, g_ba of f_ab weighted by a, and so on.
    lw = torch.cat([lb, la, la, lb])

    def update(pot):
        f_ab, g_ba, f_aa, g_bb = pot.chunk(4)
        src = torch.cat([g_ba, f_ab, f_aa, g_bb])
        return softmin(lw + src / eps)

    with torch.no_grad():
        pot = torch.zeros(4 * B, H, W, device=a.device, dtype=dt)
        for _ in range(iters):
            # Symmetric, averaged updates (stable and converge fast).
            pot = 0.5 * (pot + update(pot))

    # Final update with autograd on (envelope theorem gradient).
    f_ab, g_ba, f_aa, g_bb = update(pot.detach()).chunk(4)

    S = (a * (f_ab - f_aa)).sum(dim=(1, 2)) + (b * (g_ba - g_bb)).sum(dim=(1, 2))
    return S
