"""
Debiased Sinkhorn divergence S_eps(a, b) between distributions on a 2-D grid.

    OT_eps(a, b)  = entropic optimal transport cost
    S_eps(a, b)   = OT_eps(a, b) - 1/2 OT_eps(a, a) - 1/2 OT_eps(b, b)

S_eps is >= 0, equals 0 only when a == b, and goes to the Wasserstein-2 cost
(||x - y||^2 / 2) as eps -> 0.

Memory trick: on a regular grid the squared-Euclidean cost splits into a row part
plus a column part, C[(i,j),(k,l)] = (Cy[i,k] + Cx[j,l]). So the log-sum-exp over
all H*W source cells can be done as two 1-D log-sum-exps (one per axis). That is
O(H*W*(H+W)) per distribution instead of O((H*W)^2): for a 32x32 grid, about 16x less
memory than a dense 1024x1024 cost matrix, which is what makes it trainable.

Gradients: the Sinkhorn loop runs without autograd, then one last update is done with
autograd on. By the envelope theorem, that gives the correct gradient of OT_eps with
respect to a and b (the same scheme GeomLoss uses).
"""
import torch


def _grid_cost(n, device, dtype):
    # Cell centres in [0, 1]; cost is |x - y|^2 / 2 along one axis.
    x = (torch.arange(n, device=device, dtype=dtype) + 0.5) / n
    return (x[:, None] - x[None, :]) ** 2 / 2


def _softmin(eps, Cy, Cx, h):
    """
    Returns -eps * log sum_{k,l} exp(h[k,l] - C[(i,j),(k,l)] / eps) for every (i, j).
    h: (B, H, W) holds log-weights plus potential / eps.
    """
    # Over the column axis l:  (B, H, 1, W) - (1, 1, W, W) -> lse over last dim -> (B, H, W)
    t = torch.logsumexp(h[:, :, None, :] - Cx[None, None] / eps, dim=-1)
    # Over the row axis k:     (B, 1, H, W) with Cy (H, H) -> lse over k -> (B, H, W)
    t = torch.logsumexp(t[:, None, :, :] - Cy[None, :, :, None] / eps, dim=2)
    return -eps * t


def sinkhorn_divergence(a, b, eps=0.01, iters=50, mass_floor=1e-8):
    """
    a, b : (B, H, W) non-negative, each sums to 1 over (H, W).
    eps  : entropic blur, in squared grid units of a [0, 1] square. eps = 0.01 means a
           blur of about 0.1 of the grid width (~3 cells on a 32x32 grid).
    Returns: (B,) divergence per distribution.
    """
    B, H, W = a.shape
    dt = torch.float32 if a.dtype in (torch.float16, torch.bfloat16) else a.dtype
    a, b = a.to(dt), b.to(dt)
    Cy, Cx = _grid_cost(H, a.device, dt), _grid_cost(W, a.device, dt)

    # Cells with exactly zero mass give log(0) = -inf; floor and renormalise.
    a = (a + mass_floor) / (a + mass_floor).sum(dim=(1, 2), keepdim=True)
    b = (b + mass_floor) / (b + mass_floor).sum(dim=(1, 2), keepdim=True)
    la, lb = a.log(), b.log()

    def sm(lw, pot):  # softmin of a potential weighted by log-weights lw
        return _softmin(eps, Cy, Cx, lw + pot / eps)

    with torch.no_grad():
        f_ab, g_ba = torch.zeros_like(a), torch.zeros_like(b)
        f_aa, g_bb = torch.zeros_like(a), torch.zeros_like(b)
        for _ in range(iters):
            # Symmetric, averaged updates (stable and converge fast).
            ft, gt = sm(lb, g_ba), sm(la, f_ab)
            f_ab, g_ba = 0.5 * (f_ab + ft), 0.5 * (g_ba + gt)
            f_aa = 0.5 * (f_aa + sm(la, f_aa))
            g_bb = 0.5 * (g_bb + sm(lb, g_bb))

    # Final update with autograd on (envelope theorem gradient).
    f_ab, g_ba = sm(lb, g_ba.detach()), sm(la, f_ab.detach())
    f_aa, g_bb = sm(la, f_aa.detach()), sm(lb, g_bb.detach())

    S = (a * (f_ab - f_aa)).sum(dim=(1, 2)) + (b * (g_ba - g_bb)).sum(dim=(1, 2))
    return S