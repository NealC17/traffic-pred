"""Fused GPU kernels (Triton) for the hot spots that PyTorch runs as many small kernels.

tt_sinkhorn_plan: the temporal latent transport's (TT, --time-transport) log-domain Sinkhorn. Eager PyTorch runs
every iteration as a logsumexp + subtract over the whole (B, heads, F, S) logit tensor (a few kernel launches and
a full read / write of it each), and autograd keeps every iterate for the backward pass. Here one Triton program
holds one (F, S) plan (F <= 64 frames x S <= 64 sources: PEMS 24 x 12, BjTT 27 x 12) in registers, runs all the
iterations there and writes only the plan; the backward pass recomputes the iterates in registers instead of
storing them. Same maths as the eager loop (the reference `sinkhorn_plan_eager`, kept for the CPU, other shapes and
the tests), to float32 rounding.

On by default on CUDA when Triton is available (TRAFFICPRED_TT_KERNEL=0 turns it off, =1 forces it, e.g. with
TRITON_INTERPRET=1 to test it on the CPU). tools/bench_kernels.py checks it against the eager loop and times both.
"""
import os

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:   # no Triton (CPU-only installs, MPS): the eager loop
    triton = None

NEG = -1.0e30   # "no entry" for the padding of a plan up to the block size (finite: no inf - inf)


def sinkhorn_plan_eager(logits, col, frac, iters):
    """(B, h, F, S) logits -> plan exp(logits after `iters` Sinkhorn steps): even steps normalise the rows (each
    destination gets 1), odd steps move the columns towards log mass `col` (a float or a tensor broadcastable to
    (B, h, 1, S)), by the fraction `frac` ((h, 1, 1), unbalanced) or all the way (None)."""
    for i in range(iters):
        if i % 2 == 0:
            logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        else:
            step = col - logits.logsumexp(dim=-2, keepdim=True)
            logits = logits + (step if frac is None else frac * step)
    return logits.exp()


if triton is not None:
    _NEG = tl.constexpr(NEG)   # a jit'ed kernel may only read constexpr globals

    @triton.jit
    def _lse(x, axis: tl.constexpr):
        mx = tl.max(x, axis=axis, keep_dims=True)
        return tl.log(tl.sum(tl.exp(x - mx), axis=axis, keep_dims=True)) + mx

    @triton.jit
    def _step(l, col, frac, valid, i: tl.constexpr):
        if i % 2 == 0:
            l = l - _lse(l, 1)
        else:
            l = l + frac * (col - _lse(l, 0))
        return tl.where(valid, l, _NEG)

    @triton.jit
    def _fwd_kernel(L, COL, FRAC, P, F, S, H, ITERS: tl.constexpr, BF: tl.constexpr, BS: tl.constexpr):
        m = tl.program_id(0)
        f = tl.arange(0, BF)[:, None]
        s = tl.arange(0, BS)[None, :]
        valid = (f < F) & (s < S)
        offs = m * F * S + f * S + s
        l = tl.load(L + offs, mask=valid, other=_NEG)
        col = tl.load(COL + m * S + s, mask=s < S, other=0.0)
        frac = tl.load(FRAC + m % H)
        for i in tl.static_range(ITERS):
            l = _step(l, col, frac, valid, i)
        tl.store(P + offs, tl.exp(l), mask=valid)

    @triton.jit
    def _bwd_kernel(L, COL, FRAC, P, GP, GL, GCOL, GFRAC, F, S, H,
                    ITERS: tl.constexpr, BF: tl.constexpr, BS: tl.constexpr):
        m = tl.program_id(0)
        f = tl.arange(0, BF)[:, None]
        s = tl.arange(0, BS)[None, :]
        valid = (f < F) & (s < S)
        offs = m * F * S + f * S + s
        l0 = tl.load(L + offs, mask=valid, other=_NEG)
        col = tl.load(COL + m * S + s, mask=s < S, other=0.0)
        frac = tl.load(FRAC + m % H)
        p = tl.load(P + offs, mask=valid, other=0.0)
        g = tl.load(GP + offs, mask=valid, other=0.0) * p            # d loss / d final logits
        gcol = tl.zeros((1, BS), dtype=tl.float32)
        gfrac = tl.zeros((1, 1), dtype=tl.float32)
        for i in tl.static_range(ITERS - 1, -1, -1):                # the steps in reverse
            l = l0
            for j in tl.static_range(i):                             # recompute iterate i in registers
                l = _step(l, col, frac, valid, j)
            if i % 2 == 0:                                           # l' = l - lse_row(l)
                sm = tl.exp(l - _lse(l, 1))
                g = g - sm * tl.sum(g, axis=1, keep_dims=True)
            else:                                                    # l' = l + frac (col - lse_col(l))
                lc = _lse(l, 0)
                sm = tl.exp(l - lc)
                cs = tl.sum(g, axis=0, keep_dims=True)
                gcol += frac * cs
                gfrac += tl.sum(tl.where(s < S, cs * (col - lc), 0.0), axis=1, keep_dims=True)
                g = g - frac * sm * cs
            g = tl.where(valid, g, 0.0)
        tl.store(GL + offs, g, mask=valid)
        tl.store(GCOL + m * S + s, gcol, mask=s < S)
        tl.store(GFRAC + m + tl.zeros((1, 1), dtype=tl.int32), gfrac)

    class _TTSinkhorn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits, col, frac, iters):
            B, h, F, S = logits.shape
            L = logits.float().contiguous()
            plan = torch.empty_like(L)
            blocks = dict(BF=max(16, triton.next_power_of_2(F)), BS=max(16, triton.next_power_of_2(S)))
            _fwd_kernel[(B * h,)](L, col, frac, plan, F, S, h, ITERS=iters, **blocks)
            ctx.save_for_backward(L, col, frac, plan)
            ctx.iters, ctx.blocks = iters, blocks
            return plan

        @staticmethod
        def backward(ctx, grad_plan):
            L, col, frac, plan = ctx.saved_tensors
            B, h, F, S = L.shape
            gl, gcol = torch.empty_like(L), torch.empty_like(col)
            gfrac = torch.empty(B * h, device=L.device, dtype=torch.float32)
            _bwd_kernel[(B * h,)](L, col, frac, plan, grad_plan.float().contiguous(), gl, gcol, gfrac,
                                  F, S, h, ITERS=ctx.iters, **ctx.blocks)
            return gl, gcol, gfrac.view(B, h).sum(0), None


def _use_kernel(logits):
    flag = os.environ.get("TRAFFICPRED_TT_KERNEL", "")
    if triton is None or flag == "0":
        return False
    F, S = logits.shape[-2:]
    if F > 64 or S > 64:
        return False
    return flag == "1" or logits.is_cuda


_checked = {}   # device -> the kernel passed its self-check there (False: failed, use the eager loop)


def _key(device):   # cuda and cuda:0 are the same device
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def _self_check(device):
    """Once per device: the kernel against the eager loop on a small random plan (values and gradients), so a
    Triton build or GPU that miscompiles it falls back to the eager loop instead of training on wrong plans."""
    device = _key(device)
    if device in _checked:
        return _checked[device]
    try:
        with torch.enable_grad(), torch.autocast(device.type, enabled=False):   # also when first called in eval
            return _run_check(device)
    except Exception as e:   # no compiler for this GPU, Triton bug, ...
        print(f"[kernels] fused TT Sinkhorn unavailable on {device} ({type(e).__name__}: {e}); using the eager loop")
        _checked[device] = False
        return False


def _run_check(device):
    g = torch.Generator(device="cpu").manual_seed(0)
    lg = (3 * torch.randn(2, 4, 27, 12, generator=g)).to(device).requires_grad_(True)
    frac = (0.1 + 0.8 * torch.rand(4, 1, 1, generator=g)).to(device).requires_grad_(True)
    w = torch.randn(2, 4, 27, 12, generator=g).to(device)
    ref = sinkhorn_plan_eager(lg, 0.8, frac, 5)
    r_grads = torch.autograd.grad((ref * w).sum(), (lg, frac))
    out = _fused(lg, 0.8, frac, 5)
    o_grads = torch.autograd.grad((out * w).sum(), (lg, frac))
    err = max([(ref - out).abs().max().item()]
              + [((a - b).abs().max() / (a.abs().max() + 1e-12)).item() for a, b in zip(r_grads, o_grads)])
    ok = err < 1e-4
    if not ok:
        print(f"[kernels] fused TT Sinkhorn is off by {err:.1e} on {device}; using the eager loop")
    _checked[device] = ok
    return ok


def prepare(device):
    """Run the self-check on `device` now (train.py calls this before training). It must not run inside a model's
    forward pass during training: gradient checkpointing replays that pass and counts the tensors it saves."""
    device = torch.device(device)
    if device.type == "cuda" and triton is not None and os.environ.get("TRAFFICPRED_TT_KERNEL", "") != "0":
        ok = _self_check(device)
        print(f"[kernels] fused TT Sinkhorn {'on' if ok else 'off'} ({torch.cuda.get_device_name(device)})")


def tt_sinkhorn_plan(logits, col, frac, iters):
    """sinkhorn_plan_eager's result, fused on the GPU (see the module docstring). The kernel is used once the
    self-check has passed on this device: after prepare(), or a first call made without gradients (eval)."""
    if iters < 1 or not _use_kernel(logits):
        return sinkhorn_plan_eager(logits, col, frac, iters)
    ok = _checked.get(_key(logits.device))
    if ok is None and not torch.is_grad_enabled():
        ok = _self_check(logits.device)
    if not ok:
        return sinkhorn_plan_eager(logits, col, frac, iters)
    return _fused(logits, col, frac, iters)


def _fused(logits, col, frac, iters):
    B, h, F, S = logits.shape
    dev = logits.device
    col_t = torch.as_tensor(col, dtype=torch.float32, device=dev).expand(B, h, 1, S).reshape(B * h, S).contiguous()
    frac_t = (torch.ones(h, device=dev) if frac is None else frac.reshape(h).float()).contiguous()
    return _TTSinkhorn.apply(logits, col_t, frac_t, iters).to(logits.dtype)


def padded_sdpa(q, k, v, dropout_p=0.0):
    """F.scaled_dot_product_attention for any head size. PyTorch's memory-efficient / flash kernels need the head
    size to be a multiple of 8 (fp32: 4); otherwise it silently falls back to the plain softmax(q k^T) v, which
    holds the whole L x L score matrix (BjTT: 1260 roads, 36 GB at batch 16). Zero-padding q, k, v to the next
    multiple of 8 keeps q k^T and the first d output columns exactly the same; the scale stays 1 / sqrt(d)."""
    d = q.shape[-1]
    pad = -d % 8
    if pad == 0 or not q.is_cuda:
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
    q, k, v = (F.pad(t, (0, pad)) for t in (q, k, v))
    return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, scale=d ** -0.5)[..., :d]


def mha_self_attention(attn, x):
    """attn(x, x, x, need_weights=False)[0] for an nn.MultiheadAttention(batch_first=True) without masks, through
    padded_sdpa (same weights, same maths; attention dropout in training as attn does)."""
    B, L, D = x.shape
    h = attn.num_heads
    q, k, v = F.linear(x, attn.in_proj_weight, attn.in_proj_bias).chunk(3, dim=-1)
    q, k, v = (t.reshape(B, L, h, D // h).transpose(1, 2) for t in (q, k, v))
    out = padded_sdpa(q, k, v, attn.dropout if attn.training else 0.0)
    return attn.out_proj(out.transpose(1, 2).reshape(B, L, D))
