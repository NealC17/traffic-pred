"""Check and time the fused GPU kernels (trafficpred/kernels.py) against the eager PyTorch code they replace, on
the shapes the real runs use: the TT Sinkhorn of one temporal block (batch 16, 4 heads, 5 iterations), forward +
backward, for PEMS08 (170 sensors, 24 frames x 12 sources) and BjTT (1260 roads, 27 x 12), and the unbalanced
and chained (24 x 24) variants. Prints the largest differences, the time per call and the peak memory.

  python tools/bench_kernels.py            # needs a CUDA GPU; takes ~10 s. Run it on a free GPU (e.g. on Sol)
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trafficpred import kernels   # noqa: E402

CASES = [   # name, sequences (batch x sensors), frames, sources, unbalanced
    ("PEMS08 TT block", 16 * 170, 24, 12, False),
    ("PEMS08 unbalanced", 16 * 170, 24, 12, True),
    ("PEMS08 chain (24 x 24)", 16 * 170, 24, 24, True),
    ("BjTT TT block", 16 * 1260, 27, 12, False),
]
HEADS, ITERS = 4, 5


def timed(fn, reps=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1e3, (torch.cuda.max_memory_allocated() - base) / 2 ** 20


def main():
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")
    if kernels.triton is None:
        raise SystemExit("needs Triton (pip install triton)")
    dev = torch.device("cuda")
    print(f"{torch.cuda.get_device_name()} | torch {torch.__version__} | triton {kernels.triton.__version__}")
    for name, B, F, S, unbal in CASES:
        g = torch.Generator(device="cpu").manual_seed(0)
        lg = (3 * torch.randn(B, HEADS, F, S, generator=g)).to(dev).requires_grad_(True)
        frac = (0.1 + 0.8 * torch.rand(HEADS, 1, 1, generator=g)).to(dev).requires_grad_(True) if unbal else None
        w = torch.randn(B, HEADS, F, S, generator=g).to(dev)
        col = float(torch.log(torch.tensor(F / S)))
        inputs = [t for t in (lg, frac) if t is not None]

        def step(fn):
            p = fn(lg, col, frac, ITERS)
            return p, torch.autograd.grad((p * w).sum(), inputs)

        (p0, g0), (p1, g1) = step(kernels.sinkhorn_plan_eager), step(kernels._fused)
        err = max([(p0 - p1).abs().max().item()]
                  + [((a - b).abs().max() / (a.abs().max() + 1e-12)).item() for a, b in zip(g0, g1)])
        te, me = timed(lambda: step(kernels.sinkhorn_plan_eager))
        tf, mf = timed(lambda: step(kernels._fused))
        print(f"{name:24s} {B * HEADS:7d} plans {F}x{S}: max diff {err:.1e} | eager {te:7.2f} ms {me:7.0f} MB | "
              f"fused {tf:7.2f} ms {mf:7.0f} MB | {te / tf:4.1f}x faster, {me / max(mf, 1e-9):4.1f}x less memory")


if __name__ == "__main__":
    main()
