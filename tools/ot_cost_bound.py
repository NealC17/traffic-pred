"""Would a traffic-shaped OT cost help? An oracle bound: how much of a model's error is mass in the wrong
place along cheap moves, for different cost geometries.

A cost that is 0 on a set of allowed moves (origin cell -> k other cells) and infinite elsewhere. Every
cell of a forecast P may send any share of its mass along its allowed moves, once (one hop), with the
truth Y known (an oracle): the moved forecast P' minimizes ||P' - Y||^2. The RMSE left over is what no
loss or head built on that cost could go below by moving mass; what it removes is the most such a cost
can explain. Compared at the same k (same degrees of freedom), per allowed-move set:

  grid      the k nearest cells (squared grid distance: the cost used so far)
  od        the k most frequent trip destinations / origins in the train period (the OD kernel)
  random    k random cells (control: what any k extra degrees of freedom remove)

If 'od' doesn't beat 'grid' clearly, a traffic-shaped cost has nothing to add. If both are near
'random', the error isn't displaced mass at all.

  python tools/ot_cost_bound.py checkpoints/workspaces/tuned_models_nyctaxi_in8_ge10/distvolformer_tuned_nyctaxi_in8_out1.pt
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_published import predictions                                    # noqa: E402
from trafficpred.data import files                                         # noqa: E402
from trafficpred.data.datasets import DATASETS                             # noqa: E402


def neighbours(kind, k, H, W, kernel, rng):
    """(N, k) allowed destination cells per cell."""
    N = H * W
    yy, xx = np.divmod(np.arange(N), W)
    if kind == "grid":
        d = (yy[:, None] - yy[None]) ** 2 + (xx[:, None] - xx[None]) ** 2 + rng.random((N, N)) * 1e-3
    elif kind == "od":
        sym = kernel + kernel.T                    # trips either way
        d = -sym + rng.random((N, N)) * 1e-9
    else:
        d = rng.random((N, N))
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, 1)[:, :k]


def moved_rmse(P, Y, nbr, keep, steps, lr):
    """P, Y: (B, N) counts; nbr: (N, k). The oracle one-hop move of P toward Y; RMSE on keep."""
    P_t, Y_t = torch.tensor(P), torch.tensor(Y)
    B, N = P_t.shape
    k = nbr.shape[1]
    idx = torch.tensor(nbr)
    logits = torch.zeros(B, N, k + 1, dtype=torch.float64, requires_grad=True)   # last: stay
    with torch.no_grad():
        logits[..., -1] = 4.0                                                     # start near "stay"
    opt = torch.optim.Adam([logits], lr=lr)
    for _ in range(steps):
        share = torch.softmax(logits, -1)
        out = P_t[..., None] * share[..., :k]                                     # (B, N, k) sent i -> nbr
        moved = P_t * share[..., -1]
        moved = moved.index_add(1, idx.reshape(-1), out.reshape(B, -1))
        loss = ((moved - Y_t) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        share = torch.softmax(logits, -1)
        moved = (P_t * share[..., -1]).index_add(1, idx.reshape(-1), (P_t[..., None] * share[..., :k]).reshape(B, -1))
        return float(np.sqrt(((moved.numpy() - Y) ** 2)[keep].mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoint")
    ap.add_argument("--threshold", type=float, default=10.5, help="10.5 = STDN's true > 10")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    torch.set_num_threads(max(1, min(8, os.cpu_count() // 2)))

    ck, p, t = predictions(args.checkpoint, "cpu", "test", args.batch_size)
    spec = DATASETS[ck["dataset"]]
    P = p[:, 0].clamp_min(0).numpy()                                              # (B, C, H, W)
    Y = t[:, 0].numpy()
    B, C, H, W = P.shape
    kernel = np.load(os.path.join(files.ROOT, os.path.dirname(spec.od_file), "od_kernel.npy")).astype(np.float64)
    rng = np.random.default_rng(0)
    P2, Y2 = P.reshape(B * C, H * W), Y.reshape(B * C, H * W)
    keep = Y2 >= args.threshold - 1e-3
    base = np.sqrt(((P2 - Y2) ** 2)[keep].mean())
    total = np.sqrt(((P2 * (Y2.sum(1, keepdims=True) / np.maximum(P2.sum(1, keepdims=True), 1e-9)) - Y2) ** 2)[keep].mean())
    print(f"{spec.title} {os.path.basename(args.checkpoint)}: test RMSE on cells with true >= {args.threshold:g}")
    print(f"  forecast as is {base:.3f}   (rescaled to the true frame total: {total:.3f})")
    print(f"  oracle one-hop moves, RMSE left (share of MSE removed):")
    print(f"  {'k':>4}" + "".join(f"{kind:>20}" for kind in ("grid", "od", "random")))
    for k in args.ks:
        row = f"  {k:4d}"
        for kind in ("grid", "od", "random"):
            r = moved_rmse(P2, Y2, neighbours(kind, k, H, W, kernel, rng), keep, args.steps, args.lr)
            row += f"      {r:7.3f} ({100 * (1 - r ** 2 / base ** 2):4.1f}%)"
        print(row, flush=True)


if __name__ == "__main__":
    main()
