"""Spatial (operational) error of forecasts: the GeOT error of Wiedemann, Uscidda & Raubal (STRC 2025),
"Bridging prediction and optimization in on-demand transportation systems with Optimal Transport".

For every test map (one channel of one target frame) the exact partial OT between the predicted and the true
counts, with the cost the grid distance between cell centres (in cells). The two maps have different totals,
so, as in the paper (after Chapel et al. 2020), a dummy location takes the difference: the side with less
mass gets the excess at the dummy, and moving mass to or from it costs phi. The error is the minimal total
cost (trips x cells) of turning the forecast into the truth:

  phi = 0           spatial error only: the total's mismatch is free (the paper's Fig. 2 / Sec. 3.3 metric)
  phi = q10 of C    the paper's business-cost setting (its footnote 2): a total mismatch costs as much as a
                    short relocation

Also: RMSE on all cells and the STDN cut (true > 10), for the same windows. Per run, mean over seeds (± spread).

  python tools/geot_eval.py checkpoints/workspaces/unbalanced_ot_nyctaxi --step 1
"""
import argparse
import collections
import glob
import os
import re
import statistics as st
import sys

import numpy as np
import ot
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_published import predictions                                    # noqa: E402
from trafficpred.utils import pick_device                                  # noqa: E402


def partial_ot(pred, true, C_ext, phi):
    """pred, true: (n,) non-negative counts; C_ext: (n+1, n+1) with the dummy row / column set to phi."""
    sp, st_ = pred.sum(), true.sum()
    s = min(sp, st_)
    a = np.append(pred, st_ - s)          # source: prediction + dummy holding the true excess
    b = np.append(true, sp - s)           # target: truth + dummy holding the predicted excess
    if a.sum() <= 0:
        return 0.0
    return float(ot.emd2(a, b, C_ext, numItermax=1_000_000))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", help="checkpoints or workspace directories")
    ap.add_argument("--step", type=int, default=1, help="which forecast step to score (default t+1)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-windows", type=int, default=0, help="score only the first N windows (0 = all)")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    device = pick_device()
    torch.set_num_threads(max(1, min(8, os.cpu_count() // 2)))

    paths = []
    for p in args.paths:
        paths += sorted(glob.glob(os.path.join(p, "*.pt"))) if os.path.isdir(p) else [p]
    paths = [p for p in paths if not p.endswith(".last.pt")]

    results = collections.defaultdict(list)
    C_ext = {}
    for path in paths:
        ck, pred, true = predictions(path, device, args.split, args.batch_size)
        if args.step > pred.shape[1]:
            print(f"skip {os.path.basename(path)}: only {pred.shape[1]} steps", file=sys.stderr)
            continue
        P = pred[:, args.step - 1].clamp_min(0).numpy()        # (B, C, H, W) raw counts
        Y = true[:, args.step - 1].numpy()
        if args.max_windows:
            P, Y = P[:args.max_windows], Y[:args.max_windows]
        B, Cc, H, W = P.shape
        if (H, W) not in C_ext:
            yy, xx = np.divmod(np.arange(H * W), W)
            C = np.sqrt((yy[:, None] - yy[None]) ** 2 + (xx[:, None] - xx[None]) ** 2)
            q10 = float(np.quantile(C[C > 0], 0.10))
            mats = {}
            for name, phi in (("phi0", 0.0), ("phiq10", q10)):
                M = np.full((H * W + 1, H * W + 1), phi)
                M[:H * W, :H * W] = C
                M[-1, -1] = 0.0
                mats[name] = M
            C_ext[(H, W)] = (mats, q10)
        mats, q10 = C_ext[(H, W)]
        geo = {k: [] for k in mats}
        for b in range(B):
            for c in range(Cc):
                p_, t_ = P[b, c].ravel().astype(np.float64), Y[b, c].ravel().astype(np.float64)
                for k, M in mats.items():
                    geo[k].append(partial_ot(p_, t_, M, k))
        err = P - Y
        keep = Y >= 10.5 - 1e-3
        out = dict(phi0=np.mean(geo["phi0"]), phiq10=np.mean(geo["phiq10"]),
                   rmse=float(np.sqrt((err ** 2).mean())), rmse10=float(np.sqrt((err[keep] ** 2).mean())),
                   total=float(np.abs(P.sum((-2, -1)) - Y.sum((-2, -1))).mean()))
        run = re.sub(r"_seed\d+$", "", os.path.basename(path)[:-3])
        results[run].append(out)
        print(f"done {os.path.basename(path)}", file=sys.stderr, flush=True)

    print(f"{args.split}, step t+{args.step}: GeOT error per map (exact partial OT, cost = grid distance in cells; "
          f"trips x cells), phi = 0 and phi = q10 of the cost ({q10:.2f} cells); |total error| per map; RMSE")
    head = ["GeOT phi=0", "GeOT phi=q10", "|total err|", "RMSE all", "RMSE >10"]
    print(f"  {'run':46}{'n':>3}" + "".join(f"{h:>16}" for h in head))
    keys = ("phi0", "phiq10", "total", "rmse", "rmse10")
    for run, seeds in sorted(results.items(), key=lambda kv: np.mean([s["phi0"] for s in kv[1]])):
        cells = []
        for k in keys:
            v = [s[k] for s in seeds]
            cells.append(f"{np.mean(v):9.2f}" + (f"±{st.stdev(v):<5.2f}" if len(v) > 1 else " " * 6))
        print(f"  {run[:46]:46}{len(seeds):3d}" + "".join(f"{c:>16}" for c in cells), flush=True)


if __name__ == "__main__":
    main()
