"""Empirical risk analysis of next-step forecasts, STDN cut (cells with true > 10, both channels together).

Per run (each statistic computed per seed, then averaged over seeds):

  1. generalisation: RMSE on the train windows (the empirical risk training minimises), val and test, and the
     gap test - train. A run with a lower train RMSE but no lower test RMSE fits the train period better
     without generalising better.
  2. tail risk on test: each test window's RMSE (over its cells > 10), its median, 90th and 99th percentile,
     and CVaR 95 = the mean RMSE of the worst 5% of windows. Two runs with the same mean can differ here.
  3. risk by demand: test RMSE in each quartile of the window's true total (low to high demand).
  4. against --baseline: the test RMSE difference with a 95% interval from a block bootstrap over days
     (contiguous blocks of one day of windows, since neighbouring windows' errors are correlated), the
     share of resamples where the run is better, and the share of windows where it is better. This
     measures how much the test period's particular days decide a gap; the seed spread is separate.

  python tools/risk_eval.py checkpoints/workspaces/modes_nyctaxi_in8 checkpoints/workspaces/weather_nyctaxi_in8 \\
      --match '^distvolformer_(none|modes)' --baseline distvolformer_none
"""
import argparse
import collections
import glob
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_published import predictions                                    # noqa: E402
from trafficpred.utils import pick_device                                  # noqa: E402

STEPS_PER_DAY = {"nyctaxi": 48, "nycbike2": 48, "taxibj": 48, "bikenyc": 24}


def per_window(pred, true, threshold):
    """(N,) summed squared error and count of the kept cells (true > threshold) per window, step t+1."""
    pred, true = pred[:, :1], true[:, :1]
    keep = true >= threshold - 1e-3
    se = (((pred - true) * keep) ** 2).flatten(1).sum(1)
    return se.numpy(), keep.flatten(1).sum(1).double().numpy(), true.flatten(1).sum(1).numpy()


def rmse(se, n):
    return float(np.sqrt(se.sum() / max(n.sum(), 1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", help="checkpoints or workspace directories")
    ap.add_argument("--match", help="only runs whose name matches this regex")
    ap.add_argument("--baseline", help="regex of the run to compare every other run with (part 4)")
    ap.add_argument("--threshold", type=float, default=10.5)
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    device = pick_device()

    paths = []
    for p in args.paths:
        paths += sorted(glob.glob(os.path.join(p, "*.pt"))) if os.path.isdir(p) else [p]
    paths = [p for p in paths if not p.endswith(".last.pt") and "copy_last" not in os.path.basename(p)]
    paths = [p for p in paths if os.path.exists(p[:-3] + ".done")   # skip runs still training
             or not glob.glob(os.path.join(os.path.dirname(p), "*.done"))]   # (folders without markers: all)
    if args.match:
        paths = [p for p in paths if re.search(args.match, os.path.basename(p))]

    runs = collections.defaultdict(list)   # run -> per seed {split: (se, n)}, test totals
    dataset, ref_true = None, None
    for path in paths:
        seed = {}
        for split in ("train", "val", "test"):
            ck, pred, true = predictions(path, device, split, args.batch_size)
            dataset = dataset or ck["dataset"]
            if ck["dataset"] != dataset:
                raise SystemExit("give runs of one dataset at a time")
            se, n, total = per_window(pred, true, args.threshold)
            seed[split] = (se, n)
            if split == "test":
                if ref_true is None:
                    ref_true = total
                elif len(total) != len(ref_true) or not np.allclose(total, ref_true):
                    raise SystemExit(f"{os.path.basename(path)} was scored on other test windows than the first run")
        runs[re.sub(r"_seed\d+$", "", os.path.basename(path)[:-3])].append(seed)
        print(f"scored {os.path.basename(path)}", file=sys.stderr, flush=True)
    if not runs:
        raise SystemExit("no finished runs matched")

    def seed_mean(run, f):
        return float(np.mean([f(s) for s in runs[run]]))

    order = sorted(runs, key=lambda r: seed_mean(r, lambda s: rmse(*s["test"])))
    print(f"{dataset}: empirical risk analysis, t+1, cells with true > {args.threshold:g} (Start and End together); "
          "every number is a mean over seeds")

    print("\n1. generalisation: RMSE on each split (train = the empirical risk minimised; fold-val runs also trained on val)")
    print(f"  {'run':44}{'n':>3}{'train':>9}{'val':>9}{'test':>9}{'test - train':>14}")
    for r in order:
        tr, va, te = (seed_mean(r, lambda s, k=k: rmse(*s[k])) for k in ("train", "val", "test"))
        print(f"  {r[:44]:44}{len(runs[r]):3d}{tr:9.2f}{va:9.2f}{te:9.2f}{te - tr:+14.2f}")

    def window_rmse(s):
        se, n = s["test"]
        ok = n > 0
        return np.sqrt(se[ok] / n[ok])

    print("\n2. tail risk: per-window test RMSE (windows with any cell > threshold)")
    print(f"  {'run':44}{'mean':>9}{'median':>9}{'p90':>9}{'p99':>9}{'CVaR95':>9}")
    for r in order:
        q = lambda f: seed_mean(r, lambda s: f(window_rmse(s)))
        cvar = q(lambda w: np.sort(w)[-max(1, len(w) // 20):].mean())
        print(f"  {r[:44]:44}{seed_mean(r, lambda s: rmse(*s['test'])):9.2f}{q(np.median):9.2f}"
              f"{q(lambda w: np.quantile(w, .9)):9.2f}{q(lambda w: np.quantile(w, .99)):9.2f}{cvar:9.2f}")

    edges = np.quantile(ref_true, [0, .25, .5, .75, 1])
    bins = np.clip(np.searchsorted(edges, ref_true, side="right") - 1, 0, 3)
    print("\n3. risk by demand: test RMSE per quartile of the window's true total (Q1 = quietest)")
    print(f"  {'run':44}" + "".join(f"{f'Q{k + 1}':>9}" for k in range(4)))
    for r in order:
        cells = [seed_mean(r, lambda s, k=k: rmse(s["test"][0][bins == k], s["test"][1][bins == k])) for k in range(4)]
        print(f"  {r[:44]:44}" + "".join(f"{c:9.2f}" for c in cells))

    if args.baseline:
        base = [r for r in runs if re.search(args.baseline, r)]
        if len(base) != 1:
            raise SystemExit(f"--baseline matches {len(base)} runs: {base}")
        base = base[0]
        day = STEPS_PER_DAY.get(dataset, 48)
        N = len(ref_true)
        starts = np.arange(0, N, day)
        rng = np.random.default_rng(0)
        draws = rng.integers(0, len(starts), size=(args.boot, len(starts)))
        blocks = [np.arange(s, min(s + day, N)) for s in starts]

        def pooled(run):   # per-window squared error and count, averaged over the run's seeds
            return (np.mean([s["test"][0] for s in runs[run]], 0), np.mean([s["test"][1] for s in runs[run]], 0))

        bse, bn = pooled(base)
        print(f"\n4. against {base}: test RMSE difference (run - baseline; < 0 = run better), 95% interval from a "
              f"bootstrap over the {len(starts)} test days")
        print(f"  {'run':44}{'diff':>9}{'95% interval':>20}{'P(better)':>11}{'windows better':>16}")
        for r in order:
            if r == base:
                continue
            se, n = pooled(r)
            diff = rmse(se, n) - rmse(bse, bn)
            boot = []
            for d in draws:
                idx = np.concatenate([blocks[k] for k in d])
                boot.append(rmse(se[idx], n[idx]) - rmse(bse[idx], bn[idx]))
            lo, hi = np.quantile(boot, [.025, .975])
            ok = n > 0
            better = float(np.mean(se[ok] < bse[ok]))
            print(f"  {r[:44]:44}{diff:+9.2f}{f'[{lo:+.2f}, {hi:+.2f}]':>20}{np.mean(np.array(boot) < 0):11.0%}{better:16.0%}")


if __name__ == "__main__":
    main()
