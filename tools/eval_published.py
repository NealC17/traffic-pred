"""Score checkpoints the way published grid-flow papers do, to compare with their tables.

Per forecast step and per channel, in raw units: RMSE, MAE and MAPE over the test cells whose true
value is at least --threshold (0 = every cell). Checkpoints of the same run with different seeds
are averaged (mean and spread).

  # STDN (Yao et al. AAAI'19) on NYCTaxi: next step, pick-up / drop-off apart, cells < 10 dropped
  python tools/eval_published.py checkpoints/workspaces/published_nyctaxi --threshold 10

  # DeepSTN+ / ST-ResNet on BikeNYC: next hour, every cell, both channels together
  python tools/eval_published.py checkpoints/workspaces/published_bikenyc
"""
import argparse
import collections
import glob
import os
import re
import statistics as st
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trafficpred.checkpoints import load_model, splits_for


@torch.no_grad()
def score(path, threshold, device):
    model, ck = load_model(path, device)
    splits = splits_for(ck)
    lo, hi = ck["scale"]
    loader = torch.utils.data.DataLoader(splits.test, batch_size=128)
    T, C = ck["t_out"], len(ck["channels"])
    sums = {k: torch.zeros(T, C + 1, dtype=torch.float64) for k in ("se", "ae", "ape", "n")}
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        out = model(x, *[b.to(device) for b in batch[2:]])
        pred = out[0] if isinstance(out, tuple) else out
        pred = pred * (hi - lo) + lo
        true = y * (hi - lo) + lo
        keep = (true >= threshold) if threshold > 0 else torch.ones_like(true, dtype=torch.bool)
        err = (pred - true) * keep
        ape = (err.abs() / true.clamp_min(1e-6)) * keep
        for key, v in (("se", err ** 2), ("ae", err.abs()), ("ape", ape), ("n", keep.double())):
            per = v.double().sum(dim=(0, 3, 4)).cpu()            # (T, C)
            sums[key][:, :C] += per
            sums[key][:, C] += per.sum(dim=1)
    n = sums["n"].clamp_min(1)
    return ck, {"rmse": (sums["se"] / n).sqrt(), "mae": sums["ae"] / n, "mape": 100 * sums["ape"] / n}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="checkpoints, or workspace directories (every *.pt in them)")
    ap.add_argument("--threshold", type=float, default=0, help="only cells with true value >= this (STDN: 10)")
    ap.add_argument("--steps", type=int, default=1, help="forecast steps to show (default 1: t+1)")
    ap.add_argument("--match", help="only runs whose name matches this regex")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = []
    for p in args.paths:
        paths += sorted(glob.glob(os.path.join(p, "*.pt"))) if os.path.isdir(p) else [p]
    paths = [p for p in paths if not p.endswith(".last.pt")]
    groups, channels = collections.defaultdict(list), None
    for p in paths:
        name = re.sub(r"_seed\d+$", "", os.path.basename(p)[:-3])
        if args.match and not re.search(args.match, name):
            continue
        ck, m = score(p, args.threshold, device)
        channels = list(ck["channels"]) + ["all"]
        groups[name].append(m)
    print(f"test cells with true value >= {args.threshold:g}; mean over seeds (± spread when > 1 seed)")
    for t in range(args.steps):
        print(f"\n== t+{t + 1}")
        print(f"  {'run':58s} n  " + "  ".join(f"{c + ' RMSE':>14s} {'MAE':>5s} {'MAPE':>6s}" for c in channels))
        rows = sorted(groups.items(), key=lambda kv: st.mean(float(m["rmse"][t, -1]) for m in kv[1]))
        for name, ms in rows:
            cells = []
            for c in range(len(channels)):
                r = [float(m["rmse"][t, c]) for m in ms]
                spread = f"±{st.stdev(r):.2f}" if len(r) > 1 else "     "
                mape = (f"{st.mean(float(m['mape'][t, c]) for m in ms):5.1f}%" if args.threshold > 0
                        else "     -")   # MAPE is undefined on empty cells
                cells.append(f"{st.mean(r):8.3f}{spread} {st.mean(float(m['mae'][t, c]) for m in ms):5.2f} {mape}")
            print(f"  {name[:58]:58s} {len(ms)}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
