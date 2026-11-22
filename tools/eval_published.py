"""Score checkpoints the way published grid-flow papers do, to compare with their tables.

Per forecast step and per channel, in raw units: RMSE, MAE and MAPE over the test cells whose true
value is at least --threshold (0 = every cell). Checkpoints of the same run with different seeds
are averaged (mean and spread). --ensemble adds, per run, the score of the seeds' averaged
prediction; --blend scores the average of whole runs (each run's seeds averaged first).
--split val scores the validation period instead, so a blend can be chosen there and only the
chosen one scored on test (--search-blends K ranks every blend of 2..K runs).

  # STDN (Yao et al. AAAI'19) on NYCTaxi: next step, pick-up / drop-off apart, only cells with true > 10
  python tools/eval_published.py checkpoints/workspaces/published_nyctaxi --threshold 10.5

  # DeepSTN+ / ST-ResNet on BikeNYC: next hour, every cell, both channels together
  python tools/eval_published.py checkpoints/workspaces/published_bikenyc

  # seed ensembles, and ST-ResNet + DistVolFormer averaged (each regex picks one run)
  python tools/eval_published.py checkpoints/workspaces/nextstep_nyctaxi_in8 --threshold 10 --ensemble \
      --blend '^stresnet' 'sinkhorn_l6-30_nyctaxi'

  # choose a blend on val, then score only that one on test
  python tools/eval_published.py checkpoints/workspaces/nextstep_nyctaxi_in8 --threshold 10 --split val --search-blends 3
"""
import argparse
import collections
import glob
import itertools
import os
import re
import statistics as st
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trafficpred.checkpoints import load_model, splits_for
from trafficpred.utils import pick_device, predict


@torch.no_grad()
def predictions(path, device, split="test", batch_size=128):
    """The checkpoint's predictions and targets on that split in raw units, (N, T, C, H, W) on the CPU."""
    model, ck = load_model(path, device)
    splits = splits_for(ck)
    lo, hi = ck["scale"]
    preds, trues = [], []
    for batch in torch.utils.data.DataLoader(getattr(splits, split), batch_size=batch_size):
        x, y = batch[0].to(device), batch[1]
        ext = batch[2].to(device) if len(batch) > 2 else None
        out = predict(model, x, ext)   # the external features go in as ext=, as in training
        pred = out[0] if isinstance(out, tuple) else out
        preds.append((pred.float().cpu() * (hi - lo) + lo).double())
        trues.append((y.float() * (hi - lo) + lo).double())
    return ck, torch.cat(preds), torch.cat(trues)


def metrics(pred, true, threshold):
    # targets come back from [0, 1] with float error (a true 10 can read 9.99998): never let that decide
    keep = (true >= threshold - 1e-3) if threshold > 0 else torch.ones_like(true, dtype=torch.bool)
    err = (pred - true) * keep
    ape = (err.abs() / true.clamp_min(1e-6)) * keep
    sums = {}
    for key, v in (("se", err ** 2), ("ae", err.abs()), ("ape", ape), ("n", keep.double())):
        per = v.sum(dim=(0, 3, 4))                               # (T, C)
        sums[key] = torch.cat([per, per.sum(dim=1, keepdim=True)], dim=1)
    n = sums["n"].clamp_min(1)
    return {"rmse": (sums["se"] / n).sqrt(), "mae": sums["ae"] / n, "mape": 100 * sums["ape"] / n}


def same_targets(name, true, reference):
    if reference is not None and (true.shape != reference.shape or not torch.equal(true, reference)):
        raise SystemExit(f"{name}: its windows differ from the other runs' (another split or horizon?); "
                         "it can't be averaged with them")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="checkpoints, or workspace directories (every *.pt in them)")
    ap.add_argument("--threshold", type=float, default=0, help="only cells with true value >= this; counts are integers, so 10.5 = STDN's true > 10, 5.5 = ST-SSL's true > 5")
    ap.add_argument("--steps", type=int, default=1, help="forecast steps to show (default 1: t+1)")
    ap.add_argument("--match", help="only runs whose name matches this regex")
    ap.add_argument("--split", choices=("test", "val"), default="test",
                    help="which period to score (default test; val to choose blends / configs fairly)")
    ap.add_argument("--ensemble", action="store_true",
                    help="also score each run's seed ensemble (the seeds' predictions averaged)")
    ap.add_argument("--blend", nargs="+", action="append", metavar="REGEX",
                    help="also score the average of these runs (one regex per run, each must match exactly "
                         "one run; each run's seeds are averaged first). Repeat for more blends")
    ap.add_argument("--search-blends", type=int, metavar="K",
                    help="also score every blend of 2..K runs and show the best 10 (use with --split val)")
    ap.add_argument("--batch-size", type=int, default=128,
                    help="windows per forward pass (lower it for big models: ST-SSL DistVolFormer needs 16)")
    args = ap.parse_args()
    device = pick_device()
    paths = []
    for p in args.paths:
        paths += sorted(glob.glob(os.path.join(p, "*.pt"))) if os.path.isdir(p) else [p]
    paths = [p for p in paths if not p.endswith(".last.pt")]
    groups, channels = collections.defaultdict(list), None
    sums, counts, reference = {}, collections.Counter(), None   # per run: sum of the seeds' predictions
    keep_preds = args.ensemble or args.blend or args.search_blends
    for p in paths:
        name = re.sub(r"_seed\d+$", "", os.path.basename(p)[:-3])
        if args.match and not re.search(args.match, name):
            continue
        print(f"scoring {os.path.basename(p)}", file=sys.stderr, flush=True)
        ck, pred, true = predictions(p, device, args.split, args.batch_size)
        channels = list(ck["channels"]) + ["all"]
        groups[name].append(metrics(pred, true, args.threshold))
        if keep_preds:
            same_targets(name, true, reference)
            reference = true
            sums[name] = sums[name] + pred if name in sums else pred
            counts[name] += 1
    ensembles = {}
    if args.ensemble:
        for name in sums:
            ensembles[f"{name} [ens]"] = (counts[name], metrics(sums[name] / counts[name], reference, args.threshold))
    for regexes in args.blend or []:
        members = []
        for rx in regexes:
            hits = [n for n in sums if re.search(rx, n)]
            if len(hits) != 1:
                raise SystemExit(f"--blend {rx!r} matches {len(hits)} runs ({', '.join(hits) or 'none'}); "
                                 "it must match exactly one")
            members.append(hits[0])
        blend = sum(sums[n] / counts[n] for n in members) / len(members)
        label = "[blend] " + " + ".join(members)
        ensembles[label] = (sum(counts[n] for n in members), metrics(blend, reference, args.threshold))
    if args.search_blends:
        means = {n: sums[n] / counts[n] for n in sums}
        found = []
        for k in range(2, args.search_blends + 1):
            for members in itertools.combinations(sorted(means), k):
                m = metrics(sum(means[n] for n in members) / k, reference, args.threshold)
                found.append((float(m["rmse"][0, -1]), members, m))
        for _, members, m in sorted(found, key=lambda f: f[0])[:10]:
            ensembles["[search] " + " + ".join(members)] = (sum(counts[n] for n in members), m)
    print(f"{args.split} cells with true value >= {args.threshold:g}; mean over seeds (± spread when > 1 seed)"
          + ("; [ens] / [blend] rows: one score of the averaged prediction (n = models in it)"
             if ensembles else ""))
    for label, (n, m) in ensembles.items():
        groups[label] = [m] * n   # one prediction, so no spread
    singles = set(ensembles)
    for t in range(args.steps):
        print(f"\n== t+{t + 1}")
        width = max(58, *map(len, groups))
        print(f"  {'run':{width}s} n  " + "  ".join(f"{c + ' RMSE':>14s} {'MAE':>5s} {'MAPE':>6s}" for c in channels))
        rows = sorted(groups.items(), key=lambda kv: st.mean(float(m["rmse"][t, -1]) for m in kv[1]))
        for name, ms in rows:
            cells = []
            for c in range(len(channels)):
                r = [float(m["rmse"][t, c]) for m in ms]
                spread = f"±{st.stdev(r):.2f}" if len(r) > 1 and name not in singles else "     "
                mape = (f"{st.mean(float(m['mape'][t, c]) for m in ms):5.1f}%" if args.threshold > 0
                        else "     -")   # MAPE is undefined on empty cells
                cells.append(f"{st.mean(r):8.3f}{spread} {st.mean(float(m['mae'][t, c]) for m in ms):5.2f} {mape}")
            print(f"  {name:{width}s} {len(ms)}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
