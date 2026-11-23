"""Runs against the published STDN-protocol results (next step, RMSE / MAPE on cells with true > 10, per
channel Start / End), with a verdict on whether each margin is beyond seed noise.

Per run (seeds scored apart): mean and spread (sample sd) over seeds, and the seed ensemble (the seeds'
predictions averaged; published numbers are single models, so the seed mean is the fair comparison).
Verdict per target and channel, from the seed mean m, the seed sd s and n seeds:

  BETTER      m + 2 s / sqrt(n) < target   (beats it by more than ~2 standard errors)
  better?     m < target, within that margin (not distinguishable from a tie)
  worse       m > target

  python tools/compare_published.py checkpoints/workspaces/frame_encoder_nyctaxi_in8
"""
import argparse
import collections
import glob
import math
import os
import re
import statistics as st
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_published import metrics, predictions                           # noqa: E402
from trafficpred.utils import pick_device                                  # noqa: E402

# verified from the papers' own tables (see the project's published-targets notes); RMSE Start / End, MAPE %
TARGETS = {
    "nyctaxi": [("STDN (AAAI'19)", (24.10, 19.05), (16.30, 16.25)),
                ("Wang et al. 2020, no external data", (23.69, 18.78), None),
                ("Wang et al. 2020, with weather / holidays", (22.37, 17.78), (15.45, 15.52))],
    "nycbike2": [("STDN (AAAI'19)", (8.85, 8.15), (21.84, 20.87)),
                 ("Wang et al. 2020, with weather / holidays", (8.32, 7.66), (21.09, 19.82))],
}


def verdict(values, target):
    m = st.mean(values)
    if m > target:
        return "worse"
    se = st.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else float("inf")
    return "BETTER" if m + 2 * se < target else "better?"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", help="checkpoints or workspace directories")
    ap.add_argument("--match", help="only runs whose name matches this regex")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    device = pick_device()

    paths = []
    for p in args.paths:
        paths += sorted(glob.glob(os.path.join(p, "*.pt"))) if os.path.isdir(p) else [p]
    paths = [p for p in paths if not p.endswith(".last.pt") and "copy_last" not in os.path.basename(p)]
    if args.match:
        paths = [p for p in paths if re.search(args.match, os.path.basename(p))]

    runs = collections.defaultdict(list)       # run -> [(rmse (S, E), mape (S, E), pred)]
    dataset, true_ref = None, None
    for path in paths:
        ck, pred, true = predictions(path, device, "test", args.batch_size)
        if ck["t_out"] != 1 and pred.shape[1] != 1:
            pred, true = pred[:, :1], true[:, :1]
        dataset = dataset or ck["dataset"]
        if ck["dataset"] != dataset:
            raise SystemExit("give runs of one dataset at a time")
        if true_ref is None:
            true_ref = true
        elif not torch.equal(true, true_ref):
            raise SystemExit(f"{os.path.basename(path)} was scored on other windows than the first run")
        m = metrics(pred, true, 10.5)
        runs[re.sub(r"_seed\d+$", "", os.path.basename(path)[:-3])].append(
            ((m["rmse"][0, 0].item(), m["rmse"][0, 1].item()), (m["mape"][0, 0].item(), m["mape"][0, 1].item()), pred))
        print(f"scored {os.path.basename(path)}", file=sys.stderr, flush=True)
    if dataset not in TARGETS:
        raise SystemExit(f"no published STDN-protocol targets for {dataset}")

    print(f"{dataset}, STDN protocol (test, t+1, cells with true > 10): seed mean ± sd, then the seed ensemble")
    print(f"  {'run':42}{'n':>3}{'Start RMSE':>16}{'End RMSE':>16}{'Start MAPE':>13}{'End MAPE':>11}"
          f"{'ens RMSE S / E':>18}")
    for name, target, mape in TARGETS[dataset]:
        mp = f"{mape[0]:12.2f}%{mape[1]:10.2f}%" if mape else f"{'-':>13}{'-':>11}"
        print(f"  {'published: ' + name:45.45}{target[0]:13.2f}{target[1]:16.2f}{mp}")
    order = sorted(runs.items(), key=lambda kv: st.mean(sum(r[0]) for r in kv[1]))
    for run, seeds in order:
        rm = [[s[0][c] for s in seeds] for c in (0, 1)]
        mp = [[s[1][c] for s in seeds] for c in (0, 1)]
        sd = lambda v: f"±{st.stdev(v):.2f}" if len(v) > 1 else "     "
        ens = metrics(torch.stack([s[2] for s in seeds]).mean(0), true_ref, 10.5)["rmse"][0]
        print(f"  {run[:42]:42}{len(seeds):3d}{st.mean(rm[0]):11.2f}{sd(rm[0]):>5}{st.mean(rm[1]):11.2f}{sd(rm[1]):>5}"
              f"{st.mean(mp[0]):12.2f}%{st.mean(mp[1]):10.2f}%{ens[0].item():10.2f} / {ens[1].item():.2f}")

    print("\nverdict vs each published result (RMSE Start / End; BETTER = by more than 2 standard errors over seeds)")
    for run, seeds in order:
        rm = [[s[0][c] for s in seeds] for c in (0, 1)]
        parts = []
        for name, target, _ in TARGETS[dataset]:
            short = name.split(" (")[0].replace("Wang et al. 2020, ", "Wang ")
            parts.append(f"{short}: {verdict(rm[0], target[0])} / {verdict(rm[1], target[1])}")
        print(f"  {run[:42]:42} " + "   ".join(parts))


if __name__ == "__main__":
    main()
