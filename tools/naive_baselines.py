"""Score training-free baselines on the same windows and metric as tools/eval_published.py.

  last        the most recent frame (persistence)
  day         the same time one day back
  week        the same time one week back (seasonal naive)
  ha_day      mean over the train period for that time of day
  ha_week     mean over the train period for that time of day and day of week

The targets are taken from the dataset's own split windows (and checked against them), so the
numbers line up with the trained models' rows for the same --tin.

  python tools/naive_baselines.py --dataset nyctaxi --tin 8 --threshold 10
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.eval_published import metrics
from trafficpred.data import files
from trafficpred.data.datasets import DATASETS, get_splits


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tin", type=int, required=True, help="input length of the models compared with")
    ap.add_argument("--threshold", type=float, default=0)
    ap.add_argument("--split", choices=("val", "test"), default="test")
    args = ap.parse_args()

    spec = DATASETS[args.dataset]
    *head, last = spec.files()
    if head:
        raise SystemExit("only single-file grid datasets (NYCTaxi, NYCBike2, ...) are supported")
    frames = np.asarray(files.load(last), dtype=np.float64)              # (T, C, H, W)
    cut_test = len(frames) - spec.test_frames
    cut_val = cut_test - spec.val_frames
    start, stop = (cut_val, cut_test) if args.split == "val" else (cut_test, len(frames))
    targets = np.arange(start + args.tin, stop)                          # next-step targets of the windows

    splits = get_splits(args.dataset, args.tin, 1)
    ds = getattr(splits, args.split)
    true_ref = torch.stack([ds[i][1][0] for i in range(len(ds))]).double() * splits.span + splits.lo
    true = torch.from_numpy(frames[targets])
    if true.shape != true_ref.shape or (true - true_ref).abs().max() > 1e-3:
        raise SystemExit("targets don't match the split's windows; window alignment changed?")

    spd, week = spec.steps_per_day, 7 * spec.steps_per_day
    train = frames[:cut_val]
    slot = np.arange(len(frames))
    ha_day = np.stack([train[s::spd].mean(0) for s in range(spd)])
    ha_week = np.stack([train[s::week].mean(0) for s in range(week)])
    preds = {"last": frames[targets - 1], "day": frames[targets - spd], "week": frames[targets - week],
             "ha_day": ha_day[slot[targets] % spd], "ha_week": ha_week[slot[targets] % week]}

    names = spec.channels
    print(f"{args.dataset} {args.split}, cells with true value >= {args.threshold:g}, {len(targets)} targets")
    print(f"  {'baseline':10}" + "".join(f"{c + ' RMSE':>13}{'MAE':>8}" for c in (*names, "all")))
    for name, p in preds.items():
        m = metrics(torch.from_numpy(p)[:, None], true[:, None], args.threshold)
        print(f"  {name:10}" + "".join(f"{m['rmse'][0, k].item():13.3f}{m['mae'][0, k].item():8.2f}"
                                      for k in range(len(names) + 1)))


if __name__ == "__main__":
    main()
