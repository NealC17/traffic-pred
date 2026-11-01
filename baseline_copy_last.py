"""
Copy-last-frame reference: predict every future frame as the last observed one.

Scores it with the same evaluate() as every model and logs it to W&B as its own run
(named copy_last), with the same test/* metrics at epoch 0 and at --epochs, so it draws
a flat reference line across every test chart. Nothing to train.

    python baseline_copy_last.py --tin 4 --tout 4 --epochs 200 --save checkpoints/copy_last.pt
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from devices import pick_device
import data
from tracking import Tracker, CopyLast, evaluate, fmt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tin", type=int, default=4)
    ap.add_argument("--tout", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=200,
                    help="x-axis length of the reference line on the W&B test charts")
    ap.add_argument("--save", default="checkpoints/copy_last.pt", help="where to store the metrics")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    lo, hi = data._flow_scale
    _, test = data.taxibj_splits(args.tin, args.tout)
    loader = DataLoader(test, batch_size=64)
    device = pick_device()

    name = "copy_last" + ("" if (args.tin, args.tout) == (4, 4) else f"_in{args.tin}_out{args.tout}")
    tracker = Tracker(name=name, job_type="baseline", enabled=not args.no_wandb,
                      config=dict(model="copy_last", t_in=args.tin, t_out=args.tout, params=0))
    m = evaluate(CopyLast(args.tout), loader, device, hi - lo, lo)
    print(fmt(m))
    for epoch in (0, args.epochs):                    # two points -> a flat line on each chart
        tracker.log_eval(epoch, m)
    tracker.log_best(0, m)

    if os.path.dirname(args.save):
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
    torch.save({"model": "copy_last", "test_metrics": m, "t_in": args.tin, "t_out": args.tout},
               args.save)
    tracker.finish()


if __name__ == "__main__":
    main()
