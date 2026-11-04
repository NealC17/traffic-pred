"""
Copy-last-frame reference: predict every future frame as the last observed one.

Scores it with the same evaluate() as every model, on val and test, and logs it to W&B as
its own run (named copy_last): the val/* metrics at epoch 0 and at --epochs, so it draws a
flat reference line across every val chart, and the test/* metrics once like every model's
final result. Nothing to train (and nothing to select), so its test numbers are unbiased.

    python baseline_copy_last.py --tin 4 --tout 4 --epochs 200 --save checkpoints/copy_last.pt
    python baseline_copy_last.py --train-files 2013 --val-files 2014 --test-files 2015 2016
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
    data.add_split_args(ap)
    args = ap.parse_args()

    split = data.split_kwargs(args)
    train, val, test = data.taxibj_splits(args.tin, args.tout, val=True, **split)
    lo, hi = data.scale_of(train)                     # same scale the models use
    device = pick_device()

    name = "copy_last" + ("" if (args.tin, args.tout) == (4, 4) else f"_in{args.tin}_out{args.tout}")
    tracker = Tracker(name=name, job_type="baseline", enabled=not args.no_wandb,
                      config=dict(model="copy_last", t_in=args.tin, t_out=args.tout, params=0,
                                  **split))
    model = CopyLast(args.tout)
    m_val = evaluate(model, DataLoader(val, batch_size=64), device, hi - lo, lo)
    m = evaluate(model, DataLoader(test, batch_size=64), device, hi - lo, lo)
    print(fmt(m_val, "val"))
    for epoch in (0, args.epochs):                    # two points -> a flat line on each chart
        tracker.log_eval(epoch, m_val)
    tracker.log_reference(val=m_val, test=m)          # so vs_copy_last/* = 1
    tracker.log_best(0, m_val)
    tracker.log_test(m, epoch=args.epochs)

    if os.path.dirname(args.save):
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
    torch.save({"model": "copy_last", "val_metrics": m_val, "test_metrics": m,
                "t_in": args.tin, "t_out": args.tout}, args.save)
    tracker.finish()


if __name__ == "__main__":
    main()
