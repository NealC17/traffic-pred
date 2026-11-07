import argparse
import os

import torch

from trafficpred import astgnn
from trafficpred.data import add_data_args, splits_from_args
from trafficpred.metrics import SPLITS, evaluate, final_line, fmt, summary_table
from trafficpred.utils import pick_device, save_atomic

DESCRIPTION = """Run the official ASTGNN (github.com/guoshnBJTU/ASTGNN) on any dataset, with this
project's windows, split and metrics. Every grid cell is a graph node linked to its
neighbours; ASTGNN predicts one channel per run, so each channel is its own run.

  python astgnn.py run --dataset bikenyc          all three steps below in one go (what train_all.sh uses)

  1) python astgnn.py prepare --dataset taxibj --train-files 2013 --val-files 2014 --test-files 2015 2016
  2) cd external/ASTGNN and run the printed command for every channel (ASTGNN's own
     train_ASTGNN.py, seeded, plus its predictions on the train and val sets)
  3) python astgnn.py evaluate --dataset taxibj --train-files 2013 --val-files 2014 --test-files 2015 2016
"""


def parse_args():
    ap = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["prepare", "evaluate", "run"])
    ap.add_argument("--astgnn-root", default=astgnn.DEFAULT_ROOT, help="the cloned ASTGNN repo")
    ap.add_argument("--neighbors", type=int, choices=[4, 8], default=8)
    ap.add_argument("--epochs", type=int, help="ASTGNN training epochs (default 100)")
    ap.add_argument("--fine-tune-epochs", type=int, help="ASTGNN fine-tuning epochs (default 50)")
    ap.add_argument("--batch-size", type=int, help="ASTGNN batch size (default 4)")
    ap.add_argument("--save", help="evaluate / run: where to store the metrics")
    ap.add_argument("--no-save", "--no_save", action="store_true",
                    help="evaluate / run: don't write the metrics file (ASTGNN still writes its own "
                         "data and experiment files under external/)")
    ap.add_argument("--seed", type=int, default=42, help="seed set before ASTGNN's training script runs")
    add_data_args(ap)
    return ap.parse_args()


def overrides(args):
    chosen = {k: getattr(args, k) for k in ("epochs", "fine_tune_epochs", "batch_size")
              if getattr(args, k) is not None}
    if args.epochs is not None and args.fine_tune_epochs is None:
        chosen["fine_tune_epochs"] = max(1, args.epochs // 2)
    return chosen


def prepare(args, splits):
    configs = astgnn.prepare(splits, args.astgnn_root, args.neighbors, overrides(args))
    print(f"\nnext, one run per channel:\n    cd {os.path.relpath(args.astgnn_root)}")
    for config in configs:
        print(f"    PYTHONPATH={os.path.abspath('.')} " + " ".join(astgnn.train_command(config, args.seed)))
    return configs


def score(args, splits):
    scores = {}
    for split in SPLITS:
        pred, y_raw = astgnn.collect_predictions(splits, args.astgnn_root, split)
        loader = astgnn.prediction_loader(pred, y_raw, splits.lo, splits.hi)
        scores[split] = evaluate(torch.nn.Identity(), loader, pick_device(), splits.span, splits.lo,
                                 desc=f"  ASTGNN ({split})")
    print(fmt(scores["test"], "test"))
    print(summary_table(scores))
    name = os.environ.get("WANDB_NAME", "astgnn")
    print(final_line(name, scores, "-"), flush=True)
    if args.no_save:
        return
    save = args.save or os.path.join("checkpoints", args.dataset, f"astgnn_{astgnn.run_tag(splits)}.metrics.pt")
    save_atomic({"model_name": "astgnn", "train_metrics": scores["train"], "val_metrics": scores["val"],
                 "test_metrics": scores["test"], "dataset": args.dataset, "split": splits.split,
                 "t_in": splits.t_in, "t_out": splits.t_out, "seed": args.seed}, save)
    print(f"saved -> {save}")


def main():
    args = parse_args()
    splits = splits_from_args(args)
    if args.action == "prepare":
        prepare(args, splits)
    elif args.action == "evaluate":
        score(args, splits)
    else:
        configs = astgnn.prepare(splits, args.astgnn_root, args.neighbors, overrides(args))
        astgnn.train(args.astgnn_root, configs, args.seed)
        score(args, splits)


if __name__ == "__main__":
    main()
