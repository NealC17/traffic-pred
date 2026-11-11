import argparse
import os
from dataclasses import asdict

import torch

from trafficpred.data import DATASETS, add_data_args, splits_from_args
from trafficpred.losses import LOSSES, PARTS_LOSSES, WEIGHTED, default_options, make_loss, needs_parts
from trafficpred.metrics import SELECT
from trafficpred.models import MODELS, add_model_args, build_model, configure_model
from trafficpred.tracking import Tracker
from trafficpred.training import Trainer, TrainSettings
from trafficpred.utils import count_parameters, device_info, pick_device, seed_everything

EXAMPLES = """examples:
  python train.py --model distvol --loss sinkhorn
  python train.py --model stdcformer --dataset bikenyc
  python train.py --model predformer --dataset mmnist --amp
  python train.py --model cnn --dataset synthetic --epochs 1          # smoke test, no data files
  python train.py --model copy_last                                  # reference, nothing to train
  python train.py --model mlp --train-files 2013 --val-files 2014 --test-files 2015 2016
  python train.py --model stresnet --loss mse --resume               # continue <save>.last.pt
  python train.py --model stresnet --external                        # + TaxiBJ weather / calendar data
"""


def parse_args():
    ap = argparse.ArgumentParser(description="Train any model on any dataset, pick the checkpoint "
                                             "on val, score it once on test.",
                                 epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--loss", choices=(*LOSSES, "none"),
                    help="default: sinkhorn for cnn/distvol/distvolformer, masked_mae for stdcformer, mse "
                         "otherwise. The sinkhorn losses (" + ", ".join(PARTS_LOSSES) + ") need a model "
                         "that predicts shape and volume separately: cnn, distvol, distvolformer, stresnet")
    add_data_args(ap)

    g = ap.add_argument_group("training (unset values use the model's defaults)")
    g.add_argument("--epochs", type=int)
    g.add_argument("--batch-size", type=int)
    g.add_argument("--lr", type=float)
    g.add_argument("--weight-decay", type=float)
    g.add_argument("--clip", type=float, help="gradient norm clip (0 = off)")
    g.add_argument("--schedule", choices=("onecycle", "plateau"))
    g.add_argument("--optimizer", choices=("adamw", "adam"))
    g.add_argument("--plateau-patience", type=int, help="epochs without improvement before halving the lr")
    g.add_argument("--patience", type=int, help="early stopping after N epochs without improvement (0 = off)")
    g.add_argument("--select", default=SELECT, help="val metric that picks the checkpoint")
    g.add_argument("--amp", action="store_true", help="bf16 autocast on CUDA")
    g.add_argument("--compile", action="store_true", help="torch.compile the model")
    g.add_argument("--workers", type=int, default=2)
    g.add_argument("--eval-batch-size", type=int, help="default 128 (stdcformer: its training batch size)")
    g.add_argument("--data-on-gpu", choices=("auto", "yes", "no"), default="auto")
    g.add_argument("--curves", choices=("all", "val"), default="all",
                   help="splits scored every epoch for the W&B RMSE / MAE / MSE / Sinkhorn / TV / Frame MSE "
                        "panels: all = val, test and a val-sized train sample (default); val = val only")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--save", help="default: checkpoints/<dataset>/<run name>.pt")
    g.add_argument("--resume", action="store_true", help="continue from <save minus .pt>.last.pt")
    g.add_argument("--no-save", "--no_save", action="store_true",
                   help="write no checkpoints: train, select on val and score on test in memory only")
    g.add_argument("--no-wandb", action="store_true")
    g.add_argument("--params-only", action="store_true", help="print the model size and exit")

    g = ap.add_argument_group(
        "sinkhorn losses",
        "sinkhorn = 1 L1 + 10 Sinkhorn + 1 volume; sinkhorn_only = Sinkhorn; sinkhorn_vol = 10 Sinkhorn + "
        "1 volume; sinkhorn_vol_uw = (Sinkhorn, volume) with learned uncertainty weights; "
        "sinkhorn_adaptive = all three, reweighted like Adam so each term's parameter gradient matches "
        "the L1 term's (--l1/--l2/--l3 then scale those shares, default 1 each). A weight or "
        "Sinkhorn setting that differs from the loss's default is added to the run name (e.g. _l1-0)")
    g.add_argument("--l1", type=float, help="weight on ||X_hat - X||_1 (fixed-weight and adaptive losses)")
    g.add_argument("--l2", type=float, help="weight on Sinkhorn S_eps(p_hat, p) (fixed-weight and adaptive losses)")
    g.add_argument("--l3", type=float, help="weight on |N_hat - N| (fixed-weight and adaptive losses)")
    g.add_argument("--adapt-beta", type=float, help="sinkhorn_adaptive: running-average decay (default 0.95)")
    g.add_argument("--adapt-every", type=int,
                   help="sinkhorn_adaptive: measure the gradients every N steps (default 10)")
    g.add_argument("--eps", type=float, help="Sinkhorn blur (default 0.01)")
    g.add_argument("--sinkhorn-iters", type=int, help="default 30")

    add_model_args(ap)
    return ap.parse_args()


def choose_loss(args, spec):
    loss = args.loss or spec.loss
    if (loss == "none") != (spec.loss == "none"):
        raise SystemExit(f"--model {args.model} needs --loss {spec.loss}" if spec.loss == "none"
                         else "--loss none only works with --model copy_last (nothing to train)")
    if needs_parts(loss) and not spec.parts:
        raise SystemExit(f"--loss {loss} needs a model that predicts shape and volume separately "
                         f"(cnn, distvol, distvolformer, stresnet); --model {args.model} doesn't")
    return loss


def choose_loss_options(args, loss):
    options = default_options(loss)
    given = {k: getattr(args, k) for k in ("l1", "l2", "l3", "eps", "sinkhorn_iters", "adapt_beta", "adapt_every")
             if getattr(args, k) is not None}
    unused = sorted(set(given) - set(options))
    if unused:
        flags = ", ".join("--" + k.replace("_", "-") for k in unused)
        raise SystemExit(f"{flags} doesn't apply to --loss {loss}"
                         + ("; it fits " + ", ".join(sorted({*WEIGHTED, "sinkhorn_adaptive"}))
                            if loss in PARTS_LOSSES else ""))
    return {**options, **given}


def option_suffix(loss, options):
    def fmt(v):
        return f"{v:g}" if isinstance(v, float) else str(v)
    changed = [(k, v) for k, v in options.items() if v != default_options(loss).get(k)]
    return "".join(f"_{k}-{fmt(v)}" for k, v in changed)


def training_settings(args, spec):
    defaults = spec.train_defaults(args.dataset)
    chosen = {k: getattr(args, k) if getattr(args, k) is not None else v for k, v in defaults.items()}
    eval_batch_size = args.eval_batch_size or (chosen["batch_size"] if args.model == "stdcformer" else 128)
    return TrainSettings(**chosen, select=args.select, amp=args.amp, compile=args.compile,
                         workers=args.workers, eval_batch_size=eval_batch_size,
                         data_on_gpu=args.data_on_gpu, curves=args.curves)


def run_name(args, splits, loss, loss_options):
    name = args.model if loss == "none" else f"{args.model}_{loss}{option_suffix(loss, loss_options)}"
    if args.dataset != "taxibj":
        name = f"{args.dataset}_{name}"
    if (splits.t_in, splits.t_out) != DATASETS[args.dataset].horizon:
        name += f"_in{splits.t_in}_out{splits.t_out}"
    if splits.ext_dim:
        name += "_ext"
    return name


def main():
    args = parse_args()
    if args.external and args.model != "stresnet":
        raise SystemExit("--external only works with --model stresnet")
    spec = MODELS[args.model]
    loss_name = choose_loss(args, spec)
    loss_options = choose_loss_options(args, loss_name)
    seed_everything(args.seed)

    splits = splits_from_args(args)
    model_kwargs = configure_model(args.model, args, splits)
    model = build_model(args.model, model_kwargs)
    device = pick_device()
    C, H, W = splits.shape
    print(f"{args.model} ({loss_name} loss): {count_parameters(model) / 1e6:.2f}M params | "
          f"{args.dataset} {C}x{H}x{W}, {splits.t_in} in -> {splits.t_out} out | "
          f"{len(splits.train)} train / {len(splits.val)} val / {len(splits.test)} test windows | {device}")
    if args.params_only:
        return

    settings = training_settings(args, spec)
    name = run_name(args, splits, loss_name, loss_options)
    save = None if args.no_save else args.save or os.path.join("checkpoints", args.dataset, f"{name}.pt")
    if args.no_save and args.resume:
        raise SystemExit("--resume needs saved checkpoints; drop --no-save")
    run = dict(model_name=args.model, model_kwargs=model_kwargs, dataset=args.dataset,
               t_in=splits.t_in, t_out=splits.t_out, split=splits.split,
               scale=(splits.lo, splits.hi), channels=splits.channels, loss=loss_name,
               loss_options=loss_options, seed=args.seed)
    loss = None if loss_name == "none" else make_loss(loss_name, **loss_options)

    torch.backends.cudnn.benchmark = device.type == "cuda"
    trainer = Trainer(model, run, splits, loss, settings, save, device)
    tracker_state = trainer.try_resume() if args.resume else None
    tracker = Tracker(name=name, job_type=args.model, enabled=not args.no_wandb,
                      config=dict(vars(args), **run, **asdict(settings), params=count_parameters(model),
                                  **device_info()),
                      run_id=tracker_state and tracker_state.get("run_id"),
                      group=tracker_state and tracker_state.get("group"))
    if tracker_state:
        tracker.restore(tracker_state)
    trainer.fit(tracker)


if __name__ == "__main__":
    main()
