import argparse
import os
from dataclasses import asdict

import torch

from trafficpred.data import DATASETS, add_data_args, splits_from_args
from trafficpred.losses import LOSSES, make_loss
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
"""


def parse_args():
    ap = argparse.ArgumentParser(description="Train any model on any dataset, pick the checkpoint "
                                             "on val, score it once on test.",
                                 epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--loss", choices=(*LOSSES, "none"),
                    help="default: sinkhorn for cnn/distvol, masked_mae for stdcformer, mse otherwise")
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
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--save", help="default: checkpoints/<dataset>/<run name>.pt")
    g.add_argument("--resume", action="store_true", help="continue from <save minus .pt>.last.pt")
    g.add_argument("--no-save", "--no_save", action="store_true",
                   help="write no checkpoints: train, select on val and score on test in memory only")
    g.add_argument("--no-wandb", action="store_true")
    g.add_argument("--params-only", action="store_true", help="print the model size and exit")

    g = ap.add_argument_group("sinkhorn loss (L1 + Sinkhorn + volume)")
    g.add_argument("--l1", type=float, default=1.0, help="weight on ||X_hat - X||_1")
    g.add_argument("--l2", type=float, default=10.0, help="weight on Sinkhorn S_eps(p_hat, p)")
    g.add_argument("--l3", type=float, default=1.0, help="weight on |N_hat - N|")
    g.add_argument("--eps", type=float, default=0.01, help="Sinkhorn blur")
    g.add_argument("--sinkhorn-iters", type=int, default=30)

    add_model_args(ap)
    return ap.parse_args()


def choose_loss(args, spec):
    loss = args.loss or spec.losses[0]
    if loss not in spec.losses:
        raise SystemExit(f"--model {args.model} supports --loss {' / '.join(spec.losses)}")
    return loss


def training_settings(args, spec):
    defaults = spec.train_defaults(args.dataset)
    chosen = {k: getattr(args, k) if getattr(args, k) is not None else v for k, v in defaults.items()}
    eval_batch_size = args.eval_batch_size or (chosen["batch_size"] if args.model == "stdcformer" else 128)
    return TrainSettings(**chosen, select=args.select, amp=args.amp, compile=args.compile,
                         workers=args.workers, eval_batch_size=eval_batch_size,
                         data_on_gpu=args.data_on_gpu)


def run_name(args, splits, loss):
    name = args.model if loss == "none" else f"{args.model}_{loss}"
    if args.dataset != "taxibj":
        name = f"{args.dataset}_{name}"
    if (splits.t_in, splits.t_out) != DATASETS[args.dataset].horizon:
        name += f"_in{splits.t_in}_out{splits.t_out}"
    return name


def main():
    args = parse_args()
    spec = MODELS[args.model]
    loss_name = choose_loss(args, spec)
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
    name = run_name(args, splits, loss_name)
    save = None if args.no_save else args.save or os.path.join("checkpoints", args.dataset, f"{name}.pt")
    if args.no_save and args.resume:
        raise SystemExit("--resume needs saved checkpoints; drop --no-save")
    loss_options = dict(l1=args.l1, l2=args.l2, l3=args.l3, eps=args.eps,
                        sinkhorn_iters=args.sinkhorn_iters) if loss_name == "sinkhorn" else {}
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
