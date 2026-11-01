"""
Train PredFormer on a dataset from data.py.

    python train.py --dataset taxibj
    python train.py --dataset mmnist --epochs 200 --amp
    python train.py --dataset taxibj --impl official             # official repo's model (STS)
    python train.py --dataset taxibj --impl official --variant TSST
    python train.py --dataset taxibj --t-in 6 --t-out 6          # 3 h in -> 3 h out
    python train.py --dataset taxibj --t-in 4 --t-out 8          # 2 h in -> 4 h out (rollout)

Uses the paper's hyperparameters from predformer.CONFIGS (any flag you pass
overrides them): MSE loss, AdamW, one-cycle LR schedule.

Saves to checkpoints/<dataset>/:
    last.pt  - after every epoch (resume with --resume)
    best.pt  - lowest test MSE seen so far
Each checkpoint stores the model config, so run_viz.py can rebuild the model.
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from devices import pick_device
import data
from tracking import Tracker, CopyLast, evaluate, fmt
from predformer import PredFormer, CONFIGS, VARIANTS


class Horizon(nn.Module):
    """PredFormer always outputs as many frames as it gets (T_out == T_in).
    This adapts it to any t_out the way the official repo does: keep the first
    t_out frames if t_out < t_in, or feed predictions back in as the next input
    (autoregressive rollout) until t_out frames exist if t_out > t_in."""

    def __init__(self, model, t_in, t_out):
        super().__init__()
        self.model, self.t_in, self.t_out = model, t_in, t_out

    def forward(self, x):
        outs, n, cur = [], 0, x
        while n < self.t_out:
            cur = self.model(cur)
            outs.append(cur)
            n += cur.shape[1]
        return torch.cat(outs, dim=1)[:, :self.t_out]


def get_splits(name, t_in=4, t_out=4):
    if name == "taxibj":
        return data.taxibj_splits(t_in, t_out)
    if name == "mmnist":
        if (t_in, t_out) != (10, 10):
            raise SystemExit("Moving MNIST sequences are fixed at 10 in -> 10 out")
        return data.moving_mnist_splits()
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(CONFIGS), default="taxibj")
    ap.add_argument("--impl", choices=["mine", "official"], default="mine",
                    help="'mine' = predformer.py, 'official' = github.com/yyyujintang/PredFormer")
    ap.add_argument("--variant", choices=VARIANTS,
                    help="default: TSST for predformer.py; the official script's choice for official")
    ap.add_argument("--t-in", type=int, help="input frames (TaxiBJ default 4)")
    ap.add_argument("--t-out", type=int, help="frames to predict (TaxiBJ default 4)")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--weight-decay", type=float)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=1, help="epochs between test evals")
    ap.add_argument("--amp", action="store_true", help="bf16 autocast (CUDA recommended)")
    ap.add_argument("--resume", action="store_true", help="continue from last.pt")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-wandb", action="store_true", help="don't log to Weights & Biases")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = CONFIGS[args.dataset]
    hp = dict(cfg["train"])
    for k in ("epochs", "batch_size", "lr", "weight_decay"):
        if getattr(args, k) is not None:
            hp[k] = getattr(args, k)

    t_def = cfg["model"]["in_shape"][0]
    t_in, t_out = args.t_in or t_def, args.t_out or t_def

    device = pick_device()
    run_name = args.dataset if args.impl == "mine" else f"{args.dataset}_official"
    if (t_in, t_out) != (t_def, t_def):
        run_name += f"_in{t_in}_out{t_out}"
    out_dir = os.path.join(args.out, run_name)
    os.makedirs(out_dir, exist_ok=True)

    # ---- data ----
    train_set, test_set = get_splits(args.dataset, t_in, t_out)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=hp["batch_size"], shuffle=True,
                              num_workers=args.workers, pin_memory=pin, drop_last=True,
                              persistent_workers=args.workers > 0)
    test_loader = DataLoader(test_set, batch_size=hp["batch_size"] * 2, shuffle=False,
                             num_workers=args.workers, pin_memory=pin)
    print(f"{args.dataset}: {len(train_set)} train / {len(test_set)} test samples, "
          f"device={device}")

    # ---- model / optim ----
    if args.impl == "official":
        from official_predformer import build_official, official_config, DEFAULT_VARIANT
        variant = args.variant or DEFAULT_VARIANT[args.dataset]
        model_cfg = official_config(args.dataset, variant, pre_seq=t_in, after_seq=t_out)
        core = build_official(variant=variant, config=model_cfg).to(device)
    else:
        model_cfg = dict(cfg["model"])
        model_cfg["in_shape"] = (t_in, *model_cfg["in_shape"][1:])
        if args.variant:
            model_cfg["variant"] = args.variant
            if args.variant not in ("full", "fac_st", "fac_ts"):   # e.g. TST: 8 -> 6 blocks
                model_cfg["num_blocks"] -= model_cfg["num_blocks"] % len(args.variant)
        variant = model_cfg["variant"]
        core = PredFormer(**model_cfg).to(device)
    model = core if t_out == t_in else Horizon(core, t_in, t_out)
    print(f"PredFormer {variant} ({args.impl}), {t_in} in -> {t_out} out: "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")

    # raw-count conversion for the shared metrics: TaxiBJ min-max scale, Moving MNIST /255
    lo, hi = data._flow_scale if args.dataset == "taxibj" else (0.0, 255.0)
    tracker = Tracker(
        name=f"predformer{'_official' if args.impl == 'official' else ''}_mse"
             + ("" if (t_in, t_out) == (t_def, t_def) else f"_in{t_in}_out{t_out}"),
        config={**{k: v for k, v in model_cfg.items() if isinstance(v, (int, float, str))},
                **hp,
                **dict(model="predformer", impl=args.impl, variant=variant, loss="mse",
                       dataset=args.dataset, t_in=t_in, t_out=t_out,
                       params=sum(p.numel() for p in model.parameters()))},
        job_type="predformer", enabled=not args.no_wandb)

    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    total_steps = hp["epochs"] * len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=hp["lr"], total_steps=total_steps)

    start_epoch, best = 0, float("inf")
    last_path, best_path = os.path.join(out_dir, "last.pt"), os.path.join(out_dir, "best.pt")
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device)
        core.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed from epoch {ck['epoch']} (best test MSE {best:.4f})")

    def save(path, epoch, extra=None):
        torch.save(dict(model=core.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict(), epoch=epoch, best=best,
                        config=model_cfg, impl=args.impl, variant=variant,
                        t_in=t_in, t_out=t_out,
                        dataset=args.dataset, **(extra or {})), path)

    # ---- reference + live progress ----
    tracker.log_reference(evaluate(CopyLast(t_out), test_loader, device, hi - lo, lo,
                                   desc="  copy-last reference"))
    tracker.start(hp["epochs"], len(train_loader), start_epoch)

    # ---- train ----
    epochs = tqdm(range(start_epoch, hp["epochs"]), desc=f"PredFormer {variant}",
                  initial=start_epoch, total=hp["epochs"], unit="ep", dynamic_ncols=True, mininterval=1)
    if start_epoch and os.path.exists(best_path):        # resumed: report the earlier best too
        ck_best = torch.load(best_path, map_location="cpu")
        if "test_metrics" in ck_best:
            tracker.log_best(ck_best["epoch"] + 1, ck_best["test_metrics"])
    for epoch in epochs:
        model.train()
        t0, running = time.time(), 0.0
        bar = tqdm(train_loader, desc=f"  epoch {epoch + 1}", leave=False, dynamic_ncols=True, mininterval=1)
        for i, (x, y) in enumerate(bar, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.amp):
                pred = model(x)
            loss = F.mse_loss(pred.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item()
            bar.set_postfix(loss=f"{running / i:.5f}")
            tracker.step(loss.item(), sched.get_last_lr()[0])

        msg = (f"epoch {epoch + 1}/{hp['epochs']}  train_loss {running / len(train_loader):.5f}  "
               f"lr {sched.get_last_lr()[0]:.2e}  {time.time() - t0:.0f}s")

        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == hp["epochs"]:
            m = evaluate(model, test_loader, device, hi - lo, lo)
            msg += "  | " + fmt(m)
            tracker.log_eval(epoch + 1, m, train_loss=running / len(train_loader),
                             epoch_time=time.time() - t0, model=model, dataset=test_set,
                             device=device, span=hi - lo, lo=lo, last=epoch + 1 == hp["epochs"])
            if m["mse_frame"] < best:                    # same criterion as before (per-frame MSE)
                best = m["mse_frame"]
                save(best_path, epoch, dict(test_mse=m["mse_frame"], test_mae=m["mae_frame"],
                                            test_metrics=m))
                tracker.log_best(epoch + 1, m)
                msg += "  *"
        save(last_path, epoch)
        epochs.set_postfix(best_mse=f"{best:.4f}")
        tqdm.write(msg)

    print(f"done. best test MSE {best:.4f} -> {best_path}")
    tracker.finish()


if __name__ == "__main__":
    main()