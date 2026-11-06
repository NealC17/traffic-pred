"""
Train DistVolNet on TaxiBJ using data.py's taxibj_splits().

  python train_distvol.py --tin 4 --tout 4 --epochs 50 --save checkpoints/distvol_best.pt
  python train_distvol.py --model cnn --tin 4 --tout 4 --epochs 50   # baseline CNN, same loss
  python train_distvol.py --model cnn --loss mse                   # baseline CNN, plain MSE
  python train_distvol.py --model stresnet --loss mse              # ST-ResNet (paper: MSE)
  python train_distvol.py --synthetic --epochs 2 --width 32 --sinkhorn_iters 10   # smoke test
  python train_distvol.py ... --resume       # continue from <save>.last.pt if it exists
  python train_distvol.py ... --seed 1       # another seed (default 0); use its own --save path
  python train_distvol.py --train-files 2013 --val-files 2014 --test-files 2015 2016
  python train_distvol.py --train-size 5000 --val-size 500 --test-size 0.5

--loss sinkhorn (default): DistVolLoss = L1 + Sinkhorn + volume
--loss mse:                plain MSE on the predicted counts

ST-ResNet (stresnet.py) uses --width as its filter count (64, as in the paper), --st-units
residual units (12, the paper's TaxiBJ setting) and --st-bn for batch norm. Only the closeness
branch is used: taxibj_splits windows hold consecutive frames, so there are no period / trend
frames or external factors to feed it.

The checkpoint's config["model"] says which class to rebuild ("distvol", "cnn" or "stresnet"),
config["loss"] which loss it was trained with.

Resume: every epoch writes <save minus .pt>.last.pt (model, optimizer, LR schedule, epoch,
best score, W&B run id). With --resume the run continues from it, in the same W&B run.
train_all.sh passes --resume, so a job killed by its time limit picks up where it stopped.

Speed (CUDA): the whole dataset is copied to the GPU once (--data-on-gpu, default on CUDA),
the step loop never waits on the GPU except for the progress bar every 50 steps, AdamW is
fused, and cuDNN autotunes its convolutions. --amp (bf16 model) and --compile (torch.compile)
are opt-in: they change the numerics slightly, so runs trained without them aren't
bit-comparable. Sinkhorn always runs in float32.

Data: taxibj_splits(tin, tout, val=True): test = the last 4 weeks of TaxiBJ, val = the
4 weeks before that, train = everything earlier, with the same min-max scaling as
PredFormer. Windows never cross file or split boundaries. Run from the folder that contains
data.py and datasets/.

Protocol: every epoch is scored on val and the checkpoint with the lowest val --select
metric (default mse, the same as every other model) is kept. When training ends it is
scored once on test; those are the numbers to report (W&B test/*, and the saved
checkpoint's "test_metrics").
"""
import argparse, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from devices import pick_device
from tracking import Tracker, CopyLast, evaluate, fmt, SELECT
from distvol import DistVolNet, DistVolLoss
from baseline_cnn import BaselineCNN
from stresnet import STResNet
from seeding import seed_everything


# ---------------------------------------------------------------------------
# Synthetic data (smoke test only; does not import data.py)
# ---------------------------------------------------------------------------
class _Windows(Dataset):
    def __init__(self, frames, tin, tout):
        self.f, self.tin, self.tout = torch.as_tensor(frames), tin, tout

    def __len__(self):
        return len(self.f) - self.tin - self.tout + 1

    def __getitem__(self, i):
        j = i + self.tin
        return self.f[i:j], self.f[j:j + self.tout]


def synthetic_splits(tin, tout, n=1500, H=32, W=32, seed=0):
    """Gaussian hotspots drifting across the grid, with a daily cycle in total volume."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    centers = rng.uniform(4, 28, size=(5, 2)); vel = rng.normal(0, 0.3, size=(5, 2))
    f = np.zeros((n, 2, H, W), dtype=np.float32)
    for t in range(n):
        centers = np.clip(centers + vel, 2, 29)
        vol = 1.0 + 0.8 * math.sin(2 * math.pi * t / 48)
        for c in range(2):
            f[t, c] = vol * sum(np.exp(-((yy - cy - c) ** 2 + (xx - cx) ** 2) / 8)
                                for cy, cx in centers)
    f += rng.uniform(0, 0.02, f.shape).astype(np.float32)
    split, vsplit = int(0.7 * n), int(0.85 * n)
    lo, hi = float(f[:split].min()), float(f[:split].max())
    f = (f - lo) / (hi - lo)
    return (_Windows(f[:split], tin, tout), _Windows(f[split:vsplit], tin, tout),
            _Windows(f[vsplit:], tin, tout), (lo, hi))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
class MSELoss(nn.Module):
    """Plain MSE on the counts, with the same call signature as DistVolLoss."""
    def forward(self, x_hat, p_hat, N_hat, y):
        loss = F.mse_loss(x_hat, y)
        return loss, {"mse": loss.detach()}


class DeviceBatches:
    """Whole dataset stacked into two tensors on the device; yields (x, y) batches like a
    DataLoader (shuffle / drop_last), without worker processes or host-to-GPU copies."""

    def __init__(self, dataset, batch_size, device, shuffle=False, drop_last=False, workers=0):
        xs, ys = [], []
        loader = DataLoader(dataset, batch_size=512, num_workers=workers)
        for x, y in tqdm(loader, desc="  copying data to the GPU", leave=False, mininterval=1):
            xs.append(x); ys.append(y)
        self.x = torch.cat(xs).to(device)
        self.y = torch.cat(ys).to(device)
        self.bs, self.shuffle, self.drop_last = batch_size, shuffle, drop_last

    def __len__(self):
        n = len(self.x)
        return n // self.bs if self.drop_last else math.ceil(n / self.bs)

    def __iter__(self):
        n = len(self.x)
        order = torch.randperm(n, device=self.x.device) if self.shuffle else None
        for i in range(len(self)):
            sl = slice(i * self.bs, min((i + 1) * self.bs, n))
            if order is None:
                yield self.x[sl], self.y[sl]
            else:
                idx = order[sl]
                yield self.x[idx], self.y[idx]


# Settings that must match for --resume to continue a checkpoint (others, like --workers or
# --amp, can change between the two halves of a run).
RESUME_KEYS = ("model", "loss", "tin", "tout", "width", "blocks", "st_units", "st_bn",
               "st_paper_init",
               "epochs", "batch_size", "lr",
               "l1", "l2", "l3", "eps", "sinkhorn_iters", "synthetic", "select",
               "train_files", "val_files", "test_files", "train_size", "val_size", "test_size")


def save_atomic(obj, path):
    """torch.save to a temp file, then rename: a job killed mid-write leaves the old file."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="smoke test without datasets/")
    ap.add_argument("--tin", type=int, default=4, help="input frames (half-hour steps)")
    ap.add_argument("--tout", type=int, default=4, help="frames to predict")
    ap.add_argument("--model", choices=["distvol", "cnn", "stresnet"], default="distvol",
                    help="distvol = DistVolNet; cnn = BaselineCNN (same loss, plain conv stack); "
                         "stresnet = ST-ResNet (Zhang et al. 2017), closeness branch")
    ap.add_argument("--loss", choices=["sinkhorn", "mse"], default="sinkhorn",
                    help="sinkhorn = DistVolLoss (L1 + Sinkhorn + volume); mse = plain MSE")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6, help="residual blocks (cnn only)")
    ap.add_argument("--st-units", type=int, default=12,
                    help="ST-ResNet residual units (paper: 12 for TaxiBJ)")
    ap.add_argument("--st-bn", action="store_true", help="batch norm in ST-ResNet residual units")
    ap.add_argument("--st-paper-init", action="store_true",
                    help="ST-ResNet: keep the default output bias (starts at 0.5) instead of "
                         "starting at the mean training value; with MSE it tends to collapse to 0")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l1", type=float, default=1.0, help="weight on ||X_hat - X||_1")
    ap.add_argument("--l2", type=float, default=10.0, help="weight on Sinkhorn S_eps(p_hat, p)")
    ap.add_argument("--l3", type=float, default=1.0, help="weight on |N_hat - N|")
    ap.add_argument("--eps", type=float, default=0.01, help="Sinkhorn blur")
    ap.add_argument("--sinkhorn_iters", type=int, default=30)
    ap.add_argument("--no-wandb", action="store_true", help="don't log to Weights & Biases")
    ap.add_argument("--seed", type=int, default=0,
                    help="seeds the initial weights, shuffle order and dropout (default 0)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <save>.last.pt (and its W&B run) if it exists")
    ap.add_argument("--select", default=SELECT,
                    help="val metric to pick the checkpoint by (lower is better); keep the "
                         "default so every model is selected the same way")
    ap.add_argument("--eval-batch-size", type=int, default=128,
                    help="val/test batch size (no effect on results, only speed)")
    ap.add_argument("--data-on-gpu", choices=["auto", "yes", "no"], default="auto",
                    help="keep the whole dataset on the GPU (auto = on CUDA)")
    ap.add_argument("--amp", action="store_true",
                    help="bf16 autocast for the model on CUDA (faster; Sinkhorn stays float32)")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model (CUDA)")
    ap.add_argument("--save", type=str, default=None,
                    help="default: checkpoints/<model>_<loss>.pt")
    import data                                       # loads datasets/ on import
    data.add_split_args(ap)
    args = ap.parse_args()
    args.save = args.save or f"checkpoints/{args.model}_{args.loss}.pt"
    last_path = os.path.splitext(args.save)[0] + ".last.pt"
    seed_everything(args.seed)

    if args.synthetic:
        train, val, test, (lo, hi) = synthetic_splits(args.tin, args.tout)
    else:
        if data._flow_scale is None:
            raise SystemExit("No TaxiBJ flow files found under datasets/ "
                             "(expected names starting with TaxiBJ/TaxiBJ2...).")
        train, val, test = data.taxibj_splits(args.tin, args.tout, val=True,
                                              **data.split_kwargs(args))
        lo, hi = data.scale_of(train)                 # train files only, for a file split
    span = hi - lo

    x0, y0 = train[0]
    assert x0.shape[0] == args.tin and y0.shape[0] == args.tout, (x0.shape, y0.shape)
    _, C, H, W = x0.shape
    assert H % 4 == 0 and W % 4 == 0, "encoder downsamples twice; H and W must be divisible by 4"

    device = pick_device()
    cuda = device.type == "cuda"
    if cuda:
        torch.backends.cudnn.benchmark = True          # fixed shapes: autotune the convolutions

    tl = vl = sl = None                                # train / val / test loaders
    if args.data_on_gpu == "yes" or (args.data_on_gpu == "auto" and cuda):
        try:
            tl = DeviceBatches(train, args.batch_size, device, shuffle=True, drop_last=True,
                               workers=args.workers)
            vl = DeviceBatches(val, args.eval_batch_size, device, workers=args.workers)
            sl = DeviceBatches(test, args.eval_batch_size, device, workers=args.workers)
            n = sum(b.x.numel() + b.y.numel() for b in (tl, vl, sl))
            print(f"data on {device}: {n * 4 / 2**30:.2f} GiB")
        except torch.cuda.OutOfMemoryError:
            tl = vl = sl = None
            torch.cuda.empty_cache()
            print("dataset doesn't fit on the GPU; using DataLoader")
    if tl is None:
        tl = DataLoader(train, args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.workers, persistent_workers=args.workers > 0,
                        pin_memory=cuda)
        vl = DataLoader(val, args.eval_batch_size, num_workers=args.workers, pin_memory=cuda)
        sl = DataLoader(test, args.eval_batch_size, num_workers=args.workers, pin_memory=cuda)

    if args.model == "cnn":
        net = BaselineCNN(args.tin, args.tout, C, H, W, args.width, args.blocks).to(device)
    elif args.model == "stresnet":
        init_mean = None
        if not args.st_paper_init:                     # mean target over up to 2000 windows
            idx = np.linspace(0, len(train) - 1, min(len(train), 2000)).astype(int)
            init_mean = float(np.mean([train[i][1].float().mean().item() for i in idx]))
            print(f"ST-ResNet output starts at the mean training value {init_mean:.5f}")
        net = STResNet(args.tin, args.tout, C, H, W, filters=args.width, units=args.st_units,
                       bn=args.st_bn, init_mean=init_mean).to(device)
    else:
        net = DistVolNet(args.tin, args.tout, C, H, W, args.width).to(device)
    model = torch.compile(net) if args.compile else net      # checkpoints always save `net`
    crit = (MSELoss() if args.loss == "mse"
            else DistVolLoss(args.l1, args.l2, args.l3, args.eps, args.sinkhorn_iters))
    try:
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4, fused=cuda)
    except (TypeError, RuntimeError):
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(tl))
    use_amp = args.amp and cuda

    # ---- resume ---------------------------------------------------------------------
    start_epoch, best, resume_state = 0, float("inf"), None
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        diff = {k: (ck["config"].get(k), vars(args)[k]) for k in RESUME_KEYS
                if ck["config"].get(k) != vars(args)[k]}
        if diff:
            print(f"!!! {last_path} was trained with different settings {diff}; starting over")
        else:
            # The seed only matters at the start; a resumed run restores the saved RNG state.
            # Keep the checkpoint's seed on record (checkpoints from before --seed have none).
            old_seed = ck["config"].get("seed")
            if old_seed != args.seed:
                started = ("before --seed existed (unseeded)" if old_seed is None
                           else f"with seed {old_seed}")
                print(f"note: {last_path} was started {started}; continuing it as it was")
                args.seed = old_seed
            net.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            start_epoch, best, resume_state = ck["epoch"], ck["best"], ck["tracker"]
            torch.set_rng_state(ck["rng"]["cpu"].cpu())           # map_location moved it
            if cuda and ck["rng"].get("cuda") is not None:
                torch.cuda.set_rng_state(ck["rng"]["cuda"].cpu())
            print(f"resuming from {last_path}: {start_epoch}/{args.epochs} epochs done, "
                  f"best val {args.select} {best:.5f}")
    elif args.resume:
        print(f"no {last_path}; starting from scratch")

    print(f"{args.model} ({args.loss} loss): {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M params | "
          f"tin={args.tin} tout={args.tout} | {len(train)} train / {len(val)} val / {len(test)} test "
          f"windows | {device}")

    if os.path.dirname(args.save):
        os.makedirs(os.path.dirname(args.save), exist_ok=True)

    run_tag = f"{args.model}_{args.loss}" + ("" if (args.tin, args.tout) == (4, 4)
                                               else f"_in{args.tin}_out{args.tout}")
    tracker = Tracker(name=run_tag, job_type=args.model, enabled=not args.no_wandb,
                      config=dict(vars(args), params=sum(p.numel() for p in net.parameters())),
                      run_id=resume_state and resume_state.get("run_id"),
                      group=resume_state and resume_state.get("group"))
    if resume_state:
        tracker.restore(resume_state)
    tracker.log_reference(
        val=evaluate(CopyLast(args.tout), vl, device, span, lo, desc="  copy-last (val)"),
        test=evaluate(CopyLast(args.tout), sl, device, span, lo, desc="  copy-last (test)"))
    tracker.start(args.epochs, len(tl), start_epoch=start_epoch)

    POSTFIX_EVERY = 50                                     # progress-bar loss: one GPU sync per 50 steps
    epochs = tqdm(range(start_epoch + 1, args.epochs + 1), desc=f"{args.model} ({args.loss})", unit="ep",
                  initial=start_epoch, total=args.epochs, dynamic_ncols=True, mininterval=1)
    for ep in epochs:
        model.train(); t0 = time.time(); agg = {}; running = 0.0
        bar = tqdm(tl, desc=f"  epoch {ep}", leave=False, dynamic_ncols=True, mininterval=1)
        for i, (x, y) in enumerate(bar, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                x_hat, p_hat, N_hat = model(x, return_parts=True)
            loss, parts = crit(x_hat.float(), p_hat.float(), N_hat.float(), y)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
            for k, v in parts.items():                     # stays on the device until the epoch ends
                agg[k] = agg.get(k, 0.0) + v / len(tl)
            loss = loss.detach()
            running = running + loss
            if i % POSTFIX_EVERY == 0:
                bar.set_postfix(loss=f"{loss.item():.5f}")
            tracker.step(loss, sched.get_last_lr()[0])
        running = float(running)
        m = evaluate(net, vl, device, span, lo)
        tracker.log_eval(ep, m, train_loss=running / len(tl),
                         train_parts={k: float(v) for k, v in agg.items()},
                         epoch_time=time.time() - t0, model=net, dataset=val, device=device,
                         span=span, lo=lo, last=ep == args.epochs)
        tag = ""
        if m[args.select] < best:                          # select on VAL, never on test
            best, tag = m[args.select], " *"
            save_atomic({"model": net.state_dict(), "config": vars(args), "epoch": ep,
                         "shape": (C, H, W), "scale": (lo, hi), "val_metrics": m}, args.save)
            tracker.log_best(ep, m)
        save_atomic({"model": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                     "epoch": ep, "best": best, "tracker": tracker.state(), "config": vars(args),
                     "rng": {"cpu": torch.get_rng_state(),
                             "cuda": torch.cuda.get_rng_state() if cuda else None}}, last_path)
        terms = " ".join(f"{k} {float(v):.5f}" for k, v in agg.items())
        epochs.set_postfix(best_val=f"{best:.5f}")
        tqdm.write(f"ep {ep:3d} | {terms} | {fmt(m)} | {time.time() - t0:.0f}s{tag}")

    # ---- final: the selected checkpoint, scored once on the test set ----------------
    if os.path.exists(args.save):
        ck = torch.load(args.save, map_location=device, weights_only=False)
        if "val_metrics" in ck:
            net.load_state_dict(ck["model"])
            m_test = evaluate(net, sl, device, span, lo, desc="  test")
            ck["test_metrics"] = m_test
            save_atomic(ck, args.save)
            tracker.log_test(m_test, epoch=args.epochs)
        else:
            print(f"!!! {args.save} is from before the val split; not scoring it on test")
    tracker.finish()


if __name__ == "__main__":
    main()