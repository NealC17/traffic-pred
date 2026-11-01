"""
Train DistVolNet on TaxiBJ using data.py's taxibj_splits().

  python train_distvol.py --tin 4 --tout 4 --epochs 50 --save checkpoints/distvol_best.pt
  python train_distvol.py --model cnn --tin 4 --tout 4 --epochs 50   # baseline CNN, same loss
  python train_distvol.py --model cnn --loss mse                   # baseline CNN, plain MSE
  python train_distvol.py --synthetic --epochs 2 --width 32 --sinkhorn_iters 10   # smoke test

--loss sinkhorn (default): DistVolLoss = L1 + Sinkhorn + volume
--loss mse:                plain MSE on the predicted counts

The checkpoint's config["model"] says which class to rebuild ("distvol" or "cnn"),
config["loss"] which loss it was trained with.

Data: taxibj_splits(tin, tout) trains on every TaxiBJ flow file except the last 4 weeks
and tests on those 4 weeks, with the same min-max scaling as PredFormer. Windows never
cross file boundaries. Run from the folder that contains data.py and datasets/.
"""
import argparse, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from devices import pick_device
from tracking import Tracker, CopyLast, evaluate, fmt
from distvol import DistVolNet, DistVolLoss
from baseline_cnn import BaselineCNN


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
    split = int(0.8 * n)
    lo, hi = float(f[:split].min()), float(f[:split].max())
    f = (f - lo) / (hi - lo)
    return _Windows(f[:split], tin, tout), _Windows(f[split:], tin, tout), (lo, hi)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
class MSELoss(nn.Module):
    """Plain MSE on the counts, with the same call signature as DistVolLoss."""
    def forward(self, x_hat, p_hat, N_hat, y):
        loss = F.mse_loss(x_hat, y)
        return loss, {"mse": loss.item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="smoke test without datasets/")
    ap.add_argument("--tin", type=int, default=4, help="input frames (half-hour steps)")
    ap.add_argument("--tout", type=int, default=4, help="frames to predict")
    ap.add_argument("--model", choices=["distvol", "cnn"], default="distvol",
                    help="distvol = DistVolNet; cnn = BaselineCNN (same loss, plain conv stack)")
    ap.add_argument("--loss", choices=["sinkhorn", "mse"], default="sinkhorn",
                    help="sinkhorn = DistVolLoss (L1 + Sinkhorn + volume); mse = plain MSE")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6, help="residual blocks (cnn only)")
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
    ap.add_argument("--save", type=str, default=None,
                    help="default: checkpoints/<model>_<loss>.pt")
    args = ap.parse_args()
    args.save = args.save or f"checkpoints/{args.model}_{args.loss}.pt"

    if args.synthetic:
        train, test, (lo, hi) = synthetic_splits(args.tin, args.tout)
    else:
        import data                                   # loads datasets/ on import
        if data._flow_scale is None:
            raise SystemExit("No TaxiBJ flow files found under datasets/ "
                             "(expected names starting with TaxiBJ/TaxiBJ2...).")
        train, test = data.taxibj_splits(args.tin, args.tout)
        lo, hi = data._flow_scale
    span = hi - lo

    x0, y0 = train[0]
    assert x0.shape[0] == args.tin and y0.shape[0] == args.tout, (x0.shape, y0.shape)
    _, C, H, W = x0.shape
    assert H % 4 == 0 and W % 4 == 0, "encoder downsamples twice; H and W must be divisible by 4"

    tl = DataLoader(train, args.batch_size, shuffle=True, drop_last=True,
                    num_workers=args.workers, persistent_workers=args.workers > 0)
    vl = DataLoader(test, args.batch_size, num_workers=args.workers)

    device = pick_device()
    if args.model == "cnn":
        model = BaselineCNN(args.tin, args.tout, C, H, W, args.width, args.blocks).to(device)
    else:
        model = DistVolNet(args.tin, args.tout, C, H, W, args.width).to(device)
    crit = (MSELoss() if args.loss == "mse"
            else DistVolLoss(args.l1, args.l2, args.l3, args.eps, args.sinkhorn_iters))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(tl))

    print(f"{args.model} ({args.loss} loss): {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params | "
          f"tin={args.tin} tout={args.tout} | {len(train)} train / {len(test)} test windows | {device}")

    if os.path.dirname(args.save):
        os.makedirs(os.path.dirname(args.save), exist_ok=True)

    run_tag = f"{args.model}_{args.loss}" + ("" if (args.tin, args.tout) == (4, 4)
                                               else f"_in{args.tin}_out{args.tout}")
    tracker = Tracker(name=run_tag, job_type=args.model, enabled=not args.no_wandb,
                      config=dict(vars(args), params=sum(p.numel() for p in model.parameters())))
    tracker.log_reference(evaluate(CopyLast(args.tout), vl, device, span, lo,
                                   desc="  copy-last reference"))
    tracker.start(args.epochs, len(tl))

    best = float("inf")
    epochs = tqdm(range(1, args.epochs + 1), desc=f"{args.model} ({args.loss})", unit="ep", dynamic_ncols=True, mininterval=1)
    for ep in epochs:
        model.train(); t0 = time.time(); agg = {}; running = 0.0
        bar = tqdm(tl, desc=f"  epoch {ep}", leave=False, dynamic_ncols=True, mininterval=1)
        for i, (x, y) in enumerate(bar, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x_hat, p_hat, N_hat = model(x, return_parts=True)
            loss, parts = crit(x_hat, p_hat, N_hat, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v / len(tl)
            running += loss.item()
            bar.set_postfix(loss=f"{loss.item():.5f}")
            tracker.step(loss.item(), sched.get_last_lr()[0])
        m = evaluate(model, vl, device, span, lo)
        tracker.log_eval(ep, m, train_loss=running / len(tl), epoch_time=time.time() - t0,
                         model=model, dataset=test, device=device, span=span, lo=lo,
                         last=ep == args.epochs)
        tag = ""
        if m["mae_flow"] < best:                           # same criterion as before: test MAE (counts)
            best, tag = m["mae_flow"], " *"
            torch.save({"model": model.state_dict(), "config": vars(args),
                        "shape": (C, H, W), "scale": (lo, hi), "test_metrics": m}, args.save)
            tracker.log_best(ep, m)
        terms = " ".join(f"{k} {v:.5f}" for k, v in agg.items())
        epochs.set_postfix(best_mae=f"{best:.4f}")
        tqdm.write(f"ep {ep:3d} | {terms} | {fmt(m)} | {time.time() - t0:.0f}s{tag}")
    tracker.finish()


if __name__ == "__main__":
    main()