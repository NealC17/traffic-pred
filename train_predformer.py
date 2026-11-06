"""
Train PredFormer on a dataset from data.py.
 
    python train.py --dataset taxibj
    python train.py --dataset mmnist --epochs 200 --amp
 
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
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
 
import data
from seeding import seed_everything
from predformer import PredFormer, CONFIGS
 
 
def get_splits(name):
    if name == "taxibj":
        return data.taxibj_splits()
    if name == "mmnist":
        return data.moving_mnist_splits()
    raise ValueError(name)
 
 
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
 
 
@torch.no_grad()
def evaluate(model, loader, device, amp):
    """Returns (MSE, MAE) per frame: squared/absolute error summed over the
    pixels of one frame, averaged over all predicted frames (OpenSTL-style),
    in the [0, 1] scale data.py produces."""
    model.eval()
    se = ae = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            pred = model(x)
        pred = pred.float().clamp(0, 1)
        se += (pred - y).pow(2).sum(dim=(2, 3, 4)).sum().item()
        ae += (pred - y).abs().sum(dim=(2, 3, 4)).sum().item()
        n += y.shape[0] * y.shape[1]
    return se / n, ae / n
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(CONFIGS), default="taxibj")
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
    args = ap.parse_args()
 
    seed_everything(args.seed)
    cfg = CONFIGS[args.dataset]
    hp = dict(cfg["train"])
    for k in ("epochs", "batch_size", "lr", "weight_decay"):
        if getattr(args, k) is not None:
            hp[k] = getattr(args, k)
 
    device = pick_device()
    out_dir = os.path.join(args.out, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
 
    # ---- data ----
    train_set, test_set = get_splits(args.dataset)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=hp["batch_size"], shuffle=True,
                              num_workers=args.workers, pin_memory=pin, drop_last=True,
                              persistent_workers=args.workers > 0)
    test_loader = DataLoader(test_set, batch_size=hp["batch_size"] * 2, shuffle=False,
                             num_workers=args.workers, pin_memory=pin)
    print(f"{args.dataset}: {len(train_set)} train / {len(test_set)} test samples, "
          f"device={device}")
 
    # ---- model / optim ----
    model = PredFormer(**cfg["model"]).to(device)
    print(f"PredFormer {cfg['model']['variant']}: "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")
 
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    total_steps = hp["epochs"] * len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=hp["lr"], total_steps=total_steps)
 
    start_epoch, best = 0, float("inf")
    last_path, best_path = os.path.join(out_dir, "last.pt"), os.path.join(out_dir, "best.pt")
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed from epoch {ck['epoch']} (best test MSE {best:.4f})")
 
    def save(path, epoch, extra=None):
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict(), epoch=epoch, best=best,
                        config=cfg["model"], dataset=args.dataset, **(extra or {})), path)
 
    # ---- train ----
    for epoch in tqdm(range(start_epoch, hp["epochs"])):
        model.train()
        t0, running = time.time(), 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.amp):
                pred = model(x)
            loss = F.mse_loss(pred.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item()
 
        msg = (f"epoch {epoch + 1}/{hp['epochs']}  train_loss {running / len(train_loader):.5f}  "
               f"lr {sched.get_last_lr()[0]:.2e}  {time.time() - t0:.0f}s")
 
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == hp["epochs"]:
            mse, mae = evaluate(model, test_loader, device, args.amp)
            msg += f"  | test MSE {mse:.4f}  MAE {mae:.4f}"
            if mse < best:
                best = mse
                save(best_path, epoch, dict(test_mse=mse, test_mae=mae))
                msg += "  *"
        save(last_path, epoch)
        print(msg, flush=True)
 
    print(f"done. best test MSE {best:.4f} -> {best_path}")
 
 
if __name__ == "__main__":
    main()
