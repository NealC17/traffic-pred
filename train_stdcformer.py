"""Train STDCformer (He et al., arXiv:2412.02942) on TaxiBJ through data.py.

The 32x32 grid becomes N = 1024 regions (one per cell), linked to their neighbouring
cells; the Laplacian eigenvectors of that grid graph are the spatial confounder.
TaxiBJ windows carry no timestamps, so the temporal confounder falls back to learned
embeddings of each step's position in the window (see README_stdcformer.md).

    python train_stdcformer.py                                  # paper split, ~9M params
    python train_stdcformer.py --d_model 64 --enc_layers 5      # the paper's own size (~1.2M)
    python train_stdcformer.py --train-files 2013 --val-files 2014 --test-files 2015
    python train_stdcformer.py --resume                         # continue an interrupted run
    python train_stdcformer.py --params_only                    # print the size and exit
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from stdcformer import STDCConfig, STDCformer, count_parameters, laplacian_pe


def grid_adjacency(h, w, neighbors=8):
    """(h*w, h*w) adjacency of an h x w grid; cell (r, c) -> index r*w + c."""
    steps = [(0, 1), (1, 0)] + ([(1, 1), (1, -1)] if neighbors == 8 else [])
    A = np.zeros((h * w, h * w), np.float32)
    for r in range(h):
        for c in range(w):
            for dr, dc in steps:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    A[r * w + c, rr * w + cc] = A[rr * w + cc, r * w + c] = 1.0
    return A


def to_nodes(frames):
    """(B, T, C, H, W) -> (B, T, H*W, C)"""
    B, T, C, H, W = frames.shape
    return frames.permute(0, 1, 3, 4, 2).reshape(B, T, H * W, C)


def to_grid(nodes, H, W):
    """(B, T, H*W, C) -> (B, T, C, H, W)"""
    B, T, N, C = nodes.shape
    return nodes.reshape(B, T, H, W, C).permute(0, 1, 4, 2, 3)


def get_args():
    import data
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    m = ap.add_argument_group("model")
    m.add_argument("--d_model", type=int, default=128)
    m.add_argument("--heads", type=int, default=8)
    m.add_argument("--enc_layers", type=int, default=11)
    m.add_argument("--dec_layers", type=int, default=None, help="default: same as --enc_layers")
    m.add_argument("--cta_layers", type=int, default=1)
    m.add_argument("--lap_dim", type=int, default=8)
    m.add_argument("--neighbors", type=int, default=8, choices=(4, 8), help="grid graph connectivity")
    m.add_argument("--dropout", type=float, default=0.1)
    m.add_argument("--no_node_id", action="store_true", help="drop the per-cell ID embedding")
    t = ap.add_argument_group("training")
    t.add_argument("--t_in", type=int, default=4)
    t.add_argument("--t_out", type=int, default=4)
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--batch_size", type=int, default=16)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--weight_decay", type=float, default=0.0)
    t.add_argument("--loss", choices=("mae", "mse"), default="mae", help="paper uses MAE")
    t.add_argument("--patience", type=int, default=50, help="early-stopping patience (epochs)")
    t.add_argument("--clip", type=float, default=5.0)
    t.add_argument("--amp", action="store_true", help="bf16 autocast on CUDA (less memory, faster)")
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--save", default="checkpoints/stdcformer.pt")
    t.add_argument("--resume", action="store_true", help="continue from <save>_last.pt if it exists")
    t.add_argument("--params_only", action="store_true")
    data.add_split_args(ap)
    return ap.parse_args()


def seed_all(seed):
    try:
        from seeding import seed_everything
        seed_everything(seed)
    except ImportError:
        import random
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def build_model(args, H, W, device):
    cfg = STDCConfig(num_nodes=H * W, in_dim=2, out_dim=2, input_len=args.t_in, output_len=args.t_out,
                     d_model=args.d_model, num_heads=args.heads, enc_layers=args.enc_layers,
                     dec_layers=args.dec_layers or args.enc_layers, cta_layers=args.cta_layers,
                     dropout=args.dropout, lap_dim=args.lap_dim, use_node_id=not args.no_node_id,
                     use_time_of_day=False, use_day_of_week=False, use_holiday=False,
                     steps_per_day=48)
    return STDCformer(cfg).to(device), cfg


@torch.no_grad()
def evaluate(model, loader, static, lo, hi, device, amp, H, W, desc):
    """Metrics on the [0, 1] scale data.py uses (MSE, MAE, per-frame MSE) and on raw counts."""
    model.eval()
    n, se, ae, se_raw, ae_raw = 0, 0.0, 0.0, 0.0, 0.0
    se_step = None
    for x, y in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            p = model({"x": to_nodes(x), **static})
        p = to_grid(p.float(), H, W).clamp(0, 1)
        d = p - y
        n += d.numel()
        se += d.pow(2).sum().item(); ae += d.abs().sum().item()
        dr = d * (hi - lo)
        se_raw += dr.pow(2).sum().item(); ae_raw += dr.abs().sum().item()
        s = d.pow(2).sum(dim=(0, 2, 3, 4))
        se_step = s if se_step is None else se_step + s
    per_step = (se_step / (n / d.shape[1])).tolist()
    return {"mse": se / n, "mae": ae / n, "mse_per_step": per_step,
            "rmse_raw": (se_raw / n) ** 0.5, "mae_raw": ae_raw / n}


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    amp = args.amp and device == "cuda"

    import data
    train, val, test = data.taxibj_splits(args.t_in, args.t_out, val=True, **data.split_kwargs(args))
    lo, hi = data.scale_of(train)
    x0, _ = train[0]
    _, C, H, W = x0.shape
    assert C == 2, f"expected inflow/outflow channels, got {C}"

    seed_all(args.seed)
    model, cfg = build_model(args, H, W, device)
    print(f"STDCformer: {count_parameters(model):,} params | grid {H}x{W} -> {H*W} regions | "
          f"windows train/val/test = {len(train)}/{len(val)}/{len(test)} | device {device}"
          + (f" ({torch.cuda.get_device_name()})" if device == "cuda" else ""))
    if args.params_only:
        return

    static = {"lap_pe": torch.from_numpy(laplacian_pe(grid_adjacency(H, W, args.neighbors),
                                                      args.lap_dim)).to(device)}
    kw = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=device == "cuda",
              persistent_workers=args.workers > 0)
    tr_loader = DataLoader(train, shuffle=True, drop_last=True, **kw)
    va_loader = DataLoader(val, shuffle=False, **kw)
    te_loader = DataLoader(test, shuffle=False, **kw)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
    loss_fn = torch.nn.L1Loss() if args.loss == "mae" else torch.nn.MSELoss()

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    last_path = os.path.splitext(args.save)[0] + "_last.pt"
    start, best, bad = 1, float("inf"), 0
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device)
        if ck["config"] != cfg.__dict__:
            raise SystemExit(f"{last_path} has a different model config; delete it or drop --resume")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start, best, bad = ck["epoch"] + 1, ck["best"], ck["bad"]
        torch.set_rng_state(ck["rng"])
        print(f"resumed from {last_path} at epoch {start} (best val MAE {best:.5f})")

    for ep in range(start, args.epochs + 1):
        model.train(); t0, tot, cnt = time.time(), 0.0, 0
        bar = tqdm(tr_loader, desc=f"epoch {ep}/{args.epochs}", dynamic_ncols=True)
        for x, y in bar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                p = model({"x": to_nodes(x), **static})
            loss = loss_fn(p.float(), to_nodes(y))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            tot += loss.item(); cnt += 1
            bar.set_postfix(loss=f"{tot / cnt:.5f}")

        v = evaluate(model, va_loader, static, lo, hi, device, amp, H, W, "val")
        sched.step(v["mae"])
        improved = v["mae"] < best
        if improved:
            best, bad = v["mae"], 0
            torch.save({"model": model.state_dict(), "config": cfg.__dict__, "args": vars(args),
                        "scale": (lo, hi), "epoch": ep, "val": v}, args.save)
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "config": cfg.__dict__, "epoch": ep, "best": best, "bad": bad,
                    "rng": torch.get_rng_state()}, last_path)
        print(f"epoch {ep:3d} | train {args.loss} {tot / cnt:.5f} | val MSE {v['mse']:.5f} MAE {v['mae']:.5f} "
              f"| raw RMSE {v['rmse_raw']:.2f} MAE {v['mae_raw']:.2f} | lr {opt.param_groups[0]['lr']:.1e} "
              f"| {time.time() - t0:.0f}s" + (" *" if improved else ""))
        if bad >= args.patience:
            print(f"early stop: no val improvement for {args.patience} epochs")
            break

    model.load_state_dict(torch.load(args.save, map_location=device)["model"])
    res = evaluate(model, te_loader, static, lo, hi, device, amp, H, W, "test")
    print("TEST", json.dumps(res, indent=2))
    with open(os.path.splitext(args.save)[0] + "_test.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
