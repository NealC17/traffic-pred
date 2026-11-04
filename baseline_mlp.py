import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import time
from tqdm import tqdm
from devices import pick_device
from tracking import Tracker, CopyLast, evaluate, fmt, SELECT
class MLP(nn.Module):
    """Fully connected baseline. Flattens every input value (T_in x C x H x W =
    4 x 2 x 32 x 32 = 8192 for TaxiBJ) into one vector, passes it through
    hidden layers, and outputs T_out x C x H x W values reshaped into frames.
 
    It has no notion of space or time: neighboring cells and consecutive
    frames are just different input positions, so it has to learn every
    spatial/temporal relationship from scratch. That's what makes it a useful
    floor for the CNN and PredFormer results."""
 
    def __init__(self, t_in=4, t_out=4, channels=2, height=32, width=32,
                 hidden=512, depth=3, dropout=0.1):
        super().__init__()
        self.t_in, self.t_out, self.c = t_in, t_out, channels
        d_in = t_in * channels * height * width
        d_out = t_out * channels * height * width
        layers, d = [], d_in
        for _ in range(depth - 1):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        layers.append(nn.Linear(d, d_out))
        self.net = nn.Sequential(*layers)
 
    def forward(self, x):
        B, T, C, H, W = x.shape
        y = self.net(x.reshape(B, -1))                    # (B, T_out*C*H*W)
        return y.view(B, self.t_out, C, H, W)




def fit(model, train_set, val_set, test_set, epochs=50, batch_size=16, lr=1e-3,
        weight_decay=1e-2, device=None, save=None, workers=0, run_name=None, wandb=True,
        extra_config=None):
    """Train with MSE loss, AdamW and a one-cycle LR schedule (the same recipe
    train.py uses for PredFormer, so the comparison is fair). Evaluates on
    val_set every epoch and keeps the weights with the lowest val MSE; those are
    scored once on test_set at the end. Returns the model with the selected weights.
    `save`: optional path to store those weights (and their val / test metrics).
    Logs to Weights & Biases (see tracking.py) unless wandb=False."""
    import data
    try:
        lo, hi = data.scale_of(train_set)                 # train files only, for a file split
    except TypeError:
        lo, hi = data._flow_scale or (0.0, 1.0)
    device = device or pick_device()
    model = model.to(device)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=workers)
    val_loader = DataLoader(val_set, batch_size=batch_size * 2, num_workers=workers)
    test_loader = DataLoader(test_set, batch_size=batch_size * 2, num_workers=workers)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr,
                                                total_steps=epochs * len(train_loader))
    print(f"{type(model).__name__}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, "
          f"{len(train_set)} train / {len(val_set)} val / {len(test_set)} test, device={device}")
 
    tracker = Tracker(
        name=run_name or "mlp_mse", job_type="mlp", enabled=wandb,
        config=dict(model=type(model).__name__.lower(), loss="mse", epochs=epochs,
                    batch_size=batch_size, lr=lr, weight_decay=weight_decay,
                    t_in=getattr(model, "t_in", None), t_out=getattr(model, "t_out", None),
                    params=sum(p.numel() for p in model.parameters()), **(extra_config or {})))
    tracker.log_reference(
        val=evaluate(CopyLast(model.t_out), val_loader, device, hi - lo, lo, desc="  copy-last (val)"),
        test=evaluate(CopyLast(model.t_out), test_loader, device, hi - lo, lo, desc="  copy-last (test)"))
    tracker.start(epochs, len(train_loader))

    best, best_state, best_epoch, best_val = float("inf"), None, None, None
    bar_ep = tqdm(range(epochs), desc=type(model).__name__, unit="ep", dynamic_ncols=True, mininterval=1)
    for epoch in bar_ep:
        model.train()
        t0, running = time.time(), 0.0
        bar = tqdm(train_loader, desc=f"  epoch {epoch + 1}", leave=False, dynamic_ncols=True, mininterval=1)
        for i, (x, y) in enumerate(bar, 1):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = F.mse_loss(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item()
            bar.set_postfix(loss=f"{running / i:.5f}")
            tracker.step(loss.item(), sched.get_last_lr()[0])
        m = evaluate(model, val_loader, device, hi - lo, lo)
        tracker.log_eval(epoch + 1, m, train_loss=running / len(train_loader),
                         epoch_time=time.time() - t0, model=model, dataset=val_set,
                         device=device, span=hi - lo, lo=lo, last=epoch + 1 == epochs)
        mark = ""
        if m[SELECT] < best:                              # select on VAL, never on test
            best, mark, best_epoch, best_val = m[SELECT], "  *", epoch + 1, m
            tracker.log_best(epoch + 1, m)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if save:
                torch.save(best_state, save)
        bar_ep.set_postfix(best_val_mse=f"{best:.5f}")
        tqdm.write(f"epoch {epoch + 1}/{epochs}  train_loss {running / len(train_loader):.5f}  "
                   f"{fmt(m)}  {time.time() - t0:.0f}s{mark}")
 
    # final: the selected weights, scored once on the test set
    model.load_state_dict(best_state)
    m_test = evaluate(model, test_loader, device, hi - lo, lo, desc="  test")
    tracker.log_test(m_test, epoch=epochs)
    if save:
        torch.save(best_state, save)                      # weights only, as before
        torch.save({"epoch": best_epoch, "val_metrics": best_val, "test_metrics": m_test},
                   os.path.splitext(save)[0] + ".metrics.pt")
    tracker.finish()
    model.eval()
    return model


if __name__ == "__main__":
    import argparse
    import data

    ap = argparse.ArgumentParser(description="Train the MLP baseline on TaxiBJ with MSE loss.")
    ap.add_argument("--tin", type=int, default=4)
    ap.add_argument("--tout", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--save", default="checkpoints/mlp_mse.pt")
    ap.add_argument("--no-wandb", action="store_true", help="don't log to Weights & Biases")
    data.add_split_args(ap)
    args = ap.parse_args()
    split = data.split_kwargs(args)

    if os.path.dirname(args.save):
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
    train_set, val_set, test_set = data.taxibj_splits(args.tin, args.tout, val=True, **split)
    fit(MLP(t_in=args.tin, t_out=args.tout, hidden=args.hidden, depth=args.depth),
        train_set, val_set, test_set, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, save=args.save, wandb=not args.no_wandb, extra_config=split,
        run_name="mlp_mse" + ("" if (args.tin, args.tout) == (4, 4) else f"_in{args.tin}_out{args.tout}"))
