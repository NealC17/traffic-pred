import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
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




@torch.no_grad()
def evaluate(model, loader, device):
    """(MSE, MAE) per frame in the [0, 1] scale, same definition as train.py:
    error summed over one frame's pixels, averaged over all predicted frames."""
    model.eval()
    se = ae = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).clamp(0, 1)
        se += (pred - y).pow(2).sum(dim=(2, 3, 4)).sum().item()
        ae += (pred - y).abs().sum(dim=(2, 3, 4)).sum().item()
        n += y.shape[0] * y.shape[1]
    return se / n, ae / n
 
 
def fit(model, train_set, test_set, epochs=50, batch_size=16, lr=1e-3,
        weight_decay=1e-2, device=None, save=None, workers=0):
    """Train with MSE loss, AdamW and a one-cycle LR schedule (the same recipe
    train.py uses for PredFormer, so the comparison is fair). Evaluates on
    test_set every epoch; returns the model with the best test-MSE weights.
    `save`: optional path to store those weights."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=workers)
    test_loader = DataLoader(test_set, batch_size=batch_size * 2, num_workers=workers)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr,
                                                total_steps=epochs * len(train_loader))
    print(f"{type(model).__name__}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, "
          f"{len(train_set)} train / {len(test_set)} test, device={device}")
 
    best, best_state = float("inf"), None
    for epoch in tqdm(range(epochs)):
        model.train()
        t0, running = time.time(), 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = F.mse_loss(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item()
        mse, mae = evaluate(model, test_loader, device)
        mark = ""
        if mse < best:
            best, mark = mse, "  *"
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if save:
                torch.save(best_state, save)
        print(f"epoch {epoch + 1}/{epochs}  train_loss {running / len(train_loader):.5f}  "
              f"test MSE {mse:.4f}  MAE {mae:.4f}  {time.time() - t0:.0f}s{mark}", flush=True)
 
    model.load_state_dict(best_state)
    model.eval()
    return model


if __name__ == "__main__":
    import argparse
    import os
    from data import taxibj_splits

    ap = argparse.ArgumentParser(description="Train the MLP baseline on TaxiBJ with MSE loss.")
    ap.add_argument("--tin", type=int, default=4)
    ap.add_argument("--tout", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--save", default="checkpoints/mlp_mse.pt")
    args = ap.parse_args()

    if os.path.dirname(args.save):
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
    train_set, test_set = taxibj_splits(args.tin, args.tout)
    fit(MLP(t_in=args.tin, t_out=args.tout, hidden=args.hidden, depth=args.depth),
        train_set, test_set, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, save=args.save)   
