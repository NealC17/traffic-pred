import numpy as np
import torch
from tqdm import tqdm

from trafficpred.sinkhorn import sinkhorn_divergence, split_flow
from trafficpred.utils import predict, to_device

SINKHORN_EPS = 0.01
SINKHORN_ITERS = 30
HEADLINE = ("mse", "sinkhorn", "dist_tv", "volume_err")
SELECT = "mse"
SPLITS = ("train", "val", "test")
REPORTED = ("mse_flow", "rmse_flow", "mae_flow")
REPORTED_NAMES = {"mse_flow": "mse", "rmse_flow": "rmse", "mae_flow": "mae"}
PANELS = {"RMSE": "rmse_flow", "MAE": "mae_flow", "MSE": "mse_flow", "Sinkhorn": "sinkhorn",
          "TV": "dist_tv", "Frame MSE": "mse_frame"}


def panel_metrics(by_split):
    return {f"{name}/{split}": m[key] for split, m in by_split.items() for name, key in PANELS.items()}


class _Totals:
    NAMES = ("se", "ae", "sinkhorn", "volume_abs", "volume_total", "dist_se", "tv")

    def __init__(self, steps, device):
        dtype = torch.float32 if device.type == "mps" else torch.float64
        self.sums = {k: torch.zeros(steps, dtype=dtype, device=device) for k in self.NAMES}
        self.cells = self.frames = self.maps = 0

    def add(self, name, per_step):
        self.sums[name] += per_step.to(self.sums[name].dtype)

    def numpy(self):
        return {k: v.double().cpu() for k, v in self.sums.items()}


@torch.no_grad()
def evaluate(model, loader, device, span, lo=0.0, amp=False, desc="  eval"):
    model.eval()
    totals = None
    for batch in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True, mininterval=1):
        x, y, ext = to_device(batch, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            pred = predict(model, x, ext)
        pred = pred.float().clamp(0, 1)
        B, T, C, H, W = y.shape
        totals = totals or _Totals(T, x.device)

        diff = pred - y
        totals.add("se", diff.pow(2).sum(dim=(0, 2, 3, 4)))
        totals.add("ae", diff.abs().sum(dim=(0, 2, 3, 4)))

        p_hat, _ = split_flow(pred)
        p, _ = split_flow(y.clamp_min(0))
        sk = sinkhorn_divergence(p_hat.reshape(-1, H, W), p.reshape(-1, H, W),
                                 eps=SINKHORN_EPS, iters=SINKHORN_ITERS).reshape(B, T, C)
        totals.add("sinkhorn", sk.sum(dim=(0, 2)))
        dp = p_hat - p
        totals.add("dist_se", (dp * (H * W)).pow(2).sum(dim=(0, 2, 3, 4)))
        totals.add("tv", (0.5 * dp.abs().sum(dim=(-2, -1))).sum(dim=(0, 2)))

        n_hat = pred.sum(dim=(-2, -1)) * span + lo * H * W
        n = y.sum(dim=(-2, -1)) * span + lo * H * W
        totals.add("volume_abs", (n_hat - n).abs().sum(dim=(0, 2)))
        totals.add("volume_total", n.abs().sum(dim=(0, 2)))

        totals.cells += y.numel()
        totals.frames += B * T
        totals.maps += B * T * C
    return _summarise(totals, span)


def _summarise(totals, span):
    s = totals.numpy()
    T = len(s["se"])
    cells, maps = totals.cells, totals.maps
    cells_per_step, maps_per_step = cells / T, maps / T
    cells_per_frame = cells / totals.frames
    mse, mae = s["se"].sum().item() / cells, s["ae"].sum().item() / cells
    volume_total = max(s["volume_total"].sum().item(), 1e-12)
    m = {
        "mse": mse,
        "mae": mae,
        "mse_frame": mse * cells_per_frame,
        "mae_frame": mae * cells_per_frame,
        "mse_flow": mse * span ** 2,
        "rmse_flow": float(np.sqrt(mse)) * span,
        "mae_flow": mae * span,
        "sinkhorn": s["sinkhorn"].sum().item() / maps,
        "dist_mse": s["dist_se"].sum().item() / cells,
        "dist_tv": s["tv"].sum().item() / maps,
        "volume_err": s["volume_abs"].sum().item() / volume_total,
        "volume_mae_flow": s["volume_abs"].sum().item() / maps,
    }
    for t in range(T):
        m[f"mse_t{t + 1}"] = s["se"][t].item() / cells_per_step
        m[f"sinkhorn_t{t + 1}"] = s["sinkhorn"][t].item() / maps_per_step
        m[f"dist_mse_t{t + 1}"] = s["dist_se"][t].item() / cells_per_step
        m[f"dist_tv_t{t + 1}"] = s["tv"][t].item() / maps_per_step
        m[f"volume_err_t{t + 1}"] = s["volume_abs"][t].item() / max(s["volume_total"][t].item(), 1e-12)
    return m


def final_line(name, by_split, best_epoch):
    values = "  ".join(f"{split}_{REPORTED_NAMES[k]}={by_split[split][k]:.6f}"
                       for split in SPLITS if split in by_split for k in REPORTED)
    return f"FINAL {name}  {values}  best_epoch={best_epoch}"


def summary_table(by_split):
    rows = [f"{'':6s}{'MSE':>14s}{'RMSE':>12s}{'MAE':>12s}   (raw units)"]
    for split in SPLITS:
        if split in by_split:
            m = by_split[split]
            rows.append(f"{split:6s}{m['mse_flow']:14.4f}{m['rmse_flow']:12.4f}{m['mae_flow']:12.4f}")
    return "\n".join(rows)


def fmt(m, split="val"):
    return (f"{split} MSE {m['mse']:.5f}  Sinkhorn {m['sinkhorn']:.5f}  TV {m['dist_tv']:.4f}  "
            f"volume err {m['volume_err']:.2%}  | MAE {m['mae_flow']:.3f} RMSE {m['rmse_flow']:.3f} "
            f"(raw units)  frame MSE {m['mse_frame']:.3f}")
