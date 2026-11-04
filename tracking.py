"""
Shared evaluation + Weights & Biases logging for every model.

Every training script scores its model with evaluate() below, so all runs report the same
metrics whatever loss they were trained with, and their W&B charts overlay.

Protocol (train / val / test):
    Every epoch the model is scored on the VALIDATION set (data.taxibj_splits(val=True)),
    and the checkpoint with the lowest val SELECT metric (val MSE, the same for every model)
    is kept. After training, that checkpoint is scored ONCE on the TEST set. The test numbers
    are never used to make a decision, so they're an unbiased estimate.

Metrics (predictions are clamped to the data range [0, 1] first, as in OpenSTL):
  Everything at once (amount AND placement):
    mse, mae               per cell, in the [0, 1] scale data.py produces
    mse_frame, mae_frame   summed over a frame's cells, averaged over frames
                           (OpenSTL / PredFormer-paper convention)
    rmse_flow, mae_flow    per cell, in raw taxi counts
  WHERE the traffic is (each map normalised to sum to 1, so volume is removed):
    sinkhorn               debiased Sinkhorn divergence between the predicted and actual
                           distribution: distance-aware (a one-cell shift costs little)
                           (same function and eps / iters as DistVolLoss)
    dist_mse               per-cell squared error between the distributions, scaled by H*W
                           so a uniform map is all ones: ignores distance
    dist_tv                total variation, 1/2 sum|p_hat - p| per map, in [0, 1]: the
                           fraction of traffic predicted in the wrong cells
  HOW MUCH traffic there is (each map summed to its total, so placement is removed):
    volume_err             relative error of each map's total, sum|N_hat - N| / sum N
    volume_mae_flow        mean |N_hat - N| per map, in taxi counts
  Per prediction step (t+1 = first future frame), to show how error grows with the horizon:
    mse_t1.., sinkhorn_t1.., dist_mse_t1.., dist_tv_t1.., volume_err_t1..

Reference: copy-last-frame (predict every future frame = the last input frame), scored with
the same evaluate() on val and test. Each run stores it as copy_last_val/* and
copy_last_test/*, and reports its test metrics relative to it (vs_copy_last/*: < 1 means
better than copying). baseline_copy_last.py also logs it as its own W&B run named copy_last,
a flat line on every val chart.

W&B, per run:
    progress/percent, progress/epoch, progress/eta_min, progress/it_per_s,
    train/loss, train/lr              ~20 points per epoch: the live training progress
    val/<every metric above>          once per evaluated epoch (charts against "epoch")
    epoch/train_loss, epoch/time_s    once per epoch
    epoch/train_<term>                each term of a multi-part loss (DistVolLoss: l1,
                                      sinkhorn, volume), once per epoch
    predictions                       actual | predicted | |error| heatmaps for one fixed val
                                      window (t+1 and last step), every VIZ_EVERY evaluations
    test/<every metric above>         once, at the end: the selected checkpoint on test
    test_results                      a one-row table of the headline test metrics
    summary: mse, sinkhorn, dist_tv, volume_err   final TEST numbers of the selected checkpoint
             test/<metric>, best_val/<metric>, best_epoch,
             copy_last_test/<metric>, vs_copy_last/<metric>
In the W&B runs table, add the progress/percent column to see how far each run is.
To compare models, add the summary columns mse, sinkhorn, dist_tv, volume_err to the runs
table (or a bar-chart panel over test/mse, test/sinkhorn, ...).

Resuming: pass run_id (and group) to Tracker to continue an existing W&B run: same run page,
charts continue where they stopped. Tracker.run_id / Tracker.group give the values to store
in a checkpoint. A run that died mid-training shows as "crashed" until it is resumed.

Environment variables (all optional): WANDB_PROJECT (default "traffic-pred"), WANDB_ENTITY,
WANDB_NAME, WANDB_RUN_GROUP, WANDB_MODE ("offline" to log locally and `wandb sync` later,
"disabled" to turn W&B off). Without W&B installed or reachable, training still runs.
"""

import io
import os
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from distvol import split_flow
from sinkhorn import sinkhorn_divergence

SINKHORN_EPS = 0.01        # same defaults as DistVolLoss / train_distvol.py
SINKHORN_ITERS = 30
PROGRESS_POINTS = 20       # progress/train-loss points logged per epoch
VIZ_EVERY = 10             # evaluations between prediction images (plus the first and the last)
HEADLINE = ("mse", "sinkhorn", "dist_tv", "volume_err")
SELECT = "mse"             # val metric every script picks its checkpoint by (lower is better)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def sinkhorn_per_map(pred, y, eps=SINKHORN_EPS, iters=SINKHORN_ITERS):
    """Sinkhorn divergence for every map, shaped like pred's leading dims (B, T, C).
    Negative predictions are clipped to 0 (mass can't be negative)."""
    H, W = y.shape[-2:]
    p_hat, _ = split_flow(pred.float().clamp_min(0))
    p, _ = split_flow(y.float().clamp_min(0))
    s = sinkhorn_divergence(p_hat.reshape(-1, H, W), p.reshape(-1, H, W), eps=eps, iters=iters)
    return s.reshape(y.shape[:-2])


class CopyLast(nn.Module):
    """Reference model: every future frame = the last observed frame."""

    def __init__(self, t_out):
        super().__init__()
        self.t_out = t_out

    def forward(self, x):
        return x[:, -1:].expand(-1, self.t_out, *x.shape[2:])


@torch.no_grad()
def evaluate(model, loader, device, span, lo=0.0, desc="  eval"):
    """All test metrics (see module docstring). span, lo: the data's min-max scaling
    (raw = scaled * span + lo), used for the count-based metrics. Returns a dict of floats."""
    model.eval()
    T = None
    se_t = ae_t = sk_t = vabs_t = vtot_t = dse_t = tv_t = None
    n_cells = n_frames = n_maps = 0
    for x, y in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True, mininterval=1):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(x)
        if isinstance(pred, tuple):                       # forward(..., return_parts=True)
            pred = pred[0]
        pred = pred.float().clamp(0, 1)
        B, T, C, H, W = y.shape
        if se_t is None:
            # Accumulate on the device (float64 where supported) and copy back once at the
            # end: a .cpu() per batch makes the CPU wait for the GPU five times a batch.
            acc = torch.float32 if x.device.type == "mps" else torch.float64
            se_t, ae_t, sk_t, vabs_t, vtot_t, dse_t, tv_t = (
                torch.zeros(T, dtype=acc, device=x.device) for _ in range(7))
        d = pred - y
        se_t += d.pow(2).sum(dim=(0, 2, 3, 4)).to(se_t.dtype)
        ae_t += d.abs().sum(dim=(0, 2, 3, 4)).to(se_t.dtype)
        # distribution-only errors: both maps normalised to sum to 1 (volume removed)
        p_hat, _ = split_flow(pred)                                    # pred is in [0, 1]
        p, _ = split_flow(y.clamp_min(0))
        sk = sinkhorn_divergence(p_hat.reshape(-1, H, W), p.reshape(-1, H, W),
                                 eps=SINKHORN_EPS, iters=SINKHORN_ITERS).reshape(B, T, C)
        sk_t += sk.sum(dim=(0, 2)).to(se_t.dtype)
        dp = p_hat - p
        dse_t += (dp * (H * W)).pow(2).sum(dim=(0, 2, 3, 4)).to(se_t.dtype)
        tv_t += (0.5 * dp.abs().sum(dim=(-2, -1))).sum(dim=(0, 2)).to(se_t.dtype)
        # totals per map in raw counts: sum(scaled) * span + lo * H * W
        N_hat = pred.sum(dim=(-2, -1)) * span + lo * H * W            # (B, T, C)
        N = y.sum(dim=(-2, -1)) * span + lo * H * W
        vabs_t += (N_hat - N).abs().sum(dim=(0, 2)).to(se_t.dtype)
        vtot_t += N.abs().sum(dim=(0, 2)).to(se_t.dtype)
        n_cells += y.numel()
        n_frames += B * T
        n_maps += B * T * C

    se_t, ae_t, sk_t, vabs_t, vtot_t, dse_t, tv_t = (
        t.double().cpu() for t in (se_t, ae_t, sk_t, vabs_t, vtot_t, dse_t, tv_t))
    cells_t, maps_t = n_cells / T, n_maps / T                       # per prediction step
    cells_per_frame = n_cells / n_frames
    mse, mae = se_t.sum().item() / n_cells, ae_t.sum().item() / n_cells
    m = {
        "mse": mse,
        "mae": mae,
        "mse_frame": mse * cells_per_frame,
        "mae_frame": mae * cells_per_frame,
        "rmse_flow": float(np.sqrt(mse)) * span,
        "mae_flow": mae * span,
        "sinkhorn": sk_t.sum().item() / n_maps,
        "dist_mse": dse_t.sum().item() / n_cells,
        "dist_tv": tv_t.sum().item() / n_maps,
        "volume_err": vabs_t.sum().item() / max(vtot_t.sum().item(), 1e-12),
        "volume_mae_flow": vabs_t.sum().item() / n_maps,
    }
    for t in range(T):
        m[f"mse_t{t + 1}"] = se_t[t].item() / cells_t
        m[f"sinkhorn_t{t + 1}"] = sk_t[t].item() / maps_t
        m[f"dist_mse_t{t + 1}"] = dse_t[t].item() / cells_t
        m[f"dist_tv_t{t + 1}"] = tv_t[t].item() / maps_t
        m[f"volume_err_t{t + 1}"] = vabs_t[t].item() / max(vtot_t[t].item(), 1e-12)
    return m


def fmt(m, split="val"):
    """One-line summary of evaluate()'s dict for the console."""
    return (f"{split} MSE {m['mse']:.5f}  Sinkhorn {m['sinkhorn']:.5f}  TV {m['dist_tv']:.4f}  "
            f"volume err {m['volume_err']:.2%}  | MAE {m['mae_flow']:.3f} RMSE {m['rmse_flow']:.3f} "
            f"(counts)  frame MSE {m['mse_frame']:.3f}")


# ---------------------------------------------------------------------------
# W&B
# ---------------------------------------------------------------------------
class Tracker:
    """Everything a run sends to W&B (see module docstring). With W&B disabled, missing or
    offline it still prints the final line train_all.sh collects into results.tsv."""

    def __init__(self, name, config, job_type, enabled=True, run_id=None, group=None):
        """run_id: continue this existing W&B run (resume) instead of starting a new one.
        group: W&B group; default $WANDB_RUN_GROUP. When resuming, pass the run's stored group
        so it stays where it was even if this job has a different group."""
        self.run = None
        self.wandb = None
        self.run_id = run_id
        self.group = group or os.environ.get("WANDB_RUN_GROUP")
        self.name = os.environ.get("WANDB_NAME", name)
        self.best = None                  # (epoch, val metrics) of the saved checkpoint
        self.test = None                  # test metrics of that checkpoint (log_test)
        self.reference = None             # copy-last-frame metrics on test
        self.reference_val = None         # copy-last-frame metrics on val
        self._evals = 0
        self._total = 0
        if not enabled or os.environ.get("WANDB_MODE") == "disabled":
            return
        try:
            import wandb
        except ImportError:
            print("[wandb] not installed (pip install wandb); continuing without it")
            return
        self.wandb = wandb
        resuming = run_id is not None
        kwargs = dict(project=os.environ.get("WANDB_PROJECT", "traffic-pred"), name=self.name,
                      group=self.group, job_type=job_type, config=config)
        if resuming:
            kwargs.update(id=run_id, resume="allow")
        try:
            self.run = wandb.init(**kwargs)
        except Exception as e:                                # no login / no network
            print(f"[wandb] couldn't start an online run ({e}).\n"
                  f"[wandb] logging offline instead; upload later with:  wandb sync wandb/offline-run-*")
            self.run = wandb.init(mode="offline", **kwargs)
        self.run_id = self.run.id                            # stored in checkpoints for resuming
        if resuming:
            print(f"[wandb] resumed run {self.run_id} ({self.name}, group {self.group})", flush=True)
        wandb.define_metric("progress/step")
        wandb.define_metric("progress/*", step_metric="progress/step")
        wandb.define_metric("train/*", step_metric="progress/step")
        wandb.define_metric("epoch")
        # every metric is "lower is better": the summary keeps each val metric's minimum
        wandb.define_metric("val/*", step_metric="epoch", summary="min")
        wandb.define_metric("test/*", step_metric="epoch", summary="last")
        wandb.define_metric("epoch/*", step_metric="epoch")
        wandb.define_metric("predictions", step_metric="epoch")

    # ---- reference ---------------------------------------------------------
    def log_reference(self, val=None, test=None):
        """Copy-last-frame metrics on the val and/or test set (stored in the run summary)."""
        for split, metrics in (("val", val), ("test", test)):
            if metrics is None:
                continue
            if split == "val":
                self.reference_val = metrics
            else:
                self.reference = metrics
            print(f"copy-last-frame reference: {fmt(metrics, split)}", flush=True)
            if self.run is not None:
                for k, v in metrics.items():
                    self.run.summary[f"copy_last_{split}/{k}"] = v

    # ---- live training progress --------------------------------------------
    def start(self, total_epochs, steps_per_epoch, start_epoch=0):
        self._total = total_epochs * steps_per_epoch
        self._spe = steps_per_epoch
        self._every = max(1, steps_per_epoch // PROGRESS_POINTS)
        self._step = start_epoch * steps_per_epoch
        self._first = self._step
        self._t0 = time.time()
        self._loss_sum, self._loss_n = 0.0, 0

    def step(self, loss, lr):
        """Call once per training step; logs ~PROGRESS_POINTS points per epoch.
        loss may be a float or a 0-d tensor (summed on the device, read back only when logged)."""
        if torch.is_tensor(loss):
            loss = loss.detach()
        self._step += 1
        self._loss_sum += loss
        self._loss_n += 1
        if self.run is None or (self._step % self._every and self._step != self._total):
            return
        done = self._step - self._first
        rate = done / max(time.time() - self._t0, 1e-9)
        self.run.log({
            "progress/step": self._step,
            "progress/percent": 100.0 * self._step / self._total,
            "progress/epoch": self._step / self._spe,
            "progress/eta_min": (self._total - self._step) / max(rate, 1e-9) / 60,
            "progress/it_per_s": rate,
            "train/loss": float(self._loss_sum) / self._loss_n,   # mean since the last point
            "train/lr": lr,
        })
        self._loss_sum, self._loss_n = 0.0, 0

    # ---- per evaluation ------------------------------------------------------
    def log_eval(self, epoch, metrics, train_loss=None, train_parts=None, epoch_time=None,
                 model=None, dataset=None, device=None, span=1.0, lo=0.0, last=False,
                 split="val"):
        """Per-epoch evaluation (on the val set) -> val/<metric>, plus the epoch's training
        loss, the terms of a multi-part loss (train_parts: {"l1": ..., "sinkhorn": ...}) and
        every VIZ_EVERY evaluations a prediction image for one fixed window of `dataset`."""
        if self.run is None:
            return
        d = {"epoch": epoch, **{f"{split}/{k}": v for k, v in metrics.items()}}
        if train_loss is not None:
            d["epoch/train_loss"] = float(train_loss)
        for k, v in (train_parts or {}).items():
            d[f"epoch/train_{k}"] = float(v)
        if epoch_time is not None:
            d["epoch/time_s"] = epoch_time
        if model is not None and dataset is not None and (self._evals % VIZ_EVERY == 0 or last):
            fig = prediction_figure(model, dataset, device, span, lo, epoch)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=80)
            buf.seek(0)
            from PIL import Image
            d["predictions"] = self.wandb.Image(
                Image.open(buf), caption=f"{split} window {len(dataset) // 2}, epoch {epoch}")
        self._evals += 1
        self.run.log(d)

    def log_best(self, epoch, metrics):
        """Call when this epoch's weights become the saved checkpoint (metrics = val)."""
        self.best = (epoch, metrics)

    def log_test(self, metrics, epoch=None):
        """Call once, after training, with the selected checkpoint's TEST metrics.
        Logged as test/<metric> (one point, at `epoch`) and as a one-row table."""
        self.test = metrics
        print(f"selected checkpoint (epoch {self.best[0] if self.best else '?'}) on test: "
              f"{fmt(metrics, 'test')}", flush=True)
        if self.run is None:
            return
        d = {f"test/{k}": v for k, v in metrics.items()}
        if epoch is not None:
            d["epoch"] = epoch
        cols = ["run", "best_epoch"] + list(HEADLINE) + ["mae_flow", "rmse_flow", "dist_mse",
                                                         "volume_mae_flow"]
        row = [self.name, self.best[0] if self.best else None] + [metrics[k] for k in cols[2:]]
        d["test_results"] = self.wandb.Table(columns=cols, data=[row])
        self.run.log(d)

    def state(self):
        """What a training checkpoint needs to resume this tracker (see restore())."""
        return {"run_id": self.run_id, "group": self.group, "best": self.best,
                "test": self.test, "evals": self._evals}

    def restore(self, state):
        """Bring back best checkpoint / evaluation count from state() (run_id and group go
        to the constructor)."""
        self.best = state.get("best")
        self.test = state.get("test")
        self._evals = state.get("evals", 0)

    def finish(self):
        """Print the final line (train_all.sh reads it) and fill in the W&B run summary.
        The headline numbers are the TEST metrics of the selected checkpoint (log_test);
        if log_test was never called they fall back to its val metrics, marked as such."""
        if self.best is not None:
            epoch, mval = self.best
            m, split = (self.test, "test") if self.test is not None else (mval, "val")
            vals = "  ".join(f"{k}={m[k]:.6f}" for k in HEADLINE)
            print(f"FINAL {self.name}  {vals}  best_epoch={epoch}  ({split})", flush=True)
            if self.run is not None:
                s = self.run.summary
                s["best_epoch"] = epoch
                s["final_split"] = split
                for k in HEADLINE:
                    s[k] = m[k]
                for k, v in mval.items():
                    s[f"best_val/{k}"] = v
                if self.test is not None:
                    for k, v in self.test.items():
                        s[f"test/{k}"] = v
                    if self.reference is not None:
                        for k in HEADLINE:
                            ref = self.reference[k]
                            s[f"vs_copy_last/{k}"] = self.test[k] / ref if ref else float("nan")
        if self.run is not None:
            self.run.summary["progress/percent"] = 100.0
            self.run.finish()


@torch.no_grad()
def prediction_figure(model, dataset, device, span, lo, epoch, idx=None):
    """Inflow / outflow at t+1 and the last step: actual | predicted | |error|, in counts."""
    from matplotlib.figure import Figure
    idx = len(dataset) // 2 if idx is None else idx
    model.eval()
    x, y = dataset[idx]
    pred = model(x.unsqueeze(0).to(device))
    if isinstance(pred, tuple):
        pred = pred[0]
    pred = pred.squeeze(0).float().clamp(0, 1).cpu()
    to_counts = lambda t: (t * span + lo).numpy()
    y, pred = to_counts(y), to_counts(pred)
    T, C = y.shape[:2]
    names = ["Inflow", "Outflow"] if C == 2 else [f"ch{c}" for c in range(C)]
    rows = [(t, c) for t in sorted({0, T - 1}) for c in range(C)]

    fig = Figure(figsize=(10, 3.2 * len(rows)), constrained_layout=True)
    axes = fig.subplots(len(rows), 3, squeeze=False)
    for r, (t, c) in enumerate(rows):
        vmax = max(y[t, c].max(), pred[t, c].max())
        err = np.abs(pred[t, c] - y[t, c])
        a = axes[r, 0].imshow(y[t, c], cmap="viridis", vmin=0, vmax=vmax)
        axes[r, 1].imshow(pred[t, c], cmap="viridis", vmin=0, vmax=vmax)
        e = axes[r, 2].imshow(err, cmap="Reds", vmin=0)
        axes[r, 0].set_title(f"Actual {names[c]}, t+{t + 1}")
        axes[r, 1].set_title(f"Predicted {names[c]}, t+{t + 1}")
        axes[r, 2].set_title(f"|Error|  MAE {err.mean():.1f}")
        fig.colorbar(a, ax=axes[r, :2].tolist(), shrink=0.8)
        fig.colorbar(e, ax=axes[r, 2], shrink=0.8)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"epoch {epoch}")
    return fig
