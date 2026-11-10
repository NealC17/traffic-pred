import io
import os
import time

import numpy as np
import torch

from trafficpred.utils import predict
from trafficpred.metrics import (HEADLINE, PANELS, REPORTED, REPORTED_NAMES, SPLITS, final_line, fmt,
                                 panel_metrics, summary_table)

PROGRESS_POINTS = 20
VIZ_EVERY = 10


class Tracker:
    def __init__(self, name, config, job_type, enabled=True, run_id=None, group=None):
        self.run = None
        self.wandb = None
        self.run_id = run_id
        self.group = group or os.environ.get("WANDB_RUN_GROUP")
        self.name = os.environ.get("WANDB_NAME", name)
        self.best = None
        self.test = None
        self.reference = None
        self.reference_val = None
        self.final = None
        self._evals = 0
        if not enabled or os.environ.get("WANDB_MODE") == "disabled":
            return
        try:
            import wandb
        except ImportError:
            print("[wandb] not installed (pip install wandb); continuing without it")
            return
        self.wandb = wandb
        kwargs = dict(project=os.environ.get("WANDB_PROJECT", "traffic-pred"), name=self.name,
                      group=self.group, job_type=job_type, config=config)
        if run_id is not None:
            kwargs.update(id=run_id, resume="allow")
        try:
            self.run = wandb.init(**kwargs)
        except Exception as e:
            print(f"[wandb] couldn't start an online run ({e}).\n"
                  f"[wandb] logging offline instead; upload later with:  wandb sync wandb/offline-run-*")
            self.run = wandb.init(mode="offline", **kwargs)
        self.run_id = self.run.id
        if run_id is not None:
            print(f"[wandb] resumed run {self.run_id} ({self.name}, group {self.group})", flush=True)
        wandb.define_metric("progress/step")
        wandb.define_metric("progress/*", step_metric="progress/step")
        wandb.define_metric("train/*", step_metric="progress/step")
        wandb.define_metric("epoch")
        wandb.define_metric("val/*", step_metric="epoch", summary="min")
        wandb.define_metric("test/*", step_metric="epoch", summary="last")
        wandb.define_metric("epoch/*", step_metric="epoch")
        wandb.define_metric("predictions", step_metric="epoch")
        for name in PANELS:
            wandb.define_metric(f"{name}/*", step_metric="epoch")

    def log_reference(self, val=None, test=None):
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

    def start(self, total_epochs, steps_per_epoch, start_epoch=0):
        self._total = total_epochs * steps_per_epoch
        self._steps_per_epoch = steps_per_epoch
        self._every = max(1, steps_per_epoch // PROGRESS_POINTS)
        self._step = start_epoch * steps_per_epoch
        self._first = self._step
        self._t0 = time.time()
        self._loss_sum, self._loss_n = 0.0, 0

    def step(self, loss, lr):
        if torch.is_tensor(loss):
            loss = loss.detach()
        self._step += 1
        self._loss_sum += loss
        self._loss_n += 1
        if self.run is None or (self._step % self._every and self._step != self._total):
            return
        rate = (self._step - self._first) / max(time.time() - self._t0, 1e-9)
        self.run.log({
            "progress/step": self._step,
            "progress/percent": 100.0 * self._step / self._total,
            "progress/epoch": self._step / self._steps_per_epoch,
            "progress/eta_min": (self._total - self._step) / max(rate, 1e-9) / 60,
            "progress/it_per_s": rate,
            "train/loss": float(self._loss_sum) / self._loss_n,
            "train/lr": lr,
        })
        self._loss_sum, self._loss_n = 0.0, 0

    def log_eval(self, epoch, metrics, train_loss=None, train_parts=None, epoch_time=None,
                 model=None, dataset=None, device=None, span=1.0, lo=0.0, channels=None, last=False,
                 curves=None):
        if self.run is None:
            return
        d = {"epoch": epoch, **{f"val/{k}": v for k, v in metrics.items()}}
        d.update(panel_metrics({"val": metrics, **(curves or {})}))
        if train_loss is not None:
            d["epoch/train_loss"] = float(train_loss)
        for k, v in (train_parts or {}).items():
            d[f"epoch/train_{k}"] = float(v)
        if epoch_time is not None:
            d["epoch/time_s"] = epoch_time
        if model is not None and dataset is not None and (self._evals % VIZ_EVERY == 0 or last):
            fig = prediction_figure(model, dataset, device, span, lo, epoch, channels)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=80)
            buf.seek(0)
            from PIL import Image
            d["predictions"] = self.wandb.Image(
                Image.open(buf), caption=f"val window {len(dataset) // 2}, epoch {epoch}")
        self._evals += 1
        self.run.log(d)

    def log_best(self, epoch, metrics):
        self.best = (epoch, metrics)

    def log_final(self, by_split, epoch=None):
        self.final = by_split
        self.test = by_split["test"]
        print(f"selected checkpoint (epoch {self.best[0] if self.best else '?'}) on test: "
              f"{fmt(self.test, 'test')}", flush=True)
        print(summary_table(by_split), flush=True)
        if self.run is None:
            return
        d = {f"test/{k}": v for k, v in self.test.items()}
        d.update(raw_test(self.test))
        if epoch is not None:
            d["epoch"] = epoch
        cols = ["run", "best_epoch"] + [f"{split}_{REPORTED_NAMES[k]}" for split in SPLITS for k in REPORTED]
        row = [self.name, self.best[0] if self.best else None] + [by_split[s][k] for s in SPLITS for k in REPORTED]
        d["results"] = self.wandb.Table(columns=cols, data=[row])
        self.run.log(d)

    def state(self):
        return {"run_id": self.run_id, "group": self.group, "best": self.best,
                "test": self.test, "final": self.final, "evals": self._evals}

    def restore(self, state):
        self.best = state.get("best")
        self.test = state.get("test")
        self.final = state.get("final")
        self._evals = state.get("evals", 0)

    def finish(self):
        if self.best is not None and self.final is not None:
            epoch, val_metrics = self.best
            print(final_line(self.name, self.final, epoch), flush=True)
            if self.run is not None:
                s = self.run.summary
                s["best_epoch"] = epoch
                for k, v in panel_metrics(self.final).items():
                    s[k] = v
                for split, metrics in self.final.items():
                    for k in REPORTED:
                        s[f"{split}_{REPORTED_NAMES[k]}"] = metrics[k]
                for k in HEADLINE:
                    s[k] = self.test[k]
                for k, v in val_metrics.items():
                    s[f"best_val/{k}"] = v
                if self.test is not None:
                    for k, v in self.test.items():
                        s[f"test/{k}"] = v
                    for k, v in raw_test(self.test).items():
                        s[k] = v
                    if self.reference is not None:
                        for k in HEADLINE:
                            ref = self.reference[k]
                            s[f"vs_copy_last/{k}"] = self.test[k] / ref if ref else float("nan")
        if self.run is not None:
            self.run.summary["progress/percent"] = 100.0
            self.run.finish()


def raw_test(metrics):
    return {f"test/test_{REPORTED_NAMES[k]}": metrics[k] for k in REPORTED}


@torch.no_grad()
def prediction_figure(model, dataset, device, span, lo, epoch, channels=None, idx=None):
    from matplotlib.figure import Figure
    idx = len(dataset) // 2 if idx is None else idx
    model.eval()
    x, y, *ext = dataset[idx]
    ext = ext[0].unsqueeze(0).to(device) if ext else None
    pred = predict(model, x.unsqueeze(0).to(device), ext).squeeze(0).float().clamp(0, 1).cpu()
    to_raw = lambda t: (t * span + lo).numpy()
    y, pred = to_raw(y), to_raw(pred)
    T, C = y.shape[:2]
    names = list(channels) if channels else [f"Channel {c}" for c in range(C)]
    rows = [(t, c) for t in sorted({0, T - 1}) for c in range(C)]

    fig = Figure(figsize=(10, 3.2 * len(rows)), constrained_layout=True)
    axes = fig.subplots(len(rows), 3, squeeze=False)
    for r, (t, c) in enumerate(rows):
        vmax = max(y[t, c].max(), pred[t, c].max())
        err = np.abs(pred[t, c] - y[t, c])
        actual = axes[r, 0].imshow(y[t, c], cmap="viridis", vmin=0, vmax=vmax)
        axes[r, 1].imshow(pred[t, c], cmap="viridis", vmin=0, vmax=vmax)
        error = axes[r, 2].imshow(err, cmap="Reds", vmin=0)
        axes[r, 0].set_title(f"Actual {names[c]}, t+{t + 1}")
        axes[r, 1].set_title(f"Predicted {names[c]}, t+{t + 1}")
        axes[r, 2].set_title(f"|Error|  MAE {err.mean():.1f}")
        fig.colorbar(actual, ax=axes[r, :2].tolist(), shrink=0.8)
        fig.colorbar(error, ax=axes[r, 2], shrink=0.8)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"epoch {epoch}")
    return fig
