import math
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from trafficpred.metrics import evaluate, fmt
from trafficpred.models import CopyLast
from trafficpred.utils import predict, rng_state, save_atomic, set_rng_state, to_device

PROGRESS_EVERY = 50


@dataclass
class TrainSettings:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    clip: float
    schedule: str
    patience: int
    optimizer: str = "adamw"
    plateau_patience: int = 10
    select: str = "mse"
    amp: bool = False
    compile: bool = False
    workers: int = 2
    eval_batch_size: int = 128
    data_on_gpu: str = "auto"
    curves: str = "all"

    def must_match_on_resume(self):
        d = asdict(self)
        for k in ("amp", "compile", "workers", "eval_batch_size", "data_on_gpu", "patience", "curves"):
            d.pop(k)
        return d


class DeviceBatches:
    def __init__(self, dataset, batch_size, device, shuffle=False, drop_last=False, workers=0):
        parts = []
        for batch in tqdm(DataLoader(dataset, batch_size=512, num_workers=workers),
                          desc="  copying data to the GPU", leave=False, mininterval=1):
            parts.append(batch)
        self.tensors = [torch.cat(column).to(device) for column in zip(*parts)]
        self.x = self.tensors[0]
        self.batch_size, self.shuffle, self.drop_last = batch_size, shuffle, drop_last

    def view(self, batch_size, shuffle=False, drop_last=False):
        other = object.__new__(DeviceBatches)
        other.tensors, other.x = self.tensors, self.x
        other.batch_size, other.shuffle, other.drop_last = batch_size, shuffle, drop_last
        return other

    def __len__(self):
        n = len(self.x)
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)

    def __iter__(self):
        n = len(self.x)
        order = torch.randperm(n, device=self.x.device) if self.shuffle else torch.arange(n, device=self.x.device)
        for i in range(len(self)):
            idx = order[i * self.batch_size:(i + 1) * self.batch_size]
            yield tuple(t[idx] for t in self.tensors)


def _differences(mine, theirs):
    return sorted(k for part in mine for k, v in mine[part].items()
                  if theirs.get(part, {}).get(k) != v)


def make_loaders(splits, settings, device):
    cuda = device.type == "cuda"
    on_gpu = settings.data_on_gpu == "yes" or (settings.data_on_gpu == "auto" and cuda)
    plan = (("train", splits.train, settings.batch_size, True, splits.train_is_fixed),
            ("val", splits.val, settings.eval_batch_size, False, True),
            ("test", splits.test, settings.eval_batch_size, False, True))
    loaders = {}
    for name, ds, batch_size, train, fixed in plan:
        if on_gpu and fixed:
            try:
                loaders[name] = DeviceBatches(ds, batch_size, device, shuffle=train, drop_last=train,
                                              workers=settings.workers)
                continue
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"{name} split doesn't fit on the GPU; using a DataLoader")
        loaders[name] = DataLoader(ds, batch_size, shuffle=train, drop_last=train,
                                   num_workers=settings.workers, pin_memory=cuda,
                                   persistent_workers=train and settings.workers > 0)
    if settings.curves == "all":
        n = min(len(splits.train), max(len(splits.val), 1))
        sample = np.sort(np.random.default_rng(0).choice(len(splits.train), n, replace=False))
        loaders["train_sample"] = DataLoader(Subset(splits.train, sample.tolist()), settings.eval_batch_size,
                                             num_workers=settings.workers, pin_memory=cuda)
    if isinstance(loaders["train"], DeviceBatches):
        loaders["train_eval"] = loaders["train"].view(settings.eval_batch_size)
    else:
        loaders["train_eval"] = DataLoader(splits.train, settings.eval_batch_size,
                                           num_workers=settings.workers, pin_memory=cuda)
    return loaders


class Trainer:
    def __init__(self, model, run, splits, loss, settings, save_path, device):
        self.net = model.to(device)
        self.model = torch.compile(self.net) if settings.compile else self.net
        self.run = run
        self.splits = splits
        self.loss = loss.to(device) if loss is not None else None
        if hasattr(self.loss, "bind"):  # losses that look at the model's own gradients
            self.loss.bind(self.net.parameters())
        self.settings = settings
        self.tracker = None
        self.device = device
        self.best_path = save_path
        self.last_path = save_path and os.path.splitext(save_path)[0] + ".last.pt"
        self.best_checkpoint = None
        self.amp = settings.amp and device.type == "cuda"
        loaders = make_loaders(splits, settings, device)
        self.train_loader, self.val_loader, self.test_loader = loaders["train"], loaders["val"], loaders["test"]
        self.train_eval_loader = loaders["train_eval"]
        self.train_sample_loader = loaders.get("train_sample")
        self.start_epoch, self.best, self.bad_epochs = 0, float("inf"), 0
        self.opt = self.sched = None
        if any(p.requires_grad for p in self.net.parameters()):
            self._make_optimizer()

    def _make_optimizer(self):
        s = self.settings
        optimizer = torch.optim.AdamW if s.optimizer == "adamw" else torch.optim.Adam
        groups = [{"params": list(self.net.parameters())}]
        loss_params = list(self.loss.parameters()) if self.loss is not None else []
        if loss_params:  # e.g. learned uncertainty weights: trained with the model, never decayed
            groups.append({"params": loss_params, "weight_decay": 0.0})
        try:
            self.opt = optimizer(groups, lr=s.lr, weight_decay=s.weight_decay,
                                 fused=self.device.type == "cuda")
        except (TypeError, RuntimeError):
            self.opt = optimizer(groups, lr=s.lr, weight_decay=s.weight_decay)
        if s.schedule == "onecycle":
            self.sched = torch.optim.lr_scheduler.OneCycleLR(
                self.opt, s.lr, total_steps=s.epochs * len(self.train_loader))
        else:
            self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.opt, mode="min", factor=0.5, patience=s.plateau_patience, min_lr=5e-6)

    def resume_settings(self):
        run = {k: v for k, v in self.run.items() if k != "seed"}
        return {"run": run, "train": self.settings.must_match_on_resume()}

    def try_resume(self):
        if self.opt is None or self.last_path is None:
            return None
        if not os.path.exists(self.last_path):
            print(f"no {self.last_path}; starting from scratch")
            return None
        ck = torch.load(self.last_path, map_location=self.device, weights_only=False)
        changed = _differences(self.resume_settings(), ck.get("resume_settings") or {})
        if changed:
            print(f"!!! {self.last_path} was made with different {changed}; starting over")
            return None
        self.net.load_state_dict(ck["model"])
        if ck.get("loss_state"):
            self.loss.load_state_dict(ck["loss_state"])
        self.opt.load_state_dict(ck["opt"])
        self.sched.load_state_dict(ck["sched"])
        self.start_epoch, self.best, self.bad_epochs = ck["epoch"], ck["best"], ck["bad_epochs"]
        set_rng_state(ck["rng"])
        if ck.get("seed") != self.run.get("seed"):
            print(f"note: {self.last_path} was started with seed {ck.get('seed')}; continuing it as it was")
            self.run["seed"] = ck.get("seed")
        print(f"resuming from {self.last_path}: {self.start_epoch}/{self.settings.epochs} epochs done, "
              f"best val {self.settings.select} {self.best:.5f}")
        return ck["tracker"]

    def evaluate(self, model, loader, desc):
        return evaluate(model, loader, self.device, self.splits.span, self.splits.lo, self.amp, desc)

    def log_reference(self):
        reference = CopyLast(self.splits.t_out)
        self.tracker.log_reference(val=self.evaluate(reference, self.val_loader, "  copy-last (val)"),
                                   test=self.evaluate(reference, self.test_loader, "  copy-last (test)"))

    def fit(self, tracker):
        self.tracker = tracker
        if self.opt is None:
            return self._score_untrained()
        s = self.settings
        self.log_reference()
        self.tracker.start(s.epochs, len(self.train_loader), self.start_epoch)
        epochs = tqdm(range(self.start_epoch + 1, s.epochs + 1), desc=self.tracker.name, unit="ep",
                      initial=self.start_epoch, total=s.epochs, dynamic_ncols=True, mininterval=1)
        for epoch in epochs:
            t0 = time.time()
            train_loss, parts = self._train_epoch(epoch)
            m = self.evaluate(self.net, self.val_loader, "  val")
            if s.schedule == "plateau":
                self.sched.step(m[s.select])
            curves = None
            if self.train_sample_loader is not None:
                curves = {"train": self.evaluate(self.net, self.train_sample_loader, "  train sample"),
                          "test": self.evaluate(self.net, self.test_loader, "  test")}
            self.tracker.log_eval(epoch, m, train_loss=train_loss, train_parts=parts, curves=curves,
                                  epoch_time=time.time() - t0, model=self.net, dataset=self.splits.val,
                                  device=self.device, span=self.splits.span, lo=self.splits.lo,
                                  channels=self.splits.channels, last=epoch == s.epochs)
            improved = m[s.select] < self.best
            if improved:
                self.best, self.bad_epochs = m[s.select], 0
                self._save_best(epoch, m)
                self.tracker.log_best(epoch, m)
            else:
                self.bad_epochs += 1
            self._save_last(epoch)
            terms = " ".join(f"{k} {v:.5f}" for k, v in parts.items())
            epochs.set_postfix(best_val=f"{self.best:.5f}")
            tqdm.write(f"ep {epoch:3d} | {terms} | {fmt(m)} | {time.time() - t0:.0f}s"
                       + (" *" if improved else ""))
            if s.patience and self.bad_epochs >= s.patience:
                tqdm.write(f"early stop: no val improvement for {s.patience} epochs")
                break
        self._final()

    def _train_epoch(self, epoch):
        self.model.train()
        totals, running = {}, 0.0
        bar = tqdm(self.train_loader, desc=f"  epoch {epoch}", leave=False, dynamic_ncols=True,
                   mininterval=1)
        for i, batch in enumerate(bar, 1):
            x, y, ext = to_device(batch, self.device)
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp):
                if self.loss.needs_parts:
                    pred = predict(self.model, x, ext, return_parts=True)
                else:
                    pred = predict(self.model, x, ext)
            loss, parts = self.loss(pred, y)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.settings.clip > 0:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.settings.clip)
            self.opt.step()
            if self.settings.schedule == "onecycle":
                self.sched.step()
            for k, v in parts.items():
                totals[k] = totals.get(k, 0.0) + v
            loss = loss.detach()
            running = running + loss
            if i % PROGRESS_EVERY == 0:
                bar.set_postfix(loss=f"{loss.item():.5f}")
            self.tracker.step(loss, self.opt.param_groups[0]["lr"])
        n = len(self.train_loader)
        return float(running) / n, {k: float(v) / n for k, v in totals.items()}

    def _score_all(self):
        return {"train": self.evaluate(self.net, self.train_eval_loader, "  train"),
                "val": self.evaluate(self.net, self.val_loader, "  val"),
                "test": self.evaluate(self.net, self.test_loader, "  test")}

    def _score_untrained(self):
        scores = self._score_all()
        print(fmt(scores["val"], "val"))
        for epoch in (0, self.settings.epochs):
            self.tracker.log_eval(epoch, scores["val"], curves={k: scores[k] for k in ("train", "test")})
        self.tracker.log_reference(val=scores["val"], test=scores["test"])
        self.tracker.log_best(0, scores["val"])
        self._save_best(0, scores["val"], scores)
        self.tracker.log_final(scores, epoch=self.settings.epochs)
        self.tracker.finish()

    def _final(self):
        ck = self.best_checkpoint or torch.load(self.best_path, map_location="cpu", weights_only=False)
        self.net.load_state_dict(ck["model"])
        scores = self._score_all()
        ck.update(train_metrics=scores["train"], val_metrics=scores["val"], test_metrics=scores["test"])
        if self.best_path:
            save_atomic(ck, self.best_path)
        self.tracker.log_final(scores, epoch=self.settings.epochs)
        self.tracker.finish()

    def _save_best(self, epoch, val_metrics, scores=None):
        state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
        ck = dict(self.run, model=state, epoch=epoch, val_metrics=val_metrics)
        if self._loss_state():
            ck["loss_state"] = {k: v.detach().cpu().clone() for k, v in self._loss_state().items()}
        if scores is not None:
            ck.update(train_metrics=scores["train"], test_metrics=scores["test"])
        if self.best_path:
            save_atomic(ck, self.best_path)
        else:
            self.best_checkpoint = ck

    def _loss_state(self):
        return self.loss.state_dict() if self.loss is not None else {}

    def _save_last(self, epoch):
        if not self.last_path:
            return
        save_atomic(dict(self.run, model=self.net.state_dict(), opt=self.opt.state_dict(),
                         loss_state=self._loss_state(),
                         sched=self.sched.state_dict(), epoch=epoch, best=self.best,
                         bad_epochs=self.bad_epochs, tracker=self.tracker.state(), rng=rng_state(),
                         resume_settings=self.resume_settings()), self.last_path)
