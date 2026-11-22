"""Tune every model the same way with Optuna (equal budget), then confirm each model's best over seeds.

One study per model and dataset (<model>_<dataset>_in<T>). Every model gets the same budget:
the same number of trials, the same epochs per trial, the same pruning, the same objective (val
RMSE on cells whose true flow is >= 10, the STDN protocol we report, at the val-selected checkpoint;
--objective all: over every cell, the first round's; test is never looked at) and these shared choices:

  lr            log-uniform 1e-4 .. 5e-3
  weight decay  log-uniform 1e-5 .. 5e-2
  batch size    8 / 16 / 32 (stdcformer 16 / 32 / 64)
  loss          mse / mae / huber (every model), plus l1vol (L1 + volume) for the models that
                predict shape and volume (cnn, distvol, distvolformer, stresnet)
  periods       none / day week / shifted (1/2/3 days + week back, each +-1 step)

plus each model's own size / regularization knobs (MODEL_SPACES below). Trial 0 is the model's
current default configuration, so a study can only match or improve on it. Every model keeps its
own optimizer and schedule (train_defaults); trials squeeze them into --epochs.

Shortened single-seed trials are a screen. --confirm writes train_tuned_models_<dataset>_in<T>.sh:
every model's best trial at the model's full default length on seeds 42 / 7 / 123, with
train_all.sh's machinery. Report from those runs, not from the trials.

examples:
  python tune_models.py stresnet -- --dataset nyctaxi --t-in 8 --t-out 1 --until 30
  python tune_models.py --report stresnet -- --dataset nyctaxi --t-in 8 --t-out 1
  python tune_models.py --confirm stresnet distvolformer -- --dataset nyctaxi --t-in 8 --t-out 1
"""
import argparse
import gc
import math
import os
import signal
import stat
import sys
import time
import warnings

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState

from tune_loss import finished_count, interrupt, stop_at, worker_name

ROOT = os.path.dirname(os.path.abspath(__file__))
TUNING_DIR = os.path.join(ROOT, "checkpoints", "tuning", "models")
OBJECTIVES = {"ge10": "rmse_flow_ge10", "all": "rmse_flow"}   # --objective -> val metric
SEEDS = (42, 7, 123)
WANDB_PREFIX = "model_tuning"
WANDB_PROJECT = "traffic-pred-hyperparameter-tuning"
PARTS_MODELS = ("cnn", "distvol", "distvolformer", "distvolformer_direct", "distvolformer_graph", "stresnet")
PERIODS = {"none": [], "dayweek": ["--periods", "day", "week"],
           "shifted": ["--periods", "1day", "2day", "3day", "week", "--period-shift", "1"]}
LOSSES = {"mse": ["--loss", "mse"], "mae": ["--loss", "mae"], "huber": ["--loss", "huber"],
          "l1vol": ["--loss", "sinkhorn", "--l2", "0"]}

# Each model's own knobs: name -> (kind, choices or range, train.py flag). "cat" = categorical,
# "float" = uniform float. DEFAULTS are the current configurations (trial 0).
MODEL_SPACES = {
    "cnn": {"width": ("cat", [32, 64, 96], "--width"), "blocks": ("cat", [4, 6, 8], "--blocks")},
    "distvol": {"width": ("cat", [32, 64, 96], "--width")},
    "distvolformer": {"dim": ("cat", [128, 192, 256], "--dim"), "depth": ("cat", [4, 6, 8], "--num-blocks"),
                      "drop": ("float", (0.0, 0.3), "--drop"), "patch": ("cat", [1, 2], "--patch-size"),
                      "width": ("cat", [32, 64], "--width")},
    "distvolformer_direct": {"dim": ("cat", [128, 192, 256], "--dim"), "depth": ("cat", [4, 6, 8], "--num-blocks"),
                             "drop": ("float", (0.0, 0.3), "--drop"), "patch": ("cat", [1, 2], "--patch-size"),
                             "width": ("cat", [32, 64], "--width")},   # distvolformer's space: the ablation
    "distvolformer_graph": {"dim": ("cat", [128, 192, 256], "--dim"), "depth": ("cat", [4, 6, 8], "--num-blocks"),
                            "drop": ("float", (0.0, 0.3), "--drop"), "width": ("cat", [32, 64], "--width"),
                            "vol_dim": ("cat", [64, 128, 192], "--vol-dim"),
                            "vol_depth": ("cat", [2, 3, 4], "--vol-depth"),
                            "radius": ("cat", [0, 2, 4], "--graph-radius")},
    "stresnet": {"width": ("cat", [32, 64, 96], "--width"), "units": ("cat", [4, 8, 12], "--st-units"),
                 "bn": ("cat", [False, True], "--st-bn")},
    "mlp": {"hidden": ("cat", [256, 512, 1024], "--hidden"), "depth": ("cat", [2, 3, 4], "--depth"),
            "dropout": ("float", (0.0, 0.3), "--dropout")},
    "predformer": {"dim": ("cat", [128, 256], "--dim"), "blocks": ("cat", [4, 8], "--num-blocks"),
                   "patch": ("cat", [2, 5], "--patch-size"), "drop": ("float", (0.0, 0.3), "--drop")},
    "predformer_official": {},
    "stdcformer": {"d_model": ("cat", [32, 64], "--d-model"), "layers": ("cat", [3, 5], "--layers")},
}
FIXED_ARGS = {"distvolformer": ["--time-tokens"], "distvolformer_direct": ["--time-tokens"]}   # the frozen recipe; heads = dim / 32 (below)
DEFAULTS = {
    "cnn": dict(width=64, blocks=6, loss="mse"),
    "distvol": dict(width=64, loss="l1vol"),
    "distvolformer": dict(dim=192, depth=6, drop=0.1, patch=2, width=64, loss="huber"),
    "distvolformer_direct": dict(dim=192, depth=6, drop=0.1, patch=2, width=64, loss="huber"),
    "distvolformer_graph": dict(dim=192, depth=6, drop=0.1, width=32, vol_dim=128, vol_depth=3, radius=0,
                                loss="huber"),
    "stresnet": dict(width=64, units=12, bn=False, loss="mse"),
    "mlp": dict(hidden=512, depth=3, dropout=0.1, loss="mse"),
    "predformer": dict(dim=256, blocks=8, patch=2, drop=0.1, loss="mse"),
    "predformer_official": dict(loss="mse"),
    "stdcformer": dict(d_model=64, layers=5, loss="mae"),
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", choices=list(MODEL_SPACES))
    ap.add_argument("--trials", type=int, help="trials this worker runs per model (default: no cap with --until)")
    ap.add_argument("--until", type=int, default=30, metavar="N",
                    help="stop a study once it has N finished (complete or pruned) trials (default 30)")
    ap.add_argument("--epochs", type=int, default=20, help="epochs per trial (default 20)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--objective", choices=list(OBJECTIVES), default="ge10",
                    help="ge10 (default): val RMSE on cells with true flow >= 10, in <model>_<data>_ge10 "
                         "studies; all: every cell, the first round's studies")
    ap.add_argument("--no-prune", action="store_true")
    ap.add_argument("--report", action="store_true", help="print each study's best trials and exit")
    ap.add_argument("--confirm", action="store_true",
                    help="write train_tuned_models_<dataset>_in<T>.sh with every model's best trial and exit")
    ap.add_argument("--wandb-project", default=WANDB_PROJECT)
    ap.add_argument("--study-suffix", default="",
                    help="appended to the study / confirm-script names, to tune a variant (e.g. _odhead for "
                         "-- --od --od-head) apart from the model's main study")
    ap.add_argument("--stale-minutes", type=float, default=30.0)
    ap.add_argument("train_args", nargs="*", help="after --: train.py arguments for every trial "
                                                  "(the data: --dataset, --t-in, --t-out)")
    argv = sys.argv[1:]
    cut = argv.index("--") if "--" in argv else len(argv)
    args = ap.parse_args(argv[:cut])
    args.train_args = argv[cut + 1:]
    for flag in ("--model", "--loss", "--lr", "--weight-decay", "--batch-size", "--periods", "--epochs",
                 "--seed", "--save", "--resume"):
        if flag in args.train_args:
            raise SystemExit(f"{flag} is set by tune_models.py")
    get = lambda f, d: args.train_args[args.train_args.index(f) + 1] if f in args.train_args else d
    args.dataset, args.t_in = get("--dataset", "taxibj"), get("--t-in", None)
    args.tag = f"{args.dataset}" + (f"_in{args.t_in}" if args.t_in else "")
    args.tag += "" if args.objective == "all" else f"_{args.objective}"
    args.tag += args.study_suffix
    args.metric = OBJECTIVES[args.objective]
    return args


def train_defaults(model, dataset):
    from trafficpred.models import MODELS
    return MODELS[model].train_defaults(dataset)


def loss_choices(model):
    return [*(["l1vol"] if model in PARTS_MODELS else []), "mse", "mae", "huber"]


def batch_choices(model):
    return [16, 32, 64] if model == "stdcformer" else [8, 16, 32]


def default_params(model, dataset):
    t = train_defaults(model, dataset)
    loss = DEFAULTS[model]["loss"]
    p = {k: v for k, v in DEFAULTS[model].items() if k != "loss"}
    if loss not in loss_choices(model):   # stdcformer's masked_mae -> mae, the nearest shared loss
        loss = "mae"
    return dict(p, lr=t["lr"], weight_decay=t["weight_decay"], batch_size=t["batch_size"], loss=loss,
                periods="dayweek")


def suggest(trial, model):
    p = {"lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
         "weight_decay": trial.suggest_float("weight_decay", 1e-5, 5e-2, log=True),
         "batch_size": trial.suggest_categorical("batch_size", batch_choices(model)),
         "loss": trial.suggest_categorical("loss", loss_choices(model)),
         "periods": trial.suggest_categorical("periods", list(PERIODS))}
    for name, (kind, space, _) in MODEL_SPACES[model].items():
        p[name] = (trial.suggest_categorical(name, space) if kind == "cat" else
                   trial.suggest_float(name, *space))
    return p


def train_argv(model, p):
    """train.py arguments for one configuration (without the data / epochs / seed)."""
    argv = ["--model", model, *LOSSES[p["loss"]], "--lr", f"{p['lr']:.3g}",
            "--weight-decay", f"{p['weight_decay']:.3g}", "--batch-size", str(p["batch_size"]),
            *PERIODS[p["periods"]], *FIXED_ARGS.get(model, [])]
    for name, (kind, _, flag) in MODEL_SPACES[model].items():
        value = p[name]
        if isinstance(value, bool):
            argv += [flag] if value else []
        else:
            argv += [flag, f"{value:.3g}" if kind == "float" else str(value)]
    if model in ("distvolformer", "distvolformer_direct", "distvolformer_graph"):
        argv += ["--heads", str(p["dim"] // 32)]
    return argv


def describe(p):
    return " ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in p.items())


def open_study(model, args):
    os.makedirs(TUNING_DIR, exist_ok=True)
    name = f"{model}_{args.tag}"
    path = os.path.join(TUNING_DIR, f"{name}.journal")
    storage = JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path + ".lock")))
    sampler = optuna.samplers.TPESampler(multivariate=True, group=True, constant_liar=True,
                                         n_startup_trials=10, seed=None)
    pruner = (optuna.pruners.NopPruner() if args.no_prune else
              optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=max(1, args.epochs // 3)))
    study = optuna.create_study(study_name=name, storage=storage, direction="minimize",
                                sampler=sampler, pruner=pruner, load_if_exists=True)
    setup = dict(train_args=args.train_args, epochs=args.epochs, seed=args.seed)
    known = study.user_attrs.get("setup")
    if known is None:
        study.set_user_attr("setup", setup)
        study.enqueue_trial(default_params(model, args.dataset), user_attrs={"note": "default"})
    elif known != setup and not (args.report or args.confirm):
        raise SystemExit(f"study {name} was started with {known}; this run asks for {setup}")
    return study, name


def fail_stale(study, args):
    now = time.time()
    for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        if now - t.user_attrs.get("heartbeat", t.datetime_start.timestamp()) > 60 * args.stale_minutes:
            print(f"trial {t.number} is from a killed worker; failing it and queueing it again")
            study._storage.set_trial_state_values(t._trial_id, TrialState.FAIL)
            study.enqueue_trial(t.params, user_attrs={"note": f"rerun of trial {t.number}"})


def objective(model, name, args):
    import torch
    import wandb

    import train

    def run(trial):
        p = suggest(trial, model)
        args.current_trial = trial.number
        trial.set_user_attr("worker", worker_name())
        trial.set_user_attr("heartbeat", time.time())
        os.environ.update(WANDB_PROJECT=args.wandb_project, WANDB_RUN_GROUP=f"{WANDB_PREFIX}_{name}",
                          WANDB_NAME=f"{WANDB_PREFIX}_{name}_t{trial.number}", WANDB_TAGS=WANDB_PREFIX)
        print(f"\n=== {name} trial {trial.number}: {describe(p)}", flush=True)
        best = [math.inf]

        def after_epoch(epoch, val):
            best[0] = min(best[0], val[args.metric])
            trial.set_user_attr("heartbeat", time.time())
            if not math.isfinite(val[args.metric]):
                raise FloatingPointError(f"val {args.metric} is {val[args.metric]} at epoch {epoch}")
            trial.report(best[0], epoch)
            return trial.should_prune()

        argv = [*train_argv(model, p), *args.train_args, "--epochs", str(args.epochs),
                "--seed", str(args.seed), "--no-save", "--curves", "val"]
        try:
            scores = train.main(argv, epoch_callback=after_epoch)
        finally:
            if wandb.run is not None:
                wandb.finish(exit_code=1)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        args.current_trial = None
        if scores is None:
            raise optuna.TrialPruned()
        trial.set_user_attr("val_mae_flow", scores["val"]["mae_flow"])
        trial.set_user_attr("val_rmse_flow", scores["val"]["rmse_flow"])
        return scores["val"][args.metric]

    return run


def best_trials(study):
    return sorted(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)), key=lambda t: t.value)


def report(study, name):
    trials = study.get_trials(deepcopy=False)
    counts = {s.name.lower(): sum(t.state == s for t in trials) for s in TrialState}
    print(f"\n## {name}: " + ", ".join(f"{n} {k}" for k, n in counts.items() if n))
    done = best_trials(study)
    default = next((t for t in done if t.user_attrs.get("note") == "default"), None)
    for t in done[:8]:
        print(f"  t{t.number:<3d} val RMSE {t.value:7.3f}  {describe(t.params)}"
              + ("  (default)" if t is default else ""))
    if default is not None and default not in done[:8]:
        print(f"  t{default.number:<3d} val RMSE {default.value:7.3f}  (default)")


DATA_FLAGS = {"--dataset": "--dataset", "--t-in": "--tin", "--t-out": "--tout"}   # train.py -> train_all.sh


def write_confirm(studies, args):
    data, extra, rest = [], [], list(args.train_args)
    while rest:
        flag = rest.pop(0)
        if flag in DATA_FLAGS and rest:
            data += [DATA_FLAGS[flag], rest.pop(0)]
        else:
            extra.append(flag)
    lines = []
    for model, (study, name) in studies.items():
        done = best_trials(study)
        if not done:
            print(f"{name}: no finished trials; left out")
            continue
        t = done[0]
        argv = train_argv(model, t.params)   # --model M --loss L [args]
        lines.append(f'    "{model}_tuned {model} {argv[3]} {" ".join(argv[4:] + extra)}"'
                     f"   # t{t.number}, val RMSE {t.value:.3f} at {args.epochs} epochs")
    out = os.path.join(ROOT, f"train_tuned_models_{args.tag}.sh")
    with open(out, "w") as f:
        f.write(f"""#!/usr/bin/env bash
# Written by: python tune_models.py --confirm {' '.join(studies)} -- {' '.join(args.train_args)}
# Every model's best tuned configuration (equal-budget Optuna studies, {args.epochs}-epoch trials, best
# val {args.metric}) at the model's full default length on seeds {' '.join(map(str, SEEDS))}, with train_all.sh's
# machinery (its options work too). Workspace tuned_models_{args.tag}. Score with
#   python tools/eval_published.py checkpoints/workspaces/tuned_models_{args.tag} --threshold 10 --split val

SCRIPT_NAME=./train_tuned_models_{args.tag}.sh
DEFAULT_WORKSPACE=tuned_models_{args.tag}
SEEDS=({' '.join(map(str, SEEDS))})
set -- {' '.join(data)} "$@"

RUNS=(
{chr(10).join(lines)}
)

source "$(dirname "${{BASH_SOURCE[0]}}")/train_all.sh" "$@"
""")
    os.chmod(out, os.stat(out).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"wrote {out}")
    print("\n".join(lines))


def main():
    args = parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    studies = {m: open_study(m, args) for m in args.models}
    if args.report:
        for study, name in studies.values():
            report(study, name)
        return
    if args.confirm:
        return write_confirm(studies, args)
    os.chdir(ROOT)
    signal.signal(signal.SIGTERM, interrupt)
    for model, (study, name) in studies.items():
        fail_stale(study, args)
        if finished_count(study) >= args.until:
            print(f"{name} already has {finished_count(study)} finished trials")
            continue
        args.current_trial = None
        try:
            study.optimize(objective(model, name, args), n_trials=args.trials,
                           catch=(Exception,), callbacks=[stop_at(args.until)])   # OOM, W&B hiccups: fail the trial, not the study
        except KeyboardInterrupt:
            if args.current_trial is not None:
                study.enqueue_trial(study.trials[args.current_trial].params,
                                    user_attrs={"note": f"rerun of interrupted trial {args.current_trial}"})
            raise SystemExit(130)
        report(study, name)


if __name__ == "__main__":
    main()
