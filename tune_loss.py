"""Search the DistVol loss weights with Optuna, then confirm the best ones over seeds.

Every trial trains with the sinkhorn_kl loss,

    1 * L1  +  l2 * Sinkhorn  +  l3 * volume  +  l4 * KL(p || p_hat),

and the sampler picks l2 / l3 / l4 / eps (Sinkhorn's blur). Sinkhorn and KL can each be switched
off (weight 0), so the search also covers the sinkhorn, kl and plain L1 + volume losses. l1 stays
at 1: with Adam only the ratios between the weights matter, apart from gradient clipping.

A trial is a shortened train.py run (--epochs, default 20, with the one-cycle schedule squeezed to
fit) on one seed, scored by its best val RMSE; test is never looked at. Trials that are behind the
median of the earlier ones at the same epoch are stopped early (pruned). The study lives in one
journal file, so several workers (e.g. a Slurm array) can share it, and rerunning the same command
continues it: trials a killed worker left behind are failed and run again.

Shortened single-seed runs are only a screen. --confirm K writes train_tuned_<study>.sh, a
train_all.sh wrapper (like train_all_new_seeds.sh) that runs the K best weightings and the default
sinkhorn loss at full length on seeds 42 / 7 / 123 / 2025.

examples:
  python tune_loss.py --trials 40                                  # distvolformer on TaxiBJ
  python tune_loss.py --study dvf_l --trials 40 -- --model distvolformer --dim 256 --heads 8
  python tune_loss.py --fix l4=0 --study no_kl --trials 30         # Sinkhorn family only
  python tune_loss.py --report                                     # best trials, importances
  python tune_loss.py --confirm 3                                  # -> train_tuned_<study>.sh
  sbatch --array=0-3 --gres=gpu:1 --time=08:00:00 --mem=32G --cpus-per-task=4 \\
         --wrap="python tune_loss.py --trials 10"                  # 4 workers, one study
"""
import argparse
import gc
import math
import os
import socket
import stat
import time
import warnings

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState

ROOT = os.path.dirname(os.path.abspath(__file__))
TUNING_DIR = os.path.join(ROOT, "checkpoints", "tuning")
LOSS = "sinkhorn_kl"
OBJECTIVE = "rmse_flow"
DEFAULTS = dict(l1=1.0, l2=10.0, l3=1.0, l4=0.1, eps=0.01)   # the sinkhorn_kl defaults, trial 0
RANGES = dict(l2=(0.3, 1000.0), l3=(0.03, 30.0), l4=(0.003, 3.0), eps=(0.003, 0.1))
SWITCHABLE = dict(l2="use_sinkhorn", l4="use_kl")
SEEDS = (42, 7, 123, 2025)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", help="default: <model>_<dataset>, e.g. distvolformer_taxibj")
    ap.add_argument("--trials", type=int, default=20, help="trials this worker runs (default 20)")
    ap.add_argument("--epochs", type=int, default=20, help="epochs per trial (default 20)")
    ap.add_argument("--seed", type=int, default=42, help="training seed of every trial (default 42)")
    ap.add_argument("--fix", action="append", default=[], metavar="KEY=VALUE",
                    help="hold a weight fixed instead of searching it (l1, l2, l3, l4, eps; repeatable)")
    ap.add_argument("--no-prune", action="store_true", help="run every trial to the end")
    ap.add_argument("--stale-minutes", type=float, default=30.0,
                    help="a running trial with no progress for this long is from a killed worker (default 30)")
    ap.add_argument("--report", action="store_true", help="print the study's best trials and exit")
    ap.add_argument("--confirm", type=int, metavar="K",
                    help="write train_tuned_<study>.sh with the K best trials and exit")
    ap.add_argument("train_args", nargs="*",
                    help="after --: train.py arguments for every trial (default --model distvolformer)")
    args = ap.parse_args()
    args.fixed = {}
    for item in args.fix:
        key, _, value = item.partition("=")
        if key not in DEFAULTS or not value:
            raise SystemExit(f"--fix {item}: expected KEY=VALUE with KEY one of {', '.join(DEFAULTS)}")
        args.fixed[key] = float(value)
    for flag in ("--loss", "--l1", "--l2", "--l3", "--l4", "--eps", "--epochs", "--seed", "--save", "--resume"):
        if flag in args.train_args:
            raise SystemExit(f"{flag} is set by tune_loss.py; use its own options instead")
    if "--model" not in args.train_args:
        args.train_args = ["--model", "distvolformer", *args.train_args]
    if args.study is None:
        model = args.train_args[args.train_args.index("--model") + 1]
        dataset = args.train_args[args.train_args.index("--dataset") + 1] if "--dataset" in args.train_args else "taxibj"
        args.study = f"{model}_{dataset}"
    return args


def open_study(args):
    os.makedirs(TUNING_DIR, exist_ok=True)
    path = os.path.join(TUNING_DIR, f"{args.study}.journal")
    storage = JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path + ".lock")))
    sampler = optuna.samplers.TPESampler(multivariate=True, group=True, constant_liar=True,
                                         n_startup_trials=10, seed=None)
    pruner = (optuna.pruners.NopPruner() if args.no_prune else
              optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=max(1, args.epochs // 3)))
    study = optuna.create_study(study_name=args.study, storage=storage, direction="minimize",
                                sampler=sampler, pruner=pruner, load_if_exists=True)
    setup = dict(train_args=args.train_args, epochs=args.epochs, seed=args.seed, fixed=args.fixed)
    known = study.user_attrs.get("setup")
    if known is None:
        study.set_user_attr("setup", setup)
        study.enqueue_trial(sample_params(DEFAULTS, args.fixed), user_attrs={"note": "default weights"})
    elif known != setup and not (args.report or args.confirm):
        raise SystemExit(f"study {args.study} was started with {known}; this run asks for {setup}. "
                         f"Use the same options, or a new --study name")
    return study, path


def sample_params(values, fixed):
    """Values as Optuna params (the inverse of weights_from)."""
    params = {}
    for key in ("l2", "l3", "l4", "eps"):
        if key in fixed:
            continue
        if key in SWITCHABLE:
            params[SWITCHABLE[key]] = values[key] > 0
            if values[key] > 0:
                params[key] = values[key]
        elif key != "eps" or values["l2"] > 0:
            params[key] = values[key]
    return params


def suggest(trial, fixed):
    w = {"l1": 1.0, **DEFAULTS, **fixed}
    for key in ("l2", "l3", "l4"):
        if key in fixed:
            continue
        if key in SWITCHABLE and not trial.suggest_categorical(SWITCHABLE[key], [True, False]):
            w[key] = 0.0
        else:
            w[key] = trial.suggest_float(key, *RANGES[key], log=True)
    if "eps" not in fixed and w["l2"] > 0:
        w["eps"] = trial.suggest_float("eps", *RANGES["eps"], log=True)
    return w


def weights_from(trial, fixed):
    w = {**DEFAULTS, **fixed}
    p = trial.params
    for key in ("l2", "l3", "l4", "eps"):
        if key in fixed:
            continue
        if key in SWITCHABLE and not p.get(SWITCHABLE[key], True):
            w[key] = 0.0
        elif key in p:
            w[key] = p[key]
    return w


def weight_args(w):
    out = []
    for key in ("l1", "l2", "l3", "l4"):
        out += [f"--{key}", f"{w[key]:.4g}"]
    if w["l2"] > 0:
        out += ["--eps", f"{w['eps']:.4g}"]
    return out


def describe(w):
    parts = [f"{w['l2']:.3g} Sinkhorn (eps {w['eps']:.3g})" if w["l2"] > 0 else None,
             f"{w['l3']:.3g} volume", f"{w['l4']:.3g} KL" if w["l4"] > 0 else None]
    return f"{w['l1']:.3g} L1 + " + " + ".join(p for p in parts if p)


def fail_stale(study, args):
    """Fail trials a killed worker left running, and queue their weights again."""
    now = time.time()
    for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        beat = t.user_attrs.get("heartbeat", t.datetime_start.timestamp())
        if now - beat > 60 * args.stale_minutes:
            print(f"trial {t.number} hasn't progressed for {(now - beat) / 60:.0f} min "
                  f"({t.user_attrs.get('worker', '?')}); failing it and queueing its weights again")
            study._storage.set_trial_state_values(t._trial_id, TrialState.FAIL)
            study.enqueue_trial(t.params, user_attrs={"note": f"rerun of trial {t.number}"})


def objective(args):
    import torch
    import wandb

    import train

    def run(trial):
        w = suggest(trial, args.fixed)
        worker = f"{socket.gethostname()}:{os.getpid()}" + (f" job {os.environ['SLURM_JOB_ID']}"
                                                          if "SLURM_JOB_ID" in os.environ else "")
        trial.set_user_attr("worker", worker)
        trial.set_user_attr("heartbeat", time.time())
        os.environ["WANDB_RUN_GROUP"] = f"tune_{args.study}"
        os.environ["WANDB_NAME"] = f"{args.study}_t{trial.number}"
        print(f"\n=== {args.study} trial {trial.number}: {describe(w)}", flush=True)
        best = [math.inf]

        def after_epoch(epoch, val):
            best[0] = min(best[0], val[OBJECTIVE])
            trial.set_user_attr("heartbeat", time.time())
            if not math.isfinite(val[OBJECTIVE]):
                raise FloatingPointError(f"val {OBJECTIVE} is {val[OBJECTIVE]} at epoch {epoch}")
            trial.report(best[0], epoch)
            return trial.should_prune()

        argv = [*args.train_args, "--loss", LOSS, *weight_args(w), "--epochs", str(args.epochs),
                "--seed", str(args.seed), "--no-save", "--curves", "val"]
        try:
            scores = train.main(argv, epoch_callback=after_epoch)
        finally:
            if wandb.run is not None:   # a trial that crashed left its run open; don't log into it
                wandb.finish(exit_code=1)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if scores is None:
            raise optuna.TrialPruned()
        for key in ("rmse_flow", "mae_flow"):
            trial.set_user_attr(f"val_{key}", scores["val"][key])
        return scores["val"][OBJECTIVE]

    return run


def finished(study):
    return sorted((t for t in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))),
                  key=lambda t: t.value)


def report(study, args, path):
    trials = study.get_trials(deepcopy=False)
    counts = {s.name.lower(): sum(t.state == s for t in trials) for s in TrialState}
    print(f"study {args.study} ({path}): " + ", ".join(f"{n} {k}" for k, n in counts.items() if n))
    setup = study.user_attrs.get("setup", {})
    print(f"each trial: train.py {' '.join(setup.get('train_args', []))} --loss {LOSS}, "
          f"{setup.get('epochs')} epochs, seed {setup.get('seed')}; objective: best val RMSE")
    done = finished(study)
    if not done:
        print("no finished trials yet")
        return
    fixed = setup.get("fixed", {})
    default = next((t for t in done if t.user_attrs.get("note") == "default weights"), None)
    print(f"\n{'trial':>5}  {'val RMSE':>8}  {'val MAE':>7}  weights")
    for t in done[:15]:
        mark = "  (default)" if t is default else ""
        print(f"{t.number:5d}  {t.value:8.3f}  {t.user_attrs.get('val_mae_flow', float('nan')):7.3f}  "
              f"{describe(weights_from(t, fixed))}{mark}")
    if default is not None and default not in done[:15]:
        print(f"{default.number:5d}  {default.value:8.3f}  (the default weights)")
    if len(done) >= 10:
        try:
            importances = optuna.importance.get_param_importances(study)
            print("\nparameter importance (fANOVA): "
                  + ", ".join(f"{k} {v:.2f}" for k, v in importances.items()))
        except Exception as e:
            print(f"\n(no parameter importances: {e})")
    best = weights_from(done[0], fixed)
    print(f"\nbest: python train.py {' '.join(setup.get('train_args', []))} --loss {LOSS} "
          f"{' '.join(weight_args(best))}")
    print("these are single-seed shortened runs (on TaxiBJ, reruns of one config differ by ~0.1-0.3 RMSE); "
          "confirm the best with --confirm K")


def write_confirm(study, args, k):
    done = finished(study)[:k]
    if not done:
        raise SystemExit("no finished trials to confirm yet")
    setup = study.user_attrs["setup"]
    train_args = setup["train_args"]
    model = train_args[train_args.index("--model") + 1]
    i = train_args.index("--model")
    extra = train_args[:i] + train_args[i + 2:]
    tail = (" " + " ".join(extra)) if extra else ""   # train_all.sh splits RUNS lines on spaces
    lines = [f'    "{model}_sinkhorn {model} sinkhorn{tail}"']
    for rank, t in enumerate(done, 1):
        w = weights_from(t, setup.get("fixed", {}))
        lines.append(f'    "{model}_tuned{rank}_t{t.number} {model} {LOSS} {" ".join(weight_args(w))}{tail}"'
                     f"   # val RMSE {t.value:.3f} at {setup['epochs']} epochs: {describe(w)}")
    out = os.path.join(ROOT, f"train_tuned_{args.study}.sh")
    with open(out, "w") as f:
        f.write(f"""#!/usr/bin/env bash
# Written by: python tune_loss.py --study {args.study} --confirm {k}
# The {k} best loss weightings of Optuna study "{args.study}" next to the default sinkhorn loss, at
# full length on seeds {' '.join(map(str, SEEDS))}, with train_all.sh's machinery (same options; rerun to
# continue workspace "tuned_{args.study}"). Compare mean +- std over seeds before trusting a winner.

SCRIPT_NAME=./train_tuned_{args.study}.sh
DEFAULT_WORKSPACE=tuned_{args.study}
SEEDS=({' '.join(map(str, SEEDS))})

RUNS=(
{chr(10).join(lines)}
)

source "$(dirname "${{BASH_SOURCE[0]}}")/train_all.sh" "$@"
""")
    os.chmod(out, os.stat(out).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"wrote {out}:")
    print("\n".join(lines))


def main():
    args = parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    study, path = open_study(args)
    if args.report:
        return report(study, args, path)
    if args.confirm:
        return write_confirm(study, args, args.confirm)
    os.chdir(ROOT)
    fail_stale(study, args)
    study.optimize(objective(args), n_trials=args.trials, catch=(FloatingPointError, RuntimeError))
    report(study, args, path)


if __name__ == "__main__":
    main()
