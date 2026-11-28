"""The temporal latent transport (TT) follow-ups on PEMS08, one after another, with tqdm progress bars: an overall
bar over every epoch of every queued run and a bar for the batches of the current epoch, with the latest val
MAE. Made for a laptop (Apple MPS, CUDA or CPU: train.py picks the device); every run is resumable (Ctrl-C and
start it again: finished runs are skipped, the interrupted one continues from its .last.pt).

Every run is the best no-split model so far (distvolformer_direct + TT 5 steps + --adaptive-embed + --zscore +
--log-counts + --ext-input, STAEformer-sized; test MAE 14.91 at seed 42) with one change:

  tt1          1 Sinkhorn step: a softmax over the learned distance cost, no column constraint (is it the OT?)
  notopo       no learned lag topology (does the geometry over lags matter?)
  tt9          9 Sinkhorn steps (does stricter conservation help?)
  unbal        unbalanced OT: the 'every input frame sends 2' constraint is a learned KL penalty
  mass         learned source masses: each input frame's mass comes from its token
  sensortopo   one lag topology per sensor instead of one shared
  hybrid       temporal attention and TT side by side, outputs added
  periods      long history: + the 12 steps one day and one week before the targets (36 input frames)
  nodecay      no weight decay on TT's geometry (temperatures, topology positions / weight)
  flow         + the flow transport head: forecast = observed past flow moved by the last TT plan + a correction
  tt5          the reference itself, re-run on this machine (same-hardware comparison)
  attn         the attention control (no TT), as slurm/pems_ae_attention.sbatch runs on Sol

  python tools/run_tt_ablations.py                      # the default queue: tt1 notopo unbal mass tt9 sensortopo hybrid periods
  python tools/run_tt_ablations.py tt1 notopo           # just these, in this order
  python tools/run_tt_ablations.py --list               # the runs, their flags and whether they are done
  python tools/run_tt_ablations.py --seeds 42,7,123 tt1
  python tools/run_tt_ablations.py --extra "--no-vol-encoder --epochs 100" --tag _novol_e100 tt1

The PEMS08 data is downloaded on the first call if it isn't there (tools/download_pems.py, checked against
the reference hashes); after that every run reads the local .npy files.

Logs: checkpoints/workspaces/pems_pems08/logs/<run>.log (train.py's full output). When the queue is through, the
finished runs are scored (masked MAE / RMSE / MAPE, the PeMS papers' protocol) into logs/pems_tt_ablations_eval.out.
"""
import argparse
import os
import re
import shlex
import signal
import subprocess
import sys
import time

from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = os.path.join("checkpoints", "workspaces", "pems_pems08")

BASE = ("--model distvolformer_direct --dataset pems08 --t-in 12 --t-out 12 --loss sinkhorn --l2 0 --time-tokens "
        "--patch-size 1 --external --ext-input --adaptive-embed --zscore --log-counts --grad-checkpoint "
        "--eval-batch-size 16 --dim 152 --num-blocks 6 --heads 4 --mlp-dim 256 --width 16")
RUNS = {   # name -> (the change, the flags it adds to BASE)
    "tt1": ("1 Sinkhorn step", "--time-transport 1"),
    "notopo": ("no lag topology", "--time-transport 5 --tt-no-topology"),
    "unbal": ("unbalanced OT (rho 1)", "--time-transport 5 --tt-unbalanced 1.0"),
    "mass": ("learned source masses", "--time-transport 5 --tt-learned-mass"),
    "tt9": ("9 Sinkhorn steps", "--time-transport 9"),
    "sensortopo": ("a lag topology per sensor", "--time-transport 5 --tt-sensor-topology"),
    "hybrid": ("attention + TT", "--time-transport 5 --tt-hybrid"),
    "periods": ("+ day / week history", "--time-transport 5 --periods day week"),
    "nodecay": ("no weight decay on TT geometry", "--time-transport 5 --tt-no-decay"),
    "flow": ("flow transport head (+ no decay)", "--time-transport 5 --tt-no-decay --tt-flow-head"),
    "condcost": ("calendar-conditioned cost (+ flow head, no decay)",
                 "--time-transport 5 --tt-no-decay --tt-flow-head --tt-cond-cost"),
    "huber": ("Huber loss, delta 1 vehicle, instead of L1 + volume (+ flow head, no decay)",
              "--time-transport 5 --tt-no-decay --tt-flow-head --loss huber_raw"),
    "st1": ("space-time transport, 1 road hop (+ flow head, no decay)",
            "--time-transport 5 --tt-no-decay --tt-flow-head --st-transport 1"),
    "st1only": ("space-time transport in every block, no spatial attention (+ flow head, no decay)",
                "--time-transport 5 --tt-no-decay --tt-flow-head --st-transport 1 --st-only"),
    "mixed": ("STAEformer's mixed-projection output head, no target frames", "--time-transport 5 --mixed-head"),
    "chain": ("chained transport: later temporal blocks also draw on earlier future frames", "--time-transport 5 --tt-chain"),
    "staeembed": ("STAEformer's token embedding (flow + calendar, concatenated adaptive part), no --adaptive-embed",
                  "--time-transport 5 --stae-embed"),
    # STAEformer's recipe: Huber (delta 1 vehicle), Adam with coupled decay 1.5e-3 on raw-count gradients (our loss
    # is in z units, PEMS08 train std 145.6 vehicles: 1.5e-3 / 145.6 = 1.03e-5, the same relative strength), lr cut
    # x0.1 at epochs 25 / 45 / 65, no clipping, up to 300 epochs, early stopping after 30, picked by val MAE
    "recipe": ("STAEformer's training recipe on the reference",
               "--time-transport 5 --loss huber_raw --optimizer adam --weight-decay 1.03e-5 --schedule multistep "
               "--milestones 25 45 65 --lr-gamma 0.1 --clip 0 --epochs 300 --patience 30 --select mae"),
    "tt5": ("the reference", "--time-transport 5"),
    "attn": ("attention control, no TT", ""),
}
DEFAULT = ["tt1", "notopo", "unbal", "mass", "tt9", "sensortopo", "hybrid", "periods"]
EPOCHS = 50   # distvolformer_direct's default; --epochs in --extra overrides it

EPOCH_LINE = re.compile(r"^ep\s+(\d+) \|.*?MAE ([\d.]+) RMSE ([\d.]+)", re.M)
BATCH_BAR = re.compile(r"epoch (\d+):\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)")


NAME = {"tt5": "tt5ref"}   # keep the rerun apart from the original tt5 checkpoint


def run_name(key, tag, seed):
    name = f"distvolformer_direct_small_extin_{NAME.get(key, key)}_ae_z{tag}_pems08_in12_out12"
    return name if seed == 42 else f"{name}_seed{seed}"


def epochs_of(flags):
    m = re.search(r"--epochs\s+(\d+)", flags)
    return int(m.group(1)) if m else EPOCHS


def tail(path, size=200_000):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - size))
            return f.read().decode("utf-8", "replace")
    except FileNotFoundError:
        return ""


def epochs_done(log):
    return [(int(e), float(mae), float(rmse)) for e, mae, rmse in EPOCH_LINE.findall(log.replace("\r", "\n"))]


def train(key, flags, seed, tag, wandb, overall, done_before):
    name = run_name(key, tag, seed)
    log_path = os.path.join(W, "logs", name + ".log")
    base = BASE.replace("--loss sinkhorn --l2 0 ", "") if "--loss" in shlex.split(flags) else BASE   # a loss of its own
    if "--stae-embed" in shlex.split(flags):   # it has its own adaptive embedding
        base = base.replace("--adaptive-embed ", "")
    cmd = [sys.executable, "-u", "train.py", *shlex.split(base), *shlex.split(flags), "--seed", str(seed),
           "--save", os.path.join(W, name + ".pt"), "--resume"] + ([] if wandb else ["--no-wandb"])
    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1")   # Apple GPUs: run the odd unsupported op on the CPU
    total = epochs_of(flags)
    with open(log_path, "a") as log:
        log.write(f"\n=== {time.strftime('%F %T')} {shlex.join(cmd)}\n")
        log.flush()
        pos = log.tell()   # read only this attempt's output, incrementally
        child = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    batches = tqdm(total=1, desc="  epoch", position=1, leave=False, unit="batch", dynamic_ncols=True)
    best, pending = None, ""
    try:
        while True:
            running = child.poll() is None
            with open(log_path, "rb") as f:
                f.seek(pos)
                chunk = f.read().decode("utf-8", "replace")
                pos = f.tell()
            text = (pending + chunk).replace("\r", "\n")
            cut = text.rfind("\n") + 1
            lines, pending = text[:cut], text[cut:]   # keep a partial last line for the next read
            for e, mae, rmse in epochs_done(lines):
                counted = done_before.get(name, 0)
                if e > counted:
                    overall.update(e - counted)
                    done_before[name] = e
                best = mae if best is None else min(best, mae)
                overall.set_postfix_str(f"{key} seed {seed}: ep {e}/{total} val MAE {mae:.2f} (best {best:.2f})")
            if not running:
                break
            time.sleep(2)
            bars = BATCH_BAR.findall(text[-5000:])
            if bars:
                ep, n, of = map(int, bars[-1])
                if batches.total != of or ep != getattr(batches, "epoch", None):   # a new epoch: restart the clock
                    batches.reset(total=of)
                    batches.epoch = ep
                    batches.set_description(f"  {key} epoch {ep}")
                batches.n = n
                batches.refresh()
    except KeyboardInterrupt:
        os.killpg(child.pid, signal.SIGINT)
        child.wait()
        raise
    finally:
        batches.close()
    if child.returncode == 0:
        open(os.path.join(W, name + ".done"), "w").close()
        tqdm.write(f"done {name}" + (f": best val MAE {best:.2f}" if best is not None else ""))
        return True
    tqdm.write(f"!!! {name} failed (exit {child.returncode}); see {log_path}")
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("runs", nargs="*", help=f"from: {' '.join(RUNS)} (default: {' '.join(DEFAULT)})")
    ap.add_argument("--seeds", type=lambda v: [int(x) for x in v.split(",")], default=[42],
                    help="comma-separated, e.g. 42,7,123 (default 42)")
    ap.add_argument("--extra", default="", help="flags added to every run, e.g. \"--no-vol-encoder --epochs 100\"")
    ap.add_argument("--tag", default="", help="added to the run names (use it with --extra), e.g. _novol_e100")
    ap.add_argument("--wandb", action="store_true", help="log to W&B (off by default)")
    ap.add_argument("--list", action="store_true", help="show the runs and stop")
    ap.add_argument("--workspace", default=W, help=f"where checkpoints and logs go (default {W})")
    args = ap.parse_args()
    globals()["W"] = args.workspace   # train() and run_name() read the module's W
    os.chdir(REPO)
    keys = args.runs or DEFAULT
    unknown = [k for k in keys if k not in RUNS]
    if unknown:
        ap.error(f"unknown runs {unknown}; choose from {' '.join(RUNS)}")
    if args.extra and not args.tag:
        ap.error("--extra changes the runs: give them a --tag so they don't share checkpoints with the plain ones")
    data = os.path.join("datasets", "PEMS", "PEMS08")
    if all(os.path.exists(os.path.join(data, f)) for f in ("flow.npy", "hops.npy")):
        tqdm.write(f"data: {data} present, not downloading")
    else:   # once: train.py and the scoring only ever read these local files
        tqdm.write(f"data: {data} missing, downloading it once")
        sys.stdout.flush()
        if subprocess.run([sys.executable, os.path.join("tools", "download_pems.py"), "PEMS08"]).returncode:
            raise SystemExit("couldn't download PEMS08; see the message above")
    os.makedirs(os.path.join(W, "logs"), exist_ok=True)

    queue = [(k, f"{RUNS[k][1]} {args.extra}".strip(), s) for s in args.seeds for k in keys]
    if args.list:
        for k, flags, s in queue:
            name = run_name(k, args.tag, s)
            state = "done" if os.path.exists(os.path.join(W, name + ".done")) else "to do"
            print(f"{k:11s} seed {s:<4d} {state:6s} {RUNS[k][0]:28s} {flags}")
        return
    todo = [(k, f, s) for k, f, s in queue if not os.path.exists(os.path.join(W, run_name(k, args.tag, s) + ".done"))]
    total = sum(epochs_of(f) for _, f, _ in todo)
    done_before = {}
    for k, f, s in todo:   # resumed runs: count the epochs already in their .last.pt's log
        name = run_name(k, args.tag, s)
        if os.path.exists(os.path.join(W, name + ".last.pt")):
            eps = epochs_done(tail(os.path.join(W, "logs", name + ".log"), 10_000_000))
            done_before[name] = max((e for e, _, _ in eps), default=0)
    tqdm.write(f"{len(todo)} of {len(queue)} runs to train ({len(queue) - len(todo)} done); logs in {W}/logs/")
    overall = tqdm(total=total, initial=sum(done_before.values()), desc="TT ablations", position=0, unit="ep",
                   dynamic_ncols=True)
    failed = []
    try:
        for k, f, s in todo:
            tqdm.write(f"train {run_name(k, args.tag, s)}  ({RUNS[k][0]}: {f})")
            if not train(k, f, s, args.tag, args.wandb, overall, done_before):
                failed.append(run_name(k, args.tag, s))
    except KeyboardInterrupt:
        overall.close()
        print("\nstopped; run the same command again to continue where it left off")
        return
    overall.close()

    out = os.path.join("logs", "pems_tt_ablations_eval.out")
    os.makedirs("logs", exist_ok=True)
    pattern = "_ae_z" + re.escape(args.tag) + "_pems08"
    print(f"scoring into {out}")
    res = subprocess.run([sys.executable, os.path.join("tools", "eval_pems.py"), W, "--match", pattern],
                         capture_output=True, text=True)
    with open(out, "w") as f:
        f.write(res.stdout)
    print(res.stdout or res.stderr[-2000:])
    if failed:
        raise SystemExit(f"{len(failed)} runs failed: {' '.join(failed)}")


if __name__ == "__main__":
    main()
