"""How far along the training runs are, and when they'll finish: reads the epoch progress bar that train.py
writes into each run's log (epochs done, time per epoch, time left) and the latest / best val MAE.

  python tools/progress.py                 # runs in checkpoints/workspaces/pems_pems08 still training or done today
  python tools/progress.py ae_z            # only runs whose name contains "ae_z"
  python tools/progress.py --all           # every run in the workspace
  python tools/progress.py --workspace checkpoints/workspaces/taxibj_nextstep_in3

Runs that haven't started yet (in a queue) aren't known to this script; add their epochs x the per-epoch time.
"""
import argparse
import glob
import os
import re
import time

BAR = re.compile(r"(\d+)/(\d+) \[([\d:]+)<([\d:?]+),\s*([\d.]+)(s/ep|ep/s)")
EPOCH = re.compile(r"^ep\s+(\d+) \|.*?MAE ([\d.]+) RMSE ([\d.]+)", re.M)


def seconds(clock):
    if "?" in clock:
        return None
    parts = [int(p) for p in clock.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def clock(s):
    if s is None:
        return "?"
    h, m = divmod(int(s) // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"


def read_tail(path, size=400_000):
    with open(path, "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - size))
        return f.read().decode("utf-8", "replace").replace("\r", "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("match", nargs="?", default="", help="only runs whose name contains this")
    ap.add_argument("--workspace", default=os.path.join("checkpoints", "workspaces", "pems_pems08"))
    ap.add_argument("--all", action="store_true", help="also runs whose log hasn't changed in the last day")
    args = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ws = args.workspace if os.path.isabs(args.workspace) else os.path.join(root, args.workspace)
    logs = sorted(glob.glob(os.path.join(ws, "logs", "*.log")), key=os.path.getmtime)
    now = time.time()
    rows = []
    for path in logs:
        name = os.path.basename(path)[:-4]
        if args.match not in name:
            continue
        age = now - os.path.getmtime(path)
        done = os.path.exists(os.path.join(ws, name + ".done"))
        if not args.all and age > 86400:
            continue
        text = read_tail(path)
        bars = BAR.findall(text)
        epochs = EPOCH.findall(text)
        best = min((float(m) for _, m, _ in epochs), default=None)
        last = float(epochs[-1][1]) if epochs else None
        if done:
            state, left = "done", 0
        elif age > 1800:
            state, left = f"stopped? (log quiet for {clock(age)})", None
        else:
            state, left = "training", None
        if bars:
            n, total, elapsed, remain, rate, unit = bars[-1]
            n, total = int(n), int(total)
            per_ep = float(rate) if unit == "s/ep" else (1 / float(rate) if float(rate) else None)
            if left is None and state == "training":
                left = seconds(remain)
                if left is None and per_ep:
                    left = (total - n) * per_ep
            progress = f"epoch {n}/{total}"
            pace = f"{per_ep / 60:.1f} min/ep" if per_ep else "?"
        else:
            progress, pace = "starting", "?"
        val = f"val MAE {last:.2f} (best {best:.2f})" if last is not None else ""
        eta = "" if left in (None, 0) else f"~{clock(left)} left, done ~{time.strftime('%H:%M', time.localtime(now + left))}"
        rows.append((name, state, progress, pace, val, eta))
    if not rows:
        print(f"no runs{' matching ' + repr(args.match) if args.match else ''} in {ws}/logs"
              f"{'' if args.all else ' changed in the last day (try --all)'}")
        return
    width = max(len(r[0]) for r in rows)
    for name, state, progress, pace, val, eta in rows:
        print(f"{name:{width}s}  {state:9s} {progress:13s} {pace:11s} {val:28s} {eta}")


if __name__ == "__main__":
    main()
