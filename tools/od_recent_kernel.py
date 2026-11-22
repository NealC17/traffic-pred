"""Does a destination kernel from recent days move the trips under way better than the train-period one?

The --od-head prior is P(j | i) fitted once on the train split. If where trips go drifts over time,
the test weeks (the last 20 days) are served by a stale prior. A kernel from the plans of the last
K days is known at forecast time: flow[1][s] (start s, end s+1) has fully arrived by s + 1 <= t.

For each kernel, the RMSE of the carried part of End(t+1) (sum_i r_t[i] P(j | i)) on the val and test
targets, cells with true End >= --threshold:
  train slot     the current one: train plans per time-of-day slot, smoothed toward train all-day
  train all-day  train plans, every slot
  recent K       plans of the K days before the forecast day, every slot
  recent K slot  the same per slot of day (+-1 slot), smoothed toward recent K all-day
  blend          half train slot, half recent K slot

  python tools/od_recent_kernel.py --dataset nyctaxi --stdn-dir <unzipped STDN data>
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trafficpred.data import files
from trafficpred.data.datasets import DATASETS

PREFIX = {"nyctaxi": "", "nycbike2": "bike_"}
PSEUDO_TRIPS = 5


def normalize(k):
    return k / k.sum(-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", choices=tuple(PREFIX), required=True)
    ap.add_argument("--stdn-dir", required=True)
    ap.add_argument("--tin", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=10)
    ap.add_argument("--days", type=int, nargs="+", default=[7, 14, 28])
    args = ap.parse_args()

    spec = DATASETS[args.dataset]
    V = np.asarray(files.load(spec.files()[-1]), dtype=np.float32)
    T, _, H, W = V.shape
    N, spd = H * W, spec.steps_per_day
    f0_rows, f1 = [], []
    for p in ("train", "test"):
        f = np.load(os.path.join(args.stdn_dir, f"{PREFIX[args.dataset]}flow_{p}.npz"))["flow"]
        f0_rows.append(f[0].sum((-2, -1)).astype(np.float32))
        f1.append(f[1].reshape(-1, N, N).astype(np.float32))
        del f
    f0_rows, plans = np.concatenate(f0_rows), np.concatenate(f1)
    del f1
    S, E = V[:, 0], V[:, 1]
    r = (S - f0_rows).clip(0).reshape(T, N)
    carry = np.zeros((T, N), np.float32)
    carry[1:] = plans[:-1].sum(1)                                 # true arrivals at t+1 of trips under way at t
    cut_test = T - spec.test_frames
    cut_val = cut_test - spec.val_frames
    days = T // spd
    slot = np.arange(T) % spd

    train = plans[:cut_val - 1]
    all_day = normalize(train.sum(0) + 1e-3)
    slot_k = np.stack([train[slot[:len(train)] == s].sum(0) for s in range(spd)])
    slot_k = normalize(slot_k + PSEUDO_TRIPS * all_day)

    daily = plans[:days * spd].reshape(days, spd, N, N)           # plans by (day, slot)
    cum = np.concatenate([np.zeros((1, spd, N, N), np.float32), np.cumsum(daily, 0)])   # cum[d] = days < d

    def evaluate(name, kernel_for):
        """kernel_for(d, slots) -> (len(slots), N, N): the kernels for forecasts made on day d."""
        out = []
        for lo, hi in ((cut_val + args.tin, cut_test), (cut_test + args.tin, T)):
            t0 = np.arange(lo, hi) - 1                            # forecast frames; targets t0 + 1
            pred = np.empty((len(t0), N), np.float32)
            for d in np.unique(t0 // spd):
                m = t0 // spd == d
                pred[m] = np.einsum("bi,bij->bj", r[t0[m]], kernel_for(d, slot[t0[m]]))
            y, keep = carry[t0 + 1], E[t0 + 1].reshape(len(t0), N) >= args.threshold
            out.append(np.sqrt(((pred - y)[keep] ** 2).mean()))
        print(f"  {name:22} val {out[0]:7.3f}   test {out[1]:7.3f}", flush=True)

    print(f"{args.dataset}: carried part of End(t+1), RMSE on cells with End >= {args.threshold:g}")
    evaluate("train slot (current)", lambda d, s: slot_k[s])
    evaluate("train all-day", lambda d, s: np.broadcast_to(all_day, (len(s), N, N)))
    for K in args.days:
        def recent(d, s, K=K, per_slot=False):
            win = cum[d] - cum[max(d - K, 0)]                     # (spd, N, N): plans of the K days before d
            allk = normalize(win.sum(0) + 1e-3)
            if not per_slot:
                return np.broadcast_to(allk, (len(s), N, N))
            near = win[(s[:, None] + np.array([-1, 0, 1])) % spd].sum(1)
            return normalize(near + PSEUDO_TRIPS * allk)
        evaluate(f"recent {K}d all-day", recent)
        evaluate(f"recent {K}d slot", lambda d, s, K=K: recent(d, s, K, True))
        evaluate(f"blend train/{K}d slot", lambda d, s, K=K: 0.5 * slot_k[s] + 0.5 * recent(d, s, K, True))

if __name__ == "__main__":
    main()
