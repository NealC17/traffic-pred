"""What can origin-destination (OD) data add to next-step End prediction, without leaking the future?

STDN's flow arrays (flow_*.npz, bike_flow_*.npz in STDN's data.zip; axes (2, T, origin H, W, dest H, W)):
  flow[0][t]  trips that start and end in slot t
  flow[1][t]  trips that start in slot t and end in slot t+1
so  Start(t) = rowsum(flow[0][t] + flow[1][t])  and  End(t+1) = colsum(flow[0][t+1]) + colsum(flow[1][t]).
At forecast time t the in-progress mass per origin, r_t = Start(t) - rowsum(flow[0][t]), is known, but
not where it goes (colsum flow[1][t] is part of the target: never an input). Moving the known r_t
to destinations is a transport problem with known origin margins:

  carry_hat(t)[j] = sum_i r_t[i] * P(j | i, time of day)      P from train flow[1] plans (row-normalized)

Prints, on the test windows the models are scored on (cells with true End >= --threshold):
  * how End(t+1) splits into the within-slot part and the carried part
  * how well carry_hat predicts the carried part
  * training-free End predictors: last, HA, and HA(within) + carry_hat; and the oracle with the true
    carried part (a bound on what knowing the in-progress destinations would give)
  * the value of carry_hat as a feature: ridge models of End(t+1) per time-of-day slot from the counts
    alone vs the counts + r_t / carry_hat (fitted on train, scored on test)

  python tools/od_baselines.py --dataset nyctaxi --tin 8 --stdn-dir <unzipped STDN data>
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trafficpred.data import files
from trafficpred.data.datasets import DATASETS

PREFIX = {"nyctaxi": "", "nycbike2": "bike_"}


def rmse(p, y, keep):
    return float(np.sqrt(((p - y)[keep] ** 2).mean()))


def ridge(X, y, lam=1e-3):
    return np.linalg.solve(X.T @ X + lam * len(X) * np.eye(X.shape[1]), X.T @ y)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", choices=tuple(PREFIX), required=True)
    ap.add_argument("--tin", type=int, required=True, help="input length of the models compared with")
    ap.add_argument("--stdn-dir", required=True)
    ap.add_argument("--threshold", type=float, default=10)
    args = ap.parse_args()

    spec = DATASETS[args.dataset]
    V = np.asarray(files.load(spec.files()[-1]), dtype=np.float64)          # (T, 2, H, W) Start, End
    T, _, H, W = V.shape
    pre = PREFIX[args.dataset]
    parts = [np.load(os.path.join(args.stdn_dir, f"{pre}flow_{p}.npz"))["flow"] for p in ("train", "test")]
    if sum(p.shape[1] for p in parts) != T:
        raise SystemExit("flow frames don't match the dataset's frames")
    rows = np.concatenate([p.sum((-2, -1)) for p in parts], 1)              # (2, T, H, W) by origin
    cols = np.concatenate([p.sum((2, 3)) for p in parts], 1)                # (2, T, H, W) by destination
    cut_test = T - spec.test_frames
    cut_val = cut_test - spec.val_frames
    N = H * W
    spd, week = spec.steps_per_day, 7 * spec.steps_per_day
    carry_plans = parts[0][1][:cut_val - 1].reshape(-1, N, N).astype(np.float32)   # train flow[1], t < cut_val - 1
    del parts
    S, E = V[:, 0], V[:, 1]
    assert np.sqrt(((rows[0] + rows[1] - S) ** 2).mean()) < 2, "Start != rowsum(flow): wrong axes?"
    assert np.sqrt(((cols[0][1:] + cols[1][:-1] - E[1:]) ** 2).mean()) < 2, "End identity fails"

    # destination kernel P(j | i, slot of day), from train plans, smoothed toward the all-day kernel
    slot = np.arange(T) % spd
    tr_t = np.arange(len(carry_plans))
    k_all = carry_plans.sum(0) + 1e-3
    kernel = np.empty((spd, N, N), np.float32)
    for s in range(spd):
        k = carry_plans[slot[tr_t] == s].sum(0)
        kernel[s] = k + 5 * k_all / k_all.sum(1, keepdims=True)                # ~5 pseudo-trips per origin
    kernel /= kernel.sum(2, keepdims=True)

    r = (S - rows[0]).clip(0).reshape(T, N)                                 # in-progress mass at t, by origin
    carry_hat = np.einsum("ti,tij->tj", r, kernel[slot]).reshape(T, H, W)   # expected arrivals at t+1
    carry = np.zeros_like(E)
    carry[1:] = cols[1][:-1]                                                # true arrivals at t+1 from trips under way at t
    within = cols[0]

    t1 = np.arange(cut_test + args.tin, T)                                  # targets of the test windows
    keep = E[t1] >= args.threshold
    ha = lambda X: np.stack([X[:cut_val][s::week].mean(0) for s in range(week)])[np.arange(T) % week]
    HA_E, HA_W, HA_C = ha(E), ha(within), ha(carry)

    tot = E[t1][keep].sum()
    print(f"{args.dataset}: test End(t+1), {len(t1)} targets, cells with End >= {args.threshold:g}")
    print(f"  End = within-slot {100 * within[t1][keep].sum() / tot:.1f}% + carried from trips under way "
          f"{100 * carry[t1][keep].sum() / tot:.1f}%")
    print(f"  carried part: carry_hat RMSE {rmse(carry_hat[t1 - 1], carry[t1], keep):.3f}  "
          f"(HA of carried part {rmse(HA_C[t1], carry[t1], keep):.3f}, last {rmse(carry[t1 - 1], carry[t1], keep):.3f})")
    print("  End(t+1) RMSE, training-free:")
    for name, p in (("last End", E[t1 - 1]), ("HA_week End", HA_E[t1]),
                    ("HA within + HA carried", HA_W[t1] + HA_C[t1]),
                    ("HA within + carry_hat (OD)", HA_W[t1] + carry_hat[t1 - 1]),
                    ("oracle: HA within + true carried", HA_W[t1] + carry[t1])):
        print(f"    {name:34} {rmse(p, E[t1], keep):8.3f}")

    # value as a feature: ridge per slot of day, train frames -> test windows
    def feats(t, od):
        f = [E[t], S[t], E[t - 1], S[t - 1], HA_E[t + 1], np.ones_like(E[t])]
        if od:
            f += [r.reshape(T, H, W)[t], carry_hat[t]]
        return np.stack(f, -1)                                              # (n, H, W, k)
    tr = np.arange(max(args.tin, 1), cut_val - 1)
    print("  End(t+1) RMSE, ridge per slot of day on [End, Start at t, t-1, HA]:")
    for od in (False, True):
        pred = np.zeros((len(t1), H, W))
        te = t1 - 1
        for s in range(spd):
            a, b = tr[slot[tr] == s], np.where(slot[te] == s)[0]
            if len(b) == 0:
                continue
            Fa = feats(a, od).reshape(-1, 8 if od else 6)
            w = ridge(Fa, E[a + 1].reshape(-1))
            pred[b] = feats(te[b], od) @ w
        print(f"    {'+ r_t, carry_hat (OD)' if od else 'counts only':34} {rmse(pred, E[t1], keep):8.3f}")


if __name__ == "__main__":
    main()
