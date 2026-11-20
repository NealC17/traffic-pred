"""Do the surprises move? Remove the predictable part of the flow, then test whether what is left
(the anomaly) travels between cells from one frame to the next.

Anomaly A_t = (Y_t - HA) / sqrt(HA + 1), HA = the train-period mean for that time of week and cell
(the sqrt scaling evens out Poisson-like noise between busy and quiet cells). If anomalies move
(a storm front, a concert letting out, a commute wave), a cell's next anomaly is predicted by its
neighbours' current ones, beyond its own. Linear models of A_{t+tau}(x), fitted on train, scored on
test (cells with HA >= --threshold, RMSE back in raw counts):

  own          A_t(x) and the other channel's A_t(x), per time-of-day slot
  +3x3 mean    plus the mean of the 8 neighbours (one weight: a shared regional level, no direction)
  +3x3         plus the 8 neighbours' A_t (same channel), per time-of-day slot
  +5x5         plus the 24 cells within 2 steps, per time-of-day slot
  +3x3 /cell   own + 8 neighbours, separate weights for every cell (fixed directional flows)

Also printed: the pooled 3x3 neighbour weights and how asymmetric they are (advection in a fixed
direction makes w(d) differ from w(-d); diffusion keeps them symmetric).

  python tools/anomaly_transport.py --dataset nyctaxi
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trafficpred.data import files
from trafficpred.data.datasets import DATASETS


def shifted(a, r):
    """(T, C, H, W) -> (T, C, H, W, K): the values at every offset within radius r (zero outside)."""
    T, C, H, W = a.shape
    p = np.pad(a, ((0, 0), (0, 0), (r, r), (r, r)))
    offs = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1) if (dy, dx) != (0, 0)]
    return np.stack([p[:, :, r + dy:r + dy + H, r + dx:r + dx + W] for dy, dx in offs], -1), offs


def ridge(X, y, lam=1e-3):
    k = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * len(X) * np.eye(k), X.T @ y)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--threshold", type=float, default=10)
    ap.add_argument("--taus", type=int, nargs="+", default=[1, 2, 4])
    args = ap.parse_args()

    spec = DATASETS[args.dataset]
    *head, last = spec.files()
    if head:
        raise SystemExit("only single-file grid datasets are supported")
    Y = np.asarray(files.load(last), dtype=np.float64)                    # (T, C, H, W)
    T, C, H, W = Y.shape
    cut_test = T - spec.test_frames
    cut_val = cut_test - spec.val_frames
    spd, week = spec.steps_per_day, 7 * spec.steps_per_day
    slot_w = np.arange(T) % week
    ha_tab = np.stack([Y[:cut_val][s::week].mean(0) for s in range(week)])
    HA = ha_tab[slot_w]
    A = (Y - HA) / np.sqrt(HA + 1)
    nb1, offs1 = shifted(A, 1)
    nb2, _ = shifted(A, 2)
    other = A[:, ::-1]                                                    # the other channel, same cell
    slot_d = np.arange(T) % spd

    print(f"{args.dataset}: {T} frames, grid {C}x{H}x{W}; fit on frames < {cut_val}, score on frames >= {cut_test}; "
          f"cells with HA >= {args.threshold:g}")
    for tau in args.taus:
        t_all = np.arange(T - tau)
        train = t_all[t_all + tau < cut_val]
        test = t_all[t_all >= cut_test]
        target = A[t_all + tau]
        scale = np.sqrt(HA[t_all + tau] + 1)
        keep = HA[t_all + tau] >= args.threshold

        def features(kind, t):
            f = [A[t][..., None], other[t][..., None]]
            if kind in ("3x3", "3x3cell"):
                f.append(nb1[t])
            if kind == "5x5":
                f.append(nb2[t])
            if kind == "3x3mean":
                f.append(nb1[t].mean(-1, keepdims=True))
            return np.concatenate(f, -1)                                  # (n, C, H, W, k)

        def score(kind):
            pred = np.zeros((len(t_all), C, H, W))
            if kind == "3x3cell":                                          # weights per cell and channel
                Ftr, Fte = features(kind, train), features(kind, test)
                for c in range(C):
                    for i in range(H):
                        for j in range(W):
                            w = ridge(Ftr[:, c, i, j], target[train, c, i, j])
                            pred[test, c, i, j] = Fte[:, c, i, j] @ w
            else:                                                          # weights per slot of day and channel
                for s in range(spd):
                    tr, te = train[slot_d[train] == s], test[slot_d[test] == s]
                    if len(te) == 0:
                        continue
                    Ftr, Fte = features(kind, tr), features(kind, te)
                    for c in range(C):
                        m = keep[tr, c].reshape(-1)
                        if m.sum() < 10 * Ftr.shape[-1]:                   # too few busy cells in this slot
                            continue
                        X = Ftr[:, c].reshape(-1, Ftr.shape[-1])
                        w = ridge(X[m], target[tr, c].reshape(-1)[m])
                        pred[te, c] = Fte[:, c] @ w
            err = (pred[test] - target[test]) * scale[test]
            return np.sqrt((err[keep[test]] ** 2).mean())

        zero = np.sqrt((((target[test]) * scale[test])[keep[test]] ** 2).mean())
        rows = [("HA only (anomaly 0)", zero)] + [(k, score(k)) for k in ("own", "3x3mean", "3x3", "5x5", "3x3cell")]
        base = rows[1][1]
        print(f"\n  tau = {tau} step(s): RMSE of Y_(t+tau), raw counts")
        for name, r in rows:
            print(f"    {name:20} {r:8.3f}   ({100 * (1 - r ** 2 / base ** 2):+5.1f}% MSE vs own)")

        if tau == 1:                                                       # pooled 3x3 weights, all slots
            F = features("3x3", train)
            for c in range(C):
                m = keep[train, c].reshape(-1)
                w = ridge(F[:, c].reshape(-1, F.shape[-1])[m], target[train, c].reshape(-1)[m])
                grid = np.zeros((3, 3))
                grid[1, 1] = w[0]
                for (dy, dx), v in zip(offs1, w[2:]):
                    grid[dy + 1, dx + 1] = v
                nbw = grid.copy()
                nbw[1, 1] = 0
                asym = np.abs(nbw - nbw[::-1, ::-1]).sum() / 2 / max(np.abs(nbw).sum(), 1e-9)
                name = spec.channels[c] if hasattr(spec, "channels") else c
                print(f"    {name} pooled weights (own {w[0]:+.3f}, other channel {w[1]:+.3f}); neighbour "
                      f"asymmetry {asym:.2f} (0 = symmetric, 1 = one-directional)")
                for row in grid:
                    print("      " + " ".join(f"{v:+.3f}" for v in row))


if __name__ == "__main__":
    main()
