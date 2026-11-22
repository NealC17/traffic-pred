"""Does a live amplitude of the periodic pattern explain what a trained model still gets wrong?

Every cell has a usual weekly pattern HA (the train-period mean for its time of week). On some days a
region runs above or below it. The gain over the last L frames, pooled over a (2r+1)^2 region,

  g(t) = (sum of Y over the region and frames t-L+1..t + a) / (sum of HA over the same + a)

is known at forecast time t. (g - 1) * HA(t+1) is the forecast's correction if today's offset holds.
For the model's next-step predictions P on val and test, residual ridge models of Y(t+1), per channel:

  calibration   P, HA(t+1), 1                            (what a linear rescaling of P alone gives)
  + gains       plus (g - 1) * HA(t+1) for every L in --windows and r in --radii

are fitted on val and scored on test (cells with true >= --threshold). If '+ gains' removes clearly
more squared error than 'calibration', the model is missing the live amplitude and it is worth
feeding in; if not, the model already has it.

  python tools/amplitude_gain.py checkpoints/workspaces/tuned_models_nyctaxi_in8_ge10/distvolformer_tuned_nyctaxi_in8_out1*.pt
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_published import predictions                                    # noqa: E402
from trafficpred.data import files                                         # noqa: E402
from trafficpred.data.datasets import DATASETS                             # noqa: E402

PSEUDO = 20.0   # pseudo-counts that shrink a gain toward 1 when the region was quiet


def box_sum(a, r):
    """Sum over the (2r+1)^2 neighbourhood of every cell (zero padded); r < 0: the whole grid."""
    if r < 0:
        return np.broadcast_to(a.sum((-2, -1), keepdims=True), a.shape)
    c = np.pad(a, [(0, 0)] * (a.ndim - 2) + [(r + 1, r), (r + 1, r)]).cumsum(-2).cumsum(-1)
    H, W = a.shape[-2:]
    k = 2 * r + 1
    return c[..., k:k + H, k:k + W] - c[..., :H, k:k + W] - c[..., k:k + H, :W] + c[..., :H, :W]


def ridge(X, y, lam=1e-4):
    return np.linalg.solve(X.T @ X + lam * len(X) * np.eye(X.shape[1]), X.T @ y)


def target_frames(Y, true):
    """The frame index of every window's target; checks that the windows are consecutive frames."""
    first = true[0, 0].numpy()
    hits = [f for f in range(len(Y)) if np.abs(Y[f] - first).max() < 1e-2]   # targets come back rescaled
    for f0 in hits:
        idx = f0 + np.arange(len(true))
        if idx[-1] < len(Y) and np.abs(Y[idx] - true[:, 0].numpy()).max() < 1e-2:
            return idx
    raise SystemExit("could not place the windows' targets in the dataset's frames")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--threshold", type=float, default=10.5, help="10.5 = STDN's true > 10")
    ap.add_argument("--windows", type=int, nargs="+", default=[1, 2, 4, 8, 16, 48])
    ap.add_argument("--radii", type=int, nargs="+", default=[0, 1, 2, 4, -1])
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    runs = []
    for path in args.checkpoints:
        ck, pv, tv = predictions(path, device, "val", args.batch_size)
        _, pt, tt = predictions(path, device, "test", args.batch_size)
        runs.append((os.path.basename(path), pv[:, 0].numpy(), pt[:, 0].numpy(), tv, tt))
        print(f"loaded {runs[-1][0]}", file=sys.stderr, flush=True)
    spec = DATASETS[ck["dataset"]]
    Y = np.asarray(files.load(spec.files()[-1]), dtype=np.float64)          # (T, C, H, W)
    T = len(Y)
    week = 7 * spec.steps_per_day
    cut_val = T - spec.test_frames - spec.val_frames
    ha_week = np.stack([Y[:cut_val][s::week].mean(0) for s in range(week)])
    HA = ha_week[np.arange(T) % week]
    fv, ft = target_frames(Y, runs[0][3]), target_frames(Y, runs[0][4])
    for r in runs[1:]:
        if not (np.array_equal(r[3].numpy(), runs[0][3].numpy()) and np.array_equal(r[4].numpy(), runs[0][4].numpy())):
            raise SystemExit("the checkpoints were scored on different windows")

    # gain features at forecast frame t = target - 1
    cy, ch = np.cumsum(np.concatenate([np.zeros_like(Y[:1]), Y]), 0), np.cumsum(np.concatenate([np.zeros_like(HA[:1]), HA]), 0)
    def gains(frames):
        t = frames - 1
        cols, names = [], []
        for L in args.windows:
            ys, hs = cy[t + 1] - cy[np.maximum(t + 1 - L, 0)], ch[t + 1] - ch[np.maximum(t + 1 - L, 0)]
            for rad in args.radii:
                g = (box_sum(ys, rad) + PSEUDO) / (box_sum(hs, rad) + PSEUDO)
                cols.append((g - 1) * HA[frames])
                names.append(f"L{L} r{'all' if rad < 0 else rad}")
        return np.stack(cols, -1), names                                    # (n, C, H, W, k)
    Gv, names = gains(fv)
    Gt, _ = gains(ft)
    # control: the raw last frames (both channels), which the model sees too
    lastf = lambda f: np.stack([Y[f - 1, c] for c in range(Y.shape[1])] + [Y[f - 2, c] for c in range(Y.shape[1])], -1)
    Gv, Gt = np.concatenate([Gv, lastf(fv)[:, None].repeat(Y.shape[1], 1)], -1), np.concatenate([Gt, lastf(ft)[:, None].repeat(Y.shape[1], 1)], -1)
    n_gain = len(names)
    last_idx = list(range(n_gain, Gv.shape[-1]))
    Yv, Yt, Hv, Ht = Y[fv], Y[ft], HA[fv], HA[ft]
    keep_t = Yt >= args.threshold - 1e-3
    fit_v = np.maximum(Hv, 0) >= args.threshold / 2                        # fit cells chosen without the target

    print(f"{ck['dataset']} in{ck['t_in']}: test RMSE on cells with true >= {args.threshold:g}; ridge fitted on val")
    print(f"  {'run':48} {'model':>8} {'calib':>8} {'+last2':>8} {'+gains':>8} {'+both':>8} {'gain MSE %':>10}  {'best single gain':>18}")
    preds = [(n, pv, pt) for n, pv, pt, _, _ in runs]
    if len(runs) > 1:
        preds.append(("seed ensemble", np.mean([p[1] for p in preds], 0), np.mean([p[2] for p in preds], 0)))
    for name, Pv, Pt in preds:
        res = {}
        best = (np.inf, "")
        for label, extra in (("calib", []), ("last", last_idx), ("gains", list(range(n_gain))),
                             ("both", list(range(n_gain)) + last_idx)):
            pred = np.empty_like(Yt)
            for c in range(Y.shape[1]):
                def X(P, H, G):
                    base = [P[:, c], H[:, c], np.ones_like(P[:, c])]
                    return np.stack(base + [G[:, c, ..., k] for k in extra], -1)
                Xv, Xt = X(Pv, Hv, Gv), X(Pt, Ht, Gt)
                m = fit_v[:, c]
                w = ridge(Xv[m], Yv[:, c][m])
                pred[:, c] = Xt @ w
            res[label] = np.sqrt(((pred - Yt)[keep_t] ** 2).mean())
        for k in range(n_gain):                                             # one gain at a time
            pred = np.empty_like(Yt)
            for c in range(Y.shape[1]):
                Xv = np.stack([Pv[:, c], Hv[:, c], np.ones_like(Hv[:, c]), Gv[:, c, ..., k]], -1)
                Xt = np.stack([Pt[:, c], Ht[:, c], np.ones_like(Ht[:, c]), Gt[:, c, ..., k]], -1)
                m = fit_v[:, c]
                pred[:, c] = Xt @ ridge(Xv[m], Yv[:, c][m])
            best = min(best, (np.sqrt(((pred - Yt)[keep_t] ** 2).mean()), names[k]))
        model = np.sqrt(((Pt - Yt)[keep_t] ** 2).mean())
        gain_pct = 100 * (1 - res["both"] ** 2 / res["last"] ** 2)   # beyond the raw last frames
        print(f"  {name[:48]:48} {model:8.3f} {res['calib']:8.3f} {res['last']:8.3f} {res['gains']:8.3f} {res['both']:8.3f} {gain_pct:9.1f}%  "
              f"{best[0]:8.3f} ({best[1]})", flush=True)

    # training-free reference: HA scaled by a few gains, vs plain HA
    print("  training-free on test: HA_week %.3f" % np.sqrt(((Ht - Yt)[keep_t] ** 2).mean()), end="")
    for k, n in enumerate(names):
        if n in ("L4 r1", "L8 r2", "L16 rall"):
            print(f"   HA*gain({n}) {np.sqrt(((Ht + Gt[..., k] - Yt)[keep_t] ** 2).mean()):.3f}", end="")
    print()


if __name__ == "__main__":
    main()
