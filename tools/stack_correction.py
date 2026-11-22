"""A second stage on top of trained models: a ridge correction of their next-step forecasts with the
cell's level relative to its usual weekly pattern (tools/amplitude_gain.py's gain features).

Per run (seeds scored apart and averaged), per channel, residual ridge models of Y(t+1):
  calib   P, HA(t+1), 1, Y(t), Y(t-1) of both channels   (a linear rescaling of the forecast)
  gain    calib + (g - 1) * HA(t+1) for the --windows x --radii gains
fitted on one period and scored on another (cells with true >= --threshold, STDN's > 10 by default):
  val -> test        the stacking stage: fitted on the val period, used on the later test period
  train -> test      fitted on the train period (the models' own fitted forecasts, so in-sample)
  val1 -> test       fitted on the first half of val only (further from test)
  val1 -> val2       fitted on the first half of val, scored on the second half
HA is the leave-one-out train mean of the time-of-week slot (as --ha), so train targets don't see
themselves. If 'gain' helps from val but not from train, the correction follows the drift between periods
rather than something a model could learn from its inputs.

  python tools/stack_correction.py checkpoints/workspaces/tuned_models_nyctaxi_in8_ge10/{distvolformer,cnn,stresnet}_tuned_*.pt
"""
import argparse
import collections
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from amplitude_gain import PSEUDO, box_sum, ridge, target_frames       # noqa: E402
from eval_published import predictions                                    # noqa: E402
from trafficpred.data import files                                         # noqa: E402
from trafficpred.data.datasets import DATASETS, usual_pattern              # noqa: E402

MODES = (("val", "test"), ("train", "test"), ("val1", "test"), ("val1", "val2"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--threshold", type=float, default=10.5, help="10.5 = STDN's true > 10")
    ap.add_argument("--windows", type=int, nargs="+", default=[1, 2, 4, 8, 16, 48])
    ap.add_argument("--radii", type=int, nargs="+", default=[0, 1, 2, 4, -1])
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    spec = Y = HA = None
    results = collections.defaultdict(list)          # run -> per seed {mode: (model, calib, gain)}
    for path in args.checkpoints:
        per = {}
        for split in ("train", "val", "test"):
            ck, p, t = predictions(path, device, split, args.batch_size)
            per[split] = (p[:, 0].numpy(), t)
        if Y is None:
            spec = DATASETS[ck["dataset"]]
            Y = np.asarray(files.load(spec.files()[-1]), dtype=np.float64)
            cut_val = len(Y) - spec.test_frames - spec.val_frames
            C = Y.shape[1]
            HA = usual_pattern(Y, cut_val, 7 * spec.steps_per_day)[:, :C].astype(np.float64)
            cy = np.cumsum(np.concatenate([np.zeros_like(Y[:1]), Y]), 0)
            ch = np.cumsum(np.concatenate([np.zeros_like(HA[:1]), HA]), 0)

        def feats(frames, P):
            """(n, C, H, W, k) calib features and (n, C, H, W, k') gain features at forecast time frames - 1."""
            t = frames - 1
            last = [Y[t, c] for c in range(C)] + [Y[t - 1, c] for c in range(C)]
            calib = np.stack([P, HA[frames], np.ones_like(P)]
                             + [np.broadcast_to(l[:, None], P.shape) for l in last], -1)
            gains = []
            for L in args.windows:
                ys = cy[t + 1] - cy[np.maximum(t + 1 - L, 0)]
                hs = ch[t + 1] - ch[np.maximum(t + 1 - L, 0)]
                for rad in args.radii:
                    g = (box_sum(ys, rad) + PSEUDO) / (box_sum(hs, rad) + PSEUDO)
                    gains.append((g - 1) * HA[frames])
            return calib, np.concatenate([calib, np.stack(gains, -1)], -1)

        sets = {}
        for split in ("train", "val", "test"):
            P, true = per[split]
            frames = target_frames(Y, true)
            sets[split] = (frames, P, *feats(frames, P))
        fv = sets["val"]
        half = len(fv[0]) // 2
        sets["val1"] = tuple(a[:half] for a in fv)
        sets["val2"] = tuple(a[half:] for a in fv)

        def score(fit, ev):
            frames_f, _, cf, gf = sets[fit]
            frames_e, Pe, ce, ge = sets[ev]
            Yf, Ye = Y[frames_f], Y[frames_e]
            keep = Ye >= args.threshold - 1e-3
            out = [np.sqrt(((Pe - Ye)[keep] ** 2).mean())]
            for Xf, Xe in ((cf, ce), (gf, ge)):
                pred = np.empty_like(Ye)
                for c in range(C):
                    m = HA[frames_f][:, c] >= args.threshold / 2          # fit cells chosen without the target
                    w = ridge(Xf[:, c][m], Yf[:, c][m])
                    pred[:, c] = Xe[:, c] @ w
                out.append(np.sqrt(((pred - Ye)[keep] ** 2).mean()))
            return out

        run = re.sub(r"_seed\d+$", "", os.path.basename(path)[:-3])
        results[run].append({f"{a}->{b}": score(a, b) for a, b in MODES})
        print(f"done {os.path.basename(path)}", file=sys.stderr, flush=True)

    print(f"{spec.title}: RMSE on cells with true >= {args.threshold:g}; mean over seeds; "
          "model / + calib / + gain (gain's change vs model)")
    head = "".join(f"{m:>27}" for m in (f"{a}->{b}" for a, b in MODES))
    print(f"  {'run':40}{head}")
    for run, seeds in results.items():
        row = ""
        for a, b in MODES:
            m, c, g = np.mean([s[f"{a}->{b}"] for s in seeds], 0)
            row += f"   {m:6.2f} {c:6.2f} {g:6.2f} ({g - m:+.2f})"
        print(f"  {run[:40]:40}{row}  n={len(seeds)}")


if __name__ == "__main__":
    main()
