"""Convert the BjTT release (Zhang et al., "BjTT: A Large-scale Multimodal Dataset for Traffic Prediction", IEEE
T-ITS 2024; data from the authors on request, licence form at chyazhang.github.io/BjTT) into this repo's format.

The release (BjTT.zip, unzipped into datasets/BjTT/raw/):
  BjTT/data/<month>/<month>_<k>.npy   one 4-minute step each, (36, 36, 2) float64: [congestion level 1-5, speed
                                      km/h]; 0 / 0 = padding. Months 1 / 2 / 3 = Jan / Feb / Mar 2022, k from 1.
  BjTT/text/<month>/<month>_<k>.txt   the step's traffic-event text (date, time, "event on road" sentences)
  BjTT/matrix.npy                     (1260, 1260) int8 road adjacency (directed, self-loops)
  BjTT/Roads1260.json                 the 1260 roads' polylines / names

Layout: January and March hold the 1260 roads of matrix.npy row-major in the first 35 of the 36 columns
(36 x 35 = 1260; the Jan / Mar mean speed per road correlates 0.997). February holds only 1063 roads, packed 30
per row in another order that the release doesn't give (matching them to the 1260 by speed profile is ambiguous:
1063 roads land on 688), so February can't be put on the road graph and is left out. With the published
chronological 70 / 10 / 20 split of the 32,400 steps, val and test are March 5 - 13 and March 13 - 31 either way;
only the train period loses February.

  -> datasets/BjTT/speed_2022_01.npy, speed_2022_03.npy   (T, 1, 1, 1260) float32 speed (0 = missing reading)
     datasets/BjTT/hops.npy                               (1260, 1260) int16 hop distance on the undirected graph

  python -I tools/import_bjtt.py                       # reads datasets/BjTT/raw/BjTT
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from import_pems import hops                  # noqa: E402
from trafficpred.data.files import ROOT       # noqa: E402

MONTHS = {1: 11160, 3: 11160}                 # steps per kept month (31 days x 360)
ROADS, ROWS, COLS = 1260, 36, 35


def month(raw, m, n):
    steps = np.stack([np.load(os.path.join(raw, "data", str(m), f"{m}_{k}.npy"), allow_pickle=False)
                      for k in range(1, n + 1)])                         # (n, 36, 36, 2)
    if steps.shape[1:] != (36, 36, 2):
        raise SystemExit(f"month {m}: unexpected step shape {steps.shape[1:]}")
    level = steps[..., 0]
    if (level[:, :, COLS:] != 0).any() or ((level[:, :, :COLS] == 0).all(0)).any():
        raise SystemExit(f"month {m}: the roads aren't the 36 x 35 row-major layout")
    return steps[:, :, :COLS, 1].reshape(n, ROADS).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw", default=os.path.join(ROOT, "BjTT", "raw", "BjTT"))
    args = ap.parse_args()
    out = os.path.join(ROOT, "BjTT")
    for m, n in MONTHS.items():
        speed = month(args.raw, m, n)
        np.save(os.path.join(out, f"speed_2022_{m:02d}.npy"), speed[:, None, None, :])
        print(f"2022-{m:02d}: {n} steps x {ROADS} roads, speed mean {speed[speed > 0].mean():.1f} km/h, "
              f"max {speed.max():.0f}, missing (0) {np.mean(speed == 0):.2%}")
    adj = np.load(os.path.join(args.raw, "matrix.npy"), allow_pickle=False)
    if adj.shape != (ROADS, ROADS):
        raise SystemExit(f"matrix.npy is {adj.shape}, not {ROADS} x {ROADS}")
    i, j = np.nonzero(adj)
    h = hops(ROADS, [(a, b) for a, b in zip(i, j) if a != b])
    np.save(os.path.join(out, "hops.npy"), h)
    near = [(h == d).sum(1).mean() for d in (1, 2, 3)]
    print(f"graph: {int((i != j).sum())} directed edges, neighbours at 1 / 2 / 3 hops {near[0]:.1f} / {near[1]:.1f} / "
          f"{near[2]:.1f} on average, unreachable pairs {np.mean(h < 0):.1%}")


if __name__ == "__main__":
    main()
