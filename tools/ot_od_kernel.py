"""Destination kernels from the Start / End counts alone (no OD data), for --od-kernel: per time-of-day slot s,
from the train days of the default time split,

  ot    entropic OT between the mean Start map at s and the mean End map at s+1, cost = grid distance in cells,
        eps = --eps cells: the doubly constrained gravity model (Sinkhorn's two scalings are its balancing
        factors); K_s(j | i) = plan_ij / a_i
  dest  K_s(j | i) = the End share at s+1 for every origin: where trips go, ignoring where they come from

Written next to the real kernel as od_kernel_slot_{ot,dest}.npy, (steps per day, N, N).

  python tools/ot_od_kernel.py --dataset nycbike2 --eps 2
"""
import argparse
import os
import sys

import numpy as np
import ot

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trafficpred.data import files                                         # noqa: E402
from trafficpred.data.datasets import DATASETS                             # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", default="nycbike2", choices=("nyctaxi", "nycbike2"))
    ap.add_argument("--eps", type=float, default=2.0, help="entropic regularisation, in cells of the grid cost")
    args = ap.parse_args()
    spec = DATASETS[args.dataset]
    vol = np.asarray(files.load(spec.files()[-1]), np.float64)
    if vol.shape[-1] == 2:
        vol = vol.transpose(0, 3, 1, 2)
    T, _, H, W = vol.shape
    N, spd = H * W, spec.steps_per_day
    cut = T - spec.test_frames - spec.val_frames           # train frames; pairs (t, t + 1) stay inside them
    start, end = vol[:cut, 0].reshape(cut, N), vol[:cut, 1].reshape(cut, N)
    slot = np.arange(cut) % spd
    yy, xx = np.divmod(np.arange(N), W)
    C = np.sqrt((yy[:, None] - yy[None]) ** 2 + (xx[:, None] - xx[None]) ** 2)
    k_ot, k_dest = np.empty((spd, N, N)), np.empty((spd, N, N))
    for s in range(spd):
        t = np.arange(cut - 1)[slot[:-1] == s]
        a, b = start[t].mean(0) + 1e-9, end[t + 1].mean(0) + 1e-9
        a, b = a / a.sum(), b / b.sum()
        P = ot.sinkhorn(a, b, C, reg=args.eps, numItermax=5000, stopThr=1e-9, method="sinkhorn_log")
        k_ot[s] = P / P.sum(1, keepdims=True)
        k_dest[s] = b[None]
    folder = os.path.join(files.ROOT, os.path.dirname(spec.od_file))
    for name, k in (("ot", k_ot), ("dest", k_dest)):
        np.save(os.path.join(folder, f"od_kernel_slot_{name}.npy"), k.astype(np.float32))
    print(f"{args.dataset}: od_kernel_slot_ot.npy (eps {args.eps:g}) and od_kernel_slot_dest.npy, {k_ot.shape}, "
          f"from {cut} train frames")


if __name__ == "__main__":
    main()
