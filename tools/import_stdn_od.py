"""Turn STDN's origin-destination flows into per-frame input maps that leak nothing (--od in train.py).

STDN's data.zip (github.com/tangxianfeng/STDN) holds, next to the Start / End volumes we train on,
flow_{train,test}.npz (taxi) and bike_flow_{train,test}.npz (bike), axes (2, T, origin H, W, dest H, W):
  flow[0][t]  trips that start and end in slot t
  flow[1][t]  trips that start in slot t and end in slot t+1
For every frame t this writes two maps known at the end of slot t:
  in_progress[t]  r_t = Start(t) - rowsum(flow[0][t]): trips under way, by origin
  arrivals[t]     sum_i r_t[i] P(j | i, time of day): where they are expected to arrive at t+1, with
                  the destination kernel P estimated from the train split's flow[1] plans only
The destinations of flow[1][t] are part of End(t+1), the target, and are never used.

  python tools/import_stdn_od.py <unzipped STDN data dir>      # -> datasets/{NYCTaxi,NYCBike2}/od_aux.npy, od_kernel.npy
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trafficpred.data import files
from trafficpred.data.datasets import DATASETS

SOURCES = {"nyctaxi": ("", "NYCTaxi"), "nycbike2": ("bike_", "NYCBike2")}
PSEUDO_TRIPS = 5   # per origin and time-of-day slot: smooths the slot kernel toward the all-day one


def od_features(spec, volume, flow_train, flow_test):
    """(T, 2, H, W) float32: [in_progress, arrivals] per frame. flow_*: STDN's (2, T, H, W, H, W)."""
    T, _, H, W = volume.shape
    N = H * W
    if flow_train.shape[1] + flow_test.shape[1] != T:
        raise SystemExit("the flow files' frames don't add up to the dataset's frames")
    within_by_origin = np.concatenate([flow_train[0].sum((-2, -1)), flow_test[0].sum((-2, -1))])
    start = volume[:, 0]
    rows = within_by_origin + np.concatenate([flow_train[1].sum((-2, -1)), flow_test[1].sum((-2, -1))])
    if np.sqrt(((rows - start) ** 2).mean()) > 2:
        raise SystemExit("Start != row sums of the flow arrays: wrong file or axes")
    cut_val = T - spec.test_frames - spec.val_frames
    if cut_val - 1 > flow_train.shape[1]:
        raise SystemExit("the train split reaches past STDN's train file")
    plans = flow_train[1][:cut_val - 1].reshape(-1, N, N)            # only plans that ended inside train
    spd = spec.steps_per_day
    slot = np.arange(T) % spd
    all_day = plans.sum(0) + 1e-3
    all_day /= all_day.sum(1, keepdims=True)
    kernel = np.empty((spd, N, N), np.float64)
    for s in range(spd):
        kernel[s] = plans[slot[:len(plans)] == s].sum(0) + PSEUDO_TRIPS * all_day
    kernel /= kernel.sum(2, keepdims=True)
    in_progress = (start - within_by_origin).clip(0)
    arrivals = np.einsum("ti,tij->tj", in_progress.reshape(T, N), kernel[slot]).reshape(T, H, W)
    return np.stack([in_progress, arrivals], 1).astype(np.float32), all_day.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", help="the unzipped STDN data directory")
    args = ap.parse_args()
    for name, (prefix, folder) in SOURCES.items():
        paths = [os.path.join(args.src, f"{prefix}flow_{p}.npz") for p in ("train", "test")]
        if not all(os.path.exists(p) for p in paths):
            print(f"{name}: no {prefix}flow_*.npz in {args.src}, skipped")
            continue
        spec = DATASETS[name]
        volume = np.asarray(files.load(spec.files()[-1]), dtype=np.float64)
        flow_train, flow_test = (np.load(p)["flow"] for p in paths)
        aux, kernel = od_features(spec, volume, flow_train, flow_test)
        del flow_train, flow_test
        out = os.path.join(files.ROOT, folder, "od_aux.npy")
        np.save(out, aux)
        np.save(os.path.join(files.ROOT, folder, "od_kernel.npy"), kernel)
        print(f"{name}: {out} {aux.shape} (in-progress trips, expected arrivals at t+1) and od_kernel.npy "
              f"{kernel.shape} (the all-day destination kernel P(j | i) of the train split, for --od-head)")


if __name__ == "__main__":
    main()
