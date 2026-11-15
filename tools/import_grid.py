"""Convert a grid flow dataset into this repo's format: datasets/<Dir>/<name>.npy, (T, C, H, W).

Inputs are .npy or .npz arrays of shape (T, H, W, C) or (T, C, H, W); several inputs are joined
in time in the order given (e.g. a train and a test file that follow each other).

  # NYC taxi / bike (STDN, Yao et al. AAAI'19: https://github.com/tangxianfeng/STDN, data.zip)
  python tools/import_grid.py STDN/volume_train.npz STDN/volume_test.npz --out NYCTaxi/volume.npy
  python tools/import_grid.py STDN/bike_volume_train.npz STDN/bike_volume_test.npz --out NYCBike2/volume.npy

  # Yahoo! Bousai Crowd (DeepCrowd; by application: https://randd.yahoo.co.jp/en/softwaredata).
  # The published tensors are scaled to 0-1; --scale gives back (about) counts.
  python tools/import_grid.py flowio_tokyo_20170401_20170709_30min.npy --scale 1200 --out Bousai/flowio_tokyo.npy
  python tools/import_grid.py flowio_osaka_20170401_20170709_30min.npy --scale 770 --out Bousai/flowio_osaka.npy

Then  python tools/compress_data.py datasets/<Dir>/<name>.npy  writes the .npy.xz that is committed.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trafficpred.data.files import ROOT


def load(path, key=None):
    data = np.load(path, allow_pickle=False)
    if isinstance(data, np.lib.npyio.NpzFile):
        keys = data.files
        if key is None:
            if len(keys) != 1:
                raise SystemExit(f"{path} holds {keys}; pick one with --key")
            key = keys[0]
        return data[key]
    return data


def to_tchw(a, channels):
    if a.ndim != 4:
        raise SystemExit(f"expected a 4-D array (T, H, W, C) or (T, C, H, W), got shape {a.shape}")
    if a.shape[1] == channels and a.shape[-1] != channels:
        return a
    if a.shape[-1] == channels:
        return np.moveaxis(a, -1, 1)
    raise SystemExit(f"no axis of size {channels} (the channel count) in shape {a.shape}; set --channels")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help=".npy / .npz files, joined in time in this order")
    ap.add_argument("--out", required=True, help="path under datasets/, e.g. NYCTaxi/volume.npy")
    ap.add_argument("--key", help="array name inside .npz inputs (default: their only array)")
    ap.add_argument("--channels", type=int, default=2, help="channel count (default 2: inflow, outflow)")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply the values by this (default 1)")
    args = ap.parse_args()

    parts = [to_tchw(load(p, args.key), args.channels) for p in args.inputs]
    if len({p.shape[1:] for p in parts}) != 1:
        raise SystemExit(f"inputs differ in grid shape: {[p.shape for p in parts]}")
    arr = np.concatenate(parts).astype(np.float64) * args.scale
    if not np.isfinite(arr).all() or arr.min() < 0:
        raise SystemExit("the data has NaN / inf or negative values; not writing it")
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out, arr)
    T, C, H, W = arr.shape
    print(f"wrote {out}: {T} frames, {C} x {H} x {W}, values {arr.min():g} .. {arr.max():g} "
          f"(mean {arr.mean():.3g})")


if __name__ == "__main__":
    main()
