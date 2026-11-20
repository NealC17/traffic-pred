"""Import the ST-SSL datasets (Ji et al., AAAI'23) as the exact windows the paper trains and tests on.

Download https://github.com/Echo-Ji/ST-SSL_Dataset (the .npz files are Git LFS objects: fetch them
with git lfs, or from https://media.githubusercontent.com/media/Echo-Ji/ST-SSL_Dataset/main/<path>),
then

  python tools/import_stssl.py <download dir>        # holds NYCTaxi/ NYCBike1/ NYCBike2/ BJTaxi/

Writes datasets/STSSL/<name>/{train,val,test}_{x,y}.npy (float32, (samples, frames, 2, H, W), node
order row-major) and offsets.npy (each input frame's step relative to the last one). Train with
--dataset stssl_nyctaxi / stssl_nycbike1 / stssl_nycbike2 / stssl_bjtaxi.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trafficpred.data.files import ROOT

GRIDS = {"NYCTaxi": (10, 20), "NYCBike1": (16, 8), "NYCBike2": (10, 20), "BJTaxi": (32, 32)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src")
    args = ap.parse_args()
    for name, (H, W) in GRIDS.items():
        if not os.path.isdir(os.path.join(args.src, name)):
            print(f"{name}: not in {args.src}, skipped")
            continue
        out = os.path.join(ROOT, "STSSL", name)
        os.makedirs(out, exist_ok=True)
        for split in ("train", "val", "test"):
            z = np.load(os.path.join(args.src, name, f"{split}.npz"), allow_pickle=False)
            for key in ("x", "y"):
                a = z[key]                                                    # (S, L, N, C)
                a = a.reshape(a.shape[0], a.shape[1], H, W, a.shape[3]).transpose(0, 1, 4, 2, 3)
                np.save(os.path.join(out, f"{split}_{key}.npy"), np.ascontiguousarray(a, dtype=np.float32))
            if split == "train":
                np.save(os.path.join(out, "offsets.npy"), z["x_offsets"].ravel().astype(np.int64))
                if z["y_offsets"].ravel().tolist() != [1]:
                    raise SystemExit(f"{name}: expected next-step targets, got y_offsets {z['y_offsets'].ravel()}")
            print(f"{name} {split}: {z['x'].shape[0]} windows of {z['x'].shape[1]} frames, grid {H}x{W}")


if __name__ == "__main__":
    main()
