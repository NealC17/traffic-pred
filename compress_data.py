"""
Compress every .npy under datasets/ into a .npy.xz small enough for GitHub.

    python compress_data.py                  # TaxiBJ etc. (skips Moving MNIST: data.py downloads it)
    python compress_data.py --include-mnist

Lossless:
  1. Arrays of whole numbers stored as floats (TaxiBJ flows are taxi counts)
     are stored as the smallest integer type that holds them (e.g. uint16),
     which alone is 4x smaller than float64. Other floats go to float32 only
     if that round-trips exactly; otherwise they keep their dtype.
  2. The result is xz-compressed (stdlib lzma, max preset).
  3. Each file is decompressed again and compared bit-for-bit with the original.

The original dtype/shape and a checksum go into datasets/compressed.json;
data.py uses it to restore each .npy exactly as it was.

Commit the .npy.xz files and compressed.json; keep the .npy files out of git:
    datasets/**/*.npy
    datasets/MovingMNIST/
    datasets/MNIST_raw/
"""

import argparse
import glob
import hashlib
import io
import json
import lzma
import os

import numpy as np

ROOT = "datasets"
MANIFEST = os.path.join(ROOT, "compressed.json")
GITHUB_LIMIT = 100 * 2**20          # GitHub rejects files over 100 MB
GITHUB_WARN = 50 * 2**20            # and warns over 50 MB


def sha256(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def smallest_lossless(arr):
    """Smallest dtype that stores arr exactly."""
    if arr.dtype == object or not np.issubdtype(arr.dtype, np.number):
        return arr.dtype
    if np.issubdtype(arr.dtype, np.floating):
        if not np.isfinite(arr).all():
            pass
        elif np.array_equal(arr, np.round(arr)):                      # whole numbers
            lo, hi = arr.min(), arr.max()
            for dt in (np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32):
                info = np.iinfo(dt)
                if info.min <= lo and hi <= info.max:
                    return np.dtype(dt)
        if arr.dtype == np.float64 and np.array_equal(arr.astype(np.float32).astype(np.float64), arr):
            return np.dtype(np.float32)
        return arr.dtype
    lo, hi = arr.min(), arr.max()                                     # already integer
    for dt in (np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32):
        info = np.iinfo(dt)
        if info.min <= lo and hi <= info.max:
            return np.dtype(dt) if np.dtype(dt).itemsize < arr.dtype.itemsize else arr.dtype
    return arr.dtype


def compress(path):
    arr = np.load(path, allow_pickle=True)
    small = arr.astype(smallest_lossless(arr)) if arr.dtype != object else arr
    buf = io.BytesIO()
    np.save(buf, small, allow_pickle=True)
    out = path + ".xz"
    with open(out, "wb") as f:
        f.write(lzma.compress(buf.getvalue(), preset=9 | lzma.PRESET_EXTREME))

    # verify round trip
    with lzma.open(out) as f:
        back = np.lib.format.read_array(f, allow_pickle=True).astype(arr.dtype)
    same = (back.shape == arr.shape and
            (np.array_equal(back, arr) if arr.dtype == object else sha256(back) == sha256(arr)))
    if not same:
        os.remove(out)
        raise RuntimeError(f"round-trip mismatch for {path}; nothing written")

    info = dict(dtype=str(arr.dtype), stored_dtype=str(small.dtype), shape=list(arr.shape),
                sha256=None if arr.dtype == object else sha256(arr))
    return out, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-mnist", action="store_true",
                    help="also compress Moving MNIST (data.py can re-download it instead)")
    args = ap.parse_args()

    manifest = {}
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            manifest = json.load(f)

    total_in = total_out = 0
    for path in sorted(glob.glob(os.path.join(ROOT, "**", "*.npy"), recursive=True)):
        rel = os.path.relpath(path, ROOT)
        if "mnist" in rel.lower() and not args.include_mnist:
            print(f"skip  {rel}  (downloaded by data.py)")
            continue
        out, info = compress(path)
        manifest[os.path.relpath(out, ROOT).replace(os.sep, "/")] = info
        a, b = os.path.getsize(path), os.path.getsize(out)
        total_in, total_out = total_in + a, total_out + b
        flag = ("  !! over GitHub's 100 MB limit" if b > GITHUB_LIMIT else
                "  (over 50 MB: GitHub will warn)" if b > GITHUB_WARN else "")
        print(f"{rel:40s} {info['dtype']:>8s} -> {info['stored_dtype']:<8s} "
              f"{a / 2**20:8.1f} MB -> {b / 2**20:6.1f} MB  ({a / b:4.1f}x){flag}")

    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=1)
    if total_out:
        print(f"\ntotal {total_in / 2**20:.1f} MB -> {total_out / 2**20:.1f} MB "
              f"({total_in / total_out:.1f}x); manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
