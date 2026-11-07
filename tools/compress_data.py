import argparse
import glob
import hashlib
import io
import json
import lzma
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trafficpred.data.files import MANIFEST, ROOT

GITHUB_LIMIT = 100 * 2**20
GITHUB_WARN = 50 * 2**20
INT_TYPES = (np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32)


def sha256(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _smallest_int(lo, hi, current=None):
    for dt in INT_TYPES:
        info = np.iinfo(dt)
        if info.min <= lo and hi <= info.max:
            if current is None or np.dtype(dt).itemsize < current.itemsize:
                return np.dtype(dt)
            return current
    return current


def smallest_lossless(arr):
    if arr.dtype == object or not np.issubdtype(arr.dtype, np.number):
        return arr.dtype
    if np.issubdtype(arr.dtype, np.integer):
        return _smallest_int(arr.min(), arr.max(), arr.dtype)
    if np.isfinite(arr).all() and np.array_equal(arr, np.round(arr)):
        found = _smallest_int(arr.min(), arr.max())
        if found is not None:
            return found
    if arr.dtype == np.float64 and np.array_equal(arr.astype(np.float32).astype(np.float64), arr):
        return np.dtype(np.float32)
    return arr.dtype


def compress(path):
    arr = np.load(path, allow_pickle=True)
    small = arr if arr.dtype == object else arr.astype(smallest_lossless(arr))
    buf = io.BytesIO()
    np.save(buf, small, allow_pickle=True)
    out = path + ".xz"
    with open(out, "wb") as f:
        f.write(lzma.compress(buf.getvalue(), preset=9 | lzma.PRESET_EXTREME))

    with lzma.open(out) as f:
        back = np.lib.format.read_array(f, allow_pickle=True).astype(arr.dtype)
    same = back.shape == arr.shape and (
        np.array_equal(back, arr) if arr.dtype == object else sha256(back) == sha256(arr))
    if not same:
        os.remove(out)
        raise RuntimeError(f"round-trip mismatch for {path}; nothing written")
    return out, dict(dtype=str(arr.dtype), stored_dtype=str(small.dtype), shape=list(arr.shape),
                     sha256=None if arr.dtype == object else sha256(arr))


def main():
    ap = argparse.ArgumentParser(description="Losslessly compress every datasets/**/*.npy into a "
                                             ".npy.xz small enough for GitHub.")
    ap.add_argument("--include-mnist", action="store_true",
                    help="also compress Moving MNIST (it can be re-downloaded instead)")
    args = ap.parse_args()

    manifest = {}
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            manifest = json.load(f)
    total_in = total_out = 0
    for path in sorted(glob.glob(os.path.join(ROOT, "**", "*.npy"), recursive=True)):
        rel = os.path.relpath(path, ROOT)
        if "mnist" in rel.lower() and not args.include_mnist:
            print(f"skip  {rel}  (downloaded on demand)")
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
