"""
Loads every .npy under datasets/ and wraps each one as a torch Dataset.
Downloads Moving MNIST first if it's missing.

    arrays   : key -> numpy array (memory-mapped where possible)
    datasets : key -> torch.utils.data.Dataset   (same keys as `arrays`)
        TaxiBJ flow files     -> FlowWindowDataset   yields (x, y), each (4, 2, 32, 32)
        mnist_test_seq.npy    -> MovingMNISTDataset  yields (x, y), each (10, 1, 64, 64)
        other numeric arrays  -> TensorDataset over the first axis (e.g. TaxiBJext*)

Keys are the file's path under datasets/ with separators replaced by "-",
e.g. "TaxiBJ-TaxiBJ2013.npy", "MovingMNIST-mnist_test_seq.npy".

Train/test splits matching the paper: taxibj_splits(), moving_mnist_splits().
"""

import os
import glob
import urllib.request

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, ConcatDataset

ROOT = "datasets"
MMNIST_URL = "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"
TAXIBJ_PREFIX = "TaxiBJ-TaxiBJ2"      # flow files; excludes TaxiBJ-TaxiBJext*
TAXIBJ_TEST_FRAMES = 28 * 48          # last 4 weeks of half-hour slots


# ---------------------------------------------------------------------------
# Download Moving MNIST
# ---------------------------------------------------------------------------
mnist_path = os.path.join(ROOT, "MovingMNIST", "mnist_test_seq.npy")
if not os.path.exists(mnist_path):
    os.makedirs(os.path.dirname(mnist_path), exist_ok=True)
    print("Downloading Moving MNIST (~800 MB)...")
    urllib.request.urlretrieve(MMNIST_URL, mnist_path + ".part")
    os.replace(mnist_path + ".part", mnist_path)


# ---------------------------------------------------------------------------
# Unpack compressed .npy.xz files (made by compress_data.py) on first run
# ---------------------------------------------------------------------------
def _decompress_all():
    """For each datasets/**/X.npy.xz without an X.npy next to it, write X.npy
    with its original dtype restored (from datasets/compressed.json), so the
    loader below sees exactly the files that were compressed."""
    import json
    import lzma

    manifest_path = os.path.join(ROOT, "compressed.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    for xz in sorted(glob.glob(os.path.join(ROOT, "**", "*.npy.xz"), recursive=True)):
        npy = xz[:-3]
        if os.path.exists(npy):
            continue
        info = manifest.get(os.path.relpath(xz, ROOT).replace(os.sep, "/"), {})
        print(f"[data.py] decompressing {xz} ...")
        with lzma.open(xz) as f:
            arr = np.lib.format.read_array(f, allow_pickle=True)
        if "dtype" in info and info["dtype"] != "object":
            arr = arr.astype(info["dtype"])
        with open(npy + ".part", "wb") as f:        # file handle: np.save won't add ".npy"
            np.save(f, arr, allow_pickle=True)
        os.replace(npy + ".part", npy)              # never leave a half-written .npy


_decompress_all()


# ---------------------------------------------------------------------------
# Recursively load every .npy under datasets/  ->  `arrays`
# ---------------------------------------------------------------------------
def _load(path):
    try:
        return np.load(path, mmap_mode="r")          # cheap for big files
    except ValueError:                               # object arrays can't be mmapped
        return np.load(path, allow_pickle=True)


arrays = {}
for f in sorted(glob.glob(os.path.join(ROOT, "**", "*.npy"), recursive=True)):
    key = os.path.relpath(f, ROOT).replace(os.sep, "-")
    arrays[key] = _load(f)


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------
class FlowWindowDataset(Dataset):
    """Sliding windows over a (N, C, H, W) frame array.
    Returns (input, target) shaped (t_in, C, H, W) and (t_out, C, H, W),
    min-max scaled to [0, 1] with the given (lo, hi)."""

    def __init__(self, frames, t_in=4, t_out=4, scale=None):
        self.frames = frames
        self.t_in, self.t_out = t_in, t_out
        lo, hi = scale if scale is not None else (float(frames.min()), float(frames.max()))
        self.lo, self.span = lo, max(hi - lo, 1e-8)
        self.n = max(len(frames) - t_in - t_out + 1, 0)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        w = np.asarray(self.frames[i:i + self.t_in + self.t_out], dtype=np.float32)
        w = torch.from_numpy((w - self.lo) / self.span)
        return w[:self.t_in], w[self.t_in:]


class MovingMNISTDataset(Dataset):
    """Pre-generated Moving MNIST, stored as (T=20, S, 64, 64) uint8.
    Returns (input, target) shaped (10, 1, 64, 64) each, scaled to [0, 1]."""

    def __init__(self, arr, t_in=10):
        self.arr, self.t_in = arr, t_in

    def __len__(self):
        return self.arr.shape[1]

    def __getitem__(self, i):
        seq = np.asarray(self.arr[:, i], dtype=np.float32) / 255.0   # (20, 64, 64)
        seq = torch.from_numpy(seq).unsqueeze(1)                     # (20, 1, 64, 64)
        return seq[:self.t_in], seq[self.t_in:]


class MovingMNISTGenerator(Dataset):
    """Generates fresh bouncing-digit sequences every access (the paper trains on
    10,000 freshly generated sequences and tests on mnist_test_seq.npy)."""

    def __init__(self, digits, length=10000, T=20, t_in=10, size=64, num_digits=2):
        self.digits = digits            # (M, 28, 28) float32 in [0, 1]
        self.length, self.T, self.t_in = length, T, t_in
        self.size, self.num_digits = size, num_digits

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = np.random.default_rng()
        T, S, lim = self.T, self.size, self.size - 28
        seq = np.zeros((T, S, S), dtype=np.float32)
        for _ in range(self.num_digits):
            img = self.digits[rng.integers(len(self.digits))]
            theta = rng.uniform(0, 2 * np.pi)
            v = np.array([np.cos(theta), np.sin(theta)]) * 0.1 * lim
            p = rng.uniform(0, lim, 2)
            for t in range(T):
                x, y = int(p[0]), int(p[1])
                np.maximum(seq[t, y:y + 28, x:x + 28], img, out=seq[t, y:y + 28, x:x + 28])
                p = p + v
                for d in range(2):                       # bounce off walls
                    if p[d] < 0:
                        p[d], v[d] = -p[d], -v[d]
                    elif p[d] > lim:
                        p[d], v[d] = 2 * lim - p[d], -v[d]
        seq = torch.from_numpy(seq).unsqueeze(1)
        return seq[:self.t_in], seq[self.t_in:]


# ---------------------------------------------------------------------------
# Wrap every array as a torch Dataset  ->  `datasets`
# ---------------------------------------------------------------------------
def _is_numeric(arr):
    return arr.dtype != object and np.issubdtype(arr.dtype, np.number)


def _is_flow(key, arr):
    # 4-D numeric check guards against companion files (e.g. timestamps)
    # that happen to share the TaxiBJ2xxx prefix
    return key.startswith(TAXIBJ_PREFIX) and _is_numeric(arr) and arr.ndim == 4


def _taxibj_keys():
    return sorted(k for k, a in arrays.items() if _is_flow(k, a))


def taxibj_scale():
    """Min/max over the training portion of all TaxiBJ flow files."""
    keys = _taxibj_keys()
    lo, hi = np.inf, -np.inf
    for k in keys:
        a = arrays[k][:-TAXIBJ_TEST_FRAMES] if k == keys[-1] else arrays[k]
        lo, hi = min(lo, float(a.min())), max(hi, float(a.max()))
    return lo, hi


_flow_scale = taxibj_scale() if _taxibj_keys() else None

datasets = {}
for key, arr in arrays.items():
    if _is_flow(key, arr):
        datasets[key] = FlowWindowDataset(arr, scale=_flow_scale)
    elif "mnist_test_seq" in key:
        datasets[key] = MovingMNISTDataset(arr)
    elif _is_numeric(arr) and arr.ndim >= 1:
        datasets[key] = TensorDataset(torch.from_numpy(np.array(arr)))
    else:
        print(f"[data.py] skipping {key}: dtype={arr.dtype}, shape={arr.shape}")


# ---------------------------------------------------------------------------
# Train/test splits matching the paper
# ---------------------------------------------------------------------------
def taxibj_splits(t_in=4, t_out=4):
    """Train on everything except the last 4 weeks; test on the last 4 weeks.
    Windows never cross file boundaries."""
    keys = _taxibj_keys()
    train_parts = []
    for k in keys:
        frames = arrays[k][:-TAXIBJ_TEST_FRAMES] if k == keys[-1] else arrays[k]
        train_parts.append(FlowWindowDataset(frames, t_in, t_out, _flow_scale))
    test = FlowWindowDataset(arrays[keys[-1]][-TAXIBJ_TEST_FRAMES:], t_in, t_out, _flow_scale)
    return ConcatDataset(train_parts), test


def moving_mnist_splits(train_size=10000):
    """Train: generated on the fly from MNIST digits. Test: mnist_test_seq.npy."""
    from torchvision.datasets import MNIST
    mnist = MNIST(os.path.join(ROOT, "MNIST_raw"), train=True, download=True)
    digits = mnist.data.numpy().astype(np.float32) / 255.0
    test_key = next(k for k in datasets if "mnist_test_seq" in k)
    return MovingMNISTGenerator(digits, length=train_size), datasets[test_key]


if __name__ == "__main__":
    for k, ds in datasets.items():
        sample = ds[0]
        shapes = [tuple(s.shape) for s in sample]
        print(f"{k:40s} {type(ds).__name__:20s} len={len(ds):6d}  item={shapes}")
