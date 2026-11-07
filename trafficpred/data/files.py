import glob
import json
import lzma
import os

import numpy as np

from trafficpred import REPO_ROOT

ROOT = os.path.join(REPO_ROOT, "datasets")
MANIFEST = os.path.join(ROOT, "compressed.json")

_cache = {}


def path(name):
    return os.path.join(ROOT, name)


def available(pattern="**/*.npy"):
    found = set()
    for suffix in ("", ".xz"):
        for f in glob.glob(os.path.join(ROOT, pattern + suffix), recursive=True):
            found.add(os.path.relpath(f, ROOT).replace(os.sep, "/").removesuffix(".xz"))
    return sorted(found)


def load(name):
    if name not in _cache:
        npy = path(name)
        if not os.path.exists(npy):
            _decompress(name)
        try:
            _cache[name] = np.load(npy, mmap_mode="r")
        except ValueError:
            _cache[name] = np.load(npy, allow_pickle=True)
    return _cache[name]


def _decompress(name):
    xz = path(name) + ".xz"
    if not os.path.exists(xz):
        raise FileNotFoundError(f"neither {path(name)} nor {xz} exists")
    manifest = {}
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            manifest = json.load(f)
    dtype = manifest.get(name + ".xz", {}).get("dtype")
    print(f"[data] decompressing {xz}")
    with lzma.open(xz) as f:
        arr = np.lib.format.read_array(f, allow_pickle=True)
    if dtype and dtype != "object":
        arr = arr.astype(dtype)
    part = f"{path(name)}.{os.getpid()}.part"
    with open(part, "wb") as f:
        np.save(f, arr, allow_pickle=True)
    os.replace(part, path(name))
