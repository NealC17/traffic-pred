"""Download PEMS04 / PEMS08 and build this repo's files from them, e.g. on a cluster checkout (the .npy files
are gitignored). Fetches the STSGCN release from Hugging Face (Davidham3/pems-stsgcn, Apache-2.0), checks each
file's SHA-256 against the copies the paper results here were made with, then converts with
tools/import_pems.py and checks the resulting flow.npy / hops.npy byte for byte.

  python tools/download_pems.py              # PEMS08 and PEMS04
  python tools/download_pems.py PEMS08       # just one

Skips files that are already present with the right hash. Needs pyarrow (pip install pyarrow).
"""
import argparse
import hashlib
import os
import subprocess
import sys
import urllib.request
from functools import partial

print = partial(print, flush=True)   # keep the order with the importer's output

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "https://huggingface.co/datasets/Davidham3/pems-stsgcn/resolve/main"
SHA256 = {
    "PEMS08": {"PEMS08_series.parquet": "7d9143f9271f1819217de56940959e8b14a23c23824dc6cb7cc4ef641976b1b8",
               "PEMS08_edges.parquet": "942d54fad5c7c0a577b8ec28f804fcb3bc9dfae3b7d57bffacb493b3befbfd99",
               "flow.npy": "ef0dc6dac78e42fa4ab92d73d3f0b54280931a8413dde95891a77e72eac20a67",
               "hops.npy": "aa4de028f110d1fabf02e8b5a641fde20aa1fa75cd25ba7810bea3b5e36437a7"},
    "PEMS04": {"PEMS04_series.parquet": "39de0b3d83ece4543c9bd500e692dbe983cce5e6c1648dd7895b27222de2413d",
               "PEMS04_edges.parquet": "6f5dff5f034b9f3c18fecfa4f0f28eef88209d9d861ffc06c01aa960095a1449",
               "flow.npy": "02a6b79def62780a1da0fffd292166f6f387098059639d04a686f0fecb6e0a7a",
               "hops.npy": "7a1c5efec77960b04285624d6833700de3970258e3d588909fa9b653502a1b0b"},
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url, path, want):
    if os.path.exists(path) and sha256(path) == want:
        print(f"  have {os.path.relpath(path, REPO)}")
        return
    print(f"  downloading {url}")
    part = path + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "traffic-pred-downloader"})
    with urllib.request.urlopen(req, timeout=120) as r, open(part, "wb") as f:
        while block := r.read(1 << 20):
            f.write(block)
    got = sha256(part)
    if got != want:
        os.remove(part)
        raise SystemExit(f"!!! {os.path.basename(path)}: SHA-256 {got}, expected {want}; the upload changed, "
                         "so results would not be comparable. Not using it.")
    os.replace(part, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("datasets", nargs="*", help="PEMS08 and / or PEMS04 (default: both)")
    args = ap.parse_args()
    args.datasets = args.datasets or ["PEMS08", "PEMS04"]
    if set(args.datasets) - set(SHA256):
        ap.error(f"unknown {sorted(set(args.datasets) - set(SHA256))}; choose from {sorted(SHA256)}")
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        raise SystemExit("needs pyarrow: pip install pyarrow")
    for ds in args.datasets:
        print(ds)
        folder = os.path.join(REPO, "datasets", "PEMS", ds)
        os.makedirs(folder, exist_ok=True)
        for name in (f"{ds}_series.parquet", f"{ds}_edges.parquet"):
            fetch(f"{BASE}/{ds}/{name}", os.path.join(folder, name), SHA256[ds][name])
        built = [os.path.join(folder, n) for n in ("flow.npy", "hops.npy")]
        if not all(os.path.exists(p) and sha256(p) == SHA256[ds][os.path.basename(p)] for p in built):
            subprocess.run([sys.executable, "-I", os.path.join(REPO, "tools", "import_pems.py"), ds], check=True)
        for p in built:
            if sha256(p) != SHA256[ds][os.path.basename(p)]:
                raise SystemExit(f"!!! {os.path.relpath(p, REPO)} differs from the reference copy")
        print(f"  ok: {os.path.relpath(folder, REPO)}/flow.npy and hops.npy match the reference")


if __name__ == "__main__":
    main()
