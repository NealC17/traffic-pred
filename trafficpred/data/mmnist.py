import os
import socket
import time
import urllib.error
import urllib.request

import numpy as np

from trafficpred.data import files

URL = "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"
TEST_FILE = "MovingMNIST/mnist_test_seq.npy"
DIGITS_DIR = files.path("MNIST_raw")


def download(timeout=60, retries=3):
    dest = files.path(TEST_FILE)
    if not os.path.exists(dest):
        _fetch(URL, dest, timeout, retries)
    from torchvision.datasets import MNIST
    socket.setdefaulttimeout(timeout)
    try:
        MNIST(DIGITS_DIR, train=True, download=True)
    finally:
        socket.setdefaulttimeout(None)
    return dest


def digits():
    from torchvision.datasets import MNIST
    mnist = MNIST(DIGITS_DIR, train=True, download=False)
    return mnist.data.numpy().astype(np.float32) / 255.0


def _fetch(url, dest, timeout, retries):
    from tqdm import tqdm
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r, open(part, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0) or None
                with tqdm(total=total, unit="B", unit_scale=True, desc="Moving MNIST",
                          dynamic_ncols=True, mininterval=1) as bar:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                        bar.update(len(chunk))
            if total and os.path.getsize(part) != total:
                raise IOError(f"incomplete download ({os.path.getsize(part)} of {total} bytes)")
            os.replace(part, dest)
            return
        except (OSError, urllib.error.URLError, socket.timeout) as e:
            print(f"[data] download attempt {attempt}/{retries} failed: {e}")
            if attempt == retries:
                if os.path.exists(part):
                    os.remove(part)
                raise RuntimeError(
                    f"Couldn't download {url}\n"
                    f"On a machine without internet (e.g. a cluster compute node), run\n"
                    f'    python -c "from trafficpred.data import mmnist; mmnist.download()"\n'
                    f"on one that has it (a login node), or copy the file to\n    {dest}") from e
            time.sleep(5 * attempt)
