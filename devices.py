"""
One place to pick the training device: CUDA (NVIDIA) -> MPS (Apple Silicon) -> CPU.

    from devices import pick_device
    device = pick_device()

PYTORCH_ENABLE_MPS_FALLBACK=1 lets the few ops MPS doesn't implement run on the CPU
instead of raising an error. It only takes effect if set before torch is imported,
so it's set here (when this module is imported first) and in train_all.sh.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402


def pick_device(verbose=False):
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    if verbose:
        print(f"device: {dev}")
    return dev
