"""Shared random seeding for the training scripts.

seed_everything(seed) seeds Python's `random`, NumPy's global generator and PyTorch
(CPU and every CUDA device). That fixes the initial weights, the shuffle order, dropout
masks and DataLoader worker seeds (PyTorch derives those from its own generator).

Same seed -> same starting point and same batch order. Results can still differ in the
last few digits between runs on a GPU, because cuDNN autotuning and some CUDA kernels
are not bit-for-bit deterministic; that noise is far smaller than the seed-to-seed spread.
"""
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)          # NumPy's legacy seed must fit in 32 bits
    torch.manual_seed(seed)               # also seeds every CUDA device
