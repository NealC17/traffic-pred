import argparse
import contextlib
import importlib
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from trafficpred import EXTERNAL

ROOT = os.environ.get("STDCFORMER_ROOT", os.path.join(EXTERNAL, "STDCformer"))


def _official():
    root = os.path.abspath(ROOT)
    if not os.path.exists(os.path.join(root, "models", "STDC.py")):
        raise FileNotFoundError(
            f"Official STDCformer repo not found at {root}. Run\n"
            f"  git clone https://github.com/GeoX-Lab/STDCformer.git {root}\n"
            f"or set STDCFORMER_ROOT to where you cloned it.")
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module("models.STDC"), importlib.import_module("utils.utils")


def grid_adjacency(h, w, neighbors=8):
    steps = [(0, 1), (1, 0)] + ([(1, 1), (1, -1)] if neighbors == 8 else [])
    A = np.zeros((h * w, h * w), np.float32)
    for r in range(h):
        for c in range(w):
            for dr, dc in steps:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    A[r * w + c, rr * w + cc] = A[rr * w + cc, r * w + c] = 1.0
    return A


@contextlib.contextmanager
def frozen_batch_norm(module):
    norms = [m for m in module.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    saved = [m.momentum for m in norms]
    for m in norms:
        m.momentum = 0.0
    try:
        yield
    finally:
        for m, momentum in zip(norms, saved):
            m.momentum = momentum


def checkpoint_blocks(blocks):
    for block in blocks:
        forward = block.forward
        contexts = lambda block=block: (contextlib.nullcontext(), frozen_batch_norm(block))

        def checkpointed(*args, forward=forward, block=block, contexts=contexts):
            if block.training and torch.is_grad_enabled():
                return checkpoint(forward, *args, use_reentrant=False, context_fn=contexts)
            return forward(*args)

        block.forward = checkpointed


class STDCformer(nn.Module):
    def __init__(self, T_in=4, T_out=4, C=2, H=32, W=32, D=64, layers=5, K=8, lape_dim=8, neighbors=8,
                 mean=0.0, std=1.0, spatial_features=None, gradient_checkpointing=True):
        super().__init__()
        if T_in != T_out:
            raise ValueError("the official STDCformer needs t_in == t_out")
        stdc, utils = _official()
        spatial = (np.zeros((H * W, 0), np.float32) if spatial_features is None
                   else np.asarray(spatial_features, np.float32))
        args = argparse.Namespace(
            device="cpu", data_dir="", dataset="MHT", mode="train", test_dataset="MHT",
            d_input=C, d_output=C, scaler="standard", d_spatial=spatial.shape[1], d_temporal=2 * T_in,
            lape_dim=lape_dim, input_length=T_in, output_length=T_out, D=D, add_day_in_week=False,
            batch_size=1, layers=layers, K=K)
        self.net = stdc.STDC(args)
        self.net.num_nodes = H * W
        if gradient_checkpointing:
            checkpoint_blocks([*self.net.encoder, *self.net.decoder])
        self.T_in, self.T_out, self.C, self.H, self.W = T_in, T_out, C, H, W
        self.mean, self.std = mean, std
        lap = utils.calculate_laplacian(grid_adjacency(H, W, neighbors), lape_dim)
        self.register_buffer("spatial", torch.from_numpy(spatial), persistent=False)
        self.register_buffer("laplacian", torch.as_tensor(lap, dtype=torch.float32), persistent=False)
        self.register_buffer("temporal", torch.eye(2 * T_in), persistent=False)

    def _follow_device(self, device):
        for m in self.net.modules():
            if isinstance(getattr(m, "device", None), (torch.device, str)):
                m.device = device

    def forward(self, x):
        B, T, C, H, W = x.shape
        self._follow_device(x.device)
        self.net.batch_size = B
        nodes = (x.permute(0, 1, 3, 4, 2).reshape(B, T, H * W, C) - self.mean) / self.std
        nodes = torch.cat([nodes, nodes.new_zeros(B, T, H * W, 1)], dim=-1)
        window = np.tile(np.arange(T), (B, 1))
        inputs = nodes.detach().float().cpu().numpy()
        batch = (inputs, inputs, None, None, window)
        out = self.net(batch, self.spatial, self.laplacian, self.temporal) * self.std + self.mean
        return out.permute(0, 2, 3, 1).reshape(B, self.T_out, C, H, W)
