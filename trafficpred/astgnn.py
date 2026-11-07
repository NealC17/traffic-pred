import configparser
import glob
import os
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from trafficpred import EXTERNAL, REPO_ROOT
from trafficpred.data import DATASETS

DEFAULT_ROOT = os.path.join(EXTERNAL, "ASTGNN")

TRAINING = dict(use_nni=0, batch_size=4, model_name="ASTGNN", num_of_weeks=0, num_of_days=0,
                num_of_hours=1, start_epoch=0, epochs=100, fine_tune_epochs=50,
                learning_rate=0.001, direction=2, encoder_input_size=1, decoder_input_size=1,
                dropout=0.0, kernel_size=3, num_layers=4, d_model=64, nb_head=8, ScaledSAt=1,
                SE=1, smooth_layer_num=0, aware_temporal_context=1, TE=1)


def split_tag(split):
    if not any(split.get(k) for k in ("train_files", "val_files", "test_files")):
        tag = "paper"
    else:
        tag = "_".join(f"{prefix}{'-'.join(map(str, split[k]))}"
                       for prefix, k in (("tr", "train_files"), ("va", "val_files"), ("te", "test_files"))
                       if split.get(k))
    for k, prefix in (("train_size", "ntr"), ("val_size", "nva"), ("test_size", "nte")):
        if split.get(k) is not None:
            tag += f"_{prefix}{split[k]}"
    return tag


def run_tag(splits):
    tag = f"{DATASETS[splits.dataset].title}_{split_tag(splits.split)}"
    if (splits.t_in, splits.t_out) != DATASETS[splits.dataset].horizon:
        tag += f"_in{splits.t_in}_out{splits.t_out}"
    return tag


def channel_runs(splits):
    tag = run_tag(splits)
    return [(c, f"{tag}_{name.lower()}") for c, name in enumerate(splits.channels)]


def stack_raw(ds, lo, hi, batch=512):
    xs, ys = [], []
    for x, y in DataLoader(ds, batch_size=batch, shuffle=False):
        xs.append(x.numpy())
        ys.append(y.numpy())
    span = hi - lo
    return ((np.concatenate(xs) * span + lo).astype(np.float32),
            (np.concatenate(ys) * span + lo).astype(np.float32))


def grid_edges(H, W, neighbors):
    steps = [(0, 1), (1, 0)] if neighbors == 4 else [(0, 1), (1, 0), (1, 1), (1, -1)]
    edges = []
    for r in range(H):
        for c in range(W):
            for dr, dc in steps:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    edges.append((r * W + c, rr * W + cc, float(np.hypot(dr, dc))))
    return edges


def to_astgnn(x, y, channel):
    B, T_in, _, H, W = x.shape
    xs = x[:, :, channel].reshape(B, T_in, H * W).transpose(0, 2, 1)[:, :, None, :]
    ys = y[:, :, channel].reshape(B, y.shape[1], H * W).transpose(0, 2, 1)
    return np.ascontiguousarray(xs), np.ascontiguousarray(ys)


def prepare(splits, root, neighbors=8, overrides=None):
    if splits.t_in != splits.t_out:
        raise SystemExit("ASTGNN needs t_in == t_out (its attention is sized by num_for_predict)")
    if not os.path.exists(os.path.join(root, "train_ASTGNN.py")):
        raise SystemExit(f"ASTGNN repo not found at {root}; "
                         f"git clone https://github.com/guoshnBJTU/ASTGNN.git {root}")
    raw = {name: stack_raw(ds, splits.lo, splits.hi)
           for name, ds in (("train", splits.train), ("val", splits.val), ("test", splits.test))}
    _, H, W = splits.shape
    edges = grid_edges(H, W, neighbors)
    configs = []
    for channel, name in channel_runs(splits):
        _write_data(root, name, {k: to_astgnn(*raw[k], channel) for k in raw})
        _write_graph(root, name, edges)
        conf = _write_config(root, name, H * W, splits.t_in, splits.t_out, overrides or {})
        print(f"{name}: {H * W} nodes, {len(edges)} edges -> {os.path.relpath(conf, root)}")
        configs.append(os.path.relpath(conf, root))
    return configs


def train_command(config, seed=42, python="python"):
    return [python, "-u", "-m", "trafficpred.astgnn_runner", "--config", config, "--seed", str(seed)]


def train(root, configs, seed=42):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [REPO_ROOT, os.environ.get("PYTHONPATH")])))
    for config in configs:
        print(f"=== ASTGNN: {' '.join(train_command(config, seed))}  (in {root})", flush=True)
        subprocess.run(train_command(config, seed, sys.executable), cwd=root, env=env, check=True)


def _write_data(root, name, arrays):
    train_x = arrays["train"][0]
    hi = train_x.max(axis=(0, 1, 3), keepdims=True)
    lo = train_x.min(axis=(0, 1, 3), keepdims=True)
    normalise = lambda a: 2.0 * (a - lo) / np.maximum(hi - lo, 1e-8) - 1.0
    out, start = {}, 0
    for k in ("train", "val", "test"):
        x, y = arrays[k]
        out[f"{k}_x"] = normalise(x).astype(np.float32)
        out[f"{k}_target"] = y.astype(np.float32)
        out[f"{k}_timestamp"] = np.arange(start, start + len(x))[:, None]
        start += len(x)
    out["mean"], out["std"] = hi, lo
    directory = os.path.join(root, "data", name)
    os.makedirs(directory, exist_ok=True)
    np.savez_compressed(os.path.join(directory, f"{name}_r1_d0_w0.npz"), **out)


def _write_graph(root, name, edges):
    with open(os.path.join(root, "data", name, f"{name}.csv"), "w") as f:
        f.write("from,to,cost\n")
        for i, j, d in edges:
            f.write(f"{i},{j},{d}\n")


def _write_config(root, name, nodes, t_in, t_out, overrides):
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg["Data"] = dict(adj_filename=f"data/{name}/{name}.csv",
                       graph_signal_matrix_filename=f"data/{name}/{name}.npz",
                       num_of_vertices=nodes, points_per_hour=t_in, num_for_predict=t_out,
                       len_input=t_in, dataset_name=name)
    cfg["Training"] = {k: str(v) for k, v in {**TRAINING, **overrides}.items()}
    path = os.path.join(root, "configurations", f"{name}.conf")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        cfg.write(f)
    return path


def latest_output(root, name, split="test"):
    experiments = os.path.join(os.path.dirname(os.path.abspath(root)), "experiments", name)
    found = glob.glob(os.path.join(experiments, "*", f"output_epoch_*_{split}.npz"))
    if not found:
        raise SystemExit(f"no ASTGNN {split} output under {experiments}; "
                         f"has the ASTGNN run finished for {name}?")
    return max(found, key=os.path.getmtime)


def collect_predictions(splits, root, split="test"):
    _, y_raw = stack_raw(getattr(splits, split), splits.lo, splits.hi)
    B, T, C, H, W = y_raw.shape
    pred = np.zeros_like(y_raw)
    for channel, name in channel_runs(splits):
        path = latest_output(root, name, split)
        out = np.load(path)
        p, target = out["prediction"][..., 0], out["data_target_tensor"]
        if p.shape != (B, H * W, T):
            raise SystemExit(f"{path}: prediction {p.shape}, expected {(B, H * W, T)}")
        ours = y_raw[:, :, channel].reshape(B, T, H * W).transpose(0, 2, 1)
        diff = np.abs(target - ours).max()
        if diff > 1e-3 * max(1.0, np.abs(ours).max()):
            raise SystemExit(f"{path}: targets differ from this project's {split} set (max diff {diff})")
        pred[:, :, channel] = p.transpose(0, 2, 1).reshape(B, T, H, W)
        print(f"{name}: {os.path.relpath(path)}")
    return pred, y_raw


def prediction_loader(pred, y_raw, lo, hi, batch_size=64):
    to_unit = lambda a: torch.from_numpy((a - lo) / (hi - lo)).float()
    return DataLoader(TensorDataset(to_unit(pred), to_unit(y_raw)), batch_size=batch_size)
