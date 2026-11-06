"""
Run the official ASTGNN code (github.com/guoshnBJTU/ASTGNN, Guo et al., TKDE 2021) on TaxiBJ,
with the same windows, split and metrics as every other model in this project.

ASTGNN is a graph model built for road-sensor data (PEMS). To use it on the TaxiBJ grid:
  - Graph: every grid cell is a node (32 x 32 = 1024 nodes); cells are connected to their
    neighbours (--neighbors 4 = up/down/left/right, 8 = also diagonals).
  - One flow per run: ASTGNN predicts a single feature (it keeps only channel 0), so inflow
    and outflow are two separate runs; `evaluate` puts them back together.
  - Windows: ASTGNN's prepareData.py splits one sequence 60/20/20 and assumes
    points_per_hour == num_for_predict (12 for PEMS). At TaxiBJ's 30-min steps its "recent"
    window would overlap the target, and it would ignore this project's file split. So this
    script skips prepareData.py and writes its output file directly, from data.py's windows:
    identical inputs/targets, in the same order, as the other models' test set. ASTGNN's own
    normalisation is reproduced exactly (min-max of the training inputs to [-1, 1]).
  - ASTGNN needs t_in == t_out (its attention is sized by num_for_predict).
The ASTGNN repo itself is not modified.

1) Clone once:
    git clone https://github.com/guoshnBJTU/ASTGNN.git
    pip install tensorboardX scikit-learn

2) Write the data + configs (run from this project folder):
    python astgnn_taxibj.py prepare --train-files 2013 --val-files 2014 --test-files 2015 2016_1 2016_2

3) Train with their script, one run per flow (it prints these commands):
    cd ASTGNN
    python -u train_ASTGNN.py --config configurations/TaxiBJ_<tag>_inflow.conf --cuda=0
    python -u train_ASTGNN.py --config configurations/TaxiBJ_<tag>_outflow.conf --cuda=0
   (--cuda is the GPU index; their README's --cuda=1 fails on a single-GPU machine.)
   Results go to ../experiments/<dataset_name>/..., i.e. next to the ASTGNN folder.

4) Score both runs together with this project's metrics (tracking.evaluate):
    python astgnn_taxibj.py evaluate --train-files 2013 --val-files 2014 --test-files 2015 2016_1 2016_2
"""

import argparse
import configparser
import glob
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

CHANNELS = ("inflow", "outflow")

# ASTGNN's hyperparameters, from its configurations/PEMS04.conf (the paper's setting).
TRAINING = dict(use_nni=0, batch_size=4, model_name="ASTGNN", num_of_weeks=0, num_of_days=0,
                num_of_hours=1, start_epoch=0, epochs=100, fine_tune_epochs=50,
                learning_rate=0.001, direction=2, encoder_input_size=1, decoder_input_size=1,
                dropout=0.0, kernel_size=3, num_layers=4, d_model=64, nb_head=8, ScaledSAt=1,
                SE=1, smooth_layer_num=0, aware_temporal_context=1, TE=1)


def split_tag(split):
    """Short name for a file split, e.g. tr2013_va2014_te2015-2016_1-2016_2 ('paper' if none)."""
    if not any(split.get(k) for k in ("train_files", "val_files", "test_files")):
        tag = "paper"
    else:
        tag = "_".join(f"{p}{'-'.join(map(str, split[k]))}"
                       for p, k in (("tr", "train_files"), ("va", "val_files"), ("te", "test_files"))
                       if split.get(k))
    for k, p in (("train_size", "ntr"), ("val_size", "nva"), ("test_size", "nte")):
        if split.get(k) is not None:
            tag += f"_{p}{split[k]}"
    return tag


def get_splits(args):
    import data
    split = data.split_kwargs(args)
    train, val, test = data.taxibj_splits(args.tin, args.tout, val=True, **split)
    lo, hi = data.scale_of(train)
    return split, (train, val, test), (lo, hi)


def stack_raw(ds, lo, hi, batch=512):
    """All windows of a split, in order, back in raw units: x (B,T_in,C,H,W), y (B,T_out,C,H,W)."""
    xs, ys = [], []
    for x, y in DataLoader(ds, batch_size=batch, shuffle=False):
        xs.append(x.numpy()); ys.append(y.numpy())
    span = hi - lo
    return (np.concatenate(xs) * span + lo).astype(np.float32), \
           (np.concatenate(ys) * span + lo).astype(np.float32)


def grid_edges(H, W, neighbors):
    """Edges between grid cells (node id = row * W + col), each pair once."""
    steps = [(0, 1), (1, 0)] if neighbors == 4 else [(0, 1), (1, 0), (1, 1), (1, -1)]
    edges = []
    for r in range(H):
        for c in range(W):
            for dr, dc in steps:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    edges.append((r * W + c, rr * W + cc, float(np.hypot(dr, dc))))
    return edges


def to_astgnn(x, y, ch):
    """(B,T_in,C,H,W), (B,T_out,C,H,W) -> ASTGNN's x (B,N,1,T_in) and target (B,N,T_out)."""
    B, Ti, C, H, W = x.shape
    xs = x[:, :, ch].reshape(B, Ti, H * W).transpose(0, 2, 1)[:, :, None, :]
    ys = y[:, :, ch].reshape(B, y.shape[1], H * W).transpose(0, 2, 1)
    return np.ascontiguousarray(xs), np.ascontiguousarray(ys)


def prepare(args):
    if args.tin != args.tout:
        raise SystemExit("ASTGNN needs --tin == --tout (its attention is sized by num_for_predict)")
    root = os.path.abspath(args.astgnn_root)
    if not os.path.exists(os.path.join(root, "train_ASTGNN.py")):
        raise SystemExit(f"ASTGNN repo not found at {root}; "
                         f"git clone https://github.com/guoshnBJTU/ASTGNN.git")
    split, (train, val, test), (lo, hi) = get_splits(args)
    tag = split_tag(split) + ("" if (args.tin, args.tout) == (4, 4) else f"_in{args.tin}_out{args.tout}")
    print(f"windows: {len(train)} train / {len(val)} val / {len(test)} test  (split: {tag})")
    raw = {name: stack_raw(ds, lo, hi) for name, ds in (("train", train), ("val", val), ("test", test))}
    _, _, C, H, W = raw["train"][0].shape
    N = H * W

    commands = []
    for ch, cname in enumerate(CHANNELS[:C]):
        name = f"TaxiBJ_{tag}_{cname}"
        ddir = os.path.join(root, "data", name)
        os.makedirs(ddir, exist_ok=True)
        arrays = {k: to_astgnn(*raw[k], ch) for k in raw}
        # ASTGNN's MinMaxnormalization: max/min of the TRAIN inputs over (B, N, T), per feature,
        # mapped to [-1, 1]. Its file stores max under "mean" and min under "std".
        tx = arrays["train"][0]
        _max = tx.max(axis=(0, 1, 3), keepdims=True)
        _min = tx.min(axis=(0, 1, 3), keepdims=True)
        norm = lambda a: 2. * (a - _min) / (_max - _min) - 1.
        out = {}
        start = 0
        for k in ("train", "val", "test"):
            xk, yk = arrays[k]
            out[f"{k}_x"] = norm(xk).astype(np.float32)
            out[f"{k}_target"] = yk.astype(np.float32)               # raw, as in prepareData.py
            out[f"{k}_timestamp"] = np.arange(start, start + len(xk))[:, None]
            start += len(xk)
        out["mean"], out["std"] = _max, _min
        base = os.path.join(ddir, name)
        np.savez_compressed(base + "_r1_d0_w0.npz", **out)

        adj = os.path.join(ddir, f"{name}.csv")
        with open(adj, "w") as f:
            f.write("from,to,cost\n")
            for i, j, d in grid_edges(H, W, args.neighbors):
                f.write(f"{i},{j},{d}\n")

        cfg = configparser.ConfigParser()
        cfg.optionxform = str                                        # keep ScaledSAt etc. as-is
        cfg["Data"] = dict(adj_filename=f"data/{name}/{name}.csv",
                           graph_signal_matrix_filename=f"data/{name}/{name}.npz",
                           num_of_vertices=N, points_per_hour=args.tin,   # see module docstring
                           num_for_predict=args.tout, len_input=args.tin, dataset_name=name)
        training = dict(TRAINING)
        for k in ("epochs", "fine_tune_epochs", "batch_size"):
            if getattr(args, k) is not None:
                training[k] = getattr(args, k)
        cfg["Training"] = {k: str(v) for k, v in training.items()}
        conf = os.path.join(root, "configurations", f"{name}.conf")
        with open(conf, "w") as f:
            cfg.write(f)
        print(f"{cname}: {os.path.relpath(base, root)}_r1_d0_w0.npz  x {out['train_x'].shape}  "
              f"graph {N} nodes / {len(grid_edges(H, W, args.neighbors))} edges  -> {os.path.relpath(conf, root)}")
        commands.append(f"python -u train_ASTGNN.py --config configurations/{name}.conf --cuda=0")

    print("\nnext, from the ASTGNN folder (one run per flow):")
    print(f"    cd {os.path.relpath(root)}")
    for c in commands:
        print("    " + c)


def latest_output(root, name):
    """ASTGNN's final test predictions: the newest output_epoch_*_test.npz of that run
    (train_ASTGNN.py writes one after training and one after fine-tuning)."""
    exp = os.path.join(os.path.dirname(root), "experiments", name)
    files = glob.glob(os.path.join(exp, "*", "output_epoch_*_test.npz"))
    if not files:
        raise SystemExit(f"no ASTGNN test output under {exp}; has train_ASTGNN.py finished for {name}?")
    return max(files, key=os.path.getmtime)


def evaluate_outputs(args):
    from tracking import evaluate, fmt
    split, (_, _, test), (lo, hi) = get_splits(args)
    tag = split_tag(split) + ("" if (args.tin, args.tout) == (4, 4) else f"_in{args.tin}_out{args.tout}")
    root = os.path.abspath(args.astgnn_root)
    x_raw, y_raw = stack_raw(test, lo, hi)
    B, T, C, H, W = y_raw.shape

    pred = np.zeros_like(y_raw)
    for ch, cname in enumerate(CHANNELS[:C]):
        path = latest_output(root, f"TaxiBJ_{tag}_{cname}")
        o = np.load(path)
        p = o["prediction"][..., 0]                                  # (B, N, T)
        target = o["data_target_tensor"]                             # (B, N, T)
        if p.shape != (B, H * W, T):
            raise SystemExit(f"{path}: prediction {p.shape}, expected {(B, H * W, T)}")
        # sanity check: ASTGNN's targets are exactly this project's test targets
        ours = y_raw[:, :, ch].reshape(B, T, H * W).transpose(0, 2, 1)
        err = np.abs(target - ours).max()
        if err > 1e-3 * max(1.0, np.abs(ours).max()):
            raise SystemExit(f"{path}: targets differ from data.py's test set (max diff {err})")
        pred[:, :, ch] = p.transpose(0, 2, 1).reshape(B, T, H, W)
        print(f"{cname}: {os.path.relpath(path)}")

    # Score with the project's metrics: back to the [0, 1] scale, then evaluate() with an
    # identity "model" over (prediction, target) pairs.
    span = hi - lo
    to01 = lambda a: torch.from_numpy((a - lo) / span).float()
    loader = DataLoader(TensorDataset(to01(pred), to01(y_raw)), batch_size=64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = evaluate(torch.nn.Identity(), loader, device, span, lo, desc="  ASTGNN (test)")
    print(fmt(m, "test"))
    save = args.save or f"checkpoints/astgnn_{tag}.metrics.pt"
    if os.path.dirname(save):
        os.makedirs(os.path.dirname(save), exist_ok=True)
    torch.save({"model": "astgnn", "test_metrics": m, "split": split,
                "t_in": args.tin, "t_out": args.tout}, save)
    keys = ("mse", "sinkhorn", "dist_tv", "volume_err")
    print(f"FINAL astgnn  " + "  ".join(f"{k}={m[k]:.6f}" for k in keys) + "  (test)")
    print(f"saved -> {save}")


def main():
    import data
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["prepare", "evaluate"])
    ap.add_argument("--astgnn-root", default="ASTGNN", help="path to the cloned ASTGNN repo")
    ap.add_argument("--tin", type=int, default=4)
    ap.add_argument("--tout", type=int, default=4)
    ap.add_argument("--neighbors", type=int, choices=[4, 8], default=8,
                    help="grid graph: 4 = up/down/left/right, 8 = also diagonals")
    ap.add_argument("--epochs", type=int, help="override ASTGNN's 100 training epochs")
    ap.add_argument("--fine-tune-epochs", dest="fine_tune_epochs", type=int,
                    help="override ASTGNN's 50 fine-tuning epochs")
    ap.add_argument("--batch-size", dest="batch_size", type=int, help="override ASTGNN's batch size 4")
    ap.add_argument("--save", help="evaluate: where to store the metrics")
    data.add_split_args(ap)
    args = ap.parse_args()
    prepare(args) if args.action == "prepare" else evaluate_outputs(args)


if __name__ == "__main__":
    main()
