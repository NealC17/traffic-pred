#!/usr/bin/env python3
"""
Count trainable parameters for every checkpoint in checkpoints/ and write
an easy-to-read table to logs/param_sizes.txt (also printed to the terminal).

Usage (from the project root):
    python param_sizes.py
    python param_sizes.py --ckpt-dir checkpoints --out logs/param_sizes.txt
"""
import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import torch

# Let torch.load unpickle whole-model checkpoints that reference project classes.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Non-trainable entries that show up in a state_dict but aren't parameters.
BUFFER_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")
# Keys different trainers use to store the weights inside a checkpoint dict.
STATE_KEYS = ("model", "state_dict", "model_state_dict", "model_state", "net", "weights")
LOSSES = ("mse", "sinkhorn", "l1", "mae")


def load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # Older/whole-model checkpoints need full unpickling (your own files).
        return torch.load(path, map_location="cpu", weights_only=False)


def find_weights(obj):
    """Return an nn.Module or a {name: tensor} state_dict from any checkpoint layout."""
    if isinstance(obj, torch.nn.Module):
        return obj
    if isinstance(obj, dict):
        for k in STATE_KEYS:
            if k in obj and isinstance(obj[k], (dict, torch.nn.Module)):
                found = find_weights(obj[k])
                if found is not None:
                    return found
        tensors = {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
        if tensors and len(tensors) == len(obj):
            return tensors
        if not obj:
            return {}
    return None


def count_params(weights):
    if isinstance(weights, torch.nn.Module):
        return sum(p.numel() for p in weights.parameters() if p.requires_grad)
    seen, total = set(), 0
    for name, t in weights.items():
        if name.endswith(BUFFER_SUFFIXES) or not t.is_floating_point():
            continue
        key = (t.data_ptr(), t.numel(), tuple(t.shape))
        if t.numel() and key in seen:  # tied/shared weights counted once
            continue
        seen.add(key)
        total += t.numel()
    return total


def parse_name(run):
    """'cnn_sinkhorn_tr2013_va2014_...' -> ('cnn', 'sinkhorn', 'tr2013_va2014_...')"""
    m = re.match(r"(.+?)_(tr\d.*)$", run)
    prefix, split = (m.group(1), m.group(2)) if m else (run, "")
    parts = prefix.split("_")
    if len(parts) > 1 and parts[-1] in LOSSES:
        return "_".join(parts[:-1]), parts[-1], split
    return prefix, "-", split


def human(n):
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--out", default="logs/param_sizes.txt")
    ap.add_argument("--ref", default="distvol", help="model to compare sizes against")
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    # One file per run: prefer the best .pt, fall back to .last.pt. Skip metrics files.
    runs = {}
    for p in sorted(ckpt_dir.glob("*.pt")):
        name = p.name
        if name.endswith(".metrics.pt"):
            continue
        if name.endswith(".last.pt"):
            runs.setdefault(name[: -len(".last.pt")], p)
        else:
            runs[name[: -len(".pt")]] = p

    rows, errors = [], []
    for run, path in runs.items():
        model, loss, split = parse_name(run)
        try:
            weights = find_weights(load(path))
            if weights is None:
                raise ValueError("couldn't find weights in checkpoint")
            rows.append((model, loss, split, count_params(weights), path.name))
        except Exception as e:
            errors.append((run, f"{type(e).__name__}: {e}"))

    rows.sort(key=lambda r: (-r[3], r[0], r[1]))
    ref = next((r[3] for r in rows if r[0] == args.ref and r[3] > 0), None)
    splits = sorted({r[2] for r in rows})
    show_split = len(splits) > 1

    header = ["model", "loss", "params", "size", f"vs {args.ref}"] + (["split"] if show_split else [])
    table = []
    for model, loss, split, n, _ in rows:
        ratio = f"{n / ref:.2f}x" if ref else "-"
        size = human(n) if n else "0 (no weights)"
        table.append([model, loss, f"{n:,}", size, ratio] + ([split] if show_split else []))

    widths = [max(len(h), *(len(r[i]) for r in table)) if table else len(h) for i, h in enumerate(header)]
    right = {2, 3, 4}  # right-align numbers

    def fmt(cells):
        return "  ".join(c.rjust(w) if i in right else c.ljust(w)
                         for i, (c, w) in enumerate(zip(cells, widths))).rstrip()

    lines = [
        f"Trainable parameter counts — {ckpt_dir}/",
        f"Generated {datetime.now():%Y-%m-%d %H:%M}",
    ]
    if not show_split and splits and splits[0]:
        lines.append(f"Split: {splits[0]}")
    lines += ["", fmt(header), fmt(["-" * w for w in widths])] + [fmt(r) for r in table]
    if errors:
        lines += ["", "Could not read:"] + [f"  {run}: {err}" for run, err in errors]
    text = "\n".join(lines) + "\n"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text, end="")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
