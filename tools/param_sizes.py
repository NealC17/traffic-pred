import argparse
from datetime import datetime
from pathlib import Path

import torch

BUFFER_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")


def load_state(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and isinstance(ck.get("model"), dict):
        return ck["model"]
    if isinstance(ck, dict) and ck and all(torch.is_tensor(v) for v in ck.values()):
        return ck
    return {}


def count(state):
    seen, total = set(), 0
    for name, t in state.items():
        if name.endswith(BUFFER_SUFFIXES) or not t.is_floating_point():
            continue
        key = (t.data_ptr(), t.numel(), tuple(t.shape))
        if t.numel() and key in seen:
            continue
        seen.add(key)
        total += t.numel()
    return total


def human(n):
    return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n / 1e3:.1f}K" if n >= 1e3 else str(n)


def table(rows, ref_name):
    ref = next((n for run, _, n in rows if run.startswith(ref_name) and n), None)
    header = ["run", "params", "size", f"vs {ref_name}"]
    body = [[run, f"{n:,}", human(n) if n else "0 (no weights)", f"{n / ref:.2f}x" if ref else "-"]
            for run, _, n in rows]
    widths = [max(len(r[i]) for r in [header] + body) for i in range(len(header))]
    line = lambda cells: "  ".join(c.ljust(w) if i == 0 else c.rjust(w)
                                   for i, (c, w) in enumerate(zip(cells, widths))).rstrip()
    return [line(header), line(["-" * w for w in widths])] + [line(r) for r in body]


def main():
    ap = argparse.ArgumentParser(description="Count trainable parameters of every checkpoint.")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--out", default="logs/param_sizes.txt")
    ap.add_argument("--ref", default="distvol", help="run name prefix to compare sizes against")
    args = ap.parse_args()

    root = Path(args.ckpt_dir)
    rows, errors = [], []
    for path in sorted(root.rglob("*.pt")):
        if path.name.endswith((".last.pt", ".metrics.pt")):
            continue
        run = str(path.relative_to(root).with_suffix(""))
        try:
            rows.append((run, path, count(load_state(path))))
        except Exception as e:
            errors.append((run, f"{type(e).__name__}: {e}"))
    rows.sort(key=lambda r: (-r[2], r[0]))

    lines = [f"Trainable parameter counts - {root}/", f"Generated {datetime.now():%Y-%m-%d %H:%M}", ""]
    lines += table(rows, args.ref)
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
