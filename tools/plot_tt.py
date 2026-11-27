"""Look inside a trained temporal latent transport (TT, --time-transport): for every temporal block, the mean
transport plan per head (which input frames feed which frames, averaged over the val windows and sensors), the
learned lag topology (softplus(w) ||e_i - e_j|| between frame slots) and the learned temperatures eps (and rho,
with --tt-unbalanced).

  python tools/plot_tt.py checkpoints/workspaces/pems_pems08/distvolformer_direct_small_extin_tt5_ae_z_pems08_in12_out12.pt
  python tools/plot_tt.py <ckpt.pt> --batches 50 --out figs_tt

Writes <out>/<run>_plans.png and <out>/<run>_topology.png and prints the per-head eps / rho / topology weight.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # Apple GPUs: run any op MPS lacks on the CPU
import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trafficpred.checkpoints import load_model, splits_for          # noqa: E402
from trafficpred.models.distvolformer import LatentTransport       # noqa: E402
from trafficpred.utils import pick_device, predict                  # noqa: E402


def temporal_transports(model):
    """The temporal blocks' LatentTransport modules (they have n_src set), in block order."""
    return [(name, m) for name, m in model.named_modules() if isinstance(m, LatentTransport) and m.n_src]


def topology_distances(lt):
    """(heads, F, F) learned lag distances softplus(w) ||e_i - e_j||, averaged over sensors for a per-sensor one."""
    if lt.topo is None:
        return None
    w = F.softplus(lt.topo_w)[:, None, None]
    d = w * lt._dist(lt.topo, lt.topo)
    return (d.mean(0) if lt.groups > 1 else d).detach().cpu().numpy()


def frame_labels(S, F_, extra_lags):
    """Source / destination names: the recent inputs t-(k-1) .. t0, any period frames, the targets t+1 .."""
    recent = S - len(extra_lags)
    ins = [f"lag{lag}" for lag in extra_lags] + [f"t{i - recent + 1}" if i < recent - 1 else "t0"
                                                 for i in range(recent)]
    outs = [f"t+{j + 1}" for j in range(F_ - S)]
    return ins, ins + outs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoint")
    ap.add_argument("--split", default="val", choices=("train", "val", "test"))
    ap.add_argument("--batches", type=int, default=30, help="batches of the split to average the plans over")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="figs_tt")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = pick_device()
    model, ck = load_model(args.checkpoint, device)
    lts = temporal_transports(model)
    if not lts:
        raise SystemExit(f"{args.checkpoint} has no temporal latent transport (--time-transport)")
    run = os.path.basename(args.checkpoint)[:-3]
    os.makedirs(args.out, exist_ok=True)
    splits = splits_for(ck)
    for _, lt in lts:
        lt.keep_plan = True
    sums = [None] * len(lts)
    seen = 0
    loader = torch.utils.data.DataLoader(getattr(splits, args.split), batch_size=args.batch_size)
    total = min(args.batches, len(loader))
    with torch.no_grad():
        for k, batch in enumerate(tqdm(loader, total=total, desc=f"plans ({args.split})")):
            if k >= total:
                break
            x = batch[0].to(device)
            ext = batch[2].to(device) if len(batch) > 2 else None
            predict(model, x, ext)
            for i, (_, lt) in enumerate(lts):
                p = lt.last_plan.float().sum(0).cpu()                              # (heads, F, S)
                sums[i] = p if sums[i] is None else sums[i] + p
            seen += lts[0][1].last_plan.shape[0]
    plans = [(s / seen).numpy() for s in sums]
    heads, F_, S = plans[0].shape
    src_lab, dst_lab = frame_labels(S, F_, tuple(splits.extra_lags))

    print(f"{run}: {len(lts)} temporal TT blocks, {heads} heads, {S} sources -> {F_} frames, "
          f"averaged over {seen} sequences of the {args.split} split")
    for i, (name, lt) in enumerate(lts):
        eps = lt.log_eps.exp().detach().cpu().numpy()
        line = f"  block {i} ({name}): eps " + " ".join(f"{e:.3f}" for e in eps)
        if lt.topo is not None:
            line += " | topology weight " + " ".join(f"{w:.3f}" for w in F.softplus(lt.topo_w).detach().cpu().numpy())
        if lt.log_rho is not None:
            line += " | rho " + " ".join(f"{r:.3f}" for r in lt.log_rho.exp().detach().cpu().numpy())
        col = plans[i].sum(1)                                                      # (heads, S) mass sent
        line += f" | mass sent per input frame (target {F_ / S:.2f}): min {col.min():.2f} max {col.max():.2f}"
        print(line)

    step = max(1, len(dst_lab) // 12)
    fig, axes = plt.subplots(len(lts), heads, figsize=(2.6 * heads + 1, 3.6 * len(lts)), squeeze=False)
    for i in range(len(lts)):
        for h in range(heads):
            ax = axes[i][h]
            im = ax.imshow(plans[i][h], aspect="auto", cmap="viridis")
            ax.set_title(f"block {i}, head {h}", fontsize=9)
            ax.set_xticks(range(0, S, max(1, S // 6)))
            ax.set_xticklabels(src_lab[::max(1, S // 6)], fontsize=6, rotation=90)
            ax.set_yticks(range(0, F_, step))
            ax.set_yticklabels(dst_lab[::step], fontsize=6)
            ax.axhline(S - 0.5, color="w", lw=0.6)
            fig.colorbar(im, ax=ax, fraction=0.046)
        axes[i][0].set_ylabel("destination frame")
    for h in range(heads):
        axes[-1][h].set_xlabel("source (input) frame")
    fig.suptitle(f"{run}\nmean TT plan P[dest, source] on {args.split}; white line: inputs above, targets below", fontsize=9)
    fig.tight_layout()
    path = os.path.join(args.out, f"{run}_plans.png")
    fig.savefig(path, dpi=140)
    print("wrote", path)

    topo = [topology_distances(lt) for _, lt in lts]
    if any(t is not None for t in topo):
        fig, axes = plt.subplots(len(lts), heads, figsize=(2.6 * heads + 1, 3.0 * len(lts)), squeeze=False)
        for i, t in enumerate(topo):
            for h in range(heads):
                ax = axes[i][h]
                if t is None:
                    ax.axis("off")
                    continue
                im = ax.imshow(t[h], cmap="magma")
                ax.set_title(f"block {i}, head {h}", fontsize=9)
                ax.set_xticks(range(0, F_, step))
                ax.set_xticklabels(dst_lab[::step], fontsize=6, rotation=90)
                ax.set_yticks(range(0, F_, step))
                ax.set_yticklabels(dst_lab[::step], fontsize=6)
                fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"{run}\nlearned lag topology cost softplus(w) ||e_i - e_j||", fontsize=9)
        fig.tight_layout()
        path = os.path.join(args.out, f"{run}_topology.png")
        fig.savefig(path, dpi=140)
        print("wrote", path)


if __name__ == "__main__":
    main()
