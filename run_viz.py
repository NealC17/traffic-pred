"""
Load a trained PredFormer checkpoint and plot predicted vs. actual
inflow/outflow heatmaps with viz_flow.

    python run_viz.py                                   # checkpoints/taxibj/best.pt
    python run_viz.py --idx 500 --step 3 --save figs/
    python run_viz.py --browse                          # click through all test windows
    python run_viz.py --ckpt checkpoints/taxibj_official/best.pt   # official-model checkpoint
"""

import argparse
import os

import torch

from predformer import PredFormer
from data import taxibj_splits
from viz_flow import browse, plot_prediction, plot_horizon


def load_model(path, device=None):
    from devices import pick_device
    device = device or pick_device()
    ck = torch.load(path, map_location=device)
    # official-repo configs use keys like 'height'/'Ndepth' instead of 'in_shape'
    if ck.get("impl") == "official" or "Ndepth" in ck["config"]:
        from official_predformer import build_official
        model = build_official(variant=ck["variant"], config=ck["config"]).to(device)
    else:
        model = PredFormer(**ck["config"]).to(device)
    model.load_state_dict(ck["model"])
    t_in, t_out = ck.get("t_in", 4), ck.get("t_out", 4)     # older checkpoints: 4 -> 4
    if t_out != t_in:
        from train import Horizon
        model = Horizon(model, t_in, t_out)
    model.t_in, model.t_out = t_in, t_out      # viz_flow uses these to build the test set
    model.eval()
    return model, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/taxibj/best.pt")
    ap.add_argument("--idx", type=int, default=100, help="test window index")
    ap.add_argument("--step", type=int, default=0, help="future step for the 2x3 grid (0 = t+1)")
    ap.add_argument("--save", help="directory to save PNGs into")
    ap.add_argument("--browse", action="store_true",
                    help="interactive viewer with Prev/Next buttons over every test window")
    args = ap.parse_args()

    model, ck = load_model(args.ckpt)
    if ck["dataset"] != "taxibj":
        raise SystemExit(f"viz_flow plots TaxiBJ inflow/outflow; this checkpoint is '{ck['dataset']}'")
    print(f"loaded {args.ckpt} (epoch {ck['epoch'] + 1}, "
          f"test MSE {ck.get('test_mse', float('nan')):.4f})")

    _, test_set = taxibj_splits(model.t_in, model.t_out)
    if args.browse:
        browse(model, test_set, start=args.idx, step=args.step)
        return
    if args.save:
        os.makedirs(args.save, exist_ok=True)

    def out(name):
        return os.path.join(args.save, name) if args.save else None

    plot_prediction(model, test_set, idx=args.idx, step=args.step,
                    save=out(f"pred_idx{args.idx}_t{args.step + 1}.png"))
    for c, name in enumerate(["inflow", "outflow"]):
        plot_horizon(model, test_set, idx=args.idx, channel=c,
                     save=out(f"horizon_idx{args.idx}_{name}.png"))


if __name__ == "__main__":
    main()