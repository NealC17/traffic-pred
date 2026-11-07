import argparse
import os

from trafficpred.checkpoints import load_model, splits_for
from trafficpred.viz import browse, plot_horizon, plot_prediction


def main():
    ap = argparse.ArgumentParser(description="Plot a trained checkpoint's predictions on its test set.")
    ap.add_argument("ckpt", help="a checkpoint written by train.py")
    ap.add_argument("--idx", type=int, default=100, help="test window index")
    ap.add_argument("--step", type=int, default=0, help="future step for the grid (0 = t+1)")
    ap.add_argument("--save", help="directory to save PNGs into")
    ap.add_argument("--browse", action="store_true", help="interactive viewer over every test window")
    args = ap.parse_args()

    model, ck = load_model(args.ckpt)
    splits = splits_for(ck)
    test_mse = (ck.get("test_metrics") or {}).get("mse", float("nan"))
    print(f"loaded {args.ckpt}: {ck['model_name']} on {ck['dataset']} (epoch {ck['epoch']}, test MSE {test_mse:.5f})")
    idx = min(args.idx, len(splits.test) - 1)
    if args.browse:
        browse(model, splits, start=idx, step=args.step)
        return
    if args.save:
        os.makedirs(args.save, exist_ok=True)
    path = lambda name: os.path.join(args.save, name) if args.save else None
    plot_prediction(model, splits, idx=idx, step=args.step, save=path(f"pred_idx{idx}_t{args.step + 1}.png"))
    for c, name in enumerate(splits.channels):
        plot_horizon(model, splits, idx=idx, channel=c, save=path(f"horizon_idx{idx}_{name.lower()}.png"))


if __name__ == "__main__":
    main()
