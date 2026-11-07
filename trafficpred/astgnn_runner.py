import argparse
import glob
import os
import re
import runpy
import sys

from trafficpred.utils import seed_everything


def best_epoch(params_path):
    found = glob.glob(os.path.join(params_path, "output_epoch_*_test.npz"))
    if not found:
        raise SystemExit(f"ASTGNN wrote no test output under {params_path}")
    newest = max(found, key=os.path.getmtime)
    return int(re.search(r"output_epoch_(\d+)_test\.npz$", newest).group(1))


def main():
    ap = argparse.ArgumentParser(description="Run ASTGNN's unmodified train_ASTGNN.py with a fixed seed, "
                                             "then save its predictions on the train and val sets too.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    sys.argv = ["train_ASTGNN.py", "--config", args.config, "--cuda=0"]
    script = runpy.run_path("train_ASTGNN.py", run_name="astgnn")
    script["train_main"]()

    epoch = best_epoch(script["params_path"])
    loaders = script["load_graphdata_normY_channel1"](
        script["graph_signal_matrix_filename"], script["num_of_hours"], script["num_of_days"],
        script["num_of_weeks"], script["DEVICE"], script["batch_size"], shuffle=False)
    train_loader, train_target, val_loader, val_target, _, _, data_max, data_min = loaders
    for loader, target, split in ((train_loader, train_target, "train"), (val_loader, val_target, "val")):
        script["predict_main"](epoch, loader, target, data_max, data_min, split)


if __name__ == "__main__":
    main()
