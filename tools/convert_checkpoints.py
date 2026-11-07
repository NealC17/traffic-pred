import argparse
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trafficpred.checkpoints import load_model, splits_for
from trafficpred.data import SPLIT_KEYS, get_splits
from trafficpred.metrics import evaluate
from trafficpred.utils import pick_device, save_atomic

CONV_MODELS = ("cnn", "distvol", "stresnet")


def split_from_name(name):
    m = re.search(r"_tr(.+?)_va(.+?)_te(.+?)(?=_n|_seed|$)", name)
    split = {k: None for k in SPLIT_KEYS}
    if m:
        split.update(train_files=m.group(1).split("-"), val_files=m.group(2).split("-"),
                     test_files=m.group(3).split("-"))
    return split


def split_from_config(config, name):
    split = {k: config.get(k) for k in SPLIT_KEYS}
    return split if any(split.values()) else split_from_name(name)


def describe(ck, name):
    if isinstance(ck, dict) and "model_name" in ck:
        return None
    if isinstance(ck, dict) and ck.get("model") == "copy_last":
        return dict(model_name="copy_last", model_kwargs=dict(t_out=ck["t_out"]), t_in=ck["t_in"],
                    t_out=ck["t_out"], loss="none", split=split_from_name(name), state=None, epoch=0,
                    val_metrics=ck.get("val_metrics"), test_metrics=ck.get("test_metrics"))
    if isinstance(ck, dict) and "impl" in ck:
        official = ck["impl"] == "official"
        kwargs = dict(t_in=ck["t_in"], t_out=ck["t_out"], config=ck["config"])
        if official:
            kwargs["variant"] = ck["variant"]
        return dict(model_name="predformer_official" if official else "predformer", model_kwargs=kwargs,
                    t_in=ck["t_in"], t_out=ck["t_out"], loss="mse",
                    split={k: (ck.get("split") or {}).get(k) for k in SPLIT_KEYS},
                    state=ck["model"], epoch=ck["epoch"] + 1, val_metrics=ck.get("val_metrics"),
                    test_metrics=ck.get("test_metrics"))
    if isinstance(ck, dict) and isinstance(ck.get("config"), dict) and ck["config"].get("model") in CONV_MODELS:
        cfg = ck["config"]
        C, H, W = ck["shape"]
        grid = dict(T_in=cfg["tin"], T_out=cfg["tout"], C=C, H=H, W=W)
        kwargs = {"cnn": dict(grid, width=cfg["width"], blocks=cfg["blocks"]),
                  "distvol": dict(grid, width=cfg["width"]),
                  "stresnet": dict(grid, filters=cfg["width"], units=cfg.get("st_units", 12),
                                   bn=bool(cfg.get("st_bn")), init_mean=None)}[cfg["model"]]
        loss_options = (dict(l1=cfg["l1"], l2=cfg["l2"], l3=cfg["l3"], eps=cfg["eps"],
                             sinkhorn_iters=cfg["sinkhorn_iters"]) if cfg["loss"] == "sinkhorn" else {})
        return dict(model_name=cfg["model"], model_kwargs=kwargs, t_in=cfg["tin"], t_out=cfg["tout"],
                    loss=cfg["loss"], loss_options=loss_options, split=split_from_config(cfg, name),
                    seed=cfg.get("seed"), state=ck["model"], epoch=ck["epoch"],
                    val_metrics=ck.get("val_metrics"), test_metrics=ck.get("test_metrics"))
    if isinstance(ck, dict) and ck and all(torch.is_tensor(v) for v in ck.values()) and "net.0.weight" in ck:
        linears = [k for k in ck if k.endswith(".weight") and ck[k].ndim == 2]
        hidden = ck["net.0.weight"].shape[0]
        return dict(model_name="mlp", model_kwargs=dict(t_in=4, t_out=4, channels=2, height=32, width=32,
                                                        hidden=hidden, depth=len(linears), dropout=0.1),
                    t_in=4, t_out=4, loss="mse", split=split_from_name(name), state=ck,
                    epoch=None, val_metrics=None, test_metrics=None)
    raise ValueError("unrecognised checkpoint layout")


def convert(path, out_dir):
    name = os.path.splitext(os.path.basename(path))[0]
    ck = torch.load(path, map_location="cpu", weights_only=False)
    info = describe(ck, name)
    if info is None:
        return None
    if info["model_name"] == "mlp":
        metrics_file = os.path.splitext(path)[0] + ".metrics.pt"
        if os.path.exists(metrics_file):
            m = torch.load(metrics_file, map_location="cpu", weights_only=False)
            info.update(epoch=m.get("epoch"), val_metrics=m.get("val_metrics"),
                        test_metrics=m.get("test_metrics"))
    splits = get_splits("taxibj", info["t_in"], info["t_out"], **info["split"])
    new = dict(model_name=info["model_name"], model_kwargs=info["model_kwargs"], dataset="taxibj",
               t_in=info["t_in"], t_out=info["t_out"], split=splits.split, scale=(splits.lo, splits.hi),
               channels=splits.channels, loss=info["loss"], loss_options=info.get("loss_options", {}),
               seed=info.get("seed"), model=info["state"], epoch=info["epoch"],
               val_metrics=info["val_metrics"], test_metrics=info["test_metrics"], converted_from=path)
    target = os.path.join(out_dir, f"{name}.pt")
    save_atomic(new, target)
    return target


def verify(path):
    model, ck = load_model(path)
    splits = splits_for(ck)
    loader = torch.utils.data.DataLoader(splits.test, batch_size=128)
    m = evaluate(model, loader, pick_device(), splits.span, splits.lo, desc="  verify")
    recorded = (ck.get("test_metrics") or {}).get("mse")
    return m["mse"], recorded


def main():
    ap = argparse.ArgumentParser(description="Rewrite checkpoints from the old training scripts "
                                             "into the format train.py and visualize.py read.")
    ap.add_argument("paths", nargs="+", help="old .pt checkpoints (.last.pt and .metrics.pt are skipped)")
    ap.add_argument("--out", default="checkpoints/taxibj/converted")
    ap.add_argument("--verify", action="store_true",
                    help="re-score each converted checkpoint on its test set and compare to the recorded MSE")
    args = ap.parse_args()
    for path in args.paths:
        if path.endswith((".last.pt", ".metrics.pt")):
            continue
        try:
            target = convert(path, args.out)
        except Exception as e:
            print(f"skip    {path}: {type(e).__name__}: {e}")
            continue
        if target is None:
            print(f"skip    {path}: already in the new format")
            continue
        line = f"wrote   {target}"
        if args.verify:
            mse, recorded = verify(target)
            recorded_text = "n/a" if recorded is None else f"{recorded:.6f}"
            line += f"   test MSE now {mse:.6f}, recorded {recorded_text}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
