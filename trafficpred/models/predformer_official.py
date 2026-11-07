import importlib
import os
import runpy
import sys

from trafficpred import EXTERNAL

ROOT = os.environ.get("PREDFORMER_ROOT", os.path.join(EXTERNAL, "PredFormer"))

MODULES = {
    "full": "FullAttention",
    "fac_st": "FacST", "fac_ts": "FacTS",
    "ST": "Binary_ST", "TS": "Binary_TS",
    "TST": "Triplet_TST", "STS": "Triplet_STS",
    "TSST": "Quadruplet_TSST", "STTS": "Quadruplet_STTS",
}

NDEPTH = {
    "mmnist": {"full": 24, "fac_st": 12, "fac_ts": 12, "ST": 12, "TS": 12,
               "TST": 8, "STS": 8, "TSST": 6, "STTS": 6},
    "taxibj": {"full": 8, "fac_st": 4, "fac_ts": 4, "ST": 4, "TS": 4,
               "TST": 3, "STS": 3, "TSST": 2, "STTS": 2},
}

DEFAULT_VARIANT = {"mmnist": "TSST", "taxibj": "STS"}


def _importable():
    root = os.path.abspath(ROOT)
    if not os.path.isdir(os.path.join(root, "openstl")):
        raise FileNotFoundError(
            f"Official PredFormer repo not found at {root}. Run\n"
            f"  git clone https://github.com/yyyujintang/PredFormer.git {root}\n"
            f"or set PREDFORMER_ROOT to where you cloned it.")
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def config_family(dataset):
    return "mmnist" if dataset == "mmnist" else "taxibj"


def official_config(dataset, variant, t_in, t_out, C, H, W, patch_size=None):
    family = config_family(dataset)
    path = os.path.join(_importable(), "configs", family, "PredFormer.py")
    cfg = dict(runpy.run_path(path)["model_config"])
    cfg.update(Ndepth=NDEPTH[family][variant], pre_seq=t_in, after_seq=t_out,
               num_channels=C, height=H, width=W)
    if patch_size:
        cfg["patch_size"] = patch_size
    return cfg


def build_official(variant, config):
    _importable()
    module = importlib.import_module(f"openstl.models.PredFormer_{MODULES[variant]}")
    return module.PredFormer_Model(config)
