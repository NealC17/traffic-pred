"""
Use the official PredFormer model code (github.com/yyyujintang/PredFormer)
with this project's data.py / train.py / run_viz.py.

Setup (once):
    git clone https://github.com/yyyujintang/PredFormer.git
    pip install einops "timm<=0.6.11" fvcore opencv-python imageio torchvision

The clone is expected at ./PredFormer; set PREDFORMER_ROOT to put it elsewhere.

    from official_predformer import build_official
    model = build_official("taxibj", variant="STS")    # x (B,4,2,32,32) -> (B,4,2,32,32)

The official repo picks a variant by editing openstl/models/__init__.py; this
file imports the variant's module directly instead, so the clone stays untouched.
Model hyperparameters come straight from the repo's configs/<dataset>/PredFormer.py,
with Ndepth set per variant from the comments in those configs.
"""

import os
import sys
import runpy
import importlib

ROOT = os.environ.get("PREDFORMER_ROOT", "PredFormer")

# our variant names -> module names in openstl/models/
MODULES = {
    "full": "FullAttention",
    "fac_st": "FacST", "fac_ts": "FacTS",
    "ST": "Binary_ST", "TS": "Binary_TS",
    "TST": "Triplet_TST", "STS": "Triplet_STS",
    "TSST": "Quadruplet_TSST", "STTS": "Quadruplet_STTS",
}

# Ndepth per variant, from the comments in the official configs
NDEPTH = {
    "mmnist": {"full": 24, "fac_st": 12, "fac_ts": 12, "ST": 12, "TS": 12,
               "TST": 8, "STS": 8, "TSST": 6, "STTS": 6},
    "taxibj": {"full": 8, "fac_st": 4, "fac_ts": 4, "ST": 4, "TS": 4,
               "TST": 3, "STS": 3, "TSST": 2, "STTS": 2},
}

# the variant the official training script uses for each dataset
DEFAULT_VARIANT = {"mmnist": "TSST", "taxibj": "STS"}


def _ensure_importable():
    root = os.path.abspath(ROOT)
    if not os.path.isdir(os.path.join(root, "openstl")):
        raise FileNotFoundError(
            f"Official PredFormer repo not found at {root}. Run\n"
            f"  git clone https://github.com/yyyujintang/PredFormer.git\n"
            f"or set PREDFORMER_ROOT to where you cloned it.")
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def official_config(dataset, variant=None, **overrides):
    """model_config dict from the repo's config file, with Ndepth for `variant`."""
    root = _ensure_importable()
    variant = variant or DEFAULT_VARIANT[dataset]
    cfg = dict(runpy.run_path(os.path.join(root, "configs", dataset, "PredFormer.py"))["model_config"])
    cfg["Ndepth"] = NDEPTH[dataset][variant]
    cfg.update(overrides)
    return cfg


def build_official(dataset=None, variant=None, config=None):
    """Build the official PredFormer_Model.
    Pass dataset (+ optional variant) to use the repo's config, or a full
    `config` dict (e.g. one saved in a checkpoint) to rebuild exactly."""
    _ensure_importable()
    variant = variant or DEFAULT_VARIANT[dataset]
    if config is None:
        config = official_config(dataset, variant)
    module = importlib.import_module(f"openstl.models.PredFormer_{MODULES[variant]}")
    return module.PredFormer_Model(config)