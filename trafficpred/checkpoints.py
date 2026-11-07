import torch

from trafficpred.data import get_splits
from trafficpred.models import build_model
from trafficpred.utils import pick_device

RUN_KEYS = ("model_name", "model_kwargs", "dataset", "t_in", "t_out", "split", "scale",
            "channels", "loss", "seed")


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def load_model(path, device=None):
    ck = load(path)
    model = build_model(ck["model_name"], ck["model_kwargs"])
    if ck.get("model") is not None:
        model.load_state_dict(ck["model"])
    return model.to(device or pick_device()).eval(), ck


def splits_for(ck):
    return get_splits(ck["dataset"], ck["t_in"], ck["t_out"], **(ck.get("split") or {}))
