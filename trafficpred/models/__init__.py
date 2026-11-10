from dataclasses import dataclass, field
from typing import Callable

from trafficpred.models.cnn import CNN
from trafficpred.models.copy_last import CopyLast
from trafficpred.models.distvol import DistVolNet
from trafficpred.models.distvolformer import DistVolFormer
from trafficpred.models.mlp import MLP
from trafficpred.models.predformer import VARIANTS, PredFormer, with_horizon
from trafficpred.models.stresnet import STResNet


@dataclass
class ModelSpec:
    build: Callable
    configure: Callable
    losses: tuple
    train: dict
    train_by_dataset: dict = field(default_factory=dict)
    parts: bool = False

    def train_defaults(self, dataset):
        return {**self.train, **self.train_by_dataset.get(dataset, {})}


def _pick(args, name, default):
    value = getattr(args, name, None)
    return default if value is None else value


def _grid(splits):
    C, H, W = splits.shape
    return dict(T_in=splits.t_in, T_out=splits.t_out, C=C, H=H, W=W)


def _configure_cnn(args, splits):
    return dict(_grid(splits), width=_pick(args, "width", 64), blocks=_pick(args, "blocks", 6))


def _configure_distvol(args, splits):
    return dict(_grid(splits), width=_pick(args, "width", 64))


def _configure_distvolformer(args, splits):
    dim = _pick(args, "dim", 192)
    return dict(_grid(splits), width=_pick(args, "width", 64),
                patch=_pick(args, "patch_size", 2), dim=dim,
                depth=_pick(args, "num_blocks", 6), heads=_pick(args, "heads", 6),
                mlp_dim=_pick(args, "mlp_dim", 4 * dim), drop=_pick(args, "drop", 0.1))


def _configure_stresnet(args, splits):
    init_mean = None if args.st_paper_init else splits.train_mean()
    return dict(_grid(splits), filters=_pick(args, "width", 64), units=_pick(args, "st_units", 12),
                bn=bool(args.st_bn), init_mean=init_mean)


def _configure_mlp(args, splits):
    C, H, W = splits.shape
    return dict(t_in=splits.t_in, t_out=splits.t_out, channels=C, height=H, width=W,
                hidden=_pick(args, "hidden", 512), depth=_pick(args, "depth", 3),
                dropout=_pick(args, "dropout", 0.1))


def _patch_size(H, W, preferred):
    for p in (preferred, 4, 3, 2, 1):
        if H % p == 0 and W % p == 0:
            return p


def _configure_predformer(args, splits):
    C, H, W = splits.shape
    paper_mmnist = splits.dataset == "mmnist"
    drop = _pick(args, "drop", 0.0 if paper_mmnist else 0.1)
    variant = _pick(args, "variant", "TSST")
    num_blocks = _pick(args, "num_blocks", 24 if paper_mmnist else 8)
    if variant not in ("full", "fac_st", "fac_ts"):
        num_blocks -= num_blocks % len(variant)
    cfg = dict(in_shape=(splits.t_in, C, H, W),
               patch_size=_patch_size(H, W, _pick(args, "patch_size", 8 if paper_mmnist else 4)),
               dim=_pick(args, "dim", 256), heads=_pick(args, "heads", 8),
               mlp_dim=_pick(args, "mlp_dim", 1024), num_blocks=num_blocks, variant=variant,
               attn_drop=drop, drop=drop, drop_path=drop)
    return dict(t_in=splits.t_in, t_out=splits.t_out, config=cfg)


def _build_predformer(t_in, t_out, config):
    return with_horizon(PredFormer(**config), t_in, t_out)


def _configure_official(args, splits):
    from trafficpred.models import predformer_official as official
    C, H, W = splits.shape
    variant = _pick(args, "variant", official.DEFAULT_VARIANT[official.config_family(splits.dataset)])
    cfg = official.official_config(splits.dataset, variant, splits.t_in, splits.t_out, C, H, W)
    cfg["patch_size"] = _patch_size(H, W, _pick(args, "patch_size", cfg["patch_size"]))
    return dict(t_in=splits.t_in, t_out=splits.t_out, variant=variant, config=cfg)


def _build_official(t_in, t_out, variant, config):
    from trafficpred.models.predformer_official import build_official
    return with_horizon(build_official(variant, config), t_in, t_out)


def _configure_stdcformer(args, splits):
    mean, std = splits.input_stats()
    spatial = splits.spatial_features
    return dict(_grid(splits), D=_pick(args, "d_model", 64), layers=_pick(args, "layers", 5),
                K=_pick(args, "heads", 8), lape_dim=_pick(args, "lap_dim", 8),
                neighbors=_pick(args, "neighbors", 8), mean=mean, std=std,
                gradient_checkpointing=not args.no_checkpointing,
                spatial_features=None if spatial is None else spatial.tolist())


def _build_stdcformer(**kwargs):
    from trafficpred.models.stdcformer import STDCformer
    return STDCformer(**kwargs)


CONV_TRAIN = dict(epochs=50, batch_size=16, lr=1e-3, weight_decay=1e-4, clip=1.0,
                  schedule="onecycle", patience=0, optimizer="adamw", plateau_patience=10)
MLP_TRAIN = dict(CONV_TRAIN, weight_decay=1e-2, clip=0.0)
PREDFORMER_TRAIN = dict(MLP_TRAIN, epochs=200)

MODELS = {
    "copy_last": ModelSpec(CopyLast, lambda args, s: dict(t_out=s.t_out), ("none",),
                           dict(CONV_TRAIN, epochs=200)),
    "cnn": ModelSpec(CNN, _configure_cnn, ("sinkhorn", "mse", "mae"), CONV_TRAIN, parts=True),
    "distvol": ModelSpec(DistVolNet, _configure_distvol, ("sinkhorn", "mse", "mae"), CONV_TRAIN,
                         parts=True),
    "distvolformer": ModelSpec(DistVolFormer, _configure_distvolformer, ("sinkhorn", "mse", "mae"),
                               dict(CONV_TRAIN, weight_decay=1e-2), parts=True),
    "stresnet": ModelSpec(STResNet, _configure_stresnet, ("mse", "sinkhorn", "mae"), CONV_TRAIN,
                          parts=True),
    "mlp": ModelSpec(MLP, _configure_mlp, ("mse", "mae"), MLP_TRAIN),
    "predformer": ModelSpec(_build_predformer, _configure_predformer, ("mse", "mae"),
                            PREDFORMER_TRAIN, {"mmnist": dict(epochs=2000)}),
    "predformer_official": ModelSpec(_build_official, _configure_official, ("mse", "mae"),
                                     PREDFORMER_TRAIN, {"mmnist": dict(epochs=2000)}),
    "stdcformer": ModelSpec(_build_stdcformer, _configure_stdcformer, ("masked_mae", "mae", "mse"),
                            dict(CONV_TRAIN, epochs=120, batch_size=64, weight_decay=1e-4, clip=5.0,
                                 schedule="plateau", patience=50, optimizer="adam",
                                 plateau_patience=50)),
}


EXTERNAL_MODELS = ("stresnet",)


def build_model(name, kwargs):
    return MODELS[name].build(**kwargs)


def configure_model(name, args, splits):
    kwargs = MODELS[name].configure(args, splits)
    if splits.ext_dim:
        if name not in EXTERNAL_MODELS:
            raise SystemExit(f"--external only works with --model {' / '.join(EXTERNAL_MODELS)}")
        kwargs["ext_dim"] = splits.ext_dim
    return kwargs


def add_model_args(ap):
    g = ap.add_argument_group("cnn / distvol / distvolformer / stresnet")
    g.add_argument("--width", type=int,
                   help="channels (cnn, distvol, distvolformer's volume UNet) or filters (stresnet); default 64")
    g.add_argument("--blocks", type=int, help="cnn residual blocks (default 6)")
    g.add_argument("--st-units", type=int, help="stresnet residual units (default 12, the paper's)")
    g.add_argument("--st-bn", action="store_true", help="stresnet: batch norm in residual units")
    g.add_argument("--st-paper-init", action="store_true",
                   help="stresnet: output starts at 0.5 instead of the mean training value")

    g = ap.add_argument_group("mlp")
    g.add_argument("--hidden", type=int, help="hidden width (default 512)")
    g.add_argument("--depth", type=int, help="linear layers (default 3)")

    g = ap.add_argument_group("predformer / predformer_official / distvolformer",
                              "distvolformer defaults: patch 2, dim 192, 6 blocks, 6 heads, mlp 4*dim, drop 0.1")
    g.add_argument("--variant", choices=VARIANTS,
                   help="block arrangement (default TSST; official: STS, or TSST on mmnist)")
    g.add_argument("--patch-size", type=int, help="default 4 (8 on mmnist), lowered to fit the grid")
    g.add_argument("--dim", type=int, help="token width (default 256)")
    g.add_argument("--mlp-dim", type=int, help="SwiGLU hidden size (default 1024)")
    g.add_argument("--num-blocks", type=int, help="gated transformer blocks (default 8; 24 on mmnist)")
    g.add_argument("--drop", type=float, help="attention / path dropout (default 0.1; 0 on mmnist)")

    g = ap.add_argument_group("stdcformer (official code in external/STDCformer)")
    g.add_argument("--d-model", type=int, help="hidden size D (default 64, the official config)")
    g.add_argument("--layers", type=int, help="encoder and decoder blocks each (default 5)")
    g.add_argument("--lap-dim", type=int, help="Laplacian eigenvectors (default 8)")
    g.add_argument("--neighbors", type=int, choices=(4, 8), help="grid graph connectivity (default 8)")
    g.add_argument("--no-checkpointing", action="store_true",
                   help="keep every block's activations instead of recomputing them (faster, far more memory)")

    g = ap.add_argument_group("shared by several models")
    g.add_argument("--heads", type=int, help="attention heads (predformer, stdcformer; default 8)")
    g.add_argument("--dropout", type=float, help="mlp dropout (default 0.1)")
    return g
