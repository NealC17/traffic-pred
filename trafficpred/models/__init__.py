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
    loss: str  # default loss; any loss in trafficpred.losses works if the model fits it
    train: dict
    train_by_dataset: dict = field(default_factory=dict)
    parts: bool = False  # returns (x_hat, p_hat, n_hat), so it can train with the sinkhorn losses

    def train_defaults(self, dataset):
        return {**self.train, **self.train_by_dataset.get(dataset, {})}


def _pick(args, name, default):
    value = getattr(args, name, None)
    return default if value is None else value


def _aux(splits):   # only stored when on, so older checkpoints' kwargs stay valid
    if splits.split.get("od_slot"):   # _od_head strips the slot map again for the models that read it
        return {"aux": splits.aux_in - 1}
    return {"aux": splits.aux_in} if splits.aux_in else {}


def _grid(splits):
    C, H, W = splits.shape
    return dict(T_in=splits.in_frames, T_out=splits.t_out, C=C, H=H, W=W, **_aux(splits))


def _configure_cnn(args, splits):
    return dict(_grid(splits), width=_pick(args, "width", 64), blocks=_pick(args, "blocks", 6))


def _residual(args, splits):   # only stored when on, so older checkpoints' kwargs stay valid
    radius = getattr(args, "transport", None) or 0
    if radius and getattr(args, "dist_residual", False):
        raise SystemExit("--transport and --dist-residual are two different distribution heads; pick one")
    if radius:
        return {"transport": radius, "n_periodic": splits.extra_in // splits.t_out}
    return {"dist_residual": True} if getattr(args, "dist_residual", False) else {}


def _log_counts(args, splits):   # only stored when on, so older checkpoints' kwargs stay valid
    return dict(count_scale=splits.span, count_offset=splits.lo) if getattr(args, "log_counts", False) else {}


def _od_head(args, splits):   # only stored when on, so older checkpoints' kwargs stay valid
    if not getattr(args, "od_head", False):
        return {}
    import os
    from trafficpred.data import files
    from trafficpred.data.datasets import DATASETS
    spec = DATASETS[splits.dataset]
    if not splits.split.get("od") or getattr(spec, "od_file", None) is None:
        raise SystemExit("--od-head needs --od")
    if splits.lo != 0:
        raise SystemExit("--od-head moves scaled counts: it needs data whose minimum is 0")
    slot = bool(splits.split.get("od_slot"))
    kernel = os.path.join(files.ROOT, os.path.dirname(spec.od_file), "od_kernel_slot.npy" if slot else "od_kernel.npy")
    if not os.path.exists(kernel):
        raise SystemExit(f"{kernel} missing: rerun tools/import_stdn_od.py")
    return {"od_head": kernel, **({"od_slot_scale": splits.span} if slot else {})}


def _configure_distvol(args, splits):
    return dict(_grid(splits), width=_pick(args, "width", 64), **_residual(args, splits), **_log_counts(args, splits),
                **_od_head(args, splits))


def _configure_distvolformer(args, splits):
    dim = _pick(args, "dim", 192)
    return dict(_grid(splits), width=_pick(args, "width", 64),
                patch=_pick(args, "patch_size", 2), dim=dim,
                depth=_pick(args, "num_blocks", 6), heads=_pick(args, "heads", 6),
                mlp_dim=_pick(args, "mlp_dim", 4 * dim), drop=_pick(args, "drop", 0.1), **_residual(args, splits),
                **({"sinkformer": args.sinkformer} if getattr(args, "sinkformer", None) else {}),
                **({"time_tokens": True} if getattr(args, "time_tokens", False) else {}),
                **_log_counts(args, splits), **_od_head(args, splits))


def _configure_distvolformer_graph(args, splits):
    """distvolformer with a transformer volume head and a graph-transformer distribution head."""
    cfg = _configure_distvolformer(args, splits)
    cfg.pop("time_tokens", None)
    if cfg.get("sinkformer") or cfg.get("transport"):
        raise SystemExit("distvolformer_graph doesn't take --sinkformer / --transport")
    cfg.pop("patch")   # every cell is a node
    vol_dim = _pick(args, "vol_dim", 128)
    return dict(cfg, vol_transformer=True, vol_dim=vol_dim, vol_depth=_pick(args, "vol_depth", 3),
                graph=True, graph_radius=_pick(args, "graph_radius", 0))


def _configure_distvolformer_direct(args, splits):
    """Ablation: distvolformer's inputs and backbone, predicting the flow directly (no map x total)."""
    cfg = _configure_distvolformer(args, splits)
    if cfg.get("transport") or cfg.get("dist_residual"):
        raise SystemExit("distvolformer_direct doesn't take --transport / --dist-residual")
    return dict(cfg, direct=True)


def _configure_stresnet(args, splits):
    init_mean = None if args.st_paper_init else splits.train_mean()
    external = {"ext_dim": splits.ext_dim} if splits.ext_dim else {}
    branches = {}
    if getattr(args, "st_branches", False):   # only stored when on, so older checkpoints' kwargs stay valid
        if not splits.extra_lags:
            raise SystemExit("--st-branches needs periodic inputs (--periods ...)")
        branches = {"trend_idx": [i for i, lag in enumerate(splits.extra_lags) if lag in splits.trend_lags],
                    "period_idx": [i for i, lag in enumerate(splits.extra_lags) if lag not in splits.trend_lags]}
        branches = {k: v for k, v in branches.items() if v}
    return dict(_grid(splits), filters=_pick(args, "width", 64), units=_pick(args, "st_units", 12),
                bn=bool(args.st_bn), init_mean=init_mean, **external, **branches)


def _configure_mlp(args, splits):
    C, H, W = splits.shape
    return dict(t_in=splits.in_frames, t_out=splits.t_out, channels=C, height=H, width=W,
                hidden=_pick(args, "hidden", 512), depth=_pick(args, "depth", 3),
                dropout=_pick(args, "dropout", 0.1), **_aux(splits))


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
    cfg = dict(in_shape=(splits.in_frames, C, H, W),
               patch_size=_patch_size(H, W, _pick(args, "patch_size", 8 if paper_mmnist else 4)),
               dim=_pick(args, "dim", 256), heads=_pick(args, "heads", 8),
               mlp_dim=_pick(args, "mlp_dim", 1024), num_blocks=num_blocks, variant=variant,
               attn_drop=drop, drop=drop, drop_path=drop, **_aux(splits))
    return dict(t_in=splits.in_frames, t_out=splits.t_out, config=cfg)


def _build_predformer(t_in, t_out, config):
    if config.get("aux") and t_out > t_in:
        raise ValueError("predformer with --od rolls out once only: --t-out must be <= the input frames")
    return with_horizon(PredFormer(**config), t_in, t_out)


def _configure_official(args, splits):
    from trafficpred.models import predformer_official as official
    C, H, W = splits.shape
    variant = _pick(args, "variant", official.DEFAULT_VARIANT[official.config_family(splits.dataset)])
    cfg = official.official_config(splits.dataset, variant, splits.in_frames, splits.t_out, C, H, W)
    cfg["patch_size"] = _patch_size(H, W, _pick(args, "patch_size", cfg["patch_size"]))
    return dict(t_in=splits.in_frames, t_out=splits.t_out, variant=variant, config=cfg, **_aux(splits))


def _build_official(t_in, t_out, variant, config, aux=0):
    from trafficpred.models.predformer_official import AuxInput, build_official
    model = build_official(variant, config)
    if aux:   # the official code takes C channels in and out: fold the extra maps in first
        if t_out > t_in:
            raise ValueError("predformer_official with --od rolls out once only: --t-out must be <= the input frames")
        model = AuxInput(model, config["num_channels"], aux)
    return with_horizon(model, t_in, t_out)


def _configure_stdcformer(args, splits):
    if splits.aux_in:
        raise SystemExit("stdcformer doesn't take --od")
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
    "copy_last": ModelSpec(CopyLast, lambda args, s: dict(t_out=s.t_out, **({"channels": s.shape[0]} if s.aux_in else {})), "none",
                           dict(CONV_TRAIN, epochs=200)),
    "cnn": ModelSpec(CNN, _configure_cnn, "sinkhorn", CONV_TRAIN, parts=True),
    "distvol": ModelSpec(DistVolNet, _configure_distvol, "sinkhorn", CONV_TRAIN,
                         parts=True),
    "distvolformer": ModelSpec(DistVolFormer, _configure_distvolformer, "sinkhorn",
                               dict(CONV_TRAIN, weight_decay=1e-2), parts=True),
    "distvolformer_direct": ModelSpec(DistVolFormer, _configure_distvolformer_direct, "sinkhorn",
                                      dict(CONV_TRAIN, weight_decay=1e-2), parts=True),
    "distvolformer_graph": ModelSpec(DistVolFormer, _configure_distvolformer_graph, "huber",
                                     dict(CONV_TRAIN, weight_decay=1e-2), parts=True),
    "stresnet": ModelSpec(STResNet, _configure_stresnet, "mse", CONV_TRAIN,
                          parts=True),
    "mlp": ModelSpec(MLP, _configure_mlp, "mse", MLP_TRAIN),
    "predformer": ModelSpec(_build_predformer, _configure_predformer, "mse",
                            PREDFORMER_TRAIN, {"mmnist": dict(epochs=2000)}),
    "predformer_official": ModelSpec(_build_official, _configure_official, "mse",
                                     PREDFORMER_TRAIN, {"mmnist": dict(epochs=2000)}),
    "stdcformer": ModelSpec(_build_stdcformer, _configure_stdcformer, "masked_mae",
                            dict(CONV_TRAIN, epochs=120, batch_size=64, weight_decay=1e-4, clip=5.0,
                                 schedule="plateau", patience=50, optimizer="adam",
                                 plateau_patience=50)),
}


def build_model(name, kwargs):
    return MODELS[name].build(**kwargs)


def configure_model(name, args, splits):
    return MODELS[name].configure(args, splits)


def add_model_args(ap):
    g = ap.add_argument_group("cnn / distvol / distvolformer / stresnet")
    g.add_argument("--width", type=int,
                   help="channels (cnn, distvol, distvolformer's volume UNet) or filters (stresnet); default 64")
    g.add_argument("--blocks", type=int, help="cnn residual blocks (default 6)")
    g.add_argument("--dist-residual", action="store_true",
                   help="distvol / distvolformer: predict the distribution as softmax(log P_last + change), "
                        "starting from the last input frame's distribution")
    g.add_argument("--transport", type=int, metavar="R",
                   help="distvol / distvolformer: predict where traffic moves instead of the distribution: "
                        "each cell sends its share to cells within R (e.g. 2), chained over the output steps. "
                        "With --periods, the same time one period back is moved too and the candidates are "
                        "mixed per cell")
    g.add_argument("--od-head", action="store_true",
                   help="distvol / distvolformer / distvolformer_graph with --od: End(t+1) = the network's within-slot part + the trips under way at t "
                        "moved to their destinations by a transport plan (historical kernel, a learned correction "
                        "and a per-sample destination potential; every origin's known trips are conserved)")
    g.add_argument("--log-counts", action="store_true",
                   help="distvol / distvolformer: the log-flow input is log1p of the raw counts (default: "
                        "log1p of the 0-1 scaled data, which is nearly linear)")
    g.add_argument("--st-units", type=int, help="stresnet residual units (default 12, the paper's)")
    g.add_argument("--st-bn", action="store_true", help="stresnet: batch norm in residual units")
    g.add_argument("--st-branches", action="store_true",
                   help="stresnet with --periods: feed the weekly periodic frames to the trend branch and "
                        "the other periodic frames to the period branch, as in the ST-ResNet paper "
                        "(default: every frame in the closeness branch)")
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
    g.add_argument("--sinkformer", type=int, metavar="N",
                   help="distvolformer: Sinkformer attention, N Sinkhorn normalisations (rows, columns, ...) in "
                        "place of the softmax (1 = softmax; e.g. 3)")
    g.add_argument("--time-tokens", action="store_true",
                   help="distvolformer: every input frame its own tokens with a learned lag embedding, plus "
                        "learned target tokens; blocks alternate temporal and spatial attention (--num-blocks "
                        "counts both, so the size stays the same). Default: frames stacked as channels")
    g.add_argument("--vol-dim", type=int, help="distvolformer_graph: volume transformer width (default 128)")
    g.add_argument("--vol-depth", type=int, help="distvolformer_graph: volume transformer layers (default 3)")
    g.add_argument("--graph-radius", type=int,
                   help="distvolformer_graph: attend only to cells within R hops on the 8-neighbour grid "
                        "(default 0 = every cell, with a learned bias per hop distance)")
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
