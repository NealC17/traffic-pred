from dataclasses import dataclass, field
from typing import Callable

from trafficpred.models.cnn import CNN
from trafficpred.models.copy_last import CopyLast
from trafficpred.models.distvol import DistVolNet
from trafficpred.models.distvolformer import DistVolFormer
from trafficpred.models.mlp import MLP
from trafficpred.models.predformer import VARIANTS, PredFormer, with_horizon
from trafficpred.models.stid import STID
from trafficpred.models.staeformer import STAEformer
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


def _ext(splits, args=None):   # only stored when on, so older checkpoints' kwargs stay valid
    if getattr(args, "ext_input", False):
        if not splits.ext_dim:
            raise SystemExit("--ext-input needs --external (and optionally --weather)")
        return {"ext_dim": splits.ext_dim, "ext_input": True}
    if getattr(args, "ext_volume", False):
        if not splits.ext_dim:
            raise SystemExit("--ext-volume needs --external (and optionally --weather)")
        return {"ext_dim": splits.ext_dim, "ext_volume": True}
    return {"ext_dim": splits.ext_dim} if splits.ext_dim else {}


def _configure_cnn(args, splits):
    return dict(_grid(splits), width=_pick(args, "width", 64), blocks=_pick(args, "blocks", 6), **_ext(splits))


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
    suffix = f"_{splits.split['od_kernel']}" if splits.split.get("od_kernel") else ""
    kernel = os.path.join(files.ROOT, os.path.dirname(spec.od_file), f"od_kernel_slot{suffix}.npy" if slot else "od_kernel.npy")
    if not os.path.exists(kernel):
        raise SystemExit(f"{kernel} missing: rerun tools/import_stdn_od.py")
    return {"od_head": kernel, **({"od_slot_scale": splits.span} if slot else {})}


def _configure_distvol(args, splits):
    return dict(_grid(splits), width=_pick(args, "width", 64), **_residual(args, splits), **_log_counts(args, splits),
                **_od_head(args, splits), **({"frame_encoder": True} if getattr(args, "frame_encoder", False) else {}),
                **_ext(splits, args))


def _configure_distvolformer(args, splits):
    dim = _pick(args, "dim", 192)
    return dict(_grid(splits), width=_pick(args, "width", 64),
                patch=_pick(args, "patch_size", 2), dim=dim,
                depth=_pick(args, "num_blocks", 6), heads=_pick(args, "heads", 6),
                mlp_dim=_pick(args, "mlp_dim", 4 * dim), drop=_pick(args, "drop", 0.1), **_residual(args, splits),
                **({"sinkformer": args.sinkformer} if getattr(args, "sinkformer", None) else {}),
                **({"time_tokens": True} if getattr(args, "time_tokens", False) else {}),
                **_log_counts(args, splits), **_od_head(args, splits), **_ext(splits, args), **_modes(args),
                **_zscore(args, splits),
                **({"crop_pad": True} if splits.shape[1] == 1 and getattr(args, "time_tokens", False)
                   and _pick(args, "patch_size", 2) == 1 else {}))   # sensor data (PEMS): no empty token rows


def zscore_of(splits):
    """The train flow's (mean, std) in the 0-1 data scale, for --zscore."""
    if splits.train_mean is None:
        raise SystemExit("--zscore needs the train mean / std, which this dataset's split doesn't compute")
    return (splits.train_mean - splits.lo) / splits.span, splits.train_std / splits.span


def _zscore(args, splits):   # only stored when on, so older checkpoints' kwargs stay valid
    return {"zscore": zscore_of(splits)} if getattr(args, "zscore", False) else {}


def _modes(args):   # only stored when on, so older checkpoints' kwargs stay valid
    out = {}
    if getattr(args, "spatial_modes", None):
        out["spatial_modes"] = args.spatial_modes
        if getattr(args, "mode_gate", False):
            out["mode_gate"] = True
    if getattr(args, "mode_head", None):
        out.update(mode_head=args.mode_head, mode_init=args.mode_init)
    if getattr(args, "latent_transport", None):
        out["latent_transport"] = args.latent_transport
    if getattr(args, "time_transport", None):
        out["time_transport"] = args.time_transport
    if getattr(args, "adaptive_embed", False):
        out["adaptive_embed"] = True
    if getattr(args, "no_vol_encoder", False):
        out["no_vol_encoder"] = True
    if getattr(args, "mixed_head", False):
        out["mixed_head"] = True
    tt = {}   # --tt-*: variants of the temporal latent transport
    if getattr(args, "tt_no_topology", False):
        tt["topology"] = False
    if getattr(args, "tt_unbalanced", None):
        tt["unbalanced"] = args.tt_unbalanced
    if getattr(args, "tt_learned_mass", False):
        tt["learned_mass"] = True
    if getattr(args, "tt_sensor_topology", False):
        tt["sensor_topology"] = True
    if getattr(args, "tt_hybrid", False):
        tt["hybrid"] = True
    if getattr(args, "tt_flow_head", False):
        tt["flow_head"] = True
    if getattr(args, "tt_cond_cost", False):
        tt["cond_cost"] = True
    if getattr(args, "tt_chain", False):
        tt["chain"] = True
    if getattr(args, "st_transport", None):
        tt["space_time"] = args.st_transport
        if getattr(args, "st_only", False):
            tt["st_only"] = True
    if tt:
        out["tt_opts"] = tt
    return out


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
    from trafficpred.data.datasets import DATASETS
    cfg = _configure_distvolformer(args, splits)
    if cfg.get("transport") or cfg.get("dist_residual"):
        raise SystemExit("distvolformer_direct doesn't take --transport / --dist-residual")
    if getattr(args, "stae_embed", False):
        if getattr(DATASETS[splits.dataset], "graph_file", None) is None:
            raise SystemExit("--stae-embed reads the calendar features of the sensor datasets (pems04 / pems08 / bjtt)")
        cfg["stae_embed"] = DATASETS[splits.dataset].steps_per_day
    tt = cfg.get("tt_opts") or {}
    graph = getattr(DATASETS[splits.dataset], "graph_file", None)
    if tt.get("space_time") and graph:   # the road graph of the sensors
        cfg["tt_opts"] = dict(tt, graph=graph)
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


def _configure_stid(args, splits):
    from trafficpred.data.datasets import DATASETS
    if splits.aux_in:
        raise SystemExit("stid doesn't take --od")
    return dict(_grid(splits), **({"ext_dim": splits.ext_dim} if splits.ext_dim else {}),
                steps_per_day=getattr(DATASETS[splits.dataset], "steps_per_day", 288),
                drop=_pick(args, "dropout", 0.15))


def _configure_staeformer(args, splits):
    from trafficpred.data.datasets import DATASETS
    if splits.aux_in or splits.extra_in:
        raise SystemExit("staeformer takes the recent frames only (no --od / --periods)")
    if not splits.ext_dim:
        raise SystemExit("staeformer needs the time of day / day of week: add --external")
    return dict(_grid(splits), ext_dim=splits.ext_dim,
                steps_per_day=getattr(DATASETS[splits.dataset], "steps_per_day", 288),
                zscore=zscore_of(splits), drop=_pick(args, "dropout", 0.1))


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
    # BasicTS's PEMS04 / 08 STID config: Adam lr 0.002, weight decay 1e-4, batch 32, masked MAE. Adam's
    # (coupled) weight decay acts against the loss gradient, which here is on the 0-1 scale (/ span), not
    # BasicTS's z-scores (/ std): 1e-4 * std / span of the train flow keeps it equally strong. With 1e-4
    # as is, STID underfits (PEMS08 MAE 15.40 vs published 14.21); matched it gives 14.34.
    "stid": ModelSpec(STID, _configure_stid, "mae",
                      dict(CONV_TRAIN, epochs=100, batch_size=32, lr=2e-3, weight_decay=1e-4, clip=0.0,
                           optimizer="adam"),
                      {"pems08": dict(weight_decay=1e-4 * 145.6 / 1077), "pems04": dict(weight_decay=1e-4 * 156.5 / 919),
                       "bjtt": dict(weight_decay=1e-4 * 17.05 / 142)}),
    # the official PEMS recipe (model/STAEformer.yaml, train.py): Adam, wd 1.5e-3, lr 1e-3 x0.1 at epochs 25 /
    # 45 / 65, batch 16, up to 300 epochs with early stopping after 30, no clipping; Huber on the raw counts
    "staeformer": ModelSpec(STAEformer, _configure_staeformer, "huber_raw",
                            dict(CONV_TRAIN, epochs=300, batch_size=16, lr=1e-3, weight_decay=1.5e-3, clip=0.0,
                                 optimizer="adam", schedule="multistep", milestones=(25, 45, 65), lr_gamma=0.1,
                                 patience=30)),
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
    g.add_argument("--frame-encoder", action="store_true",
                   help="distvol: encode every input frame on its own (shared convs + a lag embedding), combine "
                        "them with attention across frames at each cell, then the U-Net and both heads "
                        "(default: all frames stacked as channels into the U-Net)")
    g.add_argument("--ext-input", action="store_true",
                   help="distvol / distvolformer with --external: every frame's external features (each input "
                        "frame's own calendar / weather and the target steps' calendar + last known weather) go "
                        "through a shared embedding into the encoders as constant input maps (with --time-tokens: "
                        "per frame, and the targets' onto the target tokens), instead of only conditioning the "
                        "blocks and the volume head")
    g.add_argument("--spatial-modes", type=int, metavar="R",
                   help="distvolformer --time-tokens: the spatial blocks mix the patch tokens through R learned "
                        "spatial modes (U (V^T h) W) instead of self-attention; the temporal blocks keep attention")
    g.add_argument("--latent-transport", type=int, metavar="K",
                   help="distvolformer --time-tokens: the spatial blocks move each token by entropic OT (K Sinkhorn "
                        "steps, odd) with a learned cost, the distance from a learned 'now' to a learned 'next' "
                        "embedding plus a learned sensor topology, instead of self-attention")
    g.add_argument("--adaptive-embed", action="store_true",
                   help="distvolformer --time-tokens: one learned vector per (frame, patch) in place of a patch "
                        "position + a frame embedding (STAEformer's spatio-temporal adaptive embedding)")
    g.add_argument("--no-vol-encoder", action="store_true",
                   help="distvolformer_direct: no volume U-Net; the blocks' AdaLN gets a zero code, so it is plain "
                        "LayerNorm with learned shift / scale / gate (STAEformer-like: the transformer alone)")
    g.add_argument("--zscore", action="store_true",
                   help="distvolformer: train in the papers' z-scored units: the loss is divided by the train "
                        "flow's std (in the 0-1 data scale), and the direct head predicts (x - mean) / std")
    g.add_argument("--time-transport", type=int, metavar="K",
                   help="distvolformer --time-tokens: the temporal blocks move the input frames' tokens into all "
                        "frames' (input and future) by entropic OT (K Sinkhorn steps, odd) with a learned cost "
                        "and a learned topology over the lags, instead of temporal self-attention")
    g.add_argument("--tt-no-topology", action="store_true",
                   help="--time-transport: no learned lag topology; the cost is the state distance alone")
    g.add_argument("--tt-unbalanced", type=float, metavar="RHO",
                   help="--time-transport: unbalanced OT, the input frames' marginal a KL penalty of strength RHO "
                        "(learned per head, starting here) instead of a hard 'send N / S each'")
    g.add_argument("--tt-learned-mass", action="store_true",
                   help="--time-transport: each input frame's mass comes from its token (softmax over the frames) "
                        "instead of N / S each")
    g.add_argument("--tt-sensor-topology", action="store_true",
                   help="--time-transport: one learned lag topology per sensor instead of one shared")
    g.add_argument("--tt-flow-head", action="store_true",
                   help="distvolformer_direct --time-transport: the forecast is the observed past flow moved by the "
                        "last temporal block's transport plan, plus the head's correction (zero at the start)")
    g.add_argument("--tt-no-decay", action="store_true",
                   help="--time-transport / --latent-transport: no weight decay on the transport's geometry (its "
                        "temperatures, topology positions and weights, and rho), as for norms and biases")
    g.add_argument("--tt-cond-cost", action="store_true",
                   help="--time-transport: add a cost term from the conditioning code (the target calendar), so the "
                        "lag map can depend on the time of day / day of week (zero at the start)")
    g.add_argument("--st-transport", type=int, metavar="R",
                   help="--time-transport K: the temporal blocks become one space-time transport (K Sinkhorn steps): "
                        "mass moves from any input frame of any sensor within R hops (the PEMS road graph, or the "
                        "8-neighbour patch grid) to every frame of the sensor; unbalanced (--tt-unbalanced, default 1). "
                        "Memory grows with the neighbourhood: PEMS08 at batch 16 needs ~3.3 GB per block for R = 1 "
                        "(at most 10 neighbours), ~6.6 GB for R = 2 (28): R = 2 wants a bigger GPU")
    g.add_argument("--tt-chain", action="store_true",
                   help="--time-transport: chained transport: the temporal blocks after the first take every earlier "
                        "frame as a source too (t+2 draws on the inputs and t+1, ...), unbalanced columns")
    g.add_argument("--stae-embed", action="store_true",
                   help="distvolformer_direct --time-tokens --zscore --external (PEMS): STAEformer's token embedding "
                        "(z-scored flow + time of day + day of week projected to 24, time-of-day and day-of-week "
                        "embeddings 24 + 24, a free (frame, sensor) vector for the rest), concatenated, instead of the "
                        "patch embedding of the input features")
    g.add_argument("--mixed-head", action="store_true",
                   help="distvolformer --time-tokens: STAEformer's output layer: no target frames; each patch's "
                        "final input-frame tokens, flattened, map to its T_out steps by one linear layer")
    g.add_argument("--st-only", action="store_true",
                   help="--st-transport: every block is a space-time transport (no spatial attention blocks)")
    g.add_argument("--tt-hybrid", action="store_true",
                   help="--time-transport: keep the temporal self-attention as well and add the two outputs")
    g.add_argument("--mode-gate", action="store_true",
                   help="--spatial-modes: scale each mode by g = 1 + tanh(G(Z)), G a linear map within each mode "
                        "and one across modes, from the input (zero at the start, so it starts as fixed modes)")
    g.add_argument("--mode-head", type=int, metavar="R",
                   help="distvolformer: add a low-rank term Phi a to the distribution logits, Phi R learned spatial "
                        "modes per channel, a from the volume code (zero at the start)")
    g.add_argument("--mode-init", choices=("random", "pca"), default="random",
                   help="--mode-head: start Phi at random or at the train split's top change modes (PCA of "
                        "log p(target) - log p(last frame))")
    g.add_argument("--ext-volume", action="store_true",
                   help="distvol / distvolformer with --external: the target steps' external features (as in the "
                        "default conditioning) go to the volume head only, not the FiLM / AdaLN of the map branch")
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
