#!/usr/bin/env bash
# MSE vs the Sinkhorn (DistVol) loss on one dataset, with the usual baselines, run with
# train_all.sh's machinery: every run below once per seed, in one resumable workspace
# mse_vs_sinkhorn_<dataset>[_in<N>_out<M>] (rerun the same command to continue it; --tin / --tout
# give another horizon its own workspace). Default dataset nyctaxi.
#
#   ./train_all_mse_vs_sinkhorn.sh                      nyctaxi, seeds 42 7 123 2025
#   ./train_all_mse_vs_sinkhorn.sh --dataset nycbike2   another dataset (its own W&B project)
#   ./train_all_mse_vs_sinkhorn.sh --list               the runs; other train_all.sh options work too
#
#   losses     mse | sinkhorn (1 L1 + 10 Sinkhorn + 1 volume) | sinkhorn_l2-0 (the same without the
#              Sinkhorn term) for every model that has a distribution head (cnn, distvol,
#              distvolformer, stresnet); mse only for mlp and the PredFormers
#   baselines  copy_last, mlp, cnn, stresnet; transformers: distvolformer, predformer,
#              predformer_official (200 epochs each, ~5 s per epoch on a 10x20 grid). STDCformer
#              is left out: ~4x slower per epoch.

case " $* " in
    *" -h "*|*" --help "*)
        sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        echo; echo "Options (from train_all.sh):"; echo
        bash "$(dirname "${BASH_SOURCE[0]}")/train_all.sh" --help | sed -n '/^  \.\/train_all.sh/,/^$/p'
        exit 0 ;;
esac

dataset=nyctaxi tin="" tout=""
args=("$@")
for i in "${!args[@]}"; do
    case "${args[$i]}" in
        --dataset) dataset="${args[$((i + 1))]}" ;;
        --tin) tin="${args[$((i + 1))]}" ;;
        --tout) tout="${args[$((i + 1))]}" ;;
    esac
done
case " $* " in *" --dataset "*) ;; *) set -- --dataset "$dataset" "$@" ;; esac

SCRIPT_NAME=./train_all_mse_vs_sinkhorn.sh
DEFAULT_WORKSPACE="mse_vs_sinkhorn_$dataset${tin:+_in$tin}${tout:+_out$tout}"
SEEDS=(42 7 123 2025)
RUNS=(
    "copy_last copy_last none"
    "mlp_mse mlp mse"

    "cnn_mse cnn mse"
    "cnn_sinkhorn cnn sinkhorn"
    "cnn_sinkhorn_l2-0 cnn sinkhorn --l2 0"

    "stresnet_mse stresnet mse"
    "stresnet_sinkhorn stresnet sinkhorn"
    "stresnet_sinkhorn_l2-0 stresnet sinkhorn --l2 0"

    "distvol_mse distvol mse"
    "distvol_sinkhorn distvol sinkhorn"
    "distvol_sinkhorn_l2-0 distvol sinkhorn --l2 0"

    "distvolformer_mse distvolformer mse"
    "distvolformer_sinkhorn distvolformer sinkhorn"
    "distvolformer_sinkhorn_l2-0 distvolformer sinkhorn --l2 0"

    "predformer_mse predformer mse"
    "predformer_official_mse predformer_official mse"
)

source "$(dirname "${BASH_SOURCE[0]}")/train_all.sh" "$@"
