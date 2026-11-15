#!/usr/bin/env bash
# Written by: python tune_loss.py --study distvolformer_taxibj --confirm 3
# The 3 best loss weightings of Optuna study "distvolformer_taxibj" next to the default sinkhorn loss, at
# full length on seeds 42 7 123 2025, with train_all.sh's machinery (same options; rerun to
# continue workspace "tuned_distvolformer_taxibj"). Compare mean +- std over seeds before trusting a winner.

SCRIPT_NAME=./train_tuned_distvolformer_taxibj.sh
DEFAULT_WORKSPACE=tuned_distvolformer_taxibj
SEEDS=(42 7 123 2025)

RUNS=(
    "distvolformer_sinkhorn distvolformer sinkhorn"
    "distvolformer_sinkhorn_l2-0 distvolformer sinkhorn --l2 0"   # the same without Sinkhorn
    "distvolformer_tuned1_t4 distvolformer sinkhorn_kl --l1 1 --l2 1.813 --l3 0.04087 --l4 0.03884 --eps 0.01422"   # val RMSE 17.148 at 20 epochs: 1 L1 + 1.81 Sinkhorn + 0.0409 volume + 0.0388 KL (eps 0.0142)
    "distvolformer_tuned2_t5 distvolformer sinkhorn_kl --l1 1 --l2 0 --l3 0.9813 --l4 0.05647"   # val RMSE 17.209 at 20 epochs: 1 L1 + 0.981 volume + 0.0565 KL
    "distvolformer_tuned3_t7 distvolformer sinkhorn_kl --l1 1 --l2 0 --l3 0.03396 --l4 0"   # val RMSE 17.210 at 20 epochs: 1 L1 + 0.034 volume
)

source "$(dirname "${BASH_SOURCE[0]}")/train_all.sh" "$@"
