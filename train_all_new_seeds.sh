#!/usr/bin/env bash
# Seed and model-size study of the DistVol / DistVolFormer losses, run with train_all.sh's machinery:
# every run below, once per seed, in one resumable workspace (default "new_seeds"; rerun the same
# command to continue it, e.g. after a Slurm time limit). Takes the same options as train_all.sh,
# e.g. --list, --dataset bikenyc, --epochs 5, --seeds 42 7, --workspace=<name>, run-name filters.
#
#   losses   mse | sinkhorn (1 L1 + 10 Sinkhorn + 1 volume) | sinkhorn_l2-0 (the same without the
#            Sinkhorn term: does it help?) | sinkhorn_adaptive (all three, weights set like Adam so
#            each term's parameter gradient matches the L1 term's)
#   sizes    distvol 3.95M (width 64) and distvol_l 8.72M (width 96); distvolformer 7.53M (dim 192,
#            6 heads) and distvolformer_l 9.94M (dim 256, 8 heads); for scale: predformer 8.44M,
#            predformer_official 9.49M, astgnn ~0.55M (0.28M per channel)
#   seeds    42 (the default everywhere else) and 7, 123, 2025; names get _seedN except for 42

case " $* " in
    *" -h "*|*" --help "*)
        sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        echo; echo "Options (from train_all.sh):"; echo
        bash "$(dirname "${BASH_SOURCE[0]}")/train_all.sh" --help | sed -n '/^  \.\/train_all.sh/,/^$/p'
        exit 0 ;;
esac

SCRIPT_NAME=./train_all_new_seeds.sh
DEFAULT_WORKSPACE=new_seeds
SEEDS=(42 7 123 2025)

DISTVOL_L="--width 96"
DISTVOLFORMER_L="--dim 256 --heads 8"
RUNS=(
    "copy_last copy_last none"

    "distvol_mse distvol mse"
    "distvol_sinkhorn distvol sinkhorn"
    "distvol_sinkhorn_l2-0 distvol sinkhorn --l2 0"
    "distvol_sinkhorn_adaptive distvol sinkhorn_adaptive"

    "distvol_l_mse distvol mse $DISTVOL_L"
    "distvol_l_sinkhorn distvol sinkhorn $DISTVOL_L"
    "distvol_l_sinkhorn_l2-0 distvol sinkhorn --l2 0 $DISTVOL_L"
    "distvol_l_sinkhorn_adaptive distvol sinkhorn_adaptive $DISTVOL_L"

    "distvolformer_mse distvolformer mse"
    "distvolformer_sinkhorn distvolformer sinkhorn"
    "distvolformer_sinkhorn_l2-0 distvolformer sinkhorn --l2 0"
    "distvolformer_sinkhorn_adaptive distvolformer sinkhorn_adaptive"

    "distvolformer_l_mse distvolformer mse $DISTVOLFORMER_L"
    "distvolformer_l_sinkhorn distvolformer sinkhorn $DISTVOLFORMER_L"
    "distvolformer_l_sinkhorn_l2-0 distvolformer sinkhorn --l2 0 $DISTVOLFORMER_L"
    "distvolformer_l_sinkhorn_adaptive distvolformer sinkhorn_adaptive $DISTVOLFORMER_L"

    # the strongest baseline over the same seeds (200 epochs each, so off by default):
#    "predformer_official_mse predformer_official mse"
)

source "$(dirname "${BASH_SOURCE[0]}")/train_all.sh" "$@"
