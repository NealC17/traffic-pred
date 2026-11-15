#!/usr/bin/env bash
# Runs set up like the published results they are compared with, scored afterwards with
# tools/eval_published.py: next-step prediction (4 in -> 1 out), best setups of this repo against
# copy-last and ST-ResNet (plain, and with the same periodic inputs as its period / trend branches).
# One resumable workspace published_<dataset>; seeds 42 7 123.
#
#   ./train_all_published.sh --dataset nyctaxi   STDN's setting: last 20 days test (the default
#                                                split); score with --threshold 10, per channel
#   ./train_all_published.sh --dataset bikenyc   DeepSTN+'s setting: last 14 days test (--test-days 14)
#   ./train_all_published.sh --dataset nycbike2  STDN bike: as nyctaxi (--threshold 10, per channel)
#   ./train_all_published.sh --list              the runs; other train_all.sh options work too

case " $* " in
    *" -h "*|*" --help "*)
        sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        echo; echo "Options (from train_all.sh):"; echo
        bash "$(dirname "${BASH_SOURCE[0]}")/train_all.sh" --help | sed -n '/^  \.\/train_all.sh/,/^$/p'
        exit 0 ;;
esac

dataset=nyctaxi
args=("$@")
for i in "${!args[@]}"; do
    [ "${args[$i]}" = --dataset ] && dataset="${args[$((i + 1))]}"
done
case " $* " in *" --dataset "*) ;; *) set -- --dataset "$dataset" "$@" ;; esac
case " $* " in *" --tout "*) ;; *) set -- "$@" --tin 4 --tout 1 ;; esac

extra=""
[ "$dataset" = bikenyc ] && extra="--test-days 14"
P="--periods day week"

SCRIPT_NAME=./train_all_published.sh
DEFAULT_WORKSPACE="published_$dataset"
SEEDS=(42 7 123)
RUNS=(
    "copy_last copy_last none $extra"
    "stresnet_mse stresnet mse $extra"
    "stresnet_periodic_mse stresnet mse $P $extra"
    "distvol_sinkhorn_l2-0 distvol sinkhorn --l2 0 $extra"
    "distvol_periodic_sinkhorn distvol sinkhorn $P $extra"
    "distvol_periodic_sinkhorn_l2-0 distvol sinkhorn --l2 0 $P $extra"
    "distvolformer_sinkhorn_l2-0 distvolformer sinkhorn --l2 0 $extra"
    "distvolformer_periodic_sinkhorn distvolformer sinkhorn $P $extra"
    "distvolformer_periodic_sinkhorn_l2-0 distvolformer sinkhorn --l2 0 $P $extra"
    "distvolformer_sinkformer_periodic_sinkhorn distvolformer sinkhorn --sinkformer 3 $P $extra"
    "distvolformer_sinkformer_periodic_sinkhorn_l2-0 distvolformer sinkhorn --l2 0 --sinkformer 3 $P $extra"
)

source "$(dirname "${BASH_SOURCE[0]}")/train_all.sh" "$@"
