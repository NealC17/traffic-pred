#!/usr/bin/env bash
# TaxiBJ on its classic protocol (ST-ResNet AAAI'17, ACFM MM'18): next 30 minutes, test = last 4 weeks
# (val = the 4 weeks before, train = the rest), RMSE over every cell. Inputs as in ST-ResNet: the last 3
# frames (closeness) plus the same time a day back (period) and a week back (trend). Untuned default configs
# for every model, seeds 42 7 123, one resumable workspace taxibj_nextstep_in3.
#
#   copy_last                    reference
#   stresnet_mse                 ST-ResNet, its own closeness / period / trend branches (--st-branches)
#   stresnet_ext_mse             ST-ResNet with the paper's external data (weather, calendar). The code doesn't
#                                combine --external with --periods, so this one sees the 3 recent frames only
#   cnn_mse                      plain CNN
#   distvol_{mse,sinkhorn_l2-0}  DistVol: MSE, and L1 + volume (no transport)
#   distvolformer_*              DistVolFormer: mse, L1 + volume (sinkhorn_l2-0), balanced Sinkhorn (sinkhorn),
#                                unbalanced Sinkhorn reach 0.3 (sinkhorn_unbalanced)
#
# Published targets (all-cell RMSE, raw units): ST-ResNet 16.69, ACFM 15.40. Score with
#   python tools/eval_published.py checkpoints/workspaces/taxibj_nextstep_in3 --split test --threshold 0
#
#   ./train_taxibj_nextstep.sh            all runs (rerun to resume)
#   ./train_taxibj_nextstep.sh --list     the runs; other train_all.sh options work too
case " $* " in
    *" -h "*|*" --help "*)
        sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        echo; echo "Options (from train_all.sh):"; echo
        bash "$(dirname "${BASH_SOURCE[0]}")/train_all.sh" --help | sed -n '/^  \.\/train_all.sh/,/^$/p'
        exit 0 ;;
esac

if [ "$(uname -s)" = Darwin ]; then
    export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
fi

set -- --dataset taxibj --tin 3 --tout 1 "$@"
P="--periods day week"
SCRIPT_NAME=./train_taxibj_nextstep.sh
DEFAULT_WORKSPACE=taxibj_nextstep_in3
SEEDS=(42 7 123)
RUNS=(
    "copy_last copy_last none"
    "stresnet_mse stresnet mse $P --st-branches"
    "stresnet_ext_mse stresnet mse --external"
    "cnn_mse cnn mse $P"
    "distvol_mse distvol mse $P"
    "distvol_sinkhorn_l2-0 distvol sinkhorn --l2 0 $P"
    "distvolformer_mse distvolformer mse $P"
    "distvolformer_sinkhorn_l2-0 distvolformer sinkhorn --l2 0 $P"
    "distvolformer_sinkhorn distvolformer sinkhorn $P"
    "distvolformer_sinkhorn_unbalanced distvolformer sinkhorn_unbalanced $P"
)

source "$(dirname "${BASH_SOURCE[0]}")/train_all.sh" "$@"
