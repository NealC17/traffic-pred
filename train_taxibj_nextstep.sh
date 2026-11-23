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
#   ./train_taxibj_nextstep.sh --time     time 1 epoch of each run (nothing saved, no W&B) and estimate the
#                                         full run; --time distvolformer times only matching runs
case " $* " in
    *" -h "*|*" --help "*)
        sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        echo; echo "Options (from train_all.sh):"; echo
        bash "$(dirname "${BASH_SOURCE[0]}")/train_all.sh" --help | sed -n '/^  \.\/train_all.sh/,/^$/p'
        exit 0 ;;
esac

if [ "$(uname -s)" = Darwin ]; then
    export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
fi

orig=("$@")
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

# --time: one epoch of every run (seed 42), nothing saved, no W&B; prints each run's seconds per epoch and
# the full run's estimate (EPOCHS x seeds). The workspace is not touched, so the real run starts clean.
#   ./train_taxibj_nextstep.sh --time              (add a filter, e.g. --time distvolformer)
case " $* " in
    *" --time "*)
        cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
        filters=()
        for a in "${orig[@]}"; do case "$a" in --*) ;; *) filters+=("$a") ;; esac; done
        EPOCHS=50
        total=0
        printf "%-36s %10s %12s %14s\n" run "s / epoch" "1 epoch (s)" "full est. (h)"
        for r in "${RUNS[@]}"; do
            read -r name model loss extra <<< "$r"
            if [ ${#filters[@]} -gt 0 ]; then
                hit=0; for f in "${filters[@]}"; do case "$name" in *"$f"*) hit=1 ;; esac; done
                [ "$hit" = 1 ] || continue
            fi
            t0=$(date +%s)
            # shellcheck disable=SC2086
            out=$("${PYTHON:-python}" -u train.py --model "$model" --loss "$loss" $extra \
                  --dataset taxibj --t-in 3 --t-out 1 --seed 42 --workers 0 --epochs 1 --no-save --no-wandb 2>&1 \
                  | tr '\r' '\n')
            wall=$(( $(date +%s) - t0 ))
            ep=$(printf '%s\n' "$out" | grep -E '^ *ep +1 \|' | tail -1 | sed -E 's/.*\| *([0-9.]+)s.*/\1/')
            if [ -z "$ep" ]; then
                if printf '%s\n' "$out" | grep -qiE 'error|traceback'; then
                    printf "%-36s FAILED:\n" "$name"; printf '%s\n' "$out" | grep -iE 'error' | tail -3
                    continue
                fi
                ep=$wall
            fi
            seeds=${#SEEDS[@]}; [ "$model" = copy_last ] && seeds=1
            est=$(awk -v e="$ep" -v n="$EPOCHS" -v s="$seeds" -v m="$model" \
                  'BEGIN { if (m == "copy_last") n = 0; printf "%.2f", e * n * s / 3600 }')
            total=$(awk -v t="$total" -v e="$est" 'BEGIN { printf "%.2f", t + e }')
            printf "%-36s %10s %12s %14s\n" "$name" "$ep" "$wall" "$est"
        done
        echo "total estimate: ${total} h for ${#SEEDS[@]} seeds x $EPOCHS epochs (epoch time only; add ~1 min per run)"
        exit 0 ;;
esac

source "$(dirname "${BASH_SOURCE[0]}")/train_all.sh" "$@"
