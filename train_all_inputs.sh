#!/usr/bin/env bash
# Two input / output changes to DistVol and DistVolFormer, each with MSE, the Sinkhorn loss and the
# Sinkhorn loss without its Sinkhorn term, run with train_all.sh's machinery in one resumable
# workspace inputs_<dataset>[_in<N>_out<M>]. Default dataset nyctaxi, seeds 42 7 123.
#
#   ./train_all_inputs.sh                     nyctaxi
#   ./train_all_inputs.sh --dataset nycbike2  another dataset (its own W&B project)
#   ./train_all_inputs.sh --list              the runs; other train_all.sh options work too
#
#   <model>_<loss>            as before (the reference)
#   <model>_res_<loss>        --dist-residual: the distribution is softmax(log P_last + change), so the
#                             model starts from "where traffic is now" and learns where it changes
#   <model>_periodic_<loss>   --periods day week: also the frames one day and one week before every
#                             target frame (ST-ResNet's period / trend idea); the first week of the
#                             training data has no such history and is dropped
#   <model>_res_periodic_<loss>  both
#   losses: mse | sinkhorn (1 L1 + 10 Sinkhorn + 1 volume) | sinkhorn_l2-0 (L1 + volume)

case " $* " in
    *" -h "*|*" --help "*)
        sed -n '2,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
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

SCRIPT_NAME=./train_all_inputs.sh
DEFAULT_WORKSPACE="inputs_$dataset${tin:+_in$tin}${tout:+_out$tout}"
SEEDS=(42 7 123)
RUNS=("copy_last copy_last none")
for model in distvol distvolformer; do
    for variant in "|" "res|--dist-residual" "periodic|--periods day week" "res_periodic|--dist-residual --periods day week"; do
        tag="${variant%%|*}" opts="${variant#*|}"
        prefix="$model${tag:+_$tag}"
        RUNS+=("${prefix}_mse $model mse $opts"
               "${prefix}_sinkhorn $model sinkhorn $opts"
               "${prefix}_sinkhorn_l2-0 $model sinkhorn --l2 0 $opts")
    done
done

source "$(dirname "${BASH_SOURCE[0]}")/train_all.sh" "$@"
