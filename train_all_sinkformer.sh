#!/usr/bin/env bash
# Sinkformer attention (Sander et al., AISTATS 2022) in DistVolFormer: --sinkformer 3 replaces the
# attention softmax with 3 Sinkhorn normalisations (rows, columns, rows), so the attention matrix is
# close to doubly stochastic. With the Sinkhorn loss and L1 + volume, plain and with periodic inputs,
# in one resumable workspace sinkformer_<dataset>[_in<N>_out<M>]. Default dataset nyctaxi, seeds
# 42 7 123 (the same as train_all_inputs.sh, whose distvolformer runs are the reference).
#
#   ./train_all_sinkformer.sh                     nyctaxi
#   ./train_all_sinkformer.sh --dataset bikenyc   another dataset (its own W&B project)
#   ./train_all_sinkformer.sh --list              the runs; other train_all.sh options work too

case " $* " in
    *" -h "*|*" --help "*)
        sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
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

SCRIPT_NAME=./train_all_sinkformer.sh
DEFAULT_WORKSPACE="sinkformer_$dataset${tin:+_in$tin}${tout:+_out$tout}"
SEEDS=(42 7 123)
RUNS=()
for variant in "sinkformer|--sinkformer 3" "sinkformer_periodic|--sinkformer 3 --periods day week"; do
    tag="${variant%%|*}" opts="${variant#*|}"
    RUNS+=("distvolformer_${tag}_sinkhorn distvolformer sinkhorn $opts"
           "distvolformer_${tag}_sinkhorn_l2-0 distvolformer sinkhorn --l2 0 $opts")
done

source "$(dirname "${BASH_SOURCE[0]}")/train_all.sh" "$@"
