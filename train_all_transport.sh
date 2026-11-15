#!/usr/bin/env bash
# Transport distribution heads for DistVol and DistVolFormer, each with MSE, the Sinkhorn loss and
# the Sinkhorn loss without its Sinkhorn term, run with train_all.sh's machinery in one resumable
# workspace transport_<dataset>[_in<N>_out<M>]. Default dataset nyctaxi, seeds 42 7 123 (the same
# as train_all_inputs.sh, whose plain and periodic runs are the reference).
#
#   ./train_all_transport.sh                     nyctaxi
#   ./train_all_transport.sh --dataset bikenyc   another dataset (its own W&B project)
#   ./train_all_transport.sh --list              the runs; other train_all.sh options work too
#
#   <model>_transport_<loss>           --transport 2: every cell sends its share of traffic to the
#                                      cells within 2 of it (a learned local plan, mass conserved),
#                                      starting from the last frame and chained over the output steps
#   <model>_transport_periodic_<loss>  also --periods day week: the same time yesterday and last week
#                                      are moved too, and the three candidates are mixed per cell
#   losses: mse | sinkhorn (1 L1 + 10 Sinkhorn + 1 volume) | sinkhorn_l2-0 (L1 + volume)

case " $* " in
    *" -h "*|*" --help "*)
        sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
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

SCRIPT_NAME=./train_all_transport.sh
DEFAULT_WORKSPACE="transport_$dataset${tin:+_in$tin}${tout:+_out$tout}"
SEEDS=(42 7 123)
RUNS=()
for model in distvol distvolformer; do
    for variant in "transport|--transport 2" "transport_periodic|--transport 2 --periods day week"; do
        tag="${variant%%|*}" opts="${variant#*|}"
        RUNS+=("${model}_${tag}_mse $model mse $opts"
               "${model}_${tag}_sinkhorn $model sinkhorn $opts"
               "${model}_${tag}_sinkhorn_l2-0 $model sinkhorn --l2 0 $opts")
    done
done

source "$(dirname "${BASH_SOURCE[0]}")/train_all.sh" "$@"
