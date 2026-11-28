#!/bin/bash
# Submit the BjTT runs as independent Slurm jobs (one GPU each, in parallel), from the repo root on Sol:
#   bash slurm/submit_bjtt.bash                       # copy_last, dvf_tt, dvf_attn (stid is done)
#   bash slurm/submit_bjtt.bash dvf_tt                # just these
#   SEEDS="7 123" bash slurm/submit_bjtt.bash dvf_tt dvf_attn
# Each job is slurm/bjtt.sbatch with ONLY=<key> and job name bjtt_<key>; it chains itself before its time limit and
# skips runs that are already .done. Stop all:  scancel -u $USER -n bjtt_copy_last,bjtt_stid,bjtt_dvf_tt,bjtt_dvf_attn
cd "$(dirname "$0")/.." || exit 1
keys=("$@")
[ ${#keys[@]} = 0 ] && keys=(copy_last dvf_tt dvf_attn)
for k in "${keys[@]}"; do
    case $k in copy_last|stid|dvf_tt|dvf_attn|staeformer) ;; *) echo "unknown run key $k"; exit 1 ;; esac
    if squeue -u "$USER" -h -o %j | grep -qx "bjtt_$k"; then
        echo "bjtt_$k is already queued or running; not submitting it twice"; continue
    fi
    sbatch --chdir="$PWD" -J "bjtt_$k" --export="ALL,ONLY=$k,SEEDS=${SEEDS:-42}" slurm/bjtt.sbatch
done
squeue -u "$USER" -o "%.10i %.16j %.8T %.10M %.10l %R"
