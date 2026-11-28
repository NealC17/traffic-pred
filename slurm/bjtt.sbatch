#!/bin/bash
# BjTT (1260 Beijing roads, speed every 4 min; tools/import_bjtt.py): 12 steps in -> 15 out (20 / 40 / 60 min =
# steps 5 / 10 / 15), the published chronological 70 / 10 / 20 split (val Mar 4 - 13, test Mar 14 - 31; February
# is left out of train because its road order isn't published). Every model gets the calendar (--external).
#   1. copy_last                                                     the persistence reference
#   2. stid                                                          BasicTS STID settings (weight decay scaled)
#   3. staeformer                                                    the official PEMS recipe, picked by val MAE
#   4. distvolformer_direct_small_extin_tt5_flow_ae_z                the best PEMS08 DistVolFormer (TT 5 steps,
#                                                                    no decay on TT, flow head; test MAE 14.48)
#   5. distvolformer_direct_small_extin_ae_z                         its attention control (no TT, no flow head)
# Each for every seed in SEEDS (default 42). Then everything is scored with tools/eval_pems.py into
# logs/bjtt_eval.out (masked MAE / RMSE / MAPE over the 15 steps and at steps 5 / 10 / 15, next to the BjTT paper).
#
# The job resubmits itself 15 min before its time limit (--signal) and the continuation resumes the
# interrupted run from its .last.pt; finished runs have a .done marker and are skipped.
#
# Needs on the cluster: the current (uncommitted) code and the imported data, which is licensed and not in git.
# Copy it from the local machine (run there, from the repo root):
#   ssh nchand39@login.sol.rc.asu.edu mkdir -p traffic-pred/datasets/BjTT
#   rsync -av datasets/BjTT/speed_2022_01.npy datasets/BjTT/speed_2022_03.npy datasets/BjTT/hops.npy \
#       nchand39@login.sol.rc.asu.edu:traffic-pred/datasets/BjTT/
# Submit from the repo root on Sol:
#   sbatch --chdir="$PWD" slurm/bjtt.sbatch
#   sbatch --chdir="$PWD" --export=ALL,SEEDS="42 7 123" slurm/bjtt.sbatch
#   sbatch --chdir="$PWD" --export=ALL,ONLY="staeformer dvf_tt" slurm/bjtt.sbatch    # some runs only
# Run keys for ONLY: copy_last stid staeformer dvf_tt dvf_attn. WANDB=0 turns W&B off.
# Stop the chain with  scancel -n bjtt
# Progress: logs/slurm-bjtt-<jobid>.out and checkpoints/workspaces/bjtt/logs/<run>.log
#
#SBATCH --job-name=bjtt
#SBATCH --partition=general
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --open-mode=append

SEEDS=${SEEDS:-42}
ONLY=${ONLY:-}
WANDB=${WANDB:-1}
MAX_JOBS=${MAX_JOBS:-30}        # safety cap on the number of chained jobs
JOB_INDEX=${JOB_INDEX:-1}
W=checkpoints/workspaces/bjtt
DATA=(datasets/BjTT/speed_2022_01.npy datasets/BjTT/speed_2022_03.npy datasets/BjTT/hops.npy)

COMMON=(--dataset bjtt --t-in 12 --t-out 15 --external)
# the PEMS08 DistVolFormer setup (run_tt_ablations.py BASE): STAEformer-sized, time tokens over the roads
DVF=(--model distvolformer_direct --loss sinkhorn --l2 0 --time-tokens --patch-size 1 --ext-input --adaptive-embed
     --zscore --log-counts --grad-checkpoint --eval-batch-size 16 --dim 152 --num-blocks 6 --heads 4 --mlp-dim 256
     --width 16)
KEYS=(copy_last stid staeformer dvf_tt dvf_attn)
declare -A NAME=(
    [copy_last]=copy_last_bjtt_in12_out15
    [stid]=stid_bjtt_in12_out15
    [staeformer]=staeformer_bjtt_in12_out15
    [dvf_tt]=distvolformer_direct_small_extin_tt5_flow_ae_z_bjtt_in12_out15
    [dvf_attn]=distvolformer_direct_small_extin_ae_z_bjtt_in12_out15)
flags() {
    case $1 in
        copy_last)  echo --model copy_last --loss none ;;
        stid)       echo --model stid ;;
        staeformer) echo --model staeformer --select mae ;;
        dvf_tt)     echo "${DVF[@]}" --time-transport 5 --tt-no-decay --tt-flow-head ;;
        dvf_attn)   echo "${DVF[@]}" ;;
    esac
}
[ -n "$ONLY" ] && KEYS=($ONLY)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

if [ ! -f train.py ] || [ ! -f tools/import_bjtt.py ]; then
    log "run this from the traffic_pred repo root with the current code (tools/import_bjtt.py); now in $PWD"
    exit 1
fi
for k in "${KEYS[@]}"; do
    [ -n "${NAME[$k]:-}" ] || { log "unknown run key '$k' (use: copy_last stid staeformer dvf_tt dvf_attn)"; exit 1; }
done
for f in "${DATA[@]}"; do
    if [ ! -f "$f" ]; then
        log "!!! $f is missing: BjTT is licensed and not in git; rsync the three files in datasets/BjTT/"
        log "    from the local machine (see the top of this script), then sbatch it again"
        exit 1
    fi
done
mkdir -p logs "$W/logs"
log "job ${SLURM_JOB_ID:-?} (chain job $JOB_INDEX of at most $MAX_JOBS) on $(hostname) in $PWD"
log "code: $(git rev-parse --short HEAD 2>/dev/null)$(git diff --quiet 2>/dev/null || echo ' (uncommitted changes)')"
grep -q '"bjtt"' trafficpred/data/datasets.py && grep -q -- "--tt-flow-head" trafficpred/models/__init__.py || {
    log "!!! this checkout has no bjtt dataset / --tt-flow-head code; copy the current code over"; exit 1; }
# shellcheck disable=SC1091
source setup.sh || { log "source setup.sh failed"; exit 1; }
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/    GPU: /'

resubmit() {
    if [ -z "${SLURM_JOB_ID:-}" ]; then
        log "not running under Slurm; nothing to resubmit"; return
    fi
    if [ "$JOB_INDEX" -ge "$MAX_JOBS" ]; then
        log "reached MAX_JOBS=$MAX_JOBS chained jobs; not resubmitting (sbatch it again to continue)"; return
    fi
    local copy="logs/.bjtt.$SLURM_JOB_ID.sbatch"
    scontrol write batch_script "$SLURM_JOB_ID" "$copy" >/dev/null || { log "!!! couldn't save the batch script"; return; }
    if sbatch --dependency="afterany:$SLURM_JOB_ID" --chdir="$PWD" \
              --export="ALL,JOB_INDEX=$((JOB_INDEX + 1)),SEEDS=$SEEDS,ONLY=$ONLY,WANDB=$WANDB" "$copy"; then
        log "continuation submitted; it starts after this job ends"
    else
        log "!!! sbatch failed: resubmit by hand with  sbatch --chdir=$PWD $copy"
    fi
}

CHILD=""
on_time_limit() {
    log "time limit is near: stopping and handing over to a new job"
    resubmit
    if [ -n "$CHILD" ]; then
        kill -TERM -- "-$CHILD" 2>/dev/null
        wait "$CHILD" 2>/dev/null
    fi
    exit 0
}
trap on_time_limit USR1

set -m   # job control: each background job gets its own process group, with id $!
run() {   # in its own process group, so the time-limit handler can stop all of it
    "$@" &
    CHILD=$!
    wait "$CHILD"
    local status=$?
    CHILD=""
    return $status
}

failed=0
for seed in $SEEDS; do
    for k in "${KEYS[@]}"; do
        n=${NAME[$k]}
        [ "$seed" != 42 ] && n=${n}_seed$seed
        [ "$k" = copy_last ] && [ "$seed" != 42 ] && continue    # no randomness: once is enough
        if [ -f "$W/$n.done" ]; then
            log "skip $n (done)"; continue
        fi
        log "train $n"
        extra=(--resume)
        [ "$WANDB" = 0 ] && extra+=(--no-wandb)
        # shellcheck disable=SC2046
        if run python -u train.py $(flags "$k") "${COMMON[@]}" --seed "$seed" --save "$W/$n.pt" "${extra[@]}" \
               >> "$W/logs/$n.log" 2>&1; then
            touch "$W/$n.done"; log "done $n"
        else
            log "!!! $n failed: see $W/logs/$n.log"; failed=1
        fi
    done
done
[ "$failed" = 1 ] && { log "stopping: a run failed; fix it, then sbatch this script again"; exit 1; }

log "every run is finished; scoring into logs/bjtt_eval.out"
python tools/eval_pems.py "$W" > logs/bjtt_eval.out 2>/dev/null
cat logs/bjtt_eval.out
log "all done"
