#!/bin/bash
# PEMS08 control for the temporal latent transport (TT) run: the same experiment as run_pems_ae.sh
# (STAEformer-sized DistVolFormer, --ext-input, --adaptive-embed, --zscore, --log-counts), with the temporal
# blocks left on ordinary multi-head attention, i.e. without --time-transport 5. Everything else is
# identical, so the two runs differ only in how the temporal blocks mix frames.
#   1. distvolformer_direct_small_extin_ae_z_pems08_in12_out12   (no share x total split)
#   2. distvolformer_small_extin_ae_z_pems08_in12_out12          (with the split)
# Each for every seed in SEEDS (default 42, the seed of the local TT runs). Then everything is scored with
# tools/eval_pems.py into logs/pems_pems08_attn_eval.out (masked MAE / RMSE / MAPE, as the PeMS papers).
#
# The job resubmits itself 15 min before its time limit (--signal) and the continuation resumes the
# interrupted run from its .last.pt; finished runs have a .done marker and are skipped.
#
# Needs on the cluster: the current (uncommitted) code. The PEMS08 data is downloaded on the first job if missing.
# Submit from the repo root:
#   sbatch --chdir="$PWD" slurm/pems_ae_attention.sbatch
#   sbatch --chdir="$PWD" --export=ALL,SEEDS="42 7 123" slurm/pems_ae_attention.sbatch
#   sbatch --chdir="$PWD" --export=ALL,ONLY=distvolformer_direct slurm/pems_ae_attention.sbatch   # one variant
# WANDB=0 turns W&B off. Stop the chain with  scancel -n pems_ae_attn
# Progress: logs/slurm-pems_ae_attn-<jobid>.out and checkpoints/workspaces/pems_pems08/logs/<run>.log
#
#SBATCH --job-name=pems_ae_attn
#SBATCH --partition=general
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --open-mode=append

SEEDS=${SEEDS:-42}
ONLY=${ONLY:-}
WANDB=${WANDB:-1}
MAX_JOBS=${MAX_JOBS:-20}        # safety cap on the number of chained jobs
JOB_INDEX=${JOB_INDEX:-1}
W=checkpoints/workspaces/pems_pems08
VARIANTS=(distvolformer_direct distvolformer)
[ -n "$ONLY" ] && VARIANTS=($ONLY)

# identical to run_pems_ae.sh minus --time-transport 5
COMMON=(--dataset pems08 --t-in 12 --t-out 12 --loss sinkhorn --l2 0 --time-tokens --patch-size 1
        --external --ext-input --adaptive-embed --zscore --log-counts --grad-checkpoint --eval-batch-size 16
        --dim 152 --num-blocks 6 --heads 4 --mlp-dim 256 --width 16)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

if [ ! -f train.py ] || [ ! -f tools/download_pems.py ]; then
    log "run this from the traffic_pred repo root (with tools/download_pems.py); now in $PWD"
    exit 1
fi
mkdir -p logs "$W/logs"
log "job ${SLURM_JOB_ID:-?} (chain job $JOB_INDEX of at most $MAX_JOBS) on $(hostname) in $PWD"
log "code: $(git rev-parse --short HEAD 2>/dev/null)$(git diff --quiet 2>/dev/null || echo ' (uncommitted changes)')"
grep -q -- "--time-transport" train.py trafficpred/models/__init__.py 2>/dev/null || \
    log "!!! this checkout has no --adaptive-embed / --time-transport code; copy the current code over"
# shellcheck disable=SC1091
source setup.sh || { log "source setup.sh failed"; exit 1; }

# The data, once: only the first job of a chain downloads (when flow.npy / hops.npy are missing); every later
# job and every training run just reads the local .npy files (the dataloader never downloads anything).
if [ -f datasets/PEMS/PEMS08/flow.npy ] && [ -f datasets/PEMS/PEMS08/hops.npy ]; then
    log "data: datasets/PEMS/PEMS08 present, not downloading"
else
    log "data: datasets/PEMS/PEMS08 missing, downloading it once"
    python tools/download_pems.py PEMS08 || {
        log "!!! download failed (no internet on this compute node?): run  python tools/download_pems.py PEMS08"
        log "    on the login node, then sbatch this script again"; exit 1; }
fi
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/    GPU: /'

resubmit() {
    if [ -z "${SLURM_JOB_ID:-}" ]; then
        log "not running under Slurm; nothing to resubmit"; return
    fi
    if [ "$JOB_INDEX" -ge "$MAX_JOBS" ]; then
        log "reached MAX_JOBS=$MAX_JOBS chained jobs; not resubmitting (sbatch it again to continue)"; return
    fi
    local copy="logs/.pems_ae_attn.$SLURM_JOB_ID.sbatch"
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
    for m in "${VARIANTS[@]}"; do
        n=${m}_small_extin_ae_z_pems08_in12_out12
        [ "$seed" != 42 ] && n=${n}_seed$seed
        if [ -f "$W/$n.done" ]; then
            log "skip $n (done)"; continue
        fi
        log "train $n"
        extra=(--resume)
        [ "$WANDB" = 0 ] && extra+=(--no-wandb)
        if run python -u train.py --model "$m" "${COMMON[@]}" --seed "$seed" --save "$W/$n.pt" "${extra[@]}" \
               >> "$W/logs/$n.log" 2>&1; then
            touch "$W/$n.done"; log "done $n"
        else
            log "!!! $n failed: see $W/logs/$n.log"; failed=1
        fi
    done
done
[ "$failed" = 1 ] && { log "stopping: a run failed; fix it, then sbatch this script again"; exit 1; }

log "every run is finished; scoring into logs/pems_pems08_attn_eval.out"
python tools/eval_pems.py "$W" --match '_ae_z_' > logs/pems_pems08_attn_eval.out 2>/dev/null
cat logs/pems_pems08_attn_eval.out
log "all done"
