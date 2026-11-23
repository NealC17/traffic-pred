#!/bin/bash
# The whole TaxiBJ next-step study (./train_taxibj_nextstep.sh: 10 runs x seeds 42 7 123, workspace
# checkpoints/workspaces/taxibj_nextstep_in3) as a chain of Slurm jobs that resubmits itself until every run
# has a .done marker, then scores it (test and val, every cell, seed ensembles) into logs/taxibj_eval.out.
#
# 15 min before a job's time limit (--signal), it stops the running training, submits its own continuation
# (which starts once this job has ended, so two jobs never share the workspace) and exits; the continuation
# skips finished runs and resumes the interrupted one from its last epoch. If the study ends on its own with
# runs still unfinished (a run crashed), the chain stops so you can look at it.
#
# Submit from the repo root (train_taxibj_nextstep.sh is gitignored like every *.sh: copy it over, or
# git add -f it, together with the code it needs):
#
#   sbatch --chdir="$PWD" slurm/taxibj_nextstep.sbatch
#   sbatch --chdir="$PWD" --export=ALL,FILTER=distvol_ slurm/taxibj_nextstep.sbatch   # only matching runs
#   sbatch --chdir="$PWD" --export=ALL,PULL=1 slurm/taxibj_nextstep.sbatch            # git merge --ff-only first
#
# FILTER is passed to the study script like its run filters ("distvol_" = DistVol only; "distvolformer";
# several separated by spaces). WANDB=0 turns W&B off. Put cluster settings in the #SBATCH lines below, not
# on the sbatch command line: each continuation is submitted from this job's own copy of the script, so
# other command-line options would be lost after the first job. Stop the chain with
#   scancel -n taxibj_nextstep
# Progress: logs/slurm-taxibj_nextstep-<jobid>.out, or W&B group taxibj_nextstep_in3 in project traffic-pred.
#
#SBATCH --job-name=taxibj_nextstep
#SBATCH --partition=general
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --open-mode=append

STUDY_SCRIPT=train_taxibj_nextstep.sh
STUDY_WORKSPACE=checkpoints/workspaces/taxibj_nextstep_in3
FILTER=${FILTER:-}
WANDB=${WANDB:-1}
MAX_JOBS=${MAX_JOBS:-20}        # safety cap on the number of chained jobs
JOB_INDEX=${JOB_INDEX:-1}

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

if [ ! -f train.py ] || [ ! -f "$STUDY_SCRIPT" ]; then
    log "run this from the traffic_pred repo root with $STUDY_SCRIPT in it (sbatch --chdir=<repo>); now in $PWD"
    exit 1
fi
mkdir -p logs
log "job ${SLURM_JOB_ID:-?} (chain job $JOB_INDEX of at most $MAX_JOBS) on $(hostname) in $PWD"

if [ "${PULL:-0}" = 1 ]; then
    log "updating the code: git fetch && git merge --ff-only @{u}"
    git fetch && git merge --ff-only '@{u}' || { log "couldn't fast-forward; fix the checkout and resubmit"; exit 1; }
fi
log "code: $(git rev-parse --short HEAD 2>/dev/null)$(git diff --quiet 2>/dev/null || echo ' (uncommitted changes)')"
# shellcheck disable=SC1091
source setup.sh || { log "source setup.sh failed"; exit 1; }
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/    GPU: /'

read -r -a FILTERS <<< "$FILTER"
STUDY_ARGS=("${FILTERS[@]}")
[ "$WANDB" = 0 ] && STUDY_ARGS+=(--no-wandb)

resubmit() {
    if [ -z "${SLURM_JOB_ID:-}" ]; then
        log "not running under Slurm; nothing to resubmit"; return
    fi
    if [ "$JOB_INDEX" -ge "$MAX_JOBS" ]; then
        log "reached MAX_JOBS=$MAX_JOBS chained jobs; not resubmitting (sbatch it again to continue)"; return
    fi
    local copy="logs/.taxibj_nextstep.$SLURM_JOB_ID.sbatch"
    scontrol write batch_script "$SLURM_JOB_ID" "$copy" >/dev/null || { log "!!! couldn't save the batch script"; return; }
    if sbatch --dependency="afterany:$SLURM_JOB_ID" --chdir="$PWD" \
              --export="ALL,PULL=0,JOB_INDEX=$((JOB_INDEX + 1)),FILTER=$FILTER,WANDB=$WANDB" "$copy"; then
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

study_left() {   # runs of the study (after FILTER) without a current .done marker
    local protocol tag
    protocol="$(sed -n 's/^PROTOCOL="\(.*\)"$/\1/p' train_all.sh)"
    bash "$STUDY_SCRIPT" "${FILTERS[@]}" --list | sed -n 's/.*-> \(.*\)\.pt$/\1/p' | while read -r tag; do
        grep -qx "$protocol" "$STUDY_WORKSPACE/$tag.done" 2>/dev/null || echo "$tag"
    done
}

left="$(study_left)"
if [ -n "$left" ]; then
    log "training: $(echo "$left" | wc -l | tr -d ' ') runs left${FILTER:+ (filter: $FILTER)}"
    run bash "$STUDY_SCRIPT" "${STUDY_ARGS[@]}"
    left="$(study_left)"
    if [ -n "$left" ]; then
        log "!!! the study ended with $(echo "$left" | wc -l | tr -d ' ') unfinished runs (see FAILED above):"
        echo "$left" | sed 's/^/    /'
        log "stopping the chain; fix them, then sbatch this script again"
        exit 1
    fi
fi
log "every run is finished; scoring into logs/taxibj_eval.out"

: > logs/taxibj_eval.out
for split in test val; do
    echo "## taxibj next step, $split (all cells; published test RMSE: ST-ResNet 16.69, ACFM 15.40)" >> logs/taxibj_eval.out
    python -u tools/eval_published.py "$STUDY_WORKSPACE" --split "$split" --threshold 0 --batch-size 32 --ensemble 2>&1 \
        | grep -vE "it/s|^scoring |Warning|warnings.warn" >> logs/taxibj_eval.out
done
log "all done: logs/taxibj_eval.out"
