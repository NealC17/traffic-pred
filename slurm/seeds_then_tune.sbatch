#!/bin/bash
# Finish the seed study (train_all_new_seeds.sh), and only then tune the loss weights
# (tune_loss.py), as a chain of Slurm jobs that resubmits itself until both are done:
#
#   1. seed study   runs ./train_all_new_seeds.sh until every run in it has a .done marker
#                   (finished runs are skipped, an interrupted one resumes from its last epoch)
#   2. tuning       python tune_loss.py --until 40: one GPU worker, until the study has 40
#                   finished trials; then writes train_tuned_<study>.sh (--confirm 3)
#
# 15 min before a job's time limit (--signal), it stops what is running, submits its own
# continuation (which starts once this job has ended, so two jobs never share the workspace),
# and exits. A tuning trial cut short this way is queued to run again. If the seed study ends on
# its own with runs still unfinished (a run crashed), the chain stops so you can look at it.
#
# First submission, from the repo root on Sol while the current seed-study job is still running.
# Fetching only downloads; the working tree, which the running job reads, stays as it is until
# this job merges it after that job has ended:
#
#   git fetch origin
#   git show origin/main:slurm/seeds_then_tune.sbatch > ~/seeds_then_tune.sbatch
#   # edit the #SBATCH lines in ~/seeds_then_tune.sbatch to match: scontrol show job <JOBID>
#   sbatch --dependency=afterany:<JOBID> --chdir="$PWD" --export=ALL,PULL=1 ~/seeds_then_tune.sbatch
#
# Put cluster settings in the #SBATCH lines, not on the sbatch command line: each continuation is
# submitted from this job's own copy of the script, so command-line options other than the ones
# above would be lost after the first job. Stop the chain with  scancel -n seeds_then_tune.
# Progress: logs/slurm-seeds_then_tune-<jobid>.out, or W&B: group new_seeds in project traffic-pred,
# group hyperparameter_tuning_<study> in project traffic-pred-hyperparameter-tuning. Add TUNE=0 to --export to finish the seed study only (e.g. when
# the tuning runs elsewhere: two machines on one study name would give W&B runs the same names).
#
#SBATCH --job-name=seeds_then_tune
#SBATCH --partition=general
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --open-mode=append

STUDY_SCRIPT=${STUDY_SCRIPT:-train_all_new_seeds.sh}
STUDY_WORKSPACE=${STUDY_WORKSPACE:-checkpoints/workspaces/new_seeds}
TUNE_ARGS=(--until "${TUNE_TRIALS:-40}" --trials 100000)   # e.g. add: -- --model distvolformer --dim 256 --heads 8
CONFIRM_TOP=${CONFIRM_TOP:-3}
TUNE=${TUNE:-1}                 # 0: stop after the seed study
MAX_JOBS=${MAX_JOBS:-20}        # safety cap on the number of chained jobs
JOB_INDEX=${JOB_INDEX:-1}

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

if [ ! -f train.py ] || [ ! -f "$STUDY_SCRIPT" ]; then
    log "run this from the traffic_pred repo root (sbatch --chdir=<repo>); now in $PWD"
    exit 1
fi
log "job ${SLURM_JOB_ID:-?} (chain job $JOB_INDEX of at most $MAX_JOBS) on $(hostname) in $PWD"

if [ "${PULL:-0}" = 1 ]; then
    log "updating the code: git merge --ff-only @{u}  (fetched before submitting)"
    git merge --ff-only '@{u}' || { log "couldn't fast-forward; fix the checkout and resubmit"; exit 1; }
fi
log "code: $(git rev-parse --short HEAD 2>/dev/null)$(git diff --quiet 2>/dev/null || echo ' (uncommitted changes)')"
# shellcheck disable=SC1091
source setup.sh || { log "source setup.sh failed"; exit 1; }

resubmit() {
    if [ -z "${SLURM_JOB_ID:-}" ]; then
        log "not running under Slurm; nothing to resubmit"; return
    fi
    if [ "$JOB_INDEX" -ge "$MAX_JOBS" ]; then
        log "reached MAX_JOBS=$MAX_JOBS chained jobs; not resubmitting (sbatch it again to continue)"; return
    fi
    local copy="logs/.seeds_then_tune.$SLURM_JOB_ID.sbatch"
    scontrol write batch_script "$SLURM_JOB_ID" "$copy" >/dev/null || { log "!!! couldn't save the batch script"; return; }
    if sbatch --dependency="afterany:$SLURM_JOB_ID" --chdir="$PWD" \
              --export="ALL,PULL=0,JOB_INDEX=$((JOB_INDEX + 1))" "$copy"; then
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

study_left() {   # runs of the seed study without a current .done marker
    local protocol tag
    protocol="$(sed -n 's/^PROTOCOL="\(.*\)"$/\1/p' train_all.sh)"
    bash "$STUDY_SCRIPT" --list | sed -n 's/.*-> \(.*\)\.pt$/\1/p' | while read -r tag; do
        grep -qx "$protocol" "$STUDY_WORKSPACE/$tag.done" 2>/dev/null || echo "$tag"
    done
}

left="$(study_left)"
if [ -n "$left" ]; then
    log "stage 1: seed study, $(echo "$left" | wc -l | tr -d ' ') runs left"
    run bash "$STUDY_SCRIPT"
    left="$(study_left)"
    if [ -n "$left" ]; then
        log "!!! the seed study ended with $(echo "$left" | wc -l | tr -d ' ') unfinished runs (see FAILED above):"
        echo "$left" | sed 's/^/    /'
        log "stopping the chain; fix them, then sbatch this script again (without PULL=1)"
        exit 1
    fi
fi
log "stage 1 done: every seed-study run is finished"
if [ "$TUNE" = 0 ]; then
    log "TUNE=0: not tuning; done"
    exit 0
fi

log "stage 2: tune_loss.py ${TUNE_ARGS[*]}"
if ! run python -u tune_loss.py "${TUNE_ARGS[@]}"; then
    log "!!! tune_loss.py failed; stopping the chain (rerunning this script continues the study)"
    exit 1
fi
STUDY_ARGS=()
for i in "${!TUNE_ARGS[@]}"; do   # pass the study name / train args on to --confirm
    case "${TUNE_ARGS[$i]}" in --study) STUDY_ARGS+=(--study "${TUNE_ARGS[$((i + 1))]}") ;; esac
done
DASH_ARGS=()
for i in "${!TUNE_ARGS[@]}"; do
    [ "${TUNE_ARGS[$i]}" = -- ] && DASH_ARGS=("${TUNE_ARGS[@]:$i}") && break
done
python tune_loss.py "${STUDY_ARGS[@]}" --confirm "$CONFIRM_TOP" "${DASH_ARGS[@]}"
log "all done. Next: bash train_tuned_<study>.sh (written above) to confirm the best weights over 4 seeds"
