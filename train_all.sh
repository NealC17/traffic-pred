#!/usr/bin/env bash
# Train every model / loss combination on TaxiBJ, as one titled session.
#
# Every invocation is a SESSION with a title you choose. The session gets its own folder,
# checkpoints/sessions/<title>/, and its own W&B group named <title>, so sessions never
# skip, resume or overwrite each other's runs. Give the title with --title, or the script
# asks for it (in a terminal; a Slurm job has no terminal, so pass --title there).
# Running again with the SAME title continues that session: finished runs are skipped and
# an interrupted one resumes. That's how to pick up after a Slurm time limit.
#
#   ./train_all.sh --title "wide baselines"     # new session (or continue that one)
#   ./train_all.sh                      # asks for a title, then all runs, default epochs
#   ./train_all.sh cnn distvol          # only runs whose name contains "cnn" or "distvol"
#   ./train_all.sh --epochs 5           # same epoch count for every run (quick comparison)
#   ./train_all.sh --quick              # 1 epoch each: checks the whole pipeline end to end
#   ./train_all.sh --tin 6 --tout 6     # other horizon (names get an _in6_out6 suffix)
#   ./train_all.sh --train-files 2013 --val-files 2014 --test-files 2015 2016
#                                       # split by TaxiBJ file (names get _tr2013_va2014_te2015-2016)
#   ./train_all.sh --train-size 5000 --val-size 500 --test-size 0.5
#                                       # fewer windows per split (int = count, decimal = fraction;
#                                       # names get _ntr5000_nva500_nte0.5)
#   ./train_all.sh --seed 1             # another random seed (names get a _seed1 suffix)
#   ./train_all.sh --force              # retrain even if the checkpoint already exists
#   ./train_all.sh --list               # show the runs and exit
#   ./train_all.sh --sessions           # list the existing sessions and exit
#   ./train_all.sh --no-wandb           # don't log to Weights & Biases
#
# Protocol: every run picks its checkpoint on the VALIDATION set (val MSE, the same for every
# model) and scores it once on the TEST set at the end; results.tsv holds those test numbers.
# Default split: test = last 4 weeks of TaxiBJ, val = the 4 weeks before, train = the rest.
# The split flags are passed to every run (copy_last included), so all models see the same data.
#
# Weights & Biases: every run logs to project $WANDB_PROJECT (default "traffic-pred"),
# named after the run (cnn_mse, ...), in a group named after the session title. All runs log the same metrics (MSE, MAE, Sinkhorn, ...) whatever their loss,
# so they overlay on one chart. One-time setup: `wandb login`. No internet on the node?
# Run with WANDB_MODE=offline and later `wandb sync wandb/offline-run-*`.
#
# Runs (Sinkhorn = DistVolLoss: L1 + Sinkhorn + volume, for models whose output is
# non-negative by construction; PredFormer and the MLP can output negative values,
# so they train with MSE only):
#
#   name                 script              model         loss       default epochs
#   copy_last            baseline_copy_last.py  last frame repeated (reference, no training)
#   cnn_mse              train_distvol.py    BaselineCNN   MSE        50
#   cnn_sinkhorn         train_distvol.py    BaselineCNN   Sinkhorn   50
#   distvol_mse          train_distvol.py    DistVolNet    MSE        50
#   distvol_sinkhorn     train_distvol.py    DistVolNet    Sinkhorn   50
#   stresnet_mse         train_distvol.py    ST-ResNet     MSE        50
#   stresnet_sinkhorn    train_distvol.py    ST-ResNet     Sinkhorn   50
#   mlp_mse              baseline_mlp.py     MLP           MSE        50
#   predformer_mse       train.py            PredFormer    MSE        200 (paper setting)
#                                            (ours: predformer.py, TSST)
#   predformer_official_mse  train.py        PredFormer    MSE        200
#                                            (official repo code in PredFormer/, STS)
#
# Output, in S = checkpoints/sessions/<title>/  (title with spaces etc. turned into _):
#   S/session.txt                  title, start time, command line, git commit, Slurm job IDs
#   S/<name>.pt                    best weights for each run (updated during training)
#   S/results.tsv                  final test MSE, Sinkhorn, TV and volume error of every model
#                                  (also printed at the end)
#   S/<name>.done                  written only when a run finishes; runs with a .done are skipped,
#                                  so a job killed by its time limit redoes only unfinished runs.
#                                  Runs finished before the val/test protocol (checkpoint picked on
#                                  the test set) don't count as done: they are retrained.
#   S/<name>.last.pt               end-of-epoch state of cnn_* / distvol_* runs; an unfinished run
#                                  resumes from it (same W&B run, same charts). --force starts over.
#   S/logs/<name>.log              full training log
#   S/runs[/<split>]/              PredFormer's own run folders (last.pt for --resume)
#
# Resubmitting after a time limit: run with the same --title. Finished runs are skipped and
# an interrupted cnn_*/distvol_*/stresnet_*/predformer* run continues from its last epoch,
# in its original W&B run.
#
# Progress: each run shows an overall epoch bar plus a per-epoch batch bar (tqdm).
# The .log files keep the per-epoch summary lines; bar redraws are stripped from them.

set -o pipefail      # no -u: macOS's bash 3.2 treats empty arrays as unbound
ORIG_ARGS="$(printf '%q ' "$@")"
cd "$(dirname "${BASH_SOURCE[0]}")"

# --- options -----------------------------------------------------------------
EPOCHS="" TIN=4 TOUT=4 FORCE=0 LIST=0 FILTERS=() SEED="" TITLE="" SESSIONS=0
TRAIN_FILES=() VAL_FILES=() TEST_FILES=() TRAIN_SIZE="" VAL_SIZE="" TEST_SIZE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --epochs) EPOCHS="$2"; shift 2 ;;
        --quick)  EPOCHS=1; shift ;;
        --tin)    TIN="$2"; shift 2 ;;
        --tout)   TOUT="$2"; shift 2 ;;
        --seed)   SEED="$2"; shift 2 ;;
        --title)  TITLE="$2"; shift 2 ;;
        --sessions) SESSIONS=1; shift ;;
        --force)  FORCE=1; shift ;;
        --list)   LIST=1; shift ;;
        --no-wandb) export WANDB_MODE=disabled; shift ;;
        --train-files|--val-files|--test-files)        # one or more names, up to the next option
            opt="$1"; shift; vals=()
            while [ $# -gt 0 ] && [[ "$1" != -* ]]; do vals+=("$1"); shift; done
            [ ${#vals[@]} -eq 0 ] && { echo "$opt needs at least one file name"; exit 2; }
            case "$opt" in
                --train-files) TRAIN_FILES=("${vals[@]}") ;;
                --val-files)   VAL_FILES=("${vals[@]}") ;;
                --test-files)  TEST_FILES=("${vals[@]}") ;;
            esac ;;
        --train-size) TRAIN_SIZE="$2"; shift 2 ;;
        --val-size)   VAL_SIZE="$2"; shift 2 ;;
        --test-size)  TEST_SIZE="$2"; shift 2 ;;
        -h|--help) sed -n '2,/^set -o pipefail/p' "$0" | sed '$d'; exit 0 ;;
        -*) echo "unknown option: $1 (see --help)"; exit 2 ;;
        *)  FILTERS+=("$1"); shift ;;
    esac
done

# --- session: title -> folder and W&B group ---------------------------------------
SESSIONS_DIR="checkpoints/sessions"
if [ "$SESSIONS" = 1 ]; then
    found=0
    for d in "$SESSIONS_DIR"/*/; do
        [ -f "$d/session.txt" ] || continue
        found=1
        t="$(sed -n 's/^title: //p' "$d/session.txt" | head -1)"
        n_done=$(find "$d" -maxdepth 1 -name '*.done' | wc -l | tr -d ' ')
        printf "%-32s %3s runs finished   %s\n" "$t" "$n_done" "$d"
    done
    [ "$found" = 0 ] && echo "no sessions yet in $SESSIONS_DIR/"
    exit 0
fi
if [ -z "$TITLE" ] && [ "$LIST" = 0 ]; then
    if [ -t 0 ]; then
        while [ -z "$TITLE" ]; do read -r -p "Session title: " TITLE; done
    else
        echo "No session title. Pass one with --title \"...\" (needed when there's no terminal, e.g. Slurm)."
        exit 2
    fi
fi
# folder name: letters, digits, . _ - kept; anything else (spaces, /, ...) becomes _
SLUG="$(printf '%s' "${TITLE:-<title>}" | tr -c 'A-Za-z0-9._-' '_' | sed 's/__*/_/g; s/^[_.]*//; s/_*$//')"
[ -z "$SLUG" ] && { echo "title \"$TITLE\" has no usable characters"; exit 2; }
CK="$SESSIONS_DIR/$SLUG"
CK_SHOW="$CK"; [ -z "$TITLE" ] && CK_SHOW="$SESSIONS_DIR/<title>"     # --list without a title

SUFFIX=""
[ "$TIN" != 4 ] || [ "$TOUT" != 4 ] && SUFFIX="_in${TIN}_out${TOUT}"

# --- data split: flags passed to every run, and a name suffix so splits don't collide ---
join() { local IFS=-; echo "$*"; }
SPLIT_ARGS="" SPLIT_SUFFIX=""
n_files=$(( ${#TRAIN_FILES[@]} + ${#VAL_FILES[@]} + ${#TEST_FILES[@]} ))
if [ "$n_files" -gt 0 ]; then
    if [ ${#TRAIN_FILES[@]} -eq 0 ] || [ ${#VAL_FILES[@]} -eq 0 ] || [ ${#TEST_FILES[@]} -eq 0 ]; then
        echo "splitting by file needs all three: --train-files, --val-files and --test-files"; exit 2
    fi
    SPLIT_ARGS="--train-files ${TRAIN_FILES[*]} --val-files ${VAL_FILES[*]} --test-files ${TEST_FILES[*]}"
    SPLIT_SUFFIX="_tr$(join "${TRAIN_FILES[@]}")_va$(join "${VAL_FILES[@]}")_te$(join "${TEST_FILES[@]}")"
fi
[ -n "$TRAIN_SIZE" ] && SPLIT_ARGS="$SPLIT_ARGS --train-size $TRAIN_SIZE" && SPLIT_SUFFIX="${SPLIT_SUFFIX}_ntr$TRAIN_SIZE"
[ -n "$VAL_SIZE" ]   && SPLIT_ARGS="$SPLIT_ARGS --val-size $VAL_SIZE"     && SPLIT_SUFFIX="${SPLIT_SUFFIX}_nva$VAL_SIZE"
[ -n "$TEST_SIZE" ]  && SPLIT_ARGS="$SPLIT_ARGS --test-size $TEST_SIZE"   && SPLIT_SUFFIX="${SPLIT_SUFFIX}_nte$TEST_SIZE"
SUFFIX="$SUFFIX$SPLIT_SUFFIX"
# --seed: passed to every trained run; without it each script uses its default (seed 0)
# and names stay as they were, so existing checkpoints keep matching
SEED_SUFFIX=""
[ -n "$SEED" ] && SEED_SUFFIX="_seed$SEED"
SUFFIX="$SUFFIX$SEED_SUFFIX"
# PredFormer names its own run folder (taxibj, taxibj_official, + horizon); keep each
# split's and seed's apart, so --resume never picks up another seed's last.pt
PF_OUT="$CK/runs"
[ -n "$SPLIT_SUFFIX$SEED_SUFFIX" ] && PF_OUT="$CK/runs/split${SPLIT_SUFFIX}${SEED_SUFFIX}"
# .done files carry this; older ones (checkpoint picked on the test set) don't count as finished
PROTOCOL="protocol=val-select-v1"

# --- environment ----------------------------------------------------------------
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f .env/bin/activate ]; then
    # shellcheck disable=SC1091
    source .env/bin/activate
fi
PY="${PYTHON:-python}"

# Device: CUDA -> Apple MPS -> CPU is picked inside the Python scripts (devices.py).
# Ops that MPS doesn't implement fall back to the CPU instead of crashing.
export PYTORCH_ENABLE_MPS_FALLBACK=1
# W&B: one group per session, named by its title, so a session is easy to filter in the UI
export WANDB_RUN_GROUP="$TITLE"
# Data loading workers: macOS starts workers with "spawn", which re-imports data.py and
# copies the arrays into every worker. Batches here are tiny, so load in the main process.
WORKERS=2
[ "$(uname)" = Darwin ] && WORKERS=0
if ! "$PY" -c "import torch" 2>/dev/null; then
    echo "PyTorch not found. Run:  source setup.sh"
    exit 1
fi

start_session() {   # create or continue the session folder; not for --list
mkdir -p "$CK/logs"
if [ -f "$CK/session.txt" ]; then
    old_title="$(sed -n 's/^title: //p' "$CK/session.txt" | head -1)"
    if [ "$old_title" != "$TITLE" ]; then     # e.g. "my run" and "my/run" share a folder name
        echo "folder $CK already belongs to session \"$old_title\"; pick another title"; exit 2
    fi
    n_done=$(find "$CK" -maxdepth 1 -name '*.done' | wc -l | tr -d ' ')
    echo "=== continuing session \"$TITLE\" ($n_done runs already finished)  [$CK]"
else
    {
        echo "title: $TITLE"
        echo "started: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "host: $(hostname)"
        echo "git: $(git rev-parse --short HEAD 2>/dev/null || echo n/a)$(git diff --quiet 2>/dev/null || echo ' (uncommitted changes)')"
    } > "$CK/session.txt"
    echo "=== new session \"$TITLE\"  [$CK]"
fi
echo "invoked $(date '+%Y-%m-%d %H:%M:%S')${SLURM_JOB_ID:+ (Slurm job $SLURM_JOB_ID)}: ./train_all.sh $ORIG_ARGS" >> "$CK/session.txt"
}
ESC="$(printf '\033')"     # for stripping terminal escape codes from logs (portable sed)

# --- run table: name | default epochs | command (EPOCHS_ and SAVE_ filled in below) ---
RUNS=(
    "copy_last|200|baseline_copy_last.py --tin $TIN --tout $TOUT --epochs EPOCHS_ --save SAVE_ $SPLIT_ARGS"
    "cnn_mse|50|train_distvol.py --model cnn --loss mse --tin $TIN --tout $TOUT --epochs EPOCHS_ --save SAVE_ --workers $WORKERS $SPLIT_ARGS"
    "cnn_sinkhorn|50|train_distvol.py --model cnn --loss sinkhorn --tin $TIN --tout $TOUT --epochs EPOCHS_ --save SAVE_ --workers $WORKERS $SPLIT_ARGS"
    "distvol_mse|50|train_distvol.py --model distvol --loss mse --tin $TIN --tout $TOUT --epochs EPOCHS_ --save SAVE_ --workers $WORKERS $SPLIT_ARGS"
    "distvol_sinkhorn|50|train_distvol.py --model distvol --loss sinkhorn --tin $TIN --tout $TOUT --epochs EPOCHS_ --save SAVE_ --workers $WORKERS $SPLIT_ARGS"
    "stresnet_mse|50|train_distvol.py --model stresnet --loss mse --tin $TIN --tout $TOUT --epochs EPOCHS_ --save SAVE_ --workers $WORKERS $SPLIT_ARGS"
    "stresnet_sinkhorn|50|train_distvol.py --model stresnet --loss sinkhorn --tin $TIN --tout $TOUT --epochs EPOCHS_ --save SAVE_ --workers $WORKERS $SPLIT_ARGS"
    "mlp_mse|50|baseline_mlp.py --tin $TIN --tout $TOUT --epochs EPOCHS_ --save SAVE_ $SPLIT_ARGS"
    "predformer_mse|200|train.py --dataset taxibj --impl mine --t-in $TIN --t-out $TOUT --epochs EPOCHS_ --out $PF_OUT --workers $WORKERS $SPLIT_ARGS"
    "predformer_official_mse|200|train.py --dataset taxibj --impl official --t-in $TIN --t-out $TOUT --epochs EPOCHS_ --out $PF_OUT --workers $WORKERS $SPLIT_ARGS"
)

selected() {   # run name matches any filter (or there are no filters)
    [ ${#FILTERS[@]} -eq 0 ] && return 0
    local f
    for f in "${FILTERS[@]}"; do [[ "$1" == *"$f"* ]] && return 0; done
    return 1
}

if [ "$LIST" = 1 ]; then
    for r in "${RUNS[@]}"; do
        IFS='|' read -r name ep _ <<< "$r"
        selected "$name" && printf "%-24s %s epochs -> $CK_SHOW/%s%s.pt\n" "$name" "${EPOCHS:-$ep}" "$name" "$SUFFIX"
    done
    exit 0
fi

start_session || exit 2

# --- results table: run, final test MSE / Sinkhorn / TV / volume error, best epoch -------
RESULTS="$CK/results.tsv"
record_result() {   # $1 = tag, $2 = log; reads the run's "FINAL ..." line
    local line="" i
    for i in 1 2 3 4 5 6 7 8 9 10; do          # the log is written by a background tee; give it a moment
        line="$(grep -m1 '^FINAL ' "$2" 2>/dev/null)"
        [ -n "$line" ] && break
        sleep 1
    done
    [ -z "$line" ] && return
    local header
    header="$(printf 'run\tmse\tsinkhorn\tdist_tv\tvolume_err\tbest_epoch')"
    if [ ! -f "$RESULTS" ] || [ "$(head -1 "$RESULTS")" != "$header" ]; then
        [ -f "$RESULTS" ] && mv "$RESULTS" "$RESULTS.old"          # older format: keep it aside
        echo "$header" > "$RESULTS"
    fi
    awk -v t="$1" -F'\t' '$1 != t' "$RESULTS" > "$RESULTS.tmp"      # drop this run's old row
    # FINAL <name>  mse=..  sinkhorn=..  dist_tv=..  volume_err=..  best_epoch=..  (test)
    echo "$line" | awk -v t="$1" '{
        for (i = 3; i <= NF; i++) if (index($i, "=")) { split($i, kv, "="); v[kv[1]] = kv[2] }
        printf "%s\t%s\t%s\t%s\t%s\t%s\n", t, v["mse"], v["sinkhorn"], v["dist_tv"], v["volume_err"],
               v["best_epoch"] }' >> "$RESULTS.tmp"
    mv "$RESULTS.tmp" "$RESULTS"
}

# --- train ------------------------------------------------------------------------
DONE=() SKIPPED=() FAILED=()
START=$(date +%s)
for r in "${RUNS[@]}"; do
    IFS='|' read -r name ep cmd <<< "$r"
    selected "$name" || continue
    tag="$name$SUFFIX"
    ckpt="$CK/$tag.pt"
    log="$CK/logs/$tag.log"

    done_mark="$CK/$tag.done"
    if [ -f "$done_mark" ] && [ "$FORCE" = 0 ]; then
        if grep -qx "$PROTOCOL" "$done_mark"; then
            echo "=== $tag: already finished ($done_mark), skipping (use --force to retrain)"
            SKIPPED+=("$tag"); continue
        fi
        echo "=== $tag: finished before the val/test protocol (picked on test); retraining"
    fi

    rm -f "$done_mark"
    # PredFormer continues an interrupted run from <session>/runs/<run>/last.pt,
    # train_distvol.py from <session>/<name>.last.pt (both: same W&B run where supported)
    if [[ "$name" == predformer* || "$cmd" == train_distvol.py* ]] && [ "$FORCE" = 0 ]; then
        cmd="$cmd --resume"
    fi
    # a resumed run appends to its log instead of overwriting the first part
    append=0
    [[ "$cmd" == *--resume* ]] && [ -f "$CK/$tag.last.pt" ] && append=1
    [ "$append" = 0 ] && : > "$log"
    # copy_last has no weights to seed
    [ -n "$SEED" ] && [[ "$cmd" != baseline_copy_last.py* ]] && cmd="$cmd --seed $SEED"
    cmd="${cmd//EPOCHS_/${EPOCHS:-$ep}}"
    cmd="${cmd//SAVE_/$ckpt}"
    export WANDB_NAME="$tag"
    echo "=== $tag: $PY $cmd"
    echo "    log: $log"
    t0=$(date +%s)
    # shellcheck disable=SC2086
    # Bars go to the terminal as usual; the log keeps only finished lines
    # (tqdm redraws with carriage returns, so keep the text after the last \r).
    if "$PY" -u $cmd 2>&1 | tee >(tr '\r' '\n' | sed "s/${ESC}\[[0-9;]*[A-Za-z]//g" | grep -v -E '^[[:space:]]*$|%\|' >> "$log"); then
        if [[ "$name" == predformer* ]]; then
            # train.py writes <session>/runs/<run>/best.pt; copy it next to the others
            run_dir="taxibj"
            [ "$name" = predformer_official_mse ] && run_dir="taxibj_official"
            [ "$TIN" != 4 ] || [ "$TOUT" != 4 ] && run_dir="${run_dir}_in${TIN}_out${TOUT}"
            cp "$PF_OUT/$run_dir/best.pt" "$ckpt" 2>/dev/null
        fi
        if [ -f "$ckpt" ]; then
            echo "$PROTOCOL" > "$done_mark"
            record_result "$tag" "$log"
            DONE+=("$tag ($(( ($(date +%s) - t0) / 60 )) min)")
        else
            echo "!!! $tag finished but $ckpt was not written"; FAILED+=("$tag")
        fi
    else
        echo "!!! $tag failed; see $log"; FAILED+=("$tag")
    fi
done

# --- summary ------------------------------------------------------------------------
echo
echo "=== summary ($(( ($(date +%s) - START) )) sec total)"
for t in "${DONE[@]}";    do echo "  trained  $t"; done
for t in "${SKIPPED[@]}"; do echo "  skipped  $t (already had a checkpoint)"; done
for t in "${FAILED[@]}";  do echo "  FAILED   $t  (log: $CK/logs/${t}.log)"; done
if [ -f "$RESULTS" ]; then
    echo
    echo "=== final test metrics (checkpoint picked on val, scored once on test)  [$RESULTS]"
    awk -F'\t' '{printf "  %-40s %-12s %-12s %-12s %-12s %s\n", $1, $2, $3, $4, $5, $6}' "$RESULTS"
fi
[ ${#FAILED[@]} -eq 0 ]
