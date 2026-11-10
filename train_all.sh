#!/usr/bin/env bash
set -o pipefail

usage() {
    cat <<'EOF'
Train every model on one dataset inside a workspace, and collect MSE / RMSE / MAE on the
train, val and test sets of every run into one table.

  ./train_all.sh                                   continue the most recently used workspace
  ./train_all.sh --workspace                       start a new workspace (named by date and time)
  ./train_all.sh --workspace=baselines             start a new workspace with that name
  ./train_all.sh --workspaces                      list the workspaces and exit
  ./train_all.sh --dataset bikenyc                 taxibj (default) or bikenyc
  ./train_all.sh cnn distvol                       only runs whose name contains "cnn" or "distvol"
                                                   (so "distvol" also matches the distvolformer runs)
  ./train_all.sh distvolformer                     DistVolFormer only (transformer distribution head)
  ./train_all.sh stdcformer astgnn                 STDCformer and ASTGNN only
  ./train_all.sh --epochs 5                        same epoch count for every run
  ./train_all.sh --batch-size stdcformer=8         batch size for one model (repeat for more models)
  ./train_all.sh --batch-size 32                   batch size for every model; a MODEL=N value wins
                                                   (models: copy_last cnn distvol distvolformer stresnet mlp
                                                   predformer predformer_official stdcformer astgnn)
  ./train_all.sh --quick                           1 epoch each: checks the whole pipeline
  ./train_all.sh --seed 1                          another seed (default 42; names get _seed1)
  ./train_all.sh --external                        ST-ResNet runs only, with TaxiBJ's external data
                                                   (weather, calendar) as in the paper; names get _ext
  ./train_all.sh --tin 6 --tout 6                  other horizon (names get _in6_out6)
  ./train_all.sh --train-files 2013 --val-files 2014 --test-files 2015 2016
  ./train_all.sh --train-size 5000 --val-size 500 --test-size 0.5
  ./train_all.sh --force                           retrain runs that already finished
  ./train_all.sh --no_save --quick                 check every model runs: no checkpoints, nothing
                                                   skipped, no .done or results.tsv (logs: <run>.check.log)
  ./train_all.sh --list                            show the runs and exit
  ./train_all.sh --no-wandb                        don't log to Weights & Biases

A workspace lives in checkpoints/workspaces/<name>/ and is one W&B group. Without --workspace the
most recently used workspace is continued: finished runs are skipped and an interrupted one resumes
(e.g. after a Slurm time limit). Every model uses seed 42 unless --seed is given. Each run picks
its checkpoint on val MSE; that checkpoint is then scored on train, val and test. The astgnn run
uses the official code in external/ASTGNN with its own epoch selection, cannot resume, and maps
--epochs N to N training and N/2 fine-tuning epochs.

  <run>.pt          best checkpoint with its train / val / test metrics (open with visualize.py)
  <run>.last.pt     end-of-epoch state, for resuming
  <run>.done        written when a run finishes
  logs/<run>.log    training log
  results.tsv       train / val / test MSE, RMSE, MAE (raw units) and best epoch of every run
  W&B "leaderboard" run in the workspace's group: test RMSE / MAE / MSE of every finished run,
                    refreshed after each run (by hand: python tools/leaderboard.py --workspace <name>)
EOF
}

RUNS=(
    "copy_last copy_last none"
    "cnn_mse cnn mse"
    "cnn_sinkhorn cnn sinkhorn"
    "distvol_mse distvol mse"
    "distvol_sinkhorn distvol sinkhorn"
    "distvolformer_mse distvolformer mse"
    "distvolformer_sinkhorn distvolformer sinkhorn"
    "stresnet_mse stresnet mse"
    "stresnet_sinkhorn stresnet sinkhorn"
    "mlp_mse mlp mse"
    "predformer_mse predformer mse"
    "predformer_official_mse predformer_official mse"
    "stdcformer_masked_mae stdcformer masked_mae"
    "astgnn astgnn none"
)
METRICS=(train_mse train_rmse train_mae val_mse val_rmse val_mae test_mse test_rmse test_mae best_epoch)
PROTOCOL="protocol=v3"
DEFAULT_SEED=42
WORKSPACES_DIR="checkpoints/workspaces"
LATEST_FILE="$WORKSPACES_DIR/LATEST"

ORIG_ARGS="$(printf '%q ' "$@")"
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

DATASET=taxibj EPOCHS="" TIN="" TOUT="" SEED="$DEFAULT_SEED" FORCE=0 NO_SAVE=0 LIST=0 EXTERNAL=0
NEW_WORKSPACE=0 WORKSPACE_NAME="" LIST_WORKSPACES=0
FILTERS=() TRAIN_FILES=() VAL_FILES=() TEST_FILES=() TRAIN_SIZE="" VAL_SIZE="" TEST_SIZE=""
BATCH_ALL="" BATCH_SIZES=()
while [ $# -gt 0 ]; do
    case "$1" in
        --workspace) NEW_WORKSPACE=1; shift ;;
        --workspace=*) NEW_WORKSPACE=1; WORKSPACE_NAME="${1#--workspace=}"; shift ;;
        --workspaces) LIST_WORKSPACES=1; shift ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch-size|--batch_size)
            case "$2" in
                *=*) BATCH_SIZES+=("$2") ;;
                *) BATCH_ALL="$2" ;;
            esac
            shift 2 ;;
        --quick) EPOCHS=1; shift ;;
        --tin) TIN="$2"; shift 2 ;;
        --tout) TOUT="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        --external) EXTERNAL=1; shift ;;
        --no_save|--no-save) NO_SAVE=1; shift ;;
        --list) LIST=1; shift ;;
        --no-wandb) export WANDB_MODE=disabled; shift ;;
        --train-files|--val-files|--test-files)
            opt="$1"; shift; vals=()
            while [ $# -gt 0 ] && [[ "$1" != -* ]]; do vals+=("$1"); shift; done
            [ ${#vals[@]} -eq 0 ] && { echo "$opt needs at least one file name"; exit 2; }
            case "$opt" in
                --train-files) TRAIN_FILES=("${vals[@]}") ;;
                --val-files) VAL_FILES=("${vals[@]}") ;;
                --test-files) TEST_FILES=("${vals[@]}") ;;
            esac ;;
        --train-size) TRAIN_SIZE="$2"; shift 2 ;;
        --val-size) VAL_SIZE="$2"; shift 2 ;;
        --test-size) TEST_SIZE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "unknown option: $1 (see --help)"; exit 2 ;;
        *) FILTERS+=("$1"); shift ;;
    esac
done

case "$DATASET" in
    taxibj|bikenyc) ;;
    *) echo "--dataset must be taxibj or bikenyc"; exit 2 ;;
esac

MODELS=" $(for r in "${RUNS[@]}"; do read -r _ model _ <<< "$r"; echo "$model"; done | uniq | tr '\n' ' ')"
is_count() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }
[ -n "$BATCH_ALL" ] && ! is_count "$BATCH_ALL" && { echo "--batch-size needs a positive integer or MODEL=N"; exit 2; }
for entry in "${BATCH_SIZES[@]}"; do
    model="${entry%%=*}" size="${entry#*=}"
    [[ "$MODELS" == *" $model "* ]] || { echo "--batch-size $entry: unknown model \"$model\" (models:$MODELS)"; exit 2; }
    is_count "$size" || { echo "--batch-size $entry: \"$size\" is not a positive integer"; exit 2; }
done

batch_size_for() {
    local entry found="$BATCH_ALL"
    for entry in "${BATCH_SIZES[@]}"; do [ "${entry%%=*}" = "$1" ] && found="${entry#*=}"; done
    echo "$found"
}

slug() { printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '_' | sed 's/__*/_/g; s/^[_.]*//; s/_*$//'; }
latest_workspace() { [ -f "$LATEST_FILE" ] && head -1 "$LATEST_FILE"; }

if [ "$LIST_WORKSPACES" = 1 ]; then
    latest="$(latest_workspace)"
    found=0
    for d in "$WORKSPACES_DIR"/*/; do
        [ -d "$d" ] || continue
        found=1
        name="$(basename "$d")"
        n_done=$(find "$d" -maxdepth 1 -name '*.done' | wc -l | tr -d ' ')
        mark=" "; [ "$name" = "$latest" ] && mark="*"
        printf "%s %-36s %3s runs finished\n" "$mark" "$name" "$n_done"
    done
    [ "$found" = 0 ] && echo "no workspaces yet in $WORKSPACES_DIR/" || echo "(* = most recently used)"
    exit 0
fi

if [ "$NEW_WORKSPACE" = 1 ]; then
    WORKSPACE="$(slug "${WORKSPACE_NAME:-$(date '+%Y-%m-%d_%H%M%S')}")"
    [ -z "$WORKSPACE" ] && { echo "workspace name \"$WORKSPACE_NAME\" has no usable characters"; exit 2; }
    if [ -d "$WORKSPACES_DIR/$WORKSPACE" ] && [ "$LIST" = 0 ]; then
        echo "workspace \"$WORKSPACE\" already exists; pick another name, or drop --workspace to continue the latest"
        exit 2
    fi
else
    WORKSPACE="$(latest_workspace)"
    if [ -z "$WORKSPACE" ] || [ ! -d "$WORKSPACES_DIR/$WORKSPACE" ]; then
        WORKSPACE="$(date '+%Y-%m-%d_%H%M%S')"
        NEW_WORKSPACE=1
    fi
fi
CK="$WORKSPACES_DIR/$WORKSPACE"

join() { local IFS=-; echo "$*"; }
ARGS=(--dataset "$DATASET" --seed "$SEED")
SUFFIX=""
[ "$DATASET" != taxibj ] && SUFFIX="_$DATASET"
if [ -n "$TIN$TOUT" ]; then
    [ -z "$TIN" ] || [ -z "$TOUT" ] && { echo "give both --tin and --tout"; exit 2; }
    ARGS+=(--t-in "$TIN" --t-out "$TOUT")
    SUFFIX="${SUFFIX}_in${TIN}_out${TOUT}"
fi
n_files=$(( ${#TRAIN_FILES[@]} + ${#VAL_FILES[@]} + ${#TEST_FILES[@]} ))
if [ "$n_files" -gt 0 ]; then
    if [ ${#TRAIN_FILES[@]} -eq 0 ] || [ ${#VAL_FILES[@]} -eq 0 ] || [ ${#TEST_FILES[@]} -eq 0 ]; then
        echo "splitting by file needs all three: --train-files, --val-files and --test-files"; exit 2
    fi
    ARGS+=(--train-files "${TRAIN_FILES[@]}" --val-files "${VAL_FILES[@]}" --test-files "${TEST_FILES[@]}")
    SUFFIX="${SUFFIX}_tr$(join "${TRAIN_FILES[@]}")_va$(join "${VAL_FILES[@]}")_te$(join "${TEST_FILES[@]}")"
fi
[ -n "$TRAIN_SIZE" ] && ARGS+=(--train-size "$TRAIN_SIZE") && SUFFIX="${SUFFIX}_ntr$TRAIN_SIZE"
[ -n "$VAL_SIZE" ] && ARGS+=(--val-size "$VAL_SIZE") && SUFFIX="${SUFFIX}_nva$VAL_SIZE"
[ -n "$TEST_SIZE" ] && ARGS+=(--test-size "$TEST_SIZE") && SUFFIX="${SUFFIX}_nte$TEST_SIZE"
[ "$SEED" != "$DEFAULT_SEED" ] && SUFFIX="${SUFFIX}_seed$SEED"
if [ "$EXTERNAL" = 1 ]; then
    [ "$DATASET" != taxibj ] && { echo "--external needs --dataset taxibj (only TaxiBJ has external data)"; exit 2; }
    ARGS+=(--external)
    SUFFIX="${SUFFIX}_ext"
fi
[ -n "$EPOCHS" ] && ARGS+=(--epochs "$EPOCHS")

selected() {
    [ "$EXTERNAL" = 1 ] && [[ "$1" != stresnet_* ]] && return 1
    [ ${#FILTERS[@]} -eq 0 ] && return 0
    local f
    for f in "${FILTERS[@]}"; do [[ "$1" == *"$f"* ]] && return 0; done
    return 1
}

if [ "$LIST" = 1 ]; then
    state="continuing"; [ "$NEW_WORKSPACE" = 1 ] && state="new"
    echo "workspace: $CK ($state)"
    for r in "${RUNS[@]}"; do
        read -r name model loss <<< "$r"
        script="train.py --model $model --loss $loss"
        [ "$model" = astgnn ] && script="astgnn.py run"
        batch="batch $(batch_size_for "$model")"
        [ "$batch" = "batch " ] && batch="batch default"
        [ "$model" = copy_last ] && batch="-"
        selected "$name" && printf "  %-26s %-50s %-14s -> %s%s.pt\n" "$name" "$script" "$batch" "$name" "$SUFFIX"
    done
    exit 0
fi

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f .env/bin/activate ]; then
    source .env/bin/activate
fi
PY="${PYTHON:-python}"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export WANDB_RUN_GROUP="$WORKSPACE"
WORKERS=2
[ "$(uname)" = Darwin ] && WORKERS=0
if ! "$PY" -c "import torch" 2>/dev/null; then
    echo "PyTorch not found. Run:  source setup.sh"
    exit 1
fi

open_workspace() {
    mkdir -p "$CK/logs"
    if [ "$NEW_WORKSPACE" = 1 ]; then
        {
            echo "workspace: $WORKSPACE"
            echo "created: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "host: $(hostname)"
            echo "git: $(git rev-parse --short HEAD 2>/dev/null || echo n/a)$(git diff --quiet 2>/dev/null || echo ' (uncommitted changes)')"
        } > "$CK/workspace.txt"
        echo "=== new workspace \"$WORKSPACE\"  [$CK]"
    else
        n_done=$(find "$CK" -maxdepth 1 -name '*.done' | wc -l | tr -d ' ')
        echo "=== continuing workspace \"$WORKSPACE\" ($n_done runs already finished)  [$CK]"
    fi
    echo "invoked $(date '+%Y-%m-%d %H:%M:%S')${SLURM_JOB_ID:+ (Slurm job $SLURM_JOB_ID)}: ./train_all.sh $ORIG_ARGS" >> "$CK/workspace.txt"
    echo "$WORKSPACE" > "$LATEST_FILE"
}

final_values() {
    local line="$1" program
    program='{
        for (i = 3; i <= NF; i++) if (index($i, "=")) { split($i, kv, "="); v[kv[1]] = kv[2] }
        n = split(keys, names, " ")
        for (i = 1; i <= n; i++) printf "\t%s", v[names[i]]
        printf "\n"
    }'
    echo "$line" | awk -v keys="${METRICS[*]}" "$program"
}

RESULTS="$CK/results.tsv"
HEADER="$(printf 'run'; printf '\t%s' "${METRICS[@]}")"
record_result() {
    local tag="$1" log="$2" line="" i
    for i in 1 2 3 4 5 6 7 8 9 10; do
        line="$(grep -m1 '^FINAL ' "$log" 2>/dev/null)" && break
        sleep 1
    done
    [ -z "$line" ] && return
    if [ ! -f "$RESULTS" ] || [ "$(head -1 "$RESULTS")" != "$HEADER" ]; then
        [ -f "$RESULTS" ] && mv "$RESULTS" "$RESULTS.old"
        echo "$HEADER" > "$RESULTS"
    fi
    awk -v t="$tag" -F'\t' '$1 != t' "$RESULTS" > "$RESULTS.tmp"
    printf '%s%s\n' "$tag" "$(final_values "$line")" >> "$RESULTS.tmp"
    mv "$RESULTS.tmp" "$RESULTS"
}

update_leaderboard() {
    [ "${WANDB_MODE:-}" = disabled ] && return
    "$PY" tools/leaderboard.py --workspace "$CK" > "$CK/logs/leaderboard.log" 2>&1 \
        || echo "!!! couldn't update the W&B leaderboard; see $CK/logs/leaderboard.log"
}

print_table() {
    awk -F'\t' '{
        printf "  %-40s", $1
        for (i = 2; i <= NF; i++) printf " %12s", $i
        printf "\n"
    }'
}

clean_log() {
    local esc
    esc="$(printf '\033')"
    tr '\r' '\n' | sed "s/${esc}\[[0-9;]*[A-Za-z]//g" | grep -v -E '^[[:space:]]*$|%\|'
}

open_workspace
DONE=() CHECKED=() SKIPPED=() FAILED=() LOGS=()
START=$(date +%s)
for r in "${RUNS[@]}"; do
    read -r name model loss <<< "$r"
    selected "$name" || continue
    tag="$name$SUFFIX"
    ckpt="$CK/$tag.pt"
    done_mark="$CK/$tag.done"
    if [ "$model" = astgnn ]; then
        cmd=(astgnn.py run "${ARGS[@]}")
    else
        cmd=(train.py --model "$model" --loss "$loss" "${ARGS[@]}" --workers "$WORKERS")
    fi
    batch="$(batch_size_for "$model")"
    [ -n "$batch" ] && [ "$model" != copy_last ] && cmd+=(--batch-size "$batch")

    if [ "$NO_SAVE" = 1 ]; then
        log="$CK/logs/$tag.check.log"
        cmd+=(--no-save)
        : > "$log"
    else
        log="$CK/logs/$tag.log"
        if [ -f "$done_mark" ] && [ "$FORCE" = 0 ]; then
            if grep -qx "$PROTOCOL" "$done_mark"; then
                echo "=== $tag: already finished, skipping (use --force to retrain)"
                SKIPPED+=("$tag"); continue
            fi
            echo "=== $tag: finished with an older version of the code; retraining"
        fi
        rm -f "$done_mark"
        cmd+=(--save "$ckpt")
        if [ "$model" != astgnn ] && [ "$FORCE" = 0 ]; then
            cmd+=(--resume)
            [ -f "$CK/$tag.last.pt" ] || : > "$log"
        else
            : > "$log"
        fi
    fi
    LOGS[${#LOGS[@]}]="$tag $log"

    export WANDB_NAME="$tag"
    echo "=== $tag: $PY ${cmd[*]}"
    echo "    log: $log"
    t0=$(date +%s)
    if ! "$PY" -u "${cmd[@]}" 2>&1 | tee >(clean_log >> "$log"); then
        echo "!!! $tag failed; see $log"; FAILED+=("$tag"); continue
    fi
    minutes="$(( ($(date +%s) - t0) / 60 )) min"
    if [ "$NO_SAVE" = 1 ]; then
        CHECKED+=("$tag ($minutes)")
    elif [ -f "$ckpt" ]; then
        echo "$PROTOCOL" > "$done_mark"
        record_result "$tag" "$log"
        update_leaderboard
        DONE+=("$tag ($minutes)")
    else
        echo "!!! $tag finished but $ckpt was not written"; FAILED+=("$tag")
    fi
done

log_of() {
    local entry
    for entry in "${LOGS[@]}"; do [ "${entry%% *}" = "$1" ] && { echo "${entry#* }"; return; }; done
}

echo
echo "=== summary ($(( $(date +%s) - START )) sec total)  [workspace $WORKSPACE]"
for t in "${DONE[@]}"; do echo "  trained  $t"; done
for t in "${CHECKED[@]}"; do echo "  ran      $t (nothing saved)"; done
for t in "${SKIPPED[@]}"; do echo "  skipped  $t (already finished)"; done
for t in "${FAILED[@]}"; do echo "  FAILED   $t  (log: $(log_of "$t"))"; done
echo
if [ "$NO_SAVE" = 1 ]; then
    echo "=== MSE / RMSE / MAE in raw units (checkpoint picked on val; nothing was saved)"
    {
        echo "$HEADER"
        for t in "${CHECKED[@]}"; do
            t="${t%% (*}"
            line="$(grep -m1 '^FINAL ' "$(log_of "$t")" 2>/dev/null)"
            [ -n "$line" ] && printf '%s%s\n' "$t" "$(final_values "$line")"
        done
    } | print_table
elif [ -f "$RESULTS" ]; then
    echo "=== MSE / RMSE / MAE in raw units (checkpoint picked on val)  [$RESULTS]"
    print_table < "$RESULTS"
fi
[ ${#FAILED[@]} -eq 0 ]
