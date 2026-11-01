#!/usr/bin/env bash
# Train every model / loss combination on TaxiBJ and put the best weights in checkpoints/.
#
#   ./train_all.sh                      # all runs, default epochs
#   ./train_all.sh cnn distvol          # only runs whose name contains "cnn" or "distvol"
#   ./train_all.sh --epochs 5           # same epoch count for every run (quick comparison)
#   ./train_all.sh --quick              # 1 epoch each: checks the whole pipeline end to end
#   ./train_all.sh --tin 6 --tout 6     # other horizon (names get an _in6_out6 suffix)
#   ./train_all.sh --force              # retrain even if the checkpoint already exists
#   ./train_all.sh --list               # show the runs and exit
#
# Runs (Sinkhorn = DistVolLoss: L1 + Sinkhorn + volume, for models whose output is
# non-negative by construction; PredFormer and the MLP can output negative values,
# so they train with MSE only):
#
#   name                 script              model         loss       default epochs
#   cnn_mse              train_distvol.py    BaselineCNN   MSE        50
#   cnn_sinkhorn         train_distvol.py    BaselineCNN   Sinkhorn   50
#   distvol_mse          train_distvol.py    DistVolNet    MSE        50
#   distvol_sinkhorn     train_distvol.py    DistVolNet    Sinkhorn   50
#   mlp_mse              baseline_mlp.py     MLP           MSE        50
#   predformer_mse       train.py            PredFormer    MSE        200 (paper setting)
#                                            (ours: predformer.py, TSST)
#   predformer_official_mse  train.py        PredFormer    MSE        200
#                                            (official repo code in PredFormer/, STS)
#
# Output:
#   checkpoints/<name>.pt          best weights for each run
#   checkpoints/logs/<name>.log    full training log
#   checkpoints/runs/              PredFormer's own run folders (last.pt for --resume)
#
# Progress: each run shows an overall epoch bar plus a per-epoch batch bar (tqdm).
# The .log files keep the per-epoch summary lines; bar redraws are stripped from them.

set -o pipefail      # no -u: macOS's bash 3.2 treats empty arrays as unbound
cd "$(dirname "${BASH_SOURCE[0]}")"

# --- options -----------------------------------------------------------------
EPOCHS="" TIN=4 TOUT=4 FORCE=0 LIST=0 FILTERS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --epochs) EPOCHS="$2"; shift 2 ;;
        --quick)  EPOCHS=1; shift ;;
        --tin)    TIN="$2"; shift 2 ;;
        --tout)   TOUT="$2"; shift 2 ;;
        --force)  FORCE=1; shift ;;
        --list)   LIST=1; shift ;;
        -h|--help) sed -n '2,38p' "$0"; exit 0 ;;
        -*) echo "unknown option: $1 (see --help)"; exit 2 ;;
        *)  FILTERS+=("$1"); shift ;;
    esac
done

SUFFIX=""
[ "$TIN" != 4 ] || [ "$TOUT" != 4 ] && SUFFIX="_in${TIN}_out${TOUT}"

# --- environment ----------------------------------------------------------------
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f .env/bin/activate ]; then
    # shellcheck disable=SC1091
    source .env/bin/activate
fi
PY="${PYTHON:-python}"

# Device: CUDA -> Apple MPS -> CPU is picked inside the Python scripts (devices.py).
# Ops that MPS doesn't implement fall back to the CPU instead of crashing.
export PYTORCH_ENABLE_MPS_FALLBACK=1
# Data loading workers: macOS starts workers with "spawn", which re-imports data.py and
# copies the arrays into every worker. Batches here are tiny, so load in the main process.
WORKERS=2
[ "$(uname)" = Darwin ] && WORKERS=0
if ! "$PY" -c "import torch" 2>/dev/null; then
    echo "PyTorch not found. Run:  source setup.sh"
    exit 1
fi

mkdir -p checkpoints/logs
ESC="$(printf '\033')"     # for stripping terminal escape codes from logs (portable sed)

# --- run table: name | default epochs | command (EPOCHS_ and SAVE_ filled in below) ---
RUNS=(
    "cnn_mse|50|train_distvol.py --model cnn --loss mse --tin $TIN --tout $TOUT --epochs EPOCHS_ --save SAVE_ --workers $WORKERS"
    "cnn_sinkhorn|50|train_distvol.py --model cnn --loss sinkhorn --tin $TIN --tout $TOUT --epochs EPOCHS_ --save SAVE_ --workers $WORKERS"
    "distvol_mse|50|train_distvol.py --model distvol --loss mse --tin $TIN --tout $TOUT --epochs EPOCHS_ --save SAVE_ --workers $WORKERS"
    "distvol_sinkhorn|50|train_distvol.py --model distvol --loss sinkhorn --tin $TIN --tout $TOUT --epochs EPOCHS_ --save SAVE_ --workers $WORKERS"
    "mlp_mse|50|baseline_mlp.py --tin $TIN --tout $TOUT --epochs EPOCHS_ --save SAVE_"
    "predformer_mse|200|train.py --dataset taxibj --impl mine --t-in $TIN --t-out $TOUT --epochs EPOCHS_ --out checkpoints/runs --workers $WORKERS"
    "predformer_official_mse|200|train.py --dataset taxibj --impl official --t-in $TIN --t-out $TOUT --epochs EPOCHS_ --out checkpoints/runs --workers $WORKERS"
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
        selected "$name" && printf "%-24s %s epochs -> checkpoints/%s%s.pt\n" "$name" "${EPOCHS:-$ep}" "$name" "$SUFFIX"
    done
    exit 0
fi

# --- train ------------------------------------------------------------------------
DONE=() SKIPPED=() FAILED=()
START=$(date +%s)
for r in "${RUNS[@]}"; do
    IFS='|' read -r name ep cmd <<< "$r"
    selected "$name" || continue
    tag="$name$SUFFIX"
    ckpt="checkpoints/$tag.pt"
    log="checkpoints/logs/$tag.log"

    if [ -f "$ckpt" ] && [ "$FORCE" = 0 ]; then
        echo "=== $tag: $ckpt exists, skipping (use --force to retrain)"
        SKIPPED+=("$tag"); continue
    fi

    cmd="${cmd//EPOCHS_/${EPOCHS:-$ep}}"
    cmd="${cmd//SAVE_/$ckpt}"
    echo "=== $tag: $PY $cmd"
    echo "    log: $log"
    t0=$(date +%s)
    # shellcheck disable=SC2086
    # Bars go to the terminal as usual; the log keeps only finished lines
    # (tqdm redraws with carriage returns, so keep the text after the last \r).
    if "$PY" -u $cmd 2>&1 | tee >(tr '\r' '\n' | sed "s/${ESC}\[[0-9;]*[A-Za-z]//g" | grep -v -E '^[[:space:]]*$|%\|' > "$log"); then
        if [[ "$name" == predformer* ]]; then
            # train.py writes checkpoints/runs/<run>/best.pt; copy it next to the others
            run_dir="taxibj"
            [ "$name" = predformer_official_mse ] && run_dir="taxibj_official"
            [ -n "$SUFFIX" ] && run_dir="${run_dir}${SUFFIX}"
            cp "checkpoints/runs/$run_dir/best.pt" "$ckpt" 2>/dev/null
        fi
        if [ -f "$ckpt" ]; then
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
echo "=== summary ($(( ($(date +%s) - START) / 60 )) min total)"
for t in "${DONE[@]}";    do echo "  trained  $t"; done
for t in "${SKIPPED[@]}"; do echo "  skipped  $t (already had a checkpoint)"; done
for t in "${FAILED[@]}";  do echo "  FAILED   $t  (log: checkpoints/logs/${t}.log)"; done
[ ${#FAILED[@]} -eq 0 ]
