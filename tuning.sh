#!/usr/bin/env bash
# Run the loss-weight tuning (tune_loss.py) on this machine, in the background.
#
#   ./tuning.sh [start] [tune_loss.py options]   start, or continue, the study in the background
#   ./tuning.sh fg      [tune_loss.py options]   the same in the foreground (Ctrl-C stops it)
#   ./tuning.sh status  [--study NAME]           whether it is running, and the best trials so far
#   ./tuning.sh log     [--study NAME]           follow its log
#   ./tuning.sh stop                             stop it; the trial it was in is queued again
#   ./tuning.sh confirm [K] [--study NAME]       write train_tuned_<study>.sh with the K (3) best
#
# Default: study distvolformer_taxibj until it has 40 finished trials (--until 40). Options go to
# tune_loss.py, e.g.  ./tuning.sh --study dvf_l --until 30 -- --model distvolformer --dim 256 --heads 8
# Trials log to W&B project traffic-pred-hyperparameter-tuning. The log is
# logs/hyperparameter_tuning_<study>.out. Only one study runs at a time: they share the one GPU.

cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
PY=.env/bin/python
[ -x "$PY" ] || { echo "no .env/; run:  source setup.sh"; exit 1; }
DEFAULT_UNTIL=40

running() {   # PIDs of tuning workers (not --report / --print-study calls)
    local pid a found skip argv
    for pid in $(pgrep -f "tune_loss\.py"); do
        mapfile -d '' -t argv < "/proc/$pid/cmdline" 2>/dev/null || continue
        case "${argv[0]##*/}" in python*) ;; *) continue ;; esac   # a python process ...
        found=0 skip=0
        for a in "${argv[@]:1}"; do                                 # ... running tune_loss.py
            case "$a" in
                *tune_loss.py) found=1 ;;
                --report|--print-study|--confirm) skip=1 ;;
            esac
        done
        [ "$found" = 1 ] && [ "$skip" = 0 ] && echo "$pid"
    done
}

with_until() {   # OPTS = the options, plus --until 40 unless they set --until or --trials
    local a
    OPTS=("$@")
    for a in "$@"; do
        [ "$a" = -- ] && break
        case "$a" in --until|--until=*|--trials|--trials=*) return ;; esac
    done
    OPTS=(--until "$DEFAULT_UNTIL" "$@")
}

study_of() { "$PY" tune_loss.py --print-study "$@"; }

cmd="${1:-start}"
case "$cmd" in start|fg|status|log|stop|confirm) shift ;; -*) cmd=start ;; *)
    sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac

case "$cmd" in
start|fg)
    pids="$(running)"
    if [ -n "$pids" ]; then
        echo "tuning is already running (pid $(echo $pids)); ./tuning.sh status / log / stop"
        exit 1
    fi
    study="$(study_of "$@")" || exit 1
    with_until "$@"
    log="logs/hyperparameter_tuning_$study.out"
    mkdir -p logs
    if [ "$cmd" = fg ]; then
        "$PY" -u tune_loss.py "${OPTS[@]}" 2>&1 | tee -a "$log"
        exit "${PIPESTATUS[0]}"
    fi
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') ./tuning.sh ${OPTS[*]}" >> "$log"
    nohup "$PY" -u tune_loss.py "${OPTS[@]}" >> "$log" 2>&1 < /dev/null &
    pid=$!
    sleep 3
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "tune_loss.py exited at once; end of $log:"; tail -20 "$log"; exit 1
    fi
    echo "tuning study $study in the background (pid $pid)"
    echo "  log:     $log   (./tuning.sh log)"
    echo "  results: ./tuning.sh status     stop: ./tuning.sh stop     W&B: traffic-pred-hyperparameter-tuning"
    ;;
status)
    pids="$(running)"
    if [ -n "$pids" ]; then
        for pid in $pids; do echo "running: pid $pid, $(ps -o etime= -p "$pid" | tr -d ' ') elapsed: $(tr '\0' ' ' < "/proc/$pid/cmdline")"; done
    else
        echo "not running"
    fi
    "$PY" tune_loss.py --report "$@"
    ;;
log)
    study="$(study_of "$@")" || exit 1
    log="logs/hyperparameter_tuning_$study.out"
    [ -f "$log" ] || { echo "no $log yet"; exit 1; }
    tail -f "$log" | tr '\r' '\n' | grep --line-buffered -avE '^\s*$|it/s|ep/s'   # skip progress bars
    ;;
stop)
    pids="$(running)"
    [ -z "$pids" ] && { echo "not running"; exit 0; }
    echo "stopping pid $(echo $pids) (the current trial is queued to run again)"
    kill -TERM $pids
    for _ in $(seq 60); do [ -z "$(running)" ] && { echo "stopped; ./tuning.sh continues the study"; exit 0; }; sleep 1; done
    echo "still running after 60 s; kill -9 $(echo $pids) if it is stuck"
    exit 1
    ;;
confirm)
    k=3
    case "${1:-}" in ''|-*) ;; *) k="$1"; shift ;; esac
    "$PY" tune_loss.py --confirm "$k" "$@"
    ;;
esac
