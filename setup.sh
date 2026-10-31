#!/usr/bin/env bash
# Set up and activate the project's Python environment.
#
#   source setup_env.sh          # create/update .env/ and activate it in this shell
#   source setup_env.sh --force  # reinstall packages even if requirements.txt is unchanged
#
# Must be *sourced*: a script run as ./setup_env.sh executes in a child shell,
# so the activation would vanish when it exits.
#
# What it does (safe to re-run; skips work that's already done):
#   1. checks for Python >= 3.10
#   2. creates the virtual environment .env/ (if missing)
#   3. installs requirements.txt (only when the file changed since last install)
#   4. registers a Jupyter kernel named "traffic-pred"
#   5. clones the official PredFormer repo into PredFormer/ (if missing)
#   6. activates .env/

# --- must be sourced ---------------------------------------------------------
if ! (return 0 2>/dev/null); then
    echo "Run this with:  source ${0##*/}"
    echo "(running it directly can't activate the environment in your shell)"
    exit 1
fi

_setup_env() {
    local root venv py force=0 req stamp want have
    root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    venv="$root/.env"
    req="$root/requirements.txt"
    stamp="$venv/.requirements.sha256"
    [ "${1:-}" = "--force" ] && force=1

    # 1. Python >= 3.10
    for py in python3.12 python3.11 python3.10 python3 python; do
        if command -v "$py" >/dev/null 2>&1 &&
           "$py" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
            break
        fi
        py=""
    done
    if [ -z "$py" ]; then
        echo "[setup] Python 3.10+ not found; install it and re-run." >&2
        return 1
    fi

    # 2. virtual environment
    if [ ! -x "$venv/bin/python" ]; then
        echo "[setup] creating virtual environment .env/ with $($py --version)"
        if ! "$py" -m venv "$venv"; then
            echo "[setup] venv creation failed. On Debian/Ubuntu: sudo apt install python3-venv" >&2
            return 1
        fi
        "$venv/bin/python" -m pip install -q --upgrade pip || return 1
    fi

    # 3. packages (skip if requirements.txt hasn't changed)
    if [ ! -f "$req" ]; then
        echo "[setup] $req not found" >&2
        return 1
    fi
    want="$(sha256sum "$req" | cut -d' ' -f1)"
    have="$(cat "$stamp" 2>/dev/null)"
    if [ "$force" = 1 ] || [ "$want" != "$have" ]; then
        echo "[setup] installing requirements.txt (first time can take a few minutes: torch is large)"
        if ! "$venv/bin/python" -m pip install -r "$req"; then
            echo "[setup] pip install failed; fix the error above and re-run" >&2
            return 1
        fi
        echo "$want" > "$stamp"
    else
        echo "[setup] packages up to date"
    fi

    # 4. Jupyter kernel (so notebooks can pick this environment)
    if ! "$venv/bin/jupyter" kernelspec list 2>/dev/null | grep -q "traffic-pred"; then
        "$venv/bin/python" -m ipykernel install --user --name traffic-pred \
            --display-name "Python (traffic-pred)" >/dev/null &&
            echo "[setup] Jupyter kernel 'Python (traffic-pred)' registered"
    fi

    # 5. official PredFormer code (needed only for --impl official)
    if [ ! -d "$root/PredFormer/openstl" ]; then
        if command -v git >/dev/null 2>&1; then
            echo "[setup] cloning official PredFormer repo"
            git clone -q --depth 1 https://github.com/yyyujintang/PredFormer.git "$root/PredFormer" ||
                echo "[setup] clone failed (only needed for --impl official); continuing" >&2
        else
            echo "[setup] git not found; skipping PredFormer clone (only needed for --impl official)" >&2
        fi
    fi

    # 6. activate
    # shellcheck disable=SC1091
    source "$venv/bin/activate" || return 1
    echo "[setup] activated: $(python --version) at $VIRTUAL_ENV"
    python - <<'EOF'
import torch
dev = (f"CUDA ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available()
       else "MPS" if torch.backends.mps.is_available() else "CPU only")
print(f"[setup] torch {torch.__version__}, device: {dev}")
EOF
}

if _setup_env "$@"; then unset -f _setup_env; return 0; fi
unset -f _setup_env
return 1
