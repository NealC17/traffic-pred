#!/usr/bin/env bash

if [ -n "${ZSH_VERSION:-}" ]; then
    case "${ZSH_EVAL_CONTEXT:-}" in
        *:file*) ;;
        *) echo "Run this with:  source setup.sh"; exit 1 ;;
    esac
    eval '_setup_self=${(%):-%x}'
else
    if ! (return 0 2>/dev/null); then
        echo "Run this with:  source ${0##*/}"
        echo "(running it directly can't activate the environment in your shell)"
        exit 1
    fi
    _setup_self="${BASH_SOURCE[0]}"
fi

_setup_env() {
    local root venv py force=0 req stamp want have
    root="$(cd "$(dirname "$_setup_self")" && pwd)"
    venv="$root/.env"
    req="$root/requirements.txt"
    stamp="$venv/.requirements.sha256"
    [ "${1:-}" = "--force" ] && force=1

    for py in python3.12 python3.13 python3.11 python3.10 python3 python; do
        if command -v "$py" >/dev/null 2>&1 &&
           "$py" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
            break
        fi
        py=""
    done
    if [ -z "$py" ]; then
        echo "[setup] Python 3.10+ not found." >&2
        if [ "$(uname)" = Darwin ]; then
            echo "        macOS's built-in python3 is too old; install a newer one:  brew install python@3.12" >&2
        fi
        echo "        Then re-run:  source setup.sh" >&2
        return 1
    fi

    if [ ! -x "$venv/bin/python" ]; then
        echo "[setup] creating virtual environment .env/ with $($py --version)"
        if ! "$py" -m venv "$venv"; then
            rm -rf "$venv"          # don't leave a half-built env (python but no pip) behind
            echo "[setup] venv creation failed. On Debian/Ubuntu install the venv package:" >&2
            echo "          sudo apt install python$("$py" -c 'import sys; print("%d.%d" % sys.version_info[:2])')-venv" >&2
            echo "        then run: source setup.sh" >&2
            return 1
        fi
    fi

    if ! "$venv/bin/python" -m pip --version >/dev/null 2>&1; then
        echo "[setup] .env/ has no pip; installing it"
        if ! "$venv/bin/python" -m ensurepip --upgrade >/dev/null 2>&1; then
            echo "[setup] couldn't add pip to .env/. On Debian/Ubuntu install the venv package:" >&2
            echo "          sudo apt install python$("$venv/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')-venv" >&2
            echo "        then run: source setup.sh   (it repairs .env/ in place)" >&2
            return 1
        fi
        rm -f "$stamp"              # packages were never installed into this env
    fi
    "$venv/bin/python" -m pip install -q --upgrade pip || return 1

    if [ ! -f "$req" ]; then
        echo "[setup] $req not found" >&2
        return 1
    fi
    if command -v sha256sum >/dev/null 2>&1; then     # Linux
        want="$(sha256sum "$req" | cut -d' ' -f1)"
    else                                              # macOS
        want="$(shasum -a 256 "$req" | cut -d' ' -f1)"
    fi
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

    if ! "$venv/bin/jupyter" kernelspec list 2>/dev/null | grep -q "traffic-pred"; then
        "$venv/bin/python" -m ipykernel install --user --name traffic-pred \
            --display-name "Python (traffic-pred)" >/dev/null &&
            echo "[setup] Jupyter kernel 'Python (traffic-pred)' registered"
    fi

    local name url
    for name in PredFormer ASTGNN STDCformer; do
        [ -d "$root/external/$name" ] && continue
        case "$name" in
            PredFormer) url="https://github.com/yyyujintang/PredFormer.git" ;;
            ASTGNN) url="https://github.com/guoshnBJTU/ASTGNN.git" ;;
            STDCformer) url="https://github.com/GeoX-Lab/STDCformer.git" ;;
        esac
        if command -v git >/dev/null 2>&1; then
            echo "[setup] cloning $name into external/$name"
            git clone -q --depth 1 "$url" "$root/external/$name" ||
                echo "[setup] clone of $name failed (only needed for that model); continuing" >&2
        else
            echo "[setup] git not found; skipping the $name clone" >&2
        fi
    done

    source "$venv/bin/activate" || return 1
    echo "[setup] activated: $(python --version) at $VIRTUAL_ENV"
    python - <<'EOF'
import torch
dev = (f"CUDA ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available()
       else "MPS" if torch.backends.mps.is_available() else "CPU only")
print(f"[setup] torch {torch.__version__}, device: {dev}")
EOF
}

if _setup_env "$@"; then unset -f _setup_env; unset _setup_self; return 0; fi
unset -f _setup_env; unset _setup_self
return 1
