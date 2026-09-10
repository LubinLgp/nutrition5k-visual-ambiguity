#!/bin/bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

VENV_DIR="$PROJECT_DIR/env_Fusion_Midas_Flava"
PY="$VENV_DIR/bin/python"   # pip via "$PY" -m pip : robuste à un déplacement du venv

source "$PROJECT_DIR/env/proxy.sh"
export PIP_DISABLE_PIP_VERSION_CHECK=1

echo "=== Env partagé Fusion_Midas_Flava : $PROJECT_DIR ==="

if command -v conda &>/dev/null; then
    while [[ "${CONDA_DEFAULT_ENV:-}" == "base" ]] || [[ -n "${CONDA_PREFIX:-}" ]]; do
        conda deactivate 2>/dev/null || break
    done
fi

pick_python() {
    local candidates=(
        /mnt/beegfs/projects/ftctinfer/conda_stagiaires/bin/python3
        "$(command -v python3.11 2>/dev/null || true)"
        "$(command -v python3.10 2>/dev/null || true)"
        "$(command -v python3.9 2>/dev/null || true)"
        "$(command -v python3 2>/dev/null || true)"
    )
    local c ver major minor
    for c in "${candidates[@]}"; do
        [[ -z "$c" || ! -x "$c" ]] && continue
        ver="$("$c" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo 0.0)"
        major="${ver%%.*}"
        minor="${ver#*.}"
        if [[ "$major" -eq 3 && "$minor" -ge 9 ]]; then
            echo "$c"
            return 0
        fi
    done
    return 1
}

need_recreate_venv() {
    [[ "${FORCE:-}" == "1" ]] && return 0
    [[ ! -x "$PY" ]] && return 0
    local ver major minor
    ver="$("$PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    major="${ver%%.*}"
    minor="${ver#*.}"
    [[ "$major" -eq 3 && "$minor" -ge 9 ]] && return 1
    return 0
}

PY_SYS="$(pick_python)" || { echo "ERREUR: aucun Python >= 3.9"; exit 1; }
echo "=== Python : $PY_SYS ($("$PY_SYS" --version)) ==="

if need_recreate_venv; then
    rm -rf "$VENV_DIR"
    "$PY_SYS" -m venv "$VENV_DIR"
fi

[[ -x "$PY" ]] || "$PY_SYS" -m venv "$VENV_DIR"

"$PY" -m pip install -U pip wheel

install_torch() {
    if "$PY" -m pip install --no-cache-dir torch torchvision; then
        return 0
    fi
    "$PY" -m pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cu124
}

if ! "$PY" -c "import torch" 2>/dev/null; then
    install_torch
fi

"$PY" -m pip install --no-cache-dir -r "$PROJECT_DIR/env/requirements.txt"

"$PY" -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"
"$PY" -c "import torchvision, numpy, PIL, tqdm, timm; print('deps OK')"

# Venv partagé par toutes les variantes (CNN, Transformers). Les dossiers de sortie
# (logs_*, checkpoints, depth) sont propres à chaque variante et créés par ses scripts.
echo "=== env_Fusion_Midas_Flava pret : $PY ==="
