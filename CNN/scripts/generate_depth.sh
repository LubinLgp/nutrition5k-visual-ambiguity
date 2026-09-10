#!/bin/bash
#SBATCH --job-name=midas_gen
#SBATCH --output=logs_generate/generate_%j.out
#SBATCH --error=logs_generate/generate_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$ROOT/scripts/_init.sh"
mkdir -p logs_generate depth/midas checkpoints
[[ -f "$FUSION_MIDAS_PRETRAINED/dpt_hybrid_midas.pt" ]] || {
    echo "ERREUR: $FUSION_MIDAS_PRETRAINED/dpt_hybrid_midas.pt manquant"
    exit 1
}
GEN_ARGS=()
[[ "${NO_SKIP:-0}" == "1" ]] && GEN_ARGS+=(--no-skip)
"$PYTHON" -u generate_depth.py --device "${DEVICE:-cuda}" "${GEN_ARGS[@]+"${GEN_ARGS[@]}"}" "$@"
