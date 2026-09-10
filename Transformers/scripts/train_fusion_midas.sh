#!/bin/bash
#SBATCH --job-name=swin_fusion_midas
#SBATCH --output=logs_train_fusion/midas_%j.out
#SBATCH --error=logs_train_fusion/midas_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$ROOT/scripts/_init.sh"
mkdir -p logs_train_fusion checkpoints
[[ -d depth/midas ]] && [[ -n "$(ls -A depth/midas 2>/dev/null || true)" ]] || {
    echo "ERREUR: depth/midas vide — sbatch scripts/generate_depth.sh"
    exit 1
}
CMD=(
  -u train_fusion.py
  --depth-source midas
  --epochs "${EPOCHS:-150}"
  --batch-size "${BATCH_SIZE:-8}"
  --lr "${LR:-5e-5}"
  --lr-decay "${LR_DECAY:-0.98}"
  --device "${DEVICE:-cuda}"
  --num-workers "${NUM_WORKERS:-0}"
)
FLAVA_TXT=${FLAVA_TXT:-0}
FLAVA_VISUEL=${FLAVA_VISUEL:-0}
[[ "$FLAVA_TXT" == "1" ]] && CMD+=(--flava-txt)
[[ "$FLAVA_VISUEL" == "1" ]] && CMD+=(--flava-vis)
NO_SKIP=${NO_SKIP:-0}
[[ "$NO_SKIP" == "1" ]] && CMD+=(--no-skip)
DENSITY=${DENSITY:-0}   # 1 = mode densité (masse + densités /100g, checkpoints _d)
[[ "$DENSITY" == "1" ]] && CMD+=(--density)
[[ -n "${PRETRAINED:-}" ]] && CMD+=(--pretrained "$PRETRAINED")
"$PYTHON" "${CMD[@]}"
