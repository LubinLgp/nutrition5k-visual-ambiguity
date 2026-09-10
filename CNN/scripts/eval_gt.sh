#!/bin/bash
#SBATCH --job-name=midas_eval_gt
#SBATCH --output=logs_eval/gt_%j.out
#SBATCH --error=logs_eval/gt_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$ROOT/scripts/_init.sh"
mkdir -p logs_eval
FLAVA_TXT=${FLAVA_TXT:-0}
FLAVA_VISUEL=${FLAVA_VISUEL:-0}
SUFFIX=""
if [[ "$FLAVA_TXT" == "1" || "$FLAVA_VISUEL" == "1" ]]; then
  SUFFIX="_f"; [[ "$FLAVA_TXT" == "1" ]] && SUFFIX="${SUFFIX}t"; [[ "$FLAVA_VISUEL" == "1" ]] && SUFFIX="${SUFFIX}v"
fi
NO_SKIP=${NO_SKIP:-0}
[[ "$NO_SKIP" == "1" ]] && SUFFIX="${SUFFIX}_all"
DENSITY=${DENSITY:-0}   # 1 = checkpoint mode densité (masse + densités /100g, _d)
[[ "$DENSITY" == "1" ]] && SUFFIX="${SUFFIX}_d"
CKPT="checkpoints/fusion_gt${SUFFIX}_best.pt"
[[ -f "$CKPT" ]] || {
    echo "ERREUR: $CKPT manquant — lance d'abord le train correspondant"
    exit 1
}
CMD=(
  -u eval.py
  --depth-source gt
  --device "${DEVICE:-cuda}"
  --batch-size "${BATCH_SIZE:-8}"
  --num-workers "${NUM_WORKERS:-0}"
)
[[ "$FLAVA_TXT" == "1" ]] && CMD+=(--flava-txt)
[[ "$FLAVA_VISUEL" == "1" ]] && CMD+=(--flava-vis)
[[ "$NO_SKIP" == "1" ]] && CMD+=(--no-skip)
[[ "$DENSITY" == "1" ]] && CMD+=(--density)
"$PYTHON" "${CMD[@]}"
