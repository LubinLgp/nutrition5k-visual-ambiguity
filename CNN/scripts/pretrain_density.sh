#!/bin/bash
#SBATCH --job-name=midas_pretrain
#SBATCH --output=logs_pretrain/pretrain_%j.out
#SBATCH --error=logs_pretrain/pretrain_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$ROOT/scripts/_init.sh"
mkdir -p logs_pretrain checkpoints
# Recipe1M vit dans le dossier data FRÈRE du projet (.../gluci/data). Le défaut de
# config.RECIPE1M_ROOT (calculé depuis __file__ = ROOT.parent.parent/data) tombe
# juste sans rien exporter. Pour pointer ailleurs : RECIPE1M_DATA_ROOT=/chemin sbatch ...
CMD=(
  -u pretrain_density.py
  --epochs "${EPOCHS:-30}"
  --batch-size "${BATCH_SIZE:-64}"
  --lr "${LR:-1e-4}"
  --lr-decay "${LR_DECAY:-0.98}"
  --img-size "${IMG_SIZE:-224}"
  --device "${DEVICE:-cuda}"
  --num-workers "${NUM_WORKERS:-8}"
)
FLAVA_TXT=${FLAVA_TXT:-0}
FLAVA_VISUEL=${FLAVA_VISUEL:-0}
[[ "$FLAVA_TXT" == "1" ]] && CMD+=(--flava-txt)
[[ "$FLAVA_VISUEL" == "1" ]] && CMD+=(--flava-vis)
[[ -n "${LIMIT:-}" ]] && CMD+=(--limit "$LIMIT")
RESUME=${RESUME:-1}   # 1 = reprend depuis _last si présent (jobs limités à 48h) ; 0 = repart à neuf
[[ "$RESUME" == "1" ]] && CMD+=(--resume)
"$PYTHON" "${CMD[@]}"
