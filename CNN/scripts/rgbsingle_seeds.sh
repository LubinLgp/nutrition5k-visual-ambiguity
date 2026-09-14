#!/bin/bash
# rgbsingle_seeds.sh — SOUMETTEUR (depuis CNN/, PAS via sbatch lui-même).
# VRAI RGB-only mono-branche (--rgb-only) : aucune branche profondeur, aucune CAB —
# les features RGB alimentent directement la fusion multi-échelle. À distinguer de
# rgbonly_seeds.sh (--no-depth = profondeur permutée). Soutient la conclusion de
# déploiement « une simple image RGB suffit » (réponse au reviewer).
#
# Comparer : RGB-only mono-branche vs référence RGB-D (20.1±0.3) et vs profondeur
# permutée (20.9±0.2). S'ils coïncident, la profondeur n'apporte rien au glucide.
#
#   bash scripts/rgbsingle_seeds.sh                 # graines 0..4
#   SEEDS="0 1 2" bash scripts/rgbsingle_seeds.sh
#
# ⚠️ EPOCHS/BATCH_SIZE/LR IDENTIQUES à visual_seeds.sh (comparabilité). Le carb de
#    test est imprimé par train_fusion.py (test_carb_PMAE=...).
# ⚠️ FAIS D'ABORD UN SMOKE TEST 1 ÉPOQUE (EPOCHS=1 SEEDS=0) : le modèle mono-branche
#    est nouveau (jamais entraîné), vérifie qu'il tourne avant de lancer 5 graines.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${FUSION_DATA_ROOT:-/mnt/beegfs/projects/gluci/Fusion_Midas_Flava/data/nutrition5k}"
SEEDS="${SEEDS:-0 1 2 3 4}"
EPOCHS="${EPOCHS:-150}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-5e-5}"
LOG_DIR="$PROJECT_DIR/logs_rgbsingle_seeds"
mkdir -p "$LOG_DIR" "$PROJECT_DIR/checkpoints"

echo "=== RGB-only MONO-BRANCHE (--rgb-only) | graines : $SEEDS  (epochs=$EPOCHS batch=$BATCH_SIZE lr=$LR) ==="
for s in $SEEDS; do
    JOB_ID=$(sbatch --parsable \
        --job-name="rgbs_s${s}" \
        --output="${LOG_DIR}/rgbs_s${s}_%j.out" \
        --error="${LOG_DIR}/rgbs_s${s}_%j.err" \
        --time=24:00:00 --gres=gpu:1 --cpus-per-task=8 --mem=48G \
        --wrap="
export FUSION_DATA_ROOT='${DATA_ROOT}'
source '${PROJECT_DIR}/scripts/_init.sh'
echo '--- TRAIN RGB-only mono-branche / gt / no-skip / seed=${s} ---'
\"\$PYTHON\" -u train_fusion.py --depth-source gt --no-skip --rgb-only --seed ${s} \
  --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --lr ${LR} --num-workers 4
")
    printf "  job %-9s  seed=%s\n" "$JOB_ID" "$s"
done
echo ""
echo "Fini -> carb par graine :  grep -H 'test_carb_PMAE' ${LOG_DIR}/rgbs_s*_*.out"
echo "Comparer à RGB-D 20.1±0.3 et profondeur permutée 20.9±0.2."
