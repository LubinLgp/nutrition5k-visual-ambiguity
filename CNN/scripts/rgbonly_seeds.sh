#!/bin/bash
# rgbonly_seeds.sh — SOUMETTEUR (à lancer depuis CNN/, PAS via sbatch lui-même).
# Ablation RGB-only (réponse au reviewer B15). Ré-entraîne la variante de RÉFÉRENCE
# (gt / sans FLAVA / directe / no-skip = celle qui donne carb ~20%) MAIS avec
# --no-depth : la profondeur est PERMUTÉE dans le batch (chaque RGB reçoit la
# profondeur d'un autre plat), ce qui casse la correspondance RGB↔profondeur — la
# profondeur ne porte plus aucune info alignée. Mesure sa contribution informationnelle.
#
# Comparaison attendue : carb RGB-only ≈ carb de référence (gt) → la profondeur
# n'apporte rien au glucide (la branche profondeur peut être retirée).
#
#   bash scripts/rgbonly_seeds.sh                 # graines 0..4, mêmes hyperparams que la réf
#   SEEDS="0 1 2" bash scripts/rgbonly_seeds.sh
#
# ⚠️ Garde EPOCHS/BATCH_SIZE/LR IDENTIQUES à visual_seeds.sh (sinon non comparable
#    au carb de référence 20.1±0.3 %). Le carb de test est imprimé par train_fusion.py
#    (test_carb_PMAE=...) — pas besoin de lancer eval.py.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"   # .../CNN
DATA_ROOT="${FUSION_DATA_ROOT:-/mnt/beegfs/projects/gluci/Fusion_Midas_Flava/data/nutrition5k}"
SEEDS="${SEEDS:-0 1 2 3 4}"
EPOCHS="${EPOCHS:-150}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-5e-5}"
LOG_DIR="$PROJECT_DIR/logs_rgbonly_seeds"
mkdir -p "$LOG_DIR" "$PROJECT_DIR/checkpoints"

echo "=== RGB-only (--no-depth) | graines : $SEEDS  (epochs=$EPOCHS batch=$BATCH_SIZE lr=$LR) ==="
for s in $SEEDS; do
    CKPT="checkpoints/fusion_gt_all_best_rgbonly_s${s}.pt"
    JOB_ID=$(sbatch --parsable \
        --job-name="rgb_s${s}" \
        --output="${LOG_DIR}/rgb_s${s}_%j.out" \
        --error="${LOG_DIR}/rgb_s${s}_%j.err" \
        --time=24:00:00 --gres=gpu:1 --cpus-per-task=8 --mem=48G \
        --wrap="
export FUSION_DATA_ROOT='${DATA_ROOT}'
source '${PROJECT_DIR}/scripts/_init.sh'
echo '--- TRAIN RGB-only (gt loader, depth ZÉRO) / no-skip / seed=${s} ---'
\"\$PYTHON\" -u train_fusion.py --depth-source gt --no-skip --no-depth --seed ${s} \
  --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --lr ${LR} --num-workers 4
")
    printf "  job %-9s  seed=%s  ->  %s\n" "$JOB_ID" "$s" "$CKPT"
done

echo ""
echo "Quand tout est fini, récupérer le carb RGB-only par graine :"
echo "  grep -H 'test_carb_PMAE' ${LOG_DIR}/rgb_s*_*.out"
echo "Comparer à la référence RGB-D (visual_seeds) : carb 20.1±0.3 %."
