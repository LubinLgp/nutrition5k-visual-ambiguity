#!/bin/bash
# density_seeds.sh — SOUMETTEUR (à lancer depuis CNN/, pas via sbatch lui-même).
# Entraîne le modèle DENSITÉ (gt / sans FLAVA / no-skip) sur plusieurs SEEDS, pour donner
# des barres d'erreur à la décomposition composition/portion (PAPER_PLAN §4.1, actuellement
# sur 1 seed). Chaque seed = 1 job GPU : entraînement (train_fusion.py --density --seed s)
# puis éval (eval.py --density), qui écrit le CSV par plat AVEC les colonnes de densité
# (preddens_*/truedens_*) dans resultats/fusion_gt_all_d_best_s<seed>.csv.
#
#   bash scripts/density_seeds.sh                 # seeds 0..4, hyperparams par défaut
#   SEEDS="0 1 2" BATCH_SIZE=16 bash scripts/density_seeds.sh
#
# ⚠️ Mets EPOCHS/BATCH_SIZE/LR aux MÊMES valeurs que la campagne d'origine (celle qui a
#    produit fusion_gt_all_d_best.pt), sinon les seeds ne seront pas comparables. Défauts =
#    ceux de scripts/train_fusion_gt.sh.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"   # .../CNN
DATA_ROOT="${FUSION_DATA_ROOT:-/mnt/beegfs/projects/gluci/Fusion_Midas_Flava/data/nutrition5k}"
SEEDS="${SEEDS:-0 1 2 3 4}"
EPOCHS="${EPOCHS:-150}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-5e-5}"
LOG_DIR="$PROJECT_DIR/logs_density_seeds"
mkdir -p "$LOG_DIR" "$PROJECT_DIR/resultats" "$PROJECT_DIR/checkpoints"

echo "=== Soumission seeds : $SEEDS  (density gt/none ; epochs=$EPOCHS batch=$BATCH_SIZE lr=$LR) ==="
for s in $SEEDS; do
    CKPT="checkpoints/fusion_gt_all_d_best_s${s}.pt"
    JOB_ID=$(sbatch --parsable \
        --job-name="dens_s${s}" \
        --output="${LOG_DIR}/dens_s${s}_%j.out" \
        --error="${LOG_DIR}/dens_s${s}_%j.err" \
        --time=24:00:00 --gres=gpu:1 --cpus-per-task=8 --mem=48G \
        --wrap="
export FUSION_DATA_ROOT='${DATA_ROOT}'
source '${PROJECT_DIR}/scripts/_init.sh'
echo '--- TRAIN densité gt/none/no-skip seed=${s} ---'
\"\$PYTHON\" -u train_fusion.py --depth-source gt --no-skip --density --seed ${s} \
  --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --lr ${LR} --num-workers 4
echo '--- EVAL seed=${s} (écrit resultats/fusion_gt_all_d_best_s${s}.csv) ---'
\"\$PYTHON\" -u eval.py --depth-source gt --no-skip --density --fusion-ckpt '${CKPT}'
")
    printf "  job %-9s  seed=%s  ->  %s\n" "$JOB_ID" "$s" "$CKPT"
done

echo ""
echo "Quand tout est fini, agréger (carb, densité × masse prédite ET × masse vraie) :"
echo "  grep -B1 'carb:' ${LOG_DIR}/dens_s*_*.out | grep -E 'masse (vraie|prédite)|carb:'"
