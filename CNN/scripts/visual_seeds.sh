#!/bin/bash
# visual_seeds.sh — SOUMETTEUR (à lancer depuis CNN/, pas via sbatch lui-même).
# Relance la MEILLEURE variante visuelle (gt / sans FLAVA / directe / no-skip = celle qui
# donne carb ~19.6%) sur plusieurs GRAINES, pour donner des barres d'erreur au modèle
# visuel et rendre la comparaison avec P1b (baselines) pleinement symétrique.
# Chaque graine = 1 job GPU : entraînement (train_fusion.py --seed s) puis éval (eval.py),
# qui écrit son CSV par plat (resultats/fusion_gt_all_best_s<seed>.csv).
#
#   bash scripts/visual_seeds.sh                 # graines 0..4, hyperparams par défaut
#   SEEDS="0 1 2" BATCH_SIZE=16 bash scripts/visual_seeds.sh
#
# ⚠️ IMPORTANT : mets EPOCHS/BATCH_SIZE/LR aux MÊMES valeurs que la campagne d'origine
#    (celle qui a produit fusion_gt_all_best.pt = carb 19.6%), sinon les graines ne seront
#    pas comparables à ce chiffre. Demande-les à ton collègue si tu as un doute.
#    Défauts ci-dessous = ceux de scripts/train_fusion_gt.sh.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"   # .../CNN
DATA_ROOT="${FUSION_DATA_ROOT:-/mnt/beegfs/projects/gluci/Fusion_Midas_Flava/data/nutrition5k}"
SEEDS="${SEEDS:-0 1 2 3 4}"
EPOCHS="${EPOCHS:-150}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-5e-5}"
LOG_DIR="$PROJECT_DIR/logs_visual_seeds"
mkdir -p "$LOG_DIR" "$PROJECT_DIR/resultats" "$PROJECT_DIR/checkpoints"

echo "=== Soumission graines : $SEEDS  (epochs=$EPOCHS batch=$BATCH_SIZE lr=$LR) ==="
for s in $SEEDS; do
    CKPT="checkpoints/fusion_gt_all_best_s${s}.pt"
    JOB_ID=$(sbatch --parsable \
        --job-name="vis_s${s}" \
        --output="${LOG_DIR}/vis_s${s}_%j.out" \
        --error="${LOG_DIR}/vis_s${s}_%j.err" \
        --time=24:00:00 --gres=gpu:1 --cpus-per-task=8 --mem=48G \
        --wrap="
export FUSION_DATA_ROOT='${DATA_ROOT}'
source '${PROJECT_DIR}/scripts/_init.sh'
echo '--- TRAIN gt/none/direct/no-skip seed=${s} ---'
\"\$PYTHON\" -u train_fusion.py --depth-source gt --no-skip --seed ${s} \
  --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --lr ${LR} --num-workers 4
echo '--- EVAL seed=${s} (écrit resultats/fusion_gt_all_best_s${s}.csv) ---'
\"\$PYTHON\" -u eval.py --depth-source gt --no-skip --fusion-ckpt '${CKPT}'
")
    printf "  job %-9s  seed=%s  ->  %s\n" "$JOB_ID" "$s" "$CKPT"
done

echo ""
echo "Quand tout est fini, agréger le carb par graine :"
echo "  grep -H 'carb:' ${LOG_DIR}/vis_s*_*.out"
