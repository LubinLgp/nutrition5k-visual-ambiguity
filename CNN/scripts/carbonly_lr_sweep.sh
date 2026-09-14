#!/bin/bash
# carbonly_lr_sweep.sh — SOUMETTEUR (depuis CNN/, PAS via sbatch lui-même).
# Test de dilution multi-tâche ÉQUITABLE (réponse au reviewer : la loss géométrique
# a un gradient ∂L/∂L_k = L_geo/(K·L_k) ; passer à un L1 mono-tâche change l'échelle
# effective du gradient, donc à LR/époques égaux le modèle carb-only est peut-être
# sous-entraîné). On balaie le LR pour le carb-only et on prend le meilleur.
# Si même le meilleur LR ne bat pas le multi-tâche (20.1±0.3), pas de dilution.
#
# LR par défaut : 5e-5 (réf), 2e-5, 1e-5. Chaque (LR, graine) = 1 job.
#   bash scripts/carbonly_lr_sweep.sh                    # LR∈{5e-5,2e-5,1e-5} × graines 0..2
#   LRS="1e-4 5e-5 2e-5 1e-5" SEEDS="0 1 2" bash scripts/carbonly_lr_sweep.sh
#
# ⚠️ EPOCHS/BATCH_SIZE identiques à la campagne. Carb imprimé par train_fusion.py
#    (test_carb_PMAE=...). --tag évite les collisions de checkpoint entre LR.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${FUSION_DATA_ROOT:-/mnt/beegfs/projects/gluci/Fusion_Midas_Flava/data/nutrition5k}"
LRS="${LRS:-5e-5 2e-5 1e-5}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-150}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LOG_DIR="$PROJECT_DIR/logs_carbonly_lr"
mkdir -p "$LOG_DIR" "$PROJECT_DIR/checkpoints"

echo "=== Carb-only sweep LR | LR : $LRS | graines : $SEEDS  (epochs=$EPOCHS batch=$BATCH_SIZE) ==="
for lr in $LRS; do
    tag="lr$(echo "$lr" | tr -d '.-')"       # 5e-5 -> lr5e5
    for s in $SEEDS; do
        JOB_ID=$(sbatch --parsable \
            --job-name="co_${tag}_s${s}" \
            --output="${LOG_DIR}/co_${tag}_s${s}_%j.out" \
            --error="${LOG_DIR}/co_${tag}_s${s}_%j.err" \
            --time=24:00:00 --gres=gpu:1 --cpus-per-task=8 --mem=48G \
            --wrap="
export FUSION_DATA_ROOT='${DATA_ROOT}'
source '${PROJECT_DIR}/scripts/_init.sh'
echo '--- TRAIN carb-only / gt / no-skip / lr=${lr} / seed=${s} ---'
\"\$PYTHON\" -u train_fusion.py --depth-source gt --no-skip --carb-only --lr ${lr} \
  --tag ${tag} --seed ${s} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --num-workers 4
")
        printf "  job %-9s  lr=%s seed=%s\n" "$JOB_ID" "$lr" "$s"
    done
done
echo ""
echo "Fini -> carb par (LR, graine) :  grep -H 'test_carb_PMAE' ${LOG_DIR}/*.out"
echo "Si le MEILLEUR LR ne bat pas 20.1±0.3, le plafond n'est pas un artefact multi-tâche."
