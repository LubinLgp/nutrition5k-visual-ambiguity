#!/bin/bash
# lambda_sweep.sh — SOUMETTEUR (depuis CNN/, PAS via sbatch lui-même).
# Sweep du poids λ de l'alignement FLAVA texte (réponse au reviewer : « votre
# conclusion 'FLAVA ne casse pas le mur' est peut-être due à un mauvais λ »). On
# ré-entraîne la variante gt / directe / no-skip AVEC FLAVA-texte à plusieurs λ.
# Si aucun λ ne passe sous la référence sans-FLAVA (20.1±0.3), la conclusion tient.
#
# λ par défaut : 1, 3, 10 (le défaut du code est 10). Chaque (λ, graine) = 1 job.
#   bash scripts/lambda_sweep.sh                      # λ∈{1,3,10} × graines 0..2
#   LAMBDAS="0.3 1 3 10" SEEDS="0 1 2" bash scripts/lambda_sweep.sh
#
# ⚠️ EPOCHS/BATCH_SIZE/LR IDENTIQUES à la campagne d'origine. Carb imprimé par
#    train_fusion.py (test_carb_PMAE=...). --tag évite les collisions de checkpoint.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${FUSION_DATA_ROOT:-/mnt/beegfs/projects/gluci/Fusion_Midas_Flava/data/nutrition5k}"
LAMBDAS="${LAMBDAS:-1 3 10}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-150}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-5e-5}"
LOG_DIR="$PROJECT_DIR/logs_lambda_sweep"
mkdir -p "$LOG_DIR" "$PROJECT_DIR/checkpoints"

echo "=== Sweep λ (FLAVA-texte) | λ : $LAMBDAS | graines : $SEEDS  (epochs=$EPOCHS batch=$BATCH_SIZE lr=$LR) ==="
for lam in $LAMBDAS; do
    tag="lam$(echo "$lam" | tr -d '.-')"      # 0.3 -> lam03, 10 -> lam10
    for s in $SEEDS; do
        JOB_ID=$(sbatch --parsable \
            --job-name="${tag}_s${s}" \
            --output="${LOG_DIR}/${tag}_s${s}_%j.out" \
            --error="${LOG_DIR}/${tag}_s${s}_%j.err" \
            --time=24:00:00 --gres=gpu:1 --cpus-per-task=8 --mem=48G \
            --wrap="
export FUSION_DATA_ROOT='${DATA_ROOT}'
source '${PROJECT_DIR}/scripts/_init.sh'
echo '--- TRAIN gt / no-skip / FLAVA-txt λ=${lam} / seed=${s} ---'
\"\$PYTHON\" -u train_fusion.py --depth-source gt --no-skip --flava-txt --align-lambda ${lam} \
  --tag ${tag} --seed ${s} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --lr ${LR} --num-workers 4
")
        printf "  job %-9s  λ=%s seed=%s\n" "$JOB_ID" "$lam" "$s"
    done
done
echo ""
echo "Fini -> carb par (λ, graine) :  grep -H 'test_carb_PMAE' ${LOG_DIR}/*.out"
echo "Si aucun λ ne bat 20.1±0.3, l'alignement vision-langage ne casse pas le mur."
