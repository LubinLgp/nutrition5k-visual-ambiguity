#!/bin/bash
# auxmass_seeds.sh — SOUMETTEUR (depuis CNN/, PAS via sbatch lui-même).
# EXPÉRIENCE W2 (reviewer) : tête auxiliaire supervisée sur les MASSES PAR INGRÉDIENT.
# Le modèle (gt / directe / no-skip) reçoit une supervision directe sur la quantité
# qu'on déclare manquante (les masses par ingrédient), en plus des totaux. On lit :
#   - test_carb_PMAE          : carb de la tête directe (comme d'habitude)
#   - test_carb_from_aux_PMAE : carb reconstruit depuis les masses/ingrédient PRÉDITES
#                               (× densités connues) — c.-à-d. "P2 depuis l'image"
#   - MAE masses/ingr         : à quel point l'image retrouve les masses par ingrédient
#
# Lecture : si test_carb_from_aux_PMAE reste ~20 % (loin du P2 oracle 1.3 %), l'image
# NE contient PAS la composition, même supervisée dessus -> la thèse devient très dure
# à attaquer. Si ça chute vers ~1-5 %, l'image la contient et il faut reformuler.
#
#   bash scripts/auxmass_seeds.sh                 # graines 0..4
#   SEEDS="0 1 2" AUX_LAMBDA=1.0 bash scripts/auxmass_seeds.sh
#
# ⚠️ SMOKE TEST D'ABORD (tête auxiliaire = nouveau code jamais entraîné) :
#      EPOCHS=1 SEEDS=0 bash scripts/auxmass_seeds.sh
#    puis vérifier : grep -E 'aux W2|Traceback' logs_auxmass_seeds/aux_s0_*.out|*.err
# ⚠️ EPOCHS/BATCH_SIZE/LR IDENTIQUES à visual_seeds.sh pour comparer à 20.1±0.3.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${FUSION_DATA_ROOT:-/mnt/beegfs/projects/gluci/Fusion_Midas_Flava/data/nutrition5k}"
SEEDS="${SEEDS:-0 1 2 3 4}"
EPOCHS="${EPOCHS:-150}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-5e-5}"
AUX_LAMBDA="${AUX_LAMBDA:-1.0}"
LOG_DIR="$PROJECT_DIR/logs_auxmass_seeds"
mkdir -p "$LOG_DIR" "$PROJECT_DIR/checkpoints"

echo "=== Tête aux masses/ingrédient (W2) | graines : $SEEDS (epochs=$EPOCHS batch=$BATCH_SIZE lr=$LR λ_aux=$AUX_LAMBDA) ==="
for s in $SEEDS; do
    JOB_ID=$(sbatch --parsable \
        --job-name="aux_s${s}" \
        --output="${LOG_DIR}/aux_s${s}_%j.out" \
        --error="${LOG_DIR}/aux_s${s}_%j.err" \
        --time=24:00:00 --gres=gpu:1 --cpus-per-task=8 --mem=48G \
        --wrap="
export FUSION_DATA_ROOT='${DATA_ROOT}'
source '${PROJECT_DIR}/scripts/_init.sh'
echo '--- TRAIN aux masses/ingrédient / gt / no-skip / seed=${s} ---'
\"\$PYTHON\" -u train_fusion.py --depth-source gt --no-skip --aux-ingr-mass --aux-lambda ${AUX_LAMBDA} \
  --seed ${s} --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --lr ${LR} --num-workers 4
")
    printf "  job %-9s  seed=%s\n" "$JOB_ID" "$s"
done
echo ""
echo "Fini -> les deux carb par graine :"
echo "  grep -HE 'test_carb_PMAE|carb_from_aux' ${LOG_DIR}/aux_s*_*.out"
echo "Si carb_from_aux reste ~20% (>> P2 oracle 1.3%), l'image ne récupère pas la composition."
