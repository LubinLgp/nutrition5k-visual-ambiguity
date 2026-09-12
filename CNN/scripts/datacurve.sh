#!/bin/bash
# datacurve.sh — SOUMETTEUR (à lancer depuis CNN/, PAS via sbatch lui-même).
# Courbe d'apprentissage / effet de la TAILLE DES DONNÉES (réponse au reviewer B14 :
# « le plafond ~20% vient-il d'un manque de données ? »). Ré-entraîne la variante de
# RÉFÉRENCE (gt / sans FLAVA / directe / no-skip) sur des FRACTIONS du train
# (--train-frac), val/test intacts. Si le carb baisse fortement de 25%→50%→100% du
# train, plus de données aideraient ; s'il plafonne, la limite est informationnelle.
#
# NB : frac=1.0 = le modèle de référence complet, DÉJÀ mesuré par visual_seeds.sh
#      (carb 20.1±0.3 %). Par défaut on ne relance donc QUE 0.25 et 0.5 (économie GPU) ;
#      utilise ce point 1.0 existant comme 3e point de la courbe.
#
#   bash scripts/datacurve.sh                          # FRACS="0.25 0.5" × SEEDS="0 1 2"
#   FRACS="0.25 0.5 0.75" SEEDS="0 1 2 3 4" bash scripts/datacurve.sh
#
# ⚠️ Garde EPOCHS/BATCH_SIZE/LR IDENTIQUES à visual_seeds.sh. Le carb de test est
#    imprimé par train_fusion.py (test_carb_PMAE=...) — pas besoin d'eval.py.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"   # .../CNN
DATA_ROOT="${FUSION_DATA_ROOT:-/mnt/beegfs/projects/gluci/Fusion_Midas_Flava/data/nutrition5k}"
FRACS="${FRACS:-0.25 0.5}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-150}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-5e-5}"
LOG_DIR="$PROJECT_DIR/logs_datacurve"
mkdir -p "$LOG_DIR" "$PROJECT_DIR/checkpoints"

echo "=== Courbe d'apprentissage | fracs : $FRACS | graines : $SEEDS  (epochs=$EPOCHS batch=$BATCH_SIZE lr=$LR) ==="
for f in $FRACS; do
    ftag="$(echo "$f" | sed 's/0*$//; s/\.$//; s/\./p/')"   # 0.25 -> 0p25
    for s in $SEEDS; do
        CKPT="checkpoints/fusion_gt_all_best_f${ftag}_s${s}.pt"
        JOB_ID=$(sbatch --parsable \
            --job-name="dc_f${ftag}_s${s}" \
            --output="${LOG_DIR}/dc_f${ftag}_s${s}_%j.out" \
            --error="${LOG_DIR}/dc_f${ftag}_s${s}_%j.err" \
            --time=24:00:00 --gres=gpu:1 --cpus-per-task=8 --mem=48G \
            --wrap="
export FUSION_DATA_ROOT='${DATA_ROOT}'
source '${PROJECT_DIR}/scripts/_init.sh'
echo '--- TRAIN gt / no-skip / train-frac=${f} / seed=${s} ---'
\"\$PYTHON\" -u train_fusion.py --depth-source gt --no-skip --train-frac ${f} --seed ${s} \
  --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --lr ${LR} --num-workers 4
")
        printf "  job %-9s  frac=%s seed=%s  ->  %s\n" "$JOB_ID" "$f" "$s" "$CKPT"
    done
done

echo ""
echo "Quand tout est fini, récupérer le carb par (fraction, graine) :"
echo "  grep -H 'test_carb_PMAE' ${LOG_DIR}/dc_*_*.out"
echo "3e point de la courbe (frac=1.0) = visual_seeds : carb 20.1±0.3 %."
