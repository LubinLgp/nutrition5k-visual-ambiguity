#!/bin/bash
# densitycurve.sh — SOUMETTEUR (depuis CNN/, PAS via sbatch lui-même).
# EXPÉRIENCE W5 (reviewer) : courbe d'apprentissage CÔTÉ COMPOSITION. On entraîne la
# tête DENSITÉ (gt / no-skip) sur des fractions du train, et on lit le carb reconstruit
# avec la MASSE VRAIE (composition seule) — c.-à-d. le pendant "17.6" de P1b, mais à
# ¼ et ½ des données. À comparer à la courbe P1b (baseline_prior --passthrough
# --train-frac) : si les deux descendent en parallèle, plus de données améliorent la
# composition des DEUX de la même façon (limite d'info partagée), pas seulement la portion.
#
# Sortie clé imprimée par train_fusion.py :
#   test_carb_comp_PMAE (densité × masse VRAIE)   <-- composition seule
#   test_carb_PMAE       (densité × masse prédite) <-- total, en bonus
#
#   bash scripts/densitycurve.sh                       # FRACS="0.25 0.5" × SEEDS="0 1 2"
#   FRACS="0.25 0.5 0.75" SEEDS="0 1 2 3 4" bash scripts/densitycurve.sh
#
# NB : frac=1.0 = la tête densité complète, DÉJÀ mesurée (papier : composition 17.6±0.3).
#      On ne relance donc que 0.25 et 0.5.
# ⚠️ EPOCHS/BATCH_SIZE/LR IDENTIQUES à density_seeds.sh (comparabilité).
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${FUSION_DATA_ROOT:-/mnt/beegfs/projects/gluci/Fusion_Midas_Flava/data/nutrition5k}"
FRACS="${FRACS:-0.25 0.5}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-150}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-5e-5}"
LOG_DIR="$PROJECT_DIR/logs_densitycurve"
mkdir -p "$LOG_DIR" "$PROJECT_DIR/checkpoints"

echo "=== Courbe densité (W5) | fracs : $FRACS | graines : $SEEDS (epochs=$EPOCHS batch=$BATCH_SIZE lr=$LR) ==="
for f in $FRACS; do
    ftag="$(echo "$f" | sed 's/0*$//; s/\.$//; s/\./p/')"   # 0.25 -> 0p25
    for s in $SEEDS; do
        JOB_ID=$(sbatch --parsable \
            --job-name="dc_d_f${ftag}_s${s}" \
            --output="${LOG_DIR}/dc_d_f${ftag}_s${s}_%j.out" \
            --error="${LOG_DIR}/dc_d_f${ftag}_s${s}_%j.err" \
            --time=24:00:00 --gres=gpu:1 --cpus-per-task=8 --mem=48G \
            --wrap="
export FUSION_DATA_ROOT='${DATA_ROOT}'
source '${PROJECT_DIR}/scripts/_init.sh'
echo '--- TRAIN densité / gt / no-skip / train-frac=${f} / seed=${s} ---'
\"\$PYTHON\" -u train_fusion.py --depth-source gt --no-skip --density --train-frac ${f} --seed ${s} \
  --epochs ${EPOCHS} --batch-size ${BATCH_SIZE} --lr ${LR} --num-workers 4
")
        printf "  job %-9s  frac=%s seed=%s\n" "$JOB_ID" "$f" "$s"
    done
done
echo ""
echo "Fini -> composition par (fraction, graine) :"
echo "  grep -H 'carb_comp_PMAE' ${LOG_DIR}/*.out"
echo "3e point (frac=1.0) = density_seeds : composition 17.6±0.3. Comparer à la courbe P1b."
