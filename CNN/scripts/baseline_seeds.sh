#!/bin/bash
#SBATCH --job-name=baseline_seeds
#SBATCH --output=logs_baseline/seeds_%j.out
#SBATCH --error=logs_baseline/seeds_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# Robustesse multi-graines de P1 (identité seule) et P1b (identité + masse totale vraie).
# But : confirmer la coïncidence P1b ≈ modèle visuel (~19.6% carb) sur plusieurs graines,
# et donner des barres d'erreur (moyenne ± écart-type) — cf. PAPER_PLAN §3.0 / §5.6.
# CPU suffit. Un CSV par (baseline, graine) : prior_<...>_s<seed>.csv.
#
#   SEEDS="0 1 2 3 4"  bash/sbatch scripts/baseline_seeds.sh    # pour changer la liste
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Data partagée (lecture seule) : on la fixe AVANT _init.sh pour que son `:-` la garde.
export FUSION_DATA_ROOT="${FUSION_DATA_ROOT:-/mnt/beegfs/projects/gluci/Fusion_Midas_Flava/data/nutrition5k}"
source "$ROOT/scripts/_init.sh"
mkdir -p logs_baseline baseline_results

SEEDS="${SEEDS:-0 1 2 3 4}"
echo "=== Graines : $SEEDS ==="

for s in $SEEDS; do
    echo ""; echo "########## P1 presence+mlp | seed=$s ##########"
    "$PYTHON" -u baseline_prior.py --features presence --model mlp --seed "$s" \
        --csv-out "baseline_results/prior_presence_mlp_s${s}.csv"

    echo ""; echo "########## P1b presence+mlp+true_mass | seed=$s ##########"
    "$PYTHON" -u baseline_prior.py --features presence --model mlp --add-true-mass --seed "$s" \
        --csv-out "baseline_results/prior_presence_mlp_truemass_s${s}.csv"
done

echo ""
echo "=== Terminé. Agréger les carb PMAE par baseline : ==="
echo "  grep -A6 '\\[test\\]' logs_baseline/seeds_<jobid>.out | grep 'carb:'"
