#!/bin/bash
#SBATCH --job-name=baseline_prior
#SBATCH --output=logs_baseline/prior_%j.out
#SBATCH --error=logs_baseline/prior_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# Baselines "prior-only" SANS image (PAPER_PLAN.md §5.1). Petits modèles tabulaires :
# CPU suffit, pas de GPU demandé. Lance les 3 régimes décisifs d'un coup.
#
#   P0  mean      : plancher (moyenne-train)
#   P1  presence  : identité des ingrédients SEULE (pas les quantités) -> TEST DÉCISIF
#   P2  mass      : sac pondéré par les masses vraies -> borne haute "portion connue"
#
# + P1b : identité seule MAIS avec la masse totale vraie fournie (oracle-portion) :
#   isole ce que la portion ajoute au prior d'identité.
#
# Le TEST est aligné sur le modèle visuel via --depth-source gt (mêmes plats filtrés).
# Comparer la PMAE carb de P1 à celle du modèle visuel (~19.6%) : si proche, le visuel
# ne fait guère plus que reconnaître + appliquer le prior.

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$ROOT/scripts/_init.sh"
mkdir -p logs_baseline

run() {  # $1 = étiquette, reste = args de baseline_prior.py
    echo ""; echo "########## $1 ##########"
    "$PYTHON" -u baseline_prior.py "${@:2}"
}

run "P0 mean (plancher)"                 --features mean
run "P1 presence+mlp (IDENTITÉ SEULE)"   --features presence --model mlp
run "P1b presence + true mass (oracle-portion)" --features presence --model mlp --add-true-mass
run "P2 mass+linear (portion en entrée)" --features mass --model linear

echo ""; echo "Comparer la colonne carb à la PMAE carb du modèle visuel (~19.6% gt/none/direct)."
