# Chrono : durée totale du job, affichée à la sortie (succès comme échec).
# SECONDS est un compteur bash auto-incrémenté ; le trap EXIT le lit en fin de job.
SECONDS=0
echo "=== Démarrage $(date '+%F %T') | job ${SLURM_JOB_ID:-local} ==="
_print_elapsed() {
    local s=$SECONDS
    printf '=== Durée totale : %02dh%02dm%02ds (%ds) | job %s | fin %s ===\n' \
        $((s / 3600)) $(((s % 3600) / 60)) $((s % 60)) "$s" \
        "${SLURM_JOB_ID:-local}" "$(date '+%F %T')"
}
trap _print_elapsed EXIT

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    ROOT="$SLURM_SUBMIT_DIR"
else
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
# env/ et le venv sont partagés, un cran au-dessus (racine Fusion_Midas_Flava/).
[[ -f "$ROOT/../env/cluster.env" ]] && source "$ROOT/../env/cluster.env"
export FUSION_DATA_ROOT="${FUSION_DATA_ROOT:-$ROOT/../data/nutrition5k}"
export FUSION_MIDAS_PRETRAINED="${FUSION_MIDAS_PRETRAINED:-$ROOT/checkpoints/pretrained}"
PYTHON="$ROOT/../env_Fusion_Midas_Flava/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "ERREUR: env_Fusion_Midas_Flava absent — bash ../env/setup_env.sh (depuis Fusion_Midas_Flava/)"
    exit 1
fi
cd "$ROOT"
