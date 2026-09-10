#!/bin/bash
set -euo pipefail

# Sync complet du projet Fusion_Midas_Flava (CNN + Transformers + data + poids MiDaS)
# → cluster, en miroir. La structure est préservée à l'identique pour que CNN/config.py
# (ROOT.parent/data) résolve la data sans override, exactement comme en local.
# Nutrition5k reste 100% autonome (dans le projet). Seul Recipe1M (pré-entraînement,
# gros dataset externe) est envoyé à part dans .../gluci/data/ (voir bloc plus bas).

REMOTE="${REMOTE:-labia:/mnt/beegfs/projects/gluci/Fusion_Midas_Flava}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
REMOTE_HOST="${REMOTE%%:*}"
REMOTE_PATH="${REMOTE#*:}"

echo "=== Fusion_Midas_Flava → $REMOTE ==="

# Data incluse par défaut (rsync incrémental). SKIP_DATA=1 pour itérer sur le code seul.
# Exclusion ANCRÉE sur /data/ (racine) pour ne pas toucher CNN/data/ (package Python).
DATA_EXCLUDE=()
if [[ "${SKIP_DATA:-0}" == "1" ]]; then
  echo "(SKIP_DATA=1 — dataset data/ exclu)"
  DATA_EXCLUDE=(--exclude '/data/')
fi

# checkpoints/pretrained/ (poids MiDaS + cache torch_hub) part avec le sync : projet autonome.
# Seuls les checkpoints d'entraînement (produits sur le cluster) sont exclus.
rsync -avz --progress \
  --exclude 'env_Fusion_Midas_Flava' \
  --exclude 'env_stage' \
  --exclude '__pycache__' \
  --exclude '.git' \
  --exclude '.claude' \
  --exclude '.DS_Store' \
  --exclude '*.pdf' \
  --exclude '/Litterature/' \
  --exclude '/Mini_rapport/' \
  --exclude 'depth/midas' \
  --exclude 'checkpoints/fusion_*.pt' \
  --exclude 'checkpoints/pretrain_*.pt' \
  --exclude '*.out' \
  --exclude '*.err' \
  ${DATA_EXCLUDE[@]+"${DATA_EXCLUDE[@]}"} \
  "$ROOT/" "$REMOTE/"

# --- Recipe1M (pré-entraînement densité /100g) ---------------------------------
# Gros dataset externe (~2,9 Go). Envoyé dans le dossier data de gluci, FRÈRE du
# projet (.../gluci/data/), exactement comme en local (Stage_2026/data/) : la
# structure reste identique local/cluster (ROOT.parent.parent/data). Volontairement
# HORS du sync projet ci-dessus pour ne pas re-scanner 20k images à chaque push de
# code. On n'envoie QUE le sous-ensemble utile : images + nutrition (json) +
# jointure (csv). PAS les 1,4 Go de recipe1M_layers/ (layer1/layer2), inutiles ici.
# Désactiver avec SKIP_DATA=1 ou SKIP_RECIPE1M=1.
LOCAL_DATA="$(cd "$ROOT/.." && pwd)/data"
REMOTE_DATA="${REMOTE_DATA:-$REMOTE_HOST:$(dirname "$REMOTE_PATH")/data}"
if [[ "${SKIP_DATA:-0}" == "1" || "${SKIP_RECIPE1M:-0}" == "1" ]]; then
  echo "(Recipe1M non envoyé)"
elif [[ ! -d "$LOCAL_DATA/recipe1M" ]]; then
  echo "(Recipe1M introuvable en local: $LOCAL_DATA/recipe1M — ignoré)"
else
  REMOTE_DATA_PATH="${REMOTE_DATA#*:}"
  echo "=== Recipe1M → $REMOTE_DATA ==="
  ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_DATA_PATH/recipe1M' '$REMOTE_DATA_PATH/Extraction_Recipes1M'"
  # images (jpg déjà compressés → sans -z)
  rsync -av --progress "$LOCAL_DATA/recipe1M/" "$REMOTE_DATA/recipe1M/"
  # métadonnées : nutrition /100g + jointure image<->recette (sans les layers)
  rsync -avz --progress \
    "$LOCAL_DATA/Extraction_Recipes1M/recipes_with_nutritional_info.json" \
    "$LOCAL_DATA/Extraction_Recipes1M/nutritional_images.csv" \
    "$REMOTE_DATA/Extraction_Recipes1M/"
fi

echo "=== OK ==="
echo "Cluster: ssh $REMOTE_HOST && cd $REMOTE_PATH/CNN"
echo "Poids MiDaS absents ?  bash CNN/scripts/fetch_midas.sh  (depuis une machine avec internet)"
