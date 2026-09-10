#!/bin/bash
set -euo pipefail

# Rapatrie les ARTEFACTS générés sur le cluster (logs, depth, checkpoints) vers le local,
# pour CHAQUE variante (CNN, Transformers, ...). Symétrique de to_cluster.sh.
# NE touche PAS au code source : le dossier local reste la référence.

REMOTE="${REMOTE:-labia:/mnt/beegfs/home/kebdi/gluci/Fusion_Midas_Flava}"
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Variantes à rapatrier (surchargeable : VARIANTS="CNN" bash from_cluster.sh).
read -r -a VARIANTS <<< "${VARIANTS:-CNN Transformers}"

# Dossiers d'artefacts par variante. depth/midas peut peser plusieurs Go (~2900 .npy) :
# SKIP_DEPTH=1 bash from_cluster.sh  pour ne récupérer que logs + checkpoints.
ITEMS=(logs_train_fusion logs_eval logs_generate checkpoints)
if [[ "${SKIP_DEPTH:-0}" == "1" ]]; then
  echo "(SKIP_DEPTH=1 — depth/ ignoré)"
else
  ITEMS+=(depth)
fi

echo "=== $REMOTE  →  $ROOT  (variantes : ${VARIANTS[*]}) ==="

for v in "${VARIANTS[@]}"; do
  echo "######## variante $v ########"
  mkdir -p "$ROOT/$v"
  for it in "${ITEMS[@]}"; do
    echo "--- $v/$it ---"
    # --exclude pretrained : on ne re-télécharge pas les poids (déjà en local, 470 Mo).
    rsync -avz --partial --progress \
      --exclude 'pretrained' \
      --exclude '__pycache__' \
      --exclude '.DS_Store' \
      "$REMOTE/$v/$it" "$ROOT/$v/" \
      || echo "  (ignoré : $v/$it absent sur le cluster ou erreur réseau)"
  done
done

echo "=== OK — artefacts rapatriés pour : ${VARIANTS[*]} ==="
