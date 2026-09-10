#!/bin/bash
# Télécharge les poids OFFICIELS de MiDaS (DPT_Hybrid) dans CNN/checkpoints/pretrained/.
# Source : intel-isl/MiDaS (torch.hub) — exactement les poids de base, aucun fine-tuning.
# À lancer UNE fois depuis une machine avec internet (frontale du cluster via proxy, ou Mac).
# Récupère aussi le cache torch_hub (code du modèle) → MiDaS marche ensuite hors-ligne.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # = CNN/
cd "$ROOT"
# env/ et venv partagés, un cran au-dessus (racine Fusion_Midas_Flava/).
[[ -f ../env/proxy.sh ]] && source ../env/proxy.sh

PY="$ROOT/../env_Fusion_Midas_Flava/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3)"

"$PY" - <<'PYEOF'
import torch
from config import TORCH_HUB_DIR, MIDAS_WEIGHTS, MIDAS_MODEL_TYPE

TORCH_HUB_DIR.mkdir(parents=True, exist_ok=True)
MIDAS_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
torch.hub.set_dir(str(TORCH_HUB_DIR))
print(f"Téléchargement MiDaS {MIDAS_MODEL_TYPE} (poids officiels) → {MIDAS_WEIGHTS}")
m = torch.hub.load("intel-isl/MiDaS", MIDAS_MODEL_TYPE, pretrained=True, skip_validation=True)
m.eval()
torch.save(m.state_dict(), MIDAS_WEIGHTS)
print(f"OK ({MIDAS_WEIGHTS.stat().st_size/1e6:.0f} Mo)")
PYEOF
