# Fusion_Midas_Flava — Estimation nutritionnelle RGB-D sur Nutrition5k

Estimation de la composition nutritionnelle d'un plat (calories, masse, lipides,
**glucides**, protéines) à partir d'une photo, sur **Nutrition5k**. La cible prioritaire
est le **glucide** (projet gluci-AI, suivi du diabète).

Ce dépôt contient deux variantes **parallèles** du même modèle (seul le backbone change) —
`CNN/` (ResNet-101) et `Transformers/` (Swin) — plus les **scripts d'analyse** qui
constituent la contribution scientifique du papier (échelle d'information, décomposition
composition/portion).

> **Le papier ne propose pas une nouvelle architecture.** Le modèle est une *sonde* qui
> combine des briques existantes (fusion RGB-D par attention CBAM, mixage hiérarchique,
> alignement FLAVA, tête densité de Thames, profondeur MiDaS). La contribution est
> l'**analyse** : *que mesure réellement le modèle, et d'où vient le plafond de ~20 % sur
> les glucides ?* Voir `contribution.md` et les docs de recherche (§ Contexte plus bas).

---

## 1. Structure du dépôt

```
Fusion_Midas_Flava/
├── README.md               ← ce fichier (vue d'ensemble)
├── contribution.md         ← note de résultats (français)
├── to_cluster.sh           ← sync Mac → cluster (rsync)
├── from_cluster.sh         ← sync cluster → Mac
├── env/                    ← mise en place de l'environnement (partagé CNN+Transformers)
│   ├── setup_env.sh
│   └── ...
├── CNN/                    ← variante ResNet-101 (voir CNN/README.md pour le cluster)
│   ├── config.py           ← chemins, hyperparamètres, noms de checkpoints
│   ├── train_fusion.py     ← entraînement (flags: --flava-txt/-vis, --density, --no-skip, --seed)
│   ├── eval.py             ← évaluation + export CSV par plat (resultats/)
│   ├── baseline_prior.py   ← baselines SANS image P0/P1/P1b/P2 (échelle d'information)
│   ├── models/             ← fusion.py, backbones.py, cab.py (CBAM), flava_text.py, midas.py
│   ├── losses/             ← geometric.py (GLS), density.py, align.py (InfoNCE)
│   ├── data/               ← nutrition5k.py (dataset, split), recipe_text.py, transforms.py
│   ├── utils/              ← metrics.py (PMAE), checkpoint.py, depth_io.py
│   └── scripts/            ← lanceurs SLURM (voir §4)
├── Transformers/           ← variante Swin (même code, backbone Swin)
└── (ignorés par git : env_Fusion_Midas_Flava/, data/, */depth/, */checkpoints/, logs_*/)
```

Le venv (`env_Fusion_Midas_Flava/`), le dataset (`data/`), les caches de profondeur
(`*/depth/`), les poids (`*/checkpoints/`) et les logs **ne sont pas versionnés** (voir
`.gitignore`). On les régénère (§2).

---

## 2. Mise en place

Depuis la racine `Fusion_Midas_Flava/` :

```bash
# 1. environnement (venv partagé CNN + Transformers)
bash env/setup_env.sh

# 2. données Nutrition5k attendues dans data/nutrition5k/ :
#    realsense_overhead/<dish>/rgb.png + depth_raw.png
#    metadata/dish_metadata_cafe1.csv
#    dish_ids/splits/depth_{train,test}_ids.txt   (split OFFICIEL scan-disjoint)
#    (override possible via la variable FUSION_DATA_ROOT)

# 3. poids MiDaS (variante profondeur estimée uniquement) :
bash CNN/scripts/fetch_midas.sh        # depuis une machine avec internet

# 4. profondeur MiDaS pré-calculée (une fois) :
sbatch CNN/scripts/generate_depth.sh   # remplit CNN/depth/midas/<dish>.npy
```

Détails cluster (nœuds, chemins beegfs) : voir `CNN/README.md` et `Transformers/README.md`.

---

## 3. Notions clés (à connaître avant de lancer)

- **Métrique PMAE** = MAE / moyenne(vérité sur le test) × 100 (`utils/metrics.py`). Plus bas =
  mieux. On rapporte surtout le **glucide** (~20 %).
- **Split OFFICIEL** (scan-disjoint) obligatoire : un split aléatoire *fuit* (quasi-doublons
  incrémentaux) et gonfle les scores. Le test = `depth_test_ids.txt` ; la validation = **10 %
  chronologiques de fin du train** (`data/nutrition5k.py::split_train_val`).
- **Protocole strict** : sélection de l'époque sur la **validation**, test évalué **une seule
  fois** (déjà implémenté dans `train_fusion.py`).
- **Deux têtes** : `directe` (5 totaux) ou `densité` (masse + densités/100 g, reconstruction =
  densité × masse). La tête densité permet la décomposition composition/portion.
- **Profondeur MiDaS** : relative (normalisée par image), sans échelle absolue.

---

## 4. Entraînement & évaluation (cluster, SLURM)

### Plan factoriel — 32 variantes
`2 backbones × 2 profondeurs × 4 FLAVA × 2 têtes`. Les flags se passent en **variables
d'environnement** aux scripts SLURM :

| Variable | Effet |
|---|---|
| `FLAVA_TXT=1` | alignement image↔texte (recette) |
| `FLAVA_VISUEL=1` | alignement image↔image FLAVA |
| `NO_SKIP=1` | entraîne sur TOUT (checkpoints suffixés `_all`) |
| `DENSITY=1` | tête densité (checkpoints suffixés `_d`) |
| `EPOCHS`,`BATCH_SIZE`,`LR` | hyperparamètres |

```bash
cd CNN     # (ou Transformers)
sbatch scripts/train_fusion_gt.sh                          # gt / directe / sans FLAVA
FLAVA_TXT=1 FLAVA_VISUEL=1 sbatch scripts/train_fusion_gt.sh
DENSITY=1 sbatch scripts/train_fusion_gt.sh                # tête densité
sbatch scripts/train_fusion_midas.sh                       # profondeur estimée MiDaS

sbatch scripts/eval_gt.sh          # évalue le checkpoint correspondant + écrit resultats/*.csv
DENSITY=1 sbatch scripts/eval_gt.sh
```

`eval.py` écrit les **prédictions par plat** dans `resultats/<stem_checkpoint>.csv`
(+ `_filtered.csv`). En mode densité, colonnes `preddens_*` / `truedens_*` en plus (pour la
décomposition). `--no-csv` pour désactiver.

---

## 5. Scripts d'analyse (contribution du papier)

Tous dans `CNN/scripts/` ; ils écrivent des CSV par plat réutilisables.

| Script | Ce qu'il fait | Sortie |
|---|---|---|
| `baseline_prior.py` | baselines **sans image** (P0 moyenne, P1 identité, P1b identité+masse, P2 masses/ingrédient) — l'**échelle d'information** | `baseline_results/prior_*.csv` |
| `scripts/baseline_prior.sh` | lance P0/P1/P1b/P2 d'un coup (CPU) | idem |
| `scripts/baseline_seeds.sh` | P1 et P1b sur **5 seeds** (barres d'erreur) | `prior_*_s{0..4}.csv` |
| `scripts/visual_seeds.sh` | le **modèle visuel** gt/none/direct sur 5 seeds | `resultats/fusion_gt_all_best_s{0..4}.csv` |
| `scripts/density_seeds.sh` | le **modèle densité** gt/none sur 5 seeds (décomposition compo/portion) | `resultats/fusion_gt_all_d_best_s{0..4}.csv` |

Exemples :
```bash
cd CNN
bash scripts/baseline_prior.sh                 # échelle d'information (P0→P2)
bash scripts/baseline_seeds.sh                 # P1/P1b × 5 seeds
SEEDS="0 1 2" bash scripts/visual_seeds.sh     # modèle visuel × 3 seeds
bash scripts/density_seeds.sh                  # densité × 5 seeds
```

⚠️ Pour que les seeds soient comparables à la campagne d'origine, garder les **mêmes**
`EPOCHS/BATCH_SIZE/LR` (voir en-tête de chaque script).

---

## 6. Reproduire les résultats clés du papier

1. **Plafond factoriel** : lancer les 8 variantes directes GT/MiDaS × 4 FLAVA (train + eval),
   idem côté `Transformers/`. → carb reste ~19–21 %.
2. **Échelle d'information** : `bash CNN/scripts/baseline_prior.sh` (+ `baseline_seeds.sh`) et
   `bash CNN/scripts/visual_seeds.sh`. → visuel ≈ P1b.
3. **Décomposition composition/portion** : `bash CNN/scripts/density_seeds.sh`, puis analyser
   les colonnes densité des CSV `_d` (composition = densité×masse-vraie ; portion = reste).
4. **Régression vers la moyenne** : régresser `pred = a·true + b` sur les CSV visuels
   (pente carb ≈ 0,78 vs masse ≈ 0,93).

*(Les scripts de post-traitement/figures vivent avec le brouillon du papier — voir § Contexte.)*

---

## 7. Contexte / papier (docs de recherche)

- `contribution.md` — note de résultats du collègue (dans ce dépôt).
- Docs de recherche **hors dépôt** (dossier parent) : `CONTEXT_HANDOFF.md` (contexte complet),
  `PAPER_PLAN.md` (plan/axe ICLR), `NOVELTY_CHECK.md` (état de l'art), `brouillon_papier.tex`
  (brouillon du papier avec toutes les figures). Les demander à Lubin, ou les ajouter au dépôt
  dans un dossier `docs/` si l'équipe le souhaite.

---

## 8. Pièges connus

- Ne **jamais** utiliser un split aléatoire (fuite → scores gonflés). Toujours le split officiel.
- `eval.py` est **déterministe** : pour des barres d'erreur, il faut **ré-entraîner** avec
  `--seed` (pas ré-évaluer). D'où `*_seeds.sh`.
- `data/` et `env_Fusion_Midas_Flava/` sont volumineux et **non versionnés** : les régénérer (§2).
- Le fichier `eval copie.py` à la racine est un doublon obsolète d'`eval.py` (à supprimer).
