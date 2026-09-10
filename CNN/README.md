# Fusion_Midas_Flava — CNN (ResNet-101)

Fusion RGB-D. Profondeur GT ou MiDaS gelé (pré-calculé). Features image : ResNet-101 ImageNet.
Données partagées : `../data/nutrition5k` (au niveau de `Fusion_Midas_Flava/`).

> Variante CNN du projet `Fusion_Midas_Flava`. La variante `../Transformers` (extraction par
> transformers) viendra ensuite et réutilisera la même `data/`.

## Setup (une fois — env partagé CNN + Transformers)

Le venv `env_Fusion_Midas_Flava/` et le dossier `env/` vivent à la **racine**
`Fusion_Midas_Flava/` (partagés par les deux variantes). Depuis la racine :

```bash
bash env/setup_env.sh                        # crée le venv partagé env_Fusion_Midas_Flava/
cp env/cluster.env.example env/cluster.env   # override FUSION_DATA_ROOT seulement si besoin
```

Poids MiDaS (variante `midas` uniquement) embarqués dans `checkpoints/pretrained/`
(`dpt_hybrid_midas.pt` + cache `torch_hub/`). Pour les (re)télécharger — poids OFFICIELS
intel-isl/MiDaS, projet autonome, aucune dépendance externe :

```bash
bash scripts/fetch_midas.sh   # depuis une machine avec internet (Mac, ou frontale via proxy)
```

## Sync Mac → cluster

Le script vit au niveau du projet (`Fusion_Midas_Flava/to_cluster.sh`) et envoie tout
(CNN + data + poids MiDaS) en miroir :

```bash
cd ..                              # Fusion_Midas_Flava/
bash to_cluster.sh                 # code + data + poids (rsync incrémental)
SKIP_DATA=1 bash to_cluster.sh     # sans le dataset (itérations code rapides)
```

## Jobs (depuis Fusion_Midas_Flava/CNN sur le cluster)

```bash
sbatch scripts/test_gt.sh
sbatch scripts/generate_depth.sh
sbatch scripts/train_fusion_gt.sh              # batch 16, nœuds n1-n4 (A6000 48G)
FLAVA=1 sbatch scripts/train_fusion_gt.sh      # + contrastive FLAVA
sbatch scripts/train_fusion_midas.sh
FLAVA=1 sbatch scripts/train_fusion_midas.sh
sbatch scripts/eval_gt.sh                      # évalue fusion_gt_best.pt
FLAVA=1 sbatch scripts/eval_gt.sh              # évalue fusion_gt_flava_best.pt
sbatch scripts/eval_midas.sh
FLAVA=1 sbatch scripts/eval_midas.sh
```

Chaque variante a son propre checkpoint, rien ne s'écrase :
`fusion_{gt,midas}_best.pt` (base) et `fusion_{gt,midas}_flava_best.pt` (FLAVA).
Les logs train et éval affichent `depth=… flava=…` ; le `FLAVA=1` choisit
automatiquement le bon checkpoint à l'éval.
