# CNN/resultats/ — prédictions par plat (variante ResNet-101)

Sorties d'`eval.py` (une ligne par plat du test) et des campagnes multi-graines.
Colonnes : `dish_id, pred_{calories,mass,fat,carb,protein}, true_{...}`.
En mode densité (`_d`), colonnes `preddens_*` / `truedens_*` en plus (décomposition
composition/portion). Les `_filtered` retirent les plats hors-critère (507 → 497).

## Plan factoriel (1 run par variante)
`fusion_{gt,midas}[_{ft,fv,ftv}]_all_best.csv` (tête directe) et
`..._all_d_best.csv` (tête densité) — 2 profondeurs × 4 FLAVA × 2 têtes = **16**.
- `gt` = profondeur capteur, `midas` = profondeur estimée MiDaS
- `ft` = FLAVA texte, `fv` = FLAVA image, `ftv` = les deux, (rien) = sans FLAVA

## Campagnes multi-graines (barres d'erreur)
- **Visuel** : `fusion_gt_all_best_s{0..4}.csv` (+ `_filtered`) — réf gt/directe/no-skip.
- **Densité** : `fusion_gt_all_d_best_s{0..4}_filtered.csv` — décomposition compo/portion.

## Ablations (réponses aux reviewers B14/B15)
`ablations_carb.csv` — récap du **carb PMAE** (une ligne par run) des 3 ablations :
- **rgbonly** (`--no-depth`, profondeur permutée) : contribution de la profondeur (B15).
- **carbonly** (`--carb-only`) : dilution multi-tâche (B14).
- **datacurve** (`--train-frac` 0.25/0.5, +1.0 = seeds visuels) : effet taille données (B14).

## Note de reproductibilité
Le factoriel CNN n'a que les CSV bruts (pas de `_filtered`) — re-filtrer via `eval.py`
si besoin des 497. Les seeds densité n'ont que les `_filtered`. La variante Swin
équivalente est dans `../../Transformers/resultats/` (avec `_filtered`).
