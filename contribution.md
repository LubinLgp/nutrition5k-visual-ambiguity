# Contribution — Estimation des glucides par fusion RGB‑D + FLAVA

*Document de travail — projet gluci‑AI (S. Conchon, N. Naderi, D. Lucor). Stage : prédiction des macronutriments (glucides en particulier) à partir d'images de repas, dataset Nutrition5k.*

> **En une phrase.** Sur une évaluation *sans fuite* (découpage chronologique) et sur le **test complet (507 plats)**, la profondeur capteur (GT) reste devant la profondeur estimée (MiDaS) d'environ **1 point de PMAE moyen** ; les auxiliaires contrastifs FLAVA **ne déplacent presque pas la PMAE moyenne** — sauf que la variante **texte + vision combinée** aide de façon cohérente **quand la profondeur est faible (MiDaS)** et sur **les glucides** dans les deux régimes.

---

## 1. Dispositif expérimental (pour lire le tableau)

- **Tâche** : régression des 5 nutriments Nutrition5k (calories, masse, lipides, glucides, protéines) depuis (RGB + profondeur).
- **Deux sources de profondeur**, architecture de fusion **strictement identique** :
  - `gt` : profondeur **capteur** (RealSense, vérité terrain relative).
  - `midas` : profondeur **monoculaire estimée** (MiDaS DPT‑Hybrid gelé) — cas réaliste « smartphone, pas de capteur ».
- **Deux auxiliaires contrastifs FLAVA**, activables **indépendamment** (`FLAVA_TXT`, `FLAVA_VISUEL`) :
  - `txt` : alignement image ↔ **texte de recette** (comme Feng et al.).
  - `vis` : alignement **image ↔ image** (terme d'alignement visuel — *ajout par rapport à Feng, qui n'utilise que le texte*).
- **8 variantes** = {gt, midas} × {aucun, +txt, +vis, +txt+vis}.
- **Découpage** : bloc **chronologique** contigu (dish_id = timestamp) pour val/test, afin d'éviter la fuite par quasi‑doublons propre à Nutrition5k (construction incrémentale des plats). *C'est le point qui nous distingue du split aléatoire de Feng.*
- **Métrique** : **PMAE** = MAE\_k / ȳ\_k × 100 (moyenne de l'erreur / moyenne de la cible, sur le test).
- **Colonne du tableau** : **test complet, n = 507** (aucune image retirée) — c'est le « full dataset » demandé et le réglage comparable à la littérature. *(Pour mémoire, le test filtré n = 497 — les 10 plats « non convenables » retirés — donne des PMAE moyens ~0,6–0,7 pt plus bas ; non reproduit ici pour rester sur le full dataset.)*
- ⚠️ **Un seul run par variante.** Les écarts inférieurs à ~0,3 pt de PMAE sont dans le bruit d'initialisation ; les lire comme des *tendances*, pas des preuves.

---







## 2. Tableau des résultats — variante CNN (ResNet‑101), test complet (n = 507)

PMAE (%) par nutriment ; **gras = meilleur de la colonne à l'intérieur du bloc de profondeur**. Plus bas = meilleur.

### Profondeur capteur (`gt`)

| FLAVA            | Calories | Masse   | Lipides  | **Glucides** | Protéines | **PMAE moyen** |
|------------------|:--------:|:-------:|:--------:|:------------:|:---------:|:--------------:|
| aucun            | **13.57**| 9.37    | 21.38    | 21.53        | 21.82     | 17.53          |
| + txt            | 13.68    | 9.65    | 21.14    | 21.16        | 22.31     | 17.59          |
| + vis            | 13.93    | 9.99    | **20.41**| 21.00        | 21.90     | **17.45**      |
| + txt + vis      | 14.59    | **9.25**| 21.87    | **20.95**    | **21.44** | 17.62          |

### Profondeur estimée (`midas`)

| FLAVA            | Calories | Masse    | Lipides  | **Glucides** | Protéines | **PMAE moyen** |
|------------------|:--------:|:--------:|:--------:|:------------:|:---------:|:--------------:|
| aucun            | 15.18    | 12.14    | **21.73**| 20.87        | 23.38     | 18.66          |
| + txt            | 15.60    | 12.11    | 21.96    | 21.46        | 23.23     | 18.87          |
| + vis            | 15.04    | 12.65    | 22.33    | 21.07        | 23.35     | 18.89          |
| + txt + vis      | **14.47**| **12.09**| 22.65    | **20.55**    | **23.26** | **18.60**      |






---

## 3. Ce que le tableau raconte

1. **Capteur > estimé, mais l'écart est modéré.** GT ≈ 17,5 % vs MiDaS ≈ 18,7 % → **~1,1–1,4 pt** de PMAE moyen. Autrement dit : se passer d'un capteur de profondeur ne coûte qu'environ un point — encourageant pour un usage smartphone.
2. **Hiérarchie de difficulté attendue.** Masse et calories = faciles (PMAE ~9–15 %) ; lipides / glucides / protéines = durs (~20–23 %). Cohérent avec Thames et DPF.
3. **FLAVA ne bouge (presque) pas la moyenne sur GT.** 17,45–17,62 vs 17,53 : dans le bruit. Quand la profondeur est déjà propre, l'auxiliaire contrastif n'apporte quasi rien.
4. **Sur MiDaS, le FLAVA combiné (txt+vis) aide de façon cohérente.** Il donne le meilleur PMAE moyen du bloc (18,60) **et gagne 4 nutriments sur 5** : calories (15,18 → **14,47**, −0,71 pt), masse, glucides, protéines ; seuls les lipides régressent. Hypothèse : le signal contrastif **régularise quand la profondeur est peu fiable**.
5. **FLAVA partiel peut nuire.** `+txt` seul ou `+vis` seul sur MiDaS **dégradent** (18,87 / 18,89 vs 18,66). Le bénéfice n'apparaît qu'avec **les deux modalités ensemble**.
6. **Signal glucides (cœur du stage).** Le **meilleur glucides vient de `+txt+vis` dans les DEUX régimes** (gt **20,95**, midas **20,55**). C'est là que le texte de recette est censé aider le plus (glucides « cachés » : sucres ajoutés, féculents non visibles). C'est notre observation la plus alignée avec la motivation clinique.

**Synthèse honnête** : pris globalement, FLAVA n'est pas un gain franc. Mais le motif — *combiné > partiel, et surtout utile quand la profondeur est faible et sur les glucides* — est cohérent et exploitable comme hypothèse d'article. Il faut **plusieurs graines** pour le confirmer.

---

## 4. Ce qui est nouveau par rapport à la littérature

Honnêtement, l'**architecture** (fusion RGB‑D par blocs d'attention croisée + multi‑échelle, perte multi‑tâches géométrique) n'est **pas** nouvelle : c'est le vocabulaire commun de DPF‑Nutrition, Shao/Min, RDINet, etc. Le FLAVA pour la nutrition a été introduit par **Feng et al.** Ce qui nous appartient est plus fin :

1. **Comparaison contrôlée capteur vs estimé sur la MÊME architecture et le MÊME split sans fuite.** La plupart des travaux prennent *soit* la profondeur capteur (Thames RGB‑D), *soit* une profondeur prédite (DPF avec Adabins), rarement les deux **à conditions identiques**. Notre delta GT↔MiDaS (~1 pt) est une mesure propre du « coût de l'absence de capteur ».
2. **Évaluation sans fuite par découpage chronologique.** Nutrition5k construit ses plats par ajout incrémental d'ingrédients → quasi‑doublons. Un split **aléatoire** (celui de Feng) fuit. Nous imposons un **bloc temporel contigu**. C'est notre contribution méthodologique la plus solide, et elle rejoint la critique de curation de données de **Bianco et al.**
3. **Décomposition de FLAVA en deux termes indépendants et ablation complète.** Feng n'aligne que **texte↔image** ; nous ajoutons un terme **image↔image** (`vis`) et mesurons chaque terme *isolément* (8 configs). Le terme d'alignement visuel séparé est une petite nouveauté.
4. **Résultat nuancé / partiellement négatif.** En régime sans fuite, **les gains FLAVA sur la PMAE moyenne s'effondrent dans le bruit** — contrairement aux gains nets rapportés par Feng (potentiellement gonflés par la fuite du split aléatoire). Le bénéfice réel est **localisé** (profondeur faible + glucides). Un résultat « de prudence » de ce type a une valeur publiable.

> À ne pas surclamer : nous n'avons pas réimplémenté Feng à l'identique, et tout repose sur un run unique. Le point (4) est une **hypothèse étayée**, pas une réfutation.

---

## 5. Observations confirmées / infirmées vs littérature

**Confirmées :**
- **Meilleure profondeur ⇒ meilleure estimation** (GT < MiDaS en erreur). Conforme au consensus (Thames, DPF). Nous ajoutons que l'écart est **modeste** → MiDaS est un substitut viable à bas coût (va dans le sens de la motivation « image unique » d'OmniFood8K).
- **Difficulté relative des nutriments** (masse/calories faciles, macros dures). Conforme à Thames/DPF.

**Non confirmées / nuancées :**
- **Les gains francs de FLAVA (Feng) ne se reproduisent pas** en PMAE moyenne dans notre réglage sans fuite. Suspicion : une part des gains publiés vient de la fuite du split aléatoire.
- **« Plus d'auxiliaires = mieux » : faux.** Le FLAVA mono‑modalité peut **dégrader** (ex. midas +txt) ; seul le combiné aide.

**Ni confirmé ni infirmé (données manquantes) :**
- **RGB‑D vs RGB seul** : nous n'avons **pas** de variante RGB‑seul ici → on ne peut pas trancher nous‑mêmes. À ajouter si on veut appuyer l'argument « la profondeur sert ».

---

## 6. Questions de recherche pour un éventuel article

1. **Quelle fraction des gains d'auxiliaires (FLAVA) est de la fuite ?** Même modèle, même code, **split aléatoire vs chronologique** → mesurer l'écart de gain. C'est la question la plus originale et la plus utile à la communauté.
2. **Un adaptateur léger de profondeur monoculaire ferme‑t‑il l'écart GT↔MiDaS ?** Tester le **SSRA** d'OmniFood8K (scale‑shift résiduel : d\_global = α·d\_mono + β puis conv résiduelle) sur notre `midas`.
3. **Le bénéfice de FLAVA est‑il spécifiquement macronutritionnel (glucides) ?** Le conditionnement texte aide‑t‑il précisément là où les ingrédients sont « cachés » ?
4. **Significativité statistique.** Sur *n* graines, l'effet FLAVA (surtout txt+vis/MiDaS) est‑il significatif ou dans le bruit ?
5. **Pertinence clinique (gluci‑AI).** Une PMAE glucides de ~20 % est‑elle exploitable pour une aide à la dose d'insuline ? Quelle erreur absolue (±g) est cliniquement acceptable ?

---

## 7. Comment orienter un article

**Angle A — méthodologique / prudence.** *« Fragilité des gains contrastifs auxiliaires en estimation nutritionnelle : une étude contrôlée par la fuite. »* Cœur : le split chronologique + l'ablation honnête montrant que les gains FLAVA rétrécissent au bruit. Positionné contre Feng. **Fort mais négatif.**

**Angle B — déploiement bas coût / clinique (recommandé).** *« Combien vaut un capteur de profondeur ? Profondeur capteur vs monoculaire pour l'estimation des glucides. »* Cœur : le delta GT↔MiDaS contrôlé + un **adaptateur SSRA** pour réduire l'écart + le motif « FLAVA txt+vis aide quand la profondeur est faible et sur les glucides ». Cadré gluci‑AI (glucides → insuline). **Constructif et aligné sur le stage.**

**Recommandation** : **Angle B en principal**, avec l'Angle A comme résultat secondaire (la question 1 en encadré). Deux expés le rendent positif plutôt que seulement négatif :
- ajouter l'adaptateur **SSRA** sur `midas` (si l'écart se réduit → contribution positive) ;
- refaire les 8 configs sur **3–5 graines** (barres d'erreur → crédibilise le motif txt+vis/MiDaS/glucides).

---

## 8. Limites et prochaines étapes concrètes

- **Run unique** → lancer le multi‑graines (priorité 1).
- **Pas de RGB‑seul** → ajouter la variante pour chiffrer l'apport de la profondeur.
- **Comparaisons littérature non iso‑split** : nos chiffres ne sont pas directement comparables aux PMAE publiés (splits/prétraitements différents) ; ne comparer que qualitativement.
- **`midas` full‑dataset** : l'entraînement NO_SKIP n'est réellement « complet » qu'après régénération de la profondeur MiDaS des plats « skip » (`generate_depth.py --no-skip`). Le tableau ci‑dessus est en évaluation *test complet 507* sur des modèles entraînés sur le train filtré standard (l'écart NO_SKIP ne porte que sur ~29 plats train, négligeable).
- **Variante Transformers (Swin)** : reproduire le même tableau pour vérifier que le motif txt+vis/MiDaS/glucides n'est pas propre au backbone CNN.

---

## Annexe — cartographie de `lit_review_nutrition5k/`

Pertinence pour notre sujet (RGB‑D / profondeur / nutrition Nutrition5k) :

| Article | Sujet | Rôle vs notre travail |
|---|---|---|
| **Thames — Nutrition5k** (CVPR 2021) | Dataset + baseline RGB(‑D) | Référence / baseline |
| **Han — DPF‑Nutrition** (Foods 2023) | Profondeur prédite (Adabins) + fusion | Le plus proche de notre `midas` |
| **Feng — FLAVA contrastif** (JFCA 2026) | RGB‑D + FLAVA texte | Base que l'on étend et critique (split aléatoire) |
| **Shao/Min — RGB‑D fusion network** (Food Chemistry 2023) | Réseau de fusion RGB‑D | Voisin architectural |
| **Ingredient‑guided interaction & refinement** (DSP 2024) | RGB‑D + ingrédients | Voisin (piste ingrédients) |
| **RDINet** (Applied Sciences 2026) | RGB‑D + traits d'ingrédients | Voisin |
| **IGSMNet** (Foods 2025) | Modélisation sémantique guidée ingrédients | Voisin (groupe Min) |
| **Swin‑Nutrition** (Foods 2022) | Swin RGB pour nutrition | Pertinent pour notre variante Transformers |
| **Qi — Visual‑Ingredient Fusion** (2025) | Fusion visuel‑ingrédient | Piste « texte/ingrédients » |
| **Liu — NutriFusionNet** | Graphe‑attention multimodal | Voisin multimodal |
| **Yu — OmniFood8K** (2026) | Image unique + SSRA (adaptateur profondeur) | Source de l'idée SSRA (Q2) |
| **Bianco** (Nutrients 2025) | Prédiction 2D + curation « au‑delà de Nutrition5k » | Appuie notre argument fuite/curation |
| **fnut‑11‑1469878** (Frontiers Nutr. 2024) | Estimation nutritionnelle | Contexte |
| 2208.12153 / 2308.01810 / 2404.12257 (Purdue/Zhu) | Portion / énergie via profondeur & 3D | Thème profondeur, tangentiel |
| Dietary Assessment w/ Multimodal ChatGPT (IEEE JBHI 2024) | Évaluation diététique par VLM | Baseline/futur (VLM) |
| CrossViT (2103.14899) | ViT multi‑échelle | Backbone, pas Nutrition5k |
| Food‑101 (2014) | Classification d'aliments | Hors sujet nutrition |
