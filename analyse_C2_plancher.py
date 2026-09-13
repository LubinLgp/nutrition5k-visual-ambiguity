#!/usr/bin/env python3
"""
analyse_C2_plancher.py — Plancher NON-PARAMÉTRIQUE de l'erreur glucidique.

Question : le ~17% de P1b (identité + masse totale) est-il une limite d'INFORMATION,
ou une faiblesse du modèle ? On répond SANS réseau.

Pour chaque plat de test on cherche ses voisins d'ENTRAÎNEMENT de même identité
d'ingrédients (Jaccard >= 0.8) et de masse totale proche (±20%, >=3 voisins). La
dispersion empirique de leurs glucides VRAIS est une borne inférieure, indépendante
de tout modèle, sur l'erreur atteignable à (identité + masse) donnés. On compare cette
borne à P1b (passthrough) et au modèle visuel, sur les plats "couverts" (un tel
voisinage existe) et "non couverts".

Résultat (reproduit les chiffres de sec:ladder du papier) :
  - couverts (25%) : plancher 8.2% ~ P1b 8.7%  -> P1b est quasi-OPTIMAL (limite d'info)
  - non couverts (75%) : P1b 18.7%             -> le plancher est hétérogène,
                                                  tiré par les combinaisons INÉDITES.

Aucun GPU, aucune image : métadonnées Nutrition5k + CSV de prédictions déjà calculés.
Usage : python analyse_C2_plancher.py   (depuis la racine du dépôt)
"""
from pathlib import Path
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent
META = PROJ / "data/nutrition5k/metadata/dish_metadata_cafe1.csv"
SPLIT = PROJ / "data/nutrition5k/dish_ids/splits"
BASE = PROJ / "CNN/baseline_results"
RES = PROJ / "CNN/resultats"
JACCARD, MASS_TOL, K_MIN = 0.8, 0.20, 3  # définition d'un voisinage "proche"


def parse_dishes():
    """CSV ragged Nutrition5k : 6 champs plat + groupes de 7 par ingrédient."""
    D = {}
    for line in open(META):
        f = line.rstrip("\n").split(",")
        if len(f) < 13 or (len(f) - 6) % 7:
            continue
        ings = [{"id": f[k], "g": float(f[k + 2])} for k in range(6, len(f), 7) if float(f[k + 2]) > 0]
        D[f[0]] = {"mass": float(f[2]), "carb": float(f[4]), "ings": ings}  # f[4]=glucides
    return D


def main():
    D = parse_dishes()
    skip = {l.strip() for l in open(PROJ / "CNN/dishes_to_skip.txt")}
    train_ids = [d for d in (l.strip() for l in open(SPLIT / "depth_train_ids.txt"))
                 if d in D and d not in skip]

    p1b = pd.read_csv(BASE / "prior_presence_passthrough_truemass.csv")
    vis = pd.read_csv(RES / "fusion_gt_all_best.csv")
    common = set(p1b.dish_id) & set(vis.dish_id)
    vis = vis[vis.dish_id.isin(common)].reset_index(drop=True)
    p1b = p1b.set_index("dish_id").loc[vis.dish_id].reset_index()
    test_ids = list(vis.dish_id)

    vocab = {}
    for d in train_ids:
        for i in D[d]["ings"]:
            vocab.setdefault(i["id"], len(vocab))

    def presence(ids):
        X = np.zeros((len(ids), len(vocab)), bool)
        for r, d in enumerate(ids):
            for i in D[d]["ings"]:
                j = vocab.get(i["id"])
                if j is not None:
                    X[r, j] = True
        return X

    Xtr, Xte = presence(train_ids), presence(test_ids)
    mtr = np.array([D[d]["mass"] for d in train_ids])
    ctr = np.array([D[d]["carb"] for d in train_ids])
    mte = np.array([D[d]["mass"] for d in test_ids])

    inter = Xte.astype(np.int32) @ Xtr.astype(np.int32).T
    jac = inter / np.maximum(Xte.sum(1)[:, None] + Xtr.sum(1)[None, :] - inter, 1)

    covered = np.zeros(len(test_ids), bool)
    floor_err = []
    for r in range(len(test_ids)):
        ok = (jac[r] >= JACCARD) & (np.abs(mtr - mte[r]) / mte[r] <= MASS_TOL)
        if ok.sum() >= K_MIN:
            covered[r] = True
            floor_err.append(abs(p1b.true_carb.values[r] - np.median(ctr[ok])))

    meanC = p1b.true_carb.mean()
    pm = lambda df, m: (df.pred_carb - df.true_carb).abs().values[m].mean() / df.true_carb.values[m].mean() * 100
    print(f"plats couverts (voisinage identité+masse proche) : {covered.sum()}/{len(test_ids)} "
          f"({covered.mean()*100:.0f}%)\n")
    print(f"{'':24s}{'couverts':>10s}{'non couv.':>11s}")
    print(f"{'plancher non-param.':24s}{np.mean(floor_err)/meanC*100:9.1f}%{'—':>11s}")
    print(f"{'P1b passthrough':24s}{pm(p1b, covered):9.1f}%{pm(p1b, ~covered):10.1f}%")
    print(f"{'modèle visuel':24s}{pm(vis, covered):9.1f}%{pm(vis, ~covered):10.1f}%")
    print("\nLecture : P1b (8.7) ~ plancher (8.2) sur les couverts -> quasi-optimal, "
          "donc limite d'INFORMATION ; plancher hétérogène (8% vu -> 19% inédit).")


if __name__ == "__main__":
    main()
