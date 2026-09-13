#!/usr/bin/env python3
"""
baseline_prior.py — Baselines « prior-only » SANS image, pour l'axe ICLR
« estimation nutritionnelle sous ambiguïté visuelle » (cf. PAPER_PLAN.md §5.1).

Question : quelle part de la performance vient de l'IMAGE, et quelle part vient de
simples corrélations du jeu d'entraînement (le prior du « type d'aliment typique ») ?

On entraîne des prédicteurs qui n'ont PAS accès à l'image, seulement à des features
dérivées de la liste d'ingrédients (identité et/ou masses), puis on compare leur PMAE
à celle des modèles visuels (fusion RGB-D). Trois régimes clés :

  --features mean       P0  : ignore tout, prédit la moyenne-train de chaque nutriment
                              (plancher absolu).
  --features presence   P1  : sac d'ingrédients BINAIRE (identité, PAS les quantités).
                              => le modèle « connaît l'aliment » mais doit DEVINER la
                              portion à partir du prior. Si P1 atteint ~la PMAE carb du
                              modèle visuel, alors le visuel ne fait guère plus que
                              reconnaître + appliquer le prior. C'EST LE TEST DÉCISIF.
  --features mass       P2  : sac d'ingrédients pondéré par les MASSES vraies (la portion
                              est DANS l'entrée). Borne haute « si on connaissait la
                              portion » — les totaux Nutrition5k sont ~déterministes
                              sachant les masses d'ingrédients.

Protocole IDENTIQUE à train_fusion.py : test = split officiel, val = queue chronologique
du train, sélection du meilleur epoch sur la val, test évalué une seule fois. Métrique
PMAE (utils/metrics). Un CSV par plat (dish_id, pred_*, true_*) est écrit pour brancher
directement sur le pipeline d'analyse d'erreur (même format que eval/resultats/).

Aucune image, aucun GPU nécessaire (petits modèles tabulaires).
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (
    DENSITY_NUTRIENTS,
    DEPTH_TEST_SPLIT,
    DEPTH_TRAIN_SPLIT,
    FUSION_VAL_RATIO,
    METADATA_CAFE1,
    NUTRIENT_NAMES,
    CHECKPOINT_DIR,
)
from losses.density import assemble_totals, totals_to_densities
from data.nutrition5k import (
    filter_valid_dish_ids,
    load_dish_metadata,
    load_split_ids,
    split_train_val,
)
from data.recipe_text import load_dish_ingredients
from losses.geometric import GeometricMultiTaskLoss
from utils.metrics import compute_mae_pmae

_MASS_IDX = NUTRIENT_NAMES.index("mass")


# --------------------------------------------------------------------------- #
# Features ingrédients (aucune image)
# --------------------------------------------------------------------------- #
def build_vocab(train_ids, dish_ingredients):
    """Vocabulaire des IDs d'ingrédients vus dans le TRAIN (déterministe, trié)."""
    vocab = {}
    for d in train_ids:
        for ing in dish_ingredients.get(d, []):
            vocab.setdefault(ing["id"], len(vocab))
    return vocab


def featurize(dish_ids, dish_ingredients, vocab, mode, add_true_mass=False, labels=None):
    """Matrice de features [N, F] selon `mode` :
      - 'presence' : 1 si l'ingrédient est présent (identité seule, pas de quantité) ;
      - 'mass'     : masse (g) de l'ingrédient (la portion est dans l'entrée) ;
      - 'mean'     : features nulles (le modèle ne peut sortir que la moyenne apprise).
    add_true_mass : ajoute la masse totale vraie comme feature (variante oracle-portion).
    """
    n_vocab = len(vocab)
    extra = 1 if add_true_mass else 0
    X = np.zeros((len(dish_ids), n_vocab + extra), dtype=np.float32)
    for r, d in enumerate(dish_ids):
        if mode != "mean":
            for ing in dish_ingredients.get(d, []):
                j = vocab.get(ing["id"])
                if j is None:  # ingrédient absent du train : ignoré (OOV)
                    continue
                X[r, j] = 1.0 if mode == "presence" else ing["mass_g"]
        if add_true_mass:
            X[r, n_vocab] = labels[d][NUTRIENT_NAMES[_MASS_IDX]]
    return X


def make_targets(dish_ids, labels):
    return np.array([[labels[d][n] for n in NUTRIENT_NAMES] for d in dish_ids], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Modèles
# --------------------------------------------------------------------------- #
class LinearHead(nn.Module):
    def __init__(self, in_dim, out_dim=5):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return torch.nn.functional.softplus(self.fc(x))  # nutriments positifs


class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden=256, out_dim=5, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(p),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Dropout(p),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return torch.nn.functional.softplus(self.net(x))


class MeanPredictor(nn.Module):
    """Prédit une constante apprise = moyenne-train de chaque nutriment (P0)."""
    def __init__(self, out_dim=5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        return torch.nn.functional.softplus(self.bias).unsqueeze(0).expand(x.shape[0], -1)


def build_model(name, in_dim):
    if name == "mean":
        return MeanPredictor()
    if name == "linear":
        return LinearHead(in_dim)
    if name == "mlp":
        return MLPHead(in_dim)
    raise ValueError(name)


# --------------------------------------------------------------------------- #
# Entraînement / éval
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, criterion, optimizer, device):
    train = optimizer is not None
    model.train(train)
    total, preds, gts = 0.0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        with torch.set_grad_enabled(train):
            pred = model(xb)
            loss, _ = criterion(pred, yb)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        total += loss.item()
        preds.append(pred.detach().cpu().numpy()); gts.append(yb.cpu().numpy())
    metrics = compute_mae_pmae(np.concatenate(preds), np.concatenate(gts))
    return total / max(1, len(loader)), metrics


def print_block(label, metrics, n):
    print(f"[{label}] n={n} | Mean PMAE {metrics['mean_pmae']:.2f}%")
    for name in NUTRIENT_NAMES:
        print(f"    {name}: MAE={metrics[name]['mae']:.3f} PMAE={metrics[name]['pmae']:.2f}%")


def write_csv(path, dish_ids, preds, gts):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (["dish_id"] + [f"pred_{n}" for n in NUTRIENT_NAMES]
              + [f"true_{n}" for n in NUTRIENT_NAMES])
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for i, d in enumerate(dish_ids):
            w.writerow([d] + [f"{v:.6g}" for v in preds[i]] + [f"{v:.6g}" for v in gts[i]])
    print(f"  CSV par plat -> {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", choices=["mean", "presence", "mass"], required=True,
                   help="mean=P0 plancher | presence=P1 identité seule (test décisif) | mass=P2 portion en entrée")
    p.add_argument("--model", choices=["mean", "linear", "mlp"], default=None,
                   help="défaut: mean->mean, presence->mlp, mass->linear")
    p.add_argument("--add-true-mass", action="store_true",
                   help="ajoute la masse totale vraie en feature (variante oracle-portion)")
    p.add_argument("--passthrough", action="store_true",
                   help="P1b PASSTHROUGH : prédit les DENSITÉS /100g depuis l'identité "
                        "(presence) puis reconstruit le total = densité × masse totale "
                        "VRAIE (masse exacte, pas apprise). C'est le P1b du papier (~17.1). "
                        "Nécessite --features presence, incompatible avec --add-true-mass.")
    p.add_argument("--depth-source", choices=["gt", "midas"], default="gt",
                   help="uniquement pour ALIGNER le set de test sur celui du modèle visuel (filtrage images)")
    p.add_argument("--no-skip", action="store_true", help="test complet (ne pas écarter les 'non convenables')")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-ratio", type=float, default=FUSION_VAL_RATIO)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--csv-out", type=Path, default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model_name = args.model or {"mean": "mean", "presence": "mlp", "mass": "linear"}[args.features]
    device = torch.device(args.device)
    apply_skip = not args.no_skip

    labels = load_dish_metadata(METADATA_CAFE1)
    dish_ingredients = load_dish_ingredients(METADATA_CAFE1)

    # MÊME découpage que la fusion : test officiel, val = queue chronologique du train.
    # On filtre par disponibilité image (depth-source) pour que le TEST soit EXACTEMENT
    # celui du modèle visuel -> PMAE directement comparables.
    all_train = filter_valid_dish_ids(load_split_ids(DEPTH_TRAIN_SPLIT), labels=labels,
                                      depth_source=args.depth_source, apply_skip=apply_skip)
    train_ids, val_ids = split_train_val(all_train, val_ratio=args.val_ratio)
    test_ids = filter_valid_dish_ids(load_split_ids(DEPTH_TEST_SPLIT), labels=labels,
                                     depth_source=args.depth_source, apply_skip=apply_skip)

    vocab = build_vocab(train_ids, dish_ingredients)
    tag = f"features={args.features} model={model_name}" + (" +true_mass" if args.add_true_mass else "")
    print(f"baseline_prior | {tag} | vocab={len(vocab)} | "
          f"train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")

    def loader_for(ids, shuffle):
        X = featurize(ids, dish_ingredients, vocab, args.features, args.add_true_mass, labels)
        Y = make_targets(ids, labels)
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle), ids, X, Y

    train_loader, *_ = loader_for(train_ids, True)
    val_loader, *_ = loader_for(val_ids, False)
    test_loader, test_ids_o, X_test, Y_test = loader_for(test_ids, False)

    if args.passthrough:
        # P1b PASSTHROUGH : MLP identité -> densités /100g (4 nutriments), puis
        # total = densité × masse VRAIE donnée. Miroir sans image de la tête densité.
        if args.features != "presence" or args.add_true_mass:
            raise SystemExit("--passthrough exige --features presence (sans --add-true-mass)")
        Xtr = featurize(train_ids, dish_ingredients, vocab, "presence", False, labels)
        Ytr = make_targets(train_ids, labels)
        Yva = make_targets(val_ids, labels)
        Xva = featurize(val_ids, dish_ingredients, vocab, "presence", False, labels)
        Dtr = totals_to_densities(torch.from_numpy(Ytr))  # [N,4] ordre DENSITY_NUTRIENTS
        tr_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), Dtr),
                               batch_size=args.batch_size, shuffle=True)
        model = MLPHead(len(vocab), out_dim=len(DENSITY_NUTRIENTS)).to(device)
        criterion = GeometricMultiTaskLoss(DENSITY_NUTRIENTS)  # L1 géométrique sur les densités
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        mass_va = torch.from_numpy(Yva[:, _MASS_IDX:_MASS_IDX + 1])
        best_val, best_state, best_epoch = float("inf"), None, 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            for xb, db in tr_loader:
                xb, db = xb.to(device), db.to(device)
                loss, _ = criterion(model(xb), db)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            model.eval()
            with torch.no_grad():  # sélection sur le total reconstruit avec la masse VRAIE de val
                dva = model(torch.from_numpy(Xva).to(device)).cpu()
                val_m = compute_mae_pmae(assemble_totals(dva, mass_va).numpy(), Yva)
            if val_m["mean_pmae"] < best_val:
                best_val, best_epoch = val_m["mean_pmae"], epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if epoch % 20 == 0 or epoch == 1:
                print(f"  epoch {epoch:3d}  val_PMAE={val_m['mean_pmae']:.2f}%  (best {best_val:.2f}@{best_epoch})")
        model.load_state_dict(best_state); model.eval()
        mass_te = torch.from_numpy(Y_test[:, _MASS_IDX:_MASS_IDX + 1])
        with torch.no_grad():
            dte = model(torch.from_numpy(X_test).to(device)).cpu()
            preds = assemble_totals(dte, mass_te).numpy()
        print(f"\n== P1b passthrough | best epoch {best_epoch} | val_PMAE={best_val:.2f}% ==")
    elif model_name == "mean":
        # Plancher ANALYTIQUE : prédire la moyenne-train de chaque nutriment (pas
        # d'apprentissage — le bon "predict-the-mean" est en forme fermée, pas par SGD).
        Y_train = make_targets(train_ids, labels)
        mean_vec = Y_train.mean(axis=0, keepdims=True)
        preds = np.repeat(mean_vec, len(test_ids_o), axis=0).astype(np.float32)
        print("\n== plancher analytique (moyenne-train) ==")
    else:
        in_dim = len(vocab) + (1 if args.add_true_mass else 0)
        model = build_model(model_name, in_dim).to(device)
        criterion = GeometricMultiTaskLoss()  # même loss que la fusion (comparaison propre)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_val, best_state, best_epoch = float("inf"), None, 0
        for epoch in range(1, args.epochs + 1):
            run_epoch(model, train_loader, criterion, optimizer, device)
            _, val_m = run_epoch(model, val_loader, criterion, None, device)
            if val_m["mean_pmae"] < best_val:
                best_val = val_m["mean_pmae"]; best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if epoch % 20 == 0 or epoch == 1:
                print(f"  epoch {epoch:3d}  val_PMAE={val_m['mean_pmae']:.2f}%  (best {best_val:.2f}@{best_epoch})")

        # Test une seule fois, sur le meilleur checkpoint (sélectionné sur la val).
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            preds = model(torch.from_numpy(X_test).to(device)).cpu().numpy()
        print(f"\n== best epoch {best_epoch} | val_PMAE={best_val:.2f}% ==")

    test_m = compute_mae_pmae(preds, Y_test)
    print_block("test", test_m, len(test_ids_o))

    if args.csv_out:
        csv_out = args.csv_out
    elif args.passthrough:
        csv_out = CHECKPOINT_DIR.parent / "baseline_results" / "prior_presence_passthrough_truemass.csv"
    else:
        suffix = "_truemass" if args.add_true_mass else ""
        csv_out = CHECKPOINT_DIR.parent / "baseline_results" / f"prior_{args.features}_{model_name}{suffix}.csv"
    write_csv(Path(csv_out), test_ids_o, preds, Y_test)


if __name__ == "__main__":
    main()
