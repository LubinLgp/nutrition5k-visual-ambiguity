#!/usr/bin/env python3
import argparse
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (
    DEPTH_TEST_SPLIT,
    DEPTH_TRAIN_SPLIT,
    FLAVA_ALIGN_LAMBDA,
    FLAVA_ALIGN_NEG,
    FLAVA_ALIGN_TEMPERATURE,
    FLAVA_MODEL,
    FLAVA_PROJ_DIM,
    FLAVA_VIS_LAMBDA,
    FUSION_BATCH_SIZE,
    FUSION_EPOCHS,
    FUSION_LR,
    FUSION_LR_DECAY,
    FUSION_VAL_RATIO,
    METADATA_CAFE1,
    NUTRIENT_NAMES,
    fusion_ckpt_path,
)
from data.nutrition5k import (
    FusionDataset,
    filter_valid_dish_ids,
    load_dish_metadata,
    load_rgb_image,
    load_split_ids,
    split_train_val,
)
from data.recipe_text import load_recipe_texts
from losses.align import align_term
from losses.density import DensityLoss, assemble_totals
from losses.geometric import GeometricMultiTaskLoss
from models.flava_text import FlavaBranch
from models.fusion import RGBDFusionNet
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.metrics import compute_mae_pmae


def build_ingr_mass_targets(train_ids, metadata_path):
    """Cibles de la tête auxiliaire (test W2) : vecteur de masses PAR INGRÉDIENT (g)
    sur le vocabulaire du train, + densité glucidique par ingrédient (carb/g, constante
    dans Nutrition5k) pour reconstruire le carb depuis les masses prédites."""
    per_dish, dsum, dcnt = {}, {}, {}
    for line in open(metadata_path):
        f = line.rstrip("\n").split(",")
        if len(f) < 13 or (len(f) - 6) % 7:
            continue
        ings = []
        for k in range(6, len(f), 7):
            iid, g, cb = f[k], float(f[k + 2]), float(f[k + 5])
            if g > 0:
                ings.append((iid, g))
                dsum[iid] = dsum.get(iid, 0.0) + cb / g
                dcnt[iid] = dcnt.get(iid, 0) + 1
        per_dish[f[0]] = ings
    vocab = {}
    for d in train_ids:
        for iid, _ in per_dish.get(d, []):
            vocab.setdefault(iid, len(vocab))
    V = len(vocab)
    dens_carb = np.zeros(V, np.float32)
    for iid, j in vocab.items():
        if dcnt.get(iid):
            dens_carb[j] = dsum[iid] / dcnt[iid]

    def mass_matrix(ids):
        M = np.zeros((len(ids), V), np.float32)
        for r, d in enumerate(ids):
            for iid, g in per_dish.get(d, []):
                j = vocab.get(iid)
                if j is not None:
                    M[r, j] = g
        return M
    return vocab, mass_matrix, dens_carb


class CarbOnlyLoss(torch.nn.Module):
    """Perte mono-tâche : L1 sur le SEUL glucide, indexé par sa vraie colonne
    (NUTRIENT_NAMES.index('carb')). NB : on ne peut PAS réutiliser
    GeometricMultiTaskLoss(('carb',)) car l1_per_task indexe par position (0),
    ce qui superviserait les calories au lieu du glucide."""

    def __init__(self):
        super().__init__()
        from config import NUTRIENT_NAMES as _NN
        self.idx = _NN.index("carb")

    def forward(self, pred, target):
        l = torch.mean(torch.abs(pred[:, self.idx] - target[:, self.idx]))
        return l, {"carb": l}


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_one_epoch(fusion, loader, optimizer, criterion, device, density=False,
                    flava=None, cache=None,
                    lam_txt=FLAVA_ALIGN_LAMBDA, lam_vis=FLAVA_VIS_LAMBDA,
                    tau=FLAVA_ALIGN_TEMPERATURE, n_neg=FLAVA_ALIGN_NEG,
                    no_depth=False, aux=False, aux_mat=None, dish2row=None, aux_lambda=1.0):
    fusion.train()
    use_txt = flava is not None and flava.use_text
    use_vis = flava is not None and flava.use_image
    use_flava = use_txt or use_vis
    use_amp = device.type == "cuda"
    total = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        rgb = batch["rgb"].to(device)
        depth = batch["depth"].to(device)
        if no_depth:  # ablation RGB-only : profondeur permutée dans le batch
            depth = torch.roll(depth, shifts=1, dims=0)  # casse la corresp. RGB↔profondeur
        targets = batch["targets"].to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            # mode densité : le forward renvoie (masse, densités) ; sinon les totaux.
            if density and use_flava:
                mass, dens, z_proj = fusion(rgb, depth, return_embedding=True)
                loss, _ = criterion(mass, dens, targets)
            elif density:
                mass, dens = fusion(rgb, depth)
                loss, _ = criterion(mass, dens, targets)
            elif use_flava:
                pred, z_proj = fusion(rgb, depth, return_embedding=True)
                loss, _ = criterion(pred, targets)
            elif aux:  # tête auxiliaire masses par ingrédient (test W2)
                pred, aux_pred = fusion(rgb, depth, return_aux=True)
                loss, _ = criterion(pred, targets)
                rows = [dish2row[d] for d in batch["dish_id"]]
                tgt = torch.from_numpy(aux_mat[rows]).to(device)
                loss = loss + aux_lambda * torch.mean(torch.abs(aux_pred - tgt))
            else:
                pred = fusion(rgb, depth)
                loss, _ = criterion(pred, targets)
            if use_flava:
                zv = z_proj.float()
                batch_idx = torch.tensor(
                    [cache["dish2idx"][d] for d in batch["dish_id"]], device=device
                )
                if use_txt:
                    loss = loss + lam_txt * align_term(
                        zv, cache["txt_feats"], flava.project_text,
                        cache["group"], batch_idx, n_neg, tau
                    )
                if use_vis:
                    loss = loss + lam_vis * align_term(
                        zv, cache["img_feats"], flava.project_image,
                        cache["group"], batch_idx, n_neg, tau
                    )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(fusion, loader, total_criterion, device, density=False, no_depth=False):
    """Éval (val) sur les TOTAUX. En mode densité on reconstruit le total avec la
    masse PRÉDITE (objectif bout-en-bout, pour la sélection du meilleur epoch)."""
    fusion.eval()
    total = 0.0
    preds, gts = [], []
    for batch in loader:
        rgb = batch["rgb"].to(device)
        depth = batch["depth"].to(device)
        if no_depth:  # ablation RGB-only : profondeur permutée dans le batch
            depth = torch.roll(depth, shifts=1, dims=0)  # casse la corresp. RGB↔profondeur
        targets = batch["targets"].to(device)
        if density:
            mass, dens = fusion(rgb, depth)
            pred = assemble_totals(dens, mass)
        else:
            pred = fusion(rgb, depth)
        loss, _ = total_criterion(pred, targets)
        total += loss.item()
        preds.append(pred.cpu().numpy())
        gts.append(targets.cpu().numpy())
    return total / len(loader), compute_mae_pmae(np.concatenate(preds), np.concatenate(gts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth-source", choices=["gt", "midas"], required=True)
    parser.add_argument("--epochs", type=int, default=FUSION_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=FUSION_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=FUSION_LR)
    parser.add_argument("--lr-decay", type=float, default=FUSION_LR_DECAY)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--num-workers", type=int, default=0 if platform.system() == "Darwin" else 4)
    parser.add_argument("--flava-txt", action="store_true", help="alignement vision <-> texte recette (FLAVA)")
    parser.add_argument("--flava-vis", action="store_true", help="alignement vision <-> image FLAVA")
    parser.add_argument("--align-lambda", type=float, default=FLAVA_ALIGN_LAMBDA)
    parser.add_argument("--vis-lambda", type=float, default=FLAVA_VIS_LAMBDA)
    parser.add_argument("--align-temperature", type=float, default=FLAVA_ALIGN_TEMPERATURE)
    parser.add_argument("--align-neg", type=int, default=FLAVA_ALIGN_NEG)
    parser.add_argument("--flava-model", default=FLAVA_MODEL)
    parser.add_argument("--val-ratio", type=float, default=FUSION_VAL_RATIO)
    parser.add_argument("--no-skip", action="store_true",
                        help="entraîne sur TOUT (n'écarte pas dishes_to_skip.txt)")
    parser.add_argument("--density", action="store_true",
                        help="mode densité : prédit masse + densités /100g (supervisées "
                             "directement) ; total = densité×masse calculé à l'éval. "
                             "Aligne sur le pré-entraînement densité Recipe1M")
    parser.add_argument("--pretrained", default=None, metavar="CKPT",
                        help="checkpoint pretrain_density (Recipe1M) à charger avant "
                             "le fine-tuning ; nécessite --density")
    parser.add_argument("--seed", type=int, default=None,
                        help="graine aléatoire (reproductibilité + runs multi-graines). "
                             "Si fournie, le checkpoint reçoit un suffixe _s<seed> pour "
                             "ne pas écraser les autres graines.")
    parser.add_argument("--no-depth", action="store_true",
                        help="ablation RGB-only : la profondeur est PERMUTÉE dans le "
                             "batch (roll de 1), ce qui casse la correspondance "
                             "RGB↔profondeur — la profondeur ne porte plus aucune info "
                             "alignée sur le plat, mais garde des statistiques réelles "
                             "(pas de NaN, contrairement à une entrée nulle). Mesure la "
                             "contribution informationnelle de la profondeur. "
                             "Suffixe checkpoint _rgbonly.")
    parser.add_argument("--rgb-only", action="store_true",
                        help="ablation RGB-only VRAIE (mono-branche) : aucune branche "
                             "profondeur, aucune CAB — les features RGB alimentent "
                             "directement la fusion multi-échelle. Contrairement à "
                             "--no-depth (profondeur permutée), il n'y a plus de profondeur "
                             "du tout. Suffixe checkpoint _rgbsingle.")
    parser.add_argument("--aux-ingr-mass", action="store_true",
                        help="tête auxiliaire supervisée sur les masses PAR INGRÉDIENT (test W2). "
                             "Le carb est aussi reconstruit depuis les masses prédites × densités. "
                             "Si le carb reste ~20%%, l'image ne récupère pas la composition même "
                             "supervisée dessus. Directe + sans FLAVA. Suffixe _auxmass.")
    parser.add_argument("--aux-lambda", type=float, default=1.0,
                        help="poids de la perte auxiliaire (masses par ingrédient, L1 en g).")
    parser.add_argument("--carb-only", action="store_true",
                        help="ablation mono-tâche : perte L1 sur le SEUL glucide "
                             "(sélection de l'époque sur le PMAE glucide). Teste la "
                             "dilution multi-tâche. Incompatible avec --density. "
                             "Suffixe checkpoint _carbonly.")
    parser.add_argument("--tag", default=None,
                        help="suffixe libre ajouté au nom du checkpoint (ex. sweeps λ/LR) "
                             "pour éviter les collisions entre configurations.")
    parser.add_argument("--train-frac", type=float, default=1.0,
                        help="fraction du train utilisée (courbe d'apprentissage / "
                             "taille de données). Sous-échantillonnage aléatoire "
                             "reproductible (via --seed). Suffixe checkpoint _f<frac>.")
    args = parser.parse_args()
    need_flava = args.flava_txt or args.flava_vis
    apply_skip = not args.no_skip
    if args.carb_only and args.density:
        raise ValueError("--carb-only et --density sont incompatibles (la tête densité "
                         "reconstruit les totaux à partir de masse+densités des 4 nutriments)")
    if args.rgb_only and args.no_depth:
        raise ValueError("--rgb-only (mono-branche) et --no-depth (profondeur permutée) "
                         "sont deux ablations distinctes, à ne pas combiner")
    if args.aux_ingr_mass and (args.density or need_flava or args.carb_only):
        raise ValueError("--aux-ingr-mass s'utilise avec la tête directe, sans FLAVA ni --carb-only")
    if not (0.0 < args.train_frac <= 1.0):
        raise ValueError("--train-frac doit être dans ]0, 1]")

    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    labels = load_dish_metadata(METADATA_CAFE1)
    all_train_ids = filter_valid_dish_ids(
        load_split_ids(DEPTH_TRAIN_SPLIT), labels=labels, depth_source=args.depth_source,
        apply_skip=apply_skip,
    )
    train_ids, val_ids = split_train_val(all_train_ids, val_ratio=args.val_ratio)
    if args.train_frac < 1.0:  # courbe d'apprentissage : sous-échantillonne le train (val/test intacts)
        rng = np.random.RandomState(args.seed if args.seed is not None else 0)
        n_keep = max(1, int(round(len(train_ids) * args.train_frac)))
        keep = rng.choice(len(train_ids), size=n_keep, replace=False)
        train_ids = [train_ids[i] for i in sorted(keep)]
    test_ids = filter_valid_dish_ids(
        load_split_ids(DEPTH_TEST_SPLIT), labels=labels, depth_source=args.depth_source,
        apply_skip=apply_skip,
    )
    flava_tag = ""
    if need_flava:
        parts = []
        if args.flava_txt:
            parts.append(f"txt(λ={args.align_lambda})")
        if args.flava_vis:
            parts.append(f"vis(λ={args.vis_lambda})")
        flava_tag = f" flava[{'+'.join(parts)} τ={args.align_temperature} negs={args.align_neg}]"
    head_tag = "densité (masse + densités /100g)" if args.density else "directe (totaux)"
    print(
        f"train_fusion | depth={args.depth_source}{flava_tag} | tête={head_tag} | "
        f"skip={'off (TOUT)' if args.no_skip else 'on'} | "
        f"train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}"
    )

    train_ds = FusionDataset(train_ids, labels, train=True, depth_source=args.depth_source)
    val_ds = FusionDataset(val_ids, labels, train=False, depth_source=args.depth_source)
    test_ds = FusionDataset(test_ids, labels, train=False, depth_source=args.depth_source)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    aux_dim = 0
    aux_mat = aux_dish2row = aux_dens = aux_test_mat = None
    if args.aux_ingr_mass:  # cibles masses par ingrédient (test W2)
        vocab_a, mass_matrix, aux_dens = build_ingr_mass_targets(train_ids, METADATA_CAFE1)
        aux_dim = len(vocab_a)
        aux_mat = mass_matrix(train_ids)
        aux_dish2row = {d: i for i, d in enumerate(train_ids)}
        aux_test_mat = mass_matrix(test_ids)
        print(f"  tête auxiliaire masses/ingrédient : vocab={aux_dim} | λ_aux={args.aux_lambda}")

    fusion = RGBDFusionNet(pretrained=True, proj_dim=FLAVA_PROJ_DIM, density=args.density,
                           rgb_only=args.rgb_only, aux_ingr_dim=aux_dim).to(device)
    if args.pretrained:
        if not args.density:
            raise ValueError("--pretrained nécessite --density (mêmes têtes)")
        ckpt = load_checkpoint(Path(args.pretrained))
        fusion.load_state_dict(ckpt["model"])
        print(f"  poids pré-entraînés chargés : {args.pretrained}")
    flava = None
    cache = None
    if need_flava:
        flava = FlavaBranch(
            args.flava_model, vis_dim=FLAVA_PROJ_DIM,
            use_text=args.flava_txt, use_image=args.flava_vis,
        ).to(device)
        # Cache des features FLAVA gelées (texte et/ou image) sur tout le train.
        # Groupe-recette (toujours calculé) pour exclure les quasi-doublons des négatifs.
        corpus_ids = train_ids
        all_texts = load_recipe_texts(METADATA_CAFE1)
        corpus_texts = [all_texts.get(d, "A dish with unknown ingredients") for d in corpus_ids]
        text2group: dict[str, int] = {}
        groups = [text2group.setdefault(t, len(text2group)) for t in corpus_texts]
        cache = {
            "dish2idx": {d: i for i, d in enumerate(corpus_ids)},
            "group": torch.tensor(groups, device=device),
        }
        if args.flava_txt:
            cache["txt_feats"] = flava.encode_text_corpus(corpus_texts)
        if args.flava_vis:
            feats = []
            for i in range(0, len(corpus_ids), 32):
                imgs = [load_rgb_image(d) for d in corpus_ids[i:i + 32]]
                feats.append(flava.encode_image_corpus(imgs, chunk=32))
            cache["img_feats"] = torch.cat(feats, dim=0)
        flava.free_encoders()  # features cachées : on libère les encodeurs lourds
        print(
            f"  cache FLAVA: {len(corpus_ids)} plats ({len(text2group)} recettes uniques)"
            f"{' | txt' if args.flava_txt else ''}{' | vis' if args.flava_vis else ''}"
        )

    # criterion = perte d'entraînement (densité ou directe) ; total_criterion = perte
    # géométrique sur les 5 totaux, utilisée par l'éval (val) quel que soit le mode.
    total_criterion = GeometricMultiTaskLoss()
    if args.density:
        criterion = DensityLoss()
    elif args.carb_only:
        criterion = CarbOnlyLoss()  # L1 mono-tâche sur la vraie colonne glucide
    else:
        criterion = total_criterion
    params = list(fusion.parameters())
    if flava is not None:
        params += flava.trainable_parameters()
    optimizer = torch.optim.Adam(params, lr=args.lr)
    ckpt_path = fusion_ckpt_path(args.depth_source, args.flava_txt, args.flava_vis,
                                 args.no_skip, args.density)
    # suffixes d'ablation : ne pas écraser les runs de référence
    if args.rgb_only:
        ckpt_path = ckpt_path.with_name(f"{ckpt_path.stem}_rgbsingle{ckpt_path.suffix}")
    if args.no_depth:
        ckpt_path = ckpt_path.with_name(f"{ckpt_path.stem}_rgbonly{ckpt_path.suffix}")
    if args.carb_only:
        ckpt_path = ckpt_path.with_name(f"{ckpt_path.stem}_carbonly{ckpt_path.suffix}")
    if args.aux_ingr_mass:
        ckpt_path = ckpt_path.with_name(f"{ckpt_path.stem}_auxmass{ckpt_path.suffix}")
    if args.train_frac < 1.0:
        frac_tag = f"{args.train_frac:.2f}".rstrip("0").rstrip(".").replace(".", "p")
        ckpt_path = ckpt_path.with_name(f"{ckpt_path.stem}_f{frac_tag}{ckpt_path.suffix}")
    if args.tag:
        ckpt_path = ckpt_path.with_name(f"{ckpt_path.stem}_{args.tag}{ckpt_path.suffix}")
    if args.seed is not None:  # suffixe par graine : ne pas écraser les autres runs
        ckpt_path = ckpt_path.with_name(f"{ckpt_path.stem}_s{args.seed}{ckpt_path.suffix}")
    best_pmae = float("inf")
    best_epoch = 0

    # carb-only : sélectionner l'époque sur le PMAE glucide (les autres têtes ne sont pas supervisées)
    sel_key = (lambda m: m["carb"]["pmae"]) if args.carb_only else (lambda m: m["mean_pmae"])
    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(
            fusion, train_loader, optimizer, criterion, device,
            density=args.density, flava=flava, cache=cache,
            lam_txt=args.align_lambda, lam_vis=args.vis_lambda,
            tau=args.align_temperature, n_neg=args.align_neg,
            no_depth=args.no_depth, aux=args.aux_ingr_mass, aux_mat=aux_mat,
            dish2row=aux_dish2row, aux_lambda=args.aux_lambda,
        )
        va_loss, va_metrics = evaluate(fusion, val_loader, total_criterion, device,
                                       density=args.density, no_depth=args.no_depth)
        for p in optimizer.param_groups:
            p["lr"] *= args.lr_decay
        print(
            f"Epoch {epoch}/{args.epochs}  loss={tr:.4f}  val_loss={va_loss:.4f}  "
            f"val_PMAE={va_metrics['mean_pmae']:.2f}%  val_carb={va_metrics['carb']['pmae']:.2f}%  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        if sel_key(va_metrics) < best_pmae:
            best_pmae = sel_key(va_metrics)
            best_epoch = epoch
            state = {
                "model": fusion.state_dict(),
                "epoch": epoch,
                "val_metrics": va_metrics,
                "depth_source": args.depth_source,
                "flava_txt": args.flava_txt,
                "flava_vis": args.flava_vis,
                "no_skip": args.no_skip,
                "density": args.density,
            }
            if flava is not None:
                state["flava_proj"] = flava.state_dict()
            save_checkpoint(state, ckpt_path)

    best = load_checkpoint(ckpt_path)
    fusion.load_state_dict(best["model"])
    _, test_metrics = evaluate(fusion, test_loader, total_criterion, device,
                               density=args.density, no_depth=args.no_depth)
    print(
        f"checkpoint {ckpt_path} | best epoch {best_epoch} "
        f"| val_sel={best_pmae:.2f}% | test_PMAE={test_metrics['mean_pmae']:.2f}% "
        f"| test_carb_PMAE={test_metrics['carb']['pmae']:.2f}%"
    )
    if args.density:  # W5 : carb reconstruit avec la masse VRAIE (composition seule, cf. 17.6)
        _mi = NUTRIENT_NAMES.index("mass")
        fusion.eval(); preds_tm, gts = [], []
        with torch.no_grad():
            for batch in test_loader:
                rgb = batch["rgb"].to(device); depth = batch["depth"].to(device)
                if args.no_depth:
                    depth = torch.roll(depth, shifts=1, dims=0)
                _, dens = fusion(rgb, depth)
                true_mass = batch["targets"][:, _mi:_mi + 1].to(device)
                preds_tm.append(assemble_totals(dens, true_mass).cpu().numpy())
                gts.append(batch["targets"].numpy())
        mc = compute_mae_pmae(np.concatenate(preds_tm), np.concatenate(gts))
        print(f"  [density comp] test_carb_comp_PMAE (densité × masse VRAIE) = {mc['carb']['pmae']:.2f}%")
    if args.aux_ingr_mass:  # test W2 : carb reconstruit depuis les masses PAR INGRÉDIENT prédites
        fusion.eval()
        pred_masses, ci = [], []
        _carb = NUTRIENT_NAMES.index("carb")
        with torch.no_grad():
            for batch in test_loader:
                _, a = fusion(batch["rgb"].to(device), batch["depth"].to(device), return_aux=True)
                pred_masses.append(a.cpu().numpy())
                ci.append(batch["targets"][:, _carb].numpy())
        pred_masses = np.concatenate(pred_masses); true_carb = np.concatenate(ci)
        carb_from_aux = pred_masses @ aux_dens                      # Σ masse_ingr × densité_carb/g
        mean_c = true_carb.mean()
        carb_pmae = np.mean(np.abs(carb_from_aux - true_carb)) / mean_c * 100
        mass_mae = np.mean(np.abs(pred_masses - aux_test_mat))       # MAE masses/ingrédient (g)
        print(
            f"  [aux W2] carb reconstruit depuis masses/ingrédient prédites : "
            f"test_carb_from_aux_PMAE={carb_pmae:.2f}%  | MAE masses/ingr={mass_mae:.2f} g "
            f"(P2 oracle=1.3%, tête directe={test_metrics['carb']['pmae']:.2f}%)"
        )


if __name__ == "__main__":
    main()
