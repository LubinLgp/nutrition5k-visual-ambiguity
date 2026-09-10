#!/usr/bin/env python3
"""Pré-entraînement du modèle de fusion complet sur Recipe1M (RGB + profondeur MiDaS).

MiDaS est GELÉ : sa profondeur est pré-calculée (generate_recipe1m_depth.py). On
pré-entraîne TOUT RGBDFusionNet(density=True) — rgb_backbone + depth_backbone + CAB
+ fusion + têtes — par régression des densités /100g (DENSITY_NUTRIENTS). On NE
supervise PAS la masse ici (Recipe1M ne donne pas la portion de la photo) : le
mass_head s'apprendra au fine-tuning Nutrition5k. Tout le state_dict se transfère
ensuite dans la fusion. FLAVA (texte/image) activable, désactivé par défaut.
Reprise possible (--resume) pour les jobs limités à 48h.
"""
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
    DENSITY_NUTRIENTS,
    FLAVA_ALIGN_LAMBDA,
    FLAVA_ALIGN_NEG,
    FLAVA_ALIGN_TEMPERATURE,
    FLAVA_MODEL,
    FLAVA_PROJ_DIM,
    FLAVA_VIS_LAMBDA,
    PRETRAIN_BATCH_SIZE,
    PRETRAIN_EPOCHS,
    PRETRAIN_IMG_SIZE,
    PRETRAIN_LR,
    PRETRAIN_LR_DECAY,
    pretrain_ckpt_path,
)
from data.recipe1m import Recipe1MDensityDataset, load_recipe1m_image, load_recipe1m_samples
from losses.align import align_term
from losses.geometric import GeometricMultiTaskLoss
from models.flava_text import FlavaBranch
from models.fusion import RGBDFusionNet
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.metrics import compute_mae_pmae


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_one_epoch(model, loader, optimizer, criterion, device, flava=None, cache=None,
                    lam_txt=FLAVA_ALIGN_LAMBDA, lam_vis=FLAVA_VIS_LAMBDA,
                    tau=FLAVA_ALIGN_TEMPERATURE, n_neg=FLAVA_ALIGN_NEG):
    model.train()
    use_txt = flava is not None and flava.use_text
    use_vis = flava is not None and flava.use_image
    use_flava = use_txt or use_vis
    use_amp = device.type == "cuda"
    total = 0.0
    for batch in tqdm(loader, desc="pretrain", leave=False):
        rgb = batch["rgb"].to(device)
        depth = batch["depth"].to(device)
        targets = batch["targets"].to(device)  # densités /100g (DENSITY_NUTRIENTS)
        with torch.cuda.amp.autocast(enabled=use_amp):
            if use_flava:
                _mass, dens, z_proj = model(rgb, depth, return_embedding=True)
            else:
                _mass, dens = model(rgb, depth)
            # densités uniquement — pas de masse supervisée au pré-entraînement.
            loss, _ = criterion(dens, targets)
            if use_flava:
                zv = z_proj.float()
                batch_idx = batch["idx"].to(device)
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
def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    preds, gts = [], []
    for batch in loader:
        rgb = batch["rgb"].to(device)
        depth = batch["depth"].to(device)
        targets = batch["targets"].to(device)
        _mass, dens = model(rgb, depth)
        loss, _ = criterion(dens, targets)
        total += loss.item()
        preds.append(dens.cpu().numpy())
        gts.append(targets.cpu().numpy())
    metrics = compute_mae_pmae(np.concatenate(preds), np.concatenate(gts), names=DENSITY_NUTRIENTS)
    return total / len(loader), metrics


def build_flava_cache(flava, samples, device, use_txt, use_vis):
    """Cache des features FLAVA gelées, alignées sur l'ordre de ``samples`` (idx).
    Groupe = recette (recipe_id) pour exclure les quasi-doublons des négatifs."""
    id2group: dict[str, int] = {}
    groups = [id2group.setdefault(s["recipe_id"], len(id2group)) for s in samples]
    cache = {"group": torch.tensor(groups, device=device)}
    if use_txt:
        cache["txt_feats"] = flava.encode_text_corpus([s["text"] for s in samples])
    if use_vis:
        feats = []
        for i in tqdm(range(0, len(samples), 32), desc="cache img FLAVA", leave=False):
            imgs = [load_recipe1m_image(s) for s in samples[i:i + 32]]
            feats.append(flava.encode_image_corpus(imgs, chunk=32))
        cache["img_feats"] = torch.cat(feats, dim=0)
    return cache, len(id2group)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=PRETRAIN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=PRETRAIN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=PRETRAIN_LR)
    parser.add_argument("--lr-decay", type=float, default=PRETRAIN_LR_DECAY)
    parser.add_argument("--img-size", type=int, default=PRETRAIN_IMG_SIZE)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--num-workers", type=int, default=0 if platform.system() == "Darwin" else 4)
    parser.add_argument("--flava-txt", action="store_true", help="alignement vision <-> texte recette (FLAVA)")
    parser.add_argument("--flava-vis", action="store_true", help="alignement vision <-> image FLAVA")
    parser.add_argument("--align-lambda", type=float, default=FLAVA_ALIGN_LAMBDA)
    parser.add_argument("--vis-lambda", type=float, default=FLAVA_VIS_LAMBDA)
    parser.add_argument("--align-temperature", type=float, default=FLAVA_ALIGN_TEMPERATURE)
    parser.add_argument("--align-neg", type=int, default=FLAVA_ALIGN_NEG)
    parser.add_argument("--flava-model", default=FLAVA_MODEL)
    parser.add_argument("--limit", type=int, default=0, help="limite le nb d'échantillons train (debug)")
    parser.add_argument("--resume", action="store_true",
                        help="reprend depuis le checkpoint _last s'il existe (jobs limités à 48h)")
    args = parser.parse_args()
    need_flava = args.flava_txt or args.flava_vis

    device = torch.device(args.device)
    train_samples = load_recipe1m_samples("train", require_depth=True)
    val_samples = load_recipe1m_samples("val", require_depth=True)
    if not train_samples:
        print("ERREUR: aucune profondeur Recipe1M trouvée — lance d'abord "
              "generate_recipe1m_depth.py (sbatch scripts/generate_recipe1m_depth.sh)")
        sys.exit(1)
    if args.limit > 0:
        train_samples = train_samples[:args.limit]
        val_samples = val_samples[:max(1, args.limit // 5)]

    flava_tag = ""
    if need_flava:
        parts = []
        if args.flava_txt:
            parts.append(f"txt(λ={args.align_lambda})")
        if args.flava_vis:
            parts.append(f"vis(λ={args.vis_lambda})")
        flava_tag = f" flava[{'+'.join(parts)} τ={args.align_temperature} negs={args.align_neg}]"
    print(
        f"pretrain_density | fusion RGB-D (profondeur MiDaS gelée){flava_tag} | "
        f"densités={list(DENSITY_NUTRIENTS)} (carb<-sugars proxy ; masse NON supervisée) | "
        f"train={len(train_samples)} val={len(val_samples)}"
    )

    train_ds = Recipe1MDensityDataset(train_samples, train=True, img_size=args.img_size)
    val_ds = Recipe1MDensityDataset(val_samples, train=False, img_size=args.img_size)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = RGBDFusionNet(pretrained=True, proj_dim=FLAVA_PROJ_DIM, density=True).to(device)
    flava = None
    cache = None
    if need_flava:
        flava = FlavaBranch(
            args.flava_model, vis_dim=FLAVA_PROJ_DIM,
            use_text=args.flava_txt, use_image=args.flava_vis,
        ).to(device)
        cache, n_recipes = build_flava_cache(
            flava, train_samples, device, args.flava_txt, args.flava_vis
        )
        flava.free_encoders()
        print(
            f"  cache FLAVA: {len(train_samples)} images ({n_recipes} recettes uniques)"
            f"{' | txt' if args.flava_txt else ''}{' | vis' if args.flava_vis else ''}"
        )

    criterion = GeometricMultiTaskLoss(names=DENSITY_NUTRIENTS)
    params = list(model.parameters())
    if flava is not None:
        params += flava.trainable_parameters()
    optimizer = torch.optim.Adam(params, lr=args.lr)
    best_ckpt = pretrain_ckpt_path(args.flava_txt, args.flava_vis, "best")
    last_ckpt = pretrain_ckpt_path(args.flava_txt, args.flava_vis, "last")

    # Reprise : jobs cluster limités à 48h → on peut resoumettre et continuer.
    start_epoch, best_pmae, best_epoch = 1, float("inf"), 0
    if args.resume and last_ckpt.is_file():
        prev = load_checkpoint(last_ckpt)
        model.load_state_dict(prev["model"])
        optimizer.load_state_dict(prev["optimizer"])
        if flava is not None and "flava_proj" in prev:
            flava.load_state_dict(prev["flava_proj"])
        start_epoch = prev["epoch"] + 1
        best_pmae = prev.get("best_pmae", float("inf"))
        best_epoch = prev.get("best_epoch", 0)
        print(f"  reprise depuis {last_ckpt.name} → epoch {start_epoch}/{args.epochs} "
              f"(meilleur jusqu'ici {best_pmae:.2f}% @ epoch {best_epoch})")
    elif args.resume:
        print(f"  --resume mais {last_ckpt.name} absent → démarrage à neuf")

    for epoch in range(start_epoch, args.epochs + 1):
        tr = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            flava=flava, cache=cache,
            lam_txt=args.align_lambda, lam_vis=args.vis_lambda,
            tau=args.align_temperature, n_neg=args.align_neg,
        )
        va_loss, va_metrics = evaluate(model, val_loader, criterion, device)
        for p in optimizer.param_groups:
            p["lr"] *= args.lr_decay
        print(
            f"Epoch {epoch}/{args.epochs}  loss={tr:.4f}  val_loss={va_loss:.4f}  "
            f"val_PMAE(densité)={va_metrics['mean_pmae']:.2f}%  lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        improved = va_metrics["mean_pmae"] < best_pmae
        if improved:
            best_pmae = va_metrics["mean_pmae"]
            best_epoch = epoch
        # _last : état COMPLET (modèle + optimiseur + époque) pour reprise, chaque époque.
        last_state = {
            "model": model.state_dict(),  # tout le modèle de fusion -> transfert direct
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_pmae": best_pmae,
            "best_epoch": best_epoch,
            "val_metrics": va_metrics,
            "flava_txt": args.flava_txt,
            "flava_vis": args.flava_vis,
            "density_nutrients": list(DENSITY_NUTRIENTS),
            "img_size": args.img_size,
        }
        if flava is not None:
            last_state["flava_proj"] = flava.state_dict()
        save_checkpoint(last_state, last_ckpt)
        if improved:
            save_checkpoint(last_state, best_ckpt)

    best = load_checkpoint(best_ckpt)
    print(
        f"checkpoint {best_ckpt} | best epoch {best_epoch} "
        f"| val_PMAE(densité)={best['val_metrics']['mean_pmae']:.2f}%"
    )


if __name__ == "__main__":
    main()
