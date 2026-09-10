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
    METADATA_CAFE1,
    NUTRIENT_NAMES,
    fusion_ckpt_path,
)
from data.nutrition5k import (
    FusionDataset,
    filter_valid_dish_ids,
    load_dish_metadata,
    load_split_ids,
)
from losses.density import assemble_totals
from models.fusion import RGBDFusionNet
from utils.checkpoint import load_checkpoint
from utils.metrics import compute_mae_pmae

_MASS_IDX = NUTRIENT_NAMES.index("mass")


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _print_block(label, metrics, n):
    print(f"[{label}] n={n} | Mean PMAE {metrics['mean_pmae']:.2f}%")
    for name in NUTRIENT_NAMES:
        print(f"    {name}: MAE={metrics[name]['mae']:.3f} PMAE={metrics[name]['pmae']:.2f}%")


@torch.no_grad()
def run_eval(fusion, test_ids, labels, args, device, label, density=False):
    ds = FusionDataset(test_ids, labels, train=False, depth_source=args.depth_source)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    if not density:
        preds, gts = [], []
        for batch in tqdm(loader, desc=label, leave=False):
            rgb = batch["rgb"].to(device)
            depth = batch["depth"].to(device)
            preds.append(fusion(rgb, depth).cpu().numpy())
            gts.append(batch["targets"].numpy())
        _print_block(label, compute_mae_pmae(np.concatenate(preds), np.concatenate(gts)), len(test_ids))
        return
    # Mode densité : le modèle prédit (masse, densités). On reconstruit le total de
    # DEUX façons — masse VRAIE (isole la qualité des densités) et masse PRÉDITE
    # (chaîne complète) — et on rapporte les deux.
    dens_l, mass_l, gt_l = [], [], []
    for batch in tqdm(loader, desc=label, leave=False):
        rgb = batch["rgb"].to(device)
        depth = batch["depth"].to(device)
        mass, dens = fusion(rgb, depth)
        dens_l.append(dens.cpu())
        mass_l.append(mass.cpu())
        gt_l.append(batch["targets"])
    dens = torch.cat(dens_l)
    mass_pred = torch.cat(mass_l)
    gts = torch.cat(gt_l)
    true_mass = gts[:, _MASS_IDX:_MASS_IDX + 1]
    gts_np = gts.numpy()
    _print_block(f"{label} · masse vraie", compute_mae_pmae(assemble_totals(dens, true_mass).numpy(), gts_np), len(test_ids))
    _print_block(f"{label} · masse prédite", compute_mae_pmae(assemble_totals(dens, mass_pred).numpy(), gts_np), len(test_ids))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth-source", choices=["gt", "midas"], required=True)
    parser.add_argument("--flava-txt", action="store_true", help="checkpoint entraîné avec FLAVA texte")
    parser.add_argument("--flava-vis", action="store_true", help="checkpoint entraîné avec FLAVA image")
    parser.add_argument("--no-skip", action="store_true", help="checkpoint entraîné sur TOUT (suffixe _all)")
    parser.add_argument("--density", action="store_true",
                        help="checkpoint en mode densité (masse + densités /100g, suffixe _d)")
    parser.add_argument("--fusion-ckpt", type=Path, default=None)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0 if platform.system() == "Darwin" else 4)
    args = parser.parse_args()

    device = torch.device(args.device)
    fusion_ckpt = args.fusion_ckpt or fusion_ckpt_path(
        args.depth_source, args.flava_txt, args.flava_vis, args.no_skip, args.density
    )
    labels = load_dish_metadata(METADATA_CAFE1)
    raw_test = load_split_ids(DEPTH_TEST_SPLIT)
    # Test COMPLET (sans écarter les "non convenables") et test FILTRÉ (comme à l'entraînement).
    test_full = filter_valid_dish_ids(raw_test, labels=labels, depth_source=args.depth_source, apply_skip=False)
    test_filtered = filter_valid_dish_ids(raw_test, labels=labels, depth_source=args.depth_source, apply_skip=True)
    n_removed = len(test_full) - len(test_filtered)
    print(
        f"eval | depth={args.depth_source} | flava_txt={args.flava_txt} flava_vis={args.flava_vis} | "
        f"complet={len(test_full)} (dont {n_removed} 'non convenables') filtré={len(test_filtered)} | "
        f"fusion={fusion_ckpt}"
    )

    if not fusion_ckpt.is_file():
        raise FileNotFoundError(fusion_ckpt)
    ckpt = load_checkpoint(fusion_ckpt)
    # Le checkpoint dit lui-même comment il a été entraîné : on construit le modèle
    # dans le bon mode (densité ou direct) pour que les clés correspondent.
    ckpt_density = bool(ckpt.get("density", args.density))
    fusion = RGBDFusionNet(pretrained=True, density=ckpt_density).to(device)
    # Garde-fou : cohérence checkpoint <-> éval demandée.
    ckpt_txt = bool(ckpt.get("flava_txt", False))
    ckpt_vis = bool(ckpt.get("flava_vis", False))
    ckpt_depth = ckpt.get("depth_source", args.depth_source)
    if (ckpt_txt != args.flava_txt or ckpt_vis != args.flava_vis
            or ckpt_depth != args.depth_source or ckpt_density != args.density):
        print(
            f"  ⚠️ checkpoint: depth={ckpt_depth} flava_txt={ckpt_txt} flava_vis={ckpt_vis} "
            f"density={ckpt_density} ≠ éval demandée (depth={args.depth_source} "
            f"flava_txt={args.flava_txt} flava_vis={args.flava_vis} density={args.density})"
        )
    fusion.load_state_dict(ckpt["model"])
    fusion.eval()

    run_eval(fusion, test_full, labels, args, device, "test complet", density=ckpt_density)
    run_eval(fusion, test_filtered, labels, args, device, "test filtré", density=ckpt_density)


if __name__ == "__main__":
    main()
