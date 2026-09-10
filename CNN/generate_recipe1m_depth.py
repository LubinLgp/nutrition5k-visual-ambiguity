#!/usr/bin/env python3
"""Génère la profondeur MiDaS (gelé) des photos Recipe1M, une fois, sur disque.

Analogue de generate_depth.py mais pour Recipe1M : MiDaS n'est jamais entraîné, il
sert juste à fabriquer une carte de profondeur par photo (comme le mode `midas` de
Nutrition5k). Résultat -> depth_midas/<image_id>.npy, réutilisé à chaque époque du
pré-entraînement de fusion. À lancer une seule fois (idempotent : saute l'existant)."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import MIDAS_WEIGHTS, PRETRAIN_IMG_SIZE, RECIPE1M_DEPTH_DIR
from data.recipe1m import load_recipe1m_samples, recipe1m_depth_path
from models.midas import MiDaSDepth
from PIL import Image


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--img-size", type=int, default=PRETRAIN_IMG_SIZE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    # Toutes partitions ; une image = un id unique. Pas de require_depth ici (c'est
    # justement ce qu'on produit).
    seen, samples = set(), []
    for part in ("train", "val", "test"):
        for s in load_recipe1m_samples(part):
            if s["image_id"] not in seen:
                seen.add(s["image_id"])
                samples.append(s)

    RECIPE1M_DEPTH_DIR.mkdir(parents=True, exist_ok=True)
    model = MiDaSDepth(str(MIDAS_WEIGHTS) if MIDAS_WEIGHTS.is_file() else None).to(device)
    todo = [s for s in samples if args.overwrite or not recipe1m_depth_path(s["image_id"]).is_file()]
    print(f"generate_recipe1m_depth | device={device} | img={args.img_size} | "
          f"{len(todo)}/{len(samples)} images")

    for s in tqdm(todo, desc="midas recipe1m"):
        img = Image.open(s["image_path"]).convert("RGB").resize((args.img_size, args.img_size))
        depth_m = model.predict_pil(img)
        np.save(recipe1m_depth_path(s["image_id"]), np.asarray(depth_m, dtype=np.float32))

    print("done")


if __name__ == "__main__":
    main()
