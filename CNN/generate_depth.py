#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import DEPTH_TEST_SPLIT, DEPTH_TRAIN_SPLIT, MIDAS_WEIGHTS
from data.nutrition5k import filter_valid_dish_ids, load_rgb_image, load_split_ids
from models.midas import MiDaSDepth
from utils.depth_io import midas_depth_path, save_depth_midas


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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-skip", action="store_true",
                        help="génère aussi la profondeur pour les images 'non convenables' "
                             "(dishes_to_skip) — nécessaire pour l'entraînement/éval midas NO_SKIP")
    args = parser.parse_args()

    device = torch.device(args.device)
    train_ids = load_split_ids(DEPTH_TRAIN_SPLIT)
    test_ids = load_split_ids(DEPTH_TEST_SPLIT)
    dish_ids = sorted(set(train_ids) | set(test_ids))
    dish_ids = filter_valid_dish_ids(dish_ids, depth_source="gt", apply_skip=not args.no_skip)

    model = MiDaSDepth(str(MIDAS_WEIGHTS) if MIDAS_WEIGHTS.is_file() else None).to(device)
    todo = [d for d in dish_ids if args.overwrite or not midas_depth_path(d).is_file()]
    print(f"generate_depth | device={device} | {len(todo)}/{len(dish_ids)} plats")

    for dish_id in tqdm(todo, desc="midas"):
        rgb = load_rgb_image(dish_id)
        depth_m = model.predict_pil(rgb)
        save_depth_midas(dish_id, depth_m)

    print("done")


if __name__ == "__main__":
    main()
