from pathlib import Path

import torch


def save_checkpoint(state: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)
