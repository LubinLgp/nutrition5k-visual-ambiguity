from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from config import DEPTH_MAX_M, MIDAS_MODEL_TYPE, MIDAS_WEIGHTS, TORCH_HUB_DIR


def _load_midas_core(pretrained_path: Optional[Path] = None):
    hub_dir = str(TORCH_HUB_DIR) if TORCH_HUB_DIR else None
    if hub_dir:
        TORCH_HUB_DIR.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(hub_dir)
    if pretrained_path and Path(pretrained_path).is_file():
        model = torch.hub.load(
            "intel-isl/MiDaS",
            MIDAS_MODEL_TYPE,
            pretrained=False,
            skip_validation=True,
        )
        state = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state, strict=False)
        return model
    return torch.hub.load(
        "intel-isl/MiDaS",
        MIDAS_MODEL_TYPE,
        pretrained=True,
        skip_validation=True,
    )


def _get_midas_transform():
    tf = torch.hub.load("intel-isl/MiDaS", "transforms", skip_validation=True)
    return tf.dpt_transform


class MiDaSDepth(nn.Module):
    def __init__(self, weights_path: Optional[str] = None):
        super().__init__()
        path = Path(weights_path) if weights_path else MIDAS_WEIGHTS
        self.core = _load_midas_core(path if path.is_file() else None)
        self.transform = _get_midas_transform()
        self.core.eval()
        for p in self.core.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def predict_pil(self, image: Image.Image) -> np.ndarray:
        h, w = image.size[1], image.size[0]
        img = np.array(image.convert("RGB"))
        x = self.transform(img)
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x["image"]).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(0)
        device = next(self.core.parameters()).device
        x = x.to(device)
        rel = self.core(x.float())
        if rel.dim() == 3:
            rel = rel.unsqueeze(1)
        rel = F.interpolate(rel, size=(h, w), mode="bicubic", align_corners=False)
        flat = rel.flatten(1)
        mn = flat.min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        mx = flat.max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        out = (rel - mn) / (mx - mn + 1e-6) * DEPTH_MAX_M
        return out.squeeze().cpu().numpy().astype(np.float32)
