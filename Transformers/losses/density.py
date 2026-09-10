import torch
import torch.nn as nn

from config import DENSITY_NUTRIENTS, NUTRIENT_NAMES
from losses.geometric import GeometricMultiTaskLoss

_MASS_IDX = NUTRIENT_NAMES.index("mass")
_DENSITY_COLS = [NUTRIENT_NAMES.index(n) for n in DENSITY_NUTRIENTS]


def totals_to_densities(targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Totaux Nutrition5k [B,5] (ordre NUTRIENT_NAMES) -> densités /100g [B,4]
    (ordre DENSITY_NUTRIENTS). densité_k = total_k / masse × 100 (masse pesée, > 0)."""
    mass = targets[:, _MASS_IDX:_MASS_IDX + 1].clamp_min(eps)
    cols = [targets[:, c:c + 1] / mass * 100.0 for c in _DENSITY_COLS]
    return torch.cat(cols, dim=1)


def assemble_totals(dens: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
    """densités [B,4] (/100g, ordre DENSITY_NUTRIENTS) + masse [B,1] (g) -> 5 totaux
    (ordre NUTRIENT_NAMES). Sert à l'évaluation (masse vraie OU masse prédite)."""
    totals_d = dens * mass / 100.0
    cols, di = [], 0
    for n in NUTRIENT_NAMES:
        if n == "mass":
            cols.append(mass)
        else:
            cols.append(totals_d[:, di:di + 1])
            di += 1
    return torch.cat(cols, dim=1)


class DensityLoss(nn.Module):
    """Perte du mode densité : à l'entraînement on NE prédit PAS le total, on prédit
    la **masse** (g) et les **densités /100g**, supervisées directement (moyenne
    géométrique des L1). Le total = densité × masse n'est calculé qu'à l'évaluation.
    Aligne l'entraînement sur le pré-entraînement Recipe1M (mêmes densités)."""

    def __init__(self):
        super().__init__()
        self.loss = GeometricMultiTaskLoss(("mass",) + tuple(DENSITY_NUTRIENTS))

    def forward(self, mass_pred, dens_pred, targets):
        gt_mass = targets[:, _MASS_IDX:_MASS_IDX + 1]
        gt_dens = totals_to_densities(targets)
        pred5 = torch.cat([mass_pred, dens_pred], dim=1)
        gt5 = torch.cat([gt_mass, gt_dens], dim=1)
        return self.loss(pred5, gt5)
