import torch
import torch.nn as nn

from config import NUTRIENT_NAMES


def l1_per_task(pred: torch.Tensor, target: torch.Tensor, names=NUTRIENT_NAMES) -> dict[str, torch.Tensor]:
    losses = {}
    for i, name in enumerate(names):
        losses[name] = torch.mean(torch.abs(pred[:, i] - target[:, i]))
    return losses


class GeometricMultiTaskLoss(nn.Module):
    """Moyenne géométrique des L1 par tâche. ``names`` permet de restreindre
    l'ensemble des tâches (ex. densités /100g sans la masse au pré-entraînement)."""

    def __init__(self, names=NUTRIENT_NAMES):
        super().__init__()
        self.names = tuple(names)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        per_task = l1_per_task(pred, target, self.names)
        stacked = torch.stack([per_task[n] for n in self.names])
        eps = 1e-6
        log_mean = torch.mean(torch.log(stacked + eps))
        total = torch.exp(log_mean)
        return total, per_task
