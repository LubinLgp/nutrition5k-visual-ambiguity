import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DENSITY_NUTRIENTS, FLAVA_PROJ_DIM, NUTRIENT_NAMES
from models.backbones import ResNet101MultiScale
from models.cab import CrossModalAttentionBlock


def _make_res_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    layers = []
    if stride != 1 or in_ch != out_ch:
        layers.append(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        in_ch = out_ch
    layers.extend(
        [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
    )
    return nn.Sequential(*layers)


class MultiScaleFusion(nn.Module):
    def __init__(self, channels: list[int]):
        super().__init__()
        self.transitions = nn.ModuleList()
        for i in range(1, len(channels)):
            in_ch, out_ch = channels[i - 1], channels[i]
            stride = 1 if i == 1 else 2
            self.transitions.append(_make_res_block(in_ch, out_ch, stride=stride))

    def forward(self, cross_feats: list[torch.Tensor]) -> torch.Tensor:
        f = cross_feats[0]
        for i, res in enumerate(self.transitions, start=1):
            f = cross_feats[i] + res(f)
        return f


class RGBDFusionNet(nn.Module):
    """Fusion RGB-D -> nutriments.

    Deux modes de tête :
      - direct (``density=False``, historique) : régression directe des 5 totaux ;
        forward -> pred [B,5] (ordre NUTRIENT_NAMES).
      - densité (``density=True``) : prédit la **masse** (g) + les **densités /100g**
        des nutriments SAUF la masse ; forward -> (mass [B,1], dens [B,4]). Le total
        (= densité × masse / 100) n'est PAS calculé ici : il l'est à l'évaluation, au
        choix avec la masse vraie ou la masse prédite. Aligne l'entraînement sur le
        pré-entraînement densité Recipe1M (mêmes densités).
    """

    def __init__(self, pretrained: bool = True, proj_dim: int = FLAVA_PROJ_DIM,
                 density: bool = False):
        super().__init__()
        self.density = density
        self.rgb_backbone = ResNet101MultiScale(3, pretrained=pretrained)
        self.depth_backbone = ResNet101MultiScale(1, pretrained=pretrained)
        ch = self.rgb_backbone.out_channels
        self.cabs = nn.ModuleList([CrossModalAttentionBlock(c) for c in ch])
        self.ms_fusion = MultiScaleFusion(ch)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(ch[-1], 2048)
        if density:
            self.mass_head = nn.Linear(2048, 1)
            self.density_heads = nn.ModuleDict({n: nn.Linear(2048, 1) for n in DENSITY_NUTRIENTS})
        else:
            self.heads = nn.ModuleDict({n: nn.Linear(2048, 1) for n in NUTRIENT_NAMES})
        # Tête de projection dédiée au contrastif (sans ReLU terminal) : découple
        # l'alignement vision-texte de la feature de régression et lève la
        # contrainte de l'orthant positif imposée par le ReLU.
        self.proj_head = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, proj_dim),
        )

    def encode_visual(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        rgb_feats = self.rgb_backbone(rgb)
        depth_feats = self.depth_backbone(depth)
        cross = [cab(r, d) for cab, r, d in zip(self.cabs, rgb_feats, depth_feats)]
        f4 = self.ms_fusion(cross)
        x = self.gap(f4).flatten(1)
        return self.fc(x)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor, return_embedding: bool = False):
        h = self.encode_visual(rgb, depth)
        feat = torch.relu(h)
        if self.density:
            # softplus : masse et densités sont des grandeurs positives.
            mass = F.softplus(self.mass_head(feat))
            dens = F.softplus(torch.cat([self.density_heads[n](feat) for n in DENSITY_NUTRIENTS], dim=1))
            if return_embedding:
                return mass, dens, self.proj_head(h)
            return mass, dens
        pred = torch.cat([self.heads[n](feat) for n in NUTRIENT_NAMES], dim=1)
        if return_embedding:
            return pred, self.proj_head(h)
        return pred
