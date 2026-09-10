from typing import Optional

import torch
import torch.nn.functional as F


def vision_text_contrastive_loss(
    z_vis: torch.Tensor,
    z_txt: torch.Tensor,
    temperature: float = 0.4,
    extra_txt_neg: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """InfoNCE symétrique vision <-> texte (style CLIP).

    z_vis, z_txt : [B, d], appariés (positif = diagonale).
    extra_txt_neg : négatifs texte supplémentaires [K, d] sans image appariée
    dans le batch. Ils n'ajoutent des négatifs qu'à la direction image->texte
    (colonnes en plus) ; la direction texte->image reste sur les B paires.
    Chaque image voit alors 1 positif et (B-1)+K négatifs.
    """
    v = F.normalize(z_vis, dim=-1)
    t = F.normalize(z_txt, dim=-1)
    labels = torch.arange(z_vis.size(0), device=z_vis.device)
    if extra_txt_neg is None or extra_txt_neg.numel() == 0:
        logits = v @ t.T / temperature
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    n = F.normalize(extra_txt_neg, dim=-1)
    t_all = torch.cat([t, n], dim=0)               # [B+K, d]
    logits_v2t = v @ t_all.T / temperature          # [B, B+K], positif = colonne i
    logits_t2v = t @ v.T / temperature              # [B, B]
    return 0.5 * (F.cross_entropy(logits_v2t, labels) + F.cross_entropy(logits_t2v, labels))


def sample_negatives(group: torch.Tensor, batch_idx: torch.Tensor, k: int) -> torch.Tensor:
    """k indices du corpus servant de négatifs, EXCLUANT tout ce qui partage un
    groupe (recette) avec le batch — écarte plats présents + quasi-doublons à
    recette identique (faux négatifs). Vaut pour l'alignement texte ET image.
    Partagé par l'entraînement (Nutrition5k) et le pré-entraînement (Recipe1M)."""
    if k <= 0:
        return batch_idx.new_empty(0)
    batch_groups = group[batch_idx].unique()
    eligible = (~torch.isin(group, batch_groups)).nonzero(as_tuple=True)[0]
    if eligible.numel() <= k:
        return eligible
    perm = torch.randperm(eligible.numel(), device=eligible.device)[:k]
    return eligible[perm]


def align_term(zv, feats, project, group, batch_idx, n_neg, tau):
    """Terme contrastif vision <-> (texte|image) FLAVA pour un batch : positif =
    feature FLAVA du même échantillon, négatifs = k features du corpus hors groupe.
    ``feats`` est le cache FLAVA gelé (aligné sur l'ordre du corpus), ``project`` la
    tête de projection (texte ou image)."""
    z_pos = project(feats[batch_idx]).float()
    neg_idx = sample_negatives(group, batch_idx, n_neg)
    z_neg = project(feats[neg_idx]).float() if neg_idx.numel() else None
    return vision_text_contrastive_loss(zv, z_pos, temperature=tau, extra_txt_neg=z_neg)
