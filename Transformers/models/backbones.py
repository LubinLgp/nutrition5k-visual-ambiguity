import torch.nn as nn
import timm

from config import IMAGE_H, IMAGE_W, SWIN_MODEL


class SwinMultiScale(nn.Module):
    """Backbone Swin Transformer multi-échelle — interface identique à
    ResNet101MultiScale (variante CNN) : expose `.out_channels` et renvoie une
    liste de feature maps NCHW, du plus résolu au plus profond.

    Swin produit 4 échelles [/4, /8, /16, /32] (canaux [128, 256, 512, 1024]
    pour swin_base ; [96, 192, 384, 768] pour swin_tiny), contre 5 pour ResNet
    (il a en plus le niveau /4 du stem). La fusion s'auto-dimensionne sur
    `out_channels`, seul le motif de stride change (cf. MultiScaleFusion).

    in_channels=1 (depth) : timm adapte automatiquement le patch_embed
    pré-entraîné (moyenne sur les canaux d'entrée), équivalent du swap conv1
    côté ResNet.
    """

    def __init__(
        self,
        in_channels: int = 3,
        pretrained: bool = True,
        model_name: str = SWIN_MODEL,
        img_size=(IMAGE_H, IMAGE_W),
    ):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            features_only=True,
            pretrained=pretrained,
            img_size=img_size,
            in_chans=in_channels,
        )
        self.out_channels = list(self.backbone.feature_info.channels())

    def forward(self, x):
        # timm Swin sort en NHWC -> on repasse en NCHW pour la fusion conv.
        return [f.permute(0, 3, 1, 2).contiguous() for f in self.backbone(x)]
