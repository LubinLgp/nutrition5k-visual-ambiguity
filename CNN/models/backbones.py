import torch
import torch.nn as nn
from torchvision.models import ResNet101_Weights, resnet101


class ResNet101MultiScale(nn.Module):
    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
        base = resnet101(weights=weights)
        if in_channels != 3:
            old = base.conv1
            base.conv1 = nn.Conv2d(
                in_channels,
                old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=False,
            )
            if pretrained and in_channels == 1:
                with torch.no_grad():
                    base.conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True))
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.out_channels = [64, 256, 512, 1024, 2048]

    def forward(self, x):
        feats = []
        x = self.stem(x)
        feats.append(x)
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            x = layer(x)
            feats.append(x)
        return feats

