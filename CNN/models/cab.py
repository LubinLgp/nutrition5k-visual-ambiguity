import torch
import torch.nn as nn


class CrossModalAttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Sigmoid(),
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Sigmoid(),
        )
        self.fuse = nn.Conv2d(channels * 2, channels, 1, bias=False)

    def forward(self, rgb_feat: torch.Tensor, depth_feat: torch.Tensor) -> torch.Tensor:
        add = rgb_feat + depth_feat
        ca = self.channel_attn(add)
        sa = self.spatial_attn(add.mean(dim=1, keepdim=True))
        attn = ca * sa
        rgb_e = rgb_feat * attn
        depth_e = depth_feat * attn
        return self.fuse(torch.cat([rgb_e, depth_e], dim=1))
