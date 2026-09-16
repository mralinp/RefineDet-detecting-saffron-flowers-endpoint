"""Transfer Connection Blocks (TCB): RefineDet's top-down feature fusion,
the same role FPN's lateral+upsample path plays. Each backbone feature map
gets two 3x3 convs; starting from the coarsest level, every finer level
additionally adds the upsampled coarser TCB output before its final conv.
This lets the fine, high-resolution conv4_3 map (best for localizing our
small, tightly-packed flowers) see the semantic context accumulated by the
deeper layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _TCBBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 256, has_higher: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.has_higher = has_higher
        if has_higher:
            self.deconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feat: torch.Tensor, higher: torch.Tensor | None) -> torch.Tensor:
        x = self.relu(self.conv1(feat))
        x = self.conv2(x)
        if self.has_higher:
            up = self.deconv(higher)
            if up.shape[-2:] != x.shape[-2:]:
                up = F.interpolate(up, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = x + up
        x = self.relu(x)
        return self.relu(self.conv3(x))


class TCB(nn.Module):
    def __init__(self, in_channels_per_level: tuple[int, ...], out_channels: int = 256):
        super().__init__()
        n = len(in_channels_per_level)
        self.blocks = nn.ModuleList(
            [
                _TCBBlock(in_channels_per_level[i], out_channels, has_higher=(i < n - 1))
                for i in range(n)
            ]
        )

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """`features` ordered finest -> coarsest (matches backbone output).
        Processes coarsest first, returns outputs in the same finest ->
        coarsest order as the input."""
        outputs = [None] * len(features)
        higher = None
        for i in reversed(range(len(features))):
            higher = self.blocks[i](features[i], higher)
            outputs[i] = higher
        return outputs
