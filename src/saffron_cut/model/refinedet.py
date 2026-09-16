"""RefineDet adapted from box detection to center+angle detection.

Architecture (faithful to Zhang et al., CVPR 2018, "Single-Shot Refinement
Neural Network for Object Detection"):

    image -> VGGAtrousBackbone -> 4 feature maps (conv4_3, fc7, conv6_2, conv7_2)
                                        |                        |
                                    ARM heads                 TCB (top-down fusion)
                                        |                        |
                          (objectness, center offset)     ODM heads on fused features
                                                       (objectness, center offset, angle)

The one structural change from the paper: every head regresses a 2-D
center offset (dx, dy) instead of a 4-D box offset (dx, dy, dw, dh) -- we
have no box to regress, only a point -- and the ODM gains a third output,
(sin, cos) of the cutting angle. The ARM -> ODM cascade (coarse anchor
refinement feeding a finer second-stage regression, plus negative-anchor
filtering) is preserved, since that cascade is the actual contribution of
the RefineDet paper.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .anchors import make_anchor_centers
from .backbone import VGGAtrousBackbone
from .tcb import TCB


class ARMHead(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )
        self.cls = nn.Conv2d(in_channels, 2, kernel_size=3, padding=1)
        self.loc = nn.Conv2d(in_channels, 2, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        t = self.trunk(x)
        return self.cls(t), self.loc(t)


class ODMHead(nn.Module):
    def __init__(self, in_channels: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )
        self.cls = nn.Conv2d(in_channels, 2, kernel_size=3, padding=1)
        self.loc = nn.Conv2d(in_channels, 2, kernel_size=3, padding=1)
        self.angle = nn.Conv2d(in_channels, 2, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        t = self.trunk(x)
        return self.cls(t), self.loc(t), self.angle(t)


def _flatten(t: torch.Tensor) -> torch.Tensor:
    n, c = t.shape[0], t.shape[1]
    return t.permute(0, 2, 3, 1).reshape(n, -1, c)


class RefineDet(nn.Module):
    def __init__(self, input_size: int, feature_strides: tuple[int, ...], pretrained_backbone: bool = True):
        super().__init__()
        self.input_size = input_size
        self.feature_strides = feature_strides

        self.backbone = VGGAtrousBackbone(pretrained=pretrained_backbone)
        self.arm_heads = nn.ModuleList([ARMHead(c) for c in self.backbone.out_channels])
        self.tcb = TCB(self.backbone.out_channels, out_channels=256)
        self.odm_heads = nn.ModuleList([ODMHead(256) for _ in self.backbone.out_channels])

        per_level_centers = make_anchor_centers(input_size, feature_strides)
        per_level_strides = [
            torch.full((c.shape[0],), float(s)) for c, s in zip(per_level_centers, feature_strides)
        ]
        self.register_buffer("anchor_centers", torch.cat(per_level_centers, dim=0), persistent=False)
        self.register_buffer("anchor_strides", torch.cat(per_level_strides, dim=0), persistent=False)
        self.num_anchors_per_level = [c.shape[0] for c in per_level_centers]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.backbone(x)

        arm_cls, arm_loc = [], []
        for feat, head in zip(feats, self.arm_heads):
            c, l = head(feat)
            arm_cls.append(_flatten(c))
            arm_loc.append(_flatten(l))

        tcb_feats = self.tcb(feats)
        odm_cls, odm_loc, odm_angle = [], [], []
        for feat, head in zip(tcb_feats, self.odm_heads):
            c, l, a = head(feat)
            odm_cls.append(_flatten(c))
            odm_loc.append(_flatten(l))
            odm_angle.append(_flatten(a))

        return {
            "arm_cls": torch.cat(arm_cls, dim=1),
            "arm_loc": torch.cat(arm_loc, dim=1),
            "odm_cls": torch.cat(odm_cls, dim=1),
            "odm_loc": torch.cat(odm_loc, dim=1),
            "odm_angle": torch.cat(odm_angle, dim=1),
        }


def decode_offset(base_centers: torch.Tensor, loc: torch.Tensor, strides: torch.Tensor) -> torch.Tensor:
    """base_centers, loc: (..., A, 2); strides: (A,). Offsets are predicted
    in units of the anchor's stride so targets stay near O(1)."""
    return base_centers + loc * strides.unsqueeze(-1)


def decode_angle_deg(angle_vec: torch.Tensor) -> torch.Tensor:
    """angle_vec: (..., A, 2) unnormalized (sin, cos)-like prediction -> degrees in [0, 360)."""
    ang = torch.atan2(angle_vec[..., 1], angle_vec[..., 0]) * (180.0 / torch.pi)
    return torch.remainder(ang, 360.0)
