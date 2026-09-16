"""Anchor (grid cell) centers for the 4 detection sources.

Standard object detectors anchor several box *shapes* per cell (to cover
the range of object aspect ratios/sizes) and regress a box. We only ever
need to localize a point, so there is exactly one anchor per spatial cell:
its own receptive-field center. What differs per level is just the stride.
"""

from __future__ import annotations

import torch


def make_anchor_centers(input_size: int, strides: tuple[int, ...]) -> list[torch.Tensor]:
    """Return, for each stride, an (H*W, 2) tensor of [x, y] anchor centers
    in network-input pixel space, row-major (matches a [H, W, C] head
    output flattened the same way)."""
    centers = []
    for stride in strides:
        size = input_size // stride
        shifts = (torch.arange(size, dtype=torch.float32) + 0.5) * stride
        yy, xx = torch.meshgrid(shifts, shifts, indexing="ij")
        centers.append(torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1))
    return centers
