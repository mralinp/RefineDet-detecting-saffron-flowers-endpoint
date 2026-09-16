"""Anchor <-> ground-truth assignment.

Standard SSD/RefineDet match anchors to ground truth by box IoU. We have
no ground-truth box (only a center point), so an anchor is a positive
match for a flower if the anchor's own center falls within a small radius
(`pos_radius_cells` grid cells) of that flower's center -- the direct
point-detection analogue of IoU matching. Every flower is additionally
guaranteed its single closest anchor as a positive, mirroring the
"ensure every ground truth has >=1 prior" rule from SSD-style matching, so
a flower can never be orphaned even if it falls in a gap between anchors.
"""

from __future__ import annotations

import torch


def assign_targets(
    centers: torch.Tensor, strides: torch.Tensor, gt_points: torch.Tensor, pos_radius_cells: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """centers: (A, 2), strides: (A,), gt_points: (G, 2).
    Returns (pos_mask: (A,) bool, matched_gt_idx: (A,) long, -1 where unmatched)."""
    num_anchors = centers.shape[0]
    device = centers.device
    if gt_points.shape[0] == 0:
        return (
            torch.zeros(num_anchors, dtype=torch.bool, device=device),
            torch.full((num_anchors,), -1, dtype=torch.long, device=device),
        )

    dist = torch.cdist(centers, gt_points)  # (A, G)
    radius = pos_radius_cells * strides  # (A,)
    nearest_dist, nearest_idx = dist.min(dim=1)
    pos_mask = nearest_dist < radius
    matched_idx = torch.where(pos_mask, nearest_idx, torch.full_like(nearest_idx, -1))

    closest_anchor_per_gt = dist.argmin(dim=0)  # (G,)
    pos_mask[closest_anchor_per_gt] = True
    matched_idx[closest_anchor_per_gt] = torch.arange(gt_points.shape[0], device=device)

    return pos_mask, matched_idx
