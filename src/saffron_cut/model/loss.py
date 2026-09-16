"""RefineDet's two-stage loss, adapted to center+angle targets.

ARM loss:  2-way (bg/fg) objectness cross-entropy with hard-negative
           mining (3:1 neg:pos), + SmoothL1 on the anchor -> flower center
           offset.
ODM loss:  matching is redone against the ARM-*refined* anchor centers
           (the actual RefineDet cascade), "easy" negatives that ARM is
           already very confident are background get dropped from the ODM
           loss entirely ("negative anchor filtering", RefineDet sec 3.3),
           then the same objectness + center-offset losses apply, plus a
           circular loss on the (cos, sin) cutting-angle prediction.

Angle is regressed as an unnormalized 2-vector trained towards
(cos theta, sin theta) with smooth-L1; because atan2 is used to decode it,
only the vector's *direction* matters, so this has no discontinuity at the
0/360 wraparound the way regressing the degree value directly would.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..config import Config
from .matching import assign_targets
from .refinedet import decode_offset

BG, FG = 0, 1
FALLBACK_NEG_COUNT = 15  # negatives mined when an image/crop has zero positives


def _hard_negative_mined_ce(
    logits: torch.Tensor, pos_mask: torch.Tensor, valid_mask: torch.Tensor, neg_pos_ratio: float
) -> torch.Tensor:
    """logits: (A, 2). Cross-entropy over positives (label FG) plus the
    hardest `neg_pos_ratio * num_pos` negatives (label BG) by loss value,
    restricted to `valid_mask` (e.g. ODM's negative-anchor-filtered set)."""
    device = logits.device
    targets = torch.full((logits.shape[0],), BG, dtype=torch.long, device=device)
    targets[pos_mask] = FG
    per_anchor_loss = F.cross_entropy(logits, targets, reduction="none")

    num_pos = int(pos_mask.sum().item())
    num_neg = max(int(round(neg_pos_ratio * num_pos)), FALLBACK_NEG_COUNT if num_pos == 0 else 0)

    neg_candidates = valid_mask & ~pos_mask
    neg_losses = per_anchor_loss.masked_fill(~neg_candidates, -1.0)
    num_neg = min(num_neg, int(neg_candidates.sum().item()))
    if num_neg == 0:
        selected_neg = torch.zeros_like(pos_mask)
    else:
        _, top_idx = neg_losses.topk(num_neg)
        selected_neg = torch.zeros_like(pos_mask)
        selected_neg[top_idx] = True

    keep = pos_mask | selected_neg
    denom = max(num_pos, 1)
    return per_anchor_loss[keep].sum() / denom


def compute_losses(
    outputs: dict[str, torch.Tensor],
    model,
    gt_points: list[torch.Tensor],
    gt_angles: list[torch.Tensor],
    cfg: Config,
    sample_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    device = outputs["arm_cls"].device
    anchor_centers = model.anchor_centers.to(device)
    anchor_strides = model.anchor_strides.to(device)
    batch_size = outputs["arm_cls"].shape[0]

    arm_cls_losses, arm_loc_losses = [], []
    odm_cls_losses, odm_loc_losses, odm_angle_losses = [], [], []

    for b in range(batch_size):
        pts = gt_points[b].to(device)
        angs = gt_angles[b].to(device)

        # ---- ARM ----
        arm_pos, arm_matched = assign_targets(anchor_centers, anchor_strides, pts, cfg.pos_radius_cells)
        all_valid = torch.ones_like(arm_pos)
        arm_cls_losses.append(_hard_negative_mined_ce(outputs["arm_cls"][b], arm_pos, all_valid, cfg.neg_pos_ratio))

        if arm_pos.any():
            tgt = (pts[arm_matched[arm_pos]] - anchor_centers[arm_pos]) / anchor_strides[arm_pos].unsqueeze(-1)
            arm_loc_losses.append(F.smooth_l1_loss(outputs["arm_loc"][b][arm_pos], tgt, reduction="mean"))
        else:
            arm_loc_losses.append(outputs["arm_loc"][b].sum() * 0.0)

        # ---- ODM (matched against ARM-refined centers) ----
        refined_centers = decode_offset(anchor_centers, outputs["arm_loc"][b].detach(), anchor_strides)
        odm_pos, odm_matched = assign_targets(refined_centers, anchor_strides, pts, cfg.pos_radius_cells)

        with torch.no_grad():
            arm_bg_score = F.softmax(outputs["arm_cls"][b], dim=-1)[:, BG]
            shift = (refined_centers - anchor_centers).norm(dim=-1)
            easy_negative = (arm_bg_score > cfg.arm_neg_filter_thresh) | (shift > cfg.arm_max_refine_shift)
            valid_for_odm = odm_pos | ~easy_negative

        odm_cls_losses.append(_hard_negative_mined_ce(outputs["odm_cls"][b], odm_pos, valid_for_odm, cfg.neg_pos_ratio))

        if odm_pos.any():
            matched_pts = pts[odm_matched[odm_pos]]
            loc_tgt = (matched_pts - refined_centers[odm_pos]) / anchor_strides[odm_pos].unsqueeze(-1)
            odm_loc_losses.append(F.smooth_l1_loss(outputs["odm_loc"][b][odm_pos], loc_tgt, reduction="mean"))

            matched_angles = angs[odm_matched[odm_pos]]
            rad = torch.deg2rad(matched_angles)
            angle_tgt = torch.stack([torch.cos(rad), torch.sin(rad)], dim=-1)
            odm_angle_losses.append(F.smooth_l1_loss(outputs["odm_angle"][b][odm_pos], angle_tgt, reduction="mean"))
        else:
            odm_loc_losses.append(outputs["odm_loc"][b].sum() * 0.0)
            odm_angle_losses.append(outputs["odm_angle"][b].sum() * 0.0)

    def _reduce(per_image: list[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(per_image)
        if sample_weights is None:
            return stacked.mean()
        w = sample_weights.to(stacked.device)
        return (stacked * w).sum() / w.sum().clamp_min(1e-9)

    arm_cls_loss = _reduce(arm_cls_losses)
    arm_loc_loss = _reduce(arm_loc_losses)
    odm_cls_loss = _reduce(odm_cls_losses)
    odm_loc_loss = _reduce(odm_loc_losses)
    odm_angle_loss = _reduce(odm_angle_losses)

    total = (
        arm_cls_loss
        + cfg.loc_loss_weight * arm_loc_loss
        + odm_cls_loss
        + cfg.loc_loss_weight * odm_loc_loss
        + cfg.angle_loss_weight * odm_angle_loss
    )
    return {
        "total": total,
        "arm_cls": arm_cls_loss,
        "arm_loc": arm_loc_loss,
        "odm_cls": odm_cls_loss,
        "odm_loc": odm_loc_loss,
        "odm_angle": odm_angle_loss,
    }
