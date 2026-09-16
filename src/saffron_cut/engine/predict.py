"""Decode raw model output into (point, angle, score) detections in
*original image* pixel space, and drive inference over a folder of images
(used for both `Test/` submission and the `Unlabeled/` pseudo-labeling
pass in semi-supervised training).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import Config
from ..data.csv_io import write_predictions
from ..data.dataset import InferenceImageDataset
from ..model.refinedet import decode_angle_deg, decode_offset

FG = 1


def _collate_inference(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "path": [b["path"] for b in batch],
        "letterbox": [b["letterbox"] for b in batch],
    }


def radius_nms(points: np.ndarray, scores: np.ndarray, radius: float) -> np.ndarray:
    """Greedy NMS on points (not boxes): suppress lower-score detections
    that fall within `radius` of an already-kept, higher-score detection."""
    order = np.argsort(-scores)
    kept_idx: list[int] = []
    kept_pts: list[np.ndarray] = []
    for idx in order:
        p = points[idx]
        if all(np.hypot(*(p - kp)) > radius for kp in kept_pts):
            kept_idx.append(idx)
            kept_pts.append(p)
    mask = np.zeros(len(points), dtype=bool)
    mask[kept_idx] = True
    return mask


@torch.no_grad()
def decode_batch(model, images: torch.Tensor, device: torch.device):
    """Returns (points, angles_deg, scores) each (B, A) / (B, A, 2) in
    *network-input* pixel space, before per-image confidence filtering."""
    model.eval()
    outputs = model(images.to(device))
    anchor_centers = model.anchor_centers.to(device)
    anchor_strides = model.anchor_strides.to(device)

    refined = decode_offset(anchor_centers, outputs["arm_loc"], anchor_strides)
    final_centers = decode_offset(refined, outputs["odm_loc"], anchor_strides)
    angles = decode_angle_deg(outputs["odm_angle"])
    scores = torch.softmax(outputs["odm_cls"], dim=-1)[..., FG]
    return final_centers.cpu().numpy(), angles.cpu().numpy(), scores.cpu().numpy()


def detections_for_image(
    points_input_px: np.ndarray,
    angles_deg: np.ndarray,
    scores: np.ndarray,
    letterbox_info,
    conf_thresh: float,
    nms_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep = scores > conf_thresh
    pts, angs, scs = points_input_px[keep], angles_deg[keep], scores[keep]
    if len(pts) == 0:
        return pts.reshape(0, 2), angs, scs

    pts_orig = letterbox_info.to_original(pts)
    nms_keep = radius_nms(pts_orig, scs, nms_radius)
    return pts_orig[nms_keep], angs[nms_keep], scs[nms_keep]


@torch.no_grad()
def predict_folder(model, cfg: Config, image_paths: list[Path], device: torch.device, batch_size: int = 4):
    """Run inference on a list of images. Yields (path, points_orig, angles_deg, scores)."""
    ds = InferenceImageDataset(image_paths, cfg.input_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=_collate_inference)
    for batch in loader:
        points, angles, scores = decode_batch(model, batch["image"], device)
        for i in range(len(batch["path"])):
            pts_o, angs_o, scs_o = detections_for_image(
                points[i], angles[i], scores[i], batch["letterbox"][i], cfg.infer_conf_thresh, cfg.infer_nms_radius
            )
            yield Path(batch["path"][i]), pts_o, angs_o, scs_o


def run_and_save_test(model, cfg: Config, test_dir: Path, device: torch.device) -> None:
    image_paths = sorted(test_dir.glob("*.jpg"))
    for path, pts, angs, scores in predict_folder(model, cfg, image_paths, device):
        out_csv = path.with_suffix(".csv")
        write_predictions(out_csv, pts, angs, scores)
        print(f"  {path.name}: {len(pts)} flowers -> {out_csv.name}")
