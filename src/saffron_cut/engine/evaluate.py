"""A center+angle analogue of detection AP.

There is no ground-truth box here to compute IoU against, so a predicted
flower counts as a true positive against a still-unmatched ground-truth
flower in the same image when BOTH:
    - center distance   < cfg.eval_dist_thresh   (pixels, original image)
    - circular angle gap < cfg.eval_angle_thresh (degrees)
Detections are greedily matched to ground truth in descending score order
(standard COCO/VOC AP protocol), each ground-truth flower can be matched at
most once, and AP is the area under the resulting precision-recall curve
using the PASCAL VOC all-points interpolation (Everingham et al., 2010).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import torch

from ..config import Config
from ..data.csv_io import read_labels
from .predict import predict_folder


def _angle_diff(a: np.ndarray, b: float) -> np.ndarray:
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)


def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


@torch.no_grad()
def evaluate_ap(model, cfg: Config, val_pairs: list[tuple[Path, Path]], device: torch.device) -> dict:
    if not val_pairs:
        return {"ap": float("nan"), "n_images": 0, "n_gt": 0}

    low_thresh_cfg = dataclasses.replace(cfg, infer_conf_thresh=0.01)
    image_paths = [jpg for jpg, _ in val_pairs]
    gt_by_path = {str(jpg): read_labels(csv) for jpg, csv in val_pairs}

    detections = []  # (score, path, x, y, angle)
    n_gt = 0
    for jpg, csv in val_pairs:
        n_gt += read_labels(csv).shape[0]

    for path, pts, angs, scores in predict_folder(model, low_thresh_cfg, image_paths, device):
        for (x, y), a, s in zip(pts, angs, scores):
            detections.append((float(s), str(path), float(x), float(y), float(a)))

    detections.sort(key=lambda d: -d[0])
    matched = {p: np.zeros(len(gt), dtype=bool) for p, gt in gt_by_path.items()}

    tp = np.zeros(len(detections))
    fp = np.zeros(len(detections))
    for i, (score, path, x, y, a) in enumerate(detections):
        gt = gt_by_path[path]
        gt_matched = matched[path]
        if len(gt) == 0:
            fp[i] = 1
            continue
        dist = np.hypot(gt[:, 0] - x, gt[:, 1] - y)
        ang_ok = _angle_diff(gt[:, 2], a) < cfg.eval_angle_thresh
        candidate = (dist < cfg.eval_dist_thresh) & ang_ok & ~gt_matched
        if candidate.any():
            j = np.where(candidate)[0][np.argmin(dist[candidate])]
            gt_matched[j] = True
            tp[i] = 1
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / max(n_gt, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    ap = _voc_ap(recall, precision) if len(detections) else 0.0

    return {
        "ap": ap,
        "n_images": len(val_pairs),
        "n_gt": n_gt,
        "n_detections": len(detections),
        "precision_at_op": float(precision[-1]) if len(precision) else 0.0,
        "recall_at_op": float(recall[-1]) if len(recall) else 0.0,
    }
