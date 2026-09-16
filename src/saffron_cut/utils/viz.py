"""Visualization, extending the project's own `data/plot_objects.py`
convention: each flower is drawn as a short line segment through its
center at its cutting angle."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def cosd(x: float) -> float:
    return np.cos(x / 180 * np.pi)


def sind(x: float) -> float:
    return np.sin(x / 180 * np.pi)


def draw_line_segment(image: np.ndarray, center, angle: float, color, length: float = 24, thickness: int = 2) -> None:
    x1 = center[0] - cosd(angle) * length / 2
    x2 = center[0] + cosd(angle) * length / 2
    y1 = center[1] - sind(angle) * length / 2
    y2 = center[1] + sind(angle) * length / 2
    cv2.line(image, (int(x1 + 0.5), int(y1 + 0.5)), (int(x2 + 0.5), int(y2 + 0.5)), color, thickness)
    cv2.circle(image, (int(center[0] + 0.5), int(center[1] + 0.5)), 3, color, -1)


def render(
    image_path: str | Path,
    pred_points: np.ndarray | None = None,
    pred_angles: np.ndarray | None = None,
    gt_points: np.ndarray | None = None,
    gt_angles: np.ndarray | None = None,
) -> np.ndarray:
    """Ground truth in green, predictions in red (matches the color scheme
    from the project brief's own example figure)."""
    image = cv2.imread(str(image_path))
    if gt_points is not None:
        for p, a in zip(gt_points, gt_angles):
            draw_line_segment(image, p, a, (0, 200, 0))
    if pred_points is not None:
        for p, a in zip(pred_points, pred_angles):
            draw_line_segment(image, p, a, (0, 0, 255))
    return image


def save_render(out_path: str | Path, *args, **kwargs) -> None:
    image = render(*args, **kwargs)
    cv2.imwrite(str(out_path), image)
