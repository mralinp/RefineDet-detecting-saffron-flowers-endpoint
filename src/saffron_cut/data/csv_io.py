"""Reading/writing the flower-label CSV format used by this project.

Label CSVs (Labeled/*.csv): 3 columns, no header, one row per flower:
    x, y, angle_degrees
Prediction CSVs (written into Test/*.csv): 4 columns, no header:
    x, y, angle_degrees, probability
Coordinates and angle are always in *original image* pixel space / degrees,
regardless of what resolution the network trained on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_labels(csv_path: str | Path) -> np.ndarray:
    """Return an (N, 3) float array [x, y, angle_deg]. Empty file -> (0, 3)."""
    path = Path(csv_path)
    if path.stat().st_size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr[:, :3]


def write_predictions(csv_path: str | Path, points: np.ndarray, angles: np.ndarray, scores: np.ndarray) -> None:
    """Write an (N, 4) [x, y, angle_deg, probability] CSV, no header."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    angles = np.asarray(angles, dtype=np.float64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    out = np.concatenate([points, angles[:, None], scores[:, None]], axis=1)
    np.savetxt(csv_path, out, delimiter=",", fmt="%.4f")
