"""Datasets for the three folders the project ships with:

    Labeled/   -> LabeledFlowerDataset (train split) / (val split)
    Unlabeled/ -> UnlabeledImageDataset, used only for pseudo-labeling
    Test/      -> InferenceImageDataset, used to produce the graded CSVs

Every jpg in this project is 1296x972, but nothing here assumes that --
`InferenceImageDataset` and `LabeledFlowerDataset.val` both carry a
`LetterboxInfo` back to the caller so predictions can be mapped from
network-input pixels back to original-image pixels regardless of source
resolution.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from . import transforms as T
from .csv_io import read_labels


def list_labeled_pairs(data_root: str | Path) -> list[tuple[Path, Path]]:
    labeled_dir = Path(data_root) / "Labeled"
    pairs = []
    for jpg in sorted(labeled_dir.glob("*.jpg")):
        csv = jpg.with_suffix(".csv")
        if csv.exists():
            pairs.append((jpg, csv))
    return pairs


def split_labeled(
    pairs: list[tuple[Path, Path]], val_split: int, seed: int
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    if val_split <= 0 or val_split >= len(pairs):
        return pairs, []
    rng = random.Random(seed)
    shuffled = pairs.copy()
    rng.shuffle(shuffled)
    val_pairs = sorted(shuffled[:val_split], key=lambda p: p[0].name)
    train_pairs = sorted(shuffled[val_split:], key=lambda p: p[0].name)
    return train_pairs, val_pairs


class LabeledFlowerDataset(Dataset):
    def __init__(self, pairs: list[tuple[Path, Path]], input_size: int, train: bool):
        self.pairs = pairs
        self.input_size = input_size
        self.train = train

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        jpg_path, csv_path = self.pairs[idx]
        image = cv2.imread(str(jpg_path))
        labels = read_labels(csv_path)
        points, angles = labels[:, :2].copy(), labels[:, 2].copy()

        if self.train:
            img_t, points, angles = T.train_transform(image, points, angles, self.input_size)
            info = None
        else:
            img_t, points, angles, info = T.eval_transform(image, points, angles, self.input_size)

        return {
            "image": img_t,
            "points": torch.from_numpy(points).float(),
            "angles": torch.from_numpy(angles).float(),
            "path": str(jpg_path),
            "letterbox": info,
            "weight": 1.0,
        }


class UnlabeledImageDataset(Dataset):
    """Unlabeled images for pseudo-labeling: plain letterboxed eval tensor."""

    def __init__(self, image_paths: list[Path], input_size: int):
        self.image_paths = image_paths
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.image_paths[idx]
        image = cv2.imread(str(path))
        empty = np.zeros((0, 2), dtype=np.float32)
        img_t, _, _, info = T.eval_transform(image, empty, np.zeros((0,), dtype=np.float32), self.input_size)
        return {"image": img_t, "path": str(path), "letterbox": info}


# Inference on Test/ uses the same logic as pseudo-labeling on Unlabeled/.
InferenceImageDataset = UnlabeledImageDataset


class PseudoLabeledDataset(Dataset):
    """Unlabeled images carrying model-generated (points, angles) targets,
    used as extra (down-weighted) training signal during self-training.
    Supports the same augmentations as LabeledFlowerDataset.
    """

    def __init__(self, items: list[tuple[Path, np.ndarray, np.ndarray]], input_size: int, weight: float = 1.0):
        # items: list of (image_path, points[N,2], angles[N])
        self.items = items
        self.input_size = input_size
        self.weight = weight

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        path, points, angles = self.items[idx]
        image = cv2.imread(str(path))
        img_t, points, angles = T.train_transform(image, points.copy(), angles.copy(), self.input_size)
        return {
            "image": img_t,
            "points": torch.from_numpy(points).float(),
            "angles": torch.from_numpy(angles).float(),
            "path": str(path),
            "letterbox": None,
            "weight": self.weight,
        }


def collate_labeled(batch: list[dict]) -> dict:
    images = torch.stack([b["image"] for b in batch], dim=0)
    return {
        "image": images,
        "points": [b["points"] for b in batch],
        "angles": [b["angles"] for b in batch],
        "path": [b["path"] for b in batch],
        "letterbox": [b["letterbox"] for b in batch],
        "weight": torch.tensor([b["weight"] for b in batch], dtype=torch.float32),
    }
