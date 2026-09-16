"""Image + (point, angle) augmentation.

The one subtlety that matters everywhere in this file: flower orientation
is a *direction*, not a scalar to leave alone under a geometric transform.
Every augmentation below transforms points as points (linear map + offset)
and transforms orientation as a unit direction vector (linear map only,
no translation), then re-derives the angle with `atan2`. That is the only
way to keep `angle` correct through crops, flips, rotation and the
non-square-to-square letterbox resize -- doing arithmetic directly on the
degree value (e.g. `angle = 180 - angle` for a flip) is easy to get wrong
sign conventions on and gets worse once flip and rotation compose.

All functions operate on numpy arrays: `image` is HWC uint8 BGR (as read by
cv2), `points` is (N, 2) float32 [x, y], `angles_deg` is (N,) float32 in
[0, 360).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import cv2
import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _unit_vector(angles_deg: np.ndarray) -> np.ndarray:
    rad = np.deg2rad(angles_deg)
    return np.stack([np.cos(rad), np.sin(rad)], axis=1)


def _angle_from_vector(vec: np.ndarray) -> np.ndarray:
    ang = np.rad2deg(np.arctan2(vec[:, 1], vec[:, 0]))
    return np.mod(ang, 360.0).astype(np.float32)


def _apply_affine(points: np.ndarray, angles_deg: np.ndarray, A: np.ndarray, t: np.ndarray):
    """p' = A @ p + t ; direction vectors transform by A only."""
    if len(points) == 0:
        return points, angles_deg
    new_points = points @ A.T + t
    dirs = _unit_vector(angles_deg) @ A.T
    new_angles = _angle_from_vector(dirs)
    return new_points.astype(np.float32), new_angles


@dataclass
class LetterboxInfo:
    scale: float
    pad_x: float
    pad_y: float
    src_w: int
    src_h: int

    def to_original(self, points_input_px: np.ndarray) -> np.ndarray:
        """Map points from network-input pixel space back to original image pixels."""
        p = np.asarray(points_input_px, dtype=np.float64)
        out = (p - np.array([self.pad_x, self.pad_y])) / self.scale
        return out


def letterbox(image: np.ndarray, size: int, points: np.ndarray, angles_deg: np.ndarray):
    """Isotropic resize (preserves angles) + centered padding to `size` x `size`."""
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    A = np.eye(2, dtype=np.float32) * scale
    t = np.array([pad_x, pad_y], dtype=np.float32)
    new_points, new_angles = _apply_affine(points, angles_deg, A, t)
    info = LetterboxInfo(scale=scale, pad_x=pad_x, pad_y=pad_y, src_w=w, src_h=h)
    return canvas, new_points, new_angles, info


def random_crop_and_rotate(
    image: np.ndarray,
    points: np.ndarray,
    angles_deg: np.ndarray,
    scale_range=(0.55, 1.0),
    rotate_range_deg=(-20.0, 20.0),
):
    h, w = image.shape[:2]
    frac = random.uniform(*scale_range)
    crop_w = max(32, round(w * frac))
    crop_h = max(32, round(h * frac))
    cx = random.uniform(crop_w / 2, w - crop_w / 2) if w > crop_w else w / 2
    cy = random.uniform(crop_h / 2, h - crop_h / 2) if h > crop_h else h / 2
    phi = random.uniform(*rotate_range_deg)
    rad = np.deg2rad(phi)
    R = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]], dtype=np.float32)
    dst_center = np.array([crop_w / 2, crop_h / 2], dtype=np.float32)
    src_center = np.array([cx, cy], dtype=np.float32)
    t = dst_center - R @ src_center
    M = np.concatenate([R, t[:, None]], axis=1).astype(np.float32)

    warped = cv2.warpAffine(
        image, M, (crop_w, crop_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101
    )
    new_points, new_angles = _apply_affine(points, angles_deg, R, t)

    if len(new_points):
        inside = (
            (new_points[:, 0] >= 0)
            & (new_points[:, 0] < crop_w)
            & (new_points[:, 1] >= 0)
            & (new_points[:, 1] < crop_h)
        )
        new_points, new_angles = new_points[inside], new_angles[inside]
    return warped, new_points, new_angles


def random_flip(image: np.ndarray, points: np.ndarray, angles_deg: np.ndarray, hflip_p=0.5, vflip_p=0.5):
    h, w = image.shape[:2]
    if random.random() < hflip_p:
        image = np.ascontiguousarray(image[:, ::-1])
        A = np.array([[-1, 0], [0, 1]], dtype=np.float32)
        t = np.array([w, 0], dtype=np.float32)
        points, angles_deg = _apply_affine(points, angles_deg, A, t)
    if random.random() < vflip_p:
        image = np.ascontiguousarray(image[::-1, :])
        A = np.array([[1, 0], [0, -1]], dtype=np.float32)
        t = np.array([0, h], dtype=np.float32)
        points, angles_deg = _apply_affine(points, angles_deg, A, t)
    return image, points, angles_deg


def color_jitter(image: np.ndarray, brightness=0.3, contrast=0.3, saturation=0.3) -> np.ndarray:
    img = image.astype(np.float32)
    if random.random() < 0.5:
        img *= 1.0 + random.uniform(-brightness, brightness)
    if random.random() < 0.5:
        mean = img.mean()
        img = (img - mean) * (1.0 + random.uniform(-contrast, contrast)) + mean
    if random.random() < 0.5:
        hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= 1.0 + random.uniform(-saturation, saturation)
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def to_tensor(image_bgr_uint8: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(image_bgr_uint8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(rgb.transpose(2, 0, 1)).float()


def train_transform(image: np.ndarray, points: np.ndarray, angles_deg: np.ndarray, input_size: int):
    image, points, angles_deg = random_crop_and_rotate(image, points, angles_deg)
    image, points, angles_deg = random_flip(image, points, angles_deg)
    image = color_jitter(image)
    image, points, angles_deg, _ = letterbox(image, input_size, points, angles_deg)
    return to_tensor(image), points, angles_deg


def eval_transform(image: np.ndarray, points: np.ndarray, angles_deg: np.ndarray, input_size: int):
    image, points, angles_deg, info = letterbox(image, input_size, points, angles_deg)
    return to_tensor(image), points, angles_deg, info
