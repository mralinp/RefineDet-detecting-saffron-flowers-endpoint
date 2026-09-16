"""Central configuration for the saffron center+angle detector.

All tunables live here so the dataloader, model and training loop agree on
image size, anchor strides and loss weights without passing a dozen
arguments around. Loaded from YAML via `Config.from_yaml`, overridable from
the CLI in `main.py`.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml


@dataclasses.dataclass
class Config:
    # --- data ---
    data_root: str = "."  # directory containing Labeled/, Unlabeled/, Test/
    input_size: int = 512  # network input is a square input_size x input_size image
    val_split: int = 3  # number of Labeled images held out for validation (rest train)
    val_seed: int = 0

    # --- backbone / anchors ---
    # 4 detection sources, matching RefineDet's conv4_3 / fc7 / conv6_2 / conv7_2 taps.
    feature_strides: tuple[int, ...] = (8, 16, 32, 64)
    # An anchor is a positive match for a GT flower if the anchor center lies
    # within `pos_radius_cells * stride` pixels of the flower center (see
    # engine/matching.py). Every GT also gets its single nearest anchor
    # (across all levels) forced positive so no flower is ever orphaned.
    pos_radius_cells: float = 1.0
    # ARM background-confidence threshold above which a *negative* anchor is
    # dropped from the ODM loss entirely ("negative anchor filtering", the
    # cascade trick from the RefineDet paper).
    arm_neg_filter_thresh: float = 0.99
    # ARM anchors whose refined center strays further than this many pixels
    # (in network-input space) from their original grid position are also
    # dropped from ODM, since the local ODM head can no longer see them.
    arm_max_refine_shift: float = 32.0

    # --- loss ---
    neg_pos_ratio: float = 3.0  # hard-negative mining ratio, both ARM and ODM
    loc_loss_weight: float = 1.0
    angle_loss_weight: float = 1.0

    # --- training ---
    batch_size: int = 4
    epochs: int = 120
    lr: float = 1e-3
    weight_decay: float = 5e-4
    momentum: float = 0.9
    lr_warmup_iters: int = 200
    lr_milestones: tuple[int, ...] = (80, 105)
    lr_gamma: float = 0.1
    grad_clip_norm: float = 10.0
    num_workers: int = 2
    seed: int = 0

    # --- semi-supervised self-training (bonus step) ---
    pseudo_label_conf_thresh: float = 0.8
    pseudo_label_rounds: int = 2
    pseudo_label_epochs_per_round: int = 30
    pseudo_label_max_images: int = 129
    pseudo_label_loss_weight: float = 0.5  # down-weight pseudo-labeled samples

    # --- inference ---
    infer_conf_thresh: float = 0.5
    infer_nms_radius: float = 12.0  # px, in original-image space; merges duplicate detections

    # --- evaluation (validation-set AP; no ground-truth box exists so we
    # match on center distance + angular tolerance instead of box IoU) ---
    eval_dist_thresh: float = 20.0  # px, original-image space
    eval_angle_thresh: float = 30.0  # degrees, circular difference

    # --- misc ---
    checkpoint_dir: str = "checkpoints"
    log_every: int = 10

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls(**raw)

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(dataclasses.asdict(self), f, sort_keys=False)
