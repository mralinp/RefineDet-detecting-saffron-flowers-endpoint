"""Self-training (pseudo-labeling) over `Unlabeled/`, the bonus
"Semi-Supervised / Self-Supervised" step in the project brief.

Round 0 is plain supervised training on `Labeled/`. Each following round
runs the current model over `Unlabeled/`, keeps only detections above
`cfg.pseudo_label_conf_thresh` as pseudo ground truth, and continues
training on Labeled + pseudo-labeled images together -- with the
pseudo-labeled loss down-weighted (`cfg.pseudo_label_loss_weight`) since
those targets are the model's own (imperfect) predictions, not ground
truth. This is classic self-training / bootstrapping; see the technical
report for discussion of its failure modes (confirmation bias) versus
heavier alternatives (consistency regularization, contrastive
pretraining) that were out of scope for the compute budget here.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch

from ..config import Config
from ..data.dataset import PseudoLabeledDataset
from .predict import predict_folder
from .train import train


def list_unlabeled_images(data_root: Path, max_images: int) -> list[Path]:
    files = sorted(Path(data_root, "Unlabeled").glob("*.jpg"))
    return files[:max_images]


def self_train(cfg: Config, data_root: Path, device: torch.device, output_dir: Path):
    model, val_pairs, history = train(cfg, data_root, device, output_dir / "round0", log_prefix="[round 0] ")
    all_history = {"round0": history}

    unlabeled_paths = list_unlabeled_images(data_root, cfg.pseudo_label_max_images)
    pseudo_cfg = dataclasses.replace(cfg, infer_conf_thresh=cfg.pseudo_label_conf_thresh)

    for round_idx in range(1, cfg.pseudo_label_rounds + 1):
        print(f"[round {round_idx}] pseudo-labeling {len(unlabeled_paths)} unlabeled images "
              f"(conf > {cfg.pseudo_label_conf_thresh})")
        pseudo_items = []
        for path, pts, angs, _scores in predict_folder(model, pseudo_cfg, unlabeled_paths, device):
            if len(pts) > 0:
                pseudo_items.append((path, pts, angs))
        n_flowers = sum(len(p[1]) for p in pseudo_items)
        print(f"[round {round_idx}] kept {len(pseudo_items)}/{len(unlabeled_paths)} images, {n_flowers} pseudo-flowers")

        pseudo_ds = PseudoLabeledDataset(pseudo_items, cfg.input_size, weight=cfg.pseudo_label_loss_weight)
        model, val_pairs, history = train(
            cfg,
            data_root,
            device,
            output_dir / f"round{round_idx}",
            extra_train_dataset=pseudo_ds,
            epochs=cfg.pseudo_label_epochs_per_round,
            init_model=model,
            log_prefix=f"[round {round_idx}] ",
        )
        all_history[f"round{round_idx}"] = history

    return model, val_pairs, all_history
