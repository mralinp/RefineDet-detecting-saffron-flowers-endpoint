"""Supervised fine-tuning loop: ImageNet-pretrained VGG16 backbone +
randomly-initialized ARM/TCB/ODM heads, trained end-to-end on `Labeled/`
with an SGD + warmup + multistep schedule (as in the original RefineDet
paper). `train()` is also the building block `engine/semi_supervised.py`
calls once per self-training round, with an extra pseudo-labeled dataset
mixed in and a previous round's weights as the starting point.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import cv2
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from ..config import Config
from ..data.csv_io import read_labels
from ..data.dataset import LabeledFlowerDataset, collate_labeled, list_labeled_pairs, split_labeled
from ..model.loss import compute_losses
from ..model.refinedet import RefineDet
from ..utils.seed import set_seed
from ..utils.viz import render
from .evaluate import evaluate_ap
from .predict import predict_folder


def build_optimizer(model: torch.nn.Module, cfg: Config) -> torch.optim.Optimizer:
    return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)


def lr_at(global_iter: int, epoch: int, cfg: Config) -> float:
    if global_iter < cfg.lr_warmup_iters:
        return cfg.lr * (global_iter + 1) / cfg.lr_warmup_iters
    factor = 1.0
    for milestone in cfg.lr_milestones:
        if epoch >= milestone:
            factor *= cfg.lr_gamma
    return cfg.lr * factor


def save_checkpoint(path: Path, model: torch.nn.Module, cfg: Config) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": dataclasses.asdict(cfg)}, path)


def load_checkpoint(path: Path, device: torch.device) -> tuple[RefineDet, Config]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = Config(**ckpt["cfg"])
    model = RefineDet(cfg.input_size, cfg.feature_strides, pretrained_backbone=False).to(device)
    model.load_state_dict(ckpt["model"])
    return model, cfg


def _log_qualitative_image(writer: SummaryWriter, model: RefineDet, cfg: Config, image_path: Path, gt: "object", device: torch.device, global_step: int) -> None:
    """Render GT (green) vs current-model predictions (red) on one fixed
    validation image and log it to TensorBoard's Images tab, so training
    progress is visible as images, not just loss curves."""
    was_training = model.training
    for _path, pts, angs, scores in predict_folder(model, cfg, [image_path], device, batch_size=1):
        img = render(image_path, pred_points=pts, pred_angles=angs, gt_points=gt[:, :2], gt_angles=gt[:, 2])
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        writer.add_image("val/gt_green_pred_red", rgb, global_step, dataformats="HWC")
        writer.add_scalar("val/n_predictions_on_sample", len(pts), global_step)
    if was_training:
        model.train()


def train(
    cfg: Config,
    data_root: Path,
    device: torch.device,
    output_dir: Path,
    extra_train_dataset: Dataset | None = None,
    epochs: int | None = None,
    init_model: RefineDet | None = None,
    log_prefix: str = "",
) -> tuple[RefineDet, list[tuple[Path, Path]], list[dict]]:
    set_seed(cfg.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_dir / "tb"))

    pairs = list_labeled_pairs(data_root)
    train_pairs, val_pairs = split_labeled(pairs, cfg.val_split, cfg.val_seed)
    print(f"{log_prefix}Labeled: {len(train_pairs)} train / {len(val_pairs)} val images")
    sample_image_path, sample_gt = None, None
    if val_pairs:
        sample_image_path = val_pairs[0][0]
        sample_gt = read_labels(val_pairs[0][1])

    train_ds: Dataset = LabeledFlowerDataset(train_pairs, cfg.input_size, train=True)
    if extra_train_dataset is not None and len(extra_train_dataset) > 0:
        train_ds = ConcatDataset([train_ds, extra_train_dataset])
        print(f"{log_prefix}+ {len(extra_train_dataset)} pseudo-labeled images")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_labeled,
        drop_last=False,
    )

    if init_model is not None:
        model = init_model.to(device)
    else:
        model = RefineDet(cfg.input_size, cfg.feature_strides, pretrained_backbone=True).to(device)

    optimizer = build_optimizer(model, cfg)
    iters_per_epoch = max(1, len(train_loader))
    epochs = epochs if epochs is not None else cfg.epochs

    best_ap = -1.0
    history: list[dict] = []
    for epoch in range(epochs):
        model.train()
        running: dict[str, float] = {}
        for it, batch in enumerate(train_loader):
            lr = lr_at(epoch * iters_per_epoch + it, epoch, cfg)
            for g in optimizer.param_groups:
                g["lr"] = lr

            outputs = model(batch["image"].to(device))
            losses = compute_losses(outputs, model, batch["points"], batch["angles"], cfg, sample_weights=batch["weight"])

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            global_step = epoch * iters_per_epoch + it
            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + v.item()
                writer.add_scalar(f"train/{k}", v.item(), global_step)
            writer.add_scalar("train/lr", lr, global_step)
            if (it + 1) % cfg.log_every == 0:
                msg = " ".join(f"{k}={v.item():.4f}" for k, v in losses.items())
                print(f"{log_prefix}epoch {epoch + 1}/{epochs} iter {it + 1}/{iters_per_epoch} lr={lr:.2e} {msg}")

        avg = {k: v / iters_per_epoch for k, v in running.items()}
        log_line = f"{log_prefix}epoch {epoch + 1}/{epochs} avg " + " ".join(f"{k}={v:.4f}" for k, v in avg.items())

        record = {"epoch": epoch + 1, **avg}
        do_eval = val_pairs and ((epoch + 1) % cfg.eval_every == 0 or epoch == epochs - 1)
        if do_eval:
            ap_metrics = evaluate_ap(model, cfg, val_pairs, device)
            log_line += f" val_AP={ap_metrics['ap']:.4f} P={ap_metrics['precision_at_op']:.3f} R={ap_metrics['recall_at_op']:.3f}"
            record.update({f"val_{k}": v for k, v in ap_metrics.items()})
            for k, v in ap_metrics.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(f"val/{k}", v, epoch + 1)
            if sample_image_path is not None:
                _log_qualitative_image(writer, model, cfg, sample_image_path, sample_gt, device, epoch + 1)
            if ap_metrics["ap"] > best_ap:
                best_ap = ap_metrics["ap"]
                save_checkpoint(output_dir / "best.pt", model, cfg)
        print(log_line)
        history.append(record)

    save_checkpoint(output_dir / "last.pt", model, cfg)
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    writer.close()
    return model, val_pairs, history
