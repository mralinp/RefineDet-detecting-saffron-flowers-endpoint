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

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from ..config import Config
from ..data.dataset import LabeledFlowerDataset, collate_labeled, list_labeled_pairs, split_labeled
from ..model.loss import compute_losses
from ..model.refinedet import RefineDet
from ..utils.seed import set_seed
from .evaluate import evaluate_ap


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

    pairs = list_labeled_pairs(data_root)
    train_pairs, val_pairs = split_labeled(pairs, cfg.val_split, cfg.val_seed)
    print(f"{log_prefix}Labeled: {len(train_pairs)} train / {len(val_pairs)} val images")

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

            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + v.item()
            if (it + 1) % cfg.log_every == 0:
                msg = " ".join(f"{k}={v.item():.4f}" for k, v in losses.items())
                print(f"{log_prefix}epoch {epoch + 1}/{epochs} iter {it + 1}/{iters_per_epoch} lr={lr:.2e} {msg}")

        avg = {k: v / iters_per_epoch for k, v in running.items()}
        log_line = f"{log_prefix}epoch {epoch + 1}/{epochs} avg " + " ".join(f"{k}={v:.4f}" for k, v in avg.items())

        record = {"epoch": epoch + 1, **avg}
        if val_pairs:
            ap_metrics = evaluate_ap(model, cfg, val_pairs, device)
            log_line += f" val_AP={ap_metrics['ap']:.4f} P={ap_metrics['precision_at_op']:.3f} R={ap_metrics['recall_at_op']:.3f}"
            record.update({f"val_{k}": v for k, v in ap_metrics.items()})
            if ap_metrics["ap"] > best_ap:
                best_ap = ap_metrics["ap"]
                save_checkpoint(output_dir / "best.pt", model, cfg)
        print(log_line)
        history.append(record)

    save_checkpoint(output_dir / "last.pt", model, cfg)
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    return model, val_pairs, history
