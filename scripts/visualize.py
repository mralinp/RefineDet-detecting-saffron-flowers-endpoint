#!/usr/bin/env python3
"""Render ground-truth-vs-prediction figures for the report: green line
segments are labeled flowers, red are the model's predictions.

    uv run python scripts/visualize.py --checkpoint checkpoints/round0/best.pt --out report/figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from saffron_cut.data.csv_io import read_labels  # noqa: E402
from saffron_cut.data.dataset import list_labeled_pairs, split_labeled  # noqa: E402
from saffron_cut.device import get_device  # noqa: E402
from saffron_cut.engine.predict import predict_folder  # noqa: E402
from saffron_cut.engine.train import load_checkpoint  # noqa: E402
from saffron_cut.utils.viz import save_render  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "report" / "figures"))
    ap.add_argument("--n", type=int, default=3, help="number of validation images to render")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    model, cfg = load_checkpoint(Path(args.checkpoint), device)

    pairs = list_labeled_pairs(args.data_root)
    _, val_pairs = split_labeled(pairs, cfg.val_split, cfg.val_seed)
    val_pairs = val_pairs[: args.n]

    image_paths = [jpg for jpg, _ in val_pairs]
    gt_by_path = {str(jpg): read_labels(csv) for jpg, csv in val_pairs}

    for path, pts, angs, scores in predict_folder(model, cfg, image_paths, device):
        gt = gt_by_path[str(path)]
        out_path = out_dir / f"{path.stem}_gt_vs_pred.jpg"
        save_render(out_path, path, pred_points=pts, pred_angles=angs, gt_points=gt[:, :2], gt_angles=gt[:, 2])
        print(f"{path.name}: {len(gt)} gt, {len(pts)} pred -> {out_path}")


if __name__ == "__main__":
    main()
