#!/usr/bin/env python3
"""Entry point required by the project brief: trains on Labeled/ (+
Unlabeled/ via self-training, on by default) and writes a prediction CSV
for every image in Test/, in place, next to this file.

    python main.py                       # train (+ self-train) then predict on Test/
    python main.py --no-semi-supervised  # skip the Unlabeled/ bonus stage, just train on Labeled/
    python main.py --set epochs=5 batch_size=2   # quick smoke run
    python main.py --mode predict --checkpoint checkpoints/round0/best.pt

Expects Labeled/, Unlabeled/, Test/ next to this file (or under ./data/,
for local development against this repo's own layout); see
`resolve_data_root` below.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))  # run standalone, no editable install required

from saffron_cut.config import Config  # noqa: E402
from saffron_cut.device import device_report, get_device  # noqa: E402
from saffron_cut.engine.predict import run_and_save_test  # noqa: E402
from saffron_cut.engine.semi_supervised import self_train  # noqa: E402
from saffron_cut.engine.train import load_checkpoint, train  # noqa: E402


def resolve_data_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    if (ROOT / "Labeled").exists() and (ROOT / "Test").exists():
        return ROOT
    if (ROOT / "data" / "Labeled").exists():
        return ROOT / "data"
    return ROOT


def parse_overrides(pairs: list[str]) -> dict:
    out = {}
    for p in pairs:
        key, _, value = p.partition("=")
        out[key] = yaml.safe_load(value)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--data-root", default=None, help="override auto-detected Labeled/Unlabeled/Test location")
    ap.add_argument("--mode", choices=["full", "train", "predict"], default="full")
    ap.add_argument("--checkpoint", default=None, help="resume training from / predict with this checkpoint")
    ap.add_argument("--no-semi-supervised", action="store_true", help="skip the Unlabeled/ self-training stage")
    ap.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    ap.add_argument("--set", nargs="*", default=[], metavar="key=value", help="override config fields")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config) if Path(args.config).exists() else Config()
    cfg = dataclasses.replace(cfg, **parse_overrides(args.set))

    data_root = resolve_data_root(args.data_root)
    device = get_device(args.device)
    print("=== Device ===")
    print(device_report(device))
    print(f"=== Data root: {data_root} ===")

    output_dir = Path(cfg.checkpoint_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    init_model = None
    if args.checkpoint:
        init_model, ckpt_cfg = load_checkpoint(Path(args.checkpoint), device)
        print(f"Loaded checkpoint: {args.checkpoint}")

    if args.mode == "predict":
        if init_model is None:
            for candidate in (output_dir / "best.pt", output_dir / "last.pt"):
                if candidate.exists():
                    init_model, ckpt_cfg = load_checkpoint(candidate, device)
                    print(f"Loaded checkpoint: {candidate}")
                    break
        if init_model is None:
            raise SystemExit("--mode predict needs --checkpoint, or an existing checkpoints/*.pt from a prior run")
        model, cfg = init_model, ckpt_cfg
    elif args.no_semi_supervised:
        model, _val_pairs, _history = train(cfg, data_root, device, output_dir, init_model=init_model)
    else:
        model, _val_pairs, _history = self_train(cfg, data_root, device, output_dir)

    if args.mode in ("full", "predict"):
        test_dir = data_root / "Test"
        print(f"=== Inference on {test_dir} ===")
        run_and_save_test(model, cfg, test_dir, device)

    print("Done.")


if __name__ == "__main__":
    main()
