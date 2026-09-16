"""Device selection: prefer CUDA (GPU VM / Colab), fall back to Apple
Silicon MPS for local development, then CPU.
"""

from __future__ import annotations

import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Pick the best available device.

    `prefer` can force a choice ("cuda", "mps", "cpu") for debugging; by
    default this always picks CUDA when available so the exact same code
    trains on the GPU VM / Colab without any changes, and degrades
    gracefully to MPS/CPU for local iteration on Apple Silicon.
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_report() -> str:
    lines = [
        f"cuda available : {torch.cuda.is_available()}",
        f"mps  available : {torch.backends.mps.is_available()}",
    ]
    if torch.cuda.is_available():
        lines.append(f"cuda device    : {torch.cuda.get_device_name(0)}")
        lines.append(f"cuda count     : {torch.cuda.device_count()}")
    lines.append(f"selected       : {get_device()}")
    return "\n".join(lines)
