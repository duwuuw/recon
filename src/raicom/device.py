"""Pick a usable torch device (handles unsupported CUDA GPUs)."""

from __future__ import annotations

import torch


def pick_device(*, force_cpu: bool = False, cuda_device: int = 0) -> torch.device:
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        torch.zeros(1, device=f"cuda:{cuda_device}")
        return torch.device(f"cuda:{cuda_device}")
    except RuntimeError:
        return torch.device("cpu")
