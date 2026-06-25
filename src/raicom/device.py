"""Pick a usable torch device (handles unsupported CUDA GPUs)."""

from __future__ import annotations

import torch


def pick_device(*, force_cpu: bool = False, cuda_device: int = 0) -> torch.device:
    if force_cpu:
        print("Device: cpu (--cpu)")
        return torch.device("cpu")
    if not torch.cuda.is_available():
        print(
            "警告: 当前 Python 无 CUDA，使用 CPU。"
            "请 conda activate fm 或 xianyu 后再训练（需 torch 2.x + cu128）。"
        )
        return torch.device("cpu")
    try:
        torch.zeros(1, device=f"cuda:{cuda_device}")
        name = torch.cuda.get_device_name(cuda_device)
        print(f"Device: cuda:{cuda_device} ({name})")
        return torch.device(f"cuda:{cuda_device}")
    except RuntimeError as exc:
        print(
            f"警告: CUDA 不可用 ({exc})，回退 CPU。"
            "RTX 50 系请使用 fm / xianyu 环境，勿用 base 自带 Python。"
        )
        return torch.device("cpu")
