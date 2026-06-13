#!/usr/bin/env python3
"""Train ViT-Base RoPE (from vit11.ipynb)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raicom.classifier import ClassifierTrainConfig
from raicom.cli import main_entry

if __name__ == "__main__":
    main_entry(
        ClassifierTrainConfig(
            num_classes=4,
            timm_model="vit_base_patch16_rope_224",
            checkpoint_name="vit.pth",
            curves_name="vit.png",
            batch_size=32,
            weight_decay=2.5e-4,
        )
    )
