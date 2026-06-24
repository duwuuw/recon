#!/usr/bin/env python3
"""Train DINOv2-S/14, the smallest timm DINOv2 image classifier."""

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
            timm_model="vit_small_patch14_dinov2",
            checkpoint_name="dinov2_small.pth",
            curves_name="dinov2_small.png",
            batch_size=32,
            weight_decay=2.5e-4,
            model_kwargs={"img_size": 224},
        )
    )
