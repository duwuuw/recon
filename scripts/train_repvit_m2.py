#!/usr/bin/env python3
"""Train ECAResNet50D (from resnet18.ipynb)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raicom.classifier import ClassifierTrainConfig
from raicom.cli import main_entry

if __name__ == "__main__":
    main_entry(
        ClassifierTrainConfig(
            timm_model="repvit_m2_3",
            checkpoint_name="repvit_m2_3.pth",
            curves_name="repvit_m2_3.png",
            batch_size=32,
            weight_decay=2e-4,
        )
    )
