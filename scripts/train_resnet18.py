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
            timm_model="ecaresnet50d",
            checkpoint_name="resnet50.pth",
            curves_name="resnet50.png",
            batch_size=32,
            weight_decay=2e-4,
        )
    )
