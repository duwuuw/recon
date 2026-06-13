#!/usr/bin/env python3
"""Train EfficientNet V2-S (from efficientnet.ipynb)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raicom.classifier import ClassifierTrainConfig
from raicom.cli import main_entry

if __name__ == "__main__":
    main_entry(
        ClassifierTrainConfig(
            timm_model="tf_efficientnetv2_s",
            checkpoint_name="efficientnet.pth",
            curves_name="efficientnet.png",
            batch_size=32,
            weight_decay=2.5e-4,
        )
    )
