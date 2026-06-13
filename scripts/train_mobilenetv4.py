#!/usr/bin/env python3
"""Train MobileNetV3-Large (from mobilenetv3_large_100.ipynb)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raicom.classifier import ClassifierTrainConfig
from raicom.cli import main_entry

if __name__ == "__main__":
    main_entry(
        ClassifierTrainConfig(
            timm_model="mobilenetv4_hybrid_medium",
            checkpoint_name="mobilenetv4_hybrid_medium.pth",
            curves_name="mobilenetv4_hybrid_medium.png",
            batch_size=32,
            weight_decay=2.5e-4,
        )
    )
