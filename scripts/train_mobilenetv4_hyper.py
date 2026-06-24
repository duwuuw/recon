#!/usr/bin/env python3
"""Train MobileNetV4 Hybrid Large."""

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
            timm_model="mobilenetv4_hybrid_large",
            checkpoint_name="mobilenetv4_hybrid_large.pth",
            curves_name="mobilenetv4_hybrid_large.png",
            batch_size=24,
            weight_decay=2.35e-4,
            optimizer="adam",
        )
    )
