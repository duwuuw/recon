#!/usr/bin/env python3
"""Train ConvNeXt V2 Nano (from convnext11.ipynb)."""

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
            timm_model="convnextv2_nano",
            checkpoint_name="convnext.pth",
            curves_name="convnext.png",
            batch_size=32,
            weight_decay=2.5e-4,
            save_classes_in_checkpoint=True,
        )
    )
