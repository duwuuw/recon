#!/usr/bin/env python3
"""Train MambaOut Kobe (from mambaout_kobe.ipynb)."""

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
            timm_model="mambaout_kobe",
            checkpoint_name="mambaout_kobe.pth",
            curves_name="training_curves_mamba.png",
            batch_size=56,
            weight_decay=2e-4,
        )
    )
