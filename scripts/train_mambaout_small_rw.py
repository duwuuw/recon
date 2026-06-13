#!/usr/bin/env python3
"""Train MambaOut Small RW (from mambaout_small_rw.ipynb)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raicom.classifier import ClassifierTrainConfig
from raicom.cli import main_entry

if __name__ == "__main__":
    main_entry(
        ClassifierTrainConfig(
            timm_model="mambaout_small_rw",
            checkpoint_name="mambaout_small_rw.pth",
            curves_name="mambaout_small_rw.png",
            batch_size=24,
            weight_decay=2.35e-4,
            optimizer="adam",
        )
    )
