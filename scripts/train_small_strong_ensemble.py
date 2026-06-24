#!/usr/bin/env python3
"""Alias for the small-strong timm ensemble trainer."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_ensemble_four_models_meta import main


if __name__ == "__main__":
    main()
