#!/usr/bin/env python3
"""Train one curated timm preset from the <=30M model zoo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raicom.classifier import train_classifier
from raicom.cli import add_classifier_args, build_config
from raicom.timm_presets import build_timm_preset_config, preset_keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a curated timm preset")
    parser.add_argument("model", nargs="?", metavar="model")
    parser.add_argument("--list", action="store_true", help="List available presets and exit")
    add_classifier_args(parser)
    args = parser.parse_args()
    if args.list:
        return args
    if args.model is None:
        parser.error("model is required unless --list is used")
    if args.model not in preset_keys():
        parser.error(f"unknown model {args.model!r}; run --list to see available presets")
    return args


def main() -> None:
    args = parse_args()
    if args.list:
        for key in preset_keys():
            print(key)
        return
    defaults = build_timm_preset_config(args.model)
    train_classifier(build_config(defaults, args))


if __name__ == "__main__":
    main()
