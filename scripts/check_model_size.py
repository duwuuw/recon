#!/usr/bin/env python3
"""Estimate model parameter memory (from check.ipynb)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raicom.timm_factory import create_timm_classifier


def parse_args():
    p = argparse.ArgumentParser(description="Print model parameter count and FP32 size (MB)")
    p.add_argument("--model", default="convnextv2_nano")
    p.add_argument("--num-classes", type=int, default=11)
    p.add_argument("--pretrained", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    model = create_timm_classifier(
        args.model, args.num_classes, pretrained=args.pretrained
    )
    total = sum(p.numel() for p in model.parameters())
    mb = total * 4 / 1024 / 1024
    print(f"model={args.model} num_classes={args.num_classes}")
    print(f"parameters={total:,}")
    print(f"fp32_size_mb={mb:.4f}")


if __name__ == "__main__":
    main()
