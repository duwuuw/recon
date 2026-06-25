#!/usr/bin/env python3
"""Copy a fixed 4-class × 50-image subset into data/raw/dataset/."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raicom.constants import SAMPLE_CLASS_NAMES


def parse_args():
    p = argparse.ArgumentParser(description="Build repo sample dataset (4 classes × N images)")
    p.add_argument(
        "--source",
        type=Path,
        default=Path("data/raw/dataset"),
        help="Full ImageFolder root (default: repo data/raw/dataset)",
    )
    p.add_argument(
        "--dest",
        type=Path,
        default=Path("data/raw/dataset"),
        help="Output ImageFolder root in repo",
    )
    p.add_argument("--per-class", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--classes",
        nargs="+",
        default=list(SAMPLE_CLASS_NAMES),
        help="Class folder names to include",
    )
    return p.parse_args()


def main():
    import random

    args = parse_args()
    if not args.source.is_dir():
        raise SystemExit(f"源数据不存在: {args.source}")

    if args.dest.exists():
        shutil.rmtree(args.dest)
    args.dest.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}

    for cls in args.classes:
        src_dir = args.source / cls
        if not src_dir.is_dir():
            raise SystemExit(f"缺少类别目录: {src_dir}")
        images = sorted(
            p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in image_suffixes
        )
        if len(images) < args.per_class:
            raise SystemExit(f"{cls} 仅 {len(images)} 张，不足 {args.per_class}")
        picked = rng.sample(images, args.per_class)
        dst_dir = args.dest / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in picked:
            shutil.copy2(src, dst_dir / src.name)
        print(f"{cls}: copied {len(picked)} -> {dst_dir}")

    print(f"done: {len(args.classes)} classes, {args.per_class} each -> {args.dest}")


if __name__ == "__main__":
    main()
