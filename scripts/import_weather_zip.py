#!/usr/bin/env python3
"""Import 天气识别 competition dataset from zip into data/raw/dataset/."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raicom.constants import SAMPLE_CLASS_NAMES

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}


def locate_train_zip(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"路径不存在: {path}")
    if path.is_file() and path.name.lower() == "train.zip":
        return path
    if path.is_file() and path.suffix.lower() == ".zip":
        with tempfile.TemporaryDirectory(prefix="raicom_import_") as tmp:
            tmp_path = Path(tmp)
            with zipfile.ZipFile(path) as outer:
                inner_names = [n for n in outer.namelist() if n.rstrip("/").endswith("train.zip")]
                if not inner_names:
                    raise FileNotFoundError(f"{path} 内未找到 train.zip")
                outer.extract(inner_names[0], tmp_path)
                inner = tmp_path / Path(inner_names[0])
                # import inside temp context: copy to another temp file that survives
                out = Path(tempfile.mkstemp(suffix=".train.zip", prefix="raicom_")[1])
                shutil.copy2(inner, out)
                return out
    if path.is_dir():
        found = next(path.rglob("train.zip"), None)
        if found:
            return found
    raise FileNotFoundError(f"无法从 {path} 定位 train.zip")


def parse_args():
    repo = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="从天气识别 zip 导入 ImageFolder 数据集")
    p.add_argument(
        "--zip",
        type=Path,
        dest="zip_path",
        required=True,
        help="天气识别.zip、train.zip 或已解压目录",
    )
    p.add_argument(
        "--dest",
        type=Path,
        default=repo / "data" / "raw" / "dataset",
        help="输出 ImageFolder 根目录",
    )
    p.add_argument(
        "--classes",
        nargs="+",
        default=list(SAMPLE_CLASS_NAMES),
        help="要导入的类别文件夹名",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def import_from_train_zip(train_zip: Path, dest: Path, classes: list[str], *, dry_run: bool) -> dict[str, int]:
    counts: dict[str, int] = {}
    if dest.exists() and not dry_run:
        shutil.rmtree(dest)
    if not dry_run:
        dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(train_zip) as z:
        members = z.namelist()
        prefix = "train/"
        for cls in classes:
            cls_prefix = f"{prefix}{cls}/"
            files = [
                n
                for n in members
                if n.startswith(cls_prefix) and Path(n).suffix.lower() in IMAGE_SUFFIXES
            ]
            if not files:
                raise SystemExit(f"zip 中缺少类别目录 train/{cls}/")
            counts[cls] = len(files)
            if dry_run:
                continue
            out_dir = dest / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            for member in files:
                with z.open(member) as src, open(out_dir / Path(member).name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    return counts


def main() -> None:
    args = parse_args()
    train_zip = locate_train_zip(args.zip_path)
    temp_zip = train_zip.name.startswith("raicom_") and train_zip.suffix == ".zip"
    try:
        print(f"train.zip: {train_zip}")
        print(f"dest:      {args.dest.resolve()}")
        print(f"classes:   {args.classes}")
        counts = import_from_train_zip(train_zip, args.dest, args.classes, dry_run=args.dry_run)
    finally:
        if temp_zip and train_zip.is_file():
            train_zip.unlink(missing_ok=True)

    total = sum(counts.values())
    for cls, n in counts.items():
        print(f"  {cls}: {n}")
    print(f"done: {len(counts)} classes, {total} images -> {args.dest}")


if __name__ == "__main__":
    main()
