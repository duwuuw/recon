"""Repository root and default dataset location for notebooks and scripts."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    if os.environ.get("RAICOM_REPO_ROOT"):
        return Path(os.environ["RAICOM_REPO_ROOT"]).resolve()
    return Path(__file__).resolve().parents[2]


def default_data_root(*, require_existing: bool = False) -> str:
    """ImageFolder root: env ``RAICOM_DATA_ROOT``, else existing ``data/raw/dataset`` or ``weather_dataset/dataset`` under repo, else default path under repo."""
    if os.environ.get("RAICOM_DATA_ROOT"):
        out = os.environ["RAICOM_DATA_ROOT"]
    else:
        root = repo_root()
        candidates = [
            root / "data" / "raw" / "dataset",
            root / "weather_dataset" / "dataset",
        ]
        out = None
        for p in candidates:
            if p.is_dir():
                out = str(p)
                break
        if out is None:
            out = str(root / "data" / "raw" / "dataset")
    if require_existing and not Path(out).is_dir():
        raise FileNotFoundError(
            f"数据目录不存在: {out}。请将 ImageFolder 根目录放到 data/raw/dataset 或 "
            "weather_dataset/dataset，或设置环境变量 RAICOM_DATA_ROOT。"
        )
    return out
