"""Checkpoint save/load with best-weight tracking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: Path | str,
    model: torch.nn.Module,
    *,
    classes: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"state_dict": model.state_dict()}
    if classes is not None:
        payload["classes"] = list(classes)
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(
    path: Path | str,
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any] | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        blob = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location=device)
    if isinstance(blob, dict) and "state_dict" in blob:
        model.load_state_dict(blob["state_dict"])
        return blob
    model.load_state_dict(blob)
    return None


@dataclass
class BestCheckpointTracker:
    """Save model weights whenever validation accuracy improves."""

    path: Path
    best_metric: float = 0.0
    min_delta: float = 0.0
    save_classes: bool = False

    def maybe_save(
        self,
        metric: float,
        model: torch.nn.Module,
        *,
        classes: list[str] | None = None,
        epoch: int | None = None,
    ) -> bool:
        if metric <= self.best_metric + self.min_delta:
            return False
        self.best_metric = metric
        extra = {"best_val_acc": metric}
        if epoch is not None:
            extra["epoch"] = epoch
        save_checkpoint(
            self.path,
            model,
            classes=classes if self.save_classes else None,
            extra=extra,
        )
        return True
