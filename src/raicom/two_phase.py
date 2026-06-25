"""Two-phase fine-tuning: freeze backbone, train head, then unfreeze full model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim

DEFAULT_EARLY_STOPPING_PATIENCE = 8
DEFAULT_EARLY_STOPPING_MIN_DELTA = 1e-4


@dataclass
class TwoPhaseSchedule:
    """Phase 1: classifier head only. Phase 2: full model fine-tune."""

    head_epochs: int = 30
    finetune_epochs: int = 30
    head_lr: float = 5e-4
    head_eta_min_ratio: float = 0.001
    finetune_lr: float = 2e-7
    finetune_eta_min_ratio: float = 0.001

    @property
    def total_epochs(self) -> int:
        return self.head_epochs + self.finetune_epochs

    @property
    def head_eta_min(self) -> float:
        return self.head_lr * self.head_eta_min_ratio

    @property
    def finetune_eta_min(self) -> float:
        return self.finetune_lr * self.finetune_eta_min_ratio


DEFAULT_TWO_PHASE = TwoPhaseSchedule()

# head_only_local: classifier head only, no backbone fine-tune
HEAD_ONLY_LOCAL_SCHEDULE = TwoPhaseSchedule(head_epochs=10, finetune_epochs=0)


@dataclass
class EarlyStopper:
    """Validation-accuracy early stopping state."""

    patience: int = DEFAULT_EARLY_STOPPING_PATIENCE
    min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA
    best_metric: float = 0.0
    epochs_no_improve: int = 0
    stopped_epoch: int | None = None

    @property
    def enabled(self) -> bool:
        return self.patience > 0

    def step(self, metric: float, epoch: int) -> bool:
        if not self.enabled:
            return False
        if metric > self.best_metric + self.min_delta:
            self.best_metric = metric
            self.epochs_no_improve = 0
            return False
        self.epochs_no_improve += 1
        if self.epochs_no_improve >= self.patience:
            self.stopped_epoch = epoch
            return True
        return False

    def describe(self) -> str:
        if not self.enabled:
            return "Early stopping: disabled"
        return (
            "Early stopping: phase 2 only, "
            f"patience={self.patience}, min_delta={self.min_delta:g}"
        )


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def get_timm_classifier(model: nn.Module) -> nn.Module:
    if hasattr(model, "get_classifier"):
        return model.get_classifier()
    for name in ("head", "fc", "classifier"):
        if hasattr(model, name):
            return getattr(model, name)
    raise RuntimeError(f"无法定位分类头: {type(model).__name__}")


def freeze_timm_backbone(model: nn.Module) -> None:
    """Freeze all parameters except the timm classification head."""
    _set_requires_grad(model, False)
    _set_requires_grad(get_timm_classifier(model), True)


def unfreeze_all(model: nn.Module) -> None:
    _set_requires_grad(model, True)


def freeze_gdn_head_only(model: nn.Module) -> None:
    """Freeze GDN backbone (patch embed + blocks), train ``head`` only."""
    _set_requires_grad(model, False)
    _set_requires_grad(model.head, True)


def trainable_parameters(model: nn.Module):
    return (p for p in model.parameters() if p.requires_grad)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_optimizer(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
    optimizer_name: str = "adamw",
) -> optim.Optimizer:
    params = list(trainable_parameters(model))
    if not params:
        raise RuntimeError("没有可训练参数，请检查 freeze / unfreeze 设置")
    name = optimizer_name.lower()
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    return optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def build_cosine_scheduler(
    optimizer: optim.Optimizer,
    *,
    t_max: int,
    eta_min: float,
) -> optim.lr_scheduler.CosineAnnealingLR:
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)


def describe_phase(schedule: TwoPhaseSchedule, phase: int) -> str:
    if phase == 1:
        return (
            f"阶段1 [仅分类头] epoch 1-{schedule.head_epochs}, "
            f"lr={schedule.head_lr:g}, eta_min={schedule.head_eta_min:g}"
        )
    return (
        f"阶段2 [全网络微调] epoch {schedule.head_epochs + 1}-{schedule.total_epochs}, "
        f"lr={schedule.finetune_lr:g}, eta_min={schedule.finetune_eta_min:g}"
    )
