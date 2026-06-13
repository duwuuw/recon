"""End-to-end timm classifier training with best-checkpoint saving."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from raicom.checkpoints import BestCheckpointTracker, load_checkpoint
from raicom.data import build_imagefolder_loaders
from raicom.paths import default_data_root, repo_root
from raicom.timm_factory import create_timm_classifier
from raicom.training import (
    collect_predictions,
    plot_training_curves,
    print_classification_report,
    train_one_epoch,
    validate,
)
from raicom.two_phase import (
    DEFAULT_TWO_PHASE,
    TwoPhaseSchedule,
    build_cosine_scheduler,
    build_optimizer,
    count_trainable_parameters,
    describe_phase,
    freeze_timm_backbone,
    unfreeze_all,
)


def ensure_src_on_path() -> None:
    src = repo_root() / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


@dataclass
class ClassifierTrainConfig:
    timm_model: str
    checkpoint_name: str
    curves_name: str
    pretrained: bool = True
    batch_size: int = 32
    weight_decay: float = 2.5e-4
    mixup_alpha: float = 0.205
    optimizer: str = "adamw"
    seed: int = 2023
    save_classes_in_checkpoint: bool = False
    drop_rate: float = 0.1
    num_workers: int = 0
    output_dir: Path | None = None
    data_root: str | None = None
    show_plots: bool = False
    print_test_report: bool = True
    two_phase: TwoPhaseSchedule = field(default_factory=TwoPhaseSchedule)

    @property
    def epochs(self) -> int:
        return self.two_phase.total_epochs


def default_output_dir() -> Path:
    return repo_root() / "checkpoints"


def _run_epoch(
    *,
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epoch: int,
    total_epochs: int,
    mixup_alpha: float,
    tracker: BestCheckpointTracker,
    class_names,
    ckpt_path: Path,
    phase_tag: str,
) -> tuple[float, float, float, float, float | None]:
    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        epoch,
        mixup_alpha=mixup_alpha,
    )
    val_loss, val_acc, macro_f1 = validate(
        model,
        val_loader,
        criterion,
        device,
        epoch,
        "Val",
        compute_f1=True,
    )
    scheduler.step()
    lr = scheduler.get_last_lr()[0]
    print(
        f"\n{phase_tag} Epoch {epoch:03d}/{total_epochs} | lr {lr:.2e} | "
        f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | "
        f"Val Loss {val_loss:.4f} | Val Acc {val_acc:.4f}"
        + (f" | Val F1 {macro_f1:.4f}" if macro_f1 is not None else "")
        + "\n"
    )
    if tracker.maybe_save(val_acc, model, classes=class_names, epoch=epoch):
        print(f"  -> 保存最佳权重 {ckpt_path} (val_acc={val_acc:.4f})\n")
    return train_loss, train_acc, val_loss, val_acc, macro_f1


def train_classifier(cfg: ClassifierTrainConfig, *, build_model_fn: Callable | None = None) -> dict:
    ensure_src_on_path()
    torch.manual_seed(cfg.seed)
    schedule = cfg.two_phase

    output_dir = cfg.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / cfg.checkpoint_name
    curves_path = output_dir / cfg.curves_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(describe_phase(schedule, 1))
    print(describe_phase(schedule, 2))

    data_root = cfg.data_root or default_data_root(require_existing=True)
    train_loader, val_loader, test_loader, num_classes, class_names = build_imagefolder_loaders(
        data_root,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    print(f"发现 {num_classes} 个类别: {class_names}")
    print(
        f"训练/验证/测试: {len(train_loader.dataset)}, "
        f"{len(val_loader.dataset)}, {len(test_loader.dataset)}"
    )

    if build_model_fn is None:

        def build_model_fn(n: int):
            return create_timm_classifier(
                cfg.timm_model, n, pretrained=cfg.pretrained, drop_rate=cfg.drop_rate
            )

    model = build_model_fn(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    freeze_timm_backbone(model)
    print(f"阶段1 可训练参数量: {count_trainable_parameters(model):,}")

    optimizer = build_optimizer(
        model,
        lr=schedule.head_lr,
        weight_decay=cfg.weight_decay,
        optimizer_name=cfg.optimizer,
    )
    scheduler = build_cosine_scheduler(
        optimizer,
        t_max=schedule.head_epochs,
        eta_min=schedule.head_eta_min,
    )

    tracker = BestCheckpointTracker(
        ckpt_path,
        save_classes=cfg.save_classes_in_checkpoint,
    )
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    val_f1s: list[float] = []

    for epoch in range(1, schedule.head_epochs + 1):
        tr_loss, tr_acc, va_loss, va_acc, macro_f1 = _run_epoch(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            total_epochs=schedule.total_epochs,
            mixup_alpha=cfg.mixup_alpha,
            tracker=tracker,
            class_names=class_names,
            ckpt_path=ckpt_path,
            phase_tag="[阶段1]",
        )
        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        val_losses.append(va_loss)
        val_accs.append(va_acc)
        if macro_f1 is not None:
            val_f1s.append(macro_f1)

    print("\n" + "=" * 60)
    print("切换至阶段2：解冻骨干网络")
    print("=" * 60 + "\n")
    unfreeze_all(model)
    print(f"阶段2 可训练参数量: {count_trainable_parameters(model):,}")

    optimizer = build_optimizer(
        model,
        lr=schedule.finetune_lr,
        weight_decay=cfg.weight_decay,
        optimizer_name=cfg.optimizer,
    )
    scheduler = build_cosine_scheduler(
        optimizer,
        t_max=schedule.finetune_epochs,
        eta_min=schedule.finetune_eta_min,
    )

    for offset, epoch in enumerate(
        range(schedule.head_epochs + 1, schedule.total_epochs + 1), start=1
    ):
        tr_loss, tr_acc, va_loss, va_acc, macro_f1 = _run_epoch(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            total_epochs=schedule.total_epochs,
            mixup_alpha=cfg.mixup_alpha,
            tracker=tracker,
            class_names=class_names,
            ckpt_path=ckpt_path,
            phase_tag=f"[阶段2 {offset}/{schedule.finetune_epochs}]",
        )
        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        val_losses.append(va_loss)
        val_accs.append(va_acc)
        if macro_f1 is not None:
            val_f1s.append(macro_f1)

    meta = load_checkpoint(ckpt_path, model, device)
    if meta is None:
        print("警告: 未写入 checkpoint（验证集从未提升），使用最后一轮权重做测试")
        eval_classes = class_names
    else:
        eval_classes = meta.get("classes") or class_names

    test_loss, test_acc, test_f1 = validate(
        model, test_loader, criterion, device, 0, "Test", compute_f1=True
    )
    print(f"\n最终测试集 Loss: {test_loss:.4f} | 准确率: {test_acc:.4f}", end="")
    if test_f1 is not None:
        print(f" | macro-F1: {test_f1:.4f}")
    else:
        print()

    if cfg.print_test_report:
        y_true, y_pred = collect_predictions(model, test_loader, device)
        print_classification_report("测试集", y_true, y_pred, eval_classes)

    plot_training_curves(
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        val_f1s=val_f1s or None,
        save_path=curves_path,
        show=cfg.show_plots,
    )

    return {
        "best_val_acc": tracker.best_metric,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "checkpoint": str(ckpt_path),
        "curves": str(curves_path),
    }
