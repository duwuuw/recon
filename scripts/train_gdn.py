#!/usr/bin/env python3
"""Train GDN (Gated Delta Network) classifier (from gdn.ipynb)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raicom.constants import NUM_CLASSES
from raicom.checkpoints import BestCheckpointTracker, load_checkpoint
from raicom.classifier import default_output_dir
from raicom.data import build_imagefolder_loaders
from raicom.paths import default_data_root
from raicom.training import plot_training_curves, train_one_epoch, validate
from raicom.two_phase import (
    DEFAULT_TWO_PHASE,
    DEFAULT_EARLY_STOPPING_MIN_DELTA,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    EarlyStopper,
    TwoPhaseSchedule,
    build_cosine_scheduler,
    build_optimizer,
    count_trainable_parameters,
    describe_phase,
    freeze_gdn_head_only,
    unfreeze_all,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train GDN weather classifier")
    p.add_argument("--data-root", default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--head-epochs", type=int, default=DEFAULT_TWO_PHASE.head_epochs)
    p.add_argument("--finetune-epochs", type=int, default=DEFAULT_TWO_PHASE.finetune_epochs)
    p.add_argument("--head-lr", type=float, default=DEFAULT_TWO_PHASE.head_lr)
    p.add_argument("--finetune-lr", type=float, default=DEFAULT_TWO_PHASE.finetune_lr)
    p.add_argument(
        "--early-stop",
        type=int,
        default=DEFAULT_EARLY_STOPPING_PATIENCE,
        help="阶段2早停 patience；0 关闭",
    )
    p.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=DEFAULT_EARLY_STOPPING_MIN_DELTA,
    )
    p.add_argument("--batch-size", type=int, default=56)
    p.add_argument("--show-plots", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    from raicom.models.gdn import GDN

    schedule = TwoPhaseSchedule(
        head_epochs=args.head_epochs,
        finetune_epochs=args.finetune_epochs,
        head_lr=args.head_lr,
        finetune_lr=args.finetune_lr,
    )
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print(describe_phase(schedule, 1))
    print(describe_phase(schedule, 2))

    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "gdn_best.pth"
    curves_path = output_dir / "training_curves_gdn.png"

    data_root = args.data_root or default_data_root(require_existing=True)
    train_loader, val_loader, test_loader, dataset_num_classes, class_names = build_imagefolder_loaders(
        data_root, batch_size=args.batch_size
    )
    if dataset_num_classes != NUM_CLASSES:
        raise ValueError(
            f"数据集有 {dataset_num_classes} 类，需要 {NUM_CLASSES} 类: {class_names}"
        )
    print(f"类别数 {NUM_CLASSES}: {class_names}")

    model = GDN(
        NUM_CLASSES,
        d_model=512,
        num_layers=6,
        dropout=0.1,
        mode="chunk",
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    weight_decay = 0.05

    freeze_gdn_head_only(model)
    print(f"阶段1 可训练参数: {count_trainable_parameters(model):,}")
    optimizer = build_optimizer(
        model, lr=schedule.head_lr, weight_decay=weight_decay, optimizer_name="adamw"
    )
    scheduler = build_cosine_scheduler(
        optimizer, t_max=schedule.head_epochs, eta_min=schedule.head_eta_min
    )

    tracker = BestCheckpointTracker(ckpt_path)
    train_losses, val_losses, train_accs, val_accs, val_f1s = [], [], [], [], []
    total = schedule.total_epochs

    for epoch in range(1, schedule.head_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, mixup_alpha=None
        )
        va_loss, va_acc, va_f1 = validate(
            model, val_loader, criterion, device, epoch, "Val", compute_f1=True
        )
        scheduler.step()
        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        train_accs.append(tr_acc)
        val_accs.append(va_acc)
        val_f1s.append(va_f1 or 0.0)
        print(
            f"[阶段1] Epoch {epoch:03d}/{total} | lr {scheduler.get_last_lr()[0]:.2e} | "
            f"train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f} f1 {va_f1:.4f}"
        )
        if tracker.maybe_save(va_acc, model, epoch=epoch):
            print(f"  -> 保存最佳权重 {ckpt_path} (val_acc={va_acc:.4f})")

    print("\n切换至阶段2：解冻骨干网络\n")
    unfreeze_all(model)
    print(f"阶段2 可训练参数: {count_trainable_parameters(model):,}")
    early_stopper = EarlyStopper(
        patience=args.early_stop,
        min_delta=args.early_stop_min_delta,
        best_metric=tracker.best_metric,
    )
    print(early_stopper.describe())
    optimizer = build_optimizer(
        model, lr=schedule.finetune_lr, weight_decay=weight_decay, optimizer_name="adamw"
    )
    scheduler = build_cosine_scheduler(
        optimizer, t_max=schedule.finetune_epochs, eta_min=schedule.finetune_eta_min
    )

    for offset, epoch in enumerate(
        range(schedule.head_epochs + 1, total + 1), start=1
    ):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, mixup_alpha=None
        )
        va_loss, va_acc, va_f1 = validate(
            model, val_loader, criterion, device, epoch, "Val", compute_f1=True
        )
        scheduler.step()
        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        train_accs.append(tr_acc)
        val_accs.append(va_acc)
        val_f1s.append(va_f1 or 0.0)
        print(
            f"[阶段2 {offset}/{schedule.finetune_epochs}] Epoch {epoch:03d}/{total} | "
            f"lr {scheduler.get_last_lr()[0]:.2e} | "
            f"train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f} f1 {va_f1:.4f}"
        )
        if tracker.maybe_save(va_acc, model, epoch=epoch):
            print(f"  -> 保存最佳权重 {ckpt_path} (val_acc={va_acc:.4f})")
        if early_stopper.step(va_acc, epoch):
            print(
                f"  -> 早停：阶段2 val_acc 连续 {early_stopper.patience} 轮未超过历史最佳 "
                f"(best={early_stopper.best_metric:.4f})"
            )
            break

    if load_checkpoint(ckpt_path, model, device) is None:
        print("警告: 未写入 checkpoint（验证集从未提升），使用最后一轮权重做测试")

    te_loss, te_acc, te_f1 = validate(
        model, test_loader, criterion, device, 0, "Test", compute_f1=True
    )
    print(f"测试集: loss={te_loss:.4f} acc={te_acc:.4f} macro_f1={te_f1:.4f}")
    plot_training_curves(
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        val_f1s=val_f1s,
        save_path=curves_path,
        show=args.show_plots,
    )


if __name__ == "__main__":
    main()
