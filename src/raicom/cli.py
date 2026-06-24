"""Shared CLI for classifier training scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from raicom.classifier import ClassifierTrainConfig, default_output_dir, train_classifier
from raicom.two_phase import TwoPhaseSchedule


def add_classifier_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        default=None,
        help="ImageFolder 根目录（默认 RAICOM_DATA_ROOT 或 data/raw/dataset）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="checkpoint 与曲线输出目录（默认 checkpoints/）",
    )
    parser.add_argument(
        "--head-epochs",
        type=int,
        default=None,
        help="阶段1 epoch 数（仅训练分类头，默认 80）",
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=None,
        help="阶段2 epoch 数（全网络微调，默认 20）",
    )
    parser.add_argument("--head-lr", type=float, default=None, help="阶段1 学习率（默认 5e-4）")
    parser.add_argument(
        "--finetune-lr", type=float, default=None, help="阶段2 学习率（默认 1e-6）"
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument("--seed", type=int, default=None)


def build_config(defaults: ClassifierTrainConfig, args: argparse.Namespace) -> ClassifierTrainConfig:
    two_phase = TwoPhaseSchedule(
        head_epochs=args.head_epochs
        if args.head_epochs is not None
        else defaults.two_phase.head_epochs,
        finetune_epochs=args.finetune_epochs
        if args.finetune_epochs is not None
        else defaults.two_phase.finetune_epochs,
        head_lr=args.head_lr if args.head_lr is not None else defaults.two_phase.head_lr,
        head_eta_min_ratio=defaults.two_phase.head_eta_min_ratio,
        finetune_lr=args.finetune_lr
        if args.finetune_lr is not None
        else defaults.two_phase.finetune_lr,
        finetune_eta_min_ratio=defaults.two_phase.finetune_eta_min_ratio,
    )
    return ClassifierTrainConfig(
        timm_model=defaults.timm_model,
        checkpoint_name=defaults.checkpoint_name,
        curves_name=defaults.curves_name,
        pretrained=defaults.pretrained,
        batch_size=args.batch_size if args.batch_size is not None else defaults.batch_size,
        weight_decay=defaults.weight_decay,
        mixup_alpha=defaults.mixup_alpha,
        optimizer=defaults.optimizer,
        seed=args.seed if args.seed is not None else defaults.seed,
        save_classes_in_checkpoint=defaults.save_classes_in_checkpoint,
        drop_rate=defaults.drop_rate,
        model_kwargs=dict(defaults.model_kwargs),
        num_classes=defaults.num_classes,
        num_workers=args.num_workers,
        output_dir=args.output_dir or defaults.output_dir or default_output_dir(),
        data_root=args.data_root or defaults.data_root,
        show_plots=args.show_plots,
        print_test_report=defaults.print_test_report,
        two_phase=two_phase,
    )


def main_entry(defaults: ClassifierTrainConfig) -> None:
    parser = argparse.ArgumentParser(description=f"Train {defaults.timm_model}")
    add_classifier_args(parser)
    args = parser.parse_args()
    train_classifier(build_config(defaults, args))
