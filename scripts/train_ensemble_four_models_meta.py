#!/usr/bin/env python3
"""Small-strong timm ensemble + Kaggle-style soft vote / meta learner."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raicom.checkpoints import load_checkpoint
from raicom.classifier import default_output_dir
from raicom.constants import NUM_CLASSES
from raicom.data import build_imagefolder_loaders
from raicom.ensemble import (
    collect_probabilities,
    hard_vote_accuracy,
    optimize_soft_vote_weights,
    probability_metrics,
    soft_vote_accuracy,
    train_single_backbone,
    weighted_probability_average,
    weighted_soft_vote_accuracy,
)
from raicom.paths import default_data_root
from raicom.timm_factory import create_timm_classifier
from raicom.timm_presets import TIMM_PRESETS
from raicom.two_phase import (
    DEFAULT_TWO_PHASE,
    DEFAULT_EARLY_STOPPING_MIN_DELTA,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    TwoPhaseSchedule,
)


@dataclass(frozen=True)
class BackboneSpec:
    short_name: str
    timm_name: str
    batch_size: int = 32
    image_size: int = 224
    weight_decay: float = 2.5e-4
    model_kwargs: dict[str, Any] = field(default_factory=dict)


ENSEMBLE_PRESETS: dict[str, list[str]] = {
    # Default: all pretrained, structurally diverse, and <= ~21M params with a 4-way head.
    "balanced": [
        "convnextv2_nano",
        "mobilenetv4_hybrid_medium",
        "fastvit_sa24",
        "mambaout_kobe",
        "tiny_vit_11m_224",
    ],
    # Faster sweep for small GPUs or quick leaderboard probing.
    "lite": [
        "convnextv2_pico",
        "mobilenetv4_conv_medium",
        "fastvit_s12",
        "repvit_m1_1",
        "tiny_vit_5m_224",
    ],
    # Still modest, but adds one stronger 24M-class backbone.
    "strong": [
        "convnextv2_nano",
        "mobilenetv4_hybrid_medium",
        "fastvit_sa24",
        "mambaout_kobe",
        "tiny_vit_11m_224",
        "caformer_s18",
    ],
    # Original notebook-style baseline, kept for comparison.
    "legacy": [
        "mambaout_kobe.in1k",
        "convnextv2_nano",
        "resnet50",
        "mobilenetv3_large_100",
    ],
}


def _safe_name(name: str) -> str:
    return (
        name.replace("hf-hub:", "")
        .replace("/", "_")
        .replace(".", "_")
        .replace("-", "_")
        .replace(":", "_")
    )


def _spec_from_name(name: str) -> BackboneSpec:
    preset = TIMM_PRESETS.get(name)
    if preset is not None:
        return BackboneSpec(
            short_name=preset.key,
            timm_name=preset.timm_model,
            batch_size=preset.batch_size,
            image_size=preset.image_size,
            weight_decay=preset.weight_decay,
            model_kwargs=dict(preset.model_kwargs),
        )
    return BackboneSpec(short_name=_safe_name(name), timm_name=name)


def resolve_backbone_specs(args: argparse.Namespace) -> list[BackboneSpec]:
    names = args.models or ENSEMBLE_PRESETS[args.ensemble]
    specs = [_spec_from_name(name) for name in names]
    seen: set[str] = set()
    duplicates = []
    for spec in specs:
        if spec.short_name in seen:
            duplicates.append(spec.short_name)
        seen.add(spec.short_name)
    if duplicates:
        raise ValueError(f"重复的 backbone short_name: {duplicates}")
    return specs


def parse_args():
    p = argparse.ArgumentParser(description="Train small-strong timm ensemble with soft vote and XGB")
    p.add_argument("--data-root", default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--ensemble",
        choices=sorted(ENSEMBLE_PRESETS),
        default="balanced",
        help="内置集成方案；默认 balanced 是小参数强模型组合",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="覆盖内置方案：可传 timm preset key 或完整 timm 模型名",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="默认使用所选模型 preset 中最小 batch_size",
    )
    p.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="默认从所选 preset 推断；混用不同尺寸 preset 时请显式指定",
    )
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
    p.add_argument("--cuda-device", type=int, default=0)
    p.add_argument(
        "--ensemble-f1-each-epoch",
        action="store_true",
        help="每 epoch 计算软投票 F1（很慢）",
    )
    p.add_argument(
        "--weight-search-trials",
        type=int,
        default=4096,
        help="验证集随机搜索 weighted soft vote 权重的次数",
    )
    p.add_argument(
        "--weight-metric",
        choices=("macro_f1", "accuracy"),
        default="macro_f1",
        help="weighted soft vote 权重搜索优化目标",
    )
    p.add_argument(
        "--tta-hflip",
        action="store_true",
        help="验证/测试收集概率时加入水平翻转 TTA",
    )
    p.add_argument(
        "--reuse-checkpoints",
        action="store_true",
        help="若 ensemble_ckpt_*.pth 已存在则直接加载，跳过对应骨干训练",
    )
    p.add_argument(
        "--no-pretrained",
        action="store_true",
        help="调试用：不加载 ImageNet 预训练权重，避免联网下载",
    )
    p.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    p.add_argument("--show-plots", action="store_true")
    return p.parse_args()


def setup_device(cuda_device: int, *, force_cpu: bool = False) -> torch.device:
    if not force_cpu and torch.cuda.is_available():
        device = torch.device(f"cuda:{cuda_device}")
        torch.cuda.set_device(cuda_device)
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass
        print("Device:", device, "|", torch.cuda.get_device_name(cuda_device))
        return device
    print("Device: cpu")
    return torch.device("cpu")


def _count_parameters(spec: BackboneSpec, num_classes: int) -> int:
    model = create_timm_classifier(
        spec.timm_name,
        num_classes,
        pretrained=False,
        **spec.model_kwargs,
    )
    total = sum(p.numel() for p in model.parameters())
    del model
    return total


def _load_existing_model(
    spec: BackboneSpec,
    ckpt: Path,
    num_classes: int,
    device: torch.device,
) -> torch.nn.Module:
    model = create_timm_classifier(
        spec.timm_name,
        num_classes,
        pretrained=False,
        **spec.model_kwargs,
    ).to(device)
    if load_checkpoint(ckpt, model, device) is None:
        print(f"  -> 已加载原始 state_dict checkpoint: {ckpt}")
    else:
        print(f"  -> 已加载 checkpoint payload: {ckpt}")
    model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


def _collect_prob_list(
    trained_models: dict[str, torch.nn.Module],
    order: list[str],
    dataloader,
    device: torch.device,
    num_classes: int,
    *,
    tta_hflip: bool,
) -> tuple[list[np.ndarray], np.ndarray]:
    prob_list: list[np.ndarray] = []
    labels: np.ndarray | None = None
    for short_name in order:
        p, y = collect_probabilities(
            trained_models[short_name],
            dataloader,
            device,
            num_classes,
            tta_hflip=tta_hflip,
        )
        prob_list.append(p)
        if labels is None:
            labels = y
        elif not np.array_equal(labels, y):
            raise RuntimeError(f"label order mismatch while collecting {short_name}")
    if labels is None:
        raise RuntimeError("no models were collected")
    return prob_list, labels


def _metrics(acc: float, f1: float) -> dict[str, float]:
    return {"accuracy": float(acc), "macro_f1": float(f1)}


def main():
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    args = parse_args()
    from xgboost import XGBClassifier

    torch.manual_seed(2023)
    np.random.seed(2023)

    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = setup_device(args.cuda_device, force_cpu=args.cpu)
    num_workers = 0 if sys.platform == "win32" else 4
    data_root = args.data_root or default_data_root(require_existing=True)
    specs = resolve_backbone_specs(args)
    ensemble_label = "custom" if args.models else args.ensemble
    order = [spec.short_name for spec in specs]
    inferred_sizes = {spec.image_size for spec in specs}
    if args.image_size is None:
        if len(inferred_sizes) != 1:
            raise ValueError(
                f"所选模型 preset 输入尺寸不一致: {sorted(inferred_sizes)}；"
                "请用 --image-size 显式指定统一尺寸"
            )
        image_size = next(iter(inferred_sizes))
    else:
        image_size = args.image_size
    batch_size = args.batch_size or min(spec.batch_size for spec in specs)

    schedule = TwoPhaseSchedule(
        head_epochs=args.head_epochs,
        finetune_epochs=args.finetune_epochs,
        head_lr=args.head_lr,
        finetune_lr=args.finetune_lr,
    )
    print(
        f"两阶段训练: 阶段1={schedule.head_epochs} epoch, 阶段2={schedule.finetune_epochs} epoch, "
        f"合计 {schedule.total_epochs}"
    )
    print(f"Ensemble preset: {ensemble_label}")
    print(f"Batch size: {batch_size} | image_size: {image_size} | TTA hflip: {args.tta_hflip}")
    train_loader, val_loader, test_loader, dataset_num_classes, class_names = build_imagefolder_loaders(
        data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
    )
    if dataset_num_classes != NUM_CLASSES:
        raise ValueError(
            f"数据集有 {dataset_num_classes} 类，需要 {NUM_CLASSES} 类: {class_names}"
        )
    num_classes = NUM_CLASSES
    print("Classes:", class_names)
    print("num_classes:", num_classes)
    print(
        "train/val/test:",
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
    )
    print("Backbones:")
    param_counts: dict[str, int] = {}
    for spec in specs:
        total_params = _count_parameters(spec, num_classes)
        param_counts[spec.short_name] = total_params
        print(
            f"  - {spec.short_name:28s} {total_params / 1e6:6.2f}M  "
            f"{spec.timm_name}"
        )

    trained_models = {}
    histories = {}
    ensemble_vote_val_macro_f1_log = []

    for spec in specs:
        ckpt = output_dir / f"ensemble_ckpt_{spec.short_name}.pth"
        print("\n" + "=" * 60)
        print(f"训练骨干: {spec.short_name}  ({spec.timm_name})")
        print("=" * 60)
        if args.reuse_checkpoints and ckpt.is_file():
            m = _load_existing_model(spec, ckpt, num_classes, device)
            hist = {"reused_checkpoint": True, "checkpoint": str(ckpt)}
        else:
            peer_list = list(trained_models.items())
            m, hist = train_single_backbone(
                spec.timm_name,
                train_loader,
                val_loader,
                num_classes,
                device,
                two_phase=schedule,
                weight_decay=spec.weight_decay,
                save_path=ckpt,
                peer_models_ordered=peer_list,
                val_loader_ensemble=val_loader,
                ensemble_f1_log=ensemble_vote_val_macro_f1_log,
                run_tag=spec.short_name,
                early_stopping_patience=args.early_stop,
                early_stopping_min_delta=args.early_stop_min_delta,
                ensemble_val_f1_each_epoch=args.ensemble_f1_each_epoch,
                pretrained=not args.no_pretrained,
                model_kwargs=spec.model_kwargs,
            )
        trained_models[spec.short_name] = m
        histories[spec.short_name] = hist

    if ensemble_vote_val_macro_f1_log:
        fig, ax = plt.subplots(figsize=(12, 4.8))
        y_vals = [r["macro_f1"] for r in ensemble_vote_val_macro_f1_log]
        x_vals = np.arange(1, len(y_vals) + 1)
        ax.plot(x_vals, y_vals, lw=1.3, color="tab:blue", label="软投票 macro-F1 (val)")
        prev = None
        for i, rec in enumerate(ensemble_vote_val_macro_f1_log):
            if i == 0:
                prev = rec["backbone"]
                continue
            if rec["backbone"] != prev:
                ax.axvline(i + 0.5, color="gray", ls="--", alpha=0.75)
                prev = rec["backbone"]
        ax.set_xlabel("累积 epoch")
        ax.set_ylabel("Macro-F1")
        ax.set_title("每轮末尾：软投票预测在验证集上的 macro-F1")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")
        plt.tight_layout()
        f1_plot = output_dir / "ensemble_soft_vote_val_macro_f1_per_epoch.png"
        plt.savefig(f1_plot, dpi=200)
        if args.show_plots:
            plt.show()
        else:
            plt.close()
        print(f"已保存 {f1_plot}")

    val_prob_list, y_val = _collect_prob_list(
        trained_models,
        order,
        val_loader,
        device,
        num_classes,
        tta_hflip=args.tta_hflip,
    )
    X_val = np.concatenate(val_prob_list, axis=1)
    print("Meta train features:", X_val.shape)

    acc_val_soft, f1_val_soft = soft_vote_accuracy(val_prob_list, y_val)
    acc_val_hard, f1_val_hard = hard_vote_accuracy(val_prob_list, y_val)
    best_weights, weighted_val_metrics = optimize_soft_vote_weights(
        val_prob_list,
        y_val,
        metric=args.weight_metric,
        n_trials=args.weight_search_trials,
        seed=42,
    )
    print("--- Validation set ---")
    print(f"Soft vote          Acc={acc_val_soft:.4f}  Macro-F1={f1_val_soft:.4f}")
    print(f"Hard vote          Acc={acc_val_hard:.4f}  Macro-F1={f1_val_hard:.4f}")
    print(
        "Weighted soft vote "
        f"Acc={weighted_val_metrics['accuracy']:.4f}  "
        f"Macro-F1={weighted_val_metrics['macro_f1']:.4f}  "
        f"LogLoss={weighted_val_metrics['log_loss']:.4f}"
    )
    print("Weighted soft-vote weights:")
    for short_name, weight in zip(order, best_weights):
        print(f"  {short_name:28s} {weight:.4f}")

    meta_clf = XGBClassifier(
        n_estimators=250,
        max_depth=2,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=3.0,
        min_child_weight=1,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="mlogloss",
    )
    meta_clf.fit(X_val, y_val)
    val_meta_pred = meta_clf.predict(X_val)
    val_meta_metrics = {
        "accuracy": float(accuracy_score(y_val, val_meta_pred)),
        "macro_f1": float(f1_score(y_val, val_meta_pred, average="macro")),
    }
    print(
        "Meta (XGB train on val) "
        f"Acc={val_meta_metrics['accuracy']:.4f}  Macro-F1={val_meta_metrics['macro_f1']:.4f}"
    )

    test_prob_list, y_test = _collect_prob_list(
        trained_models,
        order,
        test_loader,
        device,
        num_classes,
        tta_hflip=args.tta_hflip,
    )
    X_test = np.concatenate(test_prob_list, axis=1)
    meta_test_pred = meta_clf.predict(X_test)

    acc_soft, f1_soft = soft_vote_accuracy(test_prob_list, y_test)
    acc_hard, f1_hard = hard_vote_accuracy(test_prob_list, y_test)
    acc_weighted, f1_weighted = weighted_soft_vote_accuracy(test_prob_list, y_test, best_weights)
    weighted_test_metrics = probability_metrics(
        weighted_probability_average(test_prob_list, best_weights),
        y_test,
    )
    acc_meta = accuracy_score(y_test, meta_test_pred)
    f1_meta = f1_score(y_test, meta_test_pred, average="macro")

    print("--- Test set ---")
    print(f"Soft vote          Acc={acc_soft:.4f}  Macro-F1={f1_soft:.4f}")
    print(f"Hard vote          Acc={acc_hard:.4f}  Macro-F1={f1_hard:.4f}")
    print(f"Weighted soft vote Acc={acc_weighted:.4f}  Macro-F1={f1_weighted:.4f}")
    print(f"Meta (XGB)         Acc={acc_meta:.4f}  Macro-F1={f1_meta:.4f}")

    meta_path = output_dir / "ensemble_meta_xgboost.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(
            {
                "meta_clf": meta_clf,
                "backbone_order": order,
                "class_names": class_names,
                "weighted_soft_vote_weights": {
                    name: float(weight) for name, weight in zip(order, best_weights)
                },
                "backbone_specs": [spec.__dict__ for spec in specs],
                "tta_hflip": args.tta_hflip,
            },
            f,
        )
    print(f"已保存 meta 分类器到 {meta_path}")

    report = {
        "ensemble": ensemble_label,
        "requested_ensemble": args.ensemble,
        "class_names": list(class_names),
        "image_size": image_size,
        "batch_size": batch_size,
        "tta_hflip": args.tta_hflip,
        "pretrained": not args.no_pretrained,
        "device": str(device),
        "weight_search_trials": args.weight_search_trials,
        "weight_metric": args.weight_metric,
        "backbones": [
            {
                "short_name": spec.short_name,
                "timm_name": spec.timm_name,
                "params": param_counts[spec.short_name],
                "params_m": param_counts[spec.short_name] / 1e6,
                "weight_decay": spec.weight_decay,
                "model_kwargs": spec.model_kwargs,
            }
            for spec in specs
        ],
        "weighted_soft_vote_weights": {
            name: float(weight) for name, weight in zip(order, best_weights)
        },
        "validation": {
            "soft_vote": _metrics(acc_val_soft, f1_val_soft),
            "hard_vote": _metrics(acc_val_hard, f1_val_hard),
            "weighted_soft_vote": weighted_val_metrics,
            "meta_xgboost": val_meta_metrics,
        },
        "test": {
            "soft_vote": _metrics(acc_soft, f1_soft),
            "hard_vote": _metrics(acc_hard, f1_hard),
            "weighted_soft_vote": weighted_test_metrics,
            "meta_xgboost": _metrics(acc_meta, f1_meta),
        },
        "histories": {
            name: {
                "best_val_acc": max(hist.get("val_acc", [0.0])) if hist.get("val_acc") else None,
                "epochs_trained": hist.get("epochs_trained"),
                "early_stopped": hist.get("early_stopped"),
                "reused_checkpoint": hist.get("reused_checkpoint", False),
            }
            for name, hist in histories.items()
        },
    }
    report_path = output_dir / "ensemble_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"已保存集成报告到 {report_path}")


if __name__ == "__main__":
    main()
