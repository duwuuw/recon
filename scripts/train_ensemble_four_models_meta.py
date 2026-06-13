#!/usr/bin/env python3
"""Four-backbone ensemble + XGBoost meta learner (from ensemble_four_models_meta.ipynb)."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raicom.classifier import default_output_dir
from raicom.constants import NUM_CLASSES
from raicom.data import build_imagefolder_loaders
from raicom.ensemble import (
    collect_all_model_probs,
    collect_probabilities,
    hard_vote_accuracy,
    soft_vote_accuracy,
    train_single_backbone,
)
from raicom.paths import default_data_root
from raicom.two_phase import DEFAULT_TWO_PHASE, TwoPhaseSchedule

BACKBONE_SPECS = [
    ("mambaout_kobe", "mambaout_kobe.in1k"),
    ("convnextv2_nano", "convnextv2_nano"),
    ("resnet50", "resnet50"),
    ("mobilenetv3_large", "mobilenetv3_large_100"),
]


def parse_args():
    p = argparse.ArgumentParser(description="Train four-model ensemble with XGB meta learner")
    p.add_argument("--data-root", default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=28)
    p.add_argument("--head-epochs", type=int, default=DEFAULT_TWO_PHASE.head_epochs)
    p.add_argument("--finetune-epochs", type=int, default=DEFAULT_TWO_PHASE.finetune_epochs)
    p.add_argument("--head-lr", type=float, default=DEFAULT_TWO_PHASE.head_lr)
    p.add_argument("--finetune-lr", type=float, default=DEFAULT_TWO_PHASE.finetune_lr)
    p.add_argument("--early-stop", type=int, default=0, help="0 关闭早停（两阶段训练建议关闭）")
    p.add_argument("--cuda-device", type=int, default=0)
    p.add_argument(
        "--ensemble-f1-each-epoch",
        action="store_true",
        help="每 epoch 计算软投票 F1（很慢）",
    )
    p.add_argument("--show-plots", action="store_true")
    return p.parse_args()


def setup_device(cuda_device: int) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{cuda_device}")
        torch.cuda.set_device(cuda_device)
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass
        print("Device:", device, "|", torch.cuda.get_device_name(cuda_device))
        return device
    print("Device: cpu（未检测到 CUDA）")
    return torch.device("cpu")


def main():
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    args = parse_args()
    torch.manual_seed(2023)
    np.random.seed(2023)

    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = setup_device(args.cuda_device)
    num_workers = 0 if sys.platform == "win32" else 4
    data_root = args.data_root or default_data_root(require_existing=True)

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
    train_loader, val_loader, test_loader, dataset_num_classes, class_names = build_imagefolder_loaders(
        data_root, batch_size=args.batch_size, num_workers=num_workers
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

    trained_models = {}
    histories = {}
    ensemble_vote_val_macro_f1_log = []

    for short_name, timm_name in BACKBONE_SPECS:
        ckpt = output_dir / f"ensemble_ckpt_{short_name}.pth"
        print("\n" + "=" * 60)
        print(f"训练骨干: {short_name}  ({timm_name})")
        print("=" * 60)
        peer_list = list(trained_models.items())
        m, hist = train_single_backbone(
            timm_name,
            train_loader,
            val_loader,
            num_classes,
            device,
            two_phase=schedule,
            weight_decay=2.5e-4,
            save_path=ckpt,
            peer_models_ordered=peer_list,
            val_loader_ensemble=val_loader,
            ensemble_f1_log=ensemble_vote_val_macro_f1_log,
            run_tag=short_name,
            early_stopping_patience=args.early_stop,
            early_stopping_min_delta=1e-4,
            ensemble_val_f1_each_epoch=args.ensemble_f1_each_epoch,
        )
        trained_models[short_name] = m
        histories[short_name] = hist

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

    X_val, y_val, order = collect_all_model_probs(trained_models, val_loader, device, num_classes)
    print("Meta train features:", X_val.shape)

    meta_clf = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        min_child_weight=2,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="mlogloss",
    )
    meta_clf.fit(X_val, y_val)
    val_meta_pred = meta_clf.predict(X_val)
    print("Meta (train on val) re-fit accuracy on val:", accuracy_score(y_val, val_meta_pred))

    order = [s[0] for s in BACKBONE_SPECS]
    test_prob_list = []
    for short_name in order:
        p, y_test = collect_probabilities(trained_models[short_name], test_loader, device, num_classes)
        test_prob_list.append(p)

    X_test = np.concatenate(test_prob_list, axis=1)
    meta_test_pred = meta_clf.predict(X_test)

    acc_soft, f1_soft = soft_vote_accuracy(test_prob_list, y_test)
    acc_hard, f1_hard = hard_vote_accuracy(test_prob_list, y_test)
    acc_meta = accuracy_score(y_test, meta_test_pred)
    f1_meta = f1_score(y_test, meta_test_pred, average="macro")

    print("--- Test set ---")
    print(f"Soft vote   Acc={acc_soft:.4f}  Macro-F1={f1_soft:.4f}")
    print(f"Hard vote   Acc={acc_hard:.4f}  Macro-F1={f1_hard:.4f}")
    print(f"Meta (XGB)  Acc={acc_meta:.4f}  Macro-F1={f1_meta:.4f}")

    meta_path = output_dir / "ensemble_meta_xgboost.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(
            {"meta_clf": meta_clf, "backbone_order": order, "class_names": class_names},
            f,
        )
    print(f"已保存 meta 分类器到 {meta_path}")


if __name__ == "__main__":
    main()
