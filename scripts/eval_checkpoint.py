#!/usr/bin/env python3
"""Evaluate a saved checkpoint on val/test split and report accuracy + macro-F1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raicom.checkpoints import load_checkpoint
from raicom.classifier import default_output_dir
from raicom.constants import NUM_CLASSES
from raicom.data import build_imagefolder_loaders
from raicom.device import pick_device
from raicom.paths import default_data_root
from raicom.timm_factory import create_timm_classifier
from raicom.training import collect_predictions, print_classification_report, validate


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate checkpoint on val/test sets")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--split", choices=("val", "test", "both"), default="both")
    p.add_argument("--timm-model", default=None, help="Required for timm checkpoints")
    p.add_argument("--gdn", action="store_true", help="Load GDN architecture instead of timm")
    p.add_argument("--drop-rate", type=float, default=0.1)
    return p.parse_args()


def build_model(args: argparse.Namespace, num_classes: int, device: torch.device):
    if args.gdn:
        from raicom.models.gdn import GDN

        return GDN(num_classes, d_model=512, num_layers=6, dropout=0.1, mode="chunk").to(device)
    if not args.timm_model:
        raise SystemExit("请指定 --timm-model，或使用 --gdn")
    return create_timm_classifier(
        args.timm_model, num_classes, pretrained=False, drop_rate=args.drop_rate
    ).to(device)


def eval_split(model, loader, criterion, device, name: str):
    loss, acc, f1 = validate(model, loader, criterion, device, 0, name, compute_f1=True)
    y_true, y_pred = collect_predictions(model, loader, device)
    print(f"\n=== {name} ===")
    print(f"Loss: {loss:.4f} | Acc: {acc:.4f} | macro-F1: {f1:.4f}")
    return loss, acc, f1, y_true, y_pred


def main() -> None:
    args = parse_args()
    device = pick_device(force_cpu=args.cpu)
    data_root = args.data_root or default_data_root(require_existing=True)
    ckpt = args.checkpoint
    if not ckpt.is_file():
        raise SystemExit(f"权重不存在: {ckpt}")

    train_loader, val_loader, test_loader, num_classes, class_names = build_imagefolder_loaders(
        data_root, batch_size=args.batch_size, image_size=args.image_size
    )
    if num_classes != NUM_CLASSES:
        raise SystemExit(f"数据集 {num_classes} 类，需要 {NUM_CLASSES} 类: {class_names}")

    model = build_model(args, num_classes, device)
    meta = load_checkpoint(ckpt, model, device)
    eval_classes = (meta or {}).get("classes") or class_names
    criterion = nn.CrossEntropyLoss()

    print(f"checkpoint: {ckpt.resolve()}")
    print(f"device:     {device}")
    print(f"classes:    {eval_classes}")
    print(f"train/val/test sizes: {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}")

    results = {}
    if args.split in ("val", "both"):
        _, acc, f1, y_true, y_pred = eval_split(model, val_loader, criterion, device, "Val")
        print_classification_report("Val", y_true, y_pred, eval_classes)
        results["val"] = {"acc": acc, "macro_f1": f1}
    if args.split in ("test", "both"):
        _, acc, f1, y_true, y_pred = eval_split(model, test_loader, criterion, device, "Test")
        print_classification_report("Test", y_true, y_pred, eval_classes)
        results["test"] = {"acc": acc, "macro_f1": f1}

    if "test" in results:
        print(f"\n>> Test macro-F1 = {results['test']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
