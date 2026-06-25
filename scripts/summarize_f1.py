#!/usr/bin/env python3
"""Evaluate all known single-model checkpoints and print a macro-F1 summary table."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
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
from raicom.training import validate


@dataclass(frozen=True)
class ModelSpec:
    name: str
    checkpoint: str
    timm_model: str | None = None
    gdn: bool = False


SPECS: list[ModelSpec] = [
    ModelSpec("convnext11", "convnext.pth", "convnextv2_nano"),
    ModelSpec("dinov2_small", "dinov2_small.pth", "vit_small_patch14_dinov2"),
    ModelSpec("efficientnet", "efficientnet.pth", "tf_efficientnetv2_s"),
    ModelSpec("fasternet", "fasternet_t2.pth", "fasternet_t2"),
    ModelSpec("fastvit_s24", "fastvit_s24.pth", "fastvit_sa24.apple_dist_in1k"),
    ModelSpec("mambaout_kobe", "mambaout_kobe.pth", "mambaout_kobe"),
    ModelSpec("mambaout_small_rw", "mambaout_small_rw.pth", "mambaout_small_rw"),
    ModelSpec("mobilenetv4", "mobilenetv4_hybrid_medium.pth", "mobilenetv4_hybrid_medium"),
    ModelSpec("repvit", "repvit_m1_5.pth", "repvit_m1_5"),
    ModelSpec("repvit_m2", "repvit_m2_3.pth", "repvit_m2_3"),
    ModelSpec("resnet18", "resnet50.pth", "ecaresnet50d"),
    ModelSpec("vit11", "vit.pth", "vit_base_patch16_rope_224"),
    ModelSpec("gdn", "gdn_best.pth", gdn=True),
]


def parse_args():
    p = argparse.ArgumentParser(description="Summarize test macro-F1 for all checkpoints")
    p.add_argument("--data-root", default=None)
    p.add_argument("--checkpoint-dir", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--output", type=Path, default=Path("checkpoints/f1_summary.json"))
    return p.parse_args()


def load_model(spec: ModelSpec, num_classes: int, device: torch.device):
    if spec.gdn:
        from raicom.models.gdn import GDN

        return GDN(num_classes, d_model=512, num_layers=6, dropout=0.1, mode="chunk").to(device)
    return create_timm_classifier(
        spec.timm_model, num_classes, pretrained=False, drop_rate=0.1
    ).to(device)


def main() -> None:
    args = parse_args()
    device = pick_device(force_cpu=args.cpu)
    ckpt_dir = args.checkpoint_dir or default_output_dir()
    data_root = args.data_root or default_data_root(require_existing=True)

    _, _, test_loader, num_classes, class_names = build_imagefolder_loaders(
        data_root, batch_size=args.batch_size
    )
    if num_classes != NUM_CLASSES:
        raise SystemExit(f"需要 {NUM_CLASSES} 类，实际 {num_classes}: {class_names}")

    criterion = nn.CrossEntropyLoss()
    rows = []

    for spec in SPECS:
        path = ckpt_dir / spec.checkpoint
        row = {"model": spec.name, "checkpoint": str(path), "status": "missing"}
        if not path.is_file():
            rows.append(row)
            continue
        try:
            model = load_model(spec, num_classes, device)
            load_checkpoint(path, model, device)
            _, acc, f1 = validate(model, test_loader, criterion, device, 0, "Test", compute_f1=True)
            row.update(
                {
                    "status": "ok",
                    "test_acc": float(acc),
                    "test_macro_f1": float(f1 or 0.0),
                }
            )
        except Exception as exc:
            row.update({"status": "error", "error": str(exc)})
        rows.append(row)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ok = [r for r in rows if r.get("status") == "ok"]
    ok.sort(key=lambda r: r["test_macro_f1"], reverse=True)

    print(f"\n{'Model':<22} {'Test Acc':>10} {'Test F1':>10}  Checkpoint")
    print("-" * 72)
    for r in rows:
        if r["status"] == "ok":
            print(
                f"{r['model']:<22} {r['test_acc']:>10.4f} {r['test_macro_f1']:>10.4f}  "
                f"{Path(r['checkpoint']).name}"
            )
        elif r["status"] == "missing":
            print(f"{r['model']:<22} {'—':>10} {'—':>10}  (未训练: {Path(r['checkpoint']).name})")
        else:
            print(f"{r['model']:<22} {'ERR':>10} {'ERR':>10}  {r.get('error', '')[:30]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已保存: {args.output.resolve()}")


if __name__ == "__main__":
    main()
