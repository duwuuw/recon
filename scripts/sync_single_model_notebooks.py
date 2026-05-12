"""Regenerate single-backbone training notebooks under notebooks/ (run from repo root)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NB_DIR = REPO / "notebooks"

CELL0 = r"""import random
import sys
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from torchvision import transforms, datasets
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def set_seed(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def ensure_repo_src():
    for base in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        p = base / "src"
        if (p / "raicom").is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
            return


ensure_repo_src()
set_seed(2023)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
"""

CELL1 = "f1 = []\n"

CELL3 = r"""def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, mixup_alpha=0.205, grad_clip=1.0):
    model.train()
    running_loss = 0.0
    correct = 0.0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        images, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=mixup_alpha)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_loss = loss.item()
        running_loss += batch_loss * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += lam * (preds == labels_a).sum().item() + (1 - lam) * (preds == labels_b).sum().item()
        total += images.size(0)

        pbar.set_postfix({
            "Loss": f"{batch_loss:.4f}",
            "Acc": f"{correct / total:.4f}",
        })

    return running_loss / total, correct / total


def validate(model, dataloader, criterion, device, epoch, phase="Val", track_f1=True):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_pred, all_labels = [], []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [{phase}]", leave=False)
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)

            running_loss += loss.item() * inputs.size(0)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            all_pred.append(preds.cpu())
            all_labels.append(labels.cpu())

            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Acc": f"{correct / total:.4f}",
            })

    all_pred = torch.cat(all_pred).numpy()
    all_labels = torch.cat(all_labels).numpy()
    macro_f1 = f1_score(all_labels, all_pred, average="macro")
    if track_f1:
        f1.append(macro_f1)
    return running_loss / total, correct / total, macro_f1


def stratified_split(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    targets = np.array(dataset.targets)
    classes = np.unique(targets)
    train_idx, val_idx, test_idx = [], [], []
    rng = np.random.default_rng(random_seed)

    for cls in classes:
        cls_indices = np.where(targets == cls)[0]
        rng.shuffle(cls_indices)
        n_cls = len(cls_indices)
        n_train = int(round(train_ratio * n_cls))
        n_val = int(round(val_ratio * n_cls))
        n_test = n_cls - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_train = n_cls - n_val
        train_idx.extend(cls_indices[:n_train])
        val_idx.extend(cls_indices[n_train : n_train + n_val])
        test_idx.extend(cls_indices[n_train + n_val :])
    return train_idx, val_idx, test_idx


class SubsetWithTransform(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        if self.transform:
            x = self.transform(x)
        if not isinstance(x, torch.Tensor):
            from torchvision.transforms import functional as TF

            x = TF.to_tensor(x)
        return x, y


def compute_class_weights(dataset, indices, num_classes):
    targets = np.array(dataset.targets)[np.array(indices)]
    counts = np.bincount(targets, minlength=num_classes)
    weights = counts.sum() / np.maximum(counts, 1)
    weights = weights / weights.mean()
    print("Train class counts:", counts.tolist())
    print("CrossEntropy class weights:", np.round(weights, 3).tolist())
    return torch.tensor(weights, dtype=torch.float32)


def plot_curves(train_losses, val_losses, train_accs, val_accs, save_path="training_curves.png"):
    epochs = range(1, len(train_losses) + 1)
    epochs_for_f1 = range(1, len(f1) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, "b-", label="Training Loss")
    plt.plot(epochs, val_losses, "r-", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_accs, "b-", label="Training Accuracy")
    plt.plot(epochs, val_accs, "r-", label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(epochs_for_f1, f1, "b-", label="Val Macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Training curves saved to {save_path}")


def find_repo_root():
    for base in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if (base / "src" / "raicom").is_dir() or (base / ".git").exists():
            return base
    return Path.cwd().resolve()


def main(model=None):
    f1.clear()
    repo_root = find_repo_root()
    src_path = repo_root / "src"
    if src_path.is_dir() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from raicom.paths import default_data_root

    data_root = default_data_root()
    batch_size = 32
    epochs = 100
    head_lr = 3.5e-4
    backbone_lr = 3.5e-5
    weight_decay = 2e-4

    output_dir = repo_root / "outputs"
    checkpoint_dir = output_dir / "checkpoints"
    figure_dir = output_dir / "figures"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{EXPERIMENT_STEM}.pth"
    curve_path = figure_dir / f"training_curves_{EXPERIMENT_STEM}.png"

    print(f"Device: {device}")
    print(f"Backbone: {BACKBONE_NAME}")
    print(f"Data root: {data_root}")

    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                224, scale=(0.65, 1.0), ratio=(0.75, 1.33), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.18), ratio=(0.3, 3.3), value="random"),
        ]
    )
    transform_val = transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    transform_test = transform_val

    full_dataset = datasets.ImageFolder(root=data_root, transform=None)
    num_classes = len(full_dataset.classes)
    print(f"Found {num_classes} classes: {full_dataset.classes}")
    print(f"Total samples: {len(full_dataset)}")

    train_idx, val_idx, test_idx = stratified_split(
        full_dataset,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_seed=42,
    )
    print(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}, Test size: {len(test_idx)}")

    train_dataset = SubsetWithTransform(full_dataset, train_idx, transform=transform_train)
    val_dataset = SubsetWithTransform(full_dataset, val_idx, transform=transform_val)
    test_dataset = SubsetWithTransform(full_dataset, test_idx, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    if model is None:
        model = build_model(num_classes=num_classes).to(device)

    class_weights = compute_class_weights(full_dataset, train_idx, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(
        make_param_groups(model, backbone_lr=backbone_lr, head_lr=head_lr, weight_decay=weight_decay)
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs, eta_min=8e-7)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_f1 = 0.0
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device, epoch, "Val", track_f1=True)
        scheduler.step()

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(
            f"\nEpoch {epoch:03d}/{epochs} | "
            f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f} | Val Acc {val_acc:.4f} | Val Macro-F1 {val_f1:.4f}\n"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": full_dataset.classes,
                    "best_acc": best_acc,
                    "best_f1": best_f1,
                    "epoch": epoch,
                    "backbone": BACKBONE_NAME,
                },
                checkpoint_path,
            )
            print(f"  -> Saved best model, Val Acc {val_acc:.4f}, Val Macro-F1 {val_f1:.4f}\n")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(
        checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    )
    test_loss, test_acc, test_f1 = validate(model, test_loader, criterion, device, epoch=0, phase="Test", track_f1=False)
    print(f"\nFinal test Acc: {test_acc:.4f}, Macro-F1: {test_f1:.4f}")

    plot_curves(train_losses, val_losses, train_accs, val_accs, save_path=curve_path)
    return model
"""

CELL4 = "main()\n"


def _src(text: str) -> list[str]:
    if not text.endswith("\n"):
        text += "\n"
    return [text]


def _code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": str(uuid.uuid4())[:8],
        "metadata": {},
        "outputs": [],
        "source": _src(source),
    }


def cell2_for(backbone: str, stem: str) -> str:
    return f"""from raicom.timm_feature_classifier import (
    TimmFeatureClassifier,
    make_param_groups,
    mixup_data,
    mixup_criterion,
)

BACKBONE_NAME = "{backbone}"
EXPERIMENT_STEM = "{stem}"


def build_model(num_classes=11, pretrained=True):
    return TimmFeatureClassifier(
        BACKBONE_NAME,
        num_classes=num_classes,
        pretrained=pretrained,
    ).to(device)
"""


CELL2_TIMM_FEATURE_INLINE = r'''class __CLASS_NAME__(nn.Module):
    """__DOCSTRING__"""

    def __init__(
        self,
        model_name="__BACKBONE__",
        num_classes=11,
        pretrained=True,
        hidden_dim=512,
        dropout=0.35,
        gem_p=3.0,
    ):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.num_features = self.backbone.num_features
        gate_hidden = max(self.num_features // 4, 32)

        self.feature_norm = nn.LayerNorm(self.num_features)
        self.gem_p = nn.Parameter(torch.ones(1) * gem_p)
        self.channel_gate = nn.Sequential(
            nn.Linear(self.num_features * 2, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, self.num_features),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.num_features * 3),
            nn.Dropout(dropout),
            nn.Linear(self.num_features * 3, hidden_dim),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _as_nhwc(self, x):
        if x.ndim == 4 and x.shape[-1] == self.num_features:
            return x
        if x.ndim == 4 and x.shape[1] == self.num_features:
            return x.permute(0, 2, 3, 1).contiguous()
        if x.ndim == 3:
            return x
        if x.ndim == 2:
            return x.unsqueeze(1)
        raise ValueError(f"Unexpected feature shape from backbone: {tuple(x.shape)}")

    def _pool_features(self, features):
        features = self._as_nhwc(features)
        features = self.feature_norm(features)
        reduce_dims = (1, 2) if features.ndim == 4 else 1

        avg_pool = features.mean(dim=reduce_dims)
        max_pool = features.amax(dim=reduce_dims)
        p = self.gem_p.clamp(min=1.0, max=6.0)
        gem_source = torch.nn.functional.softplus(features)
        gem_pool = gem_source.pow(p).mean(dim=reduce_dims).pow(1.0 / p)

        gate = self.channel_gate(torch.cat([avg_pool, max_pool], dim=1))
        return torch.cat([avg_pool * gate, max_pool * gate, gem_pool * gate], dim=1)

    def forward_features(self, x):
        return self._pool_features(self.backbone.forward_features(x))

    def forward(self, x):
        return self.head(self.forward_features(x))


BACKBONE_NAME = "__BACKBONE__"
EXPERIMENT_STEM = "__STEM__"


def build_model(num_classes=11, pretrained=True):
    return __CLASS_NAME__(
        model_name=BACKBONE_NAME,
        num_classes=num_classes,
        pretrained=pretrained,
    ).to(device)


def make_param_groups(model, backbone_lr=3.5e-5, head_lr=3.5e-4, weight_decay=2e-4):
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.") and p.requires_grad]
    return [
        {"params": model.backbone.parameters(), "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
    ]


def mixup_data(x, y, alpha=0.2):
    """Return mixed images, two target tensors, and mix coefficient."""
    if alpha <= 0 or x.size(0) < 2:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute MixUp loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
'''


# Notebooks that use an inlined timm backbone + gated multi-pool neck + MLP head (same pattern as mambaout_kobe).
INLINE_TIMM_FEATURE_NOTEBOOKS: dict[str, tuple[str, str]] = {
    "convnext11.ipynb": (
        "ConvNeXtFeatureClassifier",
        "Use ConvNeXt as a pure backbone, then train a stronger task head.",
    ),
    "efficientnet.ipynb": (
        "EfficientNetFeatureClassifier",
        "Use EfficientNet (timm) as a pure backbone, then train a stronger task head.",
    ),
    "resnet18.ipynb": (
        "ResNet18FeatureClassifier",
        "Use ResNet-18 as a pure backbone, then train a stronger task head.",
    ),
    "mobilenetv3_large_100.ipynb": (
        "MobileNetV3FeatureClassifier",
        "Use MobileNetV3-Large as a pure backbone, then train a stronger task head.",
    ),
    "mambaout_small_rw.ipynb": (
        "MambaOutSmallRwFeatureClassifier",
        "Use mambaout_small_rw as a pure backbone, then train a stronger task head.",
    ),
}


def cell2_timm_feature_inline(class_name: str, docstring: str, backbone: str, stem: str) -> str:
    return (
        CELL2_TIMM_FEATURE_INLINE.replace("__CLASS_NAME__", class_name)
        .replace("__DOCSTRING__", docstring)
        .replace("__BACKBONE__", backbone)
        .replace("__STEM__", stem)
    )


def cell2_source(name: str, backbone: str, stem: str) -> str:
    meta = INLINE_TIMM_FEATURE_NOTEBOOKS.get(name)
    if meta is not None:
        return cell2_timm_feature_inline(meta[0], meta[1], backbone, stem)
    return cell2_for(backbone, stem)


NOTEBOOKS: list[tuple[str, str, str]] = [
    ("convnext11.ipynb", "convnext_tiny", "convnext_tiny_backbone_head"),
    ("resnet18.ipynb", "resnet18", "resnet18_backbone_head"),
    ("mobilenetv3_large_100.ipynb", "mobilenetv3_large_100", "mobilenetv3_large_100_backbone_head"),
    ("vit11.ipynb", "vit_base_patch16_rope_224", "vit_base_patch16_rope_224_backbone_head"),
    ("efficientnet.ipynb", "tf_efficientnetv2_s", "tf_efficientnetv2_s_backbone_head"),
    ("mambaout_small_rw.ipynb", "mambaout_small_rw", "mambaout_small_rw_backbone_head"),
    ("mambaout_kobe.ipynb", "mambaout_kobe", "mambaout_kobe_backbone_head"),
]


def write_notebook(name: str, backbone: str, stem: str) -> None:
    nb = {
        "cells": [
            _code_cell(CELL0),
            _code_cell(CELL1),
            _code_cell(cell2_source(name, backbone, stem)),
            _code_cell(CELL3),
            _code_cell(CELL4),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = NB_DIR / name
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", path)


def main_sync():
    for name, backbone, stem in NOTEBOOKS:
        write_notebook(name, backbone, stem)


if __name__ == "__main__":
    main_sync()
