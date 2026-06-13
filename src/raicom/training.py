"""Training / validation loops and curve plotting."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

from raicom.mixup import mixup_criterion, mixup_data


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    epoch: int,
    *,
    mixup_alpha: float | None = 0.205,
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        if mixup_alpha is not None and mixup_alpha > 0:
            images, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=mixup_alpha)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            _, preds = torch.max(outputs, 1)
            correct_a = (preds == labels_a).sum().item()
            correct_b = (preds == labels_b).sum().item()
            batch_correct = lam * correct_a + (1.0 - lam) * correct_b
            correct += batch_correct
            total += images.size(0)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
        loss.backward()
        optimizer.step()
        batch_loss = loss.item()
        running_loss += batch_loss * inputs.size(0)
        pbar.set_postfix(Loss=f"{batch_loss:.4f}", Acc=f"{correct / max(total, 1):.4f}")
    return running_loss / total, correct / total


@torch.no_grad()
def validate(
    model,
    dataloader,
    criterion,
    device,
    epoch: int,
    phase: str = "Val",
    *,
    compute_f1: bool = False,
):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_pred, all_labels = [], []
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [{phase}]", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        if compute_f1:
            all_pred.append(preds.cpu())
            all_labels.append(labels.cpu())
        pbar.set_postfix(Loss=f"{loss.item():.4f}", Acc=f"{correct / total:.4f}")
    macro_f1 = None
    if compute_f1 and all_pred:
        y_p = torch.cat(all_pred).numpy()
        y_t = torch.cat(all_labels).numpy()
        macro_f1 = float(f1_score(y_t, y_p, average="macro"))
    return running_loss / total, correct / total, macro_f1


@torch.no_grad()
def collect_predictions(model, dataloader, device):
    model.eval()
    all_pred, all_labels = [], []
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        all_pred.append(torch.argmax(outputs, dim=1).cpu())
        all_labels.append(labels)
    return torch.cat(all_pred).numpy(), torch.cat(all_labels).numpy()


def print_classification_report(name, y_true, y_pred, class_names):
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix

    print(f"\n========== {name} 分类报告 ==========")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    cm = confusion_matrix(y_true, y_pred)
    print("混淆矩阵（行=真实标签，列=预测标签）:")
    print(pd.DataFrame(cm, index=class_names, columns=class_names))


def plot_training_curves(
    train_losses,
    val_losses,
    train_accs,
    val_accs,
    *,
    val_f1s: list[float] | None = None,
    save_path: str | Path = "training_curves.png",
    show: bool = False,
):
    save_path = str(save_path)
    n_plots = 3 if val_f1s else 2
    fig_w = 12 if n_plots == 3 else 10
    plt.figure(figsize=(fig_w, 5))
    epochs = range(1, len(train_losses) + 1)

    plt.subplot(1, n_plots, 1)
    plt.plot(epochs, train_losses, "b-", label="Training Loss")
    plt.plot(epochs, val_losses, "r-", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, n_plots, 2)
    plt.plot(epochs, train_accs, "b-", label="Training Accuracy")
    plt.plot(epochs, val_accs, "r-", label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)

    if val_f1s:
        plt.subplot(1, n_plots, 3)
        plt.plot(range(1, len(val_f1s) + 1), val_f1s, "b-", label="Val macro F1")
        plt.xlabel("Epoch")
        plt.ylabel("F1")
        plt.title("Validation Macro F1")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()
    print(f"训练曲线已保存至 {save_path}")
