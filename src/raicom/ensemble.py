"""Multi-backbone ensemble training and meta-learner (from ensemble_four_models_meta.ipynb)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from raicom.checkpoints import load_checkpoint
from raicom.timm_factory import create_timm_classifier
from raicom.training import train_one_epoch, validate
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
    freeze_timm_backbone,
    unfreeze_all,
)


def train_single_backbone(
    model_name: str,
    train_loader,
    val_loader,
    num_classes: int,
    device: torch.device,
    *,
    two_phase: TwoPhaseSchedule | None = None,
    weight_decay: float = 2.5e-4,
    save_path: Path | str | None = None,
    peer_models_ordered=None,
    val_loader_ensemble=None,
    ensemble_f1_log=None,
    run_tag: str = "",
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA,
    ensemble_val_f1_each_epoch: bool = False,
):
    schedule = two_phase or DEFAULT_TWO_PHASE
    peer_models_ordered = peer_models_ordered or []
    model = create_timm_classifier(model_name, num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()

    print(f"[{model_name}] {describe_phase(schedule, 1)}")
    print(f"[{model_name}] {describe_phase(schedule, 2)}")

    freeze_timm_backbone(model)
    print(f"[{model_name}] 阶段1 可训练参数: {count_trainable_parameters(model):,}")

    optimizer = build_optimizer(
        model, lr=schedule.head_lr, weight_decay=weight_decay, optimizer_name="adamw"
    )
    scheduler = build_cosine_scheduler(
        optimizer, t_max=schedule.head_epochs, eta_min=schedule.head_eta_min
    )

    best_acc = 0.0
    stopped_early = False
    early_stopper: EarlyStopper | None = None
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    total_epochs = schedule.total_epochs

    def _epoch_step(epoch: int, phase_tag: str, *, allow_early_stop: bool) -> bool:
        nonlocal best_acc, stopped_early
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, mixup_alpha=0.205
        )
        va_loss, va_acc, _ = validate(
            model, val_loader, criterion, device, epoch, "Val", compute_f1=False
        )
        scheduler.step()
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        print(
            f"[{model_name}] {phase_tag} Epoch {epoch:03d}/{total_epochs} | "
            f"Train {tr_loss:.4f}/{tr_acc:.4f} | Val {va_loss:.4f}/{va_acc:.4f} | "
            f"lr {scheduler.get_last_lr()[0]:.2e}"
        )

        if (
            val_loader_ensemble is not None
            and ensemble_f1_log is not None
            and ensemble_val_f1_each_epoch
        ):
            print(
                f"  [ensemble F1] epoch {epoch}: 正在验证集上收集多模型概率（可能较慢）...",
                flush=True,
            )
            prob_list = []
            y_true = None
            for _, pmodel in peer_models_ordered:
                pr, yt = collect_probabilities(pmodel, val_loader_ensemble, device, num_classes)
                prob_list.append(pr)
                y_true = yt
            pr, yt = collect_probabilities(model, val_loader_ensemble, device, num_classes)
            prob_list.append(pr)
            if y_true is None:
                y_true = yt
            elif not np.array_equal(y_true, yt):
                raise RuntimeError("ensemble eval: label order mismatch")
            mean_p = np.mean(np.stack(prob_list, axis=0), axis=0)
            vote_pred = mean_p.argmax(axis=1)
            f1_macro = f1_score(y_true, vote_pred, average="macro")
            ensemble_f1_log.append(
                {
                    "backbone": run_tag,
                    "epoch_in_backbone": epoch,
                    "macro_f1": float(f1_macro),
                    "n_vote_models": len(prob_list),
                }
            )
            model.to(device)

        if va_acc > best_acc + early_stopping_min_delta:
            best_acc = va_acc
            if save_path:
                path = Path(save_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), path)
                print(f"  -> 保存最佳 {path} (val_acc={va_acc:.4f})")
        if allow_early_stop and early_stopper is not None and early_stopper.step(va_acc, epoch):
            print(
                f"  -> 早停：阶段2 val_acc 连续 {early_stopper.patience} 轮未超过历史最佳 "
                f"(best={early_stopper.best_metric:.4f})"
            )
            stopped_early = True
            return True
        return False

    for epoch in range(1, schedule.head_epochs + 1):
        if _epoch_step(epoch, "[阶段1]", allow_early_stop=False):
            break

    if not stopped_early and schedule.finetune_epochs > 0:
        print(f"\n[{model_name}] 切换至阶段2：解冻骨干网络\n")
        unfreeze_all(model)
        print(f"[{model_name}] 阶段2 可训练参数: {count_trainable_parameters(model):,}")
        early_stopper = EarlyStopper(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            best_metric=best_acc,
        )
        print(f"[{model_name}] {early_stopper.describe()}")
        optimizer = build_optimizer(
            model, lr=schedule.finetune_lr, weight_decay=weight_decay, optimizer_name="adamw"
        )
        scheduler = build_cosine_scheduler(
            optimizer,
            t_max=schedule.finetune_epochs,
            eta_min=schedule.finetune_eta_min,
        )
        for offset, epoch in enumerate(
            range(schedule.head_epochs + 1, total_epochs + 1), start=1
        ):
            if _epoch_step(
                epoch,
                f"[阶段2 {offset}/{schedule.finetune_epochs}]",
                allow_early_stop=True,
            ):
                break

    history["early_stopped"] = stopped_early
    history["epochs_trained"] = len(history["val_acc"])
    if stopped_early:
        print(f"  [{model_name}] 实际训练 {history['epochs_trained']}/{total_epochs} epoch")

    if save_path and os.path.isfile(save_path):
        load_checkpoint(save_path, model, device)
    model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model, history


@torch.no_grad()
def collect_probabilities(model, dataloader, device, num_classes):
    model = model.to(device)
    model.eval()
    probs_list, y_list = [], []
    for x, y in tqdm(dataloader, desc="Collect probs", leave=False):
        x = x.to(device)
        logits = model(x)
        p = torch.softmax(logits, dim=1).cpu().numpy()
        probs_list.append(p)
        y_list.append(y.numpy())
    probs = np.concatenate(probs_list, axis=0)
    labels = np.concatenate(y_list, axis=0)
    model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return probs, labels


@torch.no_grad()
def collect_all_model_probs(models_dict, dataloader, device, num_classes):
    order = list(models_dict.keys())
    parts = []
    labels = None
    for name in order:
        p, y = collect_probabilities(models_dict[name], dataloader, device, num_classes)
        parts.append(p)
        if labels is None:
            labels = y
        else:
            assert np.array_equal(labels, y)
    X = np.concatenate(parts, axis=1)
    return X, labels, order


def soft_vote_accuracy(prob_list, y_true):
    mean_p = np.mean(np.stack(prob_list, axis=0), axis=0)
    pred = mean_p.argmax(axis=1)
    return accuracy_score(y_true, pred), f1_score(y_true, pred, average="macro")


def hard_vote_accuracy(prob_list, y_true):
    preds = np.stack([p.argmax(axis=1) for p in prob_list], axis=1)
    num_c = prob_list[0].shape[1]
    final = np.array(
        [np.bincount(preds[i], minlength=num_c).argmax() for i in range(preds.shape[0])]
    )
    return accuracy_score(y_true, final), f1_score(y_true, final, average="macro")
