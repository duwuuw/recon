"""timm backbone + pooling neck + MLP classifier head (shared by single-model notebooks)."""

from __future__ import annotations

import numpy as np
import timm
import torch
import torch.nn as nn


class TimmFeatureClassifier(nn.Module):
    """timm model as feature backbone (num_classes=0), then gated multi-pool neck + head."""

    def __init__(
        self,
        model_name: str,
        num_classes: int = 11,
        *,
        pretrained: bool = True,
        hidden_dim: int = 512,
        dropout: float = 0.35,
        gem_p: float = 3.0,
    ):
        super().__init__()
        self.model_name = model_name
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

    def _as_nhwc(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4 and x.shape[-1] == self.num_features:
            return x
        if x.ndim == 4 and x.shape[1] == self.num_features:
            return x.permute(0, 2, 3, 1).contiguous()
        if x.ndim == 3:
            return x
        if x.ndim == 2:
            return x.unsqueeze(1)
        raise ValueError(f"Unexpected feature shape from backbone: {tuple(x.shape)}")

    def _pool_features(self, features: torch.Tensor) -> torch.Tensor:
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

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self._pool_features(self.backbone.forward_features(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def make_param_groups(
    model: TimmFeatureClassifier,
    *,
    backbone_lr: float = 3.5e-5,
    head_lr: float = 3.5e-4,
    weight_decay: float = 2e-4,
):
    head_params = [
        p for n, p in model.named_parameters() if not n.startswith("backbone.") and p.requires_grad
    ]
    return [
        {"params": model.backbone.parameters(), "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
    ]


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    if alpha <= 0 or x.size(0) < 2:
        return x, y, y, 1.0

    lam = float(np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
