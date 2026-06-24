"""Unified timm model construction for raicom training scripts."""

from __future__ import annotations

import timm

# timm 使用 drop_rate（非 dropout）控制分类头等处 dropout
DEFAULT_DROP_RATE = 0.1


def create_timm_classifier(
    model_name: str,
    num_classes: int,
    *,
    pretrained: bool = True,
    drop_rate: float = DEFAULT_DROP_RATE,
    **model_kwargs,
):
    return timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
        **model_kwargs,
    )
