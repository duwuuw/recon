"""Gated Delta Network image classifier (from gdn.ipynb)."""

from __future__ import annotations

import torch
import torch.nn as nn

from fla.layers import GatedDeltaNet


def _gated_delta_head_shapes(hidden_size: int, num_heads: int = 6):
    key_dim = (hidden_size * 3) // 4
    if key_dim * 4 != hidden_size * 3:
        raise ValueError("hidden_size（即 d_model）必须被 4 整除")
    if key_dim % num_heads != 0:
        raise ValueError(f"请调整 num_heads 使其整除 key_dim={key_dim}")
    head_dim = key_dim // num_heads
    return num_heads, head_dim


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, d_model=512):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans, d_model, kernel_size=patch_size, stride=patch_size, bias=False
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x + self.pos_embed


class GDN(nn.Module):
    """Patch 嵌入 + 多层 GatedDeltaNet + 均值池化 + 分类头。"""

    def __init__(
        self,
        num_classes,
        img_size=224,
        patch_size=16,
        in_chans=3,
        d_model=512,
        num_layers=6,
        dropout=0.1,
        num_heads=6,
        mode="chunk",
        use_short_conv=True,
    ):
        super().__init__()
        num_heads, head_dim = _gated_delta_head_shapes(d_model, num_heads=num_heads)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, d_model)
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                GatedDeltaNet(
                    hidden_size=d_model,
                    head_dim=head_dim,
                    num_heads=num_heads,
                    mode=mode,
                    use_short_conv=use_short_conv,
                    layer_idx=i,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.pos_drop(self.patch_embed(x))
        for blk in self.blocks:
            x = blk(x)[0]
        x = self.norm(x).mean(dim=1)
        return self.head(x)
