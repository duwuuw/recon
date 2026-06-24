"""Curated timm image-classification presets under roughly 30M parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from raicom.classifier import ClassifierTrainConfig


@dataclass(frozen=True)
class TimmPreset:
    key: str
    timm_model: str
    batch_size: int = 32
    image_size: int = 224
    weight_decay: float = 2.5e-4
    model_kwargs: dict[str, Any] = field(default_factory=dict)


def _preset(
    key: str,
    timm_model: str,
    *,
    batch_size: int = 32,
    image_size: int = 224,
    weight_decay: float = 2.5e-4,
    model_kwargs: dict[str, Any] | None = None,
) -> TimmPreset:
    return TimmPreset(
        key=key,
        timm_model=timm_model,
        batch_size=batch_size,
        image_size=image_size,
        weight_decay=weight_decay,
        model_kwargs=model_kwargs or {},
    )


TIMM_PRESETS: dict[str, TimmPreset] = {
    "convnextv2_atto": _preset("convnextv2_atto", "convnextv2_atto.fcmae_ft_in1k"),
    "convnextv2_femto": _preset("convnextv2_femto", "convnextv2_femto.fcmae_ft_in1k"),
    "convnextv2_pico": _preset("convnextv2_pico", "convnextv2_pico.fcmae_ft_in1k"),
    "convnextv2_nano": _preset("convnextv2_nano", "convnextv2_nano.fcmae_ft_in1k"),
    "convnextv2_tiny": _preset(
        "convnextv2_tiny", "convnextv2_tiny.fcmae_ft_in1k", batch_size=24
    ),
    "mobilenetv4_conv_small_050": _preset(
        "mobilenetv4_conv_small_050", "mobilenetv4_conv_small_050.e3000_r224_in1k"
    ),
    "mobilenetv4_conv_small": _preset(
        "mobilenetv4_conv_small", "mobilenetv4_conv_small.e2400_r224_in1k"
    ),
    "mobilenetv4_conv_medium": _preset(
        "mobilenetv4_conv_medium", "mobilenetv4_conv_medium.e500_r224_in1k"
    ),
    "mobilenetv4_conv_blur_medium": _preset(
        "mobilenetv4_conv_blur_medium", "mobilenetv4_conv_blur_medium.e500_r224_in1k"
    ),
    "mobilenetv4_hybrid_medium": _preset(
        "mobilenetv4_hybrid_medium", "mobilenetv4_hybrid_medium.e500_r224_in1k"
    ),
    "fastvit_t8": _preset("fastvit_t8", "fastvit_t8.apple_dist_in1k"),
    "fastvit_t12": _preset("fastvit_t12", "fastvit_t12.apple_dist_in1k"),
    "fastvit_s12": _preset("fastvit_s12", "fastvit_s12.apple_dist_in1k"),
    "fastvit_sa12": _preset("fastvit_sa12", "fastvit_sa12.apple_dist_in1k"),
    "fastvit_sa24": _preset("fastvit_sa24", "fastvit_sa24.apple_dist_in1k", batch_size=24),
    "fastvit_mci0": _preset("fastvit_mci0", "fastvit_mci0.apple_mclip"),
    "fastvit_mci1": _preset("fastvit_mci1", "fastvit_mci1.apple_mclip", batch_size=24),
    "efficientvit_b0": _preset("efficientvit_b0", "efficientvit_b0.r224_in1k"),
    "efficientvit_b1": _preset("efficientvit_b1", "efficientvit_b1.r224_in1k"),
    "efficientvit_b2": _preset("efficientvit_b2", "efficientvit_b2.r224_in1k", batch_size=24),
    "efficientvit_m0": _preset("efficientvit_m0", "efficientvit_m0.r224_in1k"),
    "efficientvit_m1": _preset("efficientvit_m1", "efficientvit_m1.r224_in1k"),
    "efficientvit_m2": _preset("efficientvit_m2", "efficientvit_m2.r224_in1k"),
    "efficientvit_m3": _preset("efficientvit_m3", "efficientvit_m3.r224_in1k"),
    "efficientvit_m4": _preset("efficientvit_m4", "efficientvit_m4.r224_in1k"),
    "efficientvit_m5": _preset("efficientvit_m5", "efficientvit_m5.r224_in1k"),
    "maxvit_rmlp_pico_rw_256": _preset(
        "maxvit_rmlp_pico_rw_256",
        "maxvit_rmlp_pico_rw_256.sw_in1k",
        image_size=256,
    ),
    "maxvit_nano_rw_256": _preset(
        "maxvit_nano_rw_256",
        "maxvit_nano_rw_256.sw_in1k",
        image_size=256,
    ),
    "maxvit_rmlp_nano_rw_256": _preset(
        "maxvit_rmlp_nano_rw_256",
        "maxvit_rmlp_nano_rw_256.sw_in1k",
        image_size=256,
    ),
    "maxvit_tiny_rw_224": _preset(
        "maxvit_tiny_rw_224", "maxvit_tiny_rw_224.sw_in1k", batch_size=24
    ),
    "maxvit_rmlp_tiny_rw_256": _preset(
        "maxvit_rmlp_tiny_rw_256",
        "maxvit_rmlp_tiny_rw_256.sw_in1k",
        batch_size=24,
        image_size=256,
    ),
    "coatnet_nano_rw_224": _preset("coatnet_nano_rw_224", "coatnet_nano_rw_224.sw_in1k"),
    "coatnet_rmlp_nano_rw_224": _preset(
        "coatnet_rmlp_nano_rw_224", "coatnet_rmlp_nano_rw_224.sw_in1k"
    ),
    "coatnet_0_rw_224": _preset(
        "coatnet_0_rw_224", "coatnet_0_rw_224.sw_in1k", batch_size=24
    ),
    "coatnet_bn_0_rw_224": _preset(
        "coatnet_bn_0_rw_224", "coatnet_bn_0_rw_224.sw_in1k", batch_size=24
    ),
    "efficientformerv2_s0": _preset(
        "efficientformerv2_s0", "efficientformerv2_s0.snap_dist_in1k"
    ),
    "efficientformerv2_s1": _preset(
        "efficientformerv2_s1", "efficientformerv2_s1.snap_dist_in1k"
    ),
    "efficientformerv2_s2": _preset(
        "efficientformerv2_s2", "efficientformerv2_s2.snap_dist_in1k"
    ),
    "efficientformerv2_l": _preset(
        "efficientformerv2_l", "efficientformerv2_l.snap_dist_in1k", batch_size=24
    ),
    "edgenext_xx_small": _preset("edgenext_xx_small", "edgenext_xx_small.in1k"),
    "edgenext_x_small": _preset("edgenext_x_small", "edgenext_x_small.in1k"),
    "edgenext_small": _preset("edgenext_small", "edgenext_small.usi_in1k"),
    "edgenext_small_rw": _preset("edgenext_small_rw", "edgenext_small_rw.sw_in1k"),
    "edgenext_base": _preset("edgenext_base", "edgenext_base.usi_in1k"),
    "fasternet_t0": _preset("fasternet_t0", "fasternet_t0.in1k"),
    "fasternet_t1": _preset("fasternet_t1", "fasternet_t1.in1k"),
    "fasternet_t2": _preset("fasternet_t2", "fasternet_t2.in1k"),
    "fasternet_s": _preset("fasternet_s", "fasternet_s.in1k", batch_size=24),
    "repvit_m1": _preset("repvit_m1", "repvit_m1.dist_in1k"),
    "repvit_m0_9": _preset("repvit_m0_9", "repvit_m0_9.dist_450e_in1k"),
    "repvit_m1_0": _preset("repvit_m1_0", "repvit_m1_0.dist_450e_in1k"),
    "repvit_m2": _preset("repvit_m2", "repvit_m2.dist_in1k"),
    "repvit_m1_1": _preset("repvit_m1_1", "repvit_m1_1.dist_450e_in1k"),
    "repvit_m3": _preset("repvit_m3", "repvit_m3.dist_in1k"),
    "repvit_m1_5": _preset("repvit_m1_5", "repvit_m1_5.dist_450e_in1k"),
    "repvit_m2_3": _preset("repvit_m2_3", "repvit_m2_3.dist_450e_in1k", batch_size=24),
    "mambaout_femto": _preset("mambaout_femto", "mambaout_femto.in1k"),
    "mambaout_kobe": _preset("mambaout_kobe", "mambaout_kobe.in1k"),
    "mambaout_tiny": _preset("mambaout_tiny", "mambaout_tiny.in1k", batch_size=24),
    "tiny_vit_5m_224": _preset("tiny_vit_5m_224", "tiny_vit_5m_224.dist_in22k_ft_in1k"),
    "tiny_vit_11m_224": _preset("tiny_vit_11m_224", "tiny_vit_11m_224.dist_in22k_ft_in1k"),
    "tiny_vit_21m_224": _preset(
        "tiny_vit_21m_224", "tiny_vit_21m_224.dist_in22k_ft_in1k", batch_size=24
    ),
    "eva02_tiny_patch14_224": _preset("eva02_tiny_patch14_224", "eva02_tiny_patch14_224.mim_in22k"),
    "eva02_small_patch14_224": _preset(
        "eva02_small_patch14_224", "eva02_small_patch14_224.mim_in22k", batch_size=24
    ),
    "dinov2_small": _preset(
        "dinov2_small",
        "vit_small_patch14_dinov2.lvd142m",
        batch_size=24,
        model_kwargs={"img_size": 224},
    ),
    "dinov2_small_reg4": _preset(
        "dinov2_small_reg4",
        "vit_small_patch14_reg4_dinov2.lvd142m",
        batch_size=24,
        model_kwargs={"img_size": 224},
    ),
    "deit3_small_patch16_224": _preset(
        "deit3_small_patch16_224", "deit3_small_patch16_224.fb_in22k_ft_in1k", batch_size=24
    ),
    "mobilevitv2_050": _preset("mobilevitv2_050", "mobilevitv2_050.cvnets_in1k"),
    "mobilevitv2_075": _preset("mobilevitv2_075", "mobilevitv2_075.cvnets_in1k"),
    "mobilevitv2_100": _preset("mobilevitv2_100", "mobilevitv2_100.cvnets_in1k"),
    "mobilevitv2_125": _preset("mobilevitv2_125", "mobilevitv2_125.cvnets_in1k"),
    "mobilevitv2_150": _preset("mobilevitv2_150", "mobilevitv2_150.cvnets_in1k"),
    "mobilevitv2_175": _preset("mobilevitv2_175", "mobilevitv2_175.cvnets_in1k"),
    "mobilevitv2_200": _preset("mobilevitv2_200", "mobilevitv2_200.cvnets_in1k", batch_size=24),
    "mobileone_s0": _preset("mobileone_s0", "mobileone_s0.apple_in1k"),
    "mobileone_s1": _preset("mobileone_s1", "mobileone_s1.apple_in1k"),
    "mobileone_s2": _preset("mobileone_s2", "mobileone_s2.apple_in1k"),
    "mobileone_s3": _preset("mobileone_s3", "mobileone_s3.apple_in1k"),
    "mobileone_s4": _preset("mobileone_s4", "mobileone_s4.apple_in1k"),
    "caformer_s18": _preset("caformer_s18", "caformer_s18.sail_in22k_ft_in1k", batch_size=24),
    "convformer_s18": _preset(
        "convformer_s18", "convformer_s18.sail_in22k_ft_in1k", batch_size=24
    ),
    "swiftformer_xs": _preset("swiftformer_xs", "swiftformer_xs.dist_in1k"),
    "swiftformer_s": _preset("swiftformer_s", "swiftformer_s.dist_in1k"),
    "swiftformer_l1": _preset("swiftformer_l1", "swiftformer_l1.dist_in1k"),
    "swiftformer_l3": _preset("swiftformer_l3", "swiftformer_l3.dist_in1k", batch_size=24),
}


def preset_keys() -> list[str]:
    return list(TIMM_PRESETS)


def build_timm_preset_config(key: str) -> ClassifierTrainConfig:
    try:
        preset = TIMM_PRESETS[key]
    except KeyError as exc:
        available = ", ".join(preset_keys())
        raise KeyError(f"Unknown timm preset {key!r}. Available presets: {available}") from exc
    return ClassifierTrainConfig(
        num_classes=4,
        timm_model=preset.timm_model,
        checkpoint_name=f"{preset.key}.pth",
        curves_name=f"{preset.key}.png",
        batch_size=preset.batch_size,
        weight_decay=preset.weight_decay,
        image_size=preset.image_size,
        model_kwargs=dict(preset.model_kwargs),
    )
