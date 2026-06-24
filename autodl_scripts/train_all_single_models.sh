#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TIMM_PRESETS=(
  convnextv2_atto
  convnextv2_femto
  convnextv2_pico
  convnextv2_nano
  convnextv2_tiny
  mobilenetv4_conv_small_050
  mobilenetv4_conv_small
  mobilenetv4_conv_medium
  mobilenetv4_conv_blur_medium
  mobilenetv4_hybrid_medium
  fastvit_t8
  fastvit_t12
  fastvit_s12
  fastvit_sa12
  fastvit_sa24
  fastvit_mci0
  fastvit_mci1
  efficientvit_b0
  efficientvit_b1
  efficientvit_b2
  efficientvit_m0
  efficientvit_m1
  efficientvit_m2
  efficientvit_m3
  efficientvit_m4
  efficientvit_m5
  maxvit_rmlp_pico_rw_256
  maxvit_nano_rw_256
  maxvit_rmlp_nano_rw_256
  maxvit_tiny_rw_224
  maxvit_rmlp_tiny_rw_256
  coatnet_nano_rw_224
  coatnet_rmlp_nano_rw_224
  coatnet_0_rw_224
  coatnet_bn_0_rw_224
  efficientformerv2_s0
  efficientformerv2_s1
  efficientformerv2_s2
  efficientformerv2_l
  edgenext_xx_small
  edgenext_x_small
  edgenext_small
  edgenext_small_rw
  edgenext_base
  fasternet_t0
  fasternet_t1
  fasternet_t2
  fasternet_s
  repvit_m1
  repvit_m0_9
  repvit_m1_0
  repvit_m2
  repvit_m1_1
  repvit_m3
  repvit_m1_5
  repvit_m2_3
  mambaout_femto
  mambaout_kobe
  mambaout_tiny
  tiny_vit_5m_224
  tiny_vit_11m_224
  tiny_vit_21m_224
  eva02_tiny_patch14_224
  eva02_small_patch14_224
  dinov2_small
  dinov2_small_reg4
  deit3_small_patch16_224
  mobilevitv2_050
  mobilevitv2_075
  mobilevitv2_100
  mobilevitv2_125
  mobilevitv2_150
  mobilevitv2_175
  mobilevitv2_200
  mobileone_s0
  mobileone_s1
  mobileone_s2
  mobileone_s3
  mobileone_s4
  caformer_s18
  convformer_s18
  swiftformer_xs
  swiftformer_s
  swiftformer_l1
  swiftformer_l3
)

LEGACY_SCRIPT_MODELS=(
  convnext11
  efficientnet
  vit11
  resnet18
  mambaout_kobe
  mambaout_small_rw
  mobilenetv4
  mobilenetv4_hyper
  fastvit_s24
  fastvit_sa36
  fasternet
  repvit
  repvit_m2
  dinov2_small
)

contains() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    [[ "$item" == "$needle" ]] && return 0
  done
  return 1
}

usage() {
  cat <<'EOF'
Usage:
  bash train_all_single_models.sh [model ...]

Examples:
  bash train_all_single_models.sh
  bash train_all_single_models.sh convnextv2_tiny mobilenetv4_conv_medium fastvit_sa24
  bash train_all_single_models.sh convnext11 train_fastvit_sa36.py

Default with no arguments runs the curated <=30M timm presets.

EOF
  printf 'Curated presets:\n'
  printf '  %s\n' "${TIMM_PRESETS[@]}"
  printf '\nLegacy script aliases still accepted:\n'
  printf '  %s\n' "${LEGACY_SCRIPT_MODELS[@]}"
}

to_target() {
  local raw="$1"
  local name="$raw"
  name="${name#train_}"
  name="${name%.py}"

  if contains "$name" "${TIMM_PRESETS[@]}"; then
    echo "preset:$name"
    return 0
  fi
  if contains "$name" "${LEGACY_SCRIPT_MODELS[@]}"; then
    echo "script:train_${name}.py"
    return 0
  fi
  if [[ "$raw" == *.py ]]; then
    echo "script:$raw"
    return 0
  fi

  echo "[train_all] unknown model: $raw" >&2
  usage >&2
  exit 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
  usage
  exit 0
fi

if [[ "$#" -eq 0 ]]; then
  SELECTED_MODELS=("${TIMM_PRESETS[@]}")
else
  SELECTED_MODELS=("$@")
fi

for model in "${SELECTED_MODELS[@]}"; do
  target="$(to_target "$model")"
  kind="${target%%:*}"
  value="${target#*:}"
  timestamp="$(date +%Y%m%d_%H%M%S)"
  if [[ "$kind" == "preset" ]]; then
    echo "[train_all] start preset $value"
    MODEL_PRESET="$value" SCRIPT_NAME="train_timm_preset.py" RUN_NAME="${value}_${timestamp}" "$SCRIPT_DIR/train_one.sh"
  else
    echo "[train_all] start $value"
    SCRIPT_NAME="$value" RUN_NAME="${value%.py}_${timestamp}" "$SCRIPT_DIR/train_one.sh"
  fi
done
