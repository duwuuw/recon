#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ALL_MODELS=(
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
)

usage() {
  cat <<'EOF'
Usage:
  bash train_all_single_models.sh [model ...]

Examples:
  bash train_all_single_models.sh
  bash train_all_single_models.sh convnext11 fastvit_s24 repvit_m2
  bash train_all_single_models.sh train_convnext11.py train_fastvit_s24.py

Available models:
  convnext11 efficientnet vit11 resnet18 mambaout_kobe mambaout_small_rw
  mobilenetv4 mobilenetv4_hyper fastvit_s24 fastvit_sa36 fasternet repvit repvit_m2
EOF
}

to_script_name() {
  local name="$1"
  name="${name#train_}"
  name="${name%.py}"

  case "$name" in
    convnext11|efficientnet|vit11|resnet18|mambaout_kobe|mambaout_small_rw|mobilenetv4|mobilenetv4_hyper|fastvit_s24|fastvit_sa36|fasternet|repvit|repvit_m2)
      echo "train_${name}.py"
      ;;
    *)
      echo "[train_all] unknown model: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
  usage
  exit 0
fi

if [[ "$#" -eq 0 ]]; then
  SELECTED_MODELS=("${ALL_MODELS[@]}")
else
  SELECTED_MODELS=("$@")
fi

for model in "${SELECTED_MODELS[@]}"; do
  script="$(to_script_name "$model")"
  echo "[train_all] start $script"
  SCRIPT_NAME="$script" RUN_NAME="${script%.py}_$(date +%Y%m%d_%H%M%S)" "$SCRIPT_DIR/train_one.sh"
done
