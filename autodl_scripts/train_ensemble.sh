#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

if [[ -f "$SCRIPT_DIR/autodl_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/autodl_env.sh"
fi

DATA_ROOT="${DATA_ROOT:-${RAICOM_DATA_ROOT:-/root/autodl-tmp/data}}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/raicom_runs}"
RUN_NAME="${RUN_NAME:-ensemble_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/$RUN_NAME}"
BATCH_SIZE="${BATCH_SIZE:-28}"
HEAD_EPOCHS="${HEAD_EPOCHS:-}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-}"
HEAD_LR="${HEAD_LR:-}"
FINETUNE_LR="${FINETUNE_LR:-}"
EARLY_STOP="${EARLY_STOP:-0}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$OUTPUT_DIR/logs"
cd "$PROJECT_DIR"

cmd=(
  python "$PROJECT_DIR/scripts/train_ensemble_four_models_meta.py"
  --data-root "$DATA_ROOT"
  --output-dir "$OUTPUT_DIR"
  --batch-size "$BATCH_SIZE"
  --early-stop "$EARLY_STOP"
  --cuda-device "$CUDA_DEVICE"
)

[[ -z "$HEAD_EPOCHS" ]] || cmd+=(--head-epochs "$HEAD_EPOCHS")
[[ -z "$FINETUNE_EPOCHS" ]] || cmd+=(--finetune-epochs "$FINETUNE_EPOCHS")
[[ -z "$HEAD_LR" ]] || cmd+=(--head-lr "$HEAD_LR")
[[ -z "$FINETUNE_LR" ]] || cmd+=(--finetune-lr "$FINETUNE_LR")

if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  extra_args_array=($EXTRA_ARGS)
  cmd+=("${extra_args_array[@]}")
fi

printf '%q ' "${cmd[@]}" | tee "$OUTPUT_DIR/logs/command.txt"
printf '\n' | tee -a "$OUTPUT_DIR/logs/command.txt"
"${cmd[@]}" 2>&1 | tee "$OUTPUT_DIR/logs/train.log"
