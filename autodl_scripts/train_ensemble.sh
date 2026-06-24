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
BATCH_SIZE="${BATCH_SIZE:-}"
IMAGE_SIZE="${IMAGE_SIZE:-}"
ENSEMBLE="${ENSEMBLE:-balanced}"
MODELS="${MODELS:-}"
HEAD_EPOCHS="${HEAD_EPOCHS:-}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-}"
HEAD_LR="${HEAD_LR:-}"
FINETUNE_LR="${FINETUNE_LR:-}"
EARLY_STOP="${EARLY_STOP:-}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
WEIGHT_SEARCH_TRIALS="${WEIGHT_SEARCH_TRIALS:-}"
WEIGHT_METRIC="${WEIGHT_METRIC:-}"
TTA_HFLIP="${TTA_HFLIP:-0}"
REUSE_CHECKPOINTS="${REUSE_CHECKPOINTS:-0}"
NO_PRETRAINED="${NO_PRETRAINED:-0}"
CPU="${CPU:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$OUTPUT_DIR/logs"
cd "$PROJECT_DIR"

cmd=(
  python "$PROJECT_DIR/scripts/train_small_strong_ensemble.py"
  --data-root "$DATA_ROOT"
  --output-dir "$OUTPUT_DIR"
  --ensemble "$ENSEMBLE"
  --cuda-device "$CUDA_DEVICE"
)

[[ -z "$BATCH_SIZE" ]] || cmd+=(--batch-size "$BATCH_SIZE")
[[ -z "$IMAGE_SIZE" ]] || cmd+=(--image-size "$IMAGE_SIZE")
if [[ -n "$MODELS" ]]; then
  # shellcheck disable=SC2206
  models_array=($MODELS)
  cmd+=(--models "${models_array[@]}")
fi
[[ -z "$HEAD_EPOCHS" ]] || cmd+=(--head-epochs "$HEAD_EPOCHS")
[[ -z "$FINETUNE_EPOCHS" ]] || cmd+=(--finetune-epochs "$FINETUNE_EPOCHS")
[[ -z "$HEAD_LR" ]] || cmd+=(--head-lr "$HEAD_LR")
[[ -z "$FINETUNE_LR" ]] || cmd+=(--finetune-lr "$FINETUNE_LR")
[[ -z "$EARLY_STOP" ]] || cmd+=(--early-stop "$EARLY_STOP")
[[ -z "$EARLY_STOP_MIN_DELTA" ]] || cmd+=(--early-stop-min-delta "$EARLY_STOP_MIN_DELTA")
[[ -z "$WEIGHT_SEARCH_TRIALS" ]] || cmd+=(--weight-search-trials "$WEIGHT_SEARCH_TRIALS")
[[ -z "$WEIGHT_METRIC" ]] || cmd+=(--weight-metric "$WEIGHT_METRIC")
[[ "$TTA_HFLIP" != "1" ]] || cmd+=(--tta-hflip)
[[ "$REUSE_CHECKPOINTS" != "1" ]] || cmd+=(--reuse-checkpoints)
[[ "$NO_PRETRAINED" != "1" ]] || cmd+=(--no-pretrained)
[[ "$CPU" != "1" ]] || cmd+=(--cpu)

if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  extra_args_array=($EXTRA_ARGS)
  cmd+=("${extra_args_array[@]}")
fi

printf '%q ' "${cmd[@]}" | tee "$OUTPUT_DIR/logs/command.txt"
printf '\n' | tee -a "$OUTPUT_DIR/logs/command.txt"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[train_ensemble] dry run, command not executed"
  exit 0
fi
"${cmd[@]}" 2>&1 | tee "$OUTPUT_DIR/logs/train.log"
