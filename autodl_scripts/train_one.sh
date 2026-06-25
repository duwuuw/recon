#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

PROJECT_DIR="$(autodl_project_dir)"

if [[ -f "$SCRIPT_DIR/autodl_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/autodl_env.sh"
fi

SCRIPT_NAME="${SCRIPT_NAME:-}"
MODEL_PRESET="${MODEL_PRESET:-}"
if [[ -n "$MODEL_PRESET" && -z "$SCRIPT_NAME" ]]; then
  SCRIPT_NAME="train_timm_preset.py"
fi
SCRIPT_NAME="${SCRIPT_NAME:-train_convnext11.py}"
DATA_ROOT="$(autodl_resolve_data_root "$PROJECT_DIR")"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/raicom_runs}"
RUN_STEM="${MODEL_PRESET:-${SCRIPT_NAME%.py}}"
RUN_NAME="${RUN_NAME:-${RUN_STEM}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/$RUN_NAME}"
BATCH_SIZE="${BATCH_SIZE:-}"
HEAD_EPOCHS="${HEAD_EPOCHS:-}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-}"
HEAD_LR="${HEAD_LR:-}"
FINETUNE_LR="${FINETUNE_LR:-}"
EARLY_STOP="${EARLY_STOP:-}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-}"
IMAGE_SIZE="${IMAGE_SIZE:-}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
DRY_RUN="${DRY_RUN:-0}"
CPU="${CPU:-0}"

SCRIPT_PATH="$PROJECT_DIR/scripts/$SCRIPT_NAME"
if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "[train_one] script not found: $SCRIPT_PATH" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR/logs"
cd "$PROJECT_DIR"

autodl_check_gpu

cmd=(python "$SCRIPT_PATH")
if [[ -n "$MODEL_PRESET" ]]; then
  cmd+=("$MODEL_PRESET")
fi
cmd+=(--data-root "$DATA_ROOT" --output-dir "$OUTPUT_DIR")

append_arg() {
  local flag="$1"
  local value="$2"
  [[ -n "$value" ]] || return 0
  cmd+=("$flag" "$value")
}

append_arg --batch-size "$BATCH_SIZE"
append_arg --head-epochs "$HEAD_EPOCHS"
append_arg --finetune-epochs "$FINETUNE_EPOCHS"
append_arg --head-lr "$HEAD_LR"
append_arg --finetune-lr "$FINETUNE_LR"
append_arg --early-stop "$EARLY_STOP"
append_arg --early-stop-min-delta "$EARLY_STOP_MIN_DELTA"
append_arg --image-size "$IMAGE_SIZE"

if [[ "$SCRIPT_NAME" != "train_gdn.py" ]]; then
  append_arg --num-workers "$NUM_WORKERS"
  append_arg --seed "$SEED"
fi

if [[ "$CPU" == "1" ]]; then
  cmd+=(--cpu)
fi

if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  extra_args_array=($EXTRA_ARGS)
  cmd+=("${extra_args_array[@]}")
fi

echo "[train_one] project: $PROJECT_DIR"
if [[ -n "$MODEL_PRESET" ]]; then
  echo "[train_one] preset: $MODEL_PRESET"
fi
echo "[train_one] data: $DATA_ROOT"
echo "[train_one] output: $OUTPUT_DIR"
printf '%q ' "${cmd[@]}" | tee "$OUTPUT_DIR/logs/command.txt"
printf '\n' | tee -a "$OUTPUT_DIR/logs/command.txt"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[train_one] dry run, command not executed"
  exit 0
fi

"${cmd[@]}" 2>&1 | tee "$OUTPUT_DIR/logs/train.log"
