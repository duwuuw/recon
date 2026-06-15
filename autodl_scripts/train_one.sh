#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

if [[ -f "$SCRIPT_DIR/autodl_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/autodl_env.sh"
fi

SCRIPT_NAME="${SCRIPT_NAME:-train_convnext11.py}"
DATA_ROOT="${DATA_ROOT:-${RAICOM_DATA_ROOT:-/root/autodl-tmp/data}}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/raicom_runs}"
RUN_NAME="${RUN_NAME:-${SCRIPT_NAME%.py}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/$RUN_NAME}"
BATCH_SIZE="${BATCH_SIZE:-}"
HEAD_EPOCHS="${HEAD_EPOCHS:-}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-}"
HEAD_LR="${HEAD_LR:-}"
FINETUNE_LR="${FINETUNE_LR:-}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

SCRIPT_PATH="$PROJECT_DIR/scripts/$SCRIPT_NAME"
if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "[train_one] script not found: $SCRIPT_PATH" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR/logs"
cd "$PROJECT_DIR"

HELP_TEXT="$(python "$SCRIPT_PATH" --help 2>&1 || true)"
cmd=(python "$SCRIPT_PATH" --data-root "$DATA_ROOT" --output-dir "$OUTPUT_DIR")

append_if_supported() {
  local flag="$1"
  local value="$2"
  [[ -n "$value" ]] || return 0
  if grep -Eq "(^|[[:space:]])$flag([,=[:space:]]|$)" <<<"$HELP_TEXT"; then
    cmd+=("$flag" "$value")
  fi
}

append_if_supported --batch-size "$BATCH_SIZE"
append_if_supported --head-epochs "$HEAD_EPOCHS"
append_if_supported --finetune-epochs "$FINETUNE_EPOCHS"
append_if_supported --head-lr "$HEAD_LR"
append_if_supported --finetune-lr "$FINETUNE_LR"
append_if_supported --num-workers "$NUM_WORKERS"
append_if_supported --seed "$SEED"

if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  extra_args_array=($EXTRA_ARGS)
  cmd+=("${extra_args_array[@]}")
fi

echo "[train_one] project: $PROJECT_DIR"
echo "[train_one] data: $DATA_ROOT"
echo "[train_one] output: $OUTPUT_DIR"
printf '%q ' "${cmd[@]}" | tee "$OUTPUT_DIR/logs/command.txt"
printf '\n' | tee -a "$OUTPUT_DIR/logs/command.txt"

"${cmd[@]}" 2>&1 | tee "$OUTPUT_DIR/logs/train.log"
