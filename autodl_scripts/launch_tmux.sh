#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${TARGET:-train_one.sh}"
SESSION_NAME="${SESSION_NAME:-raicom_train}"
TARGET_ARGS="${TARGET_ARGS:-}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "[launch_tmux] tmux not found, run in current shell"
  exec "$SCRIPT_DIR/$TARGET"
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[launch_tmux] session exists: $SESSION_NAME"
  echo "[launch_tmux] attach: tmux attach -t $SESSION_NAME"
  exit 0
fi

tmux new-session -d -s "$SESSION_NAME" "cd '$SCRIPT_DIR' && bash '$SCRIPT_DIR/$TARGET' $TARGET_ARGS"
echo "[launch_tmux] started: $SESSION_NAME"
echo "[launch_tmux] attach: tmux attach -t $SESSION_NAME"
