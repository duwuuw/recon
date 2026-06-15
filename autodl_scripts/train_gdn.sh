#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCRIPT_NAME=train_gdn.py \
BATCH_SIZE="${BATCH_SIZE:-16}" \
"$SCRIPT_DIR/train_one.sh"
