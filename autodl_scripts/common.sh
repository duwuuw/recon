#!/usr/bin/env bash
# Shared helpers for AutoDL training scripts.

autodl_project_dir() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
  echo "${PROJECT_DIR:-$(cd "$script_dir/.." && pwd)}"
}

autodl_resolve_data_root() {
  local project_dir="${1:?project_dir required}"
  if [[ -n "${DATA_ROOT:-}" ]]; then
    echo "$DATA_ROOT"
    return 0
  fi
  if [[ -n "${RAICOM_DATA_ROOT:-}" ]]; then
    echo "$RAICOM_DATA_ROOT"
    return 0
  fi
  if [[ -d "$project_dir/data/raw/dataset/cloudy" ]]; then
    echo "$project_dir/data/raw/dataset"
    return 0
  fi
  echo "/root/autodl-tmp/data"
}

autodl_check_gpu() {
  if [[ "${CPU:-0}" == "1" ]]; then
    echo "[autodl] CPU=1, skip GPU check"
    return 0
  fi
  python - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print("[autodl] ERROR: CUDA 不可用。请检查 install_env.sh 是否安装了 GPU 版 PyTorch。", file=sys.stderr)
    print("[autodl] 调试时可设 CPU=1", file=sys.stderr)
    sys.exit(1)
name = torch.cuda.get_device_name(0)
x = torch.zeros(1, device="cuda:0")
print(f"[autodl] GPU OK: cuda:0 ({name}), torch {torch.__version__}")
PY
}
