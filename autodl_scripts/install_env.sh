#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
ENV_NAME="${ENV_NAME:-raicom}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
SKIP_APT="${SKIP_APT:-0}"
SKIP_TORCH="${SKIP_TORCH:-0}"
INSTALL_GDN="${INSTALL_GDN:-0}"

echo "[install_env] project: $PROJECT_DIR"
echo "[install_env] env: $ENV_NAME"

if [[ "$SKIP_APT" != "1" ]] && command -v apt-get >/dev/null 2>&1; then
  apt-get update
  apt-get install -y git unzip tmux htop libgl1 libglib2.0-0
fi

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
  fi
  conda activate "$ENV_NAME"
else
  python3 -m venv "$PROJECT_DIR/.venv"
  # shellcheck disable=SC1091
  source "$PROJECT_DIR/.venv/bin/activate"
fi

python -m pip install --upgrade pip setuptools wheel

if [[ "$SKIP_TORCH" != "1" ]]; then
  python -m pip install torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"
fi

cd "$PROJECT_DIR"
python -m pip install -r requirements.txt
python -m pip install -e ".[train]"

if [[ "$INSTALL_GDN" == "1" ]]; then
  python -m pip install "flash-linear-attention>=0.1"
fi

cat > "$SCRIPT_DIR/autodl_env.sh" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
export PROJECT_DIR="$PROJECT_DIR"
export RAICOM_REPO_ROOT="$PROJECT_DIR"
export RAICOM_DATA_ROOT="\${RAICOM_DATA_ROOT:-/root/autodl-tmp/data}"
export HF_ENDPOINT="\${HF_ENDPOINT:-https://hf-mirror.com}"
if command -v conda >/dev/null 2>&1; then
  source "\$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
elif [[ -f "$PROJECT_DIR/.venv/bin/activate" ]]; then
  source "$PROJECT_DIR/.venv/bin/activate"
fi
EOF
chmod +x "$SCRIPT_DIR/autodl_env.sh"

echo "[install_env] done"
echo "[install_env] next: source $SCRIPT_DIR/autodl_env.sh"
