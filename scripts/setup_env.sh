#!/usr/bin/env bash
# No sudo, no system packages, and NO FlashAttention compilation.
set -euo pipefail
ROOT=${HECATEPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
ENV_DIR=${ENV_DIR:-$ROOT/.venv}
PYTHON_BOOTSTRAP=${PYTHON_BOOTSTRAP:-python3}
PIP_MIRROR=${PIP_MIRROR:-https://pypi.org/simple}
mkdir -p "$ROOT/logs" "$ROOT/cache/pip"
if [ ! -x "$ENV_DIR/bin/python" ]; then
 "$PYTHON_BOOTSTRAP" -m venv --without-pip "$ENV_DIR"
fi
PY="$ENV_DIR/bin/python"
if ! "$PY" -m pip --version >/dev/null 2>&1; then
 curl -fL https://bootstrap.pypa.io/get-pip.py -o "$ROOT/cache/get-pip.py"
 "$PY" "$ROOT/cache/get-pip.py"
fi
"$PY" -m pip install --cache-dir "$ROOT/cache/pip" torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
 --index-url https://download.pytorch.org/whl/cu124
"$PY" -m pip install --only-binary=:all: --cache-dir "$ROOT/cache/pip" --index-url "$PIP_MIRROR" -r "$ROOT/requirements.txt"
"$PY" -m pip check
"$PY" -m pip freeze > "$ROOT/requirements.lock.txt"
