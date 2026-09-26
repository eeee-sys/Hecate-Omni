#!/usr/bin/env bash
set -euo pipefail
ROOT=${SAGEPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
ENV_DIR=${ENV_DIR:-$ROOT/.venv}
REWARD_REPO=${REWARD_REPO:-sentence-transformers/all-MiniLM-L6-v2}
REWARD_REVISION=${REWARD_REVISION:-1110a243fdf4706b3f48f1d95db1a4f5529b4d41}
WEIGHTS_ROOT=${WEIGHTS_ROOT:-$ROOT/checkpoints}
REWARD_MODEL=${REWARD_MODEL:-$WEIGHTS_ROOT/reward_models/all-MiniLM-L6-v2}
DOWNLOAD_WORKERS=${DOWNLOAD_WORKERS:-4}
export HF_HOME="$ROOT/cache/huggingface"
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}
"$ENV_DIR/bin/python" - "$REWARD_REPO" "$REWARD_REVISION" "$REWARD_MODEL" "$DOWNLOAD_WORKERS" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, revision, destination, workers = sys.argv[1:]
print(snapshot_download(repo, revision=revision, local_dir=destination,
      allow_patterns=['*.json', '*.txt', '*.safetensors', '1_Pooling/*'], max_workers=int(workers)))
PY
