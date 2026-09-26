#!/usr/bin/env bash
set -euo pipefail
ROOT=${SAGEPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
ENV_DIR=${ENV_DIR:-$ROOT/.venv}
: "${PREDICTIONS:?Set PREDICTIONS to an evaluation predictions.jsonl file}"
JUDGE_MODEL=${JUDGE_MODEL:-gpt-5-nano-2025-08-07}
JUDGE_MAX_TOKENS=${JUDGE_MAX_TOKENS:-2048}
JUDGE_CONCURRENCY=${JUDGE_CONCURRENCY:-4}
JUDGE_RETRIES=${JUDGE_RETRIES:-3}
JUDGE_TIMEOUT=${JUDGE_TIMEOUT:-120}
JUDGE_LIMIT=${JUDGE_LIMIT:-0}
JUDGE_CACHE=${JUDGE_CACHE:-$ROOT/cache/judge.jsonl}
JUDGED_OUTPUT=${JUDGED_OUTPUT:-${PREDICTIONS%.jsonl}.judged.jsonl}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
"$ENV_DIR/bin/python" -u -m sagepo.judge --predictions "$PREDICTIONS" --output "$JUDGED_OUTPUT" \
  --cache "$JUDGE_CACHE" --model "$JUDGE_MODEL" --max-tokens "$JUDGE_MAX_TOKENS" \
  --concurrency "$JUDGE_CONCURRENCY" --retries "$JUDGE_RETRIES" --limit "$JUDGE_LIMIT" --timeout "$JUDGE_TIMEOUT"
