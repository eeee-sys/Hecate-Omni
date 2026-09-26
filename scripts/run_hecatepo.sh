#!/usr/bin/env bash
# Single source of launch-time settings. Every value supports an environment override.
set -euo pipefail
MODE=${1:-train}
if [ "$#" -gt 0 ]; then shift; fi
ROOT=${HECATEPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
ENV_DIR=${ENV_DIR:-$ROOT/.venv}
MODEL_PATH=${MODEL_PATH:-$ROOT/models/OmniSapiens2.0}
DATA_PATH=${DATA_PATH:-$ROOT/data/human_behavior_atlas}
WEIGHTS_ROOT=${WEIGHTS_ROOT:-$ROOT/checkpoints}
REWARD_MODEL=${REWARD_MODEL:-$WEIGHTS_ROOT/reward_models/all-MiniLM-L6-v2}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/outputs}
INDEX_ROOT=${INDEX_ROOT:-$ROOT/cache/index}
MEDIA_CACHE=${MEDIA_CACHE:-$ROOT/cache/media}
RUN_NAME=${RUN_NAME:-hecatepo_${MODE}_$(date +%Y%m%d_%H%M%S)}
GPU_IDS=${GPU_IDS:-0}
NUM_PROCESSES=${NUM_PROCESSES:-1}
SEED=${SEED:-42}

# Train only these modules inside the Thinker language decoder; never match the entire model.
LORA_TARGETS=${LORA_TARGETS:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-128}
LORA_DROPOUT=${LORA_DROPOUT:-0}
# Fixed implementation: BF16 base, FP32 adapters/states, zero B, bias=none, no DoRA/rsLoRA.
LEARNING_RATE=${LEARNING_RATE:-5e-6}
WEIGHT_DECAY=${WEIGHT_DECAY:-0}
ADAM_BETA1=${ADAM_BETA1:-0.9}
ADAM_BETA2=${ADAM_BETA2:-0.999}
ADAM_EPSILON=${ADAM_EPSILON:-1e-8}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}

# One optimizer step per global rollout batch (unless PPO_EPOCHS is explicitly increased).
GLOBAL_PROMPTS=${GLOBAL_PROMPTS:-32}
GROUP_SIZE=${GROUP_SIZE:-5}
ROLLOUT_PROMPT_BATCH=${ROLLOUT_PROMPT_BATCH:-1}
PROMPT_BATCH_MAX_PADDING_RATIO=${PROMPT_BATCH_MAX_PADDING_RATIO:-1.5}
ROLLOUT_MICRO_BATCH=${ROLLOUT_MICRO_BATCH:-5}
AUTO_ROLLOUT_BATCH=${AUTO_ROLLOUT_BATCH:-1} # generation OOM: 5 -> 2 -> 1; G remains unchanged
PPO_EPOCHS=${PPO_EPOCHS:-1}
EPOCHS=${EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:-0} # 0 = no step cap; epoch/time budget still applies
MAX_HOURS=${MAX_HOURS:-0} # 0 = no time cap; EPOCHS still applies
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
# v1 requires unwarped sampling so old/new log-probs refer to the actual sampling policy.
TASK_BALANCE_FRACTION=${TASK_BALANCE_FRACTION:-0.5}
SAMPLING_MODE=${SAMPLING_MODE:-coverage} # coverage: each row once per pass, every dataset/label first
MAX_RESAMPLE_GROUPS=${MAX_RESAMPLE_GROUPS:-0}

# HECATEPO: only gamma=1 has a specified singleton fallback in this implementation.
EMA_ALPHA=${EMA_ALPHA:-0.3}
ENTROPY_QUANTILE=${ENTROPY_QUANTILE:-0.60}
NMS_DISTANCE=${NMS_DISTANCE:-32}
ANCHOR_WINDOW=${ANCHOR_WINDOW:-8}
ANCHOR_THRESHOLD=${ANCHOR_THRESHOLD:-0.85}
GAMMA=${GAMMA:-1.0}
LAMBDA_BASE=${LAMBDA_BASE:-0.5}
LAMBDA_HIGH=${LAMBDA_HIGH:-0.5}
LAMBDA_LOW=${LAMBDA_LOW:-0.5}
CLIP_EPSILON=${CLIP_EPSILON:-0.2}
STD_EPSILON=${STD_EPSILON:-1e-6}
ADVANTAGE_MODE=${ADVANTAGE_MODE:-adaptive} # trajectory = HISPO control; fixed = constant lambda
RATIO_MODE=${RATIO_MODE:-segment} # token ablation: token-wise ratio and clip; keep segments/anchors/lambda
ANCHOR_MODE=${ANCHOR_MODE:-semantic} # random ablation: shuffle anchor-group membership, preserve sizes

# Input and output budgets. Video and audio share the same temporal crop, starting at t=0.
MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS:-4096}
TEXT_HEAD_TOKENS=${TEXT_HEAD_TOKENS:-128}
TEXT_TAIL_TOKENS=${TEXT_TAIL_TOKENS:-256}
TEXT_WINDOWS=${TEXT_WINDOWS:-4}
MAX_RESPONSE_TOKENS=${MAX_RESPONSE_TOKENS:-2048}
EVAL_RESPONSE_TOKENS=${EVAL_RESPONSE_TOKENS:-2048}
VIDEO_FRAMES=${VIDEO_FRAMES:-8}
VIDEO_SAMPLING=${VIDEO_SAMPLING:-segment_starts}
VIDEO_MAX_PIXELS=${VIDEO_MAX_PIXELS:-147456}
IMAGE_MAX_PIXELS=${IMAGE_MAX_PIXELS:-147456}
MEDIA_MAX_SECONDS=${MEDIA_MAX_SECONDS:-60} # 0 = full media; input-token overflow is an explicit error
AUDIO_MAX_SECONDS=${AUDIO_MAX_SECONDS:-$MEDIA_MAX_SECONDS}
VIDEO_MAX_SECONDS=${VIDEO_MAX_SECONDS:-$MEDIA_MAX_SECONDS}
VIDEO_MIN_PIXELS=${VIDEO_MIN_PIXELS:-3136}
MEDIA_CACHE_GROUPS=${MEDIA_CACHE_GROUPS:-1}
PREPARE_MEDIA=${PREPARE_MEDIA:-1}
MEDIA_PREPARE_WORKERS=${MEDIA_PREPARE_WORKERS:-4}
ENTROPY_CHUNK=${ENTROPY_CHUNK:-32}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2} # sdpa is a supported fallback
NUM_THREADS=${NUM_THREADS:-4}
MIN_FREE_GB=${MIN_FREE_GB:-28}

# Same task-aware terminal reward as the initial design, with real response token lengths.
TASK_REWARD_WEIGHT=${TASK_REWARD_WEIGHT:-0.8}
FORMAT_REWARD_WEIGHT=${FORMAT_REWARD_WEIGHT:-0.2}
LENGTH_REWARD_WEIGHT=${LENGTH_REWARD_WEIGHT:-0.75}
LENGTH_LIMIT=${LENGTH_LIMIT:-812}
LENGTH_BUFFER=${LENGTH_BUFFER:-128}

EVAL_EVERY=${EVAL_EVERY:-25}
SAVE_EVERY=${SAVE_EVERY:-25}
EVAL_PER_DATASET=${EVAL_PER_DATASET:-8} # 0 = full split; test requires 0; QA proxy is not final judge accuracy
QA_HOLDOUT_GROUPS=${QA_HOLDOUT_GROUPS:-32} # independent media groups, only for QA tasks missing from official validation
QA_HOLDOUT_FRACTION=${QA_HOLDOUT_FRACTION:-0.1} # cap at ceil(10% of available media groups)
KEEP_CHECKPOINTS=${KEEP_CHECKPOINTS:-3}
LOG_RESPONSES_EVERY=${LOG_RESPONSES_EVERY:-25}
SMOKE_PER_MODALITY=${SMOKE_PER_MODALITY:-1}
SMOKE_LONGEST=${SMOKE_LONGEST:-1}
TUNE_ROLLOUT=${TUNE_ROLLOUT:-0}
RESUME=${RESUME:-}
ADAPTER=${ADAPTER:-}
ADAPTER_SCALE=${ADAPTER_SCALE:-1.0}
EVAL_SPLIT=${EVAL_SPLIT:-validation}

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$ROOT/cache/huggingface"
export XDG_CACHE_HOME="$ROOT/cache"
export TMPDIR="$ROOT/cache/tmp"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$NUM_THREADS"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
mkdir -p "$ROOT/logs" "$TMPDIR" "$OUTPUT_ROOT" "$WEIGHTS_ROOT" "$INDEX_ROOT"
cd "$ROOT"
ARGS=(--mode "$MODE" --model "$MODEL_PATH" --data "$DATA_PATH" --output "$OUTPUT_ROOT"
 --weights "$WEIGHTS_ROOT" --index "$INDEX_ROOT" --reward-model "$REWARD_MODEL" --run-name "$RUN_NAME"
 --media-cache "$MEDIA_CACHE"
 --targets "$LORA_TARGETS" --seed "$SEED" --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA"
 --lora-dropout "$LORA_DROPOUT" --learning-rate "$LEARNING_RATE" --weight-decay "$WEIGHT_DECAY"
 --adam-beta1 "$ADAM_BETA1" --adam-beta2 "$ADAM_BETA2" --adam-epsilon "$ADAM_EPSILON"
 --warmup-ratio "$WARMUP_RATIO" --max-grad-norm "$MAX_GRAD_NORM"
 --global-prompts "$GLOBAL_PROMPTS" --group-size "$GROUP_SIZE" --rollout-micro-batch "$ROLLOUT_MICRO_BATCH"
 --prompt-batch-max-padding-ratio "$PROMPT_BATCH_MAX_PADDING_RATIO" --rollout-prompt-batch "$ROLLOUT_PROMPT_BATCH" --auto-rollout-batch "$AUTO_ROLLOUT_BATCH" --ppo-epochs "$PPO_EPOCHS" --epochs "$EPOCHS" --max-steps "$MAX_STEPS" --max-hours "$MAX_HOURS"
 --temperature "$TEMPERATURE" --top-p "$TOP_P" --task-balance-fraction "$TASK_BALANCE_FRACTION"
 --max-resample-groups "$MAX_RESAMPLE_GROUPS" --sampling-mode "$SAMPLING_MODE"
 --ema-alpha "$EMA_ALPHA" --entropy-quantile "$ENTROPY_QUANTILE" --nms-distance "$NMS_DISTANCE"
 --anchor-window "$ANCHOR_WINDOW" --anchor-threshold "$ANCHOR_THRESHOLD" --gamma "$GAMMA"
 --lambda-base "$LAMBDA_BASE" --lambda-high "$LAMBDA_HIGH" --lambda-low "$LAMBDA_LOW"
 --clip-epsilon "$CLIP_EPSILON" --std-epsilon "$STD_EPSILON" --advantage-mode "$ADVANTAGE_MODE" --ratio-mode "$RATIO_MODE" --anchor-mode "$ANCHOR_MODE"
 --max-prompt-tokens "$MAX_PROMPT_TOKENS" --max-response-tokens "$MAX_RESPONSE_TOKENS"
 --text-head-tokens "$TEXT_HEAD_TOKENS" --text-tail-tokens "$TEXT_TAIL_TOKENS" --text-windows "$TEXT_WINDOWS"
 --eval-response-tokens "$EVAL_RESPONSE_TOKENS" --video-frames "$VIDEO_FRAMES"
 --video-sampling "$VIDEO_SAMPLING" --video-max-pixels "$VIDEO_MAX_PIXELS" --image-max-pixels "$IMAGE_MAX_PIXELS"
 --audio-max-seconds "$AUDIO_MAX_SECONDS" --video-max-seconds "$VIDEO_MAX_SECONDS" --video-min-pixels "$VIDEO_MIN_PIXELS"
 --media-max-seconds "$MEDIA_MAX_SECONDS" --media-cache-groups "$MEDIA_CACHE_GROUPS"
 --prepare-media "$PREPARE_MEDIA" --media-prepare-workers "$MEDIA_PREPARE_WORKERS"
 --entropy-chunk "$ENTROPY_CHUNK" --attn-implementation "$ATTN_IMPLEMENTATION" --num-threads "$NUM_THREADS"
 --min-free-gb "$MIN_FREE_GB" --task-reward-weight "$TASK_REWARD_WEIGHT"
 --format-reward-weight "$FORMAT_REWARD_WEIGHT" --length-reward-weight "$LENGTH_REWARD_WEIGHT"
 --length-limit "$LENGTH_LIMIT" --length-buffer "$LENGTH_BUFFER" --eval-every "$EVAL_EVERY"
 --save-every "$SAVE_EVERY" --eval-per-dataset "$EVAL_PER_DATASET" --keep-checkpoints "$KEEP_CHECKPOINTS"
 --qa-holdout-groups "$QA_HOLDOUT_GROUPS" --qa-holdout-fraction "$QA_HOLDOUT_FRACTION"
 --tune-rollout "$TUNE_ROLLOUT" --smoke-longest "$SMOKE_LONGEST" --log-responses-every "$LOG_RESPONSES_EVERY" --smoke-per-modality "$SMOKE_PER_MODALITY"
 --resume "$RESUME" --adapter "$ADAPTER" --adapter-scale "$ADAPTER_SCALE" --eval-split "$EVAL_SPLIT")
if [ "$NUM_PROCESSES" -gt 1 ]; then
 "$ENV_DIR/bin/python" -m torch.distributed.run --standalone --nproc_per_node="$NUM_PROCESSES" \
   -m hecatepo.train "${ARGS[@]}" "$@" 2>&1 | tee "$ROOT/logs/$RUN_NAME.log"
else
 "$ENV_DIR/bin/python" -u -m hecatepo.train "${ARGS[@]}" "$@" 2>&1 | tee "$ROOT/logs/$RUN_NAME.log"
fi
