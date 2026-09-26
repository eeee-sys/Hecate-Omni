import argparse
from pathlib import Path
from .core import AlgorithmConfig


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="HECATEPO (all run settings are exposed in scripts/run_hecatepo.sh)")
    parser.add_argument("--mode", choices=["prepare", "smoke", "train", "evaluate"], required=True)
    for flag in ("model", "data", "output", "weights", "index", "reward-model", "run-name", "media-cache"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--eval-split", choices=["validation", "test"], default="validation")
    parser.add_argument("--targets", required=True)
    parser.add_argument("--attn-implementation", choices=["sdpa", "flash_attention_2", "eager"], required=True)
    parser.add_argument("--advantage-mode", choices=["adaptive", "trajectory", "fixed"], required=True)
    parser.add_argument("--ratio-mode", choices=["segment", "token"], default="segment")
    parser.add_argument("--anchor-mode", choices=["semantic", "random"], default="semantic")
    parser.add_argument("--sampling-mode", choices=["mixture", "coverage"], default="mixture")
    parser.add_argument("--auto-rollout-batch", type=int, choices=[0, 1], default=1)
    parser.add_argument("--smoke-longest", type=int, choices=[0, 1], default=1)
    parser.add_argument("--prompt-batch-max-padding-ratio", type=float, default=1.5)
    parser.add_argument("--tune-rollout", type=int, choices=[0, 1], default=0)
    parser.add_argument("--rollout-prompt-batch", type=int, default=1)
    parser.add_argument("--audio-max-seconds", type=float, default=None)
    parser.add_argument("--video-max-seconds", type=float, default=None)
    parser.add_argument("--video-min-pixels", type=int, default=3136)
    parser.add_argument("--video-sampling", choices=["segment_starts", "uniform_endpoints"], default="segment_starts")
    integer_flags = ["seed", "lora-rank", "lora-alpha", "global-prompts", "group-size", "rollout-micro-batch",
                     "max-steps", "ppo-epochs", "max-prompt-tokens", "max-response-tokens", "eval-response-tokens",
                     "video-frames", "video-max-pixels", "image-max-pixels", "media-cache-groups", "entropy-chunk",
                     "nms-distance", "anchor-window", "length-limit", "length-buffer", "eval-every", "save-every",
                     "eval-per-dataset", "keep-checkpoints", "num-threads", "smoke-per-modality", "min-free-gb",
                     "max-resample-groups", "log-responses-every", "qa-holdout-groups",
                     "text-head-tokens", "text-tail-tokens", "text-windows", "media-prepare-workers", "prepare-media"]
    float_flags = ["lora-dropout", "learning-rate", "weight-decay", "adam-beta1", "adam-beta2", "adam-epsilon",
                   "warmup-ratio", "max-grad-norm", "max-hours", "epochs", "temperature", "top-p", "media-max-seconds",
                   "ema-alpha", "entropy-quantile", "anchor-threshold", "gamma", "lambda-base", "lambda-high",
                   "lambda-low", "clip-epsilon", "std-epsilon", "task-reward-weight", "format-reward-weight",
                   "length-reward-weight", "task-balance-fraction", "adapter-scale", "qa-holdout-fraction"]
    for flag in integer_flags:
        parser.add_argument("--" + flag, type=int, required=True)
    for flag in float_flags:
        parser.add_argument("--" + flag, type=float, required=True)
    cfg = parser.parse_args(argv)
    if cfg.mode == "evaluate" and cfg.eval_split == "test" and cfg.eval_per_dataset != 0:
        parser.error("Official test evaluation requires EVAL_PER_DATASET=0 (all test samples); test subsampling is disabled")
    if cfg.audio_max_seconds is None:
        cfg.audio_max_seconds = cfg.media_max_seconds
    if cfg.video_max_seconds is None:
        cfg.video_max_seconds = cfg.media_max_seconds
    if cfg.prompt_batch_max_padding_ratio < 1:
        parser.error("Padding ratio must be >= 1")
    if cfg.rollout_prompt_batch < 1 or min(cfg.audio_max_seconds, cfg.video_max_seconds) < 0:
        parser.error("Prompt batch must be positive and media limits nonnegative")
    if not 0 < cfg.video_min_pixels <= cfg.video_max_pixels:
        parser.error("Require 0 < video_min_pixels <= video_max_pixels")
    algorithm_config(cfg).validate()
    if cfg.max_steps < 0 or cfg.max_hours < 0 or cfg.epochs <= 0:
        parser.error("max_steps=0 and max_hours=0 disable their limits; epochs must be positive")
    if cfg.group_size < 2 or min(cfg.global_prompts, cfg.rollout_micro_batch, cfg.ppo_epochs, cfg.entropy_chunk) < 1:
        parser.error("Positive batch settings and G>=2 required")
    if cfg.lora_dropout != 0:
        parser.error("v1 requires dropout=0 for matching old/new policy probabilities")
    if cfg.temperature != 1.0 or cfg.top_p != 1.0:
        parser.error("v1 samples the unwarped policy (temperature=1, top_p=1); no off-policy correction is implemented")
    if cfg.gamma != 1.0:
        parser.error("v1 supports gamma=1 only")
    if not 0 <= cfg.task_balance_fraction <= 1 or not 0 <= cfg.adapter_scale <= 1:
        parser.error("Invalid sampling mixture or adapter scale")
    if not 0 < cfg.qa_holdout_fraction < 1 or cfg.qa_holdout_groups < 0:
        parser.error("QA holdout requires 0<fraction<1 and nonnegative groups")
    if cfg.video_frames < 2 or cfg.video_frames % 2:
        parser.error("Qwen video frames must be a positive even count >=2")
    if cfg.mode == "train" and cfg.adapter_scale != 1:
        parser.error("Adapter scaling is evaluation-only")
    if cfg.mode == "train" and cfg.adapter:
        parser.error("Use --resume for training state, not --adapter")
    if cfg.mode != "evaluate" and cfg.eval_split != "validation":
        parser.error("Test split cannot be used during training or development")
    cfg.targets = cfg.targets.split(",")
    for field in ("model", "data", "output", "weights", "index", "reward_model", "media_cache"):
        setattr(cfg, field, str(Path(getattr(cfg, field)).expanduser().resolve()))
    return cfg


def algorithm_config(cfg):
    return AlgorithmConfig(**{key: getattr(cfg, key) for key in AlgorithmConfig.__dataclass_fields__})


def training_budget(row_count, epochs, global_prompts, max_steps):
    import math
    prompts = math.ceil(row_count * epochs)
    epoch_steps = math.ceil(prompts / global_prompts)
    return prompts, min(epoch_steps, max_steps) if max_steps > 0 else epoch_steps


def time_limit_reached(elapsed_seconds, max_hours):
    return max_hours > 0 and elapsed_seconds >= max_hours * 3600
