import json
import hashlib
import math
import os
import random
import shutil
import time
import traceback
from collections import defaultdict, Counter
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .config import parse_args, algorithm_config, training_budget, time_limit_reached
from .core import build_segments, group_diagnostics, segment_loss
from .data import InputBuilder, PromptSampler, atomic_json, load_index, stratified_rows, training_validation_split
from .evaluation import summarize_predictions
from .model import load_model, generate_group, old_policy_stats, current_log_probs, rollout_rows
from .rewards import RewardEngine


def emit(path, record, display=True):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    if display:
        print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)


def anchor_group_seed(seed, uid):
    return int.from_bytes(hashlib.sha256(f"{seed}:{uid}".encode()).digest()[:8], "big")


def sample_error_record(row, stage, error, trace, step=None, rank=0):
    return {"event": "sample_skipped", "step": step, "rank": rank,
            "uid": row.get("uid"), "dataset": row.get("dataset"),
            "task_group": row.get("task_group"),
            "coverage_stratum": "|".join(PromptSampler.stratum(row)),
            "stage": stage, "exception_type": type(error).__name__,
            "exception": str(error), "traceback": trace}


def distributed_info():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank, local_rank = int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
    torch.cuda.set_device(local_rank)
    return rank, world, torch.device("cuda", local_rank)


def synchronize_gradients(model, world):
    # Each loss is divided by the global response count before backward; SUM is correct.
    if world > 1:
        for parameter in model.parameters():
            if parameter.requires_grad:
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state(state["cuda"].cpu())


def save_checkpoint(model, processor, optimizer, scheduler, sampler, cfg, step, best, rank, world, is_best=False):
    root = Path(cfg.weights) / cfg.run_name
    path = root / f"step-{step:06d}"
    path.mkdir(parents=True, exist_ok=True)
    torch.save(rng_state(), path / f"rng-rank-{rank}.pt")
    if rank == 0:
        previous_best = json.loads((root / "best.json").read_text())["checkpoint"] if (root / "best.json").exists() else None
        model.save_pretrained(path, safe_serialization=True)
        processor.save_pretrained(path)
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "sampler": sampler.state_dict(),
                    "rollout_micro_batch": getattr(model, "_sagepo_rollout_micro_batch", getattr(cfg, "rollout_micro_batch", 1)),
                    "rollout_prompt_batch": getattr(model, "_sagepo_prompt_batch", getattr(cfg, "rollout_prompt_batch", 1)),
                    "step": step, "best": best, "world_size": world,
                    "best_checkpoint": str(path) if is_best else previous_best},
                   path / "training_state.pt")
        atomic_json(path / "run_config.json", vars(cfg))
        if is_best:
            atomic_json(root / "best.json", {"checkpoint": str(path), "step": step,
                                           "selection_proxy_macro": best, "formal_qa_judging_pending": True})
        best_file = root / "best.json"
        protected = json.loads(best_file.read_text())["checkpoint"] if best_file.exists() else None
        checkpoints = sorted(root.glob("step-*"))
        removable = [p for p in checkpoints[:-cfg.keep_checkpoints] if str(p) != protected] if cfg.keep_checkpoints > 0 else []
        for stale in removable:
            shutil.rmtree(stale)
    if world > 1:
        dist.barrier()
    if rank == 0:
        (path / "COMPLETE").write_text("complete\n")
    return path


@torch.no_grad()
def evaluate(model, processor, builder, reward, rows, cfg, device, output_path):
    model.eval()
    records, skipped = [], 0
    errors_path = Path(str(output_path) + ".errors.jsonl")
    with Path(output_path).open("w") as handle:
        for index, row in enumerate(rows):
            try:
                inputs, media = builder.build(row)
                response = generate_group(model, processor, inputs, 1, cfg, device, greedy=True)[0]
                score = reward.score(row, [response["text"]], [len(response["ids"])])[0]
                record = {"uid": row["uid"], "dataset": row["dataset"], "task_group": row["task_group"],
                          "question": row["problem"], "reference": row["answer"], "text": response["text"],
                          "terminated": response["terminated"], "media": media, **score}
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                records.append(record)
            except Exception as error:
                skipped += 1
                emit(errors_path, sample_error_record(row, "evaluation", error, traceback.format_exc()), False)
                print(json.dumps({"event": "evaluation_sample_skipped", "uid": row.get("uid"),
                                  "dataset": row.get("dataset"), "exception_type": type(error).__name__,
                                  "exception": str(error)}, ensure_ascii=False), flush=True)
                if isinstance(error, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
            if (index + 1) % 8 == 0:
                print(f"Evaluation {index + 1}/{len(rows)}", flush=True)
    result = summarize_predictions(records)
    result.update({"attempted_samples": len(rows), "evaluated_samples": len(records),
                   "skipped_samples": skipped})
    atomic_json(str(output_path) + ".metrics.json", result)
    return result


def smoke_rows(rows, per_modality, reader):
    groups = defaultdict(list)
    for row in rows:
        # Include real TV samples even when their declared signature says TAV.
        entry = reader.manifest(row)[row["row_offset"]]
        signature = "T" + ("A" if entry["audios"] else "") + ("V" if entry["videos"] else "")
        groups[signature].append(row)
    selected = []
    for signature in ("T", "TA", "TV", "TAV"):
        group = groups[signature]
        # Include a QA example in the audio-video slot to exercise the reward model.
        if signature == "TAV":
            group.sort(key=lambda r: (r["reward_type"] != "qa", len(r["problem"])))
        selected.extend(group[:per_modality])
    return selected


def run_smoke(model, processor, builder, reward, rows, cfg, device, output):
    algorithm = algorithm_config(cfg)
    report = []
    selected = smoke_rows(rows, cfg.smoke_per_modality, builder.reader)
    if cfg.smoke_longest and rows:
        longest = max(rows, key=lambda row: len(row["problem"]))
        if all(row["uid"] != longest["uid"] for row in selected):
            selected.append(longest)
    if not selected:
        raise ValueError("No smoke samples")
    if cfg.tune_rollout:
        from .benchmark import tune_rollout
        benchmark_rows = list(selected)
        used = {r["uid"] for r in benchmark_rows}
        for candidate in rows:
            if len(benchmark_rows) >= 8:
                break
            if candidate["uid"] not in used:
                benchmark_rows.append(candidate)
                used.add(candidate["uid"])
        tune_rollout(model, processor, builder, benchmark_rows, cfg, device, output)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.learning_rate)
    for sample_index, row in enumerate(selected):
        start = time.monotonic()
        inputs, media = builder.build(row)
        responses = generate_group(model, processor, inputs, cfg.group_size, cfg, device)
        stats = [old_policy_stats(model, inputs, r["ids"], cfg, device) for r in responses]
        initial_base_error = None
        if sample_index == 0 and not cfg.resume:
            with model.disable_adapter():
                base_stats = old_policy_stats(model, inputs, responses[0]["ids"], cfg, device)
            initial_base_error = float((base_stats.log_probs - stats[0].log_probs).abs().max())
            if initial_base_error != 0:
                raise AssertionError("Fresh zero-B LoRA changed base model output")
        scores = reward.score(row, [r["text"] for r in responses], [len(r["ids"]) for r in responses])
        segments = build_segments(stats, [r["reward"] for r in scores], algorithm,
                                  anchor_group_seed(cfg.seed, row["uid"]))
        sweep = []
        for distance in (16, 32, 64):
            for threshold in (0.80, 0.85, 0.90):
                probe = build_segments(stats, [r["reward"] for r in scores],
                                       replace(algorithm, nms_distance=distance, anchor_threshold=threshold),
                                       anchor_group_seed(cfg.seed, row["uid"]))
                sweep.append({"nms_distance": distance, "anchor_threshold": threshold, **group_diagnostics(probe)})
        emit(output / "segmentation_probe.jsonl", {"dataset": row["dataset"], "probes": sweep}, False)
        optimizer.zero_grad(set_to_none=True)
        model.train()
        maximum_difference = 0.0
        losses = []
        for response, old, parts in zip(responses, stats, segments):
            new_log_probs = current_log_probs(model, inputs, response["ids"], cfg, device)
            maximum_difference = max(maximum_difference, float((new_log_probs.detach().cpu() - old.log_probs).abs().max()))
            loss, _ = segment_loss(new_log_probs, old.log_probs, parts, cfg.clip_epsilon, cfg.ratio_mode)
            (loss / cfg.group_size).backward()
            losses.append(float(loss.detach()))
        if maximum_difference > 0.05:
            raise AssertionError(f"Old/new policy mismatch before update: {maximum_difference}")
        norm = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.max_grad_norm))
        if not math.isfinite(norm):
            raise FloatingPointError("Nonfinite smoke gradient")
        # Exercise the actual optimizer only when the real rewards create an update.
        changed = False
        if norm > 0:
            parameter = next(p for p in model.parameters() if p.requires_grad and p.grad is not None and p.grad.abs().max() > 0)
            before = parameter.detach().clone()
            optimizer.step()
            changed = bool(torch.any(before != parameter))
        model.eval()
        result = {"uid": row["uid"], "dataset": row["dataset"], "modality": media["effective_modality"],
                  "media": media, "max_old_new_logp_error": maximum_difference, "gradient_norm": norm,
                  "initial_adapter_base_logp_error": initial_base_error,
                  "adapter_updated": changed, "reward": [r["reward"] for r in scores],
                  "response_tokens": [len(r["ids"]) for r in responses],
                  "seconds": time.monotonic() - start,
                  "gpu_peak_gb": torch.cuda.max_memory_allocated() / 1024**3,
                  "rollout_micro_batch": getattr(model, "_sagepo_rollout_micro_batch", cfg.rollout_micro_batch),
                  **group_diagnostics(segments)}
        report.append(result)
        emit(output / "smoke.jsonl", result)
        emit(output / "smoke_responses.jsonl", {"uid": row["uid"], "responses": [r["text"] for r in responses]}, False)
    atomic_json(output / "smoke_summary.json", {"passed": True, "samples": report,
                "note": "Smoke updates are discarded; no production model is saved."})


def main(argv=None):
    cfg = parse_args(argv)
    Path(cfg.output).mkdir(parents=True, exist_ok=True)
    Path(cfg.index).mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg.num_threads)
    output = Path(cfg.output) / cfg.run_name
    output.mkdir(parents=True, exist_ok=True)
    if cfg.mode == "prepare":
        splits = {}
        for split in ("train", "validation"):
            split_rows, summary = load_index(cfg.data, split, cfg.index)
            splits[split] = (split_rows, summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
        if cfg.prepare_media:
            from .cache_media import warm_cache
            audit = warm_cache(splits["train"][0] + splits["validation"][0], cfg.media_cache, cfg.media_prepare_workers)
            atomic_json(Path(cfg.index) / "train_validation_media_audit.json", audit)
        _, _, holdout = training_validation_split(splits["train"][0], splits["validation"][0], cfg.index,
                                                  splits["train"][1]["fingerprint"], cfg.qa_holdout_groups, cfg.seed,
                                                  cfg.media_cache, cfg.qa_holdout_fraction)
        print(json.dumps({k: v for k, v in holdout.items() if k != "excluded_train_uids"}, ensure_ascii=False), flush=True)
        return
    rank, world, device = distributed_info()
    if cfg.mode in {"smoke", "evaluate"} and world != 1:
        raise ValueError("Use one GPU for smoke/evaluation; training supports torchrun replication")
    free_gb = torch.cuda.mem_get_info(device)[0] / 1024**3
    if free_gb < cfg.min_free_gb:
        raise RuntimeError(f"Only {free_gb:.1f} GiB free; need >= {cfg.min_free_gb} GiB; select another GPU")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    if rank == 0:
        atomic_json(output / "config.json", vars(cfg))
    if cfg.mode == "evaluate" and cfg.eval_split == "test":
        rows, summary = load_index(cfg.data, cfg.eval_split, cfg.index)
    else:
        rows, summary = load_index(cfg.data, "train", cfg.index)
        val_rows, val_summary = load_index(cfg.data, "validation", cfg.index)
        # Rank zero writes the media identity cache first; other ranks reuse it.
        if rank != 0 and world > 1:
            dist.barrier()
        rows, val_rows, holdout = training_validation_split(rows, val_rows, cfg.index, summary["fingerprint"],
                                                           cfg.qa_holdout_groups, cfg.seed, cfg.media_cache, cfg.qa_holdout_fraction)
        if rank == 0 and world > 1:
            dist.barrier()
        cfg.training_fingerprint = summary["fingerprint"]
        cfg.validation_fingerprint = val_summary["fingerprint"]
        if rank == 0:
            atomic_json(output / "qa_holdout.json", holdout)
        summary = {**summary, "effective_train_count": len(rows),
                   "qa_held_out_count": holdout["held_out_rows"]}
        if cfg.mode == "evaluate":
            rows = val_rows
            summary = {**summary, "split": "validation", "evaluation_rows": len(rows)}
    if rank == 0:
        atomic_json(output / "dataset_summary.json", summary)
        atomic_json(output / "config.json", vars(cfg))
    if cfg.mode == "train":
        existing = list((Path(cfg.weights) / cfg.run_name).glob("step-*/COMPLETE"))
        if existing and not cfg.resume:
            raise ValueError("RUN_NAME already has checkpoints; use RESUME or choose a new RUN_NAME")
    model, processor, parameters = load_model(cfg, device)
    if rank == 0:
        atomic_json(output / "trainable_parameters.json", parameters)
        print(f"Loaded {parameters['trainable_count']:,} trainable / {parameters['total_count']:,} total parameters", flush=True)
    # Keep initialization identical across ranks, then use different sampling RNG streams.
    torch.manual_seed(cfg.seed + rank)
    torch.cuda.manual_seed_all(cfg.seed + rank)
    builder, reward = InputBuilder(processor, cfg), RewardEngine(cfg)
    if cfg.mode == "evaluate":
        selected = stratified_rows(rows, cfg.eval_per_dataset, cfg.seed)
        result = evaluate(model, processor, builder, reward, selected, cfg, device, output / "predictions.jsonl")
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return
    if cfg.mode == "smoke":
        run_smoke(model, processor, builder, reward, rows, cfg, device, output)
        return
    if cfg.global_prompts < world:
        raise ValueError("global_prompts must be >= world_size")
    if rank == 0:
        val_rows = stratified_rows(val_rows, cfg.eval_per_dataset, cfg.seed)
        atomic_json(output / "validation_ids.json", [r["uid"] for r in val_rows])
    sampler = PromptSampler(rows, cfg.seed, cfg.task_balance_fraction, cfg.sampling_mode)
    coverage = Counter()
    active_coverage = Counter()
    skipped_coverage = Counter()
    expected_strata = {"|".join(PromptSampler.stratum(r)) for r in rows}
    algorithm = algorithm_config(cfg)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
                                 betas=(cfg.adam_beta1, cfg.adam_beta2), eps=cfg.adam_epsilon)
    max_prompts, steps = training_budget(len(rows), cfg.epochs, cfg.global_prompts, cfg.max_steps)
    if rank == 0:
        atomic_json(output / "training_plan.json", {"sampling_mode": cfg.sampling_mode,
            "training_rows": len(rows), "epoch_prompt_budget": max_prompts, "planned_batches": steps,
            "max_steps": cfg.max_steps, "max_hours": cfg.max_hours,
            "coverage_prefix_prompts": len(expected_strata) if cfg.sampling_mode == "coverage" else None,
            "expected_strata": sorted(expected_strata)})
    warmup = max(1, math.ceil(steps * cfg.ppo_epochs * cfg.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min((s + 1) / warmup, 1.0))
    step, best = 0, -math.inf
    if cfg.resume:
        checkpoint_dir = Path(cfg.resume)
        if not (checkpoint_dir / "COMPLETE").exists():
            raise ValueError("Cannot resume an incomplete checkpoint")
        saved_config = json.loads((checkpoint_dir / "run_config.json").read_text())
        # Explicit legacy defaults preserve old single-question checkpoint compatibility.
        for key, value in {"rollout_prompt_batch": 1, "tune_rollout": 0, "prompt_batch_max_padding_ratio": 1.5,
                "audio_max_seconds": saved_config.get("media_max_seconds", 60),
                "video_max_seconds": saved_config.get("media_max_seconds", 60),
                "video_min_pixels": 3136, "video_sampling": "segment_starts", "ratio_mode": "segment",
                "anchor_mode": "semantic"}.items():
            saved_config.setdefault(key, value)
        mutable = {"resume", "run_name", "output", "weights", "index", "max_hours", "eval_every", "save_every",
                   "keep_checkpoints", "log_responses_every", "min_free_gb", "num_threads", "smoke_per_modality"}
        differences = [key for key, value in vars(cfg).items()
                       if key not in mutable and saved_config.get(key) != value]
        if differences:
            raise ValueError(f"Resume configuration differs in {differences}; restore original training settings")
        state = torch.load(checkpoint_dir / "training_state.pt", map_location="cpu", weights_only=False)
        if state["world_size"] != world:
            raise ValueError("Exact resume requires the same world size")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        sampler.load_state_dict(state["sampler"])
        model._sagepo_rollout_micro_batch = state.get("rollout_micro_batch", cfg.rollout_micro_batch)
        model._sagepo_prompt_batch = state.get("rollout_prompt_batch", cfg.rollout_prompt_batch)
        step, best = state["step"], state["best"]
        if any(int(p.parent.name.split("-")[1]) > step for p in existing):
            raise ValueError("RUN_NAME contains later checkpoints; choose a new RUN_NAME to branch from an older checkpoint")
        if rank == 0 and state.get("best_checkpoint"):
            atomic_json(Path(cfg.weights) / cfg.run_name / "best.json",
                        {"checkpoint": state["best_checkpoint"], "selection_proxy_macro": best,
                         "formal_qa_judging_pending": True})
        restore_rng(torch.load(checkpoint_dir / f"rng-rank-{rank}.pt", map_location="cpu", weights_only=False))
    first_update_step = step + 1
    if cfg.resume and rank == 0:
        emit(output / "resume.jsonl", {"event": "checkpoint_restored", "checkpoint": cfg.resume,
             "restored_step": step, "max_hours": cfg.max_hours, "planned_batches": steps})
    started, last_saved = time.monotonic(), -1
    metrics_file = output / f"train_rank{rank}.jsonl"
    errors_file = output / f"skipped_samples_rank{rank}.jsonl"
    while step < steps:
        tick = time.monotonic()
        count = min(cfg.global_prompts, max_prompts - step * cfg.global_prompts)
        batch = sampler.take(count)
        local_rows = batch[rank::world]
        pending, diagnostics, batch_errors = [], [], []
        def record_skip(row, stage, error, trace):
            record = sample_error_record(row, stage, error, trace, step + 1, rank)
            batch_errors.append(record)
            emit(errors_file, record, False)
            print(json.dumps({key: record[key] for key in ("event", "step", "rank", "uid", "dataset",
                                                            "stage", "exception_type", "exception")},
                             ensure_ascii=False), flush=True)
            skipped_coverage[record["coverage_stratum"]] += 1
            if isinstance(error, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
        for row, inputs, media, responses in rollout_rows(
                model, processor, builder, local_rows, cfg, device, record_skip):
            try:
                scores = reward.score(row, [r["text"] for r in responses], [len(r["ids"]) for r in responses])
                for retry in range(cfg.max_resample_groups):
                    if np.std([s["reward"] for s in scores]) > cfg.std_epsilon:
                        break
                    responses = generate_group(model, processor, inputs, cfg.group_size, cfg, device)
                    scores = reward.score(row, [r["text"] for r in responses], [len(r["ids"]) for r in responses])
                equal_rewards = len({s["reward"] for s in scores}) == 1
                if equal_rewards:
                    # gamma=1 makes both levels of advantage exactly zero. Avoid
                    # expensive entropy/hidden-state passes with an identically zero loss.
                    group_info = {"skipped_identical_rewards": True}
                else:
                    stats = [old_policy_stats(model, inputs, r["ids"], cfg, device) for r in responses]
                    parts = build_segments(stats, [s["reward"] for s in scores], algorithm,
                                           anchor_group_seed(cfg.seed, row["uid"]))
                    pending.append((inputs, responses, stats, parts))
                    group_info = {"skipped_identical_rewards": False, **group_diagnostics(parts)}
                diagnostic = {"uid": row["uid"], "dataset": row["dataset"],
                              "coverage_stratum": "|".join(PromptSampler.stratum(row)), "media": media,
                              "reward_mean": float(np.mean([s["reward"] for s in scores])),
                              "reward_std": float(np.std([s["reward"] for s in scores], ddof=1)),
                              "task_reward_mean": float(np.mean([s["task_reward"] for s in scores])),
                              "format_rate": float(np.mean([s["format_reward"] for s in scores])),
                              "response_tokens": float(np.mean([len(r["ids"]) for r in responses])),
                              "truncation_rate": float(np.mean([not r["terminated"] for r in responses])),
                              **group_info}
                diagnostics.append(diagnostic)
            except Exception as error:
                record_skip(row, "reward_or_old_policy", error, traceback.format_exc())
                continue
            atomic_json(output / f"rollout_progress_rank{rank}.json", {"step": step + 1,
                "completed_prompts": len(diagnostics), "batch_prompts": len(local_rows),
                "dataset": row["dataset"], "seconds": time.monotonic() - tick,
                "last_reward_std": diagnostic["reward_std"], "last_group": group_info,
                "rollout_micro_batch": getattr(model, "_sagepo_rollout_micro_batch", cfg.rollout_micro_batch),
                "rollout_prompt_batch": getattr(model, "_sagepo_prompt_batch", cfg.rollout_prompt_batch)})
            if cfg.log_responses_every > 0 and (step + 1) % cfg.log_responses_every == 0:
                emit(output / f"responses_rank{rank}.jsonl", {"step": step + 1, "uid": row["uid"],
                     "responses": [r["text"] for r in responses], "rewards": scores}, False)
        rollout_seconds = time.monotonic() - tick
        model.train()
        losses, clips = [], []
        processed_prompts = torch.tensor(len(diagnostics), device=device)
        active_groups = torch.tensor(len(pending), device=device)
        if world > 1:
            dist.all_reduce(processed_prompts, op=dist.ReduceOp.SUM)
            dist.all_reduce(active_groups, op=dist.ReduceOp.SUM)
        processed_count = int(processed_prompts)
        active = bool(active_groups > 0)
        for _ in range(cfg.ppo_epochs):
            optimizer.zero_grad(set_to_none=True)
            for inputs, responses, stats, parts in pending:
                for response, old, segments in zip(responses, stats, parts):
                    # Skip exactly zero policy loss; do not invent reward differences.
                    if all(s.advantage == 0 for s in segments):
                        continue
                    new_log_probs = current_log_probs(model, inputs, response["ids"], cfg, device)
                    loss, info = segment_loss(new_log_probs, old.log_probs, segments, cfg.clip_epsilon, cfg.ratio_mode)
                    (loss / (processed_count * cfg.group_size)).backward()
                    losses.append(float(loss.detach()))
                    clips.append(info["clip_fraction"])
            if not active:
                norm = 0.0
                continue
            synchronize_gradients(model, world)
            norm = float(torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm))
            if not math.isfinite(norm):
                raise FloatingPointError("Nonfinite gradient norm; update aborted")
            optimizer.step()
            scheduler.step()
        model.eval()
        step += 1
        record = {"step": step, "global_prompts": count, "local_prompts": len(local_rows),
                  "processed_global_prompts": processed_count,
                  "processed_local_prompts": len(diagnostics), "skipped_local_prompts": len(batch_errors),
                  "active_global_groups": int(active_groups), "optimizer_updated": active,
                  "loss_mean": float(np.mean(losses)) if losses else 0.0, "grad_norm": norm,
                  "clip_fraction": float(np.mean(clips)) if clips else 0.0,
                  "rollout_seconds": rollout_seconds, "step_seconds": time.monotonic() - tick,
                  "lr": optimizer.param_groups[0]["lr"],
                  "rollout_prompt_batch": getattr(model, "_sagepo_prompt_batch", cfg.rollout_prompt_batch),
                  "rollout_micro_batch": getattr(model, "_sagepo_rollout_micro_batch", cfg.rollout_micro_batch),
                  "gpu_peak_gb": torch.cuda.max_memory_allocated() / 1024**3, "groups": diagnostics}
        emit(metrics_file, record)
        for item in diagnostics:
            coverage[item["coverage_stratum"]] += 1
            if not item["skipped_identical_rewards"]:
                active_coverage[item["coverage_stratum"]] += 1
        atomic_json(output / f"coverage_rank{rank}.json", {"step": step,
            "scope": "this process invocation", "sampled": dict(coverage),
            "skipped": dict(skipped_coverage), "nonzero_reward_groups": dict(active_coverage),
            "not_yet_sampled": sorted(expected_strata - set(coverage))})
        del pending
        is_best = False
        if cfg.eval_every > 0 and step % cfg.eval_every == 0:
            state_before_eval = rng_state()
            if rank == 0:
                result = evaluate(model, processor, builder, reward, val_rows, cfg, device,
                                  output / f"validation_step{step:06d}.jsonl")
                score = result["selection_proxy_macro"]
                is_best = score is not None and score > best
                if is_best:
                    best = score
                emit(output / "validation.jsonl", {"step": step, **result})
            if world > 1:
                dist.barrier()
            restore_rng(state_before_eval)
        flags = torch.tensor([float(is_best), best], device=device, dtype=torch.float64)
        if world > 1:
            dist.broadcast(flags, src=0)
        is_best, best = bool(flags[0]), float(flags[1])
        if step == first_update_step or is_best or (cfg.save_every > 0 and step % cfg.save_every == 0):
            save_checkpoint(model, processor, optimizer, scheduler, sampler, cfg, step, best, rank, world, is_best)
            last_saved = step
        stop = torch.tensor(int(time_limit_reached(time.monotonic() - started, cfg.max_hours)), device=device)
        if world > 1:
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if bool(stop):
            break
    if step != last_saved:
        save_checkpoint(model, processor, optimizer, scheduler, sampler, cfg, step, best, rank, world)
    if rank == 0:
        atomic_json(output / "finished.json", {"steps": step, "wall_seconds": time.monotonic() - started,
                    "best_validation_proxy": best if math.isfinite(best) else None,
                    "weights": str(Path(cfg.weights) / cfg.run_name)})
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
