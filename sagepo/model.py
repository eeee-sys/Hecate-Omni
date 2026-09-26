"""Synchronous HF actor/rollout: one live adapter, exact chunked vocabulary entropy.

The frozen LM projection is bypassed in teacher-forced passes to avoid materializing
the full [sequence, vocabulary] tensor. It is applied in checkpointed chunks only
at response prediction positions. No hidden-state gradient is detached for loss.
"""
from contextlib import contextmanager
import copy
import gc
import json
import re
import traceback

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .core import ResponseStats


def load_model(cfg, device):
    from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
    from peft import LoraConfig, PeftModel, get_peft_model
    processor = Qwen2_5OmniProcessor.from_pretrained(cfg.model, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    # Both processor paths have explicit budgets; the binary-video branch must not bypass resize.
    processor.image_processor.max_pixels = cfg.image_max_pixels
    if hasattr(processor, "video_processor"):
        processor.video_processor.max_pixels = cfg.video_max_pixels
    base = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        cfg.model, torch_dtype=torch.bfloat16, attn_implementation=cfg.attn_implementation,
        low_cpu_mem_usage=True, local_files_only=True,
    ).to(device)
    targets = [name for name, module in base.named_modules()
               if re.fullmatch(r"model\.layers\.\d+\.(?:self_attn|mlp)\.\w+", name)
               and name.rsplit(".", 1)[-1] in cfg.targets and isinstance(module, nn.Linear)]
    layers = base.config.text_config.num_hidden_layers
    if len(targets) != layers * len(cfg.targets):
        raise ValueError(f"LoRA module match mismatch: {len(targets)} != {layers}*{len(cfg.targets)}")
    adapter_path = cfg.resume or cfg.adapter
    if adapter_path:
        model = PeftModel.from_pretrained(base, adapter_path, is_trainable=bool(cfg.resume), autocast_adapter_dtype=True)
    else:
        model = get_peft_model(base, LoraConfig(
            r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules=targets, bias="none", task_type="CAUSAL_LM",
            init_lora_weights=True, use_rslora=False, use_dora=False,
        ))
    if cfg.adapter and cfg.adapter_scale != 1:
        for module in model.modules():
            if hasattr(module, "scaling") and isinstance(module.scaling, dict):
                for name in module.scaling:
                    module.scaling[name] *= cfg.adapter_scale
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    trainable = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
    if cfg.mode in {"train", "smoke"} and not trainable:
        raise ValueError("No trainable LoRA parameters")
    if any("lora_" not in name for name in trainable):
        raise ValueError("Unexpected trainable base-model parameters")
    model.eval()
    return model, processor, {"target_modules": targets, "trainable_parameters": trainable,
                              "trainable_count": sum(trainable.values()),
                              "total_count": sum(p.numel() for p in model.parameters())}


def to_device(inputs, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}


def repeat_prompt(inputs, n):
    repeated = {}
    for name, value in inputs.items():
        if isinstance(value, torch.Tensor):
            repeated[name] = value.repeat((n,) + (1,) * (value.ndim - 1)) if value.ndim else value
        elif isinstance(value, list):
            repeated[name] = value * n
        else:
            repeated[name] = value
    return repeated


def reset_positions(model):
    base = model.get_base_model()
    if hasattr(base, "rope_deltas"):
        base.rope_deltas = None


@torch.no_grad()
def _generate_group_once(model, processor, inputs, count, cfg, device, greedy=False):
    model.eval()
    prompt_length = inputs["input_ids"].shape[1]
    responses = []
    eos = model.generation_config.eos_token_id or model.config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos)
    for offset in range(0, count, cfg.rollout_micro_batch):
        n = min(cfg.rollout_micro_batch, count - offset)
        batch = to_device(repeat_prompt(inputs, n), device)
        reset_positions(model)
        kwargs = dict(max_new_tokens=cfg.eval_response_tokens if greedy else cfg.max_response_tokens,
                      do_sample=not greedy, num_beams=1, use_cache=True,
                      repetition_penalty=1.0, no_repeat_ngram_size=0,
                      pad_token_id=processor.tokenizer.pad_token_id,
                      eos_token_id=list(eos_ids), return_dict_in_generate=False)
        if not greedy:
            kwargs.update(temperature=1.0, top_p=1.0, top_k=0, typical_p=1.0)
        # This Thinker revision projects every prefill token by default. Generation
        # uses only the last token, avoiding a [G, prompt_length, vocab] allocation.
        head = model.get_base_model().lm_head
        hook = head.register_forward_pre_hook(lambda module, args: (args[0][:, -1:, :],))
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                generated = model.generate(**batch, **kwargs)
        finally:
            hook.remove()
        for sequence in generated[:, prompt_length:]:
            ids = sequence.detach().cpu()
            stop = next((i + 1 for i, token in enumerate(ids.tolist()) if token in eos_ids), len(ids))
            ids = ids[:stop]
            if not len(ids):
                raise RuntimeError("Empty generation")
            responses.append({"ids": ids, "text": processor.tokenizer.decode(ids, skip_special_tokens=True),
                              "terminated": int(ids[-1]) in eos_ids})
        del generated, batch
    return responses


@torch.no_grad()
def generate_group(model, processor, inputs, count, cfg, device, greedy=False):
    if greedy or not getattr(cfg, "auto_rollout_batch", 1):
        return _generate_group_once(model, processor, inputs, count, cfg, device, greedy)
    size = min(count, getattr(model, "_sagepo_rollout_micro_batch", cfg.rollout_micro_batch))
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device)
    while True:
        attempt = copy.copy(cfg)
        attempt.rollout_micro_batch = size
        try:
            result = _generate_group_once(model, processor, inputs, count, attempt, device, greedy)
            model._sagepo_rollout_micro_batch = size
            return result
        except torch.cuda.OutOfMemoryError:
            smaller = [n for n in (5, 2, 1) if n < size]
            if not smaller:
                raise
        # Leave the exception scope before releasing cached allocations.
        previous, size = size, max(smaller)
        gc.collect()
        torch.cuda.empty_cache()
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, device)
        print(json.dumps({"event": "rollout_oom_fallback", "from": previous,
                          "to": size, "group_size": count}), flush=True)

def collate_prompts(prompts, pad_token_id):
    """Left-pad text; concatenate media in prompt order, including absent modalities."""
    from torch.nn.functional import pad
    width = max(p['input_ids'].shape[1] for p in prompts)
    out = {}
    keys = set().union(*(p.keys() for p in prompts))
    sequence_keys = {'input_ids', 'attention_mask', 'token_type_ids'}
    media_keys = {'input_features', 'feature_attention_mask', 'pixel_values',
                  'pixel_values_videos', 'image_grid_thw', 'video_grid_thw',
                  'video_second_per_grid', 'audio_feature_lengths'}
    unknown = keys - sequence_keys - media_keys
    if unknown:
        raise ValueError(f'Unsupported batched processor keys: {sorted(unknown)}')
    for key in keys:
        values = [p[key] for p in prompts if key in p]
        if not all(isinstance(v, torch.Tensor) for v in values):
            raise TypeError(f'Expected tensor for {key}')
        if key in sequence_keys:
            if len(values) != len(prompts):
                raise ValueError(f'Missing sequence field {key}')
            fill = pad_token_id if key == 'input_ids' else 0
            values = [pad(v, (width-v.shape[-1], 0), value=fill) for v in values]
        elif key in {'input_features', 'feature_attention_mask'}:
            longest = max(v.shape[-1] for v in values)
            values = [pad(v, (0, longest-v.shape[-1]), value=0) for v in values]
        out[key] = torch.cat(values, dim=0)
    return out


def prompt_buckets(prompts, width, max_padding_ratio):
    """Return original indices grouped by length; every prompt appears once."""
    ordered = sorted(range(len(prompts)), key=lambda i: prompts[i]['input_ids'].shape[-1])
    groups, block = [], []
    for index in ordered:
        length = prompts[index]['input_ids'].shape[-1]
        if block and (len(block) >= width or length / prompts[block[0]]['input_ids'].shape[-1] > max_padding_ratio):
            groups.append(block)
            block = []
        block.append(index)
    if block:
        groups.append(block)
    return groups


@torch.no_grad()
def generate_prompt_batch(model, processor, prompts, count, cfg, device, greedy=False):
    """Generate G replies for each of P distinct prompts in one GPU batch.

    Group boundaries remain intact. OOM retries the whole block with restored RNG,
    eventually using the existing per-question 5 -> 2 -> 1 fallback.
    """
    if len(prompts) == 1:
        return [generate_group(model, processor, prompts[0], count, cfg, device, greedy)]
    size = min(len(prompts), getattr(model, '_sagepo_prompt_batch', cfg.rollout_prompt_batch))
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(device)
    while True:
        try:
            result = [None] * len(prompts)
            buckets = prompt_buckets(prompts, size, getattr(cfg, 'prompt_batch_max_padding_ratio', 1.5))
            for indices in buckets:
                block = [prompts[i] for i in indices]
                if len(block) == 1:
                    result[indices[0]] = generate_group(model, processor, block[0], count, cfg, device, greedy)
                    continue
                expanded = [p for p in block for _ in range(count)]
                merged = collate_prompts(expanded, processor.tokenizer.pad_token_id)
                attempt = copy.copy(cfg)
                attempt.rollout_micro_batch = 1
                # count=1 prevents repeat_prompt from expanding the already batched inputs.
                flat = _generate_group_once(model, processor, merged, 1, attempt, device, greedy)
                if len(flat) != len(expanded):
                    raise AssertionError('Batched generation changed the number of responses')
                for j, index in enumerate(indices):
                    result[index] = flat[j*count:(j+1)*count]
            model._sagepo_prompt_batch = size
            model._sagepo_actual_prompt_batch = max(map(len, buckets))
            return result
        except torch.cuda.OutOfMemoryError:
            if size <= 1 or not cfg.auto_rollout_batch:
                raise
        previous, size = size, max(1, size//2)
        # Drop references from completed blocks before retrying.
        result, merged, flat = None, None, None
        gc.collect()
        torch.cuda.empty_cache()
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, device)
        print(json.dumps({'event': 'prompt_batch_oom_fallback', 'from': previous,
                          'to': size, 'group_size': count}), flush=True)


def rollout_rows(model, processor, builder, rows, cfg, device, on_error=None):
    """Build and generate rows in bounded batches, isolating bad samples when requested."""
    width = getattr(cfg, 'rollout_prompt_batch', 1)
    for start in range(0, len(rows), width):
        block = rows[start:start+width]
        ready = []
        for row in block:
            try:
                ready.append((row, *builder.build(row)))
            except Exception as error:
                if on_error is None:
                    raise
                on_error(row, "input_build", error, traceback.format_exc())
        if not ready:
            continue
        try:
            groups = generate_prompt_batch(model, processor, [item[1] for item in ready],
                                           cfg.group_size, cfg, device)
        except Exception:
            # The failed batch does not identify the bad row. Retry each prompt alone
            # so one malformed sample cannot discard healthy samples in the same block.
            groups = []
            for row, inputs, _ in ready:
                try:
                    groups.append(generate_prompt_batch(model, processor, [inputs],
                                                        cfg.group_size, cfg, device)[0])
                except Exception as error:
                    if on_error is None:
                        raise
                    on_error(row, "generation", error, traceback.format_exc())
                    groups.append(None)
        for (row, inputs, media), responses in zip(ready, groups):
            if responses is not None:
                yield row, inputs, media, responses


@contextmanager
def hidden_projection(model):
    """Scoped frozen-head bypass, restored even if the model raises an exception."""
    base = model.get_base_model()
    head = base.lm_head
    base.lm_head = nn.Identity()
    try:
        yield head
    finally:
        base.lm_head = head


def teacher_hidden(model, inputs, response_ids, device):
    batch = to_device(inputs, device)
    prompt_length = batch["input_ids"].shape[1]
    ids = response_ids.to(device).unsqueeze(0)
    batch["input_ids"] = torch.cat((batch["input_ids"], ids), dim=1)
    batch["attention_mask"] = torch.cat((batch["attention_mask"], torch.ones_like(ids)), dim=1)
    reset_positions(model)
    with hidden_projection(model), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(**batch, use_cache=False, return_dict=True, output_hidden_states=False)
    hidden = output.logits
    if hidden.shape[-1] != model.config.text_config.hidden_size:
        raise RuntimeError("Frozen-head bypass returned unexpected hidden shape")
    return hidden[0], prompt_length


def project_log_probs(head, hidden, labels):
    logits = head(hidden.to(head.weight.dtype)).float()
    chosen = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return chosen - torch.logsumexp(logits, dim=-1)


@torch.no_grad()
def old_policy_stats(model, inputs, response_ids, cfg, device):
    model.eval()
    hidden, prompt = teacher_hidden(model, inputs, response_ids, device)
    predictions = hidden[prompt - 1:prompt + len(response_ids) - 1]
    labels = response_ids.to(device)
    head = model.get_base_model().lm_head
    log_probs, entropies = [], []
    for start in range(0, len(labels), cfg.entropy_chunk):
        stop = start + cfg.entropy_chunk
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = head(predictions[start:stop].to(head.weight.dtype)).float()
        log_z = torch.logsumexp(logits, dim=-1)
        logp = logits.gather(-1, labels[start:stop, None]).squeeze(-1) - log_z
        entropy = log_z - (torch.softmax(logits, -1) * logits).sum(-1)
        log_probs.append(logp.cpu())
        entropies.append(entropy.clamp_min(0).cpu())
    return ResponseStats(torch.cat(log_probs), torch.cat(entropies),
                         hidden[prompt:prompt + len(labels)].detach().to("cpu", dtype=torch.bfloat16))


def current_log_probs(model, inputs, response_ids, cfg, device):
    hidden, prompt = teacher_hidden(model, inputs, response_ids, device)
    predictions = hidden[prompt - 1:prompt + len(response_ids) - 1]
    labels = response_ids.to(device)
    head = model.get_base_model().lm_head
    parts = []
    for start in range(0, len(labels), cfg.entropy_chunk):
        stop = start + cfg.entropy_chunk
        # Recompute projection in backward instead of retaining vocabulary-sized activations.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            part = checkpoint(lambda h, y: project_log_probs(head, h, y),
                              predictions[start:stop], labels[start:stop], use_reentrant=False)
        parts.append(part)
    return torch.cat(parts)
