"""Real GPU throughput and numerical checks on training-only preflight samples."""
import copy
import json
import time
from pathlib import Path
import torch
from .data import atomic_json
from .model import (collate_prompts, to_device, reset_positions, hidden_projection,
                    generate_prompt_batch)


@torch.no_grad()
def last_logits(model, inputs, device):
    reset_positions(model)
    with hidden_projection(model) as head, torch.autocast('cuda', dtype=torch.bfloat16):
        hidden = model(**to_device(inputs, device), use_cache=False, return_dict=True).logits[:, -1]
        logits = head(hidden.to(head.weight.dtype)).float()
    return torch.log_softmax(logits, -1)


def tune_rollout(model, processor, builder, selected, cfg, device, output):
    built = [builder.build(row) for row in selected]
    inputs = [pair[0] for pair in built]
    model.eval()
    # Probe high-probability token distributions, mixed/missing modalities and left padding.
    singles = [last_logits(model, x, device) for x in inputs]
    merged = last_logits(model, collate_prompts(inputs, processor.tokenizer.pad_token_id), device)
    errors, kls, shifts, top1 = [], [], [], []
    for index, logp in enumerate(singles):
        labels = logp.topk(32, dim=-1).indices[0]
        errors.append(float((logp[0, labels] - merged[index, labels]).abs().max()))
        kls.append(float((logp[0].exp() * (logp[0] - merged[index])).sum()))
        shifts.append(float((logp[0].exp() - merged[index].exp()).abs().max()))
        top1.append(bool(logp[0].argmax() == merged[index].argmax()))
    del singles, merged
    atomic_json(output / 'batch_numerics.json', {'top32_logp_max_errors': errors,
        'sample_count': len(inputs), 'modalities': [x[1]['effective_modality'] for x in built],
        'kl_single_to_batch': kls, 'maximum_probability_shift': shifts, 'same_top1': top1,
        'kl_limit': 0.01, 'probability_shift_limit': 0.025,
        'passed': max(kls) <= 0.01 and max(shifts) <= 0.025,
        'note': 'BF16 batch-shaped GEMM changes quantized logits; distribution check, not bitwise equivalence.'})
    if max(kls) > 0.01 or max(shifts) > 0.025:
        raise AssertionError(f'Batched/single distribution mismatch: KL={kls}, probability shift={shifts}')
    trials = []
    for width in (1, 4, 8):
        trial = copy.copy(cfg)
        trial.rollout_prompt_batch = width
        model._hecatepo_prompt_batch = width
        model._hecatepo_rollout_micro_batch = cfg.rollout_micro_batch
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        start = time.monotonic()
        replies = generate_prompt_batch(model, processor, inputs, cfg.group_size, trial, device)
        torch.cuda.synchronize(device)
        elapsed = time.monotonic() - start
        tokens = sum(len(r['ids']) for group in replies for r in group)
        record = dict(requested_prompt_batch=width, effective_prompt_batch=model._hecatepo_prompt_batch,
            actual_largest_prompt_batch=getattr(model,"_hecatepo_actual_prompt_batch",1),
            generated_sequences=sum(map(len, replies)), seconds=elapsed, response_tokens=tokens,
            response_tokens_per_second=tokens/elapsed, prompts_per_second=len(inputs)/elapsed,
            peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved(device)/1024**3)
        trials.append(record)
        print(json.dumps({'event':'rollout_benchmark', **record}), flush=True)
        atomic_json(output / 'rollout_benchmark.json', {'trials':trials})
        del replies
    candidates = [r for r in trials if r['peak_allocated_gib'] < 40]
    best = max(candidates or trials[:1], key=lambda r:r['response_tokens_per_second'])
    cfg.rollout_prompt_batch = best['effective_prompt_batch']
    model._hecatepo_prompt_batch = cfg.rollout_prompt_batch
    atomic_json(output / 'rollout_benchmark.json', {'trials':trials,
        'selected_prompt_batch':cfg.rollout_prompt_batch,
        'selection':'highest measured generated-token throughput below 40 GiB allocated; variable-length stochastic samples',
        'note':'Preflight speed does not establish end-to-end training speed or accuracy.'})
