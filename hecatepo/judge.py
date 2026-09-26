"""Resumable official-prompt QA grading; invoked explicitly after generation.

API failures remain pending and are never converted into correct/incorrect labels.
The API key is read from the environment and is never written to files or logs.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

from .data import atomic_json
from .evaluation import summarize_predictions
from .judge_prompts import (GRADE_PROMPT, INTENTQA_GRADE_INSTRUCTION,
                           MIMEQA_GRADE_INSTRUCTION, SIQ_GRADE_INSTRUCTION)

INSTRUCTIONS = {"intentqa": INTENTQA_GRADE_INSTRUCTION,
                "mimeqa": MIMEQA_GRADE_INSTRUCTION, "siq2": SIQ_GRADE_INSTRUCTION}
SCHEMA = {"type": "json_schema", "json_schema": {"name": "Evaluation", "strict": True,
          "schema": {"type": "object", "additionalProperties": False,
                     "required": ["correct", "explanation"],
                     "properties": {"correct": {"type": "boolean"}, "explanation": {"type": "string"}}}}}


def judge_request(row, model, max_tokens):
    messages = [{"role": "user", "content": INSTRUCTIONS[row["dataset"]]},
                {"role": "user", "content": GRADE_PROMPT.format(question=row["question"],
                 candidate_answer=row["answer"], ref_answer=row["reference"])}]
    return {"model": model, "messages": messages, "max_completion_tokens": max_tokens,
            "response_format": SCHEMA}


def request_key(request):
    return hashlib.sha256(json.dumps(request, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


async def run(args):
    from openai import AsyncOpenAI
    rows = [json.loads(line) for line in Path(args.predictions).read_text().splitlines() if line.strip()]
    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            if line.strip():
                saved = json.loads(line)
                cache[saved["key"]] = saved
    pending = []
    for row in rows:
        if row["dataset"] not in INSTRUCTIONS:
            continue
        if not row.get("answer"):
            row.update(judge_correct=False, judge_explanation="No parseable final answer.")
            continue
        request = judge_request(row, args.model, args.max_tokens)
        key = request_key(request)
        if key in cache:
            row.update(judge_correct=cache[key]["correct"], judge_explanation=cache[key]["explanation"], judge_model=args.model)
        else:
            pending.append((row, request, key))
    if args.limit > 0:
        pending = pending[:args.limit]
    client = None
    if pending:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("MIT_OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY in your shell before QA grading; training does not need it.")
        client = AsyncOpenAI(api_key=api_key, max_retries=0, timeout=args.timeout)
    semaphore, errors = asyncio.Semaphore(args.concurrency), []
    with cache_path.open("a") as cache_file:
        async def one(row, request, key):
            async with semaphore:
                for attempt in range(args.retries):
                    try:
                        response = await client.chat.completions.create(**request)
                        content = response.choices[0].message.content
                        result = json.loads(content or "")
                        if type(result.get("correct")) is not bool or not isinstance(result.get("explanation"), str):
                            raise ValueError("Malformed judge result")
                        record = {"key": key, "correct": result["correct"], "explanation": result["explanation"],
                                  "model": response.model, "response_id": response.id,
                                  "usage": response.usage.model_dump() if response.usage else None}
                        cache_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        cache_file.flush()
                        row.update(judge_correct=result["correct"], judge_explanation=result["explanation"], judge_model=response.model)
                        return
                    except Exception as exc:
                        if attempt + 1 == args.retries:
                            # Exception text can contain request headers; do not persist it.
                            errors.append({"uid": row["uid"], "error_type": type(exc).__name__})
                        else:
                            await asyncio.sleep(min(2 ** attempt, 8))
        await asyncio.gather(*(one(*item) for item in pending))
    if client:
        await client.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    metrics = summarize_predictions(rows)
    metrics.update(judge_model=args.model, judge_errors=errors,
                   pending_qa=sum(r["dataset"] in INSTRUCTIONS and type(r.get("judge_correct")) is not bool for r in rows))
    atomic_json(str(output) + ".metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if errors or metrics["pending_qa"]:
        raise SystemExit("Grading incomplete; rerun with the same cache after resolving access/rate-limit issues.")


def main():
    parser = argparse.ArgumentParser()
    for name in ("predictions", "output", "cache", "model"):
        parser.add_argument("--" + name, required=True)
    for name in ("max-tokens", "concurrency", "retries", "limit", "timeout"):
        parser.add_argument("--" + name, type=int, required=True)
    args = parser.parse_args()
    if min(args.max_tokens, args.concurrency, args.retries, args.timeout) <= 0:
        parser.error("Positive tokens/concurrency/retries/timeout required")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
