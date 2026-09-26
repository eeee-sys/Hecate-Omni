"""HBA metrics; QA cosine is explicitly a development proxy, never judge accuracy."""
from collections import defaultdict
import numpy as np
from .data import TASK_MAP
from .rewards import normalized_label


def weighted_f1(truths, predictions):
    if not truths:
        return None
    total = 0.0
    for label in sorted(set(truths)):
        tp = sum(t == label and p == label for t, p in zip(truths, predictions))
        fp = sum(t != label and p == label for t, p in zip(truths, predictions))
        fn = sum(t == label and p != label for t, p in zip(truths, predictions))
        support = sum(t == label for t in truths)
        total += support * (2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    return total / len(truths)


def mean_emotion_wa(truths, predictions):
    if not truths:
        return None
    scores = []
    for label in sorted(set(truths)):
        pos = sum(t == label for t in truths)
        neg = len(truths) - pos
        tp = sum(t == label and p == label for t, p in zip(truths, predictions))
        tn = sum(t != label and p != label for t, p in zip(truths, predictions))
        scores.append(.5 * (tp / pos if pos else 0) + .5 * (tn / neg if neg else 0))
    return float(np.mean(scores))


def binary_sentiment(label):
    label = normalized_label(label)
    if label in {"highly positive", "positive", "weakly positive", "strongly positive", "slightly positive"}:
        return "positive"
    if label in {"highly negative", "negative", "weakly negative", "strongly negative", "slightly negative"}:
        return "negative"
    if label == "neutral":
        return "neutral"
    return "__invalid__"


def summarize_predictions(records):
    by_dataset = defaultdict(list)
    for row in records:
        by_dataset[row["dataset"]].append(row)
    datasets, task_metrics, task_proxy = {}, defaultdict(list), defaultdict(list)
    for name, rows in sorted(by_dataset.items()):
        task = TASK_MAP[name]
        truth = [normalized_label(r["reference"]) for r in rows]
        prediction = [normalized_label(r.get("answer")) or "__invalid__" for r in rows]
        metric, metric_name = None, "judge_accuracy_pending"
        if rows[0]["reward_type"] == "qa":
            if all(isinstance(r.get("judge_correct"), bool) for r in rows):
                metric = sum(r["judge_correct"] for r in rows) / len(rows)
                metric_name = "judge_accuracy"
            proxy = float(np.mean([r["task_reward"] for r in rows]))
        elif task == "EMO":
            metric, metric_name = mean_emotion_wa(truth, prediction), "mean_weighted_accuracy"
            proxy = metric
        elif task == "SEN":
            keep = [i for i, label in enumerate(truth) if binary_sentiment(label) != "neutral"]
            if any(binary_sentiment(truth[i]) == "__invalid__" for i in keep):
                raise ValueError(f"Unmapped sentiment target in {name}")
            metric = weighted_f1([binary_sentiment(truth[i]) for i in keep],
                                 [binary_sentiment(prediction[i]) for i in keep])
            metric_name, proxy = "binary_weighted_f1_excluding_neutral", metric
        else:
            metric, metric_name = weighted_f1(truth, prediction), "weighted_f1"
            proxy = metric
        entry = {"n": len(rows), "metric": metric_name, "score": metric,
                 "development_proxy": proxy,
                 "format_rate": float(np.mean([r["format_reward"] for r in rows])),
                 "invalid_answers": sum(r.get("answer") is None for r in rows)}
        if name == "mmsd":
            entry["protocol_warning"] = "Released mmsd subset; paper MUStARD equivalence unverified."
        datasets[name] = entry
        if metric is not None:
            task_metrics[task].append(metric)
        if proxy is not None:
            task_proxy[task].append(proxy)
    per_task = {task: float(np.mean(scores)) for task, scores in task_metrics.items()}
    proxies = {task: float(np.mean(scores)) for task, scores in task_proxy.items()}
    return {"datasets": datasets, "tasks": per_task, "task_proxies": proxies,
            "ten_task_macro": float(np.mean(list(per_task.values()))) if len(per_task) == 10 else None,
            "selection_proxy_macro": float(np.mean(list(proxies.values()))) if proxies else None,
            "selection_note": "QA uses MiniLM cosine; this is not the paper's 10-task judge score.",
            "paper_protocol_verified": False,
            "n": len(records)}
