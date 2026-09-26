"""Task-aware rewards. Only final answers enter task scoring."""
import re
import torch


def extract_answer(text):
    starts = list(re.finditer(r"\\boxed\s*\{", text))
    if len(starts) != 1:
        return None
    start = starts[0].end()
    depth = 1
    for pos in range(start, len(text)):
        if text[pos] == "{" and (pos == 0 or text[pos - 1] != "\\"):
            depth += 1
        elif text[pos] == "}" and (pos == 0 or text[pos - 1] != "\\"):
            depth -= 1
            if depth == 0:
                answer = text[start:pos].strip()
                return answer if answer else None
    return None


def normalized_label(value):
    return " ".join(str(value or "").strip().lower().split())


def valid_format(text):
    if text.count("<think>") != 1 or text.count("</think>") != 1:
        return False
    start, end = text.find("<think>"), text.find("</think>")
    box = text.find("\\boxed")
    return start >= 0 and end > start and box > end and extract_answer(text) is not None


def length_penalty(tokens, limit, buffer):
    if limit <= 0:
        return 0.0
    if buffer <= 0 or buffer > limit:
        raise ValueError("Invalid length buffer")
    return -min(1.0, max(0.0, (tokens - (limit - buffer)) / buffer))


class RewardEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.encoder = None
        self.reference_cache = {}

    def _encoder(self):
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(self.cfg.reward_model, device="cpu")
            self.encoder.eval()
        return self.encoder

    @torch.no_grad()
    def score(self, row, texts, lengths):
        answers = [extract_answer(t) for t in texts]
        if row["reward_type"] == "qa":
            model = self._encoder()
            reference = row["answer"]
            if reference not in self.reference_cache:
                self.reference_cache[reference] = model.encode(reference, convert_to_tensor=True, normalize_embeddings=True)
            embedding = model.encode([a or "" for a in answers], convert_to_tensor=True, normalize_embeddings=True)
            similarities = ((embedding @ self.reference_cache[reference] + 1) / 2).clamp(0, 1).tolist()
            task = [s if a is not None else 0.0 for s, a in zip(similarities, answers)]
        else:
            task = [float(a is not None and normalized_label(a) == normalized_label(row["answer"])) for a in answers]
        records = []
        for text, answer, n, accuracy in zip(texts, answers, lengths, task):
            fmt = float(valid_format(text))
            penalty = length_penalty(n, self.cfg.length_limit, self.cfg.length_buffer)
            score = self.cfg.task_reward_weight * accuracy + self.cfg.format_reward_weight * fmt + self.cfg.length_reward_weight * penalty
            records.append({"reward": score, "task_reward": accuracy, "format_reward": fmt,
                            "length_penalty": penalty, "answer": answer, "response_tokens": n,
                            "reward_type": row["reward_type"]})
        return records
