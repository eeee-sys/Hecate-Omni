"""Column-projected HBA indexing and on-demand embedded-media decoding."""
import hashlib
import io
import json
import math
import os
import re
import threading
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch


META_COLUMNS = ("problem", "answer", "dataset", "task", "class_label", "modality_signature")
MEDIA_COLUMNS = ("audios", "videos", "images")
FORMAT_PROMPT = ("\nFirst reason briefly from the provided evidence. Enclose your reasoning in "
                 "<think>...</think>. Then provide exactly one final answer in \\boxed{...}. "
                 "For classification, use exactly one of the requested labels. "
                 "For open questions, give a concise, direct answer.")
SYSTEM_PROMPT = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
                 "capable of perceiving auditory and visual inputs, as well as generating text and speech.")
TASK_MAP = {
    "cremad": "EMO", "meld_emotion": "EMO", "mosei_emotion": "EMO", "tess": "EMO",
    "meld_senti": "SEN", "mosei_senti": "SEN", "chsimsv2": "SEN",
    "urfunny": "HUM", "intentqa": "INT", "ptsd_in_the_wild": "PTSD",
    "mmpsy_anxiety": "ANX", "mmpsy_depression": "DEP", "daicwoz": "DEP",
    "siq2": "SOC", "mimeqa": "NVC", "mmsd": "SAR",
}
QA_DATASETS = {"intentqa", "siq2", "mimeqa"}


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def canonical_dataset(value):
    return str(value or "").strip().lower()


def load_index(data_dir, split, index_dir):
    if split not in {"train", "validation", "test"}:
        raise ValueError("Explicit train/validation/test split required")
    paths = sorted(Path(data_dir).glob(f"{split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No {split} parquet files in {data_dir}")
    shard_matches = [re.fullmatch(rf"{split}-(\d+)-of-(\d+)\.parquet", p.name) for p in paths]
    if all(shard_matches):
        expected = {int(m[2]) for m in shard_matches}
        if len(expected) != 1 or {int(m[1]) for m in shard_matches} != set(range(next(iter(expected)))):
            raise ValueError(f"Incomplete or inconsistent {split} parquet shards")
    fingerprint = hashlib.sha256(json.dumps([(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns)
                                             for p in paths]).encode()).hexdigest()
    destination = Path(index_dir) / f"{split}.json"
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved.get("fingerprint") == fingerprint:
            return saved["rows"], saved["summary"]
    rows, counts, modalities, label_counts = [], Counter(), Counter(), {}
    for path in paths:
        parquet = pq.ParquetFile(path, memory_map=True)
        available = set(parquet.schema_arrow.names)
        if not {"problem", "answer", "dataset"} <= available:
            raise ValueError(f"Missing required columns: {path}")
        columns = [c for c in META_COLUMNS if c in available]
        for rg in range(parquet.num_row_groups):
            table = parquet.read_row_group(rg, columns=columns, use_threads=False)
            for offset, row in enumerate(table.to_pylist()):
                dataset = canonical_dataset(row.get("dataset"))
                if dataset not in TASK_MAP:
                    raise ValueError(f"Unknown dataset {dataset!r}; add an explicit task mapping")
                if not isinstance(row.get("problem"), str) or not isinstance(row.get("answer"), str):
                    raise ValueError(f"Invalid prompt or target in {path.name}:{rg}:{offset}")
                uid = hashlib.sha256(f"{split}/{path.name}/{rg}/{offset}".encode()).hexdigest()[:24]
                row.update(uid=uid, dataset=dataset, split=split, file=str(path),
                           row_group=rg, row_offset=offset, task_group=TASK_MAP[dataset],
                           reward_type="qa" if dataset in QA_DATASETS else "cls")
                rows.append(row)
                counts[dataset] += 1
                modalities[str(row.get("modality_signature", "unknown"))] += 1
                if dataset not in QA_DATASETS:
                    label_counts.setdefault(dataset, Counter())[row["answer"].strip().lower()] += 1
        parquet.close()
    summary = {"split": split, "count": len(rows), "shards": len(paths),
               "datasets": dict(counts), "modalities": dict(modalities),
               "labels": {key: dict(value) for key, value in label_counts.items()},
               "fingerprint": fingerprint,
               "protocol_note": "mmsd is reported as released; equivalence to paper MUStARD is unverified."}
    atomic_json(destination, {"fingerprint": fingerprint, "summary": summary, "rows": rows})
    return rows, summary


def stratified_rows(rows, per_dataset, seed):
    rng = np.random.default_rng(seed)
    groups = {}
    for row in rows:
        groups.setdefault(row["dataset"], []).append(row)
    selected = []
    for name in sorted(groups):
        group = groups[name]
        indices = rng.permutation(len(group))[:per_dataset] if per_dataset > 0 else range(len(group))
        selected.extend(group[int(i)] for i in indices)
    return selected


def media_group_key(media, problem):
    """Match the same embedded video even when questions/audio annotations differ."""
    digest = hashlib.sha256()
    for kind in ("videos", "audios", "images"):
        blobs = media.get(kind) or []
        if blobs:
            digest.update(kind.encode())
            for blob in blobs:
                digest.update(len(blob).to_bytes(8, "big"))
                digest.update(hashlib.sha256(blob).digest())
            return digest.hexdigest()
    return hashlib.sha256(("text:" + " ".join(problem.split())).encode()).hexdigest()


def training_validation_split(train_rows, validation_rows, index_dir, fingerprint, groups_per_dataset, seed,
                              media_cache=None, max_fraction=0.1):
    """Add QA validation where absent; no question from a held-out media group trains.

    Byte identity is guaranteed; perceptual equivalence of re-encoded clips is not.
    The inherited OmniSapiens weights may already have seen these training clips.
    """
    missing = QA_DATASETS - {r["dataset"] for r in validation_rows}
    qa_rows = [r for r in train_rows if r["dataset"] in missing]
    if not missing or not qa_rows or groups_per_dataset <= 0:
        return train_rows, validation_rows, {"held_out_rows": 0, "missing_validation_tasks": sorted(missing)}
    path = Path(index_dir) / "qa_media_groups.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    if cache.get("fingerprint") != fingerprint:
        reader, keys = MediaReader(cache_dir=media_cache), {}
        for i, row in enumerate(qa_rows):
            keys[row["uid"]] = reader.group_key(row)
            if (i + 1) % 1000 == 0:
                print(f"QA media identity index {i + 1}/{len(qa_rows)}", flush=True)
        cache = {"fingerprint": fingerprint, "groups": keys}
        atomic_json(path, cache)
    keys = cache["groups"]
    rng, reserved = np.random.default_rng(seed), set()
    counts = {}
    for dataset in sorted(missing):
        candidates = sorted({keys[r["uid"]] for r in qa_rows if r["dataset"] == dataset})
        # Keep at least one source group available for RL.
        n = min(groups_per_dataset, math.ceil(len(candidates) * max_fraction), max(0, len(candidates) - 1))
        chosen = rng.permutation(candidates)[:n].tolist()
        reserved.update(chosen)
        counts[dataset] = {"total_media_groups": len(candidates), "held_out_media_groups": n}
    held_out = [{**r, "split": "validation_from_train", "media_group": keys[r["uid"]]}
                for r in qa_rows if keys[r["uid"]] in reserved]
    excluded = {r["uid"] for r in held_out}
    remaining = [r for r in train_rows if r["uid"] not in excluded]
    summary = {"held_out_rows": len(held_out), "remaining_train_rows": len(remaining), "groups": counts,
               "seed": seed, "groups_per_dataset": groups_per_dataset, "max_fraction": max_fraction,
               "identity": "sha256 of embedded video; audio/image/text fallback",
               "scope": "excluded from new HECATEPO updates; inherited checkpoint exposure is unknown",
               "excluded_train_uids": sorted(excluded)}
    atomic_json(Path(index_dir) / f"qa_holdout_seed{seed}_groups{groups_per_dataset}_fraction{max_fraction}.json", summary)
    return remaining, validation_rows + held_out, summary


class PromptSampler:
    """No dropped tail; optional task mixture is explicit sampling with replacement."""
    def __init__(self, rows, seed, task_balance_fraction=0.0, mode="mixture"):
        self.rows, self.rng = rows, np.random.default_rng(seed)
        self.balance = task_balance_fraction
        if mode not in {"mixture", "coverage"} or not rows:
            raise ValueError("Sampler requires nonempty rows and a supported mode")
        self.mode = mode
        self.strata = {}
        for i, row in enumerate(rows):
            self.strata.setdefault(self.stratum(row), []).append(i)
        self.groups = {}
        for i, row in enumerate(rows):
            self.groups.setdefault(row["task_group"], []).append(i)
        self.tasks = sorted(self.groups)
        self.order = self._new_order()
        self.cursor = 0

    @staticmethod
    def stratum(row):
        label = row.get("answer", "").strip().lower() if row.get("reward_type") == "cls" else "__qa__"
        return (row["task_group"], row.get("dataset", row["task_group"]), label)

    def _new_order(self):
        if self.mode == "mixture":
            return self.rng.permutation(len(self.rows)).tolist()
        # One representative per dataset/label first, then each remaining row once.
        keys = sorted(self.strata)
        first = [int(self.rng.choice(self.strata[keys[int(k)]])) for k in self.rng.permutation(len(keys))]
        selected = set(first)
        rest = [int(i) for i in self.rng.permutation(len(self.rows)) if int(i) not in selected]
        return first + rest

    def take(self, n):
        indices = []
        for _ in range(n):
            if self.mode == "mixture" and self.balance and self.rng.random() < self.balance:
                task = self.rng.choice(self.tasks)
                index = int(self.rng.choice(self.groups[task]))
            else:
                if self.cursor >= len(self.order):
                    self.order = self._new_order()
                    self.cursor = 0
                index = self.order[self.cursor]
                self.cursor += 1
            indices.append(index)
        return [self.rows[i] for i in indices]

    def state_dict(self):
        return {"rng": self.rng.bit_generator.state, "order": self.order, "cursor": self.cursor, "mode": self.mode}

    def load_state_dict(self, state):
        if state.get("mode", "mixture") != self.mode:
            raise ValueError("Sampler mode differs from saved checkpoint")
        self.rng.bit_generator.state = state["rng"]
        self.order, self.cursor = state["order"], state["cursor"]


class MediaReader:
    def __init__(self, cache_row_groups=1, cache_dir=None):
        self.cache = OrderedDict()
        self.capacity = cache_row_groups
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.manifests = {}

    def manifest(self, row):
        """Persist one whole row group once, with content-addressed native media."""
        key = (row["file"], row["row_group"])
        if key in self.manifests:
            return self.manifests[key]
        source = Path(row["file"])
        identity = hashlib.sha256(f"{source.resolve()}:{source.stat().st_size}:{source.stat().st_mtime_ns}:{row['row_group']}".encode()).hexdigest()
        destination = self.cache_dir / "manifests" / f"{identity}.json"
        if destination.exists():
            manifest = json.loads(destination.read_text())
        else:
            # File lock prevents torchrun ranks from staging the same group concurrently.
            import fcntl
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.with_suffix(".lock").open("w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if destination.exists():
                    manifest = json.loads(destination.read_text())
                else:
                    parquet = pq.ParquetFile(source, memory_map=True)
                    columns = [c for c in MEDIA_COLUMNS if c in parquet.schema_arrow.names]
                    table = parquet.read_row_group(row["row_group"], columns=columns, use_threads=False)
                    parquet.close()
                    print(f"Caching native media: {source.name} row group {row['row_group']} ({len(table)} rows)", flush=True)
                    manifest = []
                    for offset in range(len(table)):
                        media = table.slice(offset, 1).to_pylist()[0]
                        entry = {}
                        for kind in MEDIA_COLUMNS:
                            entry[kind] = []
                            for blob in media.get(kind) or []:
                                if not blob:
                                    continue
                                digest = hashlib.sha256(blob).hexdigest()
                                path = self.cache_dir / "blobs" / digest[:2] / digest
                                if not path.exists():
                                    path.parent.mkdir(parents=True, exist_ok=True)
                                    temp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
                                    temp.write_bytes(blob)
                                    temp.replace(path)
                                elif path.stat().st_size != len(blob):
                                    raise ValueError(f"Corrupt media cache: {path}")
                                entry[kind].append({"sha256": digest, "bytes": len(blob)})
                        manifest.append(entry)
                    atomic_json(destination, manifest)
        self.manifests[key] = manifest
        return manifest

    def group_key(self, row):
        if self.cache_dir is None:
            return media_group_key(self.read(row), row["problem"])
        entry = self.manifest(row)[row["row_offset"]]
        for kind in ("videos", "audios", "images"):
            if entry[kind]:
                digest = hashlib.sha256(kind.encode())
                for blob in entry[kind]:
                    digest.update(blob["bytes"].to_bytes(8, "big"))
                    digest.update(bytes.fromhex(blob["sha256"]))
                return digest.hexdigest()
        return media_group_key({}, row["problem"])

    def read(self, row):
        if self.cache_dir is not None:
            entry = self.manifest(row)[row["row_offset"]]
            media = {kind: [] for kind in MEDIA_COLUMNS}
            for kind in MEDIA_COLUMNS:
                for blob in entry[kind]:
                    path = self.cache_dir / "blobs" / blob["sha256"][:2] / blob["sha256"]
                    payload = path.read_bytes()
                    if len(payload) != blob["bytes"]:
                        raise ValueError(f"Truncated native media cache: {path}")
                    media[kind].append(payload)
            return media
        key = (row["file"], row["row_group"])
        if key not in self.cache:
            parquet = pq.ParquetFile(row["file"], memory_map=True)
            columns = [c for c in MEDIA_COLUMNS if c in parquet.schema_arrow.names]
            table = parquet.read_row_group(row["row_group"], columns=columns, use_threads=False)
            parquet.close()
            while len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
            self.cache[key] = table
        self.cache.move_to_end(key)
        media = self.cache[key].slice(row["row_offset"], 1).to_pylist()[0]
        return {c: [b for b in (media.get(c) or []) if b] for c in MEDIA_COLUMNS}


def decode_audio(blob, sampling_rate, max_seconds):
    import soundfile as sf
    from scipy.signal import resample_poly
    with sf.SoundFile(io.BytesIO(blob)) as source:
        sr = source.samplerate
        original_seconds = source.frames / sr
        frames = min(source.frames, round(max_seconds * sr)) if max_seconds > 0 else source.frames
        samples = source.read(frames=frames, dtype="float32", always_2d=True)
    samples = samples.mean(axis=1)
    if sr != sampling_rate:
        divisor = math.gcd(sr, sampling_rate)
        samples = resample_poly(samples, sampling_rate // divisor, sr // divisor).astype(np.float32)
    if len(samples) == 0 or not np.isfinite(samples).all():
        raise ValueError("Empty/nonfinite audio")
    return samples, {"original_seconds": original_seconds, "used_seconds": len(samples) / sampling_rate}


def decode_video(blob, nframes, max_seconds, max_pixels, sampling="segment_starts"):
    import av
    from PIL import Image
    container = av.open(io.BytesIO(blob))
    frames, actual_times = [], []
    try:
        stream = container.streams.video[0]
        stream.thread_count = 1
        start = float((stream.start_time or 0) * stream.time_base)
        duration = float(stream.duration * stream.time_base) if stream.duration else float(container.duration or 0) / av.time_base
        if duration <= 0:
            raise ValueError("Video duration missing; cannot sample timeline reliably")
        used = min(duration, max_seconds) if max_seconds > 0 else duration
        targets = np.linspace(start, start + used * (1 - 1 / max(nframes, 2)), nframes)
        if sampling == "uniform_endpoints":
            fps = float(stream.average_rate or stream.base_rate or 25)
            targets = np.linspace(start, start + max(0, used - 1 / fps), nframes)
        for target in targets:
            container.seek(int(target / stream.time_base), stream=stream, backward=True, any_frame=False)
            chosen = None
            for frame in container.decode(stream):
                chosen = frame
                if frame.time is not None and frame.time + 1e-6 >= target:
                    break
            if chosen is None:
                raise ValueError(f"Failed to decode video at {target}s")
            picture = chosen.to_image().convert("RGB")
            if picture.width * picture.height > max_pixels:
                scale = math.sqrt(max_pixels / (picture.width * picture.height))
                picture = picture.resize((max(28, int(picture.width * scale)), max(28, int(picture.height * scale))), Image.Resampling.BICUBIC)
            frames.append(np.asarray(picture))
            actual_times.append(float(chosen.time or target) - start)
    finally:
        container.close()
    return np.stack(frames), {"original_seconds": duration, "used_seconds": used,
                             "frame_times": actual_times, "sampling_fps": nframes / used}


def compress_text(text, budget, tokenizer, head_tokens, tail_tokens, windows):
    """Deterministic chronological excerpts; retain task instructions at the tail."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= budget:
        return text
    marker = "\n[transcript excerpt omitted]\n"
    overhead = len(tokenizer.encode(marker, add_special_tokens=False)) * (windows + 1)
    available = budget - overhead - head_tokens - tail_tokens
    if available < windows or min(head_tokens, tail_tokens, windows) <= 0:
        raise ValueError("Text budget too small to preserve instructions; increase MAX_PROMPT_TOKENS")
    width = available // windows
    middle_start, middle_stop = head_tokens, len(ids) - tail_tokens
    starts = np.linspace(middle_start, max(middle_start, middle_stop - width), windows).astype(int)
    spans = [ids[:head_tokens]] + [ids[start:start + width] for start in starts] + [ids[-tail_tokens:]]
    return marker.join(tokenizer.decode(part, skip_special_tokens=False) for part in spans)


class InputBuilder:
    def __init__(self, processor, cfg):
        self.processor, self.cfg = processor, cfg
        self.reader = MediaReader(cfg.media_cache_groups, getattr(cfg, "media_cache", None))

    def build(self, row):
        from PIL import Image
        media = self.reader.read(row)
        audios, audio_meta, videos, video_meta = [], [], [], []
        rate = self.processor.feature_extractor.sampling_rate
        for blob in media["audios"]:
            waveform, info = decode_audio(blob, rate, getattr(self.cfg, "audio_max_seconds", self.cfg.media_max_seconds))
            audios.append(waveform)
            audio_meta.append(info)
        for blob in media["videos"]:
            frames, info = decode_video(blob, self.cfg.video_frames, getattr(self.cfg, "video_max_seconds", self.cfg.media_max_seconds), self.cfg.video_max_pixels, getattr(self.cfg, "video_sampling", "segment_starts"))
            videos.append(frames)
            video_meta.append(info)
        images = [Image.open(io.BytesIO(blob)).convert("RGB") for blob in media["images"]]
        values = {"audio": audios, "video": videos, "image": images}
        offsets = {key: 0 for key in values}
        removed = {key: 0 for key in values}
        content = []
        for part in re.split(r"(<audio>|<video>|<image>)", row["problem"]):
            match = re.fullmatch(r"<(audio|video|image)>", part)
            if match:
                kind = match[1]
                if offsets[kind] >= len(values[kind]):
                    # A declared modality is not evidence that media is available.
                    # Empty/null fields are valid; never send a placeholder without its tensor.
                    removed[kind] += 1
                    continue
                content.append({"type": kind})
                offsets[kind] += 1
            elif part:
                content.append({"type": "text", "text": part})
        # Some releases have media columns but no literal placeholders. Add them explicitly.
        missing = [{"type": kind} for kind in ("video", "audio", "image")
                   for _ in range(len(values[kind]) - offsets[kind])]
        content = missing + content + [{"type": "text", "text": FORMAT_PROMPT}]
        messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        kwargs = {"text": [text], "return_tensors": "pt", "padding": True}
        if audios:
            kwargs.update(audio=audios, sampling_rate=rate)
        if videos:
            # Transformers 4.57.3 accepts a scalar FPS; set per-video timing below.
            # Audio/video are separate tokens, so this does not affect token expansion.
            kwargs.update(videos=videos, videos_kwargs={
                "fps": video_meta[0]["sampling_fps"], "do_sample_frames": False,
                "size": {"shortest_edge": getattr(self.cfg, "video_min_pixels", 56 * 56), "longest_edge": self.cfg.video_max_pixels},
                "use_audio_in_video": False,
            })
        if images:
            kwargs.update(images=images, images_kwargs={"min_pixels": 56 * 56,
                                                        "max_pixels": self.cfg.image_max_pixels})
        inputs = dict(self.processor(**kwargs))
        length = inputs["input_ids"].shape[-1]
        original_length = length
        compression = []
        while length > self.cfg.max_prompt_tokens:
            choices = [(len(self.processor.tokenizer.encode(part["text"], add_special_tokens=False)), i)
                       for i, part in enumerate(content[:-1]) if part["type"] == "text"]
            if not choices:
                raise ValueError(f"Media alone exceeds MAX_PROMPT_TOKENS for {row['uid']}; reduce media budgets")
            old_tokens, index = max(choices)
            budget = old_tokens - (length - self.cfg.max_prompt_tokens) - 32
            content[index]["text"] = compress_text(content[index]["text"], budget, self.processor.tokenizer,
                self.cfg.text_head_tokens, self.cfg.text_tail_tokens, self.cfg.text_windows)
            compression.append({"old_text_tokens": old_tokens, "new_text_budget": budget})
            kwargs["text"] = [self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)]
            inputs = dict(self.processor(**kwargs))
            new_length = inputs["input_ids"].shape[-1]
            if new_length >= length:
                raise ValueError("Text excerpting failed to reduce prompt length")
            length = new_length
        if videos:
            inputs["video_second_per_grid"] = torch.tensor(
                [2.0 / info["sampling_fps"] for info in video_meta], dtype=torch.float32)
        return inputs, {"prompt_tokens": length, "audio": audio_meta, "video": video_meta,
                        "original_prompt_tokens": original_length, "text_compression": compression,
                        "images": len(images),
                        "declared_modality": row.get("modality_signature"),
                        "effective_modality": "T" + ("A" if audios else "") + ("V" if videos else "") + ("I" if images else ""),
                        "removed_empty_placeholders": removed,
                        "inserted_placeholders": len(missing)}
