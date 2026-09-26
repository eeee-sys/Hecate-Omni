"""One-time native media extraction and actual-presence audit; no decoding/features."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from .data import MediaReader, atomic_json, load_index


def warm_cache(rows, directory, workers):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["file"], row["row_group"])].append(row)
    counts, mismatches, datasets = Counter(), [], defaultdict(Counter)
    def one(batch):
        entries = MediaReader(cache_dir=directory).manifest(batch[0])
        result = []
        for row in batch:
            entry = entries[row["row_offset"]]
            signature = "T" + ("A" if entry["audios"] else "") + ("V" if entry["videos"] else "") + ("I" if entry["images"] else "")
            missing = [kind[:-1] for kind in ("audios", "videos", "images")
                       if not entry[kind] and f"<{kind[:-1]}>" in row["problem"]]
            result.append((row["uid"], row["dataset"], signature, missing))
        return result
    # QA cache is also used for group-disjoint holdout construction.
    ordered = sorted(groups.values(), key=lambda batch: not any(r["reward_type"] == "qa" for r in batch))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(one, batch) for batch in ordered]
        for i, future in enumerate(as_completed(futures)):
            for uid, dataset, signature, missing in future.result():
                counts[signature] += 1
                datasets[dataset][signature] += 1
                if missing:
                    mismatches.append({"uid": uid, "missing_placeholder_media": missing})
            if (i + 1) % 10 == 0:
                print(f"Media cache ready: {i + 1}/{len(futures)} row groups", flush=True)
    return {"samples": len(rows), "row_groups": len(groups), "actual_modalities": dict(counts),
            "datasets": {k: dict(v) for k, v in datasets.items()},
            "stale_placeholder_rows": len(mismatches), "stale_placeholder_examples": mismatches[:100]}


def main():
    parser = argparse.ArgumentParser()
    for name in ("data", "index", "cache", "splits", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    rows = []
    for split in args.splits.split(","):
        rows.extend(load_index(args.data, split, args.index)[0])
    atomic_json(args.output, warm_cache(rows, args.cache, args.workers))


if __name__ == "__main__":
    main()
