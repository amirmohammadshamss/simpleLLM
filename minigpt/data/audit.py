"""Reproducible near-duplicate spot check across document-level prepared splits."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re


def shingles(text):
    words = re.findall(r"\w+", text.casefold())
    return {tuple(words[i:i + 5]) for i in range(max(1, len(words) - 4))}


def audit(source, dataset, sample_size=2000, seed=42):
    source, dataset = Path(source), Path(dataset)
    manifest = json.loads((dataset / "manifest.json").read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["source"]["sha256"]:
        raise ValueError("Source does not match prepared dataset")
    split_by_id = {}
    for name in ("train", "validation", "test"):
        for key in json.loads((dataset / f"{name}.ids.json").read_text()):
            if key in split_by_id:
                raise ValueError("Exact document leakage")
            split_by_id[key] = name
    selected = set(random.Random(seed).sample(sorted(split_by_id), min(sample_size, len(split_by_id))))
    sample = {}
    for line in source.read_text().splitlines():
        document = json.loads(line)["text"]
        key = hashlib.sha256(document.encode()).hexdigest()
        if key in selected:
            sample[key] = shingles(document)
    postings = defaultdict(list)
    for key, items in sample.items():
        for item in items:
            postings[item].append(key)
    reviewed, matches, strongest = set(), [], []
    for key, items in sorted(sample.items()):
        candidates = Counter()
        for item in items:
            # Common boilerplate cannot serve as a useful candidate key.
            if len(postings[item]) <= 32:
                candidates.update(other for other in postings[item] if split_by_id[other] != split_by_id[key])
        for other, _ in sorted(candidates.items(), key=lambda item: (-item[1], item[0]))[:5]:
            pair = tuple(sorted((key, other)))
            if pair in reviewed:
                continue
            reviewed.add(pair)
            similarity = len(items & sample[other]) / max(1, len(items | sample[other]))
            result = {"documents": pair, "splits": [split_by_id[pair[0]], split_by_id[pair[1]]], "five_word_shingle_jaccard": similarity}
            strongest.append(result)
            if similarity >= 0.8:
                matches.append(result)
    return {"dataset_manifest_sha256": hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest(), "seed": seed,
            "sampled_documents": len(sample), "candidate_pairs_scored": len(reviewed), "threshold": 0.8,
            "cross_split_matches": matches, "strongest_pairs": sorted(strongest, key=lambda item: (-item["five_word_shingle_jaccard"], item["documents"]))[:10],
            "method": "seeded document sample; top 5 cross-split candidates by shared non-boilerplate five-word shingles; exact Jaccard scoring",
            "limitation": "Spot check only; does not establish absence of all near duplicates in the full corpus."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=2000)
    args = parser.parse_args()
    if args.sample_size <= 0:
        parser.error("sample-size must be positive")
    result = audit(args.source, args.dataset, args.sample_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key not in ("strongest_pairs", "cross_split_matches")}))
    if result["cross_split_matches"]:
        raise SystemExit("Potential near-duplicate leakage found; review report before training")


if __name__ == "__main__":
    main()
