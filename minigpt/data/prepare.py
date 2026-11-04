"""Prepare local UTF-8 text or document JSONL without network access."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
from minigpt.tokenizer.char_tokenizer import CharTokenizer
from minigpt.tokenizer.bpe_tokenizer import BPETokenizer
from minigpt.tokenizer.serialization import tokenizer_hash, tokenizer_state


def digest(data):
    return hashlib.sha256(data).hexdigest()


def split_documents(documents, seed=42):
    unique = {}
    removed = {"empty_or_corrupted": 0, "exact_duplicates": 0}
    for document in documents:
        if not isinstance(document, str) or not document.strip() or "\ufffd" in document or "\x00" in document:
            removed["empty_or_corrupted"] += 1
            continue
        try:
            encoded = document.encode("utf-8")
        except UnicodeEncodeError:
            removed["empty_or_corrupted"] += 1
            continue
        key = digest(encoded)
        if key in unique:
            removed["exact_duplicates"] += 1
        unique[key] = document
    if len(unique) < 10:
        raise ValueError("Need at least 10 unique valid documents for three splits")
    keys = sorted(unique)
    random.Random(seed).shuffle(keys)
    holdout = max(1, len(keys) // 10)
    partitions = {"train": keys[2 * holdout:], "validation": keys[:holdout], "test": keys[holdout:2 * holdout]}
    return {name: [unique[key] for key in ids] for name, ids in partitions.items()}, partitions, removed


def prepare(source, output, mode="text", tokenizer_kind="char", vocab_size=8192, seed=42, source_url=None, revision=None, license_name="not reviewed", bpe_backend="reference"):
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty; prepared artifacts are immutable")
    raw = source.read_bytes()
    text = raw.decode("utf-8")
    if mode == "text":
        # Educational contiguous character ranges, not independent story documents.
        boundaries = [0, int(len(text) * 0.8), int(len(text) * 0.9), len(text)]
        splits = {name: [text[a:b]] for name, a, b in zip(("train", "validation", "test"), boundaries, boundaries[1:])}
        ids = {name: [digest(documents[0].encode())] for name, documents in splits.items()}
        removed = {}
        policy = {"type": "contiguous_character_ranges", "boundaries": boundaries, "limitation": "Educational text segments share works; not a document-level generalization benchmark."}
    else:
        documents = [json.loads(line)["text"] for line in text.splitlines() if line.strip()]
        splits, ids, removed = split_documents(documents, seed)
        policy = {"type": "document_hash_dedup_then_seeded_80_10_10", "near_duplicates": "not reviewed"}
    begin = time.perf_counter()
    tokenizer = CharTokenizer.fit("".join(splits["train"])) if tokenizer_kind == "char" else BPETokenizer.fit(splits["train"], vocab_size, backend=bpe_backend, progress=lambda data: print(json.dumps(data), flush=True))
    tokenizer_fit_seconds = time.perf_counter() - begin
    arrays = {}
    for name, documents in splits.items():
        tokens = []
        for document in documents:
            tokens.extend(tokenizer.encode(document, eos=True) if tokenizer_kind == "bpe" else tokenizer.encode(document))
        if len(tokens) < 2:
            raise ValueError(f"{name} has insufficient tokens")
        arrays[name] = np.asarray(tokens, dtype=np.uint32)
    manifest = {"version": 1, "source": {"file": source.name, "sha256": digest(raw), "url": source_url, "revision": revision, "license": license_name},
                "seed": seed, "split_policy": policy, "removed": removed, "tokenizer": tokenizer_state(tokenizer), "tokenizer_sha256": tokenizer_hash(tokenizer),
                "requested_vocab_size": vocab_size if tokenizer_kind == "bpe" else None, "actual_vocab_size": tokenizer.vocab_size,
                "tokenizer_fit_split": "train", "packing": "documents packed within each split; BPE appends EOS; never crosses split boundaries", "splits": {}}
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output / "tokenizer.json")
    for name, array in arrays.items():
        np.save(output / f"{name}.npy", array, allow_pickle=False)
        (output / f"{name}.ids.json").write_text(json.dumps(ids[name]))
        manifest["splits"][name] = {"documents": len(splits[name]), "tokens": len(array), "raw_bytes": sum(len(d.encode()) for d in splits[name]), "tokens_sha256": digest((output / f"{name}.npy").read_bytes()), "ids_sha256": digest((output / f"{name}.ids.json").read_bytes())}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    (output / "preparation_metrics.json").write_text(json.dumps({"tokenizer_fit_seconds": tokenizer_fit_seconds, "bpe_backend": bpe_backend, "total_seconds": time.perf_counter() - begin}, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=["text", "jsonl"], default="text")
    parser.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-url")
    parser.add_argument("--revision")
    parser.add_argument("--license", default="not reviewed")
    parser.add_argument("--bpe-backend", choices=["reference", "indexed"], default="reference")
    args = parser.parse_args()
    manifest = prepare(args.source, args.output, args.format, args.tokenizer, args.vocab_size, args.seed, args.source_url, args.revision, args.license, args.bpe_backend)
    print(json.dumps({"vocab_size": manifest["actual_vocab_size"], "splits": manifest["splits"]}, indent=2))


if __name__ == "__main__":
    main()
