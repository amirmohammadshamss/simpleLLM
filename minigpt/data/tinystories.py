"""Download a bounded, pinned prefix of TinyStories; no model weights are fetched."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
REPOSITORY = "roneneldan/TinyStories"
FILE = "TinyStories-train.txt"
DELIMITER = b"<|endoftext|>"


def extract_stories(response, count, output):
    buffer = b""
    prefix_hash = hashlib.sha256()
    consumed = 0
    for index in range(count):
        while DELIMITER not in buffer:
            chunk = response.read(1024 * 1024)
            if not chunk:
                raise ValueError("Source ended before the requested number of stories")
            buffer += chunk
        raw, buffer = buffer.split(DELIMITER, 1)
        prefix_hash.update(raw + DELIMITER)
        consumed += len(raw) + len(DELIMITER)
        story = raw.decode("utf-8").strip("\r\n")
        output.write(json.dumps({"source_index": index, "text": story}, ensure_ascii=False) + "\n")
        if (index + 1) % 5000 == 0:
            print(json.dumps({"downloaded_documents": index + 1, "consumed_source_bytes": consumed}), flush=True)
    return consumed, prefix_hash.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/tinystories-50000.jsonl"))
    parser.add_argument("--documents", type=int, default=50000)
    args = parser.parse_args()
    if args.documents <= 0:
        parser.error("documents must be positive")
    metadata_path = args.output.with_suffix(".source.json")
    temporary = args.output.with_suffix(".partial")
    if any(p.exists() for p in (args.output, metadata_path, temporary)):
        parser.error("Choose fresh output paths; existing data is not overwritten")
    tree_url = f"https://huggingface.co/api/datasets/{REPOSITORY}/tree/{REVISION}"
    with urllib.request.urlopen(tree_url, timeout=60) as response:
        files = json.load(response)
    entry = next(item for item in files if item["path"] == FILE)
    url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{FILE}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("w", encoding="utf-8", newline="\n") as output:
        consumed, prefix_hash = extract_stories(response, args.documents, output)
    metadata = {"repository": REPOSITORY, "revision": REVISION, "file": FILE, "source_url": url,
                "source_file_metadata": entry, "selection": "first N delimiter-terminated stories in pinned original train file", "documents": args.documents,
                "consumed_source_bytes": consumed, "consumed_prefix_sha256": prefix_hash, "output_sha256": hashlib.sha256(temporary.read_bytes()).hexdigest(),
                "transformation": "split <|endoftext|>; remove framing CR/LF only; serialize source index and unnormalized text as UTF-8 JSONL",
                "license": "CDLA-Sharing-1.0", "license_url": "https://cdla.dev/sharing-1-0/", "attribution": "TinyStories — Ronen Eldan and Yuanzhi Li; https://huggingface.co/datasets/roneneldan/TinyStories",
                "verification_limit": "Complete source file is not downloaded or rehashed; pinned repository/file metadata and consumed-prefix/output hashes identify this subset."}
    temporary.replace(args.output)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
