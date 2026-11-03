"""Download the pinned Tiny Shakespeare educational text, never model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = "6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e"
URL = f"https://raw.githubusercontent.com/karpathy/char-rnn/{REVISION}/data/tinyshakespeare/input.txt"
SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/tinyshakespeare.txt"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a fresh path")
    with urllib.request.urlopen(URL, timeout=60) as response:
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError("Unexpected dataset size")
    raw.decode("utf-8")
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError("Downloaded text does not match the pinned source hash")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    print(json.dumps({"url": URL, "revision": REVISION, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}))


if __name__ == "__main__":
    main()
