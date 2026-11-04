"""Validate the actual project BPE artifact, including serialization and Unicode."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import tempfile
from minigpt.tokenizer.bpe_tokenizer import BPETokenizer
from minigpt.tokenizer.serialization import tokenizer_hash


def validate(path, samples=1000):
    tokenizer = BPETokenizer.load(path)
    rng = random.Random(42)
    cases = ["", "Once upon a time.\n", "فارسی می‌روم!\u200c", "<bos><eos>", "你好 🙂\t\x00", "\U0010ffff"]
    for _ in range(samples):
        chars = []
        for _ in range(rng.randrange(1, 48)):
            point = rng.randrange(0x110000)
            if 0xD800 <= point <= 0xDFFF:
                point = 0x200C
            chars.append(chr(point))
        cases.append("".join(chars))
    with tempfile.TemporaryDirectory() as directory:
        saved = Path(directory) / "tokenizer.json"
        tokenizer.save(saved)
        restored = BPETokenizer.load(saved)
        for text in cases:
            ids = tokenizer.encode(text)
            if tokenizer.decode(ids) != text or restored.encode(text) != ids or restored.decode(ids) != text:
                raise AssertionError("Tokenizer round-trip/serialization mismatch")
            if tokenizer.decode(tokenizer.encode(text, bos=True, eos=True)) != text:
                raise AssertionError("Special-token handling mismatch")
    return {"samples_passed": len(cases), "seed": 42, "vocab_size": tokenizer.vocab_size, "tokenizer_sha256": tokenizer_hash(tokenizer), "artifact_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "checks": ["lossless Unicode round trips", "identical saved/restored token IDs", "special-token boundaries", "Persian/ZWNJ/astral/whitespace/control characters", "literal special-token strings remain ordinary text"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.tokenizer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
