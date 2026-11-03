import json
from pathlib import Path


class CharTokenizer:
    def __init__(self, characters):
        self.characters = list(characters)
        if len(set(self.characters)) != len(self.characters):
            raise ValueError("Duplicate vocabulary entries")
        self.stoi = {c: i for i, c in enumerate(self.characters)}

    @classmethod
    def fit(cls, text):
        if not text:
            raise ValueError("Training text is empty")
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.characters)

    def encode(self, text):
        try:
            return [self.stoi[c] for c in text]
        except KeyError as exc:
            raise ValueError(f"Character absent from training vocabulary: {exc.args[0]!r}") from exc

    def decode(self, ids):
        if any(i < 0 or i >= self.vocab_size for i in ids):
            raise ValueError("Token ID out of range")
        return "".join(self.characters[i] for i in ids)

    def save(self, path):
        Path(path).write_text(json.dumps({"version": 1, "characters": self.characters}, ensure_ascii=False))

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data["version"] != 1:
            raise ValueError("Unsupported tokenizer version")
        return cls(data["characters"])
