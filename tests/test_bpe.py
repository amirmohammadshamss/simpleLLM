import random
import pytest
from minigpt.tokenizer.bpe_tokenizer import BPETokenizer


def test_unicode_roundtrip_and_serialization(tmp_path):
    training = ["Once upon a time, a small bird sang.", "فارسی می‌روم!\n", "Hello 🙂 world"]
    tokenizer = BPETokenizer.fit(training, vocab_size=300)
    tokenizer.save(tmp_path / "bpe.json")
    restored = BPETokenizer.load(tmp_path / "bpe.json")
    rng = random.Random(42)
    alphabet = list("ab .,!?\n\tفارسی‌🙂你好")
    cases = ["", "<bos><eos>", "می‌روم", "\u0000", "\U0010ffff"]
    cases += ["".join(rng.choices(alphabet, k=rng.randrange(1, 100))) for _ in range(1000)]
    for text in cases:
        assert tokenizer.decode(tokenizer.encode(text)) == text
        assert restored.encode(text) == tokenizer.encode(text)
        assert tokenizer.decode(tokenizer.encode(text, bos=True, eos=True)) == text
    assert restored.merges == tokenizer.merges
    assert BPETokenizer.fit(training, vocab_size=300).merges == tokenizer.merges


def test_bpe_document_boundaries_and_unknown_bytes():
    tokenizer = BPETokenizer.fit(["a", "b"], 300)
    assert tokenizer.merges == []
    assert tokenizer.decode(tokenizer.encode("Unseen bytes: 🙂 فارسی")) == "Unseen bytes: 🙂 فارسی"
    with pytest.raises(ValueError):
        tokenizer.decode([1000])
    with pytest.raises(ValueError):
        BPETokenizer.fit([], 300)
