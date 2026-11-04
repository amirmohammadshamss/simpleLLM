import random
import pytest
from minigpt.tokenizer.bpe_tokenizer import BPETokenizer
from minigpt.tokenizer.bpe_tokenizer import merge_pair
from minigpt.tokenizer.fast_bpe import pieces
from minigpt.tokenizer.serialization import tokenizer_state, tokenizer_from_state, tokenizer_hash


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


@pytest.mark.parametrize("seed", range(12))
def test_indexed_training_matches_weighted_reference(seed, tmp_path):
    rng = random.Random(seed)
    documents = ["".join(rng.choices("aaaaabbbcc !?12_\nفارسی🙂", k=80)) for _ in range(12)]
    documents += ["aaaaa aaaaa ababab aaa", documents[0], documents[0]]
    chunks = [piece for document in documents for piece in pieces(document)]
    reference = BPETokenizer.fit(chunks, 310)
    indexed = BPETokenizer.fit(documents, 310, backend="indexed")
    assert indexed.merges == reference.merges
    assert "".join(pieces(documents[0])) == documents[0]
    for document in documents + ["می‌روم\t你好🙂", "<bos><eos>"]:
        expected = []
        for piece in pieces(document):
            ids = list(piece.encode())
            for token, pair in enumerate(reference.merges, 258):
                ids = merge_pair(ids, pair, token)
            expected.extend(ids)
        assert indexed.encode(document) == expected
        assert indexed.decode(expected) == document
    indexed.save(tmp_path / "tokenizer.json")
    restored = BPETokenizer.load(tmp_path / "tokenizer.json")
    assert restored.encode(documents[0]) == indexed.encode(documents[0])
    assert tokenizer_hash(tokenizer_from_state(tokenizer_state(indexed))) == tokenizer_hash(indexed)
    assert tokenizer_hash(indexed) != tokenizer_hash(BPETokenizer(indexed.merges))


def test_heap_encoder_preserves_legacy_merge_order():
    rng = random.Random(17)
    tokenizer = BPETokenizer.fit(["aaaaaa abababcd فارسی🙂 " * 5], 300)
    for _ in range(1000):
        text = "".join(rng.choices("aaaabb cd🙂فارسی", k=rng.randrange(80)))
        expected = list(text.encode())
        for token, pair in enumerate(tokenizer.merges, 258):
            expected = merge_pair(expected, pair, token)
        assert tokenizer.encode(text) == expected
