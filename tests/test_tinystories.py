import io
import hashlib
import json
import pytest
from minigpt.data.tinystories import extract_stories


class SmallChunks(io.BytesIO):
    def read(self, size):
        return super().read(min(size, 3))


def test_subset_extraction_at_chunk_and_unicode_boundaries():
    prefix = "\nFirst story.\n<|endoftext|>\nمی‌روم 🙂\n<|endoftext|>".encode()
    output = io.StringIO()
    consumed, checksum = extract_stories(SmallChunks(prefix + b"ignored third story"), 2, output)
    assert consumed == len(prefix)
    assert checksum == hashlib.sha256(prefix).hexdigest()
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert records == [{"source_index": 0, "text": "First story."}, {"source_index": 1, "text": "می‌روم 🙂"}]
    with pytest.raises(ValueError, match="ended"):
        extract_stories(io.BytesIO(b"unterminated"), 1, io.StringIO())


def test_near_duplicate_audit_detects_changed_text_across_splits(tmp_path):
    from minigpt.data.audit import audit
    from minigpt.data.prepare import prepare
    sentences = ["A little bird was flying home to see a kind friend in the green forest on a sunny day." + " " * i for i in range(12)]
    source = tmp_path / "stories.jsonl"
    source.write_text("\n".join(json.dumps({"text": text}) for text in sentences))
    dataset = tmp_path / "prepared"
    prepare(source, dataset, "jsonl", "bpe", 280)
    first = audit(source, dataset, sample_size=12)
    assert first == audit(source, dataset, sample_size=12)
    assert first["cross_split_matches"]
    assert all(item["five_word_shingle_jaccard"] == 1 for item in first["cross_split_matches"])
