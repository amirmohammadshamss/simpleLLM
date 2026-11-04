import json
import subprocess
import sys
import torch
import pytest
from minigpt.data.prepare import prepare, split_documents
from minigpt.data.prepared import load_prepared
from minigpt.model.bigram import Bigram
from minigpt.tokenizer.serialization import tokenizer_hash


def corpus(path):
    documents = [f"Once upon a time, bird number {i} flew home. فارسی می‌روم 🙂\n" for i in range(30)]
    path.write_text("\n".join(json.dumps({"text": document}, ensure_ascii=False) for document in documents + [documents[0], "", "\ufffd"]))
    return documents


def test_preparation_reproducibility_and_integrity(tmp_path):
    source = tmp_path / "stories.jsonl"
    documents = corpus(source)
    first = prepare(source, tmp_path / "one", "jsonl", "bpe", 280)
    second = prepare(source, tmp_path / "two", "jsonl", "bpe", 280)
    assert first == second
    assert first["removed"] == {"empty_or_corrupted": 2, "exact_duplicates": 1}
    tokenizer, train, validation, _ = load_prepared(tmp_path / "one")
    assert tokenizer_hash(tokenizer) == first["tokenizer_sha256"]
    assert tokenizer.eos_id in train and tokenizer.eos_id in validation
    splits, ids, _ = split_documents(documents)
    assert not set(ids["train"]).intersection(ids["validation"] + ids["test"])
    from minigpt.tokenizer.bpe_tokenizer import BPETokenizer
    assert BPETokenizer.fit(splits["train"], 280).merges == tokenizer.merges
    (tmp_path / "one" / "train.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_prepared(tmp_path / "one")


def test_bpe_training_resume_and_cli(tmp_path):
    source = tmp_path / "stories.jsonl"
    corpus(source)
    dataset = tmp_path / "dataset"
    prepare(source, dataset, "jsonl", "bpe", 280)
    config = tmp_path / "tiny.yaml"
    config.write_text("model:\n  n_layers: 1\n  d_model: 16\n  n_heads: 2\n  d_ff: 32\n  context_length: 8\n  dropout: 0.1\ntraining:\n  seed: 42\n  learning_rate: 0.001\n  gradient_accumulation_steps: 2\n  warmup_steps: 1\n")
    def run(output, extra=()):
        command = [sys.executable, "-m", "minigpt.training.train", "--dataset", str(dataset), "--config", str(config), "--output", str(output), "--steps", "4", "--batch-size", "1", "--eval-every", "2", "--device", "cpu", *extra]
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    run(full)
    run(resumed, ["--stop-after", "2"])
    run(resumed, ["--resume", str(resumed / "latest.pt")])
    a = torch.load(full / "latest.pt", weights_only=False)
    b = torch.load(resumed / "latest.pt", weights_only=False)
    assert a["step"] == b["step"] == 4
    assert a["scheduler"] == b["scheduler"]
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)
    result = subprocess.run([sys.executable, "-m", "minigpt.generate", "--checkpoint", str(full / "best.pt"), "--prompt", "Once", "--max-new-tokens", "4", "--device", "cpu"], check=True, capture_output=True, text=True, timeout=60)
    assert result.stdout.startswith("Once")
    from minigpt.export import export_checkpoint
    exported = tmp_path / "export"
    artifact = export_checkpoint(full / "best.pt", exported)
    inference = torch.load(exported / "model.pt", weights_only=True)
    assert "optimizer" not in inference and "rng" not in inference
    assert artifact["tokenizer_sha256"] == a["config"]["tokenizer_sha256"]
    restored = subprocess.run([sys.executable, "-m", "minigpt.generate", "--checkpoint", str(exported / "model.pt"), "--prompt", "Once", "--max-new-tokens", "4", "--device", "cpu"], check=True, capture_output=True, text=True, timeout=60)
    assert result.stdout == restored.stdout


def test_bigram_generation():
    model = Bigram(4)
    prompt = torch.tensor([[1, 2]])
    assert torch.equal(model.generate(prompt, 5, seed=7), model.generate(prompt, 5, seed=7))


def test_evaluation_counts_every_target_once():
    from minigpt.evaluation.evaluate import score
    model = Bigram(4)
    with torch.no_grad():
        model.table.weight.zero_()
    tokens = torch.arange(24) % 4
    result = score(model, tokens, context_length=8)
    assert result["target_tokens"] == 23
    assert result["perplexity"] == pytest.approx(4)
    assert model.training
    batched = score(model, tokens, context_length=8, batch_size=4)
    assert batched["target_tokens"] == result["target_tokens"]
    assert batched["loss"] == pytest.approx(result["loss"])


def test_evaluation_batching_preserves_real_model_loss():
    from minigpt.evaluation.evaluate import score
    from minigpt.model.gpt import GPT, GPTConfig
    model = GPT(GPTConfig(vocab_size=8, n_layers=1, d_model=16, n_heads=2, d_ff=32, context_length=8))
    tokens = torch.arange(52) % 8
    sequential = score(model, tokens, context_length=8)
    batched = score(model, tokens, context_length=8, batch_size=4)
    assert sequential["target_tokens"] == batched["target_tokens"] == 51
    assert sequential["loss"] == pytest.approx(batched["loss"], abs=1e-6)
