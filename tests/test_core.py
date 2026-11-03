import pytest
import torch
import json
import subprocess
import sys
from minigpt.data.dataset import sample_batch
from minigpt.model.gpt import GPT, GPTConfig
from minigpt.model.bigram import Bigram
from minigpt.tokenizer.char_tokenizer import CharTokenizer
from minigpt.training.checkpoint import save_checkpoint, load_checkpoint
from minigpt.runtime import seed_everything


@pytest.fixture(autouse=True)
def reproducible():
    torch.set_num_threads(1)
    seed_everything(42)


def tiny():
    return GPT(GPTConfig(vocab_size=8, n_layers=1, d_model=16, n_heads=2, d_ff=32, context_length=8, dropout=0.1))


def test_character_roundtrip(tmp_path):
    text = "Hello! فارسی می‌روم\n🙂"
    tokenizer = CharTokenizer.fit(text)
    tokenizer.save(tmp_path / "tokenizer.json")
    restored = CharTokenizer.load(tmp_path / "tokenizer.json")
    for i in range(1000):
        sample = (text * 2)[i % len(text):i % len(text) + 7]
        assert tokenizer.decode(tokenizer.encode(sample)) == sample
        assert restored.encode(sample) == tokenizer.encode(sample)
    with pytest.raises(ValueError, match="absent"):
        tokenizer.encode("字")


def test_causality_shapes_and_gradients():
    model = tiny().eval()
    x = torch.randint(8, (2, 8))
    changed = x.clone()
    changed[:, 4:] = (changed[:, 4:] + 1) % 8
    logits, loss = model(x, x)
    assert logits.shape == (2, 8, 8)
    torch.testing.assert_close(logits[:, :4], model(changed)[0][:, :4], atol=1e-7, rtol=1e-6)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.head.weight is model.tokens.weight
    with pytest.raises(ValueError):
        model(torch.zeros(1, 9, dtype=torch.long))


def test_batch_shift():
    x, y = sample_batch(torch.arange(30), 4, 8, torch.Generator().manual_seed(42))
    assert torch.equal(x + 1, y)


def test_generation():
    model = tiny()
    prompt = torch.tensor([[1, 2]])
    first = model.generate(prompt, 10, top_k=3, seed=7)
    assert torch.equal(first, model.generate(prompt, 10, top_k=3, seed=7))
    assert model.training
    assert first.shape == (1, 12)
    greedy = model.generate(prompt, 1, temperature=0)
    assert model.generate(prompt, 5, temperature=0, eos_id=greedy[0, -1].item()).shape == (1, 3)


def test_resume_exact_next_update(tmp_path):
    model = tiny()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    generator = torch.Generator().manual_seed(42)
    tokens = torch.arange(64) % 8
    def update(m, o, s, g):
        x, y = sample_batch(tokens, 2, 8, g)
        o.zero_grad()
        _, loss = m(x, y)
        loss.backward()
        o.step()
        s.step()
        return loss.detach()
    update(model, optimizer, scheduler, generator)
    path = tmp_path / "latest.pt"
    save_checkpoint(path, model, optimizer, scheduler, 1, CharTokenizer.fit("abcdefgh"), {}, generator)
    expected = update(model, optimizer, scheduler, generator)
    restored = tiny()
    opt2 = torch.optim.AdamW(restored.parameters(), lr=0.001)
    sched2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=1, gamma=0.9)
    gen2 = torch.Generator()
    assert load_checkpoint(path, restored, opt2, sched2, gen2)["step"] == 1
    actual = update(restored, opt2, sched2, gen2)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for a, b in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert scheduler.get_last_lr() == sched2.get_last_lr()


def test_bigram_learns():
    model = Bigram(4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    x = torch.tensor([[0, 1, 2, 3]])
    y = torch.tensor([[1, 2, 3, 0]])
    initial = model(x, y)[1].item()
    for _ in range(40):
        optimizer.zero_grad()
        loss = model(x, y)[1]
        loss.backward()
        optimizer.step()
    assert model(x, y)[1].item() < initial * 0.1


def test_training_and_offline_cli(tmp_path):
    train = tmp_path / "train.txt"
    validation = tmp_path / "validation.txt"
    config = tmp_path / "tiny.yaml"
    output = tmp_path / "run"
    train.write_text("abc cab bac " * 20)
    validation.write_text("cab abc bac " * 10)
    config.write_text("model:\n  n_layers: 1\n  d_model: 16\n  n_heads: 2\n  d_ff: 32\n  context_length: 8\n  dropout: 0\ntraining:\n  seed: 42\n  learning_rate: 0.001\n")
    command = [sys.executable, "-m", "minigpt.training.train", "--text", str(train), "--validation-text", str(validation), "--config", str(config), "--output", str(output), "--steps", "2", "--batch-size", "1", "--eval-every", "1", "--device", "cpu"]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    assert (output / "best.pt").exists()
    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert metrics[-1]["processed_tokens"] == 16
    subprocess.run(command + ["--resume", str(output / "latest.pt")], check=True, capture_output=True, text=True, timeout=60)
    result = subprocess.run([sys.executable, "-m", "minigpt.generate", "--checkpoint", str(output / "best.pt"), "--prompt", "abc", "--max-new-tokens", "5", "--device", "cpu"], check=True, capture_output=True, text=True, timeout=60)
    assert result.stdout.startswith("abc")
    assert len(result.stdout.rstrip("\n")) == 8


def test_main_configuration_correctness():
    config = GPTConfig(vocab_size=8192, n_layers=8, d_model=384, n_heads=6, d_ff=1536, context_length=512)
    model = GPT(config).eval()
    assert sum(p.numel() for p in model.parameters()) == 17538816
    x = torch.randint(8192, (1, 8))
    logits, loss = model(x, x)
    changed = x.clone()
    changed[:, 4:] = (changed[:, 4:] + 1) % 8192
    with torch.no_grad():
        torch.testing.assert_close(logits[:, :4], model(changed)[0][:, :4], atol=1e-6, rtol=1e-5)
    loss.backward()
    assert logits.shape == (1, 8, 8192)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
