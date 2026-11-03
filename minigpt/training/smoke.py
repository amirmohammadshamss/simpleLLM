"""Offline CPU fixed-batch overfitting and checkpoint recovery verification."""
import json
from pathlib import Path
import tempfile
import torch
import argparse
import yaml
from minigpt.model.gpt import GPT, GPTConfig
from minigpt.runtime import seed_everything, environment
from minigpt.tokenizer.char_tokenizer import CharTokenizer
from minigpt.training.checkpoint import save_checkpoint, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optionally verify the full toy/main architecture")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    args = parser.parse_args()
    if args.steps <= 0 or args.learning_rate <= 0:
        parser.error("steps and learning-rate must be positive")
    torch.set_num_threads(1)
    seed_everything(42)
    text = "Once upon a time. "
    tokenizer = CharTokenizer.fit(text)
    config = GPTConfig(tokenizer.vocab_size, n_layers=1, d_model=32, n_heads=4, d_ff=64, context_length=len(text) - 1, dropout=0)
    if args.config:
        values = dict(yaml.safe_load(args.config.read_text())["model"])
        # Preserve main vocabulary size when verifying the main architecture.
        values.setdefault("vocab_size", tokenizer.vocab_size)
        values["dropout"] = 0
        config = GPTConfig(**values)
    model = GPT(config)
    tokens = torch.tensor([tokenizer.encode(text)])
    x, y = tokens[:, :-1], tokens[:, 1:]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    initial = model(x, y)[1].item()
    for _ in range(args.steps):
        optimizer.zero_grad()
        loss = model(x, y)[1]
        if not torch.isfinite(loss):
            raise FloatingPointError("Smoke loss became nonfinite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
    final = model(x, y)[1].item()
    # Fixed before accepting this test: <0.1 CE and >90% loss reduction.
    if not final < min(0.1, initial * 0.1):
        raise AssertionError(f"Fixed-batch overfit failed: {initial} -> {final}")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "smoke.pt"
        save_checkpoint(path, model, optimizer, scheduler, args.steps, tokenizer, {}, torch.Generator())
        restored = GPT(config)
        load_checkpoint(path, restored)
        torch.testing.assert_close(model(x)[0], restored(x)[0])
    generated = model.generate(x[:, :5], 30, temperature=0)[0].tolist()
    sample = tokenizer.decode(generated) if max(generated) < tokenizer.vocab_size else str(generated)
    print(json.dumps({"environment": environment("cpu"), "initial_loss": initial, "final_loss": final, "steps": args.steps, "learning_rate": args.learning_rate, "parameters": sum(p.numel() for p in model.parameters()), "threshold": "CE <0.1 and >90% reduction", "sample": sample}))


if __name__ == "__main__":
    main()
