"""Compare project checkpoint, random GPT, and a newly trained Bigram on validation."""
import argparse
import json
import math
from pathlib import Path
import torch
from minigpt.data.dataset import sample_batch
from minigpt.data.prepared import load_prepared
from minigpt.model.gpt import GPT, GPTConfig
from minigpt.model.bigram import Bigram
from minigpt.runtime import seed_everything, select_device, environment
from minigpt.tokenizer.serialization import tokenizer_state


@torch.no_grad()
def score(model, tokens, context_length, device="cpu"):
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    try:
        for start in range(0, len(tokens) - 1, context_length):
            end = min(start + context_length, len(tokens) - 1)
            x = tokens[start:end].unsqueeze(0).to(device)
            y = tokens[start + 1:end + 1].unsqueeze(0).to(device)
            _, loss = model(x, y)
            total += loss.item() * y.numel()
            count += y.numel()
    finally:
        model.train(was_training)
    if not count or not math.isfinite(total):
        raise ValueError("Evaluation requires finite loss on nonempty targets")
    loss = total / count
    return {"loss": loss, "perplexity": math.exp(min(loss, 700)), "target_tokens": count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bigram-steps", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.bigram_steps <= 0:
        parser.error("bigram-steps must be positive")
    torch.set_num_threads(1)
    seed_everything(42)
    device = select_device(args.device)
    tokenizer, train, validation, provenance = load_prepared(args.dataset)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if state["tokenizer"] != tokenizer_state(tokenizer) or state["config"].get("dataset_manifest_sha256") != provenance["dataset_manifest_sha256"]:
        raise ValueError("Checkpoint tokenizer/dataset provenance mismatch")
    config = GPTConfig(**state["config"]["model"])
    model = GPT(config).to(device)
    model.load_state_dict(state["model"])
    random_model = GPT(config).to(device)
    bigram = Bigram(tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(bigram.parameters(), lr=0.03)
    generator = torch.Generator().manual_seed(42)
    for _ in range(args.bigram_steps):
        optimizer.zero_grad(set_to_none=True)
        x, y = sample_batch(train, 4, config.context_length, generator, device)
        _, loss = bigram(x, y)
        loss.backward()
        optimizer.step()
    result = {"environment": environment(device), "checkpoint": str(args.checkpoint), "checkpoint_step": state["step"], **provenance,
              "split": "validation", "evaluation_policy": "every target once in sequential nonoverlapping context windows; same tokenizer and targets for all models",
              "bigram_training_steps": args.bigram_steps, "bigram_processed_tokens": args.bigram_steps * 4 * config.context_length,
              "trained_gpt": score(model, validation, config.context_length, device), "random_gpt": score(random_model, validation, config.context_length, device),
              "trained_bigram": score(bigram, validation, config.context_length, device)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
