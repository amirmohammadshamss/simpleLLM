"""Compare project checkpoint, random GPT, and a newly trained Bigram on validation."""
import argparse
import json
import math
import hashlib
from pathlib import Path
import torch
from minigpt.data.dataset import sample_batch
from minigpt.data.prepared import load_prepared
from minigpt.model.gpt import GPT, GPTConfig
from minigpt.model.bigram import Bigram
from minigpt.runtime import seed_everything, select_device, environment
from minigpt.tokenizer.serialization import tokenizer_state


@torch.no_grad()
def score(model, tokens, context_length, device="cpu", batch_size=1):
    if min(context_length, batch_size) <= 0:
        raise ValueError("Context length and evaluation batch size must be positive")
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    try:
        for start in range(0, len(tokens) - 1, context_length * batch_size):
            length = min(context_length * batch_size, len(tokens) - 1 - start)
            if length == context_length * batch_size:
                x = tokens[start:start + length].reshape(batch_size, context_length).to(device)
                y = tokens[start + 1:start + length + 1].reshape(batch_size, context_length).to(device)
            else:
                # Score the final partial batch without counting padding targets.
                x = torch.zeros((batch_size, context_length), dtype=torch.long, device=device)
                y = torch.full_like(x, -100)
                x.view(-1)[:length] = tokens[start:start + length].to(device)
                y.view(-1)[:length] = tokens[start + 1:start + length + 1].to(device)
            _, loss = model(x, y)
            total += loss.item() * length
            count += length
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
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--baseline", choices=["adamw", "counts"], default="adamw")
    parser.add_argument("--bigram-checkpoint", type=Path, help="Reuse an existing matching baseline, or save a newly fitted one here")
    parser.add_argument("--eval-batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.bigram_steps <= 0:
        parser.error("bigram-steps must be positive")
    torch.set_num_threads(1)
    seed_everything(42)
    device = select_device(args.device)
    tokenizer, train, validation, provenance = load_prepared(args.dataset, holdout=args.split)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if state["tokenizer"] != tokenizer_state(tokenizer) or state["config"].get("dataset_manifest_sha256") != provenance["dataset_manifest_sha256"]:
        raise ValueError("Checkpoint tokenizer/dataset provenance mismatch")
    config = GPTConfig(**state["config"]["model"])
    model = GPT(config).to(device)
    model.load_state_dict(state["model"])
    random_model = GPT(config).to(device)
    bigram = Bigram(tokenizer.vocab_size).to(device)
    baseline_info = {"method": args.baseline, "smoothing": 0.1 if args.baseline == "counts" else None,
                     "processed_training_pairs": len(train) - 1 if args.baseline == "counts" else args.bigram_steps * 4 * config.context_length}
    if args.bigram_checkpoint and args.bigram_checkpoint.exists():
        baseline = torch.load(args.bigram_checkpoint, map_location="cpu", weights_only=False)
        if baseline["tokenizer"] != tokenizer_state(tokenizer) or baseline["provenance"] != provenance or baseline["baseline_info"] != baseline_info:
            raise ValueError("Bigram checkpoint tokenizer/data/method mismatch")
        bigram.load_state_dict(baseline["model"])
    else:
        if args.baseline == "counts":
            bigram.fit_counts(train)
        else:
            optimizer = torch.optim.AdamW(bigram.parameters(), lr=0.03)
            generator = torch.Generator().manual_seed(42)
            for _ in range(args.bigram_steps):
                optimizer.zero_grad(set_to_none=True)
                x, y = sample_batch(train, 4, config.context_length, generator, device)
                _, loss = bigram(x, y)
                loss.backward()
                optimizer.step()
        if args.bigram_checkpoint:
            args.bigram_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": bigram.state_dict(), "tokenizer": tokenizer_state(tokenizer), "provenance": provenance, "baseline_info": baseline_info}, args.bigram_checkpoint)
    result = {"environment": environment(device), "checkpoint": str(args.checkpoint), "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), "checkpoint_step": state["step"], **provenance,
              "split": args.split, "evaluation_policy": "every target once in sequential nonoverlapping context windows; same tokenizer and targets for all models", "eval_batch_size": args.eval_batch_size,
              "bigram_training": baseline_info,
              "trained_gpt": score(model, validation, config.context_length, device, args.eval_batch_size), "random_gpt": score(random_model, validation, config.context_length, device, args.eval_batch_size),
              "trained_bigram": score(bigram, validation, config.context_length, device, args.eval_batch_size)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
