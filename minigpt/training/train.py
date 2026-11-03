"""Train a random-initialized GPT on local text or a prepared token dataset."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
import torch
import yaml
from minigpt.data.dataset import sample_batch
from minigpt.model.gpt import GPT, GPTConfig
from minigpt.runtime import environment, seed_everything, select_device
from minigpt.tokenizer.char_tokenizer import CharTokenizer
from minigpt.training.checkpoint import save_checkpoint, load_checkpoint
from minigpt.tokenizer.serialization import tokenizer_hash, tokenizer_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", type=Path, help="Training-only UTF-8 text")
    parser.add_argument("--validation-text", type=Path, help="Independent validation-only text")
    parser.add_argument("--dataset", type=Path, help="Prepared directory containing manifest.json and split arrays")
    parser.add_argument("--config", type=Path, default=Path("configs/toy.yaml"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/toy"))
    parser.add_argument("--steps", type=int, default=1000, help="Total optimizer steps, including resumed progress")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int, help="Save and exit at this step without changing the total scheduling budget")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if min(args.steps, args.threads) <= 0:
        parser.error("steps and threads must be positive")
    if args.dataset and (args.text or args.validation_text):
        parser.error("Use --dataset or --text/--validation-text")
    if not args.dataset and not (args.text and args.validation_text):
        parser.error("Supply --dataset or both --text and --validation-text")
    if args.stop_after is not None and not 0 < args.stop_after <= args.steps:
        parser.error("stop-after must be within the total step budget")
    torch.set_num_threads(args.threads)
    settings = yaml.safe_load(args.config.read_text())
    training = settings["training"]
    args.batch_size = args.batch_size if args.batch_size is not None else training.get("micro_batch_size", 4)
    args.eval_every = args.eval_every if args.eval_every is not None else training.get("eval_every_steps", 50)
    args.save_every = args.save_every if args.save_every is not None else training.get("save_every_steps", args.eval_every)
    if min(args.batch_size, args.eval_every, args.save_every) <= 0:
        parser.error("batch-size, eval-every and save-every must be positive")
    for key in ("gradient_accumulation_steps", "gradient_clip_norm", "learning_rate"):
        if key in training and training[key] <= 0:
            raise ValueError(f"training.{key} must be positive")
    if training.get("warmup_steps", 0) < 0:
        raise ValueError("warmup_steps must be nonnegative")
    seed = training.get("seed", 42)
    seed_everything(seed)
    device = select_device(args.device or training.get("device", "auto"))
    if args.dataset:
        from minigpt.data.prepared import load_prepared
        tokenizer, train_tokens, val_tokens, provenance = load_prepared(args.dataset)
    else:
        text, validation = args.text.read_text(), args.validation_text.read_text()
        if args.text.resolve() == args.validation_text.resolve() or text == validation:
            raise ValueError("Training and validation must be separate text")
        tokenizer = CharTokenizer.fit(text)
        train_tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        val_tokens = torch.tensor(tokenizer.encode(validation), dtype=torch.long)
        provenance = {"train_sha256": hashlib.sha256(text.encode()).hexdigest(), "validation_sha256": hashlib.sha256(validation.encode()).hexdigest()}
    config_values = dict(settings["model"])
    if "vocab_size" in config_values and config_values["vocab_size"] != tokenizer.vocab_size:
        raise ValueError("Configured vocabulary size does not match the project tokenizer")
    config_values["vocab_size"] = tokenizer.vocab_size
    config = GPTConfig(**config_values)
    if min(len(train_tokens), len(val_tokens)) <= config.context_length:
        raise ValueError("Each training/validation stream must be longer than context_length")
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.get("learning_rate", 0.0003), betas=tuple(training.get("betas", [0.9, 0.95])), weight_decay=training.get("weight_decay", 0.1))
    warmup = training.get("warmup_steps", 0)
    def lr_factor(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1, (step - warmup) / max(1, args.steps - warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    batches = torch.Generator().manual_seed(seed)
    resolved = {"model": asdict(config), "training": training, "total_steps": args.steps, "batch_size": args.batch_size, "tokenizer_sha256": tokenizer_hash(tokenizer), **provenance}
    start, best = 0, float("inf")
    if args.resume:
        preview = torch.load(args.resume, map_location="cpu", weights_only=False)
        if preview["config"] != resolved or preview["tokenizer"] != tokenizer_state(tokenizer):
            raise ValueError("Resume configuration/data/tokenizer mismatch; use the original run settings")
        state = load_checkpoint(args.resume, model, optimizer, scheduler, batches)
        start, best = state["step"], state["best_validation_loss"]
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.resume and (args.output / "latest.pt").exists():
        raise ValueError("Output already contains a run; resume it or choose a fresh directory")
    tokenizer.save(args.output / "tokenizer.json")
    print(json.dumps({"environment": environment(device), "config": resolved, "parameters": sum(p.numel() for p in model.parameters())}), flush=True)
    accumulation = training.get("gradient_accumulation_steps", 1)
    begin = time.perf_counter()
    with (args.output / "metrics.jsonl").open("a") as log:
        for step in range(start, min(args.steps, args.stop_after or args.steps)):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for _ in range(accumulation):
                x, y = sample_batch(train_tokens, args.batch_size, config.context_length, batches, device)
                _, loss = model(x, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at step {step}; lower learning rate and inspect data")
                (loss / accumulation).backward()
                total_loss += loss.item() / accumulation
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), training.get("gradient_clip_norm", 1.0), error_if_nonfinite=True)
            learning_rate = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            final_step = step + 1 == min(args.steps, args.stop_after or args.steps)
            improved = False
            if (step + 1) % args.eval_every == 0 or final_step:
                model.eval()
                validation_generator = torch.Generator().manual_seed(seed + 1)
                with torch.no_grad():
                    losses = [model(*sample_batch(val_tokens, args.batch_size, config.context_length, validation_generator, device))[1].item() for _ in range(5)]
                val_loss = sum(losses) / len(losses)
                if not math.isfinite(val_loss):
                    raise FloatingPointError("Nonfinite validation loss")
                elapsed = time.perf_counter() - begin
                processed = (step + 1) * args.batch_size * accumulation * config.context_length
                metrics = {"step": step + 1, "train_loss": total_loss, "validation_loss": val_loss, "perplexity": math.exp(min(val_loss, 700)), "learning_rate": learning_rate, "gradient_norm": float(grad), "processed_tokens": processed, "elapsed_seconds": elapsed, "tokens_per_second": (step + 1 - start) * args.batch_size * accumulation * config.context_length / elapsed}
                if device.type == "mps":
                    metrics["device_allocated_bytes"] = torch.mps.current_allocated_memory()
                elif device.type == "cuda":
                    metrics["device_allocated_bytes"] = torch.cuda.memory_allocated(device)
                    metrics["peak_device_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                print(json.dumps(metrics), flush=True)
                log.write(json.dumps(metrics) + "\n")
                log.flush()
                improved = val_loss < best
                best = min(best, val_loss)
            names = []
            if (step + 1) % args.save_every == 0 or final_step:
                names.append("latest.pt")
            if improved:
                names.append("best.pt")
            for name in names:
                save_checkpoint(args.output / name, model, optimizer, scheduler, step + 1, tokenizer, resolved, batches, best_validation_loss=best)


if __name__ == "__main__":
    main()
