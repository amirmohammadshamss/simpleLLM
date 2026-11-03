"""Benchmark training kernels on synthetic IDs; this is not a corpus training run."""
import argparse
import json
from pathlib import Path
import platform
import resource
import statistics
import time
import torch
import yaml
from minigpt.model.gpt import GPT, GPTConfig
from minigpt.runtime import select_device, seed_everything, environment


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/tinystories.yaml"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps <= 0 or args.warmup_steps < 0:
        parser.error("steps must be positive and warmup-steps nonnegative")
    torch.set_num_threads(1)
    seed_everything(42)
    settings = yaml.safe_load(args.config.read_text())
    config = GPTConfig(**settings["model"])
    training = settings["training"]
    micro = training.get("micro_batch_size", 4)
    accumulation = training.get("gradient_accumulation_steps", 4)
    device = select_device(args.device)
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.get("learning_rate", 0.0003), betas=tuple(training.get("betas", [0.9, 0.95])), weight_decay=training.get("weight_decay", 0.1))
    ids = torch.randint(config.vocab_size, (micro, config.context_length + 1)).to(device)
    durations, losses = [], []
    sampled_peak, driver_peak = 0, 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(args.warmup_steps + args.steps):
        synchronize(device)
        begin = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(accumulation):
            _, loss = model(ids[:, :-1], ids[:, 1:])
            if not torch.isfinite(loss):
                raise FloatingPointError("Benchmark loss became nonfinite")
            (loss / accumulation).backward()
            total += loss.item() / accumulation
            if device.type == "mps":
                sampled_peak = max(sampled_peak, torch.mps.current_allocated_memory())
                driver_peak = max(driver_peak, torch.mps.driver_allocated_memory())
        torch.nn.utils.clip_grad_norm_(model.parameters(), training.get("gradient_clip_norm", 1), error_if_nonfinite=True)
        optimizer.step()
        synchronize(device)
        elapsed = time.perf_counter() - begin
        if device.type == "mps":
            sampled_peak = max(sampled_peak, torch.mps.current_allocated_memory())
            driver_peak = max(driver_peak, torch.mps.driver_allocated_memory())
        if step >= args.warmup_steps:
            durations.append(elapsed)
            losses.append(total)
        if (step + 1) % 10 == 0:
            print(json.dumps({"completed_steps_including_warmup": step + 1, "step_seconds": elapsed, "loss": total}), flush=True)
    tokens_per_step = micro * accumulation * config.context_length
    throughput = args.steps * tokens_per_step / sum(durations)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {"environment": environment(device), "config": settings, "dtype": "float32", "parameters": sum(p.numel() for p in model.parameters()),
              "synthetic_data": True, "learning_rate_policy": "constant configured learning rate; no scheduler; kernel benchmark only",
              "warmup_steps_excluded": args.warmup_steps, "measured_steps": args.steps, "tokens_per_optimizer_step": tokens_per_step,
              "measured_processed_tokens": args.steps * tokens_per_step, "elapsed_measured_seconds": sum(durations), "tokens_per_second": throughput,
              "mean_step_seconds": statistics.mean(durations), "median_step_seconds": statistics.median(durations),
              "first_ten_mean_seconds": statistics.mean(durations[:10]), "last_ten_mean_seconds": statistics.mean(durations[-10:]),
              "process_peak_rss_bytes": rss if platform.system() == "Darwin" else rss * 1024,
              "sampled_device_allocated_peak_bytes": sampled_peak if device.type == "mps" else None,
              "sampled_driver_allocated_peak_bytes": driver_peak if device.type == "mps" else None,
              "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
              "memory_limitations": "MPS samples occur after backward/optimizer updates and can miss transient peaks. Driver allocation includes cache. Process RSS excludes some device allocations. Total system unified-memory peak is not measured.",
              "finite_loss_and_gradients": True, "initial_measured_loss": losses[0], "final_measured_loss": losses[-1],
              "estimated_compute_hours": {str(budget): budget / throughput / 3600 for budget in (10_000_001, 50_000_000, 150_000_000)},
              "estimate_limitations": "Synthetic fixed batches; excludes real data loading, evaluation, checkpoints, tokenization, and thermal changes beyond this short run. Not a duration guarantee."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
