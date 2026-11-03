"""Run the offline generation CLI on fixed prompts and save reproducible outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=Path("configs/evaluation_prompts.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    prompts = json.loads(args.prompts.read_text())
    if not isinstance(prompts, list) or not prompts or any(not isinstance(p, str) for p in prompts):
        parser.error("Prompts must be a nonempty JSON list of strings")
    results = []
    for prompt in prompts:
        command = [sys.executable, "-m", "minigpt.generate", "--checkpoint", str(args.checkpoint), "--prompt", prompt, "--max-new-tokens", str(args.max_new_tokens), "--temperature", str(args.temperature), "--top-k", str(args.top_k), "--seed", str(args.seed), "--device", args.device]
        response = subprocess.run(command, capture_output=True, text=True, check=True, timeout=120)
        results.append({"prompt": prompt, "output": response.stdout.rstrip("\n")})
    output = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), "prompts_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(), "seed": args.seed, "temperature": args.temperature, "top_k": args.top_k, "max_new_tokens": args.max_new_tokens, "device": args.device, "samples": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Saved {len(results)} CLI generations to {args.output}")


if __name__ == "__main__":
    main()
