import argparse
import torch
from minigpt.model.gpt import GPT, GPTConfig
from minigpt.runtime import select_device
from minigpt.tokenizer.serialization import tokenizer_from_state


def main():
    parser = argparse.ArgumentParser(description="Generate offline from a trusted project checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tokenizer = tokenizer_from_state(state["tokenizer"])
    model = GPT(GPTConfig(**state["config"]["model"])).to(device)
    model.load_state_dict(state["model"])
    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids and hasattr(tokenizer, "bos_id"):
        prompt_ids = [tokenizer.bos_id]
    if len(prompt_ids) > model.config.context_length:
        parser.error("Prompt exceeds context length; shorten the prompt")
    if not 0 <= args.max_new_tokens <= 10000:
        parser.error("max-new-tokens must be between 0 and 10000")
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output = model.generate(ids, args.max_new_tokens, args.temperature, args.top_k, eos_id=getattr(tokenizer, "eos_id", None), seed=args.seed)
    if state["tokenizer"]["type"] == "byte_bpe":
        print(tokenizer.decode(output[0].tolist(), errors="replace"))
    else:
        print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
