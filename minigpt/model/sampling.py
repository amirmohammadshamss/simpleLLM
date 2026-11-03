import math
import torch


class GenerationMixin:
    """Seeded autoregressive sampling shared by GPT and Bigram."""
    @torch.no_grad()
    def generate(self, ids, max_new_tokens=100, temperature=0.8, top_k=None, eos_id=None, seed=42):
        if max_new_tokens < 0 or not math.isfinite(temperature) or temperature < 0 or (top_k is not None and top_k <= 0):
            raise ValueError("Invalid generation parameters")
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
            raise ValueError("Generation requires one nonempty prompt")
        # A CPU sampling generator also works on MPS, which lacks generator support.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                logits, _ = self(ids[:, -self.config.context_length:])
                scores = logits[:, -1, :]
                if temperature == 0:
                    next_id = scores.argmax(dim=-1, keepdim=True)
                else:
                    scores = scores / temperature
                    if top_k is not None:
                        cutoff = scores.topk(min(top_k, scores.shape[-1])).values[:, -1:]
                        scores = scores.masked_fill(scores < cutoff, float("-inf"))
                    next_id = torch.multinomial(scores.softmax(-1).cpu(), 1, generator=generator).to(ids.device)
                ids = torch.cat((ids, next_id), dim=1)
                if eos_id is not None and next_id.item() == eos_id:
                    break
        finally:
            self.train(was_training)
        return ids
