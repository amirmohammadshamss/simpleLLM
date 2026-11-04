import torch
from torch import nn
from torch.nn import functional as F
from .gpt import Embedding
from .sampling import GenerationMixin
from types import SimpleNamespace


class Bigram(GenerationMixin, nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.table = Embedding(vocab_size, vocab_size)
        self.config = SimpleNamespace(context_length=1)

    def forward(self, ids, targets=None):
        logits = self.table(ids)
        loss = None if targets is None else F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss

    @torch.no_grad()
    def fit_counts(self, tokens, smoothing=0.1):
        """Fit smoothed empirical conditional counts using training tokens only."""
        if tokens.ndim != 1 or len(tokens) < 2 or smoothing <= 0:
            raise ValueError("Need at least two training tokens and positive smoothing")
        size = self.table.weight.shape[0]
        tokens = tokens.to(device="cpu", dtype=torch.long)
        if tokens.min() < 0 or tokens.max() >= size:
            raise ValueError("Training token ID out of range")
        counts = torch.bincount(tokens[:-1] * size + tokens[1:], minlength=size * size).reshape(size, size).float()
        counts.add_(smoothing).log_()
        self.table.weight.copy_(counts.to(self.table.weight.device))
