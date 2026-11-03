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
