from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from .sampling import GenerationMixin


@dataclass
class GPTConfig:
    vocab_size: int
    n_layers: int = 4
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    context_length: int = 128
    dropout: float = 0.1
    tie_embeddings: bool = True

    def __post_init__(self):
        for key in ("vocab_size", "n_layers", "d_model", "n_heads", "d_ff", "context_length"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


class Embedding(nn.Module):
    def __init__(self, count, width):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(count, width) * 0.02)

    def forward(self, ids):
        return self.weight[ids]


class CausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.projection = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("mask", torch.tril(torch.ones(config.context_length, config.context_length, dtype=torch.bool)), persistent=False)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.reshape(batch, length, self.heads, width // self.heads).transpose(1, 2) for t in (q, k, v)]
        scores = q @ k.transpose(-2, -1) / math.sqrt(width // self.heads)
        scores = scores.masked_fill(~self.mask[:length, :length], float("-inf"))
        out = self.dropout(scores.softmax(dim=-1)) @ v
        return self.dropout(self.projection(out.transpose(1, 2).contiguous().view(batch, length, width)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attention = CausalAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(nn.Linear(config.d_model, config.d_ff), nn.GELU(), nn.Linear(config.d_ff, config.d_model), nn.Dropout(config.dropout))

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        return x + self.ff(self.ln2(x))


class GPT(GenerationMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokens = Embedding(config.vocab_size, config.d_model)
        self.positions = Embedding(config.context_length, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*(Block(config) for _ in range(config.n_layers)))
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._initialize)
        if config.tie_embeddings:
            self.head.weight = self.tokens.weight

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids, targets=None):
        if ids.ndim != 2 or not 0 < ids.shape[1] <= self.config.context_length:
            raise ValueError("Expected [batch, length] tokens within context limit")
        x = self.tokens(ids) + self.positions(torch.arange(ids.shape[1], device=ids.device))
        logits = self.head(self.norm(self.blocks(self.dropout(x))))
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, self.config.vocab_size), targets.reshape(-1))
        return logits, loss

