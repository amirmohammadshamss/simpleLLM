import torch


def sample_batch(tokens, batch_size, context_length, generator, device="cpu"):
    if tokens.ndim != 1 or len(tokens) <= context_length:
        raise ValueError("Need a one-dimensional token stream longer than context_length")
    starts = torch.randint(len(tokens) - context_length, (batch_size,), generator=generator)
    x = torch.stack([tokens[i:i + context_length] for i in starts.tolist()])
    y = torch.stack([tokens[i + 1:i + context_length + 1] for i in starts.tolist()])
    return x.to(device), y.to(device)
