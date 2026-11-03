import os
from pathlib import Path
import random
import tempfile
import numpy as np
import torch
from minigpt.tokenizer.serialization import tokenizer_state


def save_checkpoint(path, model, optimizer, scheduler, step, tokenizer, config, batch_generator, **extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"version": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "step": step, "config": config,
             "tokenizer": tokenizer_state(tokenizer),
             "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "batch": batch_generator.get_state()}, **extra}
    if torch.cuda.is_available():
        state["rng"]["cuda"] = torch.cuda.get_rng_state_all()
    if next(model.parameters()).device.type == "mps":
        state["rng"]["mps"] = torch.mps.get_rng_state()
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as output:
            torch.save(state, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path, model, optimizer=None, scheduler=None, batch_generator=None):
    # Full optimizer/RNG checkpoints use pickle: load only trusted local project files.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["version"] != 1:
        raise ValueError("Unsupported checkpoint version")
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch"])
    if "cuda" in state["rng"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    if "mps" in state["rng"] and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["rng"]["mps"])
    if batch_generator is not None:
        batch_generator.set_state(state["rng"]["batch"])
    return state
