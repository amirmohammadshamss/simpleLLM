import platform
import random
import subprocess
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested="auto"):
    if requested not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError("Device must be auto, cpu, mps, or cuda")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; use --device cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu")
    return torch.device(requested)


def environment(device):
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    return {"python": platform.python_version(), "torch": str(torch.__version__), "os": platform.platform(), "device": str(device), "git_commit": result.stdout.strip() if result.returncode == 0 else "uncommitted", "git_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None}
