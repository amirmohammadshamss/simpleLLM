import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from minigpt.tokenizer.serialization import tokenizer_from_state, tokenizer_hash


def load_prepared(directory, holdout="validation"):
    if holdout not in ("validation", "test"):
        raise ValueError("Holdout must be validation or test")
    directory = Path(directory)
    raw = (directory / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    if manifest["version"] != 1 or manifest["tokenizer_fit_split"] != "train":
        raise ValueError("Invalid prepared dataset manifest")
    tokenizer = tokenizer_from_state(manifest["tokenizer"])
    if tokenizer_hash(tokenizer) != manifest["tokenizer_sha256"] or tokenizer.vocab_size != manifest["actual_vocab_size"]:
        raise ValueError("Tokenizer integrity check failed")
    streams, all_ids = {}, set()
    for name in ("train", "validation", "test"):
        info = manifest["splits"][name]
        for suffix, hash_key in (("npy", "tokens_sha256"), ("ids.json", "ids_sha256")):
            if hashlib.sha256((directory / f"{name}.{suffix}").read_bytes()).hexdigest() != info[hash_key]:
                raise ValueError(f"{name} artifact hash mismatch")
        ids = json.loads((directory / f"{name}.ids.json").read_text())
        if len(ids) != len(set(ids)) or all_ids.intersection(ids):
            raise ValueError("Document IDs overlap across splits")
        all_ids.update(ids)
        array = np.load(directory / f"{name}.npy", allow_pickle=False)
        if array.ndim != 1 or len(array) != info["tokens"] or array.dtype != np.uint32 or array.max() >= tokenizer.vocab_size:
            raise ValueError(f"Invalid {name} token array")
        if name in ("train", holdout):
            streams[name] = torch.from_numpy(array.astype(np.int64))
    return tokenizer, streams["train"], streams[holdout], {"dataset_manifest_sha256": hashlib.sha256(raw).hexdigest()}
