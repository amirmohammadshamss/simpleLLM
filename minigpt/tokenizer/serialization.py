import hashlib
import json
from .char_tokenizer import CharTokenizer
from .bpe_tokenizer import BPETokenizer


def tokenizer_state(tokenizer):
    if isinstance(tokenizer, CharTokenizer):
        return {"type": "char", "characters": tokenizer.characters}
    if isinstance(tokenizer, BPETokenizer):
        state = {"type": "byte_bpe", "merges": [list(pair) for pair in tokenizer.merges]}
        if tokenizer.pretokenization is not None:
            state["pretokenization"] = tokenizer.pretokenization
        return state
    raise ValueError("Unsupported tokenizer")


def tokenizer_from_state(state):
    if state["type"] == "char":
        return CharTokenizer(state["characters"])
    if state["type"] == "byte_bpe":
        return BPETokenizer(state["merges"], state.get("pretokenization"))
    raise ValueError("Unsupported tokenizer type")


def tokenizer_hash(tokenizer):
    return hashlib.sha256(json.dumps(tokenizer_state(tokenizer), sort_keys=True, ensure_ascii=False).encode()).hexdigest()
