"""Export project-trained inference artifacts without optimizer or RNG state."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from minigpt.tokenizer.serialization import tokenizer_from_state, tokenizer_hash


def export_checkpoint(checkpoint, output):
    checkpoint, output = Path(checkpoint), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Export directory must be empty")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    tokenizer = tokenizer_from_state(state["tokenizer"])
    if tokenizer.vocab_size != state["config"]["model"]["vocab_size"]:
        raise ValueError("Checkpoint vocabulary mismatch")
    output.mkdir(parents=True, exist_ok=True)
    inference = {"format": "minigpt-inference-v1", "model": state["model"], "config": {"model": state["config"]["model"]},
                 "tokenizer": state["tokenizer"], "source_step": state["step"]}
    torch.save(inference, output / "model.pt")
    tokenizer.save(output / "tokenizer.json")
    (output / "config.json").write_text(json.dumps(inference["config"], indent=2))
    manifest = {"format": inference["format"], "source_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "source_step": state["step"], "dataset_manifest_sha256": state["config"].get("dataset_manifest_sha256"), "tokenizer_sha256": tokenizer_hash(tokenizer),
                "files": {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in ("model.pt", "tokenizer.json", "config.json")}}
    (output / "artifacts.json").write_text(json.dumps(manifest, indent=2))
    (output / "README.md").write_text("# Local MiniGPT inference export\n\nThis model was trained by this project from randomly initialized weights.\n\nRun from the installed project, without network access:\n\n```sh\npython -m minigpt.generate --checkpoint " + str(output / "model.pt") + " --prompt 'Once upon a time' --max-new-tokens 150 --temperature 0.8 --top-k 40 --device cpu\n```\n\nSee the project reports/model_card.md and reports/tinystories_license.md for scope, data, evaluation, and release considerations. No inference service is required.\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_checkpoint(args.checkpoint, args.output), indent=2))


if __name__ == "__main__":
    main()
