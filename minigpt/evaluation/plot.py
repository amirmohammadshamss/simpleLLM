"""Create standalone training-loss figures from recorded metrics."""
import argparse
import json
from pathlib import Path
import os
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output figure path, e.g. reports/main_loss.svg")
    args = parser.parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "minigpt-matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "minigpt-cache"))
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    points = {}
    for line in args.metrics.read_text().splitlines():
        item = json.loads(line)
        points[item["step"]] = item
    points = [points[key] for key in sorted(points)]
    if not points:
        parser.error("No metric records")
    fig, axes = plt.subplots(figsize=(8, 4.5), layout="constrained")
    axes.plot([p["processed_tokens"] / 1e6 for p in points], [p["train_loss"] for p in points], marker="o", markersize=3, label="Training (last optimizer step)")
    axes.plot([p["processed_tokens"] / 1e6 for p in points], [p["validation_loss"] for p in points], marker="o", markersize=3, label="Validation (fixed sampled batches)")
    axes.set(xlabel="Processed training tokens (millions)", ylabel="Cross-entropy (nats/token)", title="MiniGPT training and validation loss")
    axes.grid(alpha=0.2)
    axes.legend(frameon=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, metadata={"Description": "Recorded losses from fixed-tokenizer training; validation uses sampled batches, not the full holdout."})
    if args.output.suffix.lower() == ".svg":
        args.output.write_text("\n".join(line.rstrip() for line in args.output.read_text().splitlines()) + "\n")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
