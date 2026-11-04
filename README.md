# MiniGPT Lab

An independently implemented decoder-only GPT learning project. All model weights start randomly. Character and byte-level BPE tokenizers, reproducible local data preparation, GPT/Bigram models, resumable training, evaluation, inference export, and offline generation are implemented. The project includes educational Shakespeare experiments and a pinned TinyStories subset for its 17.5M-parameter main model. All twelve version-one acceptance criteria have evidence, including [passing hosted CI](https://github.com/amirmohammadshamss/simpleLLM/actions/runs/41904835821). See [the implementation report](reports/implementation_status.md) for results and optional follow-ups.

## Setup

Use Python 3.11–3.13 (3.12 recommended):

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
sh scripts/smoke_test.sh
```

To reproduce the recorded local dependency versions, install `requirements-lock.txt` first, then `pip install --no-deps --no-build-isolation -e .`. The lock records the tested Python 3.12 environment; other platforms may require platform-appropriate PyTorch wheels. For loss charts, use `pip install -e '.[dev,reports]'`.

The CPU smoke run uses embedded synthetic text, performs 120 fixed-batch optimizer steps, requires cross-entropy below 0.1 and a loss reduction over 90%, and checks checkpoint restoration. It needs no dataset or network after installation. This verifies mechanics, not generalization.

## Educational training

Download the pinned educational corpus and prepare it before training:

```sh
python -m minigpt.data.download
python -m minigpt.data.prepare --source data/raw/tinyshakespeare.txt --output data/shakespeare-char --source-url https://raw.githubusercontent.com/karpathy/char-rnn/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt --revision 6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e
python -m minigpt.training.train --dataset data/shakespeare-char --config configs/toy.yaml --output checkpoints/toy --steps 2000 --batch-size 4 --eval-every 200 --device cpu
python -m minigpt.training.train --dataset data/shakespeare-char --config configs/toy.yaml --output checkpoints/toy --steps 2000 --batch-size 4 --eval-every 200 --device cpu --resume checkpoints/toy/latest.pt
python -m minigpt.generate --checkpoint checkpoints/toy/best.pt --prompt 'Once' --max-new-tokens 100 --temperature 0.8 --top-k 40 --device cpu
python -m minigpt.evaluation.evaluate --checkpoint checkpoints/toy/best.pt --dataset data/shakespeare-char --output reports/validation.json
python -m minigpt.evaluation.samples --checkpoint checkpoints/toy/best.pt --output reports/samples.json
```

Prepared output directories must be empty. The character vocabulary is fitted on training only; unseen validation characters raise an error. Tiny Shakespeare uses documented contiguous 80/10/10 character ranges. These are educational segments of shared works, rather than independent story documents. See [the data card](reports/data_card.md).

Resume uses the original total-step budget, model/data configuration, and batch size. Use `--stop-after 200` to save and exit early while keeping a longer original `--steps` scheduling budget. Evaluation and latest saves use configured intervals or `--eval-every` / `--save-every` overrides. Unexpected interruption may lose work since the latest save. Existing checkpoint directories require explicit resume. Checkpoints embed tokenizer, optimizer, scheduler, and RNG/batch-generator state. Load only trusted project checkpoints, because full training checkpoints use Python pickle. Each session persists environment, Git revision/dirty state, configuration, runtime overrides, and starting step.

`--device auto` selects CUDA, then MPS, then CPU. Explicit unavailable devices fail clearly. CPU/MPS results need not be bitwise identical. Prompts must fit the configured token context; generation subsequently uses a rolling context window. Temperature zero selects greedy decoding. BPE generation stops at EOS; the character model has no EOS token. BPE generation displays invalid/incomplete UTF-8 using replacement characters, while round-trip decoding remains strict by default.

## Local document data and BPE

Supply UTF-8 JSONL containing one `{"text": "story"}` object per line. Preparation rejects empty/corrupted text, removes exact duplicates globally, splits by document before tokenization, fits BPE on training only, and writes split arrays plus document IDs and integrity/provenance hashes. Documents are packed within each split with EOS boundaries. Near-duplicate review and source/license verification remain required before using a real corpus.

```sh
python -m minigpt.data.prepare --source data/stories.jsonl --format jsonl --tokenizer bpe --vocab-size 8192 --output data/stories-bpe --source-url SOURCE --revision REVISION --license LICENSE
python -m minigpt.training.train --dataset data/stories-bpe --config configs/tinystories.yaml --output checkpoints/stories --steps 2000 --device cpu
```

The second command requires an actual 8,192-token vocabulary and sufficiently long splits; preparation reports the achieved vocabulary and training rejects mismatches. It is an example, not a recommendation to start substantial main-model training before the required benchmark. `configs/tinystories.yaml` defines the proposed 17,538,816-parameter architecture. The trainer can also accept independent `--text` and `--validation-text` files for character experiments.

`BPETokenizer.fit(training_documents, vocab_size=8192)` retains the simple educational trainer. `backend="indexed"` uses project-owned weighted piece counts, linked lists, indexed pair occurrences, and a priority queue. Its lossless Unicode pretokenization pattern is versioned with the vocabulary. Differential tests compare indexed merges against reference training on identical pieces, and compare heap encoding with ordered reference merging. BOS/EOS and literal special-token strings remain distinct. The real 8,192-token artifact passed 1,006 Unicode round trips and save/load checks.

## Reproduce the TinyStories experiment

The source is the first 50,000 delimiter-terminated stories from the pinned original `TinyStories-train.txt`, rather than a download of the full dataset. Read [the data-use record](reports/tinystories_license.md) and [data card](reports/data_card.md) for provenance, transformations, attribution, and sampling limits.

```sh
python -m minigpt.data.tinystories --documents 50000 --output data/raw/tinystories-50000.jsonl
python -m minigpt.data.prepare --source data/raw/tinystories-50000.jsonl --output data/tinystories-bpe-8192 --format jsonl --tokenizer bpe --bpe-backend indexed --vocab-size 8192 --source-url https://huggingface.co/datasets/roneneldan/TinyStories/resolve/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/TinyStories-train.txt --revision f54c09fd23315a6f9c86f9dc80f725de7d8f9c64 --license CDLA-Sharing-1.0
python -m minigpt.data.audit --source data/raw/tinystories-50000.jsonl --dataset data/tinystories-bpe-8192 --output reports/tinystories_near_duplicates.json
python -m minigpt.tokenizer.validate --tokenizer data/tinystories-bpe-8192/tokenizer.json --output reports/tinystories_tokenizer_validation.json
python -m minigpt.training.train --dataset data/tinystories-bpe-8192 --config configs/tinystories.yaml --output checkpoints/tinystories-main --steps 1221 --stop-after 100 --eval-every 100 --save-every 100 --device mps
python -m minigpt.training.train --dataset data/tinystories-bpe-8192 --config configs/tinystories.yaml --output checkpoints/tinystories-main --steps 1221 --eval-every 100 --save-every 100 --device mps --resume checkpoints/tinystories-main/latest.pt
python -m minigpt.evaluation.evaluate --checkpoint checkpoints/tinystories-main/best.pt --dataset data/tinystories-bpe-8192 --baseline counts --bigram-checkpoint checkpoints/tinystories-main/bigram.pt --eval-batch-size 4 --output reports/tinystories_validation.json --device mps
python -m minigpt.evaluation.evaluate --checkpoint checkpoints/tinystories-main/best.pt --dataset data/tinystories-bpe-8192 --baseline counts --bigram-checkpoint checkpoints/tinystories-main/bigram.pt --eval-batch-size 4 --split test --output reports/tinystories_test.json --device mps
python -m minigpt.evaluation.samples --checkpoint checkpoints/tinystories-main/best.pt --output reports/tinystories_samples.json
python -m minigpt.evaluation.plot --metrics checkpoints/tinystories-main/metrics.jsonl --output reports/tinystories_loss.svg
python -m minigpt.export --checkpoint checkpoints/tinystories-main/best.pt --output checkpoints/tinystories-export
python -m minigpt.generate --checkpoint checkpoints/tinystories-export/model.pt --prompt 'Once upon a time' --temperature 0.8 --top-k 40 --seed 42
```

Use fresh data/run/export directories when reproducing an existing experiment. CPU fallback is supported via `--device cpu`; MPS availability depends on the execution environment. The near-duplicate audit is a seeded spot check, not exhaustive proof of uniqueness. Main training uses 1,221 optimizer steps × 8,192 tokens = 10,002,432 processed tokens, sampled with replacement. See [the training budget](reports/training_budget.md). The larger 50–150M-token quality target is a later conditional experiment.

The count-fitted Bigram uses every training-token pair with fixed smoothing 0.1. It uses the same tokenizer, holdout, and target/window policy as GPT. Full evaluation scores every next-token target once; test scoring occurs only after checkpoint selection. Perplexities are comparable within this tokenizer, not directly against the character model.

The inference export removes optimizer/RNG states and includes tokenizer, model configuration, artifact hashes, and an offline command. Data/checkpoints remain gitignored and local; export does not publish or upload them.

## Architecture smoke and hardware benchmark

```sh
python -m minigpt.training.smoke --config configs/toy.yaml
python -m minigpt.training.smoke --config configs/tinystories.yaml --steps 200 --learning-rate 0.001
sh scripts/benchmark.sh --device mps
```

The full-architecture smoke uses a short synthetic sequence and disables dropout for controlled overfitting. The benchmark runs 10 warmup and 100 measured optimizer steps on synthetic token IDs, with configured micro-batches and accumulation, in FP32. It synchronizes the device for timing and records finite loss/gradients, step timings, throughput, and available memory measurements. MPS allocation samples can miss transient peaks, and process RSS does not capture all unified-memory use. The benchmark does not replace a real-data run or count toward the main model's training-token acceptance target.

## How the model learns

The tokenizer maps text to integer IDs. Learned token and position embeddings map IDs to vectors. Each Pre-LayerNorm block applies masked multi-head attention and a GELU feed-forward network with residual connections. The causal mask prevents access to future tokens. A final LayerNorm and tied linear head produce vocabulary logits; cross-entropy compares each position with the next token. Autograd computes gradients, and AdamW updates random-initialized weights. Generation repeatedly samples the next token and appends it to the rolling context.

See [the implementation checklist](IMPLEMENTATION_PLAN.md) and [the RFP](LLM_From_Scratch_RFP_EN.md) for remaining milestones, acceptance conditions, and scope. No general knowledge, conversational quality, or Persian generation quality is claimed.
