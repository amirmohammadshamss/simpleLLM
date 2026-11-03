# MiniGPT Lab

An independently implemented decoder-only GPT learning project. All model weights start randomly. Character and byte-level BPE tokenizers, local data preparation, GPT/Bigram models, training, resumable checkpoints, evaluation, and offline generation are implemented. An educational Tiny Shakespeare experiment is available locally. The main TinyStories corpus, hardware benchmark, and substantial main-model training remain outstanding.

## Setup

Use Python 3.11–3.13 (3.12 recommended):

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
sh scripts/smoke_test.sh
```

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

Resume uses the original total-step budget, model/data configuration, and batch size. Use `--stop-after 200` to save and exit early while keeping a longer original `--steps` scheduling budget. Checkpoints are saved at evaluation intervals (50 steps by default), so unexpected interruption may lose work since the latest save. Existing checkpoint directories require explicit resume. Checkpoints embed tokenizer, optimizer, scheduler, and RNG/batch-generator state. Load only trusted project checkpoints, because full training checkpoints use Python pickle.

`--device auto` selects CUDA, then MPS, then CPU. Explicit unavailable devices fail clearly. CPU/MPS results need not be bitwise identical. Prompts must fit the configured token context; generation subsequently uses a rolling context window. Temperature zero selects greedy decoding. BPE generation stops at EOS; the character model has no EOS token. BPE generation displays invalid/incomplete UTF-8 using replacement characters, while round-trip decoding remains strict by default.

## Local document data and BPE

Supply UTF-8 JSONL containing one `{"text": "story"}` object per line. Preparation rejects empty/corrupted text, removes exact duplicates globally, splits by document before tokenization, fits BPE on training only, and writes split arrays plus document IDs and integrity/provenance hashes. Documents are packed within each split with EOS boundaries. Near-duplicate review and source/license verification remain required before using a real corpus.

```sh
python -m minigpt.data.prepare --source data/stories.jsonl --format jsonl --tokenizer bpe --vocab-size 8192 --output data/stories-bpe --source-url SOURCE --revision REVISION --license LICENSE
python -m minigpt.training.train --dataset data/stories-bpe --config configs/tinystories.yaml --output checkpoints/stories --steps 2000 --device cpu
```

The second command requires an actual 8,192-token vocabulary and sufficiently long splits; preparation reports the achieved vocabulary and training rejects mismatches. It is an example, not a recommendation to start substantial main-model training before the required benchmark. `configs/tinystories.yaml` defines the proposed 17,538,816-parameter architecture. The trainer can also accept independent `--text` and `--validation-text` files for character experiments.

`BPETokenizer.fit(training_documents, vocab_size=8192)` learns ordered byte-pair merges without crossing document boundaries. It reserves BOS/EOS, treats literal special-token strings as ordinary text, and covers arbitrary valid UTF-8 without an unknown token. Tests include more than 1,000 round trips and serialization equivalence. Its simple repeated-counting trainer is educational; large-corpus performance is unbenchmarked.

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
