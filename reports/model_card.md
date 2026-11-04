# MiniGPT Lab — TinyStories 17.5M

Local experimental model, trained November 4, 2026. Its intended use is learning how a decoder-only language model is implemented and trained, and experimenting with short English story continuations. It is not a dependable conversational, factual, reasoning, or Persian-language model.

## Architecture and origin

The independently written PyTorch GPT has **17,538,816 unique parameters**: 8 Pre-LayerNorm Transformer blocks, hidden size 384, 6 attention heads, feed-forward size 1,536, GELU, learned positions, context length 512, dropout 0.1 during training, explicit causal attention, and tied token/LM-head weights. All model weights began randomly; no pretrained weights, pretrained tokenizer, or inference API was used.

The project-trained byte BPE contains 8,192 tokens, including BOS=256 and EOS=257. Its versioned Unicode-piece splitting preserves arbitrary valid UTF-8. EOS terminates stories and generation. Unicode encoding coverage does not establish generation skill in every language.

## Training data and process

Training uses a pinned prefix subset of TinyStories's original train file, with exact duplicates and empty/corrupted samples removed before seeded document-level splitting. There are 39,912 training stories / 15,340,001 training tokens and independent validation/test stories. BPE was fitted only on training. Sources, source/file IDs, transformation hashes, license handling, and the near-duplicate spot-check limitation are documented in [data_card.md](data_card.md) and [tinystories_license.md](tinystories_license.md).

The FP32 MPS run on Apple M4 / 16 GiB used AdamW, peak learning rate 0.0003, 200-step warmup, cosine decay, clipping 1.0, micro-batch 4, accumulation 4, context 512, and seed 42. It stopped after 100 diagnostic steps, then resumed the same scheduling budget and finished at step **1,221**, totaling **10,002,432 processed tokens**. Windows were sampled with replacement. Recorded active training-loop time was about **24.9 minutes**, including periodic evaluation/checkpoint work but excluding the interruption gap. The larger 50–150M-token quality target was not attempted.

Best-checkpoint selection used fixed sampled validation batches every 100 steps and at completion. Step 1,221 was selected. The test stream was reserved until that choice was fixed. CPU resume tests reproduce an uninterrupted run exactly; CPU/MPS bitwise identity is not claimed.

## Evaluation and limitations

Full validation CE is **2.06312 nats/token** (perplexity **7.87051**) versus **3.15693** (perplexity **23.49843**) for a count-fitted Bigram and **8.91407** for an untrained random GPT. Each used the same BPE, target stream, and nonoverlapping context-window policy. The Bigram uses all training pairs with fixed smoothing 0.1. Results are in [tinystories_validation.json](tinystories_validation.json); reserved test results are recorded in [tinystories_test.json](tinystories_test.json) after selection.

On the reserved test stream, GPT CE is **2.06705** (perplexity **7.90146**) over 1,925,502 targets, versus Bigram CE **3.15885** and random GPT CE **8.91536**. No weights, sampling settings, or checkpoint choice were adjusted using these test results.

Twenty fixed-prompt outputs, generated offline on CPU with temperature 0.8, top-k 40, seed 42, and at most 150 new tokens, are saved in [tinystories_samples.json](tinystories_samples.json). Outputs show story vocabulary and short grammatical phrases, but can contradict themselves, confuse entities, repeat words, stop prematurely, and produce broken syntax. One sample calls Lily both a girl and a car. Human inspection supplements rather than replaces the loss measurements.

The source prefix can be distribution-biased. The 2,000-document near-duplicate spot check is not exhaustive. The training budget is small relative to the recommended quality target. No safety/factuality benchmark, general reasoning evaluation, or Persian-language generation evaluation was performed. Perplexity cannot be directly compared to the character-tokenized Shakespeare experiment.

## Local artifacts and offline use

Training checkpoints are `checkpoints/tinystories-main/{best,latest}.pt`, with full optimizer/scheduler/RNG/tokenizer state. The inference export is `checkpoints/tinystories-export/model.pt` (approximately 67 MiB), alongside `tokenizer.json`, `config.json`, `artifacts.json`, and an offline README. Weights and data are gitignored and remain local; the export command performs no upload.

```sh
python -m minigpt.generate --checkpoint checkpoints/tinystories-export/model.pt --prompt "Once upon a time" --max-new-tokens 150 --temperature 0.8 --top-k 40 --seed 42 --device cpu
```

[Export hashes](tinystories_export.json) link the inference artifact to selected training checkpoint SHA-256 `6f953302a1202e48338c745564e5f7ff8b864371bc647aaa3cc148fa8883239a`. The inference checkpoint contains only model tensors, configuration, and tokenizer metadata; full training checkpoints use Python pickle and should be loaded only from trusted project files. The export preserves the same generated output as its training checkpoint in automated tests.

Setup/reproduction commands are in [README.md](../README.md), exact local dependency versions in [requirements-lock.txt](../requirements-lock.txt), and measurements in [experiments.md](experiments.md) and [the loss chart](tinystories_loss.svg). Current code was tested in a fresh temporary Python 3.12 environment. Hosted GitHub CI has not been run against this working tree.
