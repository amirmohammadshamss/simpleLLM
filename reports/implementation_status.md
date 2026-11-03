# Initial implementation milestone

Verified on November 3, 2026, with Python 3.12.15 and PyTorch 2.15.0 on macOS ARM64. The repository has no initial commit, so environment metadata reports `uncommitted`.

## Delivered

- Package setup, toy/main YAML configurations, seed/device/environment helpers, and CPU CI workflow.
- Training-only character vocabulary and standalone deterministic byte-level BPE with BOS/EOS, Unicode support, and serialization.
- Independently written embeddings, explicit masked multi-head attention, Pre-LayerNorm blocks, learned positions, GELU, residuals, and tied LM-head weights.
- Bigram forward/loss baseline and character-model trainer with AdamW, accumulation, clipping, warmup/cosine schedule, validation metrics, and best/latest checkpoints.
- Atomic checkpoint writes containing model, optimizer, scheduler, configuration, character tokenizer, global step, and RNG/batch-generator state.
- Offline character-model CLI with greedy/temperature/top-k decoding and seeded sampling.

## Verification

Commands:

```sh
.venv/bin/pytest -q
.venv/bin/python -m minigpt.training.smoke
```

All **9 tests passed** in 11.87 seconds. Coverage includes future-token leakage, gradients, tied weights, batch shifting, sampled/greedy generation, EOS stopping, Bigram learning, exact next-update CPU checkpoint resume, an actual train/save/load/CLI subprocess cycle, and over 1,000 BPE Unicode round trips plus serialization equivalence.

The offline CPU smoke run completed 120 optimizer steps on one fixed synthetic batch. Cross-entropy fell from **2.643312 to 0.000932**, passing the recorded requirement of CE below 0.1 and a reduction above 90%. This tiny-model test validates backpropagation; it does not measure story quality or main-model generalization. Its output repeats spaces after the memorized sequence, demonstrating the limits of this test.

Instantiating the proposed main architecture reports **17,538,816 unique parameters**. MPS support is built into PyTorch, but `is_available()` returned false in the current execution environment. CPU verification passed; MPS training remains unverified.

The CI workflow is written; a hosted CI run has not been executed.

## Remaining gates

The initial implementation does not complete Phase 1: Tiny Shakespeare acquisition/provenance, educational partitions, a full prototype experiment, and baseline text generation remain. The standalone BPE trainer is simple and unbenchmarked on large corpora. BPE is not yet integrated into training/checkpoint generation. No TinyStories subset, licensed data package, main-model training run, hardware benchmark, baseline comparison, or trained model release has been produced.

Next: complete the educational data workflow and prototype experiment, then implement reproducible document-level TinyStories preparation and connect BPE to the training pipeline. Main-model acceptance criteria remain open until their required evidence exists.
