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

## Follow-up milestone — November 4, 2026

Completed pinned Tiny Shakespeare download/preparation, character and BPE prepared-data training, tokenizer-aware checkpoints and offline generation, shared Bigram/GPT sampling, full-validation comparison, and 20 fixed-prompt CLI samples. Prepared datasets include source/split/tokenizer hashes, document IDs, exact deduplication for JSONL corpora, training-only tokenizer fitting, and integrity checks. Resumed CPU BPE training with accumulation and warmup produces the exact same final weights as uninterrupted training.

The updated suite passes **14 tests** in 15.96 seconds, including main-configuration gradients and future-token leakage. The original CPU smoke was rerun and passed. Full toy and main architecture overfitting also passed at the settings documented in [experiments.md](experiments.md).

The 2,000-step educational run processed 1.024 million character tokens. Full-validation GPT loss was 2.41143 versus 2.51638 for Bigram and 4.19669 for random GPT. See [experiments.md](experiments.md), [data_card.md](data_card.md), and the saved JSON/JSONL evidence. Generated text remains poor; no story-quality acceptance claim is made.

The earlier MPS availability failure was sandbox-specific. Outside the sandbox, MPS is available. Hardware inspection reports **Apple M4, 17,179,869,184 bytes (16 GiB) RAM**. A 17.5M-parameter FP32 synthetic benchmark completed 10 warmup and 100 measured optimizer steps at context 512, micro-batch 4, accumulation 4, with finite loss/gradients. Throughput was **10,067.6 tokens/second**, mean step time **0.8137 seconds**. The last ten steps did not slow relative to the first ten; longer-term thermal behavior is unknown.

Sampled MPS driver allocation peaked at 2,214,477,824 bytes, sampled tensor allocation at 387,592,704 bytes, and process peak RSS at 555,319,296 bytes. These measurements overlap and must not be added. Sampling can miss transient peaks, and total unified-memory peak remains unmeasured. Kernel-only estimates are about 1.38 hours for 50M tokens and 4.14 hours for 150M tokens, excluding actual data, evaluation/checkpoints, preprocessing, and longer-term performance changes. See [hardware_benchmark.json](hardware_benchmark.json).

## Main-model delivery — November 4, 2026

Downloaded a bounded 50,000-story prefix from a pinned TinyStories revision, preserving upstream file identity and consumed-prefix/output hashes. Documented the published license and project data handling in [tinystories_license.md](tinystories_license.md). Preparation removed 12 invalid/empty samples and 100 exact duplicates before seeded document-level train/validation/test splitting. The training partition has 39,912 stories / 15,340,001 tokens.

Implemented a project-owned indexed weighted-piece BPE trainer and heap encoder. Differential tests match the reference trainer/encoder under the same piece boundaries, and serialized artifacts preserve the pretokenization policy. The actual 8,192-token tokenizer passed 1,006 Unicode/Persian/ZWNJ round trips and serialization checks. Training took 2.26 seconds and preparation 5.05 seconds for this subset. A 2,000-document near-duplicate spot check scored 517 cross-split candidate pairs and found no match at the fixed 0.8 threshold; the audit is not exhaustive.

The 17,538,816-parameter random-initialized main model completed **1,221 optimizer steps / 10,002,432 processed tokens** on MPS. The 100-step real-data diagnostic was resumed with all training states and the same total scheduling budget. Best/latest checkpoints, BPE/configuration, source fingerprints, session metadata, and metric logs are retained locally. Active-loop time was approximately 24.9 minutes, including periodic validation/checkpoint work, with concurrent evaluation affecting throughput.

The selected step-1,221 checkpoint achieves full-validation loss **2.06312**, beating the all-training-pairs Bigram at **3.15693** and random GPT at **8.91407**, with matching BPE and evaluation targets. Reserved-test loss is **2.06705** (perplexity **7.90146**), compared with Bigram **3.15885**. Test scoring occurred after checkpoint selection and did not change it.

Delivered the [model card](model_card.md), updated [data card](data_card.md), [experiment report](experiments.md), [SVG](tinystories_loss.svg)/[PNG](tinystories_loss.png) loss charts, [20 offline CLI samples](tinystories_samples.json), dependency lock, and a local inference export with [artifact hashes](tinystories_export.json). The export at `checkpoints/tinystories-export/model.pt` is about 67 MiB and excludes optimizer/RNG state. Tests verify that exported and full training checkpoints produce matching seeded CLI output.

Created a fresh temporary Python 3.12 environment at `/tmp/minigpt-clean-20261104`, installed the recorded dependencies and project, and ran **31 passing tests** in 26.94 seconds plus the passing CPU smoke run. The equivalent checks are configured in `.github/workflows/ci.yml`. Source, data, and artifact records remain reviewable in this working tree; no upload or external inference was performed.

## Hosted CI acceptance — November 4, 2026

**All twelve version-one acceptance criteria are verified.** The implementation was committed as `eaa3460a833d376b169779dcada1913bdd157c2e` and pushed to the existing private repository on `codex/minigpt-v1-acceptance`. Hosted [CPU checks run 41904835821](https://github.com/amirmohammadshamss/simpleLLM/actions/runs/41904835821) passed on Ubuntu/Python 3.12: **31 tests in 22.12 seconds**, followed by a successful 120-step CPU smoke run (final loss 0.000932405). Machine-readable job, step, environment, and commit evidence is in [ci_verification.json](ci_verification.json).

The existing workflow was used. GitHub rejected an attempted workflow-file update because the configured token lacks `workflow` scope; the delivery commit retains the original workflow unchanged. The optional CPU-only installation/artifact-upload proposal is preserved as [ci_workflow_proposal.yml](ci_workflow_proposal.yml) and is not active. No broader credential permission was required to verify the implementation.

Source and reports are on the private delivery branch. Raw data and trained weights remain local and gitignored. The branch has not been merged into `main`; later acceptance-record commits change documentation only.

## Optional follow-up work

The optional 50–150M-token quality experiment and further architecture optimizations remain future work. Current samples still show broken grammar, repetition, early endings, and character inconsistencies. Near-duplicate auditing is a spot check; total system unified-memory peak was not measured. Model weights are available locally, and no public model/data release has been made.
