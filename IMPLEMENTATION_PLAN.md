# MiniGPT Lab — Implementation Checklist

Source: [LLM From Scratch RFP](LLM_From_Scratch_RFP_EN.md), version 1.0.

Checked items are implemented and locally verified; unchecked gates still block advancing to substantial training. Some reusable tokenizer/model components were built early. Complete phases in order and pass each decision gate before proceeding. This plan tracks implementation and delivery. Progress evidence is recorded in [reports/implementation_status.md](reports/implementation_status.md).

## Phase 0 — Foundations and project setup

Estimated effort: 2–4 part-time days.

- [ ] Review next-token prediction, embeddings, causal attention, residual connections, LayerNorm, cross-entropy, and Autograd.
- [x] Create the `minigpt/`, `configs/`, `tests/`, `scripts/`, and `reports/` structure described in RFP section 13.
- [x] Add `pyproject.toml`, dependency installation instructions, and test configuration.
- [x] Ignore downloaded data, checkpoints, caches, and generated training artifacts in Git.
- [x] Add validated configuration loading and deterministic seed setup.
- [x] Create `configs/toy.yaml` and `configs/tinystories.yaml`.
- [x] Implement device selection for CPU, available MPS, and optional CUDA, with explicit override.
- [x] Record Python/PyTorch versions, OS, device, resolved configuration, and Git commit for every run.
- [x] Establish project rules: independently written code, random model initialization, no pretrained models/tokenizers, and no external inference APIs.
- [ ] Gate: run a small tensor/autograd example and the initial tests on CPU; verify MPS availability on the target Mac.

## Phase 1 — Character tokenizer and educational model

Estimated effort: 4–7 part-time days. Depends on Phase 0.

- [ ] Download Tiny Shakespeare with recorded provenance and file hash.
- [ ] Define reproducible, nonoverlapping educational train/validation/test partitions before sequence creation; document boundaries.
- [x] Implement a training-only character vocabulary, `encode`, `decode`, and save/load behavior; define behavior for unseen characters.
- [x] Implement shifted input/target batches with context length 128.
- [ ] Implement a trainable Bigram baseline with loss calculation and text generation.
- [x] Implement project-owned token and learned position embeddings and a linear language-model head using PyTorch tensor operations/Autograd.
- [x] Implement explicit matrix-multiplication causal attention, head splitting/combining, and multi-head projection.
- [x] Implement Pre-LayerNorm Transformer blocks with GELU feed-forward layers, residual connections, and configurable dropout.
- [x] Assemble the educational GPT: 4 layers, hidden size 128, 4 heads, and feed-forward size 512.
- [x] Test tensor shapes, shifted targets, attention masking, finite gradients, and context limits.
- [x] Test that changing tokens after position `t` leaves logits at positions `≤ t` unchanged in evaluation mode.
- [x] Build a minimal CPU training loop and verify that loss decreases on a fixed batch.
- [x] Record a numerical near-memorization threshold after the smoke test and demonstrate that the fixed-batch run meets it.
- [ ] Save prototype generations and a short explanation of the text-to-next-token pipeline.
- [ ] Gate: the prototype trains, passes causal/gradient tests, overfits a fixed batch, and generates text locally. **AC02, AC03, AC05 (prototype evidence).**

## Phase 2 — Byte-level BPE and main data pipeline

Estimated effort: 4–7 part-time days. Depends on Phase 1.

- [x] Implement byte-level BPE from the 256 byte values with pair-frequency counting, deterministic tie handling, iterative merging, and ordered merge rules.
- [x] Reserve `<bos>` and `<eos>` IDs, decide whether `<pad>` is needed, and define explicit special-token handling.
- [ ] Implement UTF-8 encoding/decoding and tokenizer serialization with version/hash metadata.
- [ ] Target an 8,192-token vocabulary including special tokens; report the actual vocabulary size and validate model compatibility.
- [x] Pass at least 1,000 round-trip cases, including English, Persian, zero-width non-joiners, punctuation, whitespace, and mixed Unicode.
- [x] Verify identical token IDs and decoded output before and after saving/loading the tokenizer.
- [ ] Record TinyStories source, pinned revision, file IDs/hashes, sample counts, and license in `reports/data_card.md`.
- [ ] Review actual license and attribution/sharing requirements before redistributing data or model artifacts.
- [ ] Select a fixed, reproducible TinyStories subset; aim for 10–30 million tokens after tokenization.
- [ ] Remove empty/corrupted stories while preserving punctuation, paragraph breaks, and endings.
- [ ] Deduplicate exact stories across all candidate partitions and spot-check near-duplicates.
- [ ] Create document-level train/validation/test splits before tokenization and overlapping sequence creation; record split IDs, seed, and duplicate checks.
- [ ] Fit BPE exclusively on the training partition; save its training provenance and artifact hash.
- [ ] Encode each split independently with an explicit end-of-story policy and report exact document/token counts.
- [ ] Implement token storage and sequence batching without crossing split boundaries; document any packing across stories within a split.
- [ ] Test shifted targets, split isolation, tokenizer provenance, and reproducible preprocessing.
- [ ] Gate: versioned BPE passes all round-trip tests and independently reproducible, leakage-safe splits are ready. **AC04, AC06.**

## Phase 3 — Main architecture and reliable training

Estimated effort: 5–8 part-time days. Depends on Phases 1–2.

- [x] Configure the main GPT with vocabulary 8,192, 8 layers, hidden size 384, 6 heads, feed-forward size 1,536, and context length 512.
- [x] Tie token-embedding and LM-head weights; verify shared parameters and print the actual unique parameter count, expected around 17.5 million.
- [ ] Verify random initialization and audit the pipeline for pretrained-weight/tokenizer loading. **AC01.**
- [ ] Repeat architecture, causal-leakage, gradient, and fixed-batch overfitting tests for the main configuration. **AC02, AC03, AC05.**
- [ ] Implement AdamW with initial learning rate `0.0003`, betas `(0.9, 0.95)`, and weight decay `0.1`.
- [ ] Implement gradient accumulation, correct loss scaling, gradient clipping at `1.0`, 200-step warmup, and cosine learning-rate decay.
- [ ] Start with micro-batch size 4 and 4 accumulation steps; log the effective batch of 8,192 tokens per optimizer step.
- [ ] Configure evaluation and checkpoint intervals, initially every 500 optimizer steps.
- [ ] Log train/validation loss, perplexity, learning rate, gradient norm, processed-token count, tokens/second, elapsed time, and memory where available.
- [ ] Detect NaN/Inf loss and gradients and stop with actionable diagnostics.
- [ ] Save model, optimizer, scheduler, global step, RNG states, tokenizer, and configuration; preserve data-order state needed for meaningful resume.
- [ ] Retain separate best-validation and latest checkpoints; protect existing valid checkpoints from interrupted writes.
- [ ] Test interruption/resume and verify preserved progress, learning-rate state, optimizer state, and tokenizer compatibility using appropriate numeric tolerances. **AC09.**
- [ ] Keep mixed precision opt-in and `torch.compile` disabled by default.
- [ ] Add optional optimized attention only after explicit-attention output and gradient equivalence tests pass.
- [ ] Add automated tests and a short CPU smoke run to CI; exclude dataset downloads requiring large transfers and multi-hour training. **AC12.**
- [ ] Gate: main-model smoke training, checkpoint recovery, and CI pass before benchmarking.

## Phase 4 — Hardware benchmark and training budget

Depends on Phase 3. Full training duration remains unknown until measured.

- [ ] Benchmark 100–300 steps at the intended configuration on the M4 Mac with 16 GB unified memory.
- [ ] Measure tokens/second, time per step, peak memory, numerical stability, and sustained performance when measurable.
- [ ] Record measurement methods, device synchronization, warmup treatment, and unavailable metrics.
- [ ] If memory is insufficient, reduce micro-batch size first, then context length; record changes and repeat the benchmark.
- [ ] Estimate run duration from measured throughput and proposed processed-token budgets.
- [ ] Distinguish unique corpus tokens from total tokens processed across repeated passes.
- [ ] Define maximum runtime/token budget, validation-based stopping rules, and available disk space for checkpoints.
- [ ] Select a meaningful run exceeding 10 million processed tokens; target 50–150 million when the benchmark supports it.
- [ ] Gate: save the benchmark report and selected configuration/budget before launching the main run.

## Phase 5 — Main training and experiment tracking

Depends on Phase 4. Duration is determined by measured throughput and selected budget.

- [ ] Train the main model from random weights using the frozen training split and tokenizer.
- [ ] Complete more than 10 million processed tokens and retain logs proving the total. **AC07.**
- [ ] Monitor held-out validation loss, training stability, throughput, memory, and signs of overfitting/repetition.
- [ ] Preserve latest checkpoints throughout training and choose the best checkpoint by validation loss.
- [ ] Run controlled experiments only when results justify them; record configuration, seed, token budget, and outcome for each run.
- [ ] Produce training/validation loss charts and an experiment summary in `reports/experiments.md`.
- [ ] Expand toward 300 million processed tokens only if budget and validation trends justify it.
- [ ] Gate: the selected checkpoint, tokenizer, configuration, logs, and training-token evidence are complete and recoverable.

## Phase 6 — Evaluation, offline CLI, and delivery

Estimated effort: 3–5 part-time days. Depends on Phase 5; CLI implementation can begin after Phase 3.

- [ ] Evaluate the trained model, trained Bigram baseline, and random-initialized GPT with the same BPE tokenizer, splits, context policy, and loss aggregation.
- [ ] Demonstrate better validation loss than the Bigram baseline; if it fails, diagnose and improve training before acceptance. **AC08.**
- [ ] Report validation and final held-out test loss/perplexity; keep test data out of model selection.
- [ ] Use bits per byte on identical raw text if comparing different tokenizers; avoid direct cross-tokenizer perplexity comparisons.
- [ ] Create at least 20 fixed evaluation prompts and save generations with checkpoint, seed, and sampling settings.
- [ ] Review coherence, grammar, repetition, and failures as supplementary qualitative evidence.
- [ ] Implement `python -m minigpt.generate` with checkpoint, prompt, max-new-tokens, temperature, top-k, and seed arguments.
- [ ] Support greedy decoding, EOS stopping, validated sampling parameters, and explicit input/output/context limits.
- [ ] Verify fully offline generation after local artifacts are present and save at least three CLI-generated samples. **AC10.**
- [ ] Write `README.md` with clean-environment setup, data preparation, tokenizer training, CPU smoke, benchmark, train/resume, evaluation, and offline generation commands.
- [ ] Complete `reports/model_card.md`, `reports/data_card.md`, experiment reports, and documented limitations. **AC11.**
- [ ] Package project-trained weights, tokenizer, configurations, source, tests, hashes, benchmark results, loss charts, and sample outputs with applicable attribution/license material.
- [ ] Verify documented commands from a clean environment and confirm CI tests/CPU smoke pass. **AC12.**
- [ ] Gate: review evidence for every acceptance criterion below and resolve outstanding failures.

## Final acceptance checklist

- [ ] **AC01:** Evidence of random initialization and no pretrained-model loading.
- [ ] **AC02:** Independently implemented architecture passes automated correctness tests.
- [ ] **AC03:** Future-token leakage test passes within recorded tolerances.
- [ ] **AC04:** At least 1,000 lossless tokenizer round trips, including Unicode/Persian, pass.
- [ ] **AC05:** Fixed-batch loss meets the recorded near-memorization threshold.
- [ ] **AC06:** Independent train/validation/test partitions have provenance and duplicate-check records.
- [ ] **AC07:** Main-model training exceeds 10 million processed tokens.
- [ ] **AC08:** Main-model validation loss beats the Bigram baseline under matching evaluation conditions.
- [ ] **AC09:** Checkpoints preserve necessary state and resume without an unintended major reset.
- [ ] **AC10:** Offline CLI produces at least three samples with configurable temperature/top-k.
- [ ] **AC11:** README, setup instructions, reports, limitations, and at least 20 evaluation prompts are delivered.
- [ ] **AC12:** Automated tests and a CPU smoke run pass in CI without long training runs.

## Optional work after version-one acceptance

- [ ] Add a local FastAPI service with `/generate`, `/health`, and `/model-info` if needed.
- [ ] Benchmark MLX against PyTorch/MPS under comparable conditions.
- [ ] Evaluate RoPE, RMSNorm, SwiGLU, KV caching, or optimized attention through controlled ablations.
- [ ] Estimate cloud cost before attempting a 50–124 million parameter model.
- [ ] Scope a separate licensed Persian corpus and language-specific evaluation project.
- [ ] Experiment with narrow instruction tuning using this project's own trained model.

Planning estimate: approximately 4–8 weeks of part-time development, excluding uncertain experiment and substantial-training durations. Version one remains an educational English story model; general-purpose conversation and dependable Persian generation are outside its acceptance scope.
