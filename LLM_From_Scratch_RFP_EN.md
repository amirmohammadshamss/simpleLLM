# Request for Proposal (RFP)
## Design, Implementation, and Training of a Small Language Model from Scratch

**Version:** 1.0  
**Date:** November 3, 2026  
**Proposed project name:** MiniGPT Lab  
**Document language:** English  
**Project type:** Research, education, and software engineering  
**Status:** Ready for implementation

---

## 1. Executive Summary

The objective of this project is to build an **autoregressive language model from scratch**: from collecting and cleaning text, building a tokenizer, and implementing a Transformer to training the model weights, evaluating performance, saving checkpoints, and generating text locally. The model must start with **randomly initialized weights**. No pretrained weights or commercial language-model APIs may be used in its training or text-generation pipeline.

The final deliverable will be a **small language model (SLM)** that can be understood, trained, tested, and run independently. The goal is to reproduce the fundamental concepts of large language models at a practical scale, **not** to create a replacement for ChatGPT or a general-purpose intelligent assistant.

**Implementation decision:** First build a tiny prototype using a character-level tokenizer. Then train a decoder-only GPT model with approximately **17.5 million parameters** on a subset of TinyStories. Larger configurations are reserved for optional later phases.

## 2. Research Findings and Design Rationale

1. The Transformer architecture was introduced in *Attention Is All You Need*. For next-token prediction, this project will use a decoder-only variant with **causal self-attention** [R1].
2. *TinyStories* demonstrated that very small language models, even below 10 million parameters, can generate coherent English within a narrow children's-story domain. These findings must **not** be generalized to open-domain conversation, Persian, or factual world knowledge [R2].
3. **nanoGPT**, **build-nanogpt**, and **LLMs-from-scratch** are useful architectural and testing references. The project's code must be written independently; these repositories are learning and comparison resources rather than source-code dependencies [R3, R4, R5].
4. According to PyTorch and Apple documentation, **MPS** acceleration on Apple Silicon can support local training. Actual throughput and memory use must be benchmarked on the target machine; training time cannot be guaranteed in advance [R6, R7].
5. Chinchilla research shows that model size and training-token count must be selected with compute budgets in mind. For a project at this much smaller scale, published scaling relationships are **approximate guidance**, not strict rules [R8].
6. A **byte-level BPE** tokenizer is recommended because it can cover arbitrary Unicode text without requiring an unknown-token fallback. A simplified implementation will be built inside the project [R9].

## 3. Objectives and Expected Outcomes

### 3.1 Primary Objectives

- Independently understand and implement the pipeline `text → tokens → embeddings → transformer → logits → next token`.
- Implement the forward pass, cross-entropy loss calculation, and training loop using PyTorch Autograd.
- Build and train a tokenizer from scratch, including bidirectional text/token conversion.
- Train on a CPU and, where supported, on Apple Silicon through MPS.
- Generate text from the trained model with configurable temperature and top-k sampling.
- Deliver valid quantitative evaluation metrics and reproducible experiment documentation.

### 3.2 Out of Scope for Version 1

- ChatGPT-level conversation, reliable general knowledge, complex reasoning, or agentic behavior.
- Pretrained weights, fine-tuning external models, distillation, retrieval-augmented generation (RAG), or commercial inference APIs.
- RLHF, DPO, multi-GPU training, search engines, large-scale web crawling, or a sophisticated user interface.
- High-quality Persian generation in the primary model; Persian-language training would be a separate follow-up project.

## 4. Exact Definition of “From Scratch”

**Allowed:** Python, PyTorch for tensors and automatic differentiation, NumPy, dataset-loading libraries, and testing tools. Using PyTorch does not mean using a pretrained model.

**Must be implemented within the project:** educational vocabulary/tokenizer, embeddings, attention, causal masking, multi-head attention, Transformer blocks, output language-model head, sampling, training loop, checkpoint manager, and evaluation.

**Prohibited:** using `from_pretrained(...)` for a model or tokenizer, downloading pretrained model weights, initializing from third-party model checkpoints, or using external APIs to generate the final model's text.

**Clarification:** Reimplementing Autograd, GPU kernels, CUDA, or Metal is not required. Doing so would turn this into a numerical-computing framework project rather than a language-model project.

## 5. Technical Architecture

### 5.1 Conceptual Architecture

```text
Raw text / TinyStories
       ↓
Data cleaning + document-level split + deduplication
       ↓
Custom tokenizer (Character → Byte-level BPE)
       ↓
Token IDs + sequence batching
       ↓
Token Embedding + Position Embedding
       ↓
N × [LayerNorm → Masked Multi-Head Attention → Residual
     → LayerNorm → Feed-Forward (GELU) → Residual]
       ↓
Final LayerNorm → Linear LM Head → Logits
       ↓
Cross-Entropy (training) / Sampling (inference)
```

### 5.2 Proposed Model Configurations

| Property | Educational prototype | Main model |
|---|---|---|
| Architecture | Decoder-only GPT | Decoder-only GPT |
| Tokenizer | Character-level | Project-trained byte-level BPE |
| Vocabulary size | Data-dependent | 8,192 tokens including special tokens |
| Transformer layers | 4 | 8 |
| Hidden dimension | 128 | 384 |
| Attention heads | 4 | 6 |
| Feed-forward dimension | 512 | 1,536 |
| Context length | 128 | 512 |
| Approximate parameters | Fewer than 1 million | Approximately 17.5 million |
| Dataset | Tiny Shakespeare | TinyStories |

**Main-model details:** Pre-LayerNorm, learned positional embeddings, GELU activation, residual connections, tied token-embedding and language-model-head weights, configurable dropout, and causal attention masking. Attention in the educational version should use an explicit matrix-multiplication implementation. An optimized version may use `scaled_dot_product_attention` once equivalence tests pass.

**Parameter estimate:** Approximately 17.5 million parameters is an estimate for `vocab=8192`, `d_model=384`, `layers=8`, `d_ff=1536`, `context=512`, and weight tying. The implementation must print the actual parameter count.

## 6. Dataset and Data Management

### 6.1 Educational Phase

Use **Tiny Shakespeare** (the approximately 1 MB text dataset popularized by nanoGPT) to validate the complete pipeline. The purpose of this phase is to demonstrate technical correctness rather than achieve high-quality generation.

### 6.2 Main Phase

- Proposed source: **roneneldan/TinyStories**, a collection of short English stories [R2, R10].
- The dataset card reports approximately **2.14 million rows**, with raw files totaling several gigabytes. Downloading or training on the full dataset is unnecessary at the beginning.
- Initial experiments: a fixed, reproducible subset containing approximately **10–30 million tokens**.
- Substantial training: initially target **50–150 million tokens processed**. Expansion toward 300 million processed tokens should occur only if cost, training time, and validation trends justify it.
- Record the source, dataset revision, file IDs/hashes, sample counts, token counts, license, and preprocessing transformations.
- Hugging Face identifies the TinyStories license as **CDLA Sharing 1.0**. Review the actual license terms and attribution/sharing requirements before redistribution or commercial use.

### 6.3 Preprocessing and Splitting

1. Remove corrupted or empty samples while preserving punctuation, paragraph breaks, and story endings.
2. Remove exact duplicates and spot-check near-duplicates.
3. Split data at the **story/document level**, before converting text into overlapping token sequences, to prevent leakage.
4. Maintain independent Train, Validation, and Test sets. The official validation data may be split deterministically into nonoverlapping validation and test subsets.
5. Fit the tokenizer only on training data; version the vocabulary and merge rules alongside checkpoints.
6. Define an end-of-story token (`<eos>`) and avoid unintended sequence crossing between isolated data splits.

## 7. Tokenizer Development

**Stage 1:** Implement a character-level tokenizer supporting `encode`, `decode`, and vocabulary serialization/deserialization.

**Stage 2:** Implement simple byte-level BPE with pair-frequency counting, iterative pair merging, ordered merge-rule storage, encoding/decoding, and full UTF-8 support. Minimum special tokens are `<bos>`, `<eos>`, and optionally `<pad>`. A byte-level tokenizer does not inherently require `<unk>`.

**Acceptance conditions:** Exact round-trip reconstruction on ordinary and Unicode text, including Persian text, zero-width non-joiners, and punctuation; at least **1,000 sample tests**; identical output before and after saving/loading the tokenizer. Being able to tokenize Persian does **not** imply the trained model can generate Persian.

## 8. Training Requirements

### 8.1 Learning Objective

Train with **next-token prediction** and cross-entropy loss against the correct subsequent token. The causal mask must prevent each position from attending to future positions.

### 8.2 Suggested Starting Configuration

```yaml
model:
  vocab_size: 8192
  n_layers: 8
  d_model: 384
  n_heads: 6
  d_ff: 1536
  context_length: 512
  dropout: 0.1
  tie_embeddings: true
  positional_encoding: learned
training:
  optimizer: adamw
  learning_rate: 0.0003
  betas: [0.9, 0.95]
  weight_decay: 0.1
  gradient_clip_norm: 1.0
  micro_batch_size: 4
  gradient_accumulation_steps: 4
  warmup_steps: 200
  lr_schedule: cosine
  seed: 42
  eval_every_steps: 500
  save_every_steps: 500
  device: auto   # mps / cuda / cpu
```

These settings are **initial experimental hypotheses**, not hyperparameters validated for the target hardware. The effective batch in this example is `4 × 4 × 512 = 8,192` tokens per optimizer step. If out-of-memory errors occur, reduce `micro_batch_size` first, then `context_length`.

### 8.3 Mandatory Training-Loop Capabilities

- Configure a random seed and log the Python version, PyTorch version, operating system, configuration, and Git commit.
- Log training/validation loss, perplexity (only directly comparable for the same tokenizer), tokens per second, elapsed time, learning rate, gradient norm, and memory use when available.
- Detect NaN/Inf values and fail gracefully with actionable diagnostics.
- Support gradient accumulation, gradient clipping, warmup, and cosine learning-rate decay.
- Save **model, optimizer, scheduler, global step, RNG states, tokenizer, and configuration**; support resuming from interruption.
- Retain at least one `best checkpoint` chosen by validation loss and one `latest checkpoint`.
- Enable optional mixed precision only after benchmarking and numerical-validity tests. Keep `torch.compile` disabled by default in the first version to avoid unnecessary complexity.

## 9. Hardware and Performance

**Local target:** A Mac with Apple Silicon M4 and **16 GB unified memory**. A CPU fallback and cloud GPU are optional execution paths.

A 17.5-million-parameter model requires approximately **70 MB** for FP32 weights alone. Weights, gradients, and the two Adam optimizer moments together occupy approximately **280 MB** in FP32. This excludes activations, attention buffers, batches, MPS allocator overhead, the operating system, and other applications. Consequently, **measured peak memory** must be used for sizing decisions.

**Mandatory benchmark:** Run **100–300 steps** at the intended configuration and report tokens/second, peak memory, time per step, sustained-performance behavior when measurable, and training stability. Use measured throughput to estimate the time needed for the planned token budget. A full training run might take hours or days; no fixed duration is guaranteed.

**Platform note:** PyTorch/MPS is the default for its clear implementation and extensive educational documentation. **Apple MLX** is proposed only as a second-stage optimization or comparative experiment [R11].

## 10. Text Generation and API

### 10.1 Command-Line Interface

```bash
python -m minigpt.generate \
  --checkpoint checkpoints/best.pt \
  --prompt "Once upon a time" \
  --max-new-tokens 150 \
  --temperature 0.8 \
  --top-k 40 \
  --seed 42
```

Required features: greedy decoding, temperature sampling, top-k, EOS stopping, input/output length limits, a configurable random seed, and fully offline operation after the required local files are available.

### 10.2 Optional Local Service

If needed, add a FastAPI service with `POST /generate`, `GET /health`, and `GET /model-info`. A TypeScript/React web interface is **not** part of the version-one acceptance criteria. Generation requests must never be sent to third-party inference services.

## 11. Model Evaluation

| Test | Method | Purpose |
|---|---|---|
| Model correctness | Tensor-shape, causal-mask, and gradient tests | Identify logical implementation errors |
| Controlled overfitting | Train on one fixed batch | Verify the backpropagation pipeline |
| Statistical quality | Validation loss and perplexity | Measure learning beyond training data |
| Baseline comparison | Bigram and randomly initialized model with the same tokenizer | Demonstrate improvement from the Transformer |
| Text quality | 20 fixed prompts and saved generations | Inspect repetition, coherence, and grammar |
| Resumability | Interrupt and resume training | Protect training progress |
| Generalization | Nonoverlapping document-level test holdout | Reduce overly optimistic evaluation |

**Important limitation:** Perplexity and cross-entropy values of models using different tokenizers are not directly comparable. **Bits per byte** on the same raw text can be used for cross-tokenizer comparisons. Human evaluation is supplementary evidence and does not replace quantitative measurements.

## 12. Acceptance Criteria

- **AC01:** All main-model weights begin with random initialization and no pretrained-model loading occurs in the training pipeline.
- **AC02:** The manually implemented `Embedding → Transformer → LM Head` architecture and causal attention pass automated tests.
- **AC03:** The future-token leakage test passes: changing tokens after position `t` must not change logits at positions `≤ t` in evaluation mode, within numerical tolerance.
- **AC04:** The project tokenizer passes at least 1,000 round-trip tests, including Unicode and Persian, without data loss.
- **AC05:** The model can substantially reduce loss and nearly memorize a small fixed batch; the precise threshold must be recorded after a smoke test.
- **AC06:** Separate Train/Validation/Test sets exist with documented provenance and duplicate checks.
- **AC07:** The main model completes at least one meaningful training run processing more than **10 million tokens**. A quality-oriented target of **50–150 million tokens** is recommended, conditional on benchmarked budget.
- **AC08:** The trained main model achieves better validation loss than a Bigram baseline using the **same tokenizer and dataset split**.
- **AC09:** Checkpoints preserve the necessary states and resuming training does not cause an unintended major reset. Bitwise-identical behavior between CPU and MPS is not expected.
- **AC10:** The CLI generates at least three text samples from a local checkpoint with configurable temperature and top-k.
- **AC11:** Deliver `README`, experiment reports, documented limitations, setup instructions, and at least 20 evaluation prompts.
- **AC12:** Automated tests and a CPU smoke run execute in CI. Multi-hour training runs must not be part of ordinary CI.

**Not acceptance criteria:** ChatGPT-level writing, reliable general knowledge, or dependable Persian-language question answering.

## 13. Proposed Repository Structure

```text
minigpt-lab/
  README.md
  pyproject.toml
  configs/
    toy.yaml
    tinystories.yaml
  minigpt/
    tokenizer/
      char_tokenizer.py
      bpe_tokenizer.py
    data/
      download.py
      prepare.py
      dataset.py
    model/
      attention.py
      block.py
      gpt.py
    training/
      train.py
      checkpoint.py
      metrics.py
    evaluation/
      evaluate.py
      samples.py
    generate.py
    api.py                 # optional
  tests/
    test_tokenizer.py
    test_attention.py
    test_model.py
    test_training.py
    test_resume.py
  scripts/
    smoke_test.sh
    benchmark.sh
  reports/
    experiments.md
    model_card.md
    data_card.md
  checkpoints/             # gitignored
  data/                    # gitignored
```

## 14. Implementation Plan and Milestones

| Phase | Work | Deliverable / decision gate | Estimated part-time effort |
|---|---|---|---|
| 0 | Study fundamentals and initialize the project | CPU/MPS execution, tests, configuration | 2–4 days |
| 1 | Character tokenizer, Bigram, and MiniGPT | Test text generation and controlled overfitting | 4–7 days |
| 2 | Custom BPE and data pipeline | Round-trip tests and leakage-safe splits | 4–7 days |
| 3 | 17.5M architecture and training loop | Smoke tests, checkpointing, benchmark | 5–8 days |
| 4 | Main training and experiment analysis | Loss curves, samples, best-model selection | Variable; depends on benchmark |
| 5 | Evaluation, CLI, and documentation | Final report, automated tests, reproducible setup | 3–5 days |

**Overall estimate:** Approximately **4–8 weeks of part-time development** for an experienced programmer with introductory Python/PyTorch knowledge, excluding uncertain experiment and large-training durations. This is not a contractual delivery guarantee.

## 15. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Low text quality in a small model | Disappointing generations | Limit scope to English children's stories; measure against baseline |
| Insufficient unified memory | Training interruption | Smaller micro-batches, gradient accumulation, shorter context |
| Excessively long training | Wasted time and compute | Benchmark early, define budgets, stop based on validation |
| Incorrect attention/masking | Apparently successful but invalid training | Output-equivalence, gradient, and future-token leakage tests |
| Dataset leakage | Misleading results | Document-level splitting before tokenization; deduplication |
| Repetition/overfitting | Poor generalization | Holdout set, early stopping, logging, data review |
| Data-license restrictions | Redistribution risk | Data card, license review, permission-aware publishing |
| CPU/MPS numerical differences | Imperfect reproducibility | Record seeds/environment; use numeric tolerances rather than claiming bitwise identity |

## 16. Optional Future Extensions

- Experiment with **MLX** and compare training speed and memory use against PyTorch/MPS.
- Evaluate relative/RoPE positional embeddings, RMSNorm, SwiGLU, KV caching, and optimized attention methods through controlled ablations.
- Train a **50–124 million parameter** model on a cloud GPU only after validating the pipeline and estimating costs.
- Start a separate Persian-language project: licensed corpus, normalization of Persian Arabic variants of ye/kaf, zero-width non-joiners, appropriate tokenization, and language-specific evaluation.
- Apply limited **instruction tuning to the model trained by this project**, for narrow tasks such as “write a short story.” This must not be confused with the initial pretraining stage.

## 17. Final Deliverables

An accepted delivery includes independently written source code; configuration files; the tokenizer and model weights **trained by the project itself**; automated tests; data and license reports; hardware benchmarks; loss charts; generated text samples; limitation documentation; and an offline execution guide. The repository must be reproducibly set up from a clean environment using documented commands. Downloading the main dataset may be a separate setup step.

## 18. Research Sources and References

- **[R1]** Vaswani et al., *Attention Is All You Need* (2017): [Original publication](https://research.google/pubs/attention-is-all-you-need/)
- **[R2]** Eldan & Li, *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* (2023): [Paper](https://arxiv.org/abs/2305.07759)
- **[R3]** Andrej Karpathy, *nanoGPT*: [GitHub repository](https://github.com/karpathy/nanoGPT)
- **[R4]** Andrej Karpathy, *build-nanogpt*: [GitHub repository](https://github.com/karpathy/build-nanogpt)
- **[R5]** Sebastian Raschka, *LLMs-from-scratch*: [GitHub repository](https://github.com/rasbt/LLMs-from-scratch)
- **[R6]** PyTorch, *MPS backend*: [Documentation](https://docs.pytorch.org/docs/stable/notes/mps.html)
- **[R7]** Apple Developer, *Accelerated PyTorch training on Mac*: [Documentation](https://developer.apple.com/metal/pytorch/)
- **[R8]** Hoffmann et al., *Training Compute-Optimal Large Language Models* (2022): [Paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c1e2faff6f588870935f114ebe04a3e5-Abstract.html)
- **[R9]** OpenAI, *tiktoken / educational BPE*: [GitHub documentation](https://github.com/openai/tiktoken/blob/main/README.md)
- **[R10]** Hugging Face, *roneneldan/TinyStories dataset card*: [Dataset](https://huggingface.co/datasets/roneneldan/TinyStories)
- **[R11]** MLX team, *MLX Examples: Transformer Language Model*: [GitHub repository](https://github.com/ml-explore/mlx-examples)
- **[R12]** AWS, *Data splits and leakage*: [Guidance](https://docs.aws.amazon.com/prescriptive-guidance/latest/ml-operations-planning/splits-leakage.html)
- **[R13]** PyTorch, *Scaled Dot Product Attention*: [Documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention)
