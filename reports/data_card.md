# Data card — project corpora

## TinyStories main-model subset

Source: **TinyStories — Ronen Eldan and Yuanzhi Li**, [dataset](https://huggingface.co/datasets/roneneldan/TinyStories), original file `TinyStories-train.txt`, revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`. The [data-use record](tinystories_license.md) documents CDLA-Sharing-1.0 and attribution/redistribution handling.

The downloader takes the first 50,000 delimiter-terminated stories from that pinned file. This is a reproducible prefix sample, not a random sample of the full source. It consumes 45,134,794 source bytes through the final selected delimiter, with prefix SHA-256 `e99c839654a96e4f177587ce6c36a3d3f8086466fc596d1d4ccb535b264fd0f4`. The upstream full-file LFS identity and file size are recorded, but the complete upstream file was not downloaded or independently rehashed. Selected stories are serialized as UTF-8 JSONL with source indices; framing CR/LF and delimiter markers are removed. Paragraphs, punctuation, Unicode, and story text are otherwise preserved.

Preparation removed **12 empty/corrupted samples and 100 exact duplicate stories** before splitting. The remaining 49,888 story hashes are sorted, shuffled with seed 42, and partitioned at document level before tokenization. Training contains **39,912 documents / 15,340,001 tokens**, validation **4,988 documents / 1,913,490 tokens**, and test **4,988 documents / 1,925,503 tokens**. Independent arrays, document-ID lists, and hashes are retained locally. All split arrays are integrity-checked on load, and duplicate document IDs across splits are rejected.

The 8,192-token byte-level BPE is fitted exclusively on training documents. The indexed trainer uses weighted lossless Unicode pieces, deterministic merge tie breaks, and a versioned pretokenization policy. BOS=256 and EOS=257 are reserved; no padding or unknown token is needed for corpus packing. Every encoded story ends with EOS; packing crosses stories only within a split. Tokenizer training took 2.26 seconds and full preparation 5.05 seconds in the recorded environment. These timings describe this subset, not arbitrary corpus sizes.

Tokenizer SHA-256 is `eebe14bcfe8628ca9f878bfaff5f230cf1f339bfda2bac811138220446cddf3f`; prepared manifest SHA-256 is `6073baa4b3c6dc99fa80407b7ed86c33b8bcf38a21bc3b1c19551f13c6d7f90f`. The actual tokenizer passed 1,006 Unicode/Persian/ZWNJ/control-character round trips and serialization equivalence checks. Unicode coverage is a tokenizer property, not evidence of Persian generation skill.

A seed-42 near-duplicate spot check sampled 2,000 documents and scored 517 cross-split candidate pairs using five-word-shingle Jaccard similarity. No pair reached the preselected 0.8 threshold. This does not establish that all near duplicates are absent from the full subset. The manifest's initial `near_duplicates: not reviewed` field is supplemented by the separate audit report bound to that manifest hash; immutable prepared artifacts were not rewritten after auditing.

Recorded evidence: [source/file identities](tinystories_source.json), [manifest](tinystories_manifest.json), [near-duplicate audit](tinystories_near_duplicates.json), [tokenizer checks](tinystories_tokenizer_validation.json), and [preparation timings](tinystories_preparation_metrics.json). Raw stories, arrays, and weights remain local and gitignored.

## Tiny Shakespeare educational corpus

The current experiment uses Tiny Shakespeare from `karpathy/char-rnn`, pinned to revision `6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e`, file `data/tinyshakespeare/input.txt`. [Pinned source](https://raw.githubusercontent.com/karpathy/char-rnn/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt).

The downloaded file is 1,115,394 UTF-8 bytes with SHA-256 `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`. Downloading and preparing it are separate commands; no source text or training weights are committed. Distribution/license terms have not been reviewed, and this experiment does not establish permission for redistribution or commercial use.

The educational split uses contiguous character ranges before tokenization: training `[0, 892315)`, validation `[892315, 1003854)`, test `[1003854, 1115394)`. The character vocabulary contains 65 characters, fitted exclusively on training. No normalization or punctuation removal was applied. Each range is encoded independently and batches never cross split boundaries. Vocabulary, split IDs, source revision, source hash, and array hashes are recorded in [the manifest](shakespeare_manifest.json).

These ranges contain shared Shakespeare works and can share passages or linguistic patterns. They are useful for an educational mechanics experiment, not a document-level generalization claim. Exact/near-duplicate analysis of passages has not been performed. Test text has not been scored or used for checkpoint selection; its array is checked for artifact integrity only.

For future JSONL corpora, the preparation command removes empty text, replacement-character/NUL corruption, and exact document duplicates before deterministic document-level splitting. It stores document hashes per split and fits BPE only on training documents. Packing may cross document boundaries within a split, with EOS appended to every BPE document; it never crosses split boundaries. Near-duplicate review remains separate.

The main-model subset and its separate document-level audit are described above. The educational Shakespeare results remain a separate tokenizer/domain experiment and their perplexities must not be compared directly with the BPE model.
