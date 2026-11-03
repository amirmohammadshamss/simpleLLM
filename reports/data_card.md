# Data card — current educational corpus

The current experiment uses Tiny Shakespeare from `karpathy/char-rnn`, pinned to revision `6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e`, file `data/tinyshakespeare/input.txt`. [Pinned source](https://raw.githubusercontent.com/karpathy/char-rnn/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt).

The downloaded file is 1,115,394 UTF-8 bytes with SHA-256 `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`. Downloading and preparing it are separate commands; no source text or training weights are committed. Distribution/license terms have not been reviewed, and this experiment does not establish permission for redistribution or commercial use.

The educational split uses contiguous character ranges before tokenization: training `[0, 892315)`, validation `[892315, 1003854)`, test `[1003854, 1115394)`. The character vocabulary contains 65 characters, fitted exclusively on training. No normalization or punctuation removal was applied. Each range is encoded independently and batches never cross split boundaries. Vocabulary, split IDs, source revision, source hash, and array hashes are recorded in [the manifest](shakespeare_manifest.json).

These ranges contain shared Shakespeare works and can share passages or linguistic patterns. They are useful for an educational mechanics experiment, not a document-level generalization claim. Exact/near-duplicate analysis of passages has not been performed. Test text has not been scored or used for checkpoint selection; its array is checked for artifact integrity only.

For future JSONL corpora, the preparation command removes empty text, replacement-character/NUL corruption, and exact document duplicates before deterministic document-level splitting. It stores document hashes per split and fits BPE only on training documents. Packing may cross document boundaries within a split, with EOS appended to every BPE document; it never crosses split boundaries. Near-duplicate review remains separate.

TinyStories has not been downloaded or prepared. Its source revision, actual license terms, subset size, near-duplicate review, and final split/token statistics remain outstanding. The main-model data acceptance gate remains open.
