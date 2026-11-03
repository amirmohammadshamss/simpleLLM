# Educational experiments

## Tiny Shakespeare character GPT

The model uses 4 layers, hidden size 128, 4 attention heads, feed-forward size 512, context length 128, and a 65-character training-only vocabulary. It has 818,048 unique parameters. Data preparation and its limitations are documented in [the data card](data_card.md).

The initial 100-step experiment processed 51,200 tokens. Exact full-validation loss was 2.90105, trailing a 500-step trained Bigram at 2.51638. That result motivated a longer educational run rather than a quality claim.

The second experiment used 2,000 optimizer steps, batch size 4, learning rate 0.0003 with cosine decay, seed 42, and CPU execution. It processed 1,024,000 tokens, sampling training windows with replacement. This is a processed-token total, not a claim that every training token was visited. Best-checkpoint selection used five fixed validation batches every 200 steps; the best checkpoint was step 2,000. The recorded training-loop time was approximately 28.35 seconds on this environment, including periodic evaluation/checkpoint work as accumulated by the trainer. This toy timing must not be applied to the larger model.

Full-validation scoring then evaluated each of the 111,538 next-token targets once, in nonoverlapping windows of at most 128 tokens. All three models used the same tokenizer, targets, and context-window policy. Trained GPT loss was **2.41143**, random GPT loss **4.19669**, and trained Bigram loss **2.51638**. The Bigram was trained for 500 steps / 256,000 tokens; training compute was not matched between models. This demonstrates improvement in this educational setting, not acceptance of the main TinyStories model.

Machine-readable results are in [shakespeare_validation.json](shakespeare_validation.json), with [training metrics](prototype_training.jsonl). Twenty fixed prompts and saved CLI outputs are in [prototype_samples.json](prototype_samples.json). Outputs remain largely incoherent, with repetition and broken words. English story prompts are intentionally retained for future comparable model samples, but Shakespeare training is a domain mismatch for them. No reliable general knowledge, story coherence, or Persian generation is demonstrated.

Local artifacts are `checkpoints/shakespeare-prototype-2000/{best,latest}.pt`; data and checkpoints are gitignored. Test-holdout loss has not been measured or used for model selection.

## Controlled overfitting

The full educational architecture, fitted to the smoke sequence's smaller character vocabulary, reduced fixed-batch CE from 2.53912 to 0.000209 in 120 steps at learning rate 0.01 with dropout disabled. Checkpoint restoration reproduced its logits.

The 17,538,816-parameter main architecture initially failed the CE-below-0.1 threshold at learning rate 0.01: loss fell from 8.83383 to 0.62429 after 120 steps. Repeating with learning rate 0.001 and 200 steps reduced CE to **0.000285**, passing the unchanged threshold and the required reduction above 90%. This uses a short synthetic sequence, the main 8,192-way output vocabulary, and dropout disabled. It does not test sustained full-context corpus training.

## Main-model readiness

The main architecture passes shape, gradient, tied-weight, and future-token leakage checks. BPE preparation, trainer/CLI integration, data-integrity checks, and exact interrupted/resumed CPU training are tested on small local document fixtures. TinyStories data selection, license review, near-duplicate review, large-corpus BPE performance, and the more-than-10-million-token main-model experiment remain outstanding.
