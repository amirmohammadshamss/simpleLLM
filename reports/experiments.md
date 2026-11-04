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

## TinyStories main-model experiment

The pinned 50,000-story source prefix was cleaned and deduplicated before document-level splitting. The training partition contains 39,912 stories / 15,340,001 tokens. A project-trained 8,192-token byte BPE uses indexed weighted-piece training; its identity and Unicode checks are recorded in the [data card](data_card.md).

The 17,538,816-parameter main model started with random weights. It used FP32 on MPS, context 512, micro-batch 4, accumulation 4, AdamW (learning rate 0.0003, betas 0.9/0.95, weight decay 0.1), clipping 1.0, 200-step warmup, cosine decay, seed 42, and no mixed precision or compilation. The initial diagnostic stopped at step 100, saved all training states, and resumed with the unchanged total scheduling budget. A total of **1,221 optimizer steps processed 10,002,432 tokens**, with replacement from the fixed training stream.

Validation and checkpoints occurred every 100 steps and at the final step. Best-checkpoint selection used the same five fixed validation batches each time. The selected best checkpoint was step 1,221, sampled validation CE 2.08878. The recorded active-loop time was 86.66 seconds before the interruption plus 1,405.72 seconds after resume: approximately **24.9 minutes**, excluding the gap between sessions. Concurrent CPU evaluation reduced sustained throughput relative to the initial 9,454 tokens/second diagnostic. This run exceeded the approximately 20-minute planning estimate and completed the fixed token budget; it did not expand to the optional 50–150M-token quality target.

Full validation then scored each of 1,913,489 next-token targets once under the same tokenizer and nonoverlapping context-window policy. GPT CE was **2.06312** (perplexity **7.87051**), the count-fitted Bigram CE **3.15693** (perplexity **23.49843**), and random GPT CE **8.91407**. The Bigram uses every one of the 15,340,000 training pairs with smoothing 0.1, selected before held-out scoring. This is a structural baseline, not a claim of matched parameter counts or compute. The 8,192-square Bigram table has more entries than the Transformer's parameter count but conditions only on one prior token.

After checkpoint selection was fixed, reserved-test scoring over 1,925,502 targets gave GPT CE **2.06705** / perplexity **7.90146**, Bigram CE **3.15885** / perplexity **23.54356**, and random GPT CE **8.91536**. See [the test report](tinystories_test.json). The test scores did not alter training or model selection.

Evidence: [full validation](tinystories_validation.json), [training metrics](tinystories_training.jsonl), [resume-session metadata](tinystories_sessions.jsonl), [loss chart](tinystories_loss.svg), [20 exported-model CLI samples](tinystories_samples.json), and [inference artifact hashes](tinystories_export.json). Earlier step-200 validation remains in `tinystories_interim_validation.json` as diagnostic history. It did not use test data or alter the fixed final training budget.

The generations now contain recognizable story vocabulary and short grammatical phrases, but continue to lose entity identity, invent contradictory attributes, repeat phrases, and produce broken grammar. The first fixed prompt calls Lily both a girl and a car; another repeats "gave him him". These outputs are useful educational evidence, not reliable story comprehension or general conversation. All 20 outputs should be assessed alongside the quantitative metrics.

## Main-model validation and delivery status

The main architecture passes shape, gradient, tied-weight, and future-token leakage checks. BPE preparation, trainer/CLI integration, data-integrity checks, and exact interrupted/resumed CPU training are tested. The real data, tokenizer, main training, full-validation baseline comparison, and local inference export are complete. Reserved test scoring is reported separately after checkpoint selection. Both fresh-environment CPU checks and hosted CI passed; see [ci_verification.json](ci_verification.json).
