# Initial main-model training budget

Selected November 4, 2026 for the local Apple M4 / 16 GiB machine. Source preparation, tokenizer identity, split integrity, Unicode round trips, and near-duplicate spot-check records must pass before continuing beyond the initial real-data diagnostic.

The first acceptance-oriented run uses the RFP's 17,538,816-parameter architecture, FP32, context 512, micro-batch 4, accumulation 4, AdamW, 200-step warmup, cosine decay, seed 42, and the fixed 15.34M-token training corpus. Each optimizer step processes 8,192 tokens. **1,221 optimizer steps process 10,002,432 tokens**, exceeding the required 10M threshold. Token windows are sampled with replacement; this total does not mean a complete pass over every unique training token.

The synthetic benchmark suggests about 17 minutes of kernel work for this budget. A 100-step real-data diagnostic checks actual throughput, numerical stability, checkpoint saving, and validation before the remaining steps are resumed with the same total scheduling budget. The initial run targets roughly 20 minutes, with checkpointed exit available if measured runtime exceeds expectations. Evaluation and latest-checkpoint saving occur every 100 optimizer steps; the best checkpoint is selected only by validation loss. Test loss is reserved for final evaluation after model selection.

Disk availability was approximately 154 GiB before the experiment. Keep only best/latest training checkpoints in the main run directory, allowing at least 2 GiB for their atomic writes and logs. The dataset, tokenizer, configuration, and resume states remain available locally.

Stop on nonfinite loss/gradients or unavailable memory. A failed diagnostic requires investigation before extending the run. Failure to beat Bigram after the initial run requires reporting the failure and a further training/configuration decision; it does not qualify as model acceptance. A quality-oriented 50–150M-token continuation remains conditional on measured validation trends, available runtime, and a separately recorded experiment configuration. It is not silently added to this initial budget.

## Recorded outcome

The diagnostic passed at about 9,454 tokens/second and the same checkpoint resumed through all 1,221 steps. Active recorded loop time totaled about 24.9 minutes, exceeding the approximate planning estimate while completing the unchanged token budget. Concurrent full-validation CPU evaluation reduced throughput. The run remained finite and within available memory; no batch/context reduction was required. Full validation loss 2.06312 beats the count-fitted Bigram's 3.15693. The reserved test loss is 2.06705. No larger training budget was started.
