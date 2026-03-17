# Metric Minimization Experiment Report

## Overview

This report documents a sequential experiment to further reduce validation loss on the GPT-2 style model trained on Hacker News headlines. The runs were cumulative: each run added one change on top of the previous configuration. All runs used `batch_size=32` for 16 GB CPU safety.

**Baseline (from main optimization):** ~1.27  
**Starting point for this sweep:** Run 1 with 24k vocab, block_size 128, contiguous packing.

---

## Results Summary

| Run | Change Added | Final Val Loss | vs Previous |
|-----|--------------|----------------|-------------|
| 1 | vocab 24k, block 128, batch 32 | 1.305 | — |
| 2 | block_size 96 | 1.290 | ✓ improved |
| 3 | random_offset packing | **1.267** | ✓ improved |
| 4 | lr 4e-4, adam_beta2 0.95 | 1.286 | ✗ regressed |
| 5 | dropout 0.01 | 1.291 | ✗ regressed |
| 6 | SwiGLU + RMSNorm | — | failed early |

---

## Run Details

### Run 1: Larger tokenizer vocab (16k → 24k)
- **Config:** vocab_size 24_000, block_size 128, batch_size 32, pack_strategy contiguous
- **Result:** 1.305
- **Note:** Reduced from 32k to 24k for memory safety on CPU-only 16 GB machines.

### Run 2: Shorter context (block_size 128 → 96)
- **Config:** + block_size 96 (titles are short; fewer optimizer steps per epoch with same data)
- **Result:** 1.290
- **Effect:** Improved. Shorter context suited to headline length and increased step count.

### Run 3: Random-offset packing
- **Config:** + pack_strategy "random_offset"
- **Result:** 1.267
- **Effect:** Best result. Random offset each epoch exposes tokens in more contexts.

### Run 4: AdamW short-run tuning (lr 4e-4, beta2 0.95)
- **Config:** + lr 4e-4, adam_beta2 0.95
- **Result:** 1.286
- **Effect:** Regressed. Higher LR and faster beta2 likely too aggressive for this dataset size.

### Run 5: Lower dropout (0.05 → 0.01)
- **Config:** + dropout 0.01
- **Result:** 1.291
- **Effect:** Regressed. Lower dropout may have caused overfitting.

### Run 6: SwiGLU + RMSNorm
- **Config:** + mlp_type swiglu, norm_type rmsnorm
- **Result:** Run failed early (log truncated)
- **Effect:** Not completed; likely crashed or killed.

---

## Conclusion

**Best configuration from this sweep:** Run 3

- vocab_size: 24_000  
- block_size: 96  
- batch_size: 32  
- pack_strategy: random_offset  
- lr: 3e-4  
- dropout: 0.05  
- adam_beta2: 0.999  

**Changes that helped:** Larger vocab (24k), shorter context (block 96), random-offset packing.

**Changes that hurt:** Higher LR + beta2, lower dropout.

**Log directory:** `logs/metric_min_20260314-131252/`
