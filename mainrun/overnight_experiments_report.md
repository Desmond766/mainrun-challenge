# Overnight Experiments Report

**Date:** March 16, 2026  
**Goal:** Reduce validation loss below the baseline mainrun (1.266) by trying three independent approaches.

---

## Reference: Baseline Mainrun

Before the overnight experiments, the mainrun configuration achieved:

| Config | Value |
|-------|-------|
| block_size | 128 |
| batch_size | 64 |
| vocab_size | 32,000 |
| n_layer | 6 |
| n_head | 8 |
| d_model | 512 |
| lr | 3e-4 |
| dropout | 0.05 |
| warmup_frac | 0.05 |
| param_grouped_decay | False |
| Steps | 861 |

**Final validation loss: 1.266**

---

## Results Summary

| Approach | Final Val Loss | Best Val Loss | Steps | vs Baseline |
|----------|----------------|---------------|-------|-------------|
| 1 | 1.281 | 1.281 | 1,155 | ✗ +0.015 |
| 2 | 1.323 | 1.279 (step 1980) | 3,465 | ✗ +0.057 |
| 3 | 1.340 | 1.265 (step 1980) | 3,465 | ✗ +0.074 |

**Conclusion:** None of the three approaches improved on the baseline. Approaches 2 and 3 showed overfitting in later epochs (loss increased after ~step 2000).

---

## Approach 1: Hyperparameter + Data Pipeline Tweaks

**Hypothesis:** Stricter cosine decay, shorter context, and random offset packing might improve generalization.

**Config:**
- block_size: 96
- batch_size: 64
- param_grouped_decay: True
- cosine_min_ratio: 0.1 (LR decays to 10% of peak by end)
- random_offset_packing: True
- lr: 3e-4
- Other: same as baseline (vocab 32k, n_layer 6, dropout 0.05)

**Result:** 1.281 (step 1155)

**Observation:** Slightly worse than baseline. Shorter context (96 vs 128) and random offset packing did not help. The stricter cosine decay (0.1× min) may have reduced learning in later epochs.

**Log:** `logs/approach1.log`

---

## Approach 2: More Steps via Shorter Context

**Hypothesis:** More gradient steps (block_size=64, batch_size=32) would improve learning.

**Config:**
- block_size: 64
- batch_size: 32
- param_grouped_decay: True
- lr: 3e-4
- Other: same as baseline

**Result:** 1.323 final, but **best 1.279 at step 1980**

**Observation:** Loss improved until ~step 1980, then degraded. Training for 7 full epochs (3,465 steps) led to overfitting. Early stopping around step 2000 would have yielded ~1.28, still worse than baseline. Shorter context (64 tokens) may have truncated useful information for headlines.

**Log:** `logs/approach2.log`

---

## Approach 3: Full Optimization (RoPE + Label Smoothing + Higher LR)

**Hypothesis:** RoPE position encoding, label smoothing, and higher learning rate would push loss lower.

**Config:**
- block_size: 64
- batch_size: 32
- param_grouped_decay: True
- lr: 4e-4
- label_smoothing: 0.1
- use_rope: True (RoPE replaces learned position embeddings)
- Other: same as baseline

**Result:** 1.340 final, but **best 1.265 at step 1980**

**Observation:** Best intermediate loss (1.265) briefly matched baseline, but late training degraded sharply to 1.34. RoPE, label smoothing, and higher LR did not yield a net improvement. Same overfitting pattern as Approach 2.

**Log:** `logs/approach3.log`

---

## Takeaways

1. **block_size=128, batch_size=64** remains the best-tested configuration for this task.
2. **Shorter context (64 or 96)** hurt performance, likely truncating headline context.
3. **More steps** (3,465 vs 861) led to overfitting; the baseline’s 861 steps were sufficient.
4. **RoPE, label smoothing, higher LR** did not improve final loss.
5. **Early stopping** could have salvaged Approaches 2 and 3 (best ~1.27–1.28 around step 2000), but still below baseline 1.266.

---

## Log Files

- `mainrun/logs/approach1.log`
- `mainrun/logs/approach2.log`
- `mainrun/logs/approach3.log`

To extract final validation loss from any log:

```bash
grep "validation_step" mainrun/logs/approach1.log | tail -1
```
