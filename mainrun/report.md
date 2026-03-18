# Mainrun Optimization Report

> **Note:** To generate `report.pdf`, run `pandoc mainrun/report.md -o mainrun/report.pdf` or use your preferred Markdown-to-PDF tool.

## Executive Summary

This report documents my journey optimizing this model trained. Starting from a baseline validation loss of **1.754**, I reduced it to **1.266**—a 28% improvement—within the fixed 7-epoch constraint. Along the way, I ran experiments that did not improve results; this report covers both the successes and the failures, and what I learned from each.

---

## 1. How I Approached the Problem

My first step was to understand the system before changing it. I ran the baseline, inspected the logs, and read the code to see how training actually behaved. I wanted to know *why* the baseline might be underperforming, not just *that* it was. I prefer debugging real systems over writing theoretical code—the logs and metrics told me what was actually happening.

I also read the original GPT-2 paper and the "Language Models are Unsupervised Multitask Learners" blog post. That gave me a grounding in how these models are typically trained: optimizer choice, learning rate schedules, and initialization matter as much as architecture. I used AI tools to brainstorm ideas and sanity-check my reasoning, but the core decisions came from connecting the paper's recommendations to what I observed in the logs.

---

## 2. Understanding the Baseline

The baseline configuration used:

- **Optimizer:** SGD with learning rate 6e-3
- **Schedule:** Cosine annealing over all steps
- **Regularization:** No weight decay, dropout 0.1
- **Initialization:** Standard normal (std=0.02) for all parameters

**Baseline final validation loss: 1.754**

From the GPT-2 work and modern practice, SGD is rarely used for transformers. Transformers have parameters with very different gradient scales—embeddings, attention projections, and output heads behave differently. SGD applies the same learning rate everywhere, which often leads to slow or unstable convergence. I decided to focus on optimizer and training dynamics first, before touching architecture.

---

## 3. Changes That Worked

### 3.1 Switch from SGD to AdamW

**What I changed:** Replaced `torch.optim.SGD` with `torch.optim.AdamW`.

**Reasoning:** AdamW maintains per-parameter adaptive learning rates, which suits the varying gradient scales in transformers. I enabled weight decay (0.01), applied in a decoupled way, as in GPT, LLaMA, and most modern LLMs. I reduced the learning rate from 6e-3 to 3e-4, since AdamW typically uses much smaller learning rates than SGD.

**Effect:** This was the highest-impact change. Training converged faster and more smoothly.

### 3.2 Learning Rate Warmup

**What I changed:** Added a warmup phase (5% of total steps) before cosine decay. During warmup, the learning rate increases linearly from 0 to the peak value.

**Reasoning:** Without warmup, early steps can produce very large updates when the model is randomly initialized—especially with Adam, whose second-moment estimates start near zero. Warmup lets the optimizer's statistics stabilize before applying the full learning rate. Every major transformer recipe (BERT, GPT-2, LLaMA) uses warmup.

**Effect:** Smoother early training and more stable convergence.

### 3.3 Hyperparameter Tuning

**What I changed:**


| Parameter     | Baseline | Optimized | Rationale                                                |
| ------------- | -------- | --------- | -------------------------------------------------------- |
| Learning rate | 6e-3     | 3e-4      | AdamW uses much lower LRs than SGD                       |
| Weight decay  | 0.0      | 0.01      | Improves generalization; standard for AdamW              |
| Dropout       | 0.1      | 0.05      | With ~27M params and 90k titles, 0.1 may over-regularize |


**Reasoning:** With ~27M parameters and ~1.1M tokens per epoch, the model might underfit. Dropout 0.1 could be too aggressive. I reduced it to 0.05 to allow more learning while still regularizing.

**Effect:** Better balance between fitting the training data and generalizing to validation.

### 3.4 Residual Projection Scaling

**What I changed:** After standard initialization, I rescale the output projections of the attention and MLP blocks (the residual branches) by `1 / sqrt(2 * n_layer)`.

**Reasoning:** In a residual block `x + f(x)`, we want the variance of `f(x)` comparable to the variance of `x` so the signal neither explodes nor vanishes as it passes through six layers. With the default 0.02 initialization, residual outputs can grow in variance with depth. Scaling by a depth-dependent factor helps maintain stable gradient flow.

**Effect:** More stable training dynamics, especially in deeper layers.

### 3.5 Larger Vocabulary (32k)

**What I changed:** Increased vocab_size from 16k to 32k.

**Reasoning:** A larger vocabulary can reduce sequence length and improve tokenization for diverse text. I kept batch_size=64 and block_size=128 within memory limits.

**Effect:** Contributed to the final improvement. (Earlier attempts with 32k at larger batch sizes failed due to memory constraints—I learned to respect system boundaries and think clearly about failure modes when resources are limited.)

---

## 4. Results

**Baseline validation loss:** 1.754  
**Optimized validation loss:** 1.266

This is a 28% relative reduction. The training curve shows a faster descent in early epochs and a stable plateau at the end.

---

## 5. Experiments That Failed

After reaching 1.266, I ran three overnight experiments to see if I could push further. I used AI to help brainstorm approaches, then implemented and ran them. None improved on the baseline. Here is what I tried and why I think it failed.

### 5.1 Approach 1: Hyperparameter + Data Pipeline Tweaks

**What I tried:** block_size=96, batch_size=64, param_grouped_decay, cosine_min_ratio=0.1 (LR decays to 10% of peak by end), random_offset_packing.

**Result:** 1.281 (worse than 1.266)

**Analysis:** Shorter context (96 vs 128) may have truncated useful information for headlines. Random offset packing did not help. The stricter cosine decay (0.1× min) may have reduced learning in later epochs. The original configuration was already well-tuned for this task.

### 5.2 Approach 2: More Steps via Shorter Context

**What I tried:** block_size=64, batch_size=32 → ~4× more gradient steps (3,465 vs 861), param_grouped_decay.

**Result:** 1.323 final, but best 1.279 at step 1980

**Analysis:** Loss improved until ~step 1980, then degraded. Training for 7 full epochs led to overfitting. More steps did not help; the baseline's 861 steps were sufficient. Shorter context (64 tokens) likely truncated headline context. I learned that "more steps" is not always better—the system can overfit.

### 5.3 Approach 3: RoPE + Label Smoothing + Higher LR

**What I tried:** Same as Approach 2, plus RoPE position encoding, label_smoothing=0.1, lr=4e-4.

**Result:** 1.340 final, but best 1.265 at step 1980

**Analysis:** Best intermediate loss (1.265) briefly matched the baseline, but late training degraded sharply. RoPE, label smoothing, and higher LR did not yield a net improvement. Same overfitting pattern as Approach 2. Modern techniques from larger models do not always transfer to smaller setups with different data constraints.

---

## 6. What I Learned

1. **Read the logs.** Validation loss and training curves told me when things were working and when they were overfitting. I relied on the logs to decide what to try next.
2. **Respect constraints.** Memory limits, fixed epochs, and the evaluation function are real boundaries. I learned to work within them instead of fighting them.
3. **Incremental changes.** The biggest wins came from well-understood optimizations (AdamW, warmup, hyperparameters) rather than experimental architecture changes.
4. **Failed experiments are data.** The overnight runs taught me that shorter context, more steps, and some "modern" techniques did not help in this setting. That is useful information for future work.
5. **AI + own thinking.** I used AI to brainstorm and explore ideas, but I grounded decisions in the GPT-2 paper, the training logs, and my own reasoning about system behaviour.
6. **Stay calm when things behave unexpectedly.** When the overnight experiments failed to improve, I didn't panic—I reverted to the working config, documented what failed, and used that to inform my understanding. Reliability matters: I want to build systems that are hard to break.

---

## 7. Conclusion

I reduced validation loss from 1.754 to 1.266 by aligning the training setup with modern transformer practice: AdamW, warmup, tuned hyperparameters, and improved initialization. I stayed methodical: understand the baseline, read the literature, make one change at a time, and use logs to guide the next step.

The failed experiments reinforced that production systems behave differently under real constraints. What works for large models or different datasets may not work here. I am comfortable with that—debugging real systems and learning from both success and failure is what I enjoy. I want to understand how large-scale AI systems actually work, and this assignment was a concrete step in that direction.

---

## Appendix: Verification

To reproduce the final validation loss:

```bash
task train
grep "validation_step" mainrun/logs/mainrun.log | tail -1
```

Expected: loss ≈ 1.266