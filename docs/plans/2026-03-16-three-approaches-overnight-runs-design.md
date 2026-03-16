# Three Approaches Overnight Runs Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run three independent training approaches (1, 2, 3) sequentially overnight, each from the current baseline, with separate logs for comparison.

**Architecture:** Add `--approach` argument to train.py; each approach uses a self-contained config (no cumulative changes). A bash script runs all three sequentially, writing to `approach1.log`, `approach2.log`, `approach3.log`.

**Tech Stack:** Python 3.10, PyTorch, existing mainrun/train.py

---

## Approach Definitions (from brainstorming)

| Approach | block_size | batch_size | param_grouped_decay | cosine_min | random_offset | label_smoothing | lr | RoPE |
|----------|------------|------------|--------------------|------------|---------------|-----------------|-----|------|
| 1 | 96 | 64 | True | 0.1 | True | - | 3e-4 | No |
| 2 | 64 | 32 | True | 0.5 | False | - | 3e-4 | No |
| 3 | 64 | 32 | True | 0.5 | False | 0.1 | 4e-4 | Yes |

---

## Task 1: Add APPROACH_CONFIGS and --approach CLI

**Files:**
- Modify: `mainrun/train.py` (lines 38-46, 254-268)

**Step 1: Add APPROACH_CONFIGS after RUN_CONFIGS**

Add after line 45 (after RUN_CONFIGS closing brace):

```python
# Approach configs for overnight comparison (each is independent, no cumulative)
APPROACH_CONFIGS = {
    1: {
        "block_size": 96,
        "batch_size": 64,
        "param_grouped_decay": True,
        "cosine_min_ratio": 0.1,
        "random_offset_packing": True,
        "lr": 3e-4,
    },
    2: {
        "block_size": 64,
        "batch_size": 32,
        "param_grouped_decay": True,
        "cosine_min_ratio": 0.5,
        "random_offset_packing": False,
        "lr": 3e-4,
    },
    3: {
        "block_size": 64,
        "batch_size": 32,
        "param_grouped_decay": True,
        "cosine_min_ratio": 0.5,
        "random_offset_packing": False,
        "lr": 4e-4,
        "label_smoothing": 0.1,
        "use_rope": True,
    },
}
```

**Step 2: Add --approach argument and apply config**

In `main()`, add after `parser.add_argument("--log-file", ...)` (around line 264):

```python
parser.add_argument("--approach", type=int, choices=[1, 2, 3], default=None,
                    help="Approach 1/2/3 for overnight runs. Overrides hyperparameters.")
```

After `if cli.log_file and cli.run is None:` block (around line 268), add:

```python
if cli.approach is not None:
    config = APPROACH_CONFIGS[cli.approach]
    for k, v in config.items():
        if hasattr(args, k):
            setattr(args, k, v)
        else:
            setattr(args, k, v)  # for new keys like cosine_min_ratio, random_offset_packing, label_smoothing, use_rope
    if cli.log_file:
        args.log_file = cli.log_file
    else:
        args.log_file = f"./logs/approach{cli.approach}.log"
```

**Step 3: Add Hyperparameters defaults for new fields**

In `Hyperparameters` dataclass, add (after param_grouped_decay):

```python
cosine_min_ratio: float = 0.5
random_offset_packing: bool = False
label_smoothing: float = 0.0
use_rope: bool = False
```

**Step 4: Verify**

Run: `cd mainrun && python3 train.py --help`
Expected: `--approach` appears in help.

---

## Task 2: Apply cosine_min_ratio in LR scheduler

**Files:**
- Modify: `mainrun/train.py` (lines 328-332)

**Step 1: Update lr_lambda to use cosine_min_ratio**

Replace the lr_lambda function body:

```python
def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps if warmup_steps > 0 else 1.0
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    min_ratio = getattr(args, "cosine_min_ratio", 0.5)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
```

**Step 2: Verify**

Run: `cd mainrun && python3 train.py --approach 1 --log-file ./logs/test_approach1.log` (Ctrl+C after a few steps)
Expected: Training starts without error.

---

## Task 3: Implement random_offset_packing in get_batch_by_index

**Files:**
- Modify: `mainrun/train.py` (lines 107-115)
- Modify: `mainrun/train.py` (main loop, around line 357)

**Step 1: Add optional random_offset parameter to get_batch_by_index**

Change signature and body:

```python
def get_batch_by_index(split_ids: torch.Tensor, batch_idx: int, block_size: int, batch_size: int, device: torch.device, random_offset: int = 0):
    """Get a batch by index. Used with shuffled indices for per-epoch data shuffling.
    random_offset: when > 0, adds offset to ptr for random context windows (approach 1)."""
    span = block_size * batch_size + 1
    max_start = len(split_ids) - span
    if max_start < 0:
        raise ValueError("split_ids too short for batch")
    ptr = min(batch_idx * span + random_offset, max_start)
    batch = split_ids[ptr: ptr + span]
    x = batch[:-1].view(batch_size, block_size).to(device)
    y = batch[1:].view(batch_size, block_size).to(device)
    return x, y
```

**Step 2: Pass random_offset in training loop**

In the training loop, replace the get_batch_by_index call:

```python
use_random_offset = getattr(args, "random_offset_packing", False)
offset = random.randint(0, span - 1) if use_random_offset else 0
xb, yb = get_batch_by_index(train_ids, batch_idx, args.block_size, args.batch_size, device, random_offset=offset)
```

Note: `span` must be available in the loop. Add before the loop: `span = args.block_size * args.batch_size + 1`

**Step 3: Verify**

Run: `cd mainrun && python3 train.py --approach 1 --log-file ./logs/test_a1.log` (Ctrl+C after ~10 steps)
Expected: No errors.

---

## Task 4: Implement label_smoothing in loss

**Files:**
- Modify: `mainrun/train.py` (GPT.forward, line 249)
- Modify: `mainrun/train.py` (evaluate function - CANNOT change per README)

**Constraint:** README says "You cannot change the evaluate() function." So we can only add label smoothing to the *training* loss, not validation. That is acceptable—training with label smoothing often improves validation loss.

**Step 1: Pass label_smoothing to model forward or compute in training loop**

Option: Add `label_smoothing` to GPTConfig and use in F.cross_entropy. But that would require changing the forward signature. Simpler: compute loss in the training loop with label_smoothing when args.label_smoothing > 0.

Change the training step from:
```python
_, loss = model(xb, yb)
```
to:
```python
logits, _ = model(xb, None)
if getattr(args, "label_smoothing", 0) > 0:
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), reduction='mean', label_smoothing=args.label_smoothing)
else:
    _, loss = model(xb, yb)
```

Actually that would call model twice. Better: have model.forward accept an optional label_smoothing and use it when targets are provided. Or we could add a wrapper. Simplest: when label_smoothing > 0, we don't use model(xb, yb) for loss; we use model(xb, None) and then F.cross_entropy(..., label_smoothing=args.label_smoothing). So:

```python
logits, _ = model(xb, None)
ls = getattr(args, "label_smoothing", 0.0)
if ls > 0:
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), reduction='mean', label_smoothing=ls)
else:
    _, loss = model(xb, yb)
```

Wait, that still calls model twice when ls=0. Let me do:

```python
logits, loss = model(xb, yb)
ls = getattr(args, "label_smoothing", 0.0)
if ls > 0:
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), reduction='mean', label_smoothing=ls)
```

So we always get logits from model(xb, yb) - but model returns (logits, loss). We need logits. So:

```python
logits, _ = model(xb, yb)
ls = getattr(args, "label_smoothing", 0.0)
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), reduction='mean', label_smoothing=ls) if ls > 0 else _
```

No - when ls=0 we want the original loss. So:

```python
logits, loss_internal = model(xb, yb)
ls = getattr(args, "label_smoothing", 0.0)
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), reduction='mean', label_smoothing=ls) if ls > 0 else loss_internal
```

Good.

**Step 2: Implement**

In the training loop, replace:
```python
_, loss = model(xb, yb)
```

with:
```python
logits, loss = model(xb, yb)
if getattr(args, "label_smoothing", 0) > 0:
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), reduction='mean', label_smoothing=args.label_smoothing)
```

**Step 3: Verify**

Run: `cd mainrun && python3 train.py --approach 3 --log-file ./logs/test_a3.log` (Ctrl+C after a few steps)
Expected: No errors.

---

## Task 5: Implement RoPE in CausalSelfAttention

**Files:**
- Modify: `mainrun/train.py` (GPTConfig, CausalSelfAttention, Block, GPT)
- Create: RoPE helper logic inline in train.py

**Step 1: Add use_rope to GPTConfig**

In GPTConfig dataclass, add: `use_rope: bool = False`

**Step 2: Implement RoPE application**

RoPE (Rotary Position Embedding) multiplies Q and K by rotation matrices. Add before the CausalSelfAttention class:

```python
def apply_rope(q: torch.Tensor, k: torch.Tensor, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to q and k. q,k: (B, n_head, T, head_dim)."""
    B, n_head, T, d = q.size()
    device = q.device
    inv_freq = 1.0 / (10000 ** (torch.arange(0, d, 2, device=device).float() / d))
    t = torch.arange(T, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(0).unsqueeze(0)
    sin = emb.sin().unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims. x: (..., d)."""
    x1, x2 = x[..., : x.size(-1) // 2], x[..., x.size(-1) // 2 :]
    return torch.cat((-x2, x1), dim=-1)
```

**Step 3: Modify CausalSelfAttention to use RoPE when cfg.use_rope**

In CausalSelfAttention.forward, after extracting q, k, v:

```python
if getattr(self, 'use_rope', False):
    q, k = apply_rope(q, k, self.head_dim)
```

We need to pass use_rope to the attention. Add to CausalSelfAttention.__init__:
```python
self.use_rope = getattr(cfg, 'use_rope', False)
```

And in GPTConfig, ensure use_rope is passed from args. In cfg creation:
```python
cfg = GPTConfig(
    ...
    use_rope=getattr(args, "use_rope", False),
)
```

**Step 4: Remove pos_emb when use_rope**

When use_rope is True, we should not add pos_emb (RoPE replaces it). In GPT.forward:

```python
if getattr(self.cfg, 'use_rope', False):
    x = self.drop(tok)  # no pos_emb
else:
    pos = self.pos_emb[:, :T, :]
    x = self.drop(tok + pos)
```

**Step 5: Verify**

Run: `cd mainrun && python3 train.py --approach 3 --log-file ./logs/test_a3_rope.log` (Ctrl+C after ~20 steps)
Expected: No errors, loss decreases.

---

## Task 6: Wire GPTConfig.use_rope from args

**Files:**
- Modify: `mainrun/train.py` (GPTConfig instantiation, around line 291)

**Step 1: Add use_rope to GPTConfig**

In the GPTConfig instantiation, add:
```python
use_rope=getattr(args, "use_rope", False),
```

**Step 2: Add use_rope to GPTConfig dataclass**

In GPTConfig:
```python
use_rope: bool = False
```

---

## Task 7: Create overnight run script

**Files:**
- Create: `mainrun/run_overnight.sh`

**Step 1: Create script**

```bash
#!/bin/bash
# Run all three approaches sequentially. Logs: approach1.log, approach2.log, approach3.log
set -e
cd "$(dirname "$0")"

echo "=== Overnight runs: Approach 1, 2, 3 ==="
echo "Started at $(date)"
echo ""

# Ensure dataset is ready
python3 download_dataset.py 2>/dev/null || true

for APPROACH in 1 2 3; do
    echo ""
    echo "=========================================="
    echo ">>> APPROACH $APPROACH - Started at $(date)"
    echo "=========================================="
    python3 train.py --approach "$APPROACH" --log-file "./logs/approach${APPROACH}.log"
    echo ">>> APPROACH $APPROACH - Finished at $(date)"
done

echo ""
echo "=== All runs finished at $(date) ==="
echo ""
echo "Final validation losses:"
for APPROACH in 1 2 3; do
    if [ -f "./logs/approach${APPROACH}.log" ]; then
        LOSS=$(grep 'validation_step' "./logs/approach${APPROACH}.log" | tail -1 | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('loss','N/A'))" 2>/dev/null || echo "N/A")
        echo "  Approach $APPROACH: $LOSS"
    fi
done
```

**Step 4: Add optional Taskfile task**

In `Taskfile.yml`, add:

```yaml
  overnight:
    deps: [checkpoint, download]
    dir: mainrun
    cmds:
      - ./run_overnight.sh
```

Then user can run: `task overnight`


**Step 2: Make executable**

Run: `chmod +x mainrun/run_overnight.sh`

**Step 3: Verify script syntax**

Run: `bash -n mainrun/run_overnight.sh`
Expected: No output (success).

---

## Task 8: Fix setattr for new keys in approach config

**Files:**
- Modify: `mainrun/train.py` (approach config application)

**Step 1: Fix the setattr logic**

The approach config may have keys not in Hyperparameters (cosine_min_ratio, random_offset_packing, label_smoothing, use_rope). Use:

```python
if cli.approach is not None:
    config = APPROACH_CONFIGS[cli.approach]
    for k, v in config.items():
        setattr(args, k, v)
    if cli.log_file:
        args.log_file = cli.log_file
    else:
        args.log_file = f"./logs/approach{cli.approach}.log"
```

Ensure Hyperparameters has default values for cosine_min_ratio, random_offset_packing, label_smoothing, use_rope so that args has these attributes before setattr.

---

## Task 9: Dry run one approach

**Files:**
- N/A

**Step 1: Run approach 1 for 50 steps**

Run: `cd mainrun && timeout 120 python3 train.py --approach 1 --log-file ./logs/dryrun_approach1.log` (or run until ~50 steps then Ctrl+C)

**Step 2: Verify log**

Run: `grep validation_step mainrun/logs/dryrun_approach1.log | tail -1`
Expected: JSON line with loss value.

---

## Execution Handoff

After saving the plan, offer execution choice:

**"Plan complete and saved to `docs/plans/2026-03-16-three-approaches-overnight-runs-design.md`. Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?"**
