#!/bin/bash
# Sequential ablation runs: each run adds one improvement over the previous.
# Run 1: AdamW + LR + weight decay
# Run 2: Run 1 + learning rate warmup
# Run 3: Run 1+2 + residual projection scaling
# Run 4: Run 1+2+3 + dropout tuning
# Run 5: Run 4 + param-grouped weight decay

set -e
cd "$(dirname "$0")"

echo "=== Mainrun Ablation Experiments ==="
echo "Starting at $(date)"
echo ""

# Ensure dataset is downloaded
python3 download_dataset.py

for RUN in 1 2 3 4 5; do
    echo ""
    echo "=========================================="
    echo ">>> RUN $RUN of 5 - Starting at $(date)"
    echo "=========================================="
    python3 train.py --run "$RUN"
    echo ">>> RUN $RUN completed at $(date)"
    echo "    Log saved to logs/run${RUN}.log"
done

echo ""
echo "=== All 5 runs completed at $(date) ==="
echo ""
echo "Results summary (final validation loss per run):"
SUMMARY_FILE="logs/ablation_results.txt"
{
    echo "Mainrun Ablation Results - $(date)"
    echo "================================"
    echo "Baseline (from README): 1.754"
    echo ""
    echo "Run 1: AdamW + LR 3e-4 + weight_decay 0.01"
    echo "Run 2: Run 1 + learning rate warmup (5%)"
    echo "Run 3: Run 2 + residual projection scaling"
    echo "Run 4: Run 3 + dropout 0.1->0.05"
    echo "Run 5: Run 4 + param-grouped weight decay"
    echo ""
    for RUN in 1 2 3 4 5; do
        if [ -f "logs/run${RUN}.log" ]; then
            LOSS=$(grep "validation_step" logs/run${RUN}.log | tail -1 | jq -r '.loss // "N/A"' 2>/dev/null || grep -o '"loss":[^,}]*' logs/run${RUN}.log | tail -1 | cut -d: -f2)
            echo "Run $RUN final val loss: $LOSS"
        else
            echo "Run $RUN: (log not found)"
        fi
    done
} | tee "$SUMMARY_FILE"
echo ""
echo "Summary saved to $SUMMARY_FILE"
