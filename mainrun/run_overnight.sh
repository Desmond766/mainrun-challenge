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
