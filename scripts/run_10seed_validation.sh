#!/bin/bash
# 10-seed statistical validation for key model comparisons
# Usage: ./scripts/run_10seed_validation.sh <GPU_ID>

set -e

GPU=${1:-5}
OUTPUT_DIR="results/10seed_validation"
SEEDS="42 123 456 789 1000 1111 2222 3333 4444 5555"

mkdir -p "$OUTPUT_DIR"

echo "10-Seed Statistical Validation"
echo "GPU: $GPU"
echo "Output: $OUTPUT_DIR"
echo "========================================"

# Run each model separately for each dataset and seed combination
# The benchmark script handles one (model, dataset, seed) at a time via lists

for dataset in esol lipophilicity; do
    for model in dko_gated dko_invariants attention; do
        for seed in $SEEDS; do
            echo ""
            echo ">>> $dataset / $model / seed=$seed"

            CUDA_VISIBLE_DEVICES=$GPU python -u -m dko.experiments.main_benchmark \
                --models "$model" \
                --datasets "$dataset" \
                --seeds "$seed" \
                --device "cuda:0" \
                --output-dir "$OUTPUT_DIR/$model" \
                2>&1 | tail -5
        done
    done
done

echo ""
echo "========================================"
echo "All experiments complete!"
echo "Run: python scripts/analyze_10seed.py"
