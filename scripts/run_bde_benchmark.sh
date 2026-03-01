#!/bin/bash
# BDE benchmark on GPU 5
set -e
GPU=5
SEEDS="42 123 456"
MODELS="dko_gated dko_invariants dko_first_order attention mean_ensemble"
OUTPUT_DIR="results/marcel_benchmark/bde"
mkdir -p "$OUTPUT_DIR"

echo "BDE Benchmark on GPU $GPU"
for model in $MODELS; do
    for seed in $SEEDS; do
        echo "[$(date +%H:%M:%S)] $model / bde / seed=$seed"
        CUDA_VISIBLE_DEVICES=$GPU python -u -m dko.experiments.main_benchmark \
            --models "$model" --datasets "bde" --seeds "$seed" \
            --device "cuda:0" --output-dir "$OUTPUT_DIR/$model" \
            2>&1 || echo "  FAILED: $model / bde / seed=$seed"
    done
done
echo "BDE benchmark done!"
