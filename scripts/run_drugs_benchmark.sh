#!/bin/bash
# Run Drugs-75K benchmark on GPU 6
# 75K molecules = large dataset, so use only key models
# 3 models × 3 targets × 3 seeds = 27 experiments

set -e

GPU=6
SEEDS="42 123 456"
MODELS="dko_gated dko_invariants attention"
TARGETS="drugs_ip drugs_ea drugs_chi"
OUTPUT_DIR="results/marcel_benchmark/drugs"
mkdir -p "$OUTPUT_DIR"

echo "Drugs-75K Benchmark on GPU $GPU"
echo "================================"
echo "Models: $MODELS"
echo "Targets: $TARGETS"
echo "Seeds: $SEEDS"
echo ""

total=0
done_count=0
for model in $MODELS; do
    for target in $TARGETS; do
        for seed in $SEEDS; do
            total=$((total + 1))
        done
    done
done
echo "Total experiments: $total"
echo ""

for model in $MODELS; do
    for target in $TARGETS; do
        for seed in $SEEDS; do
            done_count=$((done_count + 1))
            echo "[$(date +%H:%M:%S)] [$done_count/$total] $model / $target / seed=$seed"
            CUDA_VISIBLE_DEVICES=$GPU python -u -m dko.experiments.main_benchmark \
                --models "$model" --datasets "$target" --seeds "$seed" \
                --device "cuda:0" --output-dir "$OUTPUT_DIR/$model" \
                2>&1 || echo "  FAILED: $model / $target / seed=$seed"
        done
    done
done

echo ""
echo "Drugs-75K benchmark complete!"
