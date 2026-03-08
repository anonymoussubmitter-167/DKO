#!/bin/bash
# Run dko_first_order on Drugs on GPU 8
set -e

GPU=8
SEEDS="42 123 456"
MODELS="dko_first_order"
TARGETS="drugs_ip drugs_ea drugs_chi"
OUTPUT_DIR="results/marcel_benchmark/drugs"
mkdir -p "$OUTPUT_DIR"

echo "Drugs-75K (GPU $GPU) - dko_first_order"
echo "========================================"

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

echo "Drugs-75K (GPU $GPU) done!"
