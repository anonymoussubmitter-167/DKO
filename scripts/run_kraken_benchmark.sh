#!/bin/bash
# Run Kraken benchmark on GPU 9
# Key DKO variants + baselines on 4 Sterimol targets

set -e

GPU=9
OUTPUT_DIR="results/kraken_benchmark"
SEEDS="42 123 456"
DATASETS="kraken_B5 kraken_L kraken_burB5 kraken_burL"

# Models to test (most promising DKO variants + baselines)
MODELS="dko_gated dko_invariants dko_eigenspectrum dko_residual dko_first_order attention mean_ensemble"

mkdir -p "$OUTPUT_DIR"

echo "Kraken Benchmark"
echo "================"
echo "GPU: $GPU"
echo "Models: $MODELS"
echo "Datasets: $DATASETS"
echo "Seeds: $SEEDS"
echo ""

total=0
done=0

# Count total experiments
for model in $MODELS; do
    for dataset in $DATASETS; do
        for seed in $SEEDS; do
            total=$((total + 1))
        done
    done
done
echo "Total experiments: $total"
echo ""

for model in $MODELS; do
    for dataset in $DATASETS; do
        for seed in $SEEDS; do
            done=$((done + 1))
            echo "[$done/$total] $model / $dataset / seed=$seed"

            CUDA_VISIBLE_DEVICES=$GPU python -u -m dko.experiments.main_benchmark \
                --models "$model" \
                --datasets "$dataset" \
                --seeds "$seed" \
                --device "cuda:0" \
                --output-dir "$OUTPUT_DIR/$model" \
                2>&1 || echo "  FAILED: $model / $dataset / seed=$seed"
        done
    done
done

echo ""
echo "Kraken benchmark complete!"
echo "Results in: $OUTPUT_DIR"
