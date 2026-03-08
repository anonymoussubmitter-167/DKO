#!/bin/bash
# Run full MARCEL benchmark: Drugs-75K (3 targets) + BDE + Kraken (4 targets)
# Uses separate GPUs for efficiency

set -e

SEEDS="42 123 456"

# Key models to test
MODELS="dko_gated dko_invariants dko_first_order attention mean_ensemble"

echo "MARCEL Full Benchmark"
echo "====================="
echo ""

# --- GPU 9: Kraken (already running separately) ---

# --- GPU 5: BDE ---
echo "=== BDE on GPU 5 ==="
BDE_DIR="results/marcel_benchmark/bde"
mkdir -p "$BDE_DIR"

for model in $MODELS; do
    for seed in $SEEDS; do
        echo "BDE: $model / seed=$seed"
        CUDA_VISIBLE_DEVICES=5 python -u -m dko.experiments.main_benchmark \
            --models "$model" \
            --datasets "bde" \
            --seeds "$seed" \
            --device "cuda:0" \
            --output-dir "$BDE_DIR/$model" \
            2>&1 || echo "  FAILED: $model / bde / seed=$seed"
    done
done

echo ""
echo "BDE benchmark complete!"

# --- Drugs-75K on GPU 5 (after BDE finishes) ---
echo "=== Drugs-75K on GPU 5 ==="
DRUGS_DIR="results/marcel_benchmark/drugs"
mkdir -p "$DRUGS_DIR"

for model in $MODELS; do
    for target in drugs_ip drugs_ea drugs_chi; do
        for seed in $SEEDS; do
            echo "Drugs: $model / $target / seed=$seed"
            CUDA_VISIBLE_DEVICES=5 python -u -m dko.experiments.main_benchmark \
                --models "$model" \
                --datasets "$target" \
                --seeds "$seed" \
                --device "cuda:0" \
                --output-dir "$DRUGS_DIR/$model" \
                2>&1 || echo "  FAILED: $model / $target / seed=$seed"
        done
    done
done

echo ""
echo "Drugs benchmark complete!"
echo "Full MARCEL benchmark done!"
