#!/bin/bash
# Run Phase 3 variants on remaining 4 datasets
# Usage: ./scripts/run_phase3_remaining.sh <GPU_ID>

set -e

GPU=${1:-6}
OUTPUT_DIR="results/phase3_remaining"
SEEDS="42 123 456"

mkdir -p "$OUTPUT_DIR"

echo "Phase 3 Remaining Datasets"
echo "GPU: $GPU"
echo "Output: $OUTPUT_DIR"
echo "========================================"

for dataset in freesolv qm9_homo bace bbbp; do
    for model in dko_eigenspectrum dko_invariants dko_lowrank dko_residual dko_crossattn dko_gated dko_router; do
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
