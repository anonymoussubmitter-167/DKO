#!/bin/bash
# Run Phase 3 on remaining datasets using GPU 7
# Split the work: GPU 6 does freesolv+qm9_homo, GPU 7 does bace+bbbp

set -e

GPU=7
OUTPUT_DIR="results/phase3_remaining"
SEEDS="42 123 456"

echo "Phase 3 (GPU 7) - BACE and BBBP"
echo "========================================"

for dataset in bace bbbp; do
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

echo "Done!"
