#!/bin/bash
# Phase 2: Full Benchmark with All Fixes
# One dataset per GPU, 8 models x 3 seeds each
#
# GPU allocation:
#   GPU 1: esol
#   GPU 3: freesolv
#   GPU 4: lipophilicity
#   GPU 5: bace
#   GPU 6: bbbp
#   GPU 7: qm9_gap
#   GPU 8: qm9_homo
#   GPU 9: qm9_lumo

set -e
cd "$(dirname "$0")/.."

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/benchmark_fixed_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

# Models to benchmark (all fixed DKO variants + baselines)
# Note: dko_diagonal and dko_separate_nets are included as DKO variants
MODELS="dko dko_first_order dko_diagonal dko_separate_nets attention deepsets mean_ensemble single_conformer"
SEEDS="42 123 456"

echo "======================================"
echo "Phase 2: Full Benchmark (All Fixes Applied)"
echo "Results: $RESULTS_DIR"
echo "Models: $MODELS"
echo "Seeds: $SEEDS"
echo "======================================"

# Dataset -> GPU mapping
declare -A DATASET_GPU
DATASET_GPU[esol]=1
DATASET_GPU[freesolv]=3
DATASET_GPU[lipophilicity]=4
DATASET_GPU[bace]=5
DATASET_GPU[bbbp]=6
DATASET_GPU[qm9_gap]=7
DATASET_GPU[qm9_homo]=8
DATASET_GPU[qm9_lumo]=9

for DATASET in esol freesolv lipophilicity bace bbbp qm9_gap qm9_homo qm9_lumo; do
    GPU=${DATASET_GPU[$DATASET]}
    echo ""
    echo "Launching $DATASET on GPU $GPU..."

    (
        CUDA_VISIBLE_DEVICES=$GPU python -m dko.experiments.main_benchmark \
            --datasets "$DATASET" \
            --models $MODELS \
            --seeds $SEEDS \
            --output-dir "$RESULTS_DIR/$DATASET" \
            --device cuda
    ) > "$RESULTS_DIR/${DATASET}.log" 2>&1 &

    echo "  PID: $!"
done

echo ""
echo "All benchmark jobs launched. Waiting for completion..."
echo "Monitor with: tail -f $RESULTS_DIR/*.log"
wait

echo ""
echo "======================================"
echo "Phase 2 Complete!"
echo "======================================"
echo "Results directory: $RESULTS_DIR"
echo ""

# Aggregate all results
echo "Aggregating results..."
python scripts/aggregate_results.py --results-dir "$RESULTS_DIR" 2>/dev/null || true

echo "Done!"
