#!/bin/bash
# Phase 1: Ablation on ESOL + Feature Quality Analysis
# GPU allocation: 3-9 for ablation, CPU for feature quality
#
# This runs 7 ablation configs x 3 seeds on ESOL to isolate which fix matters most.

set -e
cd "$(dirname "$0")/.."

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/ablation_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "======================================"
echo "Phase 1: Ablation Study + Feature Quality"
echo "Results: $RESULTS_DIR"
echo "======================================"

# --- Feature Quality Analysis (CPU) ---
echo ""
echo "[Phase 1a] Feature Quality Analysis (CPU)..."
mkdir -p "results/feature_quality_${TIMESTAMP}"
CUDA_VISIBLE_DEVICES="" python scripts/feature_quality_analysis.py \
    --output-dir "results/feature_quality_${TIMESTAMP}" \
    > "results/feature_quality_${TIMESTAMP}/output.log" 2>&1 &
FQ_PID=$!
echo "  Feature quality PID: $FQ_PID"

# --- Ablation Study (GPUs 3-9) ---
echo ""
echo "[Phase 1b] Ablation Study on ESOL..."

# Helper function to run one ablation config across seeds
run_config() {
    local CONFIG=$1 GPU=$2 LR=$3 KDIM=$4 NORM=$5 SREG=$6 MP=$7
    mkdir -p "$RESULTS_DIR/$CONFIG"
    for SEED in 42 123 456; do
        echo "  [GPU $GPU] $CONFIG seed=$SEED"
        CUDA_VISIBLE_DEVICES=$GPU python scripts/run_ablation_single.py \
            --dataset esol \
            --model dko \
            --seed $SEED \
            --lr $LR \
            --kernel-output-dim $KDIM \
            --norm-dim $NORM \
            --sigma-reg $SREG \
            --mixed-precision $MP \
            --output-dir "$RESULTS_DIR/$CONFIG"
    done
}

#                   Config              GPU  LR     KDIM  NORM  SREG   MP
run_config baseline         3    1e-5   32    1_2   1e-4   False > "$RESULTS_DIR/baseline.log" 2>&1 &
run_config fix_lr           4    1e-4   32    1_2   1e-4   False > "$RESULTS_DIR/fix_lr.log" 2>&1 &
run_config fix_kdim         5    1e-5   64    1_2   1e-4   False > "$RESULTS_DIR/fix_kdim.log" 2>&1 &
run_config fix_norm         6    1e-5   32    1     1e-2   False > "$RESULTS_DIR/fix_norm.log" 2>&1 &
run_config fix_lr_kdim      7    1e-4   64    1_2   1e-4   False > "$RESULTS_DIR/fix_lr_kdim.log" 2>&1 &
run_config fix_lr_kdim_norm 8    1e-4   64    1     1e-2   False > "$RESULTS_DIR/fix_lr_kdim_norm.log" 2>&1 &
run_config all_fixes        9    1e-4   64    1     1e-2   True  > "$RESULTS_DIR/all_fixes.log" 2>&1 &

echo ""
echo "All jobs launched. Waiting for completion..."
echo "Monitor with: tail -f $RESULTS_DIR/*.log"
wait

echo ""
echo "======================================"
echo "Phase 1 Complete!"
echo "======================================"
echo "Feature quality results: results/feature_quality_${TIMESTAMP}/"
echo "Ablation results: $RESULTS_DIR/"
