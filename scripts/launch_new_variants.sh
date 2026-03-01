#!/bin/bash
# =============================================================================
# DKO Experiment Suite v2 — GPU Launch Script
#
# Available GPUs: 0 (light), 5, 6 (free), 7, 8, 9 (light)
# Datasets: ESOL, QM9-Gap, QM9-LUMO, Lipophilicity
#
# Step 1: Analysis scripts (parallel with Step 2)
# Step 2: 7 new variant models × 4 datasets × 3 seeds = 84 experiments
# Step 3: Curriculum + Optuna (manual, after reviewing Step 2)
# =============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="results/new_variants_${TIMESTAMP}/logs"
RESULTS_DIR="results/new_variants_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

SEEDS="42 123 456"
DATASETS="esol qm9_gap qm9_lumo lipophilicity"
MAX_EPOCHS=300

echo "============================================================"
echo "DKO Experiment Suite v2 — Launch"
echo "============================================================"
echo "Timestamp: $TIMESTAMP"
echo "Results:   $RESULTS_DIR"
echo "Logs:      $LOG_DIR"
echo "Seeds:     $SEEDS"
echo "Datasets:  $DATASETS"
echo "============================================================"

# =============================================================================
# Step 1: Analysis scripts (CPU + 1 GPU)
# =============================================================================
echo ""
echo "--- Step 1: Analysis Scripts ---"

# G — Feature Variance Audit (CPU)
echo "Launching: Feature Variance Audit (CPU)"
nohup python scripts/feature_variance_audit.py \
    > "$LOG_DIR/exp_G_variance_audit.log" 2>&1 &
PID_G=$!
echo "  PID: $PID_G"

# R — SCC Quartile Analysis (CPU)
echo "Launching: SCC Quartile Analysis (CPU)"
nohup python scripts/scc_quartile_analysis.py \
    > "$LOG_DIR/exp_R_scc_quartile.log" 2>&1 &
PID_R=$!
echo "  PID: $PID_R"

# T — Synthetic Validation (GPU 0)
echo "Launching: Synthetic Validation (GPU 0)"
nohup env CUDA_VISIBLE_DEVICES=0 python scripts/synthetic_validation.py \
    --device cuda:0 \
    > "$LOG_DIR/exp_T_synthetic.log" 2>&1 &
PID_T=$!
echo "  PID: $PID_T"

# =============================================================================
# Step 2: New Variant Benchmark (6 GPUs)
# =============================================================================
echo ""
echo "--- Step 2: New Variant Benchmark (84 experiments) ---"

run_variant() {
    local GPU=$1
    local MODEL=$2
    local LOGNAME="${MODEL}"

    echo "Launching: $MODEL on GPU $GPU"
    nohup env CUDA_VISIBLE_DEVICES=$GPU python -m dko.experiments.main_benchmark \
        --models "$MODEL" \
        --datasets $DATASETS \
        --seeds $SEEDS \
        --output-dir "$RESULTS_DIR/$MODEL" \
        --device cuda:0 \
        > "$LOG_DIR/exp_${LOGNAME}.log" 2>&1 &

    local PID=$!
    echo "  PID: $PID"
    echo "$PID" >> "$LOG_DIR/pids.txt"
}

# GPU 5: dko_eigenspectrum
run_variant 5 dko_eigenspectrum

# GPU 6: dko_invariants
run_variant 6 dko_invariants

# GPU 7: dko_gated
run_variant 7 dko_gated

# GPU 8: dko_residual
run_variant 8 dko_residual

# GPU 9: dko_lowrank
run_variant 9 dko_lowrank

# GPU 0: dko_crossattn + dko_router (sequential on same GPU after synthetic finishes)
echo "Launching: dko_crossattn + dko_router on GPU 0 (after synthetic validation)"
nohup bash -c "
    # Wait for synthetic validation to finish
    wait $PID_T 2>/dev/null || true

    echo 'Synthetic done. Starting dko_crossattn...'
    CUDA_VISIBLE_DEVICES=0 python -m dko.experiments.main_benchmark \
        --models dko_crossattn \
        --datasets $DATASETS \
        --seeds $SEEDS \
        --output-dir '$RESULTS_DIR/dko_crossattn' \
        --device cuda:0 \
        >> '$LOG_DIR/exp_dko_crossattn.log' 2>&1

    echo 'dko_crossattn done. Starting dko_router...'
    CUDA_VISIBLE_DEVICES=0 python -m dko.experiments.main_benchmark \
        --models dko_router \
        --datasets $DATASETS \
        --seeds $SEEDS \
        --output-dir '$RESULTS_DIR/dko_router' \
        --device cuda:0 \
        >> '$LOG_DIR/exp_dko_router.log' 2>&1

    echo 'GPU 0 experiments complete.'
" > "$LOG_DIR/exp_gpu0_sequence.log" 2>&1 &
PID_GPU0=$!
echo "  PID: $PID_GPU0"

echo ""
echo "============================================================"
echo "All experiments launched!"
echo "============================================================"
echo ""
echo "Monitor with:"
echo "  tail -f $LOG_DIR/exp_*.log"
echo ""
echo "Check progress:"
echo "  python scripts/monitor_jobs.py 2>/dev/null || ps aux | grep main_benchmark"
echo ""
echo "PIDs:"
echo "  G (variance audit): $PID_G"
echo "  R (SCC quartile):   $PID_R"
echo "  T (synthetic):      $PID_T"
echo "  GPU 0 (crossattn+router): $PID_GPU0"
echo ""
echo "--- Step 3 (manual, after Step 2 completes) ---"
echo "Curriculum learning:"
echo "  CUDA_VISIBLE_DEVICES=5 python scripts/run_curriculum.py --models dko_eigenspectrum dko_residual --device cuda:0"
echo ""
echo "Optuna tuning:"
echo "  CUDA_VISIBLE_DEVICES=6 python scripts/run_hyperopt_campaign.py --device cuda:0"
echo ""
