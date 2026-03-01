#!/bin/bash
# Quick progress check for running experiments

echo "=== EXPERIMENT PROGRESS $(date) ==="
echo ""

# 10-seed validation
echo "10-SEED VALIDATION (GPU 5):"
count=$(grep -c "Test metrics" results/10seed_validation.log 2>/dev/null || echo "0")
echo "  $count/60 complete"

# Phase 3 GPU6
echo ""
echo "PHASE 3 GPU6 (freesolv, qm9_homo):"
count=$(grep -c "Test metrics" results/phase3_gpu6.log 2>/dev/null || echo "0")
echo "  $count/42 complete"

# Phase 3 GPU7
echo ""
echo "PHASE 3 GPU7 (bace, bbbp):"
count=$(grep -c "Test metrics" results/phase3_gpu7.log 2>/dev/null || echo "0")
echo "  $count/42 complete"

# GPU usage
echo ""
echo "GPU MEMORY:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep -E "^[567],"

echo ""
echo "Run './scripts/check_progress.sh' to check again"
