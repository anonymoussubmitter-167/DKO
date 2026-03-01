#!/bin/bash
#SBATCH --job-name=dko_hyperopt
#SBATCH --output=logs/hyperopt_%j.out
#SBATCH --error=logs/hyperopt_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --signal=B:TERM@120

# Hyperparameter optimization SLURM script for DKO
#
# Usage:
#   sbatch scripts/slurm_hyperopt.sh esol 50
#   sbatch scripts/slurm_hyperopt.sh freesolv 100 --sampler tpe

set -e

# Print job info
echo "=========================================="
echo "DKO Hyperparameter Optimization"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Load modules
module purge 2>/dev/null || true
module load python/3.9 2>/dev/null || module load python 2>/dev/null || true
module load cuda/11.8 2>/dev/null || module load cuda 2>/dev/null || true
module load cudnn/8.6 2>/dev/null || module load cudnn 2>/dev/null || true

# Activate virtual environment
if [ -f ~/venv/dko/bin/activate ]; then
    source ~/venv/dko/bin/activate
elif [ -f ~/miniconda3/envs/dko/bin/activate ]; then
    source ~/miniconda3/envs/dko/bin/activate
elif command -v conda &> /dev/null; then
    conda activate dko 2>/dev/null || true
fi

# Verify GPU
echo ""
echo "GPU Information:"
nvidia-smi || echo "No GPU available"

# Set environment variables
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID:-0}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Parse arguments
DATASET=${1:-"esol"}
N_TRIALS=${2:-50}
SAMPLER=${3:-"tpe"}
STUDY_NAME=${4:-"dko_${DATASET}_$(date +%Y%m%d)"}

echo ""
echo "Hyperopt Configuration:"
echo "  Dataset: $DATASET"
echo "  N Trials: $N_TRIALS"
echo "  Sampler: $SAMPLER"
echo "  Study Name: $STUDY_NAME"
echo "=========================================="

# Create output directory
OUTPUT_DIR="experiments/hyperopt/${STUDY_NAME}"
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Signal handler
cleanup() {
    echo ""
    echo "Received termination signal, saving hyperopt state..."
    exit 0
}
trap cleanup SIGTERM SIGINT

# Run hyperparameter optimization
echo ""
echo "Starting hyperparameter optimization..."
python scripts/run_hyperopt.py \
    --dataset "$DATASET" \
    --n-trials $N_TRIALS \
    --sampler "$SAMPLER" \
    --study-name "$STUDY_NAME" \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/hyperopt.log"

HYPEROPT_EXIT_CODE=$?

# Save job info
cat > "$OUTPUT_DIR/job_info.json" <<EOF
{
    "slurm_job_id": "$SLURM_JOB_ID",
    "node": "$SLURM_NODELIST",
    "dataset": "$DATASET",
    "n_trials": $N_TRIALS,
    "sampler": "$SAMPLER",
    "study_name": "$STUDY_NAME",
    "start_time": "$(date -Iseconds)",
    "exit_code": $HYPEROPT_EXIT_CODE
}
EOF

echo ""
echo "=========================================="
echo "Hyperopt completed with exit code: $HYPEROPT_EXIT_CODE"
echo "End Time: $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="

exit $HYPEROPT_EXIT_CODE
