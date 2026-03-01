#!/bin/bash
#SBATCH --job-name=dko_batch
#SBATCH --output=logs/batch_%A_%a.out
#SBATCH --error=logs/batch_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --signal=B:TERM@120

# Batch submission script for DKO experiments
# Supports SLURM array jobs for running multiple experiments
#
# Usage:
#   sbatch --array=0-4 scripts/slurm_submit_batch.sh configs/experiments/
#   sbatch --array=0-9%3 scripts/slurm_submit_batch.sh configs/experiments/  # max 3 concurrent

set -e

# Print job info
echo "=========================================="
echo "SLURM Batch Job Information"
echo "=========================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
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

# Set environment variables
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID:-0}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Parse arguments
CONFIG_DIR=${1:-"configs/experiments"}
SEED_START=${2:-42}

# Create logs directory
mkdir -p logs

# Get list of config files
CONFIG_FILES=($(find "$CONFIG_DIR" -name "*.yaml" | sort))
NUM_CONFIGS=${#CONFIG_FILES[@]}

if [ $NUM_CONFIGS -eq 0 ]; then
    echo "ERROR: No config files found in $CONFIG_DIR"
    exit 1
fi

echo ""
echo "Found $NUM_CONFIGS config files"

# Get config for this array task
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    CONFIG_IDX=$SLURM_ARRAY_TASK_ID
else
    CONFIG_IDX=0
fi

if [ $CONFIG_IDX -ge $NUM_CONFIGS ]; then
    echo "ERROR: Array task ID ($CONFIG_IDX) exceeds number of configs ($NUM_CONFIGS)"
    exit 1
fi

CONFIG_FILE="${CONFIG_FILES[$CONFIG_IDX]}"
CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
SEED=$((SEED_START + SLURM_ARRAY_TASK_ID % 5))

echo ""
echo "Running Configuration:"
echo "  Config Index: $CONFIG_IDX"
echo "  Config File: $CONFIG_FILE"
echo "  Config Name: $CONFIG_NAME"
echo "  Seed: $SEED"
echo "=========================================="

# Output directory
OUTPUT_DIR="experiments/${CONFIG_NAME}_seed${SEED}_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

# Check for resume checkpoint
RESUME_ARG=""
if [ -f "$OUTPUT_DIR/checkpoints/last_model.pt" ]; then
    echo "Found checkpoint, will resume training"
    RESUME_ARG="--resume-from $OUTPUT_DIR/checkpoints/last_model.pt"
fi

# Signal handler for graceful shutdown
cleanup() {
    echo ""
    echo "Received termination signal, saving checkpoint..."
    # The training script handles checkpointing internally
    exit 0
}
trap cleanup SIGTERM SIGINT

# Run training
echo ""
echo "Starting training..."
python scripts/train_single_experiment.py \
    --config "$CONFIG_FILE" \
    --experiment-name "${CONFIG_NAME}_seed${SEED}" \
    --seed $SEED \
    $RESUME_ARG \
    2>&1 | tee "$OUTPUT_DIR/training.log"

TRAIN_EXIT_CODE=$?

# Save batch job info
cat > "$OUTPUT_DIR/batch_info.json" <<EOF
{
    "array_job_id": "$SLURM_ARRAY_JOB_ID",
    "array_task_id": "$SLURM_ARRAY_TASK_ID",
    "job_id": "$SLURM_JOB_ID",
    "node": "$SLURM_NODELIST",
    "config_file": "$CONFIG_FILE",
    "config_name": "$CONFIG_NAME",
    "seed": $SEED,
    "start_time": "$(date -Iseconds)",
    "exit_code": $TRAIN_EXIT_CODE
}
EOF

echo ""
echo "=========================================="
echo "Batch task completed with exit code: $TRAIN_EXIT_CODE"
echo "End Time: $(date)"
echo "Output saved to: $OUTPUT_DIR"
echo "=========================================="

exit $TRAIN_EXIT_CODE
