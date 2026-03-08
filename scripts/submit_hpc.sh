#!/bin/bash
#SBATCH --job-name=dko_training
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --signal=B:TERM@120

# =============================================================================
# DKO HPC Training Submission Script (Bulletproof Edition)
# =============================================================================
#
# Usage:
#   sbatch submit_hpc.sh [config_file] [experiment_name]
#
# Examples:
#   sbatch submit_hpc.sh configs/experiments/dko_lipophilicity.yaml
#   sbatch --gres=gpu:v100:1 submit_hpc.sh configs/experiments/dko_esol.yaml esol_run1
#   sbatch --mail-user=your@email.com submit_hpc.sh configs/base_config.yaml
#
# Arguments:
#   config_file: Path to experiment configuration (default: configs/base_config.yaml)
#   experiment_name: Optional experiment name (default: auto-generated)
#
# Features:
#   - Automatic checkpoint resumption on requeue
#   - Graceful signal handling for preemption
#   - Environment validation before training
#   - Comprehensive logging and error reporting
#   - Automatic retry on transient failures
#
# =============================================================================

# ============================================================================
# CONFIGURATION
# ============================================================================

# Retry configuration
MAX_RETRIES=3
RETRY_DELAY=30  # seconds

# Checkpoint directory
CHECKPOINT_DIR="checkpoints"

# Required disk space (in KB)
MIN_DISK_SPACE=10485760  # 10 GB

# Required memory check (percentage of requested that must be available)
MIN_MEMORY_PERCENT=80

# ============================================================================
# ERROR HANDLING AND CLEANUP
# ============================================================================

# Track if we received a signal
RECEIVED_SIGNAL=0
TRAINING_PID=0

# Cleanup function
cleanup() {
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running cleanup..."

    if [ $TRAINING_PID -ne 0 ] && kill -0 $TRAINING_PID 2>/dev/null; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Sending SIGTERM to training process (PID: $TRAINING_PID)"
        kill -TERM $TRAINING_PID 2>/dev/null

        # Wait up to 60 seconds for graceful shutdown
        for i in $(seq 1 60); do
            if ! kill -0 $TRAINING_PID 2>/dev/null; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training process terminated gracefully"
                break
            fi
            sleep 1
        done

        # Force kill if still running
        if kill -0 $TRAINING_PID 2>/dev/null; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Force killing training process"
            kill -9 $TRAINING_PID 2>/dev/null
        fi
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cleanup complete"
}

# Signal handler
handle_signal() {
    local signal=$1
    RECEIVED_SIGNAL=1
    echo ""
    echo "=========================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Received signal: $signal"
    echo "=========================================="

    if [ "$signal" = "TERM" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Job being preempted or terminated"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checkpoint will be saved automatically by training script"
    fi

    cleanup

    # If preempted and requeue is enabled, exit with special code
    if [ "$signal" = "TERM" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Exiting for requeue..."
        exit 0  # Exit cleanly so SLURM can requeue
    fi

    exit 1
}

# Set up signal handlers
trap 'handle_signal TERM' SIGTERM
trap 'handle_signal INT' SIGINT
trap 'handle_signal HUP' SIGHUP
trap cleanup EXIT

# ============================================================================
# LOGGING FUNCTIONS
# ============================================================================

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1"
}

log_warn() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] $1" >&2
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $1" >&2
}

# ============================================================================
# ENVIRONMENT VALIDATION
# ============================================================================

validate_environment() {
    log_info "Validating environment..."
    local errors=0

    # Check Python
    if ! command -v python &> /dev/null; then
        log_error "Python not found in PATH"
        errors=$((errors + 1))
    else
        log_info "Python: $(python --version 2>&1)"
    fi

    # Check PyTorch
    if ! python -c "import torch" 2>/dev/null; then
        log_error "PyTorch not installed or not importable"
        errors=$((errors + 1))
    else
        local torch_version=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
        log_info "PyTorch: $torch_version"
    fi

    # Check CUDA
    local cuda_available=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
    if [ "$cuda_available" != "True" ]; then
        log_warn "CUDA not available - will run on CPU (this may be very slow)"
    else
        local gpu_count=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
        local gpu_name=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
        log_info "CUDA available: $gpu_count GPU(s) - $gpu_name"
    fi

    # Check DKO package
    if ! python -c "import dko" 2>/dev/null; then
        log_error "DKO package not installed. Run: pip install -e ."
        errors=$((errors + 1))
    else
        log_info "DKO package: OK"
    fi

    # Check config file exists
    if [ ! -f "$CONFIG_FILE" ]; then
        log_error "Config file not found: $CONFIG_FILE"
        errors=$((errors + 1))
    else
        log_info "Config file: $CONFIG_FILE"
    fi

    # Check disk space
    local available_space=$(df -k . | awk 'NR==2 {print $4}')
    if [ "$available_space" -lt "$MIN_DISK_SPACE" ]; then
        log_error "Insufficient disk space: ${available_space}KB available, ${MIN_DISK_SPACE}KB required"
        errors=$((errors + 1))
    else
        log_info "Disk space: $(numfmt --to=iec-i --suffix=B $((available_space * 1024))) available"
    fi

    # Check RDKit (required for molecular processing)
    if ! python -c "from rdkit import Chem" 2>/dev/null; then
        log_warn "RDKit not installed - conformer generation will fail"
    else
        log_info "RDKit: OK"
    fi

    if [ $errors -gt 0 ]; then
        log_error "Environment validation failed with $errors error(s)"
        return 1
    fi

    log_info "Environment validation passed"
    return 0
}

# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

find_latest_checkpoint() {
    local exp_name=$1
    local checkpoint_pattern="${CHECKPOINT_DIR}/${exp_name}*.pt"

    # Look for preemption checkpoint first
    if [ -f "${CHECKPOINT_DIR}/preemption_checkpoint.pt" ]; then
        echo "${CHECKPOINT_DIR}/preemption_checkpoint.pt"
        return 0
    fi

    # Look for latest regular checkpoint
    local latest=$(ls -t ${checkpoint_pattern} 2>/dev/null | head -n1)
    if [ -n "$latest" ]; then
        echo "$latest"
        return 0
    fi

    return 1
}

# ============================================================================
# MAIN SCRIPT
# ============================================================================

# Print header
echo "=========================================="
echo "DKO Training Job (Bulletproof Edition)"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Job Name: ${SLURM_JOB_NAME:-local}"
echo "Array Task: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "GPUs: ${SLURM_GPUS:-N/A}"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-$(nproc)}"
echo "Memory: ${SLURM_MEM_PER_NODE:-N/A}"
echo "Working Dir: $(pwd)"
echo "Started: $(date)"
echo "Restart Count: ${SLURM_RESTART_COUNT:-0}"
echo "=========================================="

# Arguments
CONFIG_FILE=${1:-configs/base_config.yaml}
EXPERIMENT_NAME=${2:-""}

# Auto-generate experiment name if not provided
if [ -z "$EXPERIMENT_NAME" ]; then
    # Extract base name from config file
    config_base=$(basename "$CONFIG_FILE" .yaml)
    EXPERIMENT_NAME="${config_base}_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
fi

log_info "Config: $CONFIG_FILE"
log_info "Experiment: $EXPERIMENT_NAME"

# Create necessary directories
mkdir -p logs
mkdir -p "$CHECKPOINT_DIR"
mkdir -p results

# Load modules (with fallbacks for different clusters)
log_info "Loading modules..."

# Try different module configurations
load_modules() {
    # Try CUDA 11.8 first, then 11.7, then 12.x
    for cuda_version in cuda/11.8 cuda/11.7 cuda/12.0 cuda; do
        if module load $cuda_version 2>/dev/null; then
            log_info "Loaded: $cuda_version"
            break
        fi
    done

    # Try Python module
    for python_version in python/3.10 python/3.9 python/3.11 python; do
        if module load $python_version 2>/dev/null; then
            log_info "Loaded: $python_version"
            break
        fi
    done

    # Try conda/anaconda
    for conda_module in anaconda3 anaconda conda miniconda3; do
        if module load $conda_module 2>/dev/null; then
            log_info "Loaded: $conda_module"
            break
        fi
    done
}

# Only load modules if we're on SLURM
if [ -n "$SLURM_JOB_ID" ]; then
    load_modules 2>/dev/null || log_warn "Module loading not available or failed"
fi

# Activate conda environment (try multiple methods)
activate_conda() {
    # Method 1: source activate
    if source activate dko 2>/dev/null; then
        log_info "Activated conda environment: dko (source activate)"
        return 0
    fi

    # Method 2: conda activate
    if conda activate dko 2>/dev/null; then
        log_info "Activated conda environment: dko (conda activate)"
        return 0
    fi

    # Method 3: Direct path activation
    if [ -f "$HOME/miniconda3/envs/dko/bin/activate" ]; then
        source "$HOME/miniconda3/envs/dko/bin/activate"
        log_info "Activated conda environment: dko (direct path)"
        return 0
    fi

    log_warn "Could not activate conda environment 'dko' - using system Python"
    return 1
}

activate_conda

# Set environment variables
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-$(nproc)}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-$(nproc)}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-$(nproc)}

# CUDA configuration
if [ -n "$SLURM_LOCALID" ]; then
    export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
fi

# PyTorch settings for reproducibility and performance
export CUBLAS_WORKSPACE_CONFIG=:16:8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Navigate to project directory
cd "${SLURM_SUBMIT_DIR:-$(dirname $(dirname $(realpath $0)))}"

# Validate environment
echo ""
if ! validate_environment; then
    log_error "Environment validation failed. Exiting."
    exit 1
fi
echo ""

# Check for existing checkpoint to resume from
RESUME_CHECKPOINT=""
checkpoint=$(find_latest_checkpoint "$EXPERIMENT_NAME")
if [ -n "$checkpoint" ]; then
    log_info "Found checkpoint to resume from: $checkpoint"
    RESUME_CHECKPOINT="--resume $checkpoint"

    # If this is a restarted job, log it
    if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then
        log_info "This is restart #${SLURM_RESTART_COUNT} - resuming from checkpoint"
    fi
fi

# Build training command
TRAIN_CMD="python scripts/train_single_experiment.py --config $CONFIG_FILE --experiment-name $EXPERIMENT_NAME $RESUME_CHECKPOINT"

log_info "Training command: $TRAIN_CMD"
echo ""
echo "=========================================="
echo "Starting Training"
echo "=========================================="
echo ""

# Run training with retry logic
attempt=0
success=false

while [ $attempt -lt $MAX_RETRIES ] && [ "$success" = "false" ]; do
    attempt=$((attempt + 1))

    if [ $attempt -gt 1 ]; then
        log_info "Retry attempt $attempt of $MAX_RETRIES (waiting ${RETRY_DELAY}s)..."
        sleep $RETRY_DELAY

        # Check for new checkpoint after failure
        new_checkpoint=$(find_latest_checkpoint "$EXPERIMENT_NAME")
        if [ -n "$new_checkpoint" ] && [ "$new_checkpoint" != "$checkpoint" ]; then
            checkpoint="$new_checkpoint"
            RESUME_CHECKPOINT="--resume $checkpoint"
            TRAIN_CMD="python scripts/train_single_experiment.py --config $CONFIG_FILE --experiment-name $EXPERIMENT_NAME $RESUME_CHECKPOINT"
            log_info "Found newer checkpoint: $checkpoint"
        fi
    fi

    log_info "Starting training (attempt $attempt/$MAX_RETRIES)..."

    # Run training in background so we can catch signals
    $TRAIN_CMD &
    TRAINING_PID=$!

    # Wait for training to complete
    wait $TRAINING_PID
    exit_code=$?
    TRAINING_PID=0

    if [ $exit_code -eq 0 ]; then
        success=true
        log_info "Training completed successfully"
    elif [ $RECEIVED_SIGNAL -eq 1 ]; then
        log_info "Training interrupted by signal - checkpoint should be saved"
        exit 0  # Exit cleanly for requeue
    else
        log_warn "Training failed with exit code: $exit_code"

        # Check for common recoverable errors
        if [ $exit_code -eq 137 ]; then
            log_warn "Process killed (likely OOM) - consider reducing batch size"
        elif [ $exit_code -eq 139 ]; then
            log_error "Segmentation fault - this may indicate a bug"
        fi
    fi
done

if [ "$success" = "false" ]; then
    log_error "Training failed after $MAX_RETRIES attempts"
    exit 1
fi

# Print completion info
echo ""
echo "=========================================="
echo "Job Completed Successfully"
echo "=========================================="
echo "Finished: $(date)"
echo "Duration: $SECONDS seconds"
echo "Experiment: $EXPERIMENT_NAME"
echo "=========================================="

exit 0
