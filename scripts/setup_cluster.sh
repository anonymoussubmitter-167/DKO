#!/bin/bash
# Complete cluster setup script for DKO project
#
# Usage:
#   bash scripts/setup_cluster.sh              # Full setup with venv
#   bash scripts/setup_cluster.sh --no-venv    # Skip venv creation
#   bash scripts/setup_cluster.sh --no-tests   # Skip validation tests

set -e

echo "=========================================="
echo "DKO PROJECT - CLUSTER SETUP"
echo "=========================================="

# Parse arguments
INSTALL_VENV=true
RUN_TESTS=true

for arg in "$@"; do
    case $arg in
        --no-venv)
            INSTALL_VENV=false
            shift
            ;;
        --no-tests)
            RUN_TESTS=false
            shift
            ;;
        --help)
            echo "Usage: bash scripts/setup_cluster.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --no-venv    Skip virtual environment creation"
            echo "  --no-tests   Skip validation tests"
            echo "  --help       Show this help message"
            exit 0
            ;;
    esac
done

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo ""
echo "Project root: $PROJECT_ROOT"

# [1] Create directory structure
echo ""
echo "[1/6] Creating directory structure..."
mkdir -p data
mkdir -p logs
mkdir -p experiments
mkdir -p checkpoints
mkdir -p analysis
mkdir -p cluster_tests
mkdir -p configs/experiments

echo "  Created directories:"
echo "    data/           - Dataset storage"
echo "    logs/           - SLURM job logs"
echo "    experiments/    - Experiment outputs"
echo "    checkpoints/    - Model checkpoints"
echo "    analysis/       - Results analysis"
echo "    cluster_tests/  - Deployment tests"

# [2] Setup virtual environment
if [ "$INSTALL_VENV" = "true" ]; then
    echo ""
    echo "[2/6] Setting up virtual environment..."

    # Try to load Python module (cluster-specific)
    module purge 2>/dev/null || true
    module load python/3.9 2>/dev/null || module load python 2>/dev/null || true
    module load cuda/11.8 2>/dev/null || module load cuda 2>/dev/null || true

    VENV_PATH="${HOME}/venv/dko"

    if [ -d "$VENV_PATH" ]; then
        echo "  Virtual environment already exists at $VENV_PATH"
        echo "  Activating..."
        source "$VENV_PATH/bin/activate"
    else
        echo "  Creating virtual environment at $VENV_PATH..."
        python -m venv "$VENV_PATH"
        source "$VENV_PATH/bin/activate"

        # Upgrade pip
        pip install --upgrade pip

        # Install PyTorch with CUDA
        echo "  Installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

        # Install RDKit
        echo "  Installing RDKit..."
        pip install rdkit-pypi

        # Install other dependencies
        echo "  Installing other dependencies..."
        pip install numpy scipy scikit-learn pandas
        pip install pyyaml tqdm
        pip install optuna
        pip install matplotlib seaborn
        pip install pytest

        echo "  [OK] Virtual environment created and packages installed"
    fi
else
    echo ""
    echo "[2/6] Skipping virtual environment setup"
fi

# [3] Make scripts executable
echo ""
echo "[3/6] Making scripts executable..."
chmod +x scripts/*.sh 2>/dev/null || true
chmod +x scripts/*.py 2>/dev/null || true
echo "  [OK] Scripts are executable"

# [4] Install DKO package
echo ""
echo "[4/6] Installing DKO package..."
if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
    pip install -e . 2>/dev/null || echo "  [WARN] Could not install package in editable mode"
else
    echo "  [INFO] No setup.py found, using PYTHONPATH"
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
fi

# Verify installation
python -c "import dko; print('  [OK] DKO package importable')" 2>/dev/null || echo "  [WARN] DKO package not importable"

# [5] Download minimal test data
echo ""
echo "[5/6] Checking test data..."
if [ -f "data/.downloaded" ]; then
    echo "  [OK] Test data already downloaded"
else
    python -c "
import sys
sys.path.insert(0, '.')
from pathlib import Path

try:
    from dko.data.datasets import get_dataset
    print('  Downloading ESOL dataset (smallest)...')
    dataset = get_dataset('esol', root=Path('./data'), split='train')
    print(f'  [OK] Downloaded: {len(dataset)} samples')
    Path('data/.downloaded').touch()
except ImportError:
    print('  [WARN] Dataset loader not available, skipping')
except Exception as e:
    print(f'  [WARN] Could not download: {e}')
" 2>/dev/null || echo "  [WARN] Data download skipped"
fi

# [6] Run validation
if [ "$RUN_TESTS" = "true" ]; then
    echo ""
    echo "[6/6] Running validation tests..."

    if [ -f "scripts/validate_cluster_ready.py" ]; then
        python scripts/validate_cluster_ready.py

        if [ $? -eq 0 ]; then
            echo "  [OK] Validation passed"
        else
            echo "  [WARN] Validation had issues - review output above"
        fi
    else
        echo "  [WARN] Validation script not found, skipping"
    fi
else
    echo ""
    echo "[6/6] Skipping validation tests"
fi

# Print next steps
echo ""
echo "=========================================="
echo "SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate environment (if not already):"
echo "   source ~/venv/dko/bin/activate"
echo ""
echo "2. Run comprehensive validation:"
echo "   python scripts/validate_cluster_ready.py"
echo ""
echo "3. Submit a test job:"
echo "   sbatch scripts/submit_hpc.sh configs/experiments/dko_esol.yaml"
echo ""
echo "4. Monitor jobs:"
echo "   python scripts/monitor_jobs.py --status"
echo ""
echo "5. Submit batch experiments:"
echo "   python scripts/submit_all_experiments.py --dry-run"
echo ""
echo "6. Aggregate results after completion:"
echo "   python scripts/aggregate_results.py"
echo ""
echo "=========================================="
