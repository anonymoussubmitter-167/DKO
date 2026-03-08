# Distribution Kernel Operators (DKO) for Molecular Property Prediction

Code for the paper: *"When Does Conformer Geometry Help? Complementarity of 3D Ensemble Statistics and 2D Fingerprints for Molecular Property Prediction"* (under review).

## Overview

This project investigates when three-dimensional conformer ensemble statistics complement two-dimensional molecular fingerprints for property prediction. Using Distribution Kernel Operators (DKOs), we extract mean and covariance features from conformer ensembles and evaluate 13 model configurations across 14 regression targets spanning MoleculeNet, QM9, and MARCEL benchmarks.

### Key Findings

- **Selective complementarity**: Hybrid FP+conformer features yield 9.9% RMSE reduction on ESOL, 3.9% on FreeSolv, and 4.2% on QM9-HOMO
- **Performance hierarchy**: SchNet ≈ FP+3D > FP+XGBoost > neural conformer ensembles
- **Mechanistic insight**: Conformer mean features carry 2–8× more information per feature than fingerprint bits; covariance features contribute <2% of model signal
- **Property taxonomy**: Solvation properties benefit from conformer geometry; electronic and steric properties do not

## Installation

### Requirements

- Python 3.9+
- CUDA 11.8+ (for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/anonymoussubmitter-167/DKO.git
cd DKO

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Optional: RDKit Installation

If RDKit installation fails via pip, use conda:

```bash
conda install -c conda-forge rdkit
```

## Quick Start

### 1. Prepare Datasets

```bash
# Download and preprocess all datasets
python scripts/prepare_datasets.py --all

# Or prepare specific datasets
python scripts/prepare_datasets.py --datasets esol freesolv lipophilicity
```

### 2. Run Experiments

```bash
# Run main benchmark on all datasets
python main_benchmark.py --models all --datasets all --seeds 42 123 456

# Run on specific dataset with specific model
python main_benchmark.py --models dko_gated --datasets esol --seeds 42

# Run hybrid FP+conformer experiments
python scripts/run_hybrid_10seed.py
```

### 3. Analyze Results

```bash
# Compile results across experiments
python scripts/compile_results.py
```

## Project Structure

```
DKO/
├── dko/
│   ├── models/              # Model implementations
│   │   ├── __init__.py      # MODEL_REGISTRY with all 13 configurations
│   │   ├── dko.py           # Distribution Kernel Operator (base)
│   │   ├── dko_variants.py  # 7 DKO variants (gated, invariants, etc.)
│   │   ├── attention.py     # Set Transformer conformer aggregation
│   │   ├── deepsets.py      # DeepSets baseline
│   │   └── ensemble_baselines.py  # Mean/single conformer baselines
│   ├── data/                # Data loading and processing
│   │   ├── datasets.py      # ConformerDataset, DataLoader creation
│   │   ├── conformers.py    # ETKDG conformer generation
│   │   ├── features.py      # Geometric + 3D descriptor extraction
│   │   └── splits.py        # Random and scaffold splitting
│   ├── training/            # Training infrastructure
│   │   ├── trainer.py       # Training loop with early stopping
│   │   └── evaluator.py     # Evaluation metrics
│   └── analysis/            # Analysis utilities
│       ├── scc.py           # Conformer diversity metric
│       └── visualization.py
├── configs/                 # YAML configuration files
├── data/conformers/         # Pre-computed conformer pickles
├── scripts/                 # Experiment and analysis scripts
├── results/                 # Experiment results
├── main_benchmark.py        # Main entry point
├── requirements.txt
└── setup.py
```

## Models

### DKO Variants (9 configurations)
| Model | Covariance Representation | Summary Dim |
|-------|--------------------------|-------------|
| dko_gated | Learned sigmoid gate fusing μ/Σ streams | 64 |
| dko_invariants | 5 scalar statistics (tr, log-det, ‖·‖_F, λ₁/λₖ, spectral) | 5 |
| dko_eigenspectrum | Top-k eigenvalues | 64 |
| dko_lowrank | Top-k eigenvalues + eigenvector projection | 128 |
| dko_residual | μ prediction + learned Σ correction | 64 |
| dko_crossattn | Cross-attention (μ queries, Σ keys/values) | 64 |
| dko_router | Diversity-based MoE routing | 64 |
| dko_first_order | μ-only (no covariance, ablation) | 0 |
| dko_diagonal | Per-feature variances only (ablation) | 1024 |

### Baselines (4 configurations)
- **attention**: Set Transformer conformer aggregation
- **mean_ensemble**: Mean pooling over conformers
- **single_conformer**: Lowest-energy conformer only
- **DeepSets**: Permutation-invariant aggregation

### Hybrid (non-neural)
- **FP+XGBoost**: Morgan fingerprints (2048-bit) with XGBoost
- **FP+μ+σ**: Hybrid fingerprint + conformer statistics with XGBoost
- **FP+3D**: Hybrid with 28 physicochemical 3D descriptors (PMI, SASA, USR)

## Datasets

| Dataset | Property | N | Category |
|---------|----------|---|----------|
| ESOL | Aqueous solubility | 1,128 | Solvation |
| FreeSolv | Hydration ΔG | 642 | Solvation |
| Lipophilicity | LogP | 4,200 | Solvation |
| QM9-Gap | HOMO–LUMO gap | 133K | Electronic |
| QM9-HOMO | HOMO energy | 133K | Electronic |
| QM9-LUMO | LUMO energy | 133K | Electronic |
| BDE | Bond dissociation energy | 5,915 | Electronic |
| Drugs-75K (×3) | Electronic properties | 75K | Electronic |
| Kraken (×4) | Sterimol B5/L/burB5/burL | 1,552 | Steric |

## Testing

```bash
pytest tests/
```

## License

MIT License - see LICENSE file for details.
