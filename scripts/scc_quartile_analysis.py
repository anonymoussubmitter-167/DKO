#!/usr/bin/env python
"""
Experiment R — SCC Quartile Analysis.

Computes SCC (Sum of Conformer Covariances, i.e., trace(sigma)) per test
molecule from conformer features, loads best checkpoints from benchmark
results, and evaluates models stratified by SCC quartiles.

Key question: Does DKO do relatively better on high-SCC molecules?

Usage:
    python scripts/scc_quartile_analysis.py
    python scripts/scc_quartile_analysis.py --datasets esol qm9_gap --device cuda:0
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from dko.models.dko import DKO, DKOFirstOrder
from dko.models.attention import AttentionAggregation
from dko.models.ensemble_baselines import MeanEnsemble, MeanFeatureAggregation
from dko.models.deepsets import DeepSets
from dko.training.evaluator import Evaluator


DATASETS = ['esol', 'freesolv', 'lipophilicity', 'qm9_gap', 'qm9_homo', 'qm9_lumo']
DATA_DIR = project_root / 'data' / 'conformers'
RESULTS_DIR = project_root / 'results' / 'benchmark_fixed'
FIXED_DIM = 256

# Models to analyze (those that have checkpoints in benchmark_fixed)
MODELS_TO_ANALYZE = [
    'dko', 'dko_first_order', 'attention', 'mean_ensemble',
    'mfa', 'deepsets', 'single_conformer', 'boltzmann_ensemble',
]


def compute_scc_per_molecule(features_list, weights_list=None, fixed_dim=256):
    """Compute SCC (trace of covariance matrix) for each molecule."""
    scc_values = []

    for i in range(len(features_list)):
        mol_features = features_list[i]
        if len(mol_features) < 2:
            scc_values.append(0.0)
            continue

        # Pad/truncate
        fixed_features = []
        for conf_feat in mol_features:
            conf_feat = np.array(conf_feat).flatten()
            if len(conf_feat) >= fixed_dim:
                fixed_features.append(conf_feat[:fixed_dim])
            else:
                padded = np.zeros(fixed_dim)
                padded[:len(conf_feat)] = conf_feat
                fixed_features.append(padded)

        features = np.array(fixed_features)
        n_conf = features.shape[0]

        # Weights
        if weights_list is not None and weights_list[i] is not None:
            w = np.array(weights_list[i])
            w = w / w.sum()
        else:
            w = np.ones(n_conf) / n_conf

        # Weighted mean
        mu = np.sum(w[:, np.newaxis] * features, axis=0)

        # Weighted variance (sum of per-feature variances = trace of sigma)
        variances = np.sum(w[:, np.newaxis] * (features - mu[np.newaxis, :]) ** 2, axis=0)
        scc = variances.sum()
        scc_values.append(float(scc))

    return np.array(scc_values)


def find_best_checkpoint(dataset_name, model_name):
    """Find the best checkpoint for a model+dataset from benchmark results."""
    # Look for checkpoints in various locations
    search_paths = [
        RESULTS_DIR / dataset_name / 'checkpoints' / f'{model_name}_best.pt',
        RESULTS_DIR / f'{dataset_name}_{model_name}_seed42' / 'best_model.pt',
        RESULTS_DIR / 'checkpoints' / f'{dataset_name}_{model_name}_best.pt',
    ]

    for path in search_paths:
        if path.exists():
            return path

    return None


def stratify_by_scc(predictions, labels, scc_values, n_bins=4):
    """Stratify evaluation results by SCC quartiles."""
    # Create quartile bins
    bin_edges = np.percentile(scc_values, np.linspace(0, 100, n_bins + 1))
    bin_indices = np.digitize(scc_values, bin_edges[1:-1])

    results = {}
    for bin_idx in range(n_bins):
        mask = (bin_indices == bin_idx)
        if mask.sum() == 0:
            continue

        bin_preds = predictions[mask]
        bin_labels = labels[mask]
        bin_scc = scc_values[mask]

        # Compute RMSE and MAE
        rmse = np.sqrt(np.mean((bin_preds - bin_labels) ** 2))
        mae = np.mean(np.abs(bin_preds - bin_labels))

        results[f'Q{bin_idx + 1}'] = {
            'n_samples': int(mask.sum()),
            'scc_range': (float(bin_scc.min()), float(bin_scc.max())),
            'scc_mean': float(bin_scc.mean()),
            'rmse': float(rmse),
            'mae': float(mae),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="SCC Quartile Analysis (Experiment R)")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--n-bins", type=int, default=4, help="Number of SCC bins (4=quartiles)")
    args = parser.parse_args()

    datasets = args.datasets or DATASETS
    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR

    print("SCC Quartile Analysis — Experiment R")
    print(f"Device: {args.device}")
    print(f"Results dir: {results_dir}")
    print(f"Bins: {args.n_bins}")

    all_results = {}

    for dataset_name in datasets:
        dataset_path = DATA_DIR / dataset_name
        test_path = dataset_path / 'test.pkl'
        if not test_path.exists():
            print(f"\nSkipping {dataset_name}: test.pkl not found")
            continue

        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*70}")

        # Load test data
        with open(test_path, 'rb') as f:
            test_data = pickle.load(f)

        # Check task type
        unique_labels = set(test_data['labels'])
        is_classification = len(unique_labels) <= 2 and all(
            l in [0, 1, 0.0, 1.0] for l in unique_labels
        )
        if is_classification:
            print(f"  Skipping: classification task")
            continue

        # Compute SCC for test molecules
        features_list = test_data['features']
        weights_list = test_data.get('boltzmann_weights', [None] * len(features_list))
        scc_values = compute_scc_per_molecule(features_list, weights_list, FIXED_DIM)

        print(f"  Test molecules: {len(scc_values)}")
        print(f"  SCC range: [{scc_values.min():.4f}, {scc_values.max():.4f}]")
        print(f"  SCC mean: {scc_values.mean():.4f} +/- {scc_values.std():.4f}")

        # Quartile boundaries
        quartile_edges = np.percentile(scc_values, [0, 25, 50, 75, 100])
        print(f"  SCC quartiles: {[f'{q:.3f}' for q in quartile_edges]}")

        # Load benchmark results to get predictions
        benchmark_results_path = results_dir / 'benchmark_results.json'
        if benchmark_results_path.exists():
            with open(benchmark_results_path, 'r') as f:
                benchmark_data = json.load(f)
        else:
            print(f"  No benchmark_results.json found at {results_dir}")
            benchmark_data = {}

        dataset_results = {}

        # For each model, check if we have results
        for model_name in MODELS_TO_ANALYZE:
            if dataset_name in benchmark_data and model_name in benchmark_data[dataset_name]:
                seed_results = benchmark_data[dataset_name][model_name]
                # Get the test RMSE/MAE from first seed
                for sr in seed_results:
                    if isinstance(sr, dict) and 'test_metrics' in sr:
                        metrics = sr['test_metrics']
                        print(f"  {model_name}: RMSE={metrics.get('rmse', 'N/A'):.4f}, "
                              f"MAE={metrics.get('mae', 'N/A'):.4f}")
                        break

        all_results[dataset_name] = {
            'n_test': len(scc_values),
            'scc_stats': {
                'mean': float(scc_values.mean()),
                'std': float(scc_values.std()),
                'min': float(scc_values.min()),
                'max': float(scc_values.max()),
                'quartiles': quartile_edges.tolist(),
            },
        }

    # Summary
    print(f"\n{'='*70}")
    print("SCC QUARTILE ANALYSIS SUMMARY")
    print(f"{'='*70}")

    print(f"\n{'Dataset':<20} {'N_test':>8} {'SCC Mean':>10} {'SCC Std':>10} {'SCC Max':>10}")
    print(f"{'-'*20} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
    for name, r in all_results.items():
        s = r['scc_stats']
        print(f"{name:<20} {r['n_test']:>8} {s['mean']:>10.4f} {s['std']:>10.4f} {s['max']:>10.4f}")

    print(f"\nNote: Full per-quartile model evaluation requires loading model checkpoints.")
    print(f"Run the full benchmark with --models to generate per-quartile breakdowns.")

    # Save results
    output_path = project_root / 'results' / 'scc_quartile_analysis.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
