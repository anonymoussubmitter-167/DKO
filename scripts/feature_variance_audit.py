#!/usr/bin/env python
"""
Experiment G — Feature Variance Audit.

Loads precomputed conformer data for all datasets, computes covariance matrices
using the same logic as trainer._compute_mu_sigma, analyzes the eigenvalue
spectrum via diagonal proxy (matching the model's actual approach for D>256),
and validates against true eigenvalues on a subsample.

Reports:
  - Eigenvalue spectrum (mean/std across molecules)
  - Effective rank (number of eigenvalues needed for 95% variance)
  - Cumulative variance in top-k eigenvalues
  - Diagonal proxy vs true eigvalsh comparison

This informs the choice of k for DKO variants A/C/D/E/S.

Usage:
    python scripts/feature_variance_audit.py
    python scripts/feature_variance_audit.py --datasets esol qm9_gap
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


DATASETS = ['esol', 'freesolv', 'lipophilicity', 'qm9_gap', 'qm9_homo', 'qm9_lumo', 'bace', 'bbbp']
DATA_DIR = project_root / 'data' / 'conformers'
MAX_MOLECULES = 2000  # Subsample large datasets for speed
MAX_FEATURE_DIM = 1024  # Match dataloader's max_feature_dim


def _pad_truncate(feat, dim):
    """Pad or truncate a feature vector to a fixed dimension."""
    feat = np.array(feat).flatten()
    if len(feat) >= dim:
        return feat[:dim]
    padded = np.zeros(dim)
    padded[:len(feat)] = feat
    return padded


def compute_sigma_diags(features_list, weights_list=None, max_mol=MAX_MOLECULES,
                        feature_dim=MAX_FEATURE_DIM):
    """Compute diagonal of sigma for each molecule (fast O(D) per molecule).

    Returns sorted diagonal values in descending order — this matches
    the proxy used by DKO variants when D > 256.
    """
    n_total = len(features_list)
    indices = np.arange(n_total)
    if n_total > max_mol:
        rng = np.random.RandomState(42)
        indices = rng.choice(n_total, max_mol, replace=False)

    all_diags = []
    D = feature_dim

    for idx in indices:
        mol_features = features_list[idx]
        if len(mol_features) < 2:
            continue

        # Pad/truncate conformer features to fixed dimension (matches dataloader)
        features = np.array([_pad_truncate(f, D) for f in mol_features])

        n_conf = features.shape[0]

        # Per-sample normalization across conformers
        feat_mean = features.mean(axis=0, keepdims=True)
        feat_std = features.std(axis=0, keepdims=True)
        feat_std = np.maximum(feat_std, 1e-6)
        features = (features - feat_mean) / feat_std

        # Weights
        if weights_list is not None and idx < len(weights_list) and weights_list[idx] is not None:
            w = np.array(weights_list[idx])
            if len(w) == n_conf:
                w = w / w.sum()
            else:
                w = np.ones(n_conf) / n_conf
        else:
            w = np.ones(n_conf) / n_conf

        # Weighted mean
        mu = np.sum(w[:, np.newaxis] * features, axis=0)

        # Diagonal of weighted covariance (fast — no full matrix needed)
        centered = features - mu[np.newaxis, :]
        centered = np.clip(centered, -10.0, 10.0)
        weighted_sq = (centered ** 2) * w[:, np.newaxis]
        diag = weighted_sq.sum(axis=0)  # (D,)

        # Regularization
        diag += 1e-2

        # Sort descending
        diag_sorted = np.sort(diag)[::-1]
        all_diags.append(diag_sorted)

    return np.array(all_diags), D


def compute_true_eigenvalues(features_list, weights_list=None, n_sample=100,
                             feature_dim=MAX_FEATURE_DIM):
    """Compute true eigenvalues for a small subsample to validate the diagonal proxy."""
    n_total = len(features_list)
    rng = np.random.RandomState(123)

    # Only consider molecules with >= 2 conformers
    valid_indices = [i for i in range(n_total) if len(features_list[i]) >= 2]
    sample_indices = rng.choice(valid_indices, min(n_sample, len(valid_indices)), replace=False)

    all_eigvals = []
    all_diags = []
    D = feature_dim

    for idx in sample_indices:
        mol_features = features_list[idx]

        # Pad/truncate to fixed dim
        features = np.array([_pad_truncate(f, D) for f in mol_features])
        n_conf = features.shape[0]

        # Per-sample normalization
        feat_mean = features.mean(axis=0, keepdims=True)
        feat_std = features.std(axis=0, keepdims=True)
        feat_std = np.maximum(feat_std, 1e-6)
        features = (features - feat_mean) / feat_std

        # Weights
        if weights_list is not None and idx < len(weights_list) and weights_list[idx] is not None:
            w = np.array(weights_list[idx])
            if len(w) == n_conf:
                w = w / w.sum()
            else:
                w = np.ones(n_conf) / n_conf
        else:
            w = np.ones(n_conf) / n_conf

        mu = np.sum(w[:, np.newaxis] * features, axis=0)
        centered = features - mu[np.newaxis, :]
        centered = np.clip(centered, -10.0, 10.0)
        weighted_centered = centered * np.sqrt(w[:, np.newaxis])
        sigma = weighted_centered.T @ weighted_centered
        sigma += 1e-2 * np.eye(D)

        eigvals = np.linalg.eigvalsh(sigma)
        eigvals = np.sort(eigvals)[::-1]
        all_eigvals.append(eigvals)

        diag_sorted = np.sort(np.diag(sigma))[::-1]
        all_diags.append(diag_sorted)

    return np.array(all_eigvals), np.array(all_diags)


def analyze_spectrum(values, dataset_name, n_molecules, feature_dim, label="Diagonal proxy"):
    """Analyze eigenvalue/diagonal spectrum."""
    D = values.shape[1]

    mean_vals = values.mean(axis=0)
    std_vals = values.std(axis=0)

    total_var = mean_vals.sum()

    print(f"\n  {label} — Feature dimension: {D}")
    print(f"  Top-20 values (mean +/- std across {len(values)} molecules):")
    print(f"    {'k':>4}  {'Mean':>12}  {'Std':>12}  {'Cumulative %':>14}")
    print(f"    {'-'*4}  {'-'*12}  {'-'*12}  {'-'*14}")

    cumulative = 0.0
    for k in range(min(20, D)):
        cumulative += mean_vals[k]
        pct = 100.0 * cumulative / total_var if total_var > 0 else 0.0
        print(f"    {k+1:>4}  {mean_vals[k]:>12.4f}  {std_vals[k]:>12.4f}  {pct:>13.1f}%")

    # Effective rank at various thresholds
    print(f"\n  Effective rank (values needed for X% of total variance):")
    rank_results = {}
    for threshold in [0.80, 0.90, 0.95, 0.99]:
        ranks = []
        for vals in values:
            total = vals.sum()
            if total <= 0:
                ranks.append(D)
                continue
            cumsum = np.cumsum(vals) / total
            rank = np.searchsorted(cumsum, threshold) + 1
            ranks.append(min(rank, D))
        ranks = np.array(ranks)
        rank_results[f"{threshold*100:.0f}%"] = {
            'mean': float(ranks.mean()),
            'median': float(np.median(ranks)),
            'p25': float(np.percentile(ranks, 25)),
            'p75': float(np.percentile(ranks, 75)),
        }
        print(f"    {threshold*100:.0f}%: mean={ranks.mean():.1f}, "
              f"median={np.median(ranks):.0f}, "
              f"p25={np.percentile(ranks, 25):.0f}, "
              f"p75={np.percentile(ranks, 75):.0f}")

    # Cumulative variance for specific k values
    print(f"\n  Cumulative variance captured by top-k values:")
    cumvar_results = {}
    for k in [3, 5, 10, 15, 20, 30, 50, 100]:
        if k > D:
            break
        pcts = []
        for vals in values:
            total = vals.sum()
            if total > 0:
                pcts.append(100.0 * vals[:k].sum() / total)
            else:
                pcts.append(0.0)
        pcts = np.array(pcts)
        cumvar_results[f"k={k}"] = {
            'mean_pct': float(pcts.mean()),
            'std_pct': float(pcts.std()),
            'min_pct': float(pcts.min()),
            'max_pct': float(pcts.max()),
        }
        print(f"    k={k:>3}: {pcts.mean():.1f}% +/- {pcts.std():.1f}% "
              f"(min={pcts.min():.1f}%, max={pcts.max():.1f}%)")

    return {
        'n_molecules_analyzed': int(len(values)),
        'feature_dim': int(D),
        'total_variance': float(total_var),
        'mean_top20': mean_vals[:20].tolist(),
        'effective_rank': rank_results,
        'cumulative_variance': cumvar_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Feature Variance Audit (Experiment G)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Datasets to audit (default: all)")
    parser.add_argument("--max-molecules", type=int, default=MAX_MOLECULES,
                        help="Max molecules per dataset (subsample for speed)")
    parser.add_argument("--eigval-sample", type=int, default=50,
                        help="Molecules for true eigenvalue validation")
    args = parser.parse_args()

    datasets = args.datasets or DATASETS
    results = {}

    print("Feature Variance Audit — Experiment G")
    print(f"Data directory: {DATA_DIR}")
    print(f"Max molecules per dataset: {args.max_molecules}")
    print(f"True eigenvalue sample size: {args.eigval_sample}")

    for dataset_name in datasets:
        dataset_path = DATA_DIR / dataset_name / 'train.pkl'
        if not dataset_path.exists():
            print(f"\nSkipping {dataset_name}: {dataset_path} not found")
            continue

        t0 = time.time()
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*70}")

        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)

        features_list = data['features']
        weights_list = data.get('boltzmann_weights', [None] * len(features_list))
        n_total = len(features_list)
        print(f"  Total molecules: {n_total}")

        # Fast diagonal analysis (all/subsampled molecules)
        print(f"  Computing diagonal proxy...")
        diag_values, feature_dim = compute_sigma_diags(
            features_list, weights_list, max_mol=args.max_molecules
        )
        if len(diag_values) == 0:
            print(f"  Skipping: no valid molecules")
            continue

        diag_result = analyze_spectrum(
            diag_values, dataset_name, n_total, feature_dim,
            label=f"Diagonal proxy (n={len(diag_values)})"
        )

        dataset_result = {
            'n_molecules_total': n_total,
            'feature_dim': feature_dim,
            'diagonal_proxy': diag_result,
        }

        # True eigenvalue validation on small subsample
        if args.eigval_sample > 0:
            print(f"\n  Computing true eigenvalues for {args.eigval_sample} molecules...")
            try:
                true_eigvals, true_diags = compute_true_eigenvalues(
                    features_list, weights_list, n_sample=args.eigval_sample
                )
                if len(true_eigvals) > 0:
                    eigval_result = analyze_spectrum(
                        true_eigvals, dataset_name, n_total, feature_dim,
                        label=f"True eigenvalues (n={len(true_eigvals)})"
                    )
                    dataset_result['true_eigenvalues'] = eigval_result

                    # Compare diagonal proxy vs true eigenvalues
                    diag_cumvar = true_diags.cumsum(axis=1) / true_diags.sum(axis=1, keepdims=True)
                    eig_cumvar = true_eigvals.cumsum(axis=1) / true_eigvals.sum(axis=1, keepdims=True)
                    for k in [5, 10, 20]:
                        if k <= feature_dim:
                            diag_pct = diag_cumvar[:, k-1].mean() * 100
                            eig_pct = eig_cumvar[:, k-1].mean() * 100
                            print(f"\n    Proxy validation (k={k}): "
                                  f"diag={diag_pct:.1f}%, eig={eig_pct:.1f}%, "
                                  f"gap={abs(diag_pct-eig_pct):.1f}pp")
            except Exception as e:
                print(f"  True eigenvalue computation failed: {e}")

        elapsed = time.time() - t0
        print(f"\n  Done in {elapsed:.1f}s")
        results[dataset_name] = dataset_result

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY: Recommended k for each dataset")
    print(f"{'='*70}")
    print(f"  {'Dataset':<20} {'N_mol':>8} {'D':>6} {'k(90%)':>8} {'k(95%)':>8} {'k(99%)':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")

    for name, r in results.items():
        proxy = r['diagonal_proxy']
        eff = proxy.get('effective_rank', {})
        k90 = eff.get('90%', {}).get('median', '?')
        k95 = eff.get('95%', {}).get('median', '?')
        k99 = eff.get('99%', {}).get('median', '?')
        print(f"  {name:<20} {r['n_molecules_total']:>8} {r['feature_dim']:>6} "
              f"{k90:>8} {k95:>8} {k99:>8}")

    print(f"\nRecommendation: Use k=10 for most variants (A/D/E/S), k=5 for low-rank (C)")

    # Save results
    output_path = project_root / 'results' / 'feature_variance_audit.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
