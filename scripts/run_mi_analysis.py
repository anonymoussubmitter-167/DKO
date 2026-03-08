#!/usr/bin/env python3
"""
Experiment 5: Mutual Information Quantification.

Computes MI between different feature sets and targets to show that
sigma provides additional information beyond FP+mu.

Feature sets analyzed:
- FP (Morgan fingerprints, 2048-bit)
- mu (conformer mean, 256-dim)
- sigma (5 scalar invariants)
- FP + mu
- FP + mu + sigma

Also computes conditional MI: MI(sigma; y | FP, mu) by measuring
the residual information sigma adds beyond FP+mu.

Datasets: ESOL, FreeSolv, Lipophilicity, QM9-Gap
"""

import argparse
import json
import pickle
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.feature_selection import mutual_info_regression

sys.path.insert(0, str(Path(__file__).parent.parent))

from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator

DATASETS = ["esol", "freesolv", "lipophilicity", "qm9_gap"]

_FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(2048)
    return _FP_GEN.GetFingerprintAsNumPy(mol).astype(np.float64)


def compute_conformer_stats(features, max_dim=256):
    """Compute mu and sigma stats from conformer features.

    Uses fast variance-based invariants (O(n*d)) instead of
    eigendecomposition (O(d^3)) for tractability.
    """
    padded = []
    for conf_feat in features:
        cf = np.array(conf_feat).flatten()
        if len(cf) < max_dim:
            cf = np.pad(cf, (0, max_dim - len(cf)))
        else:
            cf = cf[:max_dim]
        padded.append(cf)

    conformers = np.array(padded)
    if len(conformers) == 0:
        return np.zeros(max_dim), np.zeros(5)

    mu = np.mean(conformers, axis=0)

    if len(conformers) < 2:
        return mu, np.zeros(5)

    centered = conformers - mu
    variances = np.mean(centered ** 2, axis=0)

    total_var = np.sum(variances)
    max_var = np.max(variances)
    mean_var = np.mean(variances)
    sorted_var = np.sort(variances)[::-1]
    top5_var = np.sum(sorted_var[:5])
    effective_rank = float(np.sum(variances > 0.01 * total_var)) if total_var > 0 else 0

    return mu, np.array([total_var, max_var, mean_var, top5_var, effective_rank])


def load_features(dataset_name, splits=("train", "val")):
    """Load and precompute all features for a dataset (train+val combined)."""
    all_fp = []
    all_mu = []
    all_sigma = []
    all_labels = []

    for split in splits:
        path = Path(f"data/conformers/{dataset_name}/{split}.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)

        smiles_list = data["smiles"]
        labels = [float(y) for y in data["labels"]]
        features_list = data["features"]

        for smi, mol_feats, y in zip(smiles_list, features_list, labels):
            all_fp.append(smiles_to_fingerprint(smi))
            mu, sigma = compute_conformer_stats(mol_feats)
            all_mu.append(mu)
            all_sigma.append(sigma)
            all_labels.append(y)

    return {
        "fp": np.array(all_fp),
        "mu": np.array(all_mu),
        "sigma": np.array(all_sigma),
        "labels": np.array(all_labels),
    }


def reduce_features(features, max_dim=50):
    """Reduce high-dim features by selecting most variable columns.

    Faster than PCA for MI estimation: selects the max_dim columns
    with highest variance. For fingerprints, this selects the most
    informative bits.
    """
    if features.shape[1] <= max_dim:
        return features
    # Select columns with highest variance
    variances = np.var(features, axis=0)
    top_indices = np.argsort(variances)[::-1][:max_dim]
    return features[:, top_indices]


def compute_mi(features, targets, n_neighbors=5, n_runs=3, max_dim=50):
    """Compute MI with dimensionality reduction for high-dim features.

    Selects the max_dim most variable features for tractable MI estimation.
    This gives a lower bound on total MI (we only measure top features).
    """
    n_samples = min(len(targets), 1000)
    features_reduced = reduce_features(features, max_dim)
    actual_dim = features_reduced.shape[1]

    total_mis = []
    for run in range(n_runs):
        idx = np.random.RandomState(run).choice(len(targets), n_samples, replace=False)
        mi = mutual_info_regression(
            features_reduced[idx], targets[idx],
            n_neighbors=n_neighbors, random_state=run,
        )
        total_mis.append(np.sum(mi))

    return {
        "total_mi_mean": float(np.mean(total_mis)),
        "total_mi_std": float(np.std(total_mis)),
        "mean_per_feature": float(np.mean(total_mis) / actual_dim),
        "n_features_original": int(features.shape[1]),
        "n_features_reduced": int(actual_dim),
    }


def compute_conditional_mi(fp, mu, sigma, targets, n_neighbors=5, max_dim=50):
    """Compute conditional MI: MI(sigma; y | FP, mu).

    Approximated as: MI(FP+mu+sigma; y) - MI(FP+mu; y)
    This measures the additional information sigma provides beyond FP+mu.
    """
    fp_mu = np.hstack([fp, mu])
    fp_mu_sigma = np.hstack([fp, mu, sigma])

    mi_fp_mu = compute_mi(fp_mu, targets, n_neighbors, max_dim=max_dim)
    mi_fp_mu_sigma = compute_mi(fp_mu_sigma, targets, n_neighbors, max_dim=max_dim)

    conditional_mi = mi_fp_mu_sigma["total_mi_mean"] - mi_fp_mu["total_mi_mean"]

    return {
        "mi_fp_mu": mi_fp_mu["total_mi_mean"],
        "mi_fp_mu_sigma": mi_fp_mu_sigma["total_mi_mean"],
        "conditional_mi_sigma": float(conditional_mi),
        "relative_gain_pct": float(conditional_mi / (mi_fp_mu["total_mi_mean"] + 1e-10) * 100),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--output-dir", default="results/mi_analysis")
    parser.add_argument("--n-neighbors", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Mutual Information Analysis")
    print(f"Datasets: {args.datasets}")
    print("=" * 70)

    all_results = {}

    feature_set_configs = [
        ("fp", lambda d: d["fp"], "FP (2048-bit Morgan)"),
        ("mu", lambda d: d["mu"], "Mu (256-dim conformer mean)"),
        ("sigma", lambda d: d["sigma"], "Sigma (5 scalar invariants)"),
        ("fp_mu", lambda d: np.hstack([d["fp"], d["mu"]]), "FP + Mu"),
        ("fp_sigma", lambda d: np.hstack([d["fp"], d["sigma"]]), "FP + Sigma"),
        ("mu_sigma", lambda d: np.hstack([d["mu"], d["sigma"]]), "Mu + Sigma"),
        ("fp_mu_sigma", lambda d: np.hstack([d["fp"], d["mu"], d["sigma"]]), "FP + Mu + Sigma"),
    ]

    for dataset in args.datasets:
        print(f"\n=== {dataset.upper()} ===")
        try:
            data = load_features(dataset)
            targets = data["labels"]

            dataset_results = {}

            # Compute MI for each feature set
            print(f"\n  {'Feature Set':<30} {'Total MI':<15} {'MI/feat':<12} {'Orig Dim':<10} {'PCA Dim':<8}")
            print("  " + "-" * 75)

            for key, feat_fn, label in feature_set_configs:
                features = feat_fn(data)
                mi = compute_mi(features, targets, args.n_neighbors)
                dataset_results[key] = {
                    "label": label,
                    **mi,
                }
                print(f"  {label:<30} {mi['total_mi_mean']:.4f}+/-{mi['total_mi_std']:.4f}  "
                      f"{mi['mean_per_feature']:.6f}    {mi['n_features_original']:<10} "
                      f"{mi['n_features_reduced']}")

            # Conditional MI
            cond_mi = compute_conditional_mi(
                data["fp"], data["mu"], data["sigma"], targets, args.n_neighbors
            )
            dataset_results["conditional_mi"] = cond_mi

            print(f"\n  Conditional MI Analysis:")
            print(f"    MI(FP+mu; y)       = {cond_mi['mi_fp_mu']:.4f}")
            print(f"    MI(FP+mu+sigma; y) = {cond_mi['mi_fp_mu_sigma']:.4f}")
            print(f"    MI(sigma; y | FP,mu) = {cond_mi['conditional_mi_sigma']:.4f} "
                  f"({cond_mi['relative_gain_pct']:.1f}% relative gain)")

            all_results[dataset] = dataset_results

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Total MI by Feature Set")
    print("=" * 70)

    header = f"{'Feature Set':<25}"
    for ds in args.datasets:
        header += f" {ds:<15}"
    print(header)
    print("-" * (25 + 15 * len(args.datasets)))

    for key, _, label in feature_set_configs:
        row = f"{label:<25}"
        for ds in args.datasets:
            if ds in all_results and key in all_results[ds]:
                mi = all_results[ds][key]["total_mi_mean"]
                row += f" {mi:<15.4f}"
            else:
                row += f" {'N/A':<15}"
        print(row)

    print(f"\n{'Conditional MI(sigma|FP,mu)':<25}", end="")
    for ds in args.datasets:
        if ds in all_results and "conditional_mi" in all_results[ds]:
            cmi = all_results[ds]["conditional_mi"]["conditional_mi_sigma"]
            pct = all_results[ds]["conditional_mi"]["relative_gain_pct"]
            print(f" {cmi:.4f}({pct:+.1f}%)", end="  ")
        else:
            print(f" {'N/A':<15}", end="")
    print()

    # Save
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "datasets": args.datasets,
            "n_neighbors": args.n_neighbors,
        },
        "results": all_results,
    }
    with open(output_dir / "mi_analysis_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_dir}/mi_analysis_results.json")


if __name__ == "__main__":
    main()
