#!/usr/bin/env python3
"""
Hybrid Fingerprint + Conformer experiment.

Tests whether DKO second-order statistics add value on top of fingerprints.
Approach: Train XGBoost on [fingerprint | mu_stats | sigma_stats] features.
"""

import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import xgboost as xgb
except ImportError:
    print("ERROR: xgboost required")
    sys.exit(1)

from rdkit import Chem
from rdkit.Chem import AllChem


DATASETS = ["esol", "lipophilicity", "freesolv", "qm9_gap", "qm9_homo", "qm9_lumo"]
SEEDS = [42, 123, 456]


def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)


def compute_conformer_stats(features, max_dim=1024):
    """Compute mu (mean) and sigma statistics from conformer features."""
    # Pad/truncate all conformers to same dimension
    padded = []
    for conf_feat in features:
        if len(conf_feat) < max_dim:
            padded.append(np.pad(conf_feat, (0, max_dim - len(conf_feat))))
        else:
            padded.append(conf_feat[:max_dim])

    conformers = np.array(padded)  # (n_conf, D)

    if len(conformers) == 0:
        return np.zeros(max_dim), np.zeros(5)

    # First-order: mean
    mu = np.mean(conformers, axis=0)  # (D,)

    # Second-order: scalar invariants of covariance
    centered = conformers - mu
    cov = centered.T @ centered / max(len(conformers) - 1, 1)

    # Diagonal of covariance (variance per feature)
    diag = np.diag(cov)

    # Scalar invariants
    total_var = np.sum(diag)
    max_var = np.max(diag)
    mean_var = np.mean(diag)
    # Top-k eigenvalues via diagonal proxy
    sorted_diag = np.sort(diag)[::-1]
    top5_var = np.sum(sorted_diag[:5])
    effective_rank = np.sum(diag > 0.01 * total_var) if total_var > 0 else 0

    sigma_stats = np.array([total_var, max_var, mean_var, top5_var, effective_rank])

    return mu, sigma_stats


def load_and_featurize(dataset_name, split, feature_sets):
    """Load data and compute feature vectors."""
    data_path = Path(f"data/conformers/{dataset_name}/{split}.pkl")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    smiles_list = data["smiles"]
    labels = np.array([float(y) for y in data["labels"]])
    features_list = data["features"]

    all_features = []
    for i, (smi, mol_feats) in enumerate(zip(smiles_list, features_list)):
        feat_parts = []

        if "fp" in feature_sets:
            fp = smiles_to_fingerprint(smi)
            feat_parts.append(fp)

        if "mu" in feature_sets:
            mu, _ = compute_conformer_stats(mol_feats)
            feat_parts.append(mu)

        if "sigma" in feature_sets:
            _, sigma_stats = compute_conformer_stats(mol_feats)
            feat_parts.append(sigma_stats)

        all_features.append(np.concatenate(feat_parts))

    return np.array(all_features), labels


def run_experiment(dataset_name, seed, feature_sets, label):
    """Run single experiment with given feature combination."""
    np.random.seed(seed)

    train_X, train_y = load_and_featurize(dataset_name, "train", feature_sets)
    val_X, val_y = load_and_featurize(dataset_name, "val", feature_sets)
    test_X, test_y = load_and_featurize(dataset_name, "test", feature_sets)

    # Combine train + val
    full_X = np.vstack([train_X, val_X])
    full_y = np.concatenate([train_y, val_y])

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=4,
        verbosity=0,
        tree_method='hist',
    )

    model.fit(full_X, full_y)

    test_pred = model.predict(test_X)
    rmse = np.sqrt(mean_squared_error(test_y, test_pred))
    mae = mean_absolute_error(test_y, test_pred)
    r2 = r2_score(test_y, test_pred)

    return {
        "dataset": dataset_name,
        "features": label,
        "feature_dim": train_X.shape[1],
        "seed": seed,
        "test_rmse": float(rmse),
        "test_mae": float(mae),
        "test_r2": float(r2),
    }


def main():
    output_dir = Path("results/hybrid_experiment")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Feature combinations to test
    configs = [
        (["fp"], "FP only"),
        (["mu"], "Mu only"),
        (["sigma"], "Sigma only"),
        (["fp", "mu"], "FP + Mu"),
        (["fp", "sigma"], "FP + Sigma"),
        (["fp", "mu", "sigma"], "FP + Mu + Sigma"),
        (["mu", "sigma"], "Mu + Sigma"),
    ]

    print("Hybrid FP + Conformer Experiment")
    print("=" * 70)

    results = []

    for dataset in DATASETS:
        print(f"\n{dataset}:")
        for feature_sets, label in configs:
            rmses = []
            for seed in SEEDS:
                result = run_experiment(dataset, seed, feature_sets, label)
                results.append(result)
                rmses.append(result["test_rmse"])

            mean_rmse = np.mean(rmses)
            std_rmse = np.std(rmses)
            dim = results[-1]["feature_dim"]
            print(f"  {label:20s} (dim={dim:5d}): RMSE={mean_rmse:.4f} ± {std_rmse:.4f}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "raw_results": results,
    }
    with open(output_dir / "hybrid_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_dir / 'hybrid_results.json'}")


if __name__ == "__main__":
    main()
