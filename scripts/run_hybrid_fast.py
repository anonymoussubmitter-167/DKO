#!/usr/bin/env python3
"""
Fast hybrid FP + Conformer experiment.
Precomputes all features once, then runs XGBoost with different combinations.
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
    print("ERROR: xgboost required"); sys.exit(1)

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


def compute_conformer_stats(features, max_dim=256):
    """Compute mu and sigma stats. Use smaller max_dim for speed."""
    padded = []
    for conf_feat in features:
        if len(conf_feat) < max_dim:
            padded.append(np.pad(conf_feat, (0, max_dim - len(conf_feat))))
        else:
            padded.append(conf_feat[:max_dim])

    conformers = np.array(padded)
    if len(conformers) == 0:
        return np.zeros(max_dim), np.zeros(5)

    mu = np.mean(conformers, axis=0)

    # Second-order: scalar invariants (fast, no full covariance)
    centered = conformers - mu
    variances = np.mean(centered ** 2, axis=0)  # per-feature variance

    total_var = np.sum(variances)
    max_var = np.max(variances)
    mean_var = np.mean(variances)
    sorted_var = np.sort(variances)[::-1]
    top5_var = np.sum(sorted_var[:5])
    effective_rank = float(np.sum(variances > 0.01 * total_var)) if total_var > 0 else 0

    sigma_stats = np.array([total_var, max_var, mean_var, top5_var, effective_rank])
    return mu, sigma_stats


def precompute_features(dataset_name, split):
    """Precompute all feature types for a split."""
    path = Path(f"data/conformers/{dataset_name}/{split}.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)

    smiles_list = data["smiles"]
    labels = np.array([float(y) for y in data["labels"]])
    features_list = data["features"]

    n = len(smiles_list)
    fp_features = np.zeros((n, 2048))
    mu_features = np.zeros((n, 256))
    sigma_features = np.zeros((n, 5))

    for i, (smi, mol_feats) in enumerate(zip(smiles_list, features_list)):
        fp_features[i] = smiles_to_fingerprint(smi)
        mu, sigma = compute_conformer_stats(mol_feats, max_dim=256)
        mu_features[i] = mu
        sigma_features[i] = sigma

    return {
        "labels": labels,
        "fp": fp_features,
        "mu": mu_features,
        "sigma": sigma_features,
    }


def run_xgb(train_X, train_y, test_X, test_y, seed):
    model = xgb.XGBRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=seed,
        n_jobs=4, verbosity=0, tree_method='hist',
    )
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    rmse = float(np.sqrt(mean_squared_error(test_y, pred)))
    mae = float(mean_absolute_error(test_y, pred))
    r2 = float(r2_score(test_y, pred))
    return rmse, mae, r2


def main():
    output_dir = Path("results/hybrid_experiment")
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        (["fp"], "FP only"),
        (["mu"], "Mu only"),
        (["sigma"], "Sigma only"),
        (["fp", "mu"], "FP + Mu"),
        (["fp", "sigma"], "FP + Sigma"),
        (["fp", "mu", "sigma"], "FP + Mu + Sigma"),
        (["mu", "sigma"], "Mu + Sigma"),
    ]

    print("Fast Hybrid FP + Conformer Experiment")
    print("=" * 70)

    all_results = []

    for dataset in DATASETS:
        print(f"\nPrecomputing features for {dataset}...")
        train_data = precompute_features(dataset, "train")
        val_data = precompute_features(dataset, "val")
        test_data = precompute_features(dataset, "test")

        # Combine train + val
        combined = {}
        for key in ["fp", "mu", "sigma"]:
            combined[key] = np.vstack([train_data[key], val_data[key]])
        combined_y = np.concatenate([train_data["labels"], val_data["labels"]])
        test_y = test_data["labels"]

        print(f"\n{dataset}:")
        for feat_keys, label in configs:
            rmses, maes, r2s = [], [], []
            for seed in SEEDS:
                train_X = np.hstack([combined[k] for k in feat_keys])
                test_X = np.hstack([test_data[k] for k in feat_keys])

                rmse, mae, r2 = run_xgb(train_X, combined_y, test_X, test_y, seed)
                rmses.append(rmse)
                maes.append(mae)
                r2s.append(r2)

                all_results.append({
                    "dataset": dataset, "features": label, "seed": seed,
                    "rmse": rmse, "mae": mae, "r2": r2,
                    "feature_dim": train_X.shape[1],
                })

            dim = all_results[-1]["feature_dim"]
            print(f"  {label:20s} (dim={dim:5d}): "
                  f"RMSE={np.mean(rmses):.4f}±{np.std(rmses):.4f}  "
                  f"MAE={np.mean(maes):.4f}±{np.std(maes):.4f}")

    output = {
        "timestamp": datetime.now().isoformat(),
        "raw_results": all_results,
    }
    with open(output_dir / "hybrid_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_dir / 'hybrid_results.json'}")


if __name__ == "__main__":
    main()
