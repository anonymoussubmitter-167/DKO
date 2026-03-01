#!/usr/bin/env python3
"""
Experiment 6: Per-Feature Attribution (XGBoost Importance).

Trains hybrid XGBoost (FP+mu+sigma) and extracts feature_importances_
(gain-based) to show which feature groups contribute most.

Reports:
1. Top-20 features with importance scores
2. Aggregate importance by feature group (FP vs mu vs sigma %)
3. Per-dataset comparison

Datasets: ESOL, FreeSolv, QM9-Gap
"""

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import xgboost as xgb
except ImportError:
    print("ERROR: xgboost required"); sys.exit(1)

from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator

DATASETS = ["esol", "freesolv", "qm9_gap"]
SEEDS = [42, 123, 456]

_FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# Feature group ranges
FP_DIM = 2048
MU_DIM = 256
SIGMA_DIM = 5
TOTAL_DIM = FP_DIM + MU_DIM + SIGMA_DIM  # 2309

SIGMA_NAMES = ["total_var", "max_var", "mean_var", "top5_var", "effective_rank"]


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


def precompute_features(dataset_name, split):
    """Precompute all feature types for a split."""
    path = Path(f"data/conformers/{dataset_name}/{split}.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)

    smiles_list = data["smiles"]
    labels = np.array([float(y) for y in data["labels"]])
    features_list = data["features"]

    n = len(smiles_list)
    fp = np.zeros((n, FP_DIM))
    mu = np.zeros((n, MU_DIM))
    sigma = np.zeros((n, SIGMA_DIM))

    for i, (smi, mol_feats) in enumerate(zip(smiles_list, features_list)):
        fp[i] = smiles_to_fingerprint(smi)
        mu[i], sigma[i] = compute_conformer_stats(mol_feats)

    return fp, mu, sigma, labels


def get_feature_name(idx):
    """Map feature index to human-readable name."""
    if idx < FP_DIM:
        return f"FP_bit_{idx}"
    elif idx < FP_DIM + MU_DIM:
        return f"mu_dim_{idx - FP_DIM}"
    else:
        sigma_idx = idx - FP_DIM - MU_DIM
        if sigma_idx < len(SIGMA_NAMES):
            return f"sigma_{SIGMA_NAMES[sigma_idx]}"
        return f"sigma_{sigma_idx}"


def get_feature_group(idx):
    """Get feature group for an index."""
    if idx < FP_DIM:
        return "FP"
    elif idx < FP_DIM + MU_DIM:
        return "mu"
    else:
        return "sigma"


def analyze_importances(importances):
    """Analyze feature importances by group."""
    fp_imp = np.sum(importances[:FP_DIM])
    mu_imp = np.sum(importances[FP_DIM:FP_DIM + MU_DIM])
    sigma_imp = np.sum(importances[FP_DIM + MU_DIM:])

    total = fp_imp + mu_imp + sigma_imp + 1e-10

    # Top-20 features
    top_indices = np.argsort(importances)[::-1][:20]
    top_features = []
    for idx in top_indices:
        top_features.append({
            "rank": len(top_features) + 1,
            "index": int(idx),
            "name": get_feature_name(idx),
            "group": get_feature_group(idx),
            "importance": float(importances[idx]),
        })

    # Per-sigma feature importances
    sigma_detail = {}
    for i, name in enumerate(SIGMA_NAMES):
        feat_idx = FP_DIM + MU_DIM + i
        if feat_idx < len(importances):
            sigma_detail[name] = float(importances[feat_idx])

    return {
        "group_importance": {
            "FP": float(fp_imp),
            "mu": float(mu_imp),
            "sigma": float(sigma_imp),
        },
        "group_importance_pct": {
            "FP": float(fp_imp / total * 100),
            "mu": float(mu_imp / total * 100),
            "sigma": float(sigma_imp / total * 100),
        },
        "top_20_features": top_features,
        "sigma_detail": sigma_detail,
        "n_nonzero_features": int(np.sum(importances > 0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--output-dir", default="results/feature_attribution")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Feature Attribution Analysis (XGBoost Importance)")
    print(f"Datasets: {args.datasets}")
    print(f"Seeds: {args.seeds}")
    print("=" * 70)

    all_results = {}

    for dataset in args.datasets:
        print(f"\n=== {dataset.upper()} ===")
        try:
            # Load data
            train_fp, train_mu, train_sigma, train_y = precompute_features(dataset, "train")
            val_fp, val_mu, val_sigma, val_y = precompute_features(dataset, "val")
            test_fp, test_mu, test_sigma, test_y = precompute_features(dataset, "test")

            # Combine train + val
            trainval_X = np.vstack([
                np.hstack([train_fp, train_mu, train_sigma]),
                np.hstack([val_fp, val_mu, val_sigma]),
            ])
            trainval_y = np.concatenate([train_y, val_y])
            test_X = np.hstack([test_fp, test_mu, test_sigma])

            # Average importances across seeds
            avg_importances = np.zeros(TOTAL_DIM)
            rmses = []

            for seed in args.seeds:
                model = xgb.XGBRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8, random_state=seed,
                    n_jobs=4, verbosity=0, tree_method='hist',
                )
                model.fit(trainval_X, trainval_y)

                pred = model.predict(test_X)
                rmse = float(np.sqrt(mean_squared_error(test_y, pred)))
                rmses.append(rmse)

                avg_importances += model.feature_importances_

            avg_importances /= len(args.seeds)

            analysis = analyze_importances(avg_importances)
            analysis["model_rmse_mean"] = float(np.mean(rmses))
            analysis["model_rmse_std"] = float(np.std(rmses))

            all_results[dataset] = analysis

            # Print results
            pct = analysis["group_importance_pct"]
            print(f"\n  Group Importance (%):")
            print(f"    FP:    {pct['FP']:.1f}%")
            print(f"    mu:    {pct['mu']:.1f}%")
            print(f"    sigma: {pct['sigma']:.1f}%")

            print(f"\n  Sigma feature detail:")
            for name, imp in analysis["sigma_detail"].items():
                print(f"    {name:<20}: {imp:.6f}")

            print(f"\n  Top-10 features:")
            for feat in analysis["top_20_features"][:10]:
                print(f"    #{feat['rank']}: {feat['name']:<25} ({feat['group']}) "
                      f"imp={feat['importance']:.6f}")

            print(f"\n  Model RMSE: {analysis['model_rmse_mean']:.4f}+/-{analysis['model_rmse_std']:.4f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Cross-dataset summary
    print("\n" + "=" * 70)
    print("CROSS-DATASET SUMMARY: Feature Group Contribution (%)")
    print("=" * 70)
    print(f"{'Dataset':<15} {'FP %':<10} {'mu %':<10} {'sigma %':<10} {'RMSE':<15}")
    print("-" * 60)

    for dataset in args.datasets:
        if dataset in all_results:
            r = all_results[dataset]
            pct = r["group_importance_pct"]
            print(f"{dataset:<15} {pct['FP']:<10.1f} {pct['mu']:<10.1f} {pct['sigma']:<10.1f} "
                  f"{r['model_rmse_mean']:.4f}")

    # Save
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "datasets": args.datasets,
            "seeds": args.seeds,
            "feature_groups": {
                "FP": f"bits 0-{FP_DIM-1} (Morgan fingerprint)",
                "mu": f"dims {FP_DIM}-{FP_DIM+MU_DIM-1} (conformer mean)",
                "sigma": f"dims {FP_DIM+MU_DIM}-{TOTAL_DIM-1} ({', '.join(SIGMA_NAMES)})",
            },
        },
        "results": all_results,
    }
    with open(output_dir / "feature_attribution_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_dir}/feature_attribution_results.json")


if __name__ == "__main__":
    main()
