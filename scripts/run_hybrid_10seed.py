#!/usr/bin/env python3
"""
Run 10-seed validation for hybrid FP+conformer features.
Addresses Critique #4: Hybrid as centerpiece.
"""

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    import xgboost as xgb
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


DATASETS = ["esol", "freesolv", "lipophilicity", "qm9_gap", "qm9_homo", "qm9_lumo"]
SEEDS = [42, 123, 456, 789, 1000, 1111, 2222, 3333, 4444, 5555]
FEATURE_SETS = ["fp_only", "mu_only", "sigma_only", "fp_mu", "fp_sigma", "fp_mu_sigma", "mu_sigma"]


def load_data(dataset_name: str, split: str = "train", max_dim: int = 256):
    """Load dataset with conformer features."""
    data_path = Path(f"data/conformers/{dataset_name}/{split}.pkl")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    smiles_list = data["smiles"]
    raw_features = data["features"]
    weights = data.get("boltzmann_weights", None)
    targets = np.array(data["labels"]).squeeze()

    # Compute mu (mean) and sigma stats for each molecule
    mu_list = []
    sigma_stats_list = []

    for i, mol_feats in enumerate(raw_features):
        # Pad/truncate conformer features
        mol_feats_padded = []
        for conf_feat in mol_feats:
            conf_feat = np.array(conf_feat).flatten()
            if len(conf_feat) > max_dim:
                conf_feat = conf_feat[:max_dim]
            elif len(conf_feat) < max_dim:
                conf_feat = np.pad(conf_feat, (0, max_dim - len(conf_feat)))
            mol_feats_padded.append(conf_feat)

        mol_feats_arr = np.array(mol_feats_padded)

        # Compute weighted mean (mu)
        if weights is not None and len(weights[i]) == len(mol_feats):
            w = np.array(weights[i])
            w = w / (w.sum() + 1e-10)
            mu = np.average(mol_feats_arr, axis=0, weights=w)
            # Weighted covariance
            centered = mol_feats_arr - mu
            sigma = np.einsum('i,ij,ik->jk', w, centered, centered)
        else:
            mu = mol_feats_arr.mean(axis=0)
            sigma = np.cov(mol_feats_arr.T) if len(mol_feats_arr) > 1 else np.zeros((max_dim, max_dim))

        mu_list.append(mu)

        # Extract sigma statistics (5 scalar invariants)
        try:
            trace = np.trace(sigma)
            frob = np.linalg.norm(sigma, 'fro')
            eigvals = np.linalg.eigvalsh(sigma)
            eigvals = np.sort(eigvals)[::-1]  # Descending
            log_det = np.sum(np.log(np.abs(eigvals) + 1e-10))
            lambda_ratio = eigvals[0] / (np.sum(eigvals) + 1e-10)
            spectral_ratio = eigvals[0] / (eigvals[1] + 1e-10) if len(eigvals) > 1 else 1.0
            sigma_stats = np.array([trace, log_det, frob, lambda_ratio, spectral_ratio])
        except Exception:
            sigma_stats = np.zeros(5)

        sigma_stats_list.append(sigma_stats)

    mu_arr = np.array(mu_list)
    sigma_arr = np.array(sigma_stats_list)

    # Compute fingerprints
    fps = []
    if RDKIT_AVAILABLE:
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                fps.append(np.array(fp))
            else:
                fps.append(np.zeros(2048))
    else:
        fps = [np.zeros(2048) for _ in smiles_list]

    fp_arr = np.array(fps)

    return {
        "smiles": smiles_list,
        "fp": fp_arr,
        "mu": mu_arr,
        "sigma": sigma_arr,
        "targets": targets,
    }


def get_features(data: dict, feature_set: str) -> np.ndarray:
    """Get feature matrix for given feature set."""
    if feature_set == "fp_only":
        return data["fp"]
    elif feature_set == "mu_only":
        return data["mu"]
    elif feature_set == "sigma_only":
        return data["sigma"]
    elif feature_set == "fp_mu":
        return np.hstack([data["fp"], data["mu"]])
    elif feature_set == "fp_sigma":
        return np.hstack([data["fp"], data["sigma"]])
    elif feature_set == "fp_mu_sigma":
        return np.hstack([data["fp"], data["mu"], data["sigma"]])
    elif feature_set == "mu_sigma":
        return np.hstack([data["mu"], data["sigma"]])
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")


def run_experiment(dataset: str, feature_set: str, seed: int) -> dict:
    """Run single XGBoost experiment."""
    # Load data
    train_data = load_data(dataset, "train")
    val_data = load_data(dataset, "val")
    test_data = load_data(dataset, "test")

    # Get features
    X_train = get_features(train_data, feature_set)
    X_val = get_features(val_data, feature_set)
    X_test = get_features(test_data, feature_set)

    y_train = train_data["targets"]
    y_val = val_data["targets"]
    y_test = test_data["targets"]

    # Combine train + val
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    # Train XGBoost with GPU
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=seed,
        tree_method="gpu_hist",
        device="cuda:8",
    )
    model.fit(X_trainval, y_trainval)

    # Evaluate
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    return {
        "dataset": dataset,
        "feature_set": feature_set,
        "seed": seed,
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }


def compute_statistics(results: list) -> dict:
    """Compute statistics and significance tests."""
    # Group by dataset and feature_set
    grouped = {}
    for r in results:
        key = (r["dataset"], r["feature_set"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r["rmse"])

    stats_results = {}
    for (dataset, feature_set), rmses in grouped.items():
        rmses = np.array(rmses)
        stats_results[f"{dataset}_{feature_set}"] = {
            "rmse_mean": float(np.mean(rmses)),
            "rmse_std": float(np.std(rmses)),
            "rmse_values": rmses.tolist(),
            "n_seeds": len(rmses),
        }

    # T-tests: fp_mu_sigma vs fp_only
    for dataset in DATASETS:
        fp_key = f"{dataset}_fp_only"
        hybrid_key = f"{dataset}_fp_mu_sigma"

        if fp_key in stats_results and hybrid_key in stats_results:
            fp_rmses = stats_results[fp_key]["rmse_values"]
            hybrid_rmses = stats_results[hybrid_key]["rmse_values"]

            t_stat, p_value = stats.ttest_ind(hybrid_rmses, fp_rmses, equal_var=False)
            improvement = (np.mean(fp_rmses) - np.mean(hybrid_rmses)) / np.mean(fp_rmses) * 100

            stats_results[f"{dataset}_hybrid_vs_fp"] = {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant_0.05": p_value < 0.05,
                "significant_0.01": p_value < 0.01,
                "hybrid_better": np.mean(hybrid_rmses) < np.mean(fp_rmses),
                "improvement_pct": float(improvement),
            }

    return stats_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--feature-sets", nargs="+", default=FEATURE_SETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--output-dir", default="results/hybrid_10seed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(args.datasets) * len(args.feature_sets) * len(args.seeds)
    print(f"Hybrid Feature 10-Seed Validation")
    print(f"Datasets: {args.datasets}")
    print(f"Feature sets: {args.feature_sets}")
    print(f"Seeds: {args.seeds}")
    print(f"Total experiments: {total}")
    print("=" * 70)

    results = []
    completed = 0

    for dataset in args.datasets:
        print(f"\n=== {dataset} ===")
        for feature_set in args.feature_sets:
            for seed in args.seeds:
                completed += 1
                try:
                    result = run_experiment(dataset, feature_set, seed)
                    results.append(result)
                    print(f"  [{completed}/{total}] {feature_set} seed={seed}: RMSE={result['rmse']:.4f}")

                    # Save incrementally
                    with open(output_dir / "results_partial.json", "w") as f:
                        json.dump(results, f, indent=2)

                except Exception as e:
                    print(f"  [{completed}/{total}] {feature_set} seed={seed}: ERROR - {e}")

    # Compute statistics
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    stats_results = compute_statistics(results)

    # Print summary
    for dataset in args.datasets:
        print(f"\n{dataset.upper()}:")
        for fs in args.feature_sets:
            key = f"{dataset}_{fs}"
            if key in stats_results:
                s = stats_results[key]
                print(f"  {fs:15s}: {s['rmse_mean']:.4f} +/- {s['rmse_std']:.4f}")

        # Print hybrid vs FP comparison
        ttest_key = f"{dataset}_hybrid_vs_fp"
        if ttest_key in stats_results:
            t = stats_results[ttest_key]
            sig = "***" if t["significant_0.01"] else ("**" if t["significant_0.05"] else "ns")
            if t["hybrid_better"]:
                print(f"  >> Hybrid improves by {t['improvement_pct']:.1f}% (p={t['p_value']:.4f} {sig})")
            else:
                print(f"  >> FP alone is better (p={t['p_value']:.4f})")

    # Save final results
    final_output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "datasets": args.datasets,
            "feature_sets": args.feature_sets,
            "seeds": args.seeds,
        },
        "raw_results": results,
        "statistics": stats_results,
    }

    with open(output_dir / "hybrid_validation_results.json", "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"\nResults saved to {output_dir}/hybrid_validation_results.json")


if __name__ == "__main__":
    main()
