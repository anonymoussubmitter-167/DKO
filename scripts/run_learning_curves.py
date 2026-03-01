#!/usr/bin/env python3
"""Learning curve experiment: how do methods scale with training data?

Tests FP+XGB vs Hybrid XGB at 10%, 25%, 50%, 75%, 100% of training data.
Addresses: "Will neural methods catch up with more data?"
"""

import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime

import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Config ──
DATASETS = ["esol", "freesolv", "lipophilicity", "qm9_gap"]
FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]
SEEDS = [42, 123, 456]
OUTPUT_DIR = Path("results/learning_curves")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(2048)
    return _FP_GEN.GetFingerprintAsNumPy(mol).astype(np.float64)


def compute_conformer_stats(features, max_dim=256):
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


def load_split(dataset_name, split):
    path = Path(f"data/conformers/{dataset_name}/{split}.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["smiles"], np.array([float(y) for y in data["labels"]]), data["features"]


def precompute_all(smiles_list, features_list):
    n = len(smiles_list)
    fp = np.zeros((n, 2048))
    mu = np.zeros((n, 256))
    sigma = np.zeros((n, 5))
    for i, (smi, feats) in enumerate(zip(smiles_list, features_list)):
        fp[i] = smiles_to_fingerprint(smi)
        mu[i], sigma[i] = compute_conformer_stats(feats, max_dim=256)
    return fp, mu, sigma


def run_xgb(train_X, train_y, test_X, test_y, seed):
    model = xgb.XGBRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=seed,
        n_jobs=4, verbosity=0, tree_method='hist',
    )
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    return (float(np.sqrt(mean_squared_error(test_y, pred))),
            float(mean_absolute_error(test_y, pred)),
            float(r2_score(test_y, pred)))


def main():
    print("Learning Curve Experiment")
    print("=" * 60)

    all_results = []

    for dataset in DATASETS:
        print(f"\n=== {dataset.upper()} ===")

        # Load data
        train_smi, train_y, train_feats = load_split(dataset, "train")
        val_smi, val_y, val_feats = load_split(dataset, "val")
        test_smi, test_y, test_feats = load_split(dataset, "test")

        # Combine train+val
        trainval_smi = list(train_smi) + list(val_smi)
        trainval_y = np.concatenate([train_y, val_y])
        trainval_feats = list(train_feats) + list(val_feats)

        # Precompute features
        tv_fp, tv_mu, tv_sigma = precompute_all(trainval_smi, trainval_feats)
        te_fp, te_mu, te_sigma = precompute_all(test_smi, test_feats)

        n_total = len(trainval_y)

        for frac in FRACTIONS:
            n_train = max(10, int(frac * n_total))  # At least 10 samples

            for seed in SEEDS:
                # Subsample training data
                rng = np.random.RandomState(seed)
                indices = rng.choice(n_total, size=n_train, replace=False)

                sub_fp = tv_fp[indices]
                sub_mu = tv_mu[indices]
                sub_sigma = tv_sigma[indices]
                sub_y = trainval_y[indices]

                # Feature configs
                configs = {
                    "FP-only": (sub_fp, te_fp),
                    "FP+Mu+Sigma": (np.hstack([sub_fp, sub_mu, sub_sigma]),
                                    np.hstack([te_fp, te_mu, te_sigma])),
                    "Mu-only": (sub_mu, te_mu),
                }

                for name, (train_X, test_X) in configs.items():
                    rmse, mae, r2 = run_xgb(train_X, sub_y, test_X, test_y, seed)
                    all_results.append({
                        "dataset": dataset,
                        "features": name,
                        "fraction": frac,
                        "n_train": n_train,
                        "seed": seed,
                        "rmse": rmse,
                        "mae": mae,
                        "r2": r2,
                    })

            # Print summary for this fraction
            print(f"\n  Fraction={frac:.0%} (n={n_train}):")
            for name in ["FP-only", "FP+Mu+Sigma", "Mu-only"]:
                subset = [r for r in all_results
                          if r["dataset"] == dataset
                          and r["features"] == name
                          and r["fraction"] == frac]
                rmses = [r["rmse"] for r in subset]
                print(f"    {name:<15s}: RMSE={np.mean(rmses):.4f}±{np.std(rmses):.4f}")

    # Summary: hybrid improvement at each fraction
    print("\n\n" + "=" * 60)
    print("HYBRID IMPROVEMENT BY TRAINING SIZE")
    print("=" * 60)
    for dataset in DATASETS:
        print(f"\n{dataset.upper()}:")
        for frac in FRACTIONS:
            fp_rmses = [r["rmse"] for r in all_results
                        if r["dataset"] == dataset and r["features"] == "FP-only"
                        and r["fraction"] == frac]
            hybrid_rmses = [r["rmse"] for r in all_results
                            if r["dataset"] == dataset and r["features"] == "FP+Mu+Sigma"
                            and r["fraction"] == frac]
            if fp_rmses and hybrid_rmses:
                imp = (np.mean(fp_rmses) - np.mean(hybrid_rmses)) / np.mean(fp_rmses) * 100
                n_train = int(frac * len(trainval_y)) if dataset == DATASETS[-1] else "?"
                print(f"  {frac:5.0%}: FP={np.mean(fp_rmses):.4f}, "
                      f"Hybrid={np.mean(hybrid_rmses):.4f}, Δ={imp:+.1f}%")

    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {"datasets": DATASETS, "fractions": FRACTIONS, "seeds": SEEDS},
        "raw_results": all_results,
    }
    with open(OUTPUT_DIR / "learning_curve_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
