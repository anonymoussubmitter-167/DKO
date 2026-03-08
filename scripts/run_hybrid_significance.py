#!/usr/bin/env python3
"""10-seed statistical significance test for hybrid improvement.

Tests H0: FP+Mu+Sigma has same RMSE as FP-only.
Uses paired t-test (same train/test split per seed) for maximum power.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats

import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Config ──
DATASETS = ["esol", "freesolv", "lipophilicity", "qm9_gap"]
SEEDS = list(range(10))  # 10 seeds: 0-9
OUTPUT_DIR = Path("results/hybrid_significance")
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
    rmse = float(np.sqrt(mean_squared_error(test_y, pred)))
    mae = float(mean_absolute_error(test_y, pred))
    r2 = float(r2_score(test_y, pred))
    return rmse, mae, r2


def main():
    print("10-Seed Hybrid Significance Test")
    print("=" * 60)

    all_results = []
    summary = {}

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

        # Feature configs
        configs = {
            "FP-only": (tv_fp, te_fp),
            "FP+Mu": (np.hstack([tv_fp, tv_mu]), np.hstack([te_fp, te_mu])),
            "FP+Mu+Sigma": (np.hstack([tv_fp, tv_mu, tv_sigma]),
                            np.hstack([te_fp, te_mu, te_sigma])),
            "Mu-only": (tv_mu, te_mu),
        }

        dataset_results = {name: {"rmse": [], "mae": [], "r2": []} for name in configs}

        for seed in SEEDS:
            for name, (train_X, test_X) in configs.items():
                rmse, mae, r2 = run_xgb(train_X, trainval_y, test_X, test_y, seed)
                dataset_results[name]["rmse"].append(rmse)
                dataset_results[name]["mae"].append(mae)
                dataset_results[name]["r2"].append(r2)
                all_results.append({
                    "dataset": dataset, "features": name,
                    "seed": seed, "rmse": rmse, "mae": mae, "r2": r2,
                })

        # Print summary
        print(f"\n  {'Config':<15s} {'RMSE (mean±std)':<22s} {'R² (mean±std)':<22s}")
        print(f"  {'-'*60}")
        for name in configs:
            r = dataset_results[name]
            print(f"  {name:<15s} {np.mean(r['rmse']):.4f}±{np.std(r['rmse']):.4f}      "
                  f"{np.mean(r['r2']):.4f}±{np.std(r['r2']):.4f}")

        # Paired t-test: FP+Mu+Sigma vs FP-only
        fp_rmses = np.array(dataset_results["FP-only"]["rmse"])
        hybrid_rmses = np.array(dataset_results["FP+Mu+Sigma"]["rmse"])
        fp_mu_rmses = np.array(dataset_results["FP+Mu"]["rmse"])

        # One-sided paired t-test: H0: hybrid >= fp, H1: hybrid < fp
        t_stat, p_two = stats.ttest_rel(hybrid_rmses, fp_rmses)
        p_one = p_two / 2 if t_stat < 0 else 1 - p_two / 2

        t_stat_mu, p_two_mu = stats.ttest_rel(fp_mu_rmses, fp_rmses)
        p_one_mu = p_two_mu / 2 if t_stat_mu < 0 else 1 - p_two_mu / 2

        improvement = (np.mean(fp_rmses) - np.mean(hybrid_rmses)) / np.mean(fp_rmses) * 100
        improvement_mu = (np.mean(fp_rmses) - np.mean(fp_mu_rmses)) / np.mean(fp_rmses) * 100

        print(f"\n  FP+Mu+Sigma vs FP-only:")
        print(f"    Improvement: {improvement:+.1f}%")
        print(f"    Paired t-test: t={t_stat:.3f}, p={p_one:.6f} (one-sided)")
        print(f"    Significant at p<0.05: {'YES' if p_one < 0.05 else 'NO'}")
        print(f"    Significant at p<0.01: {'YES' if p_one < 0.01 else 'NO'}")

        print(f"\n  FP+Mu vs FP-only:")
        print(f"    Improvement: {improvement_mu:+.1f}%")
        print(f"    Paired t-test: t={t_stat_mu:.3f}, p={p_one_mu:.6f} (one-sided)")

        # Also test: does sigma add to FP+Mu?
        t_stat_s, p_two_s = stats.ttest_rel(hybrid_rmses, fp_mu_rmses)
        p_one_s = p_two_s / 2 if t_stat_s < 0 else 1 - p_two_s / 2
        imp_s = (np.mean(fp_mu_rmses) - np.mean(hybrid_rmses)) / np.mean(fp_mu_rmses) * 100

        print(f"\n  FP+Mu+Sigma vs FP+Mu (marginal sigma):")
        print(f"    Improvement: {imp_s:+.1f}%")
        print(f"    Paired t-test: t={t_stat_s:.3f}, p={p_one_s:.6f} (one-sided)")

        summary[dataset] = {
            "fp_rmse_mean": float(np.mean(fp_rmses)),
            "fp_rmse_std": float(np.std(fp_rmses)),
            "hybrid_rmse_mean": float(np.mean(hybrid_rmses)),
            "hybrid_rmse_std": float(np.std(hybrid_rmses)),
            "fp_mu_rmse_mean": float(np.mean(fp_mu_rmses)),
            "fp_mu_rmse_std": float(np.std(fp_mu_rmses)),
            "improvement_pct": float(improvement),
            "t_stat": float(t_stat),
            "p_value_one_sided": float(p_one),
            "significant_005": bool(p_one < 0.05),
            "significant_001": bool(p_one < 0.01),
            "sigma_marginal_improvement_pct": float(imp_s),
            "sigma_marginal_p_value": float(p_one_s),
        }

    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {"datasets": DATASETS, "n_seeds": len(SEEDS)},
        "summary": summary,
        "raw_results": all_results,
    }
    with open(OUTPUT_DIR / "hybrid_significance_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
