#!/usr/bin/env python3
"""Scaffold split validation: re-run key comparisons with scaffold splits.

Addresses the #1 critique for MoleculeNet papers: random splits can leak
information via structurally similar molecules in train/test.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Config ──
DATASETS = ["esol", "freesolv", "lipophilicity"]
SEEDS = [42, 123, 456]
OUTPUT_DIR = Path("results/scaffold_splits")
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
    sigma_stats = np.array([total_var, max_var, mean_var, top5_var, effective_rank])
    return mu, sigma_stats


def get_scaffold(smiles):
    """Get Murcko scaffold for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles  # Use original SMILES as its own scaffold
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=False
        )
        return scaffold
    except:
        return smiles


def scaffold_split(smiles_list, labels, features_list, seed=42, frac_train=0.8, frac_val=0.1, frac_test=0.1):
    """Split data by Murcko scaffolds (standard MoleculeNet approach)."""
    rng = np.random.RandomState(seed)

    # Group molecules by scaffold
    scaffold_to_indices = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        scaffold = get_scaffold(smi)
        scaffold_to_indices[scaffold].append(i)

    # Sort scaffolds by size (largest first for determinism)
    scaffold_sets = list(scaffold_to_indices.values())
    scaffold_sets.sort(key=lambda x: len(x), reverse=True)

    # Assign scaffolds to splits
    n = len(smiles_list)
    train_cutoff = int(frac_train * n)
    val_cutoff = int((frac_train + frac_val) * n)

    train_idx, val_idx, test_idx = [], [], []
    for scaffold_set in scaffold_sets:
        if len(train_idx) + len(scaffold_set) <= train_cutoff:
            train_idx.extend(scaffold_set)
        elif len(train_idx) + len(val_idx) + len(scaffold_set) <= val_cutoff:
            val_idx.extend(scaffold_set)
        else:
            test_idx.extend(scaffold_set)

    # Shuffle within each split for randomness
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return train_idx, val_idx, test_idx


def load_all_data(dataset_name):
    """Load and merge all splits for a dataset."""
    all_smiles = []
    all_labels = []
    all_features = []

    for split in ["train", "val", "test"]:
        path = Path(f"data/conformers/{dataset_name}/{split}.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        all_smiles.extend(data["smiles"])
        all_labels.extend([float(y) for y in data["labels"]])
        all_features.extend(data["features"])

    return all_smiles, np.array(all_labels), all_features


def precompute_features(smiles_list, features_list, indices):
    """Compute features for a subset of molecules."""
    n = len(indices)
    fp_features = np.zeros((n, 2048))
    mu_features = np.zeros((n, 256))
    sigma_features = np.zeros((n, 5))

    for j, idx in enumerate(indices):
        fp_features[j] = smiles_to_fingerprint(smiles_list[idx])
        mu, sigma = compute_conformer_stats(features_list[idx], max_dim=256)
        mu_features[j] = mu
        sigma_features[j] = sigma

    return {"fp": fp_features, "mu": mu_features, "sigma": sigma_features}


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
    print("Scaffold Split Validation")
    print("=" * 60)

    all_results = []

    # Feature configs to test
    feature_configs = {
        "FP-only": ["fp"],
        "FP+Mu+Sigma": ["fp", "mu", "sigma"],
        "Mu-only": ["mu"],
    }

    for dataset in DATASETS:
        print(f"\n=== {dataset.upper()} ===")
        smiles_list, labels, features_list = load_all_data(dataset)
        n_total = len(smiles_list)
        print(f"  Total molecules: {n_total}")

        for seed in SEEDS:
            # Create scaffold split
            train_idx, val_idx, test_idx = scaffold_split(
                smiles_list, labels, features_list, seed=seed
            )
            trainval_idx = train_idx + val_idx

            print(f"  Seed {seed}: train={len(train_idx)}, val={len(val_idx)}, "
                  f"test={len(test_idx)}, scaffolds used")

            # Precompute features
            trainval_feats = precompute_features(smiles_list, features_list, trainval_idx)
            test_feats = precompute_features(smiles_list, features_list, test_idx)
            trainval_y = labels[trainval_idx]
            test_y = labels[test_idx]

            for config_name, feat_keys in feature_configs.items():
                train_X = np.hstack([trainval_feats[k] for k in feat_keys])
                test_X = np.hstack([test_feats[k] for k in feat_keys])

                rmse, mae, r2 = run_xgb(train_X, trainval_y, test_X, test_y, seed)

                result = {
                    "dataset": dataset,
                    "split_type": "scaffold",
                    "features": config_name,
                    "seed": seed,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                    "n_train": len(trainval_idx),
                    "n_test": len(test_idx),
                }
                all_results.append(result)
                print(f"    {config_name}: RMSE={rmse:.4f}, R²={r2:.4f}")

    # Also run random splits for direct comparison
    print("\n\n=== RANDOM SPLIT COMPARISON ===")
    for dataset in DATASETS:
        print(f"\n=== {dataset.upper()} (random) ===")
        smiles_list, labels, features_list = load_all_data(dataset)

        for seed in SEEDS:
            rng = np.random.RandomState(seed)
            n = len(smiles_list)
            perm = rng.permutation(n)
            train_cut = int(0.8 * n)
            val_cut = int(0.9 * n)
            train_idx = perm[:train_cut].tolist()
            val_idx = perm[train_cut:val_cut].tolist()
            test_idx = perm[val_cut:].tolist()
            trainval_idx = train_idx + val_idx

            trainval_feats = precompute_features(smiles_list, features_list, trainval_idx)
            test_feats = precompute_features(smiles_list, features_list, test_idx)
            trainval_y = labels[trainval_idx]
            test_y = labels[test_idx]

            for config_name, feat_keys in feature_configs.items():
                train_X = np.hstack([trainval_feats[k] for k in feat_keys])
                test_X = np.hstack([test_feats[k] for k in feat_keys])

                rmse, mae, r2 = run_xgb(train_X, trainval_y, test_X, test_y, seed)

                result = {
                    "dataset": dataset,
                    "split_type": "random",
                    "features": config_name,
                    "seed": seed,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                    "n_train": len(trainval_idx),
                    "n_test": len(test_idx),
                }
                all_results.append(result)
                print(f"    {config_name}: RMSE={rmse:.4f}, R²={r2:.4f}")

    # Summary
    print("\n\n" + "=" * 60)
    print("SUMMARY: Scaffold vs Random Split")
    print("=" * 60)
    for dataset in DATASETS:
        print(f"\n{dataset.upper()}:")
        for config_name in feature_configs:
            for split_type in ["random", "scaffold"]:
                subset = [r for r in all_results
                          if r["dataset"] == dataset
                          and r["features"] == config_name
                          and r["split_type"] == split_type]
                rmses = [r["rmse"] for r in subset]
                r2s = [r["r2"] for r in subset]
                print(f"  {split_type:10s} {config_name:15s}: "
                      f"RMSE={np.mean(rmses):.4f}±{np.std(rmses):.4f}, "
                      f"R²={np.mean(r2s):.4f}")

    # Compute hybrid improvement under both splits
    print("\n\nHYBRID IMPROVEMENT (FP+Mu+Sigma vs FP-only):")
    for dataset in DATASETS:
        for split_type in ["random", "scaffold"]:
            fp_rmses = [r["rmse"] for r in all_results
                        if r["dataset"] == dataset and r["features"] == "FP-only"
                        and r["split_type"] == split_type]
            hybrid_rmses = [r["rmse"] for r in all_results
                            if r["dataset"] == dataset and r["features"] == "FP+Mu+Sigma"
                            and r["split_type"] == split_type]
            improvement = (np.mean(fp_rmses) - np.mean(hybrid_rmses)) / np.mean(fp_rmses) * 100
            print(f"  {dataset:15s} {split_type:10s}: {improvement:+.1f}%")

    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "datasets": DATASETS,
            "seeds": SEEDS,
            "split_types": ["scaffold", "random"],
            "feature_configs": list(feature_configs.keys()),
        },
        "raw_results": all_results,
    }
    with open(OUTPUT_DIR / "scaffold_split_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'scaffold_split_results.json'}")


if __name__ == "__main__":
    main()
