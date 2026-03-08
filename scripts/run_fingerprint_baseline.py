#!/usr/bin/env python3
"""
Fingerprint + XGBoost baseline for all datasets.

This provides a lower bound: if conformer-based methods can't beat fingerprints,
they're not adding value.

Features: Morgan fingerprints (ECFP4, radius=2, 2048 bits)
Model: XGBoost with default hyperparameters + light tuning
"""

import argparse
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("WARNING: xgboost not installed, using sklearn GradientBoosting")
    from sklearn.ensemble import GradientBoostingRegressor

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("ERROR: rdkit required for fingerprint generation")
    sys.exit(1)


DATASETS = ["esol", "freesolv", "lipophilicity", "qm9_gap", "qm9_homo", "qm9_lumo", "bace", "bbbp"]
SEEDS = [42, 123, 456]


def smiles_to_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Convert SMILES to Morgan fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)


def load_dataset_with_smiles(dataset_name: str, split: str) -> tuple:
    """Load dataset and extract SMILES + targets."""
    data_path = Path(f"data/conformers/{dataset_name}/{split}.pkl")

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    # Data is a dict with parallel arrays: 'smiles', 'labels', 'features', etc.
    if isinstance(data, dict) and "smiles" in data:
        smiles_list = data["smiles"]
        targets = np.array([float(y) for y in data["labels"]])
        return smiles_list, targets

    # Fallback for list-of-dict format
    smiles_list = []
    targets = []

    for item in data:
        # Try different keys for SMILES
        smiles = item.get("smiles") or item.get("SMILES") or item.get("canonical_smiles")
        if smiles is None:
            # Try to get from metadata
            smiles = item.get("metadata", {}).get("smiles")

        if smiles is None:
            print(f"Warning: No SMILES found for item in {dataset_name}/{split}")
            continue

        target = item.get("target") or item.get("y") or item.get("label")
        if target is None:
            target = item.get("targets")
        if isinstance(target, (list, np.ndarray)):
            target = target[0] if len(target) > 0 else None

        if target is not None:
            smiles_list.append(smiles)
            targets.append(float(target))

    return smiles_list, np.array(targets)


def run_fingerprint_baseline(dataset_name: str, seed: int) -> dict:
    """Run fingerprint baseline on a dataset."""
    np.random.seed(seed)

    # Load data
    train_smiles, train_y = load_dataset_with_smiles(dataset_name, "train")
    val_smiles, val_y = load_dataset_with_smiles(dataset_name, "val")
    test_smiles, test_y = load_dataset_with_smiles(dataset_name, "test")

    if len(train_smiles) == 0:
        return {"error": f"No data found for {dataset_name}"}

    # Generate fingerprints
    print(f"  Generating fingerprints for {len(train_smiles)} train, {len(test_smiles)} test molecules...", flush=True)
    train_X = np.array([smiles_to_fingerprint(s) for s in train_smiles])
    val_X = np.array([smiles_to_fingerprint(s) for s in val_smiles])
    test_X = np.array([smiles_to_fingerprint(s) for s in test_smiles])
    print(f"  Fingerprints done. Training XGBoost...", flush=True)

    # Combine train + val for final training
    full_train_X = np.vstack([train_X, val_X])
    full_train_y = np.concatenate([train_y, val_y])

    # Train model
    if HAS_XGBOOST:
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=4,
            verbosity=1,
            tree_method='hist',  # Faster histogram-based method
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=seed,
        )

    model.fit(full_train_X, full_train_y)
    print(f"  Model trained. Evaluating...", flush=True)

    # Predict
    train_pred = model.predict(full_train_X)
    test_pred = model.predict(test_X)

    # Metrics
    train_rmse = np.sqrt(mean_squared_error(full_train_y, train_pred))
    test_rmse = np.sqrt(mean_squared_error(test_y, test_pred))
    test_mae = mean_absolute_error(test_y, test_pred)
    test_r2 = r2_score(test_y, test_pred)

    return {
        "dataset": dataset_name,
        "model": "fingerprint_xgboost",
        "seed": seed,
        "n_train": len(full_train_X),
        "n_test": len(test_X),
        "train_rmse": float(train_rmse),
        "test_rmse": float(test_rmse),
        "test_mae": float(test_mae),
        "test_r2": float(test_r2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--output-dir", type=str, default="results/fingerprint_baseline")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Fingerprint + XGBoost Baseline")
    print(f"Datasets: {args.datasets}")
    print(f"Seeds: {args.seeds}")
    print("=" * 70)

    results = []

    for dataset in args.datasets:
        print(f"\n{dataset}:")
        for seed in args.seeds:
            print(f"  seed={seed}...", end=" ", flush=True)
            try:
                result = run_fingerprint_baseline(dataset, seed)
                results.append(result)
                if "error" not in result:
                    print(f"RMSE={result['test_rmse']:.4f}")
                else:
                    print(f"ERROR: {result['error']}")
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()

    # Aggregate results
    print("\n" + "=" * 70)
    print("SUMMARY (mean +/- std over 3 seeds)")
    print("=" * 70)

    summary = {}
    for dataset in args.datasets:
        dataset_results = [r for r in results if r.get("dataset") == dataset and "error" not in r]
        if dataset_results:
            rmses = [r["test_rmse"] for r in dataset_results]
            maes = [r["test_mae"] for r in dataset_results]
            r2s = [r["test_r2"] for r in dataset_results]

            summary[dataset] = {
                "rmse_mean": float(np.mean(rmses)),
                "rmse_std": float(np.std(rmses)),
                "mae_mean": float(np.mean(maes)),
                "mae_std": float(np.std(maes)),
                "r2_mean": float(np.mean(r2s)),
                "r2_std": float(np.std(r2s)),
            }

            print(f"{dataset:20s}: RMSE={np.mean(rmses):.4f}+/-{np.std(rmses):.4f}, "
                  f"MAE={np.mean(maes):.4f}+/-{np.std(maes):.4f}, "
                  f"R2={np.mean(r2s):.4f}+/-{np.std(r2s):.4f}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "fingerprint": "Morgan (ECFP4)",
            "radius": 2,
            "n_bits": 2048,
            "model": "XGBoost" if HAS_XGBOOST else "GradientBoosting",
        },
        "raw_results": results,
        "summary": summary,
    }

    with open(output_dir / "fingerprint_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_dir / 'fingerprint_results.json'}")


if __name__ == "__main__":
    main()
