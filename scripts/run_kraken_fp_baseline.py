#!/usr/bin/env python3
"""
Fingerprint + XGBoost baseline on Kraken datasets.
Reports both RMSE and MAE (MARCEL uses MAE).
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


DATASETS = ["kraken_B5", "kraken_L", "kraken_burB5", "kraken_burL"]
SEEDS = [42, 123, 456]


def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)


def load_split(dataset_name, split):
    path = Path(f"data/conformers/{dataset_name}/{split}.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["smiles"], np.array(data["labels"], dtype=float)


def run_experiment(dataset_name, seed):
    train_smiles, train_y = load_split(dataset_name, "train")
    val_smiles, val_y = load_split(dataset_name, "val")
    test_smiles, test_y = load_split(dataset_name, "test")

    # Compute fingerprints
    train_X = np.array([smiles_to_fingerprint(s) for s in train_smiles])
    val_X = np.array([smiles_to_fingerprint(s) for s in val_smiles])
    test_X = np.array([smiles_to_fingerprint(s) for s in test_smiles])

    # Combine train + val
    full_X = np.vstack([train_X, val_X])
    full_y = np.concatenate([train_y, val_y])

    model = xgb.XGBRegressor(
        n_estimators=200,
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
    rmse = float(np.sqrt(mean_squared_error(test_y, test_pred)))
    mae = float(mean_absolute_error(test_y, test_pred))
    r2 = float(r2_score(test_y, test_pred))

    return {"dataset": dataset_name, "seed": seed, "rmse": rmse, "mae": mae, "r2": r2}


def main():
    output_dir = Path("results/kraken_benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Kraken Fingerprint Baseline")
    print("=" * 50)

    results = []
    for dataset in DATASETS:
        rmses, maes = [], []
        for seed in SEEDS:
            r = run_experiment(dataset, seed)
            results.append(r)
            rmses.append(r["rmse"])
            maes.append(r["mae"])
            print(f"  {dataset} seed={seed}: RMSE={r['rmse']:.4f}, MAE={r['mae']:.4f}")

        print(f"  {dataset} MEAN: RMSE={np.mean(rmses):.4f}±{np.std(rmses):.4f}, "
              f"MAE={np.mean(maes):.4f}±{np.std(maes):.4f}")
        print()

    output = {
        "timestamp": datetime.now().isoformat(),
        "method": "Morgan FP + XGBoost",
        "results": results,
    }
    out_path = output_dir / "fingerprint_baseline.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
