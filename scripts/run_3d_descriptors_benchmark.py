#!/usr/bin/env python3
"""
Experiment 1: Enhanced 3D Descriptors Benchmark.

Benchmarks the 28 enhanced 3D descriptors (PMI, SASA, USR, etc.) implemented
in dko/data/features_3d.py but never evaluated on downstream tasks.

For each dataset:
1. Load SMILES from existing pickles
2. Generate conformers with RDKit ETKDG
3. Extract 28 enhanced 3D features per conformer
4. Compute mu (mean) and sigma invariants from 28-D features
5. Run XGBoost on: 3D-only, FP+3D, FP+3D+sigma vs FP-only baseline

Seeds: 42, 123, 456
"""

import argparse
import json
import pickle
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import xgboost as xgb
except ImportError:
    print("ERROR: xgboost required"); sys.exit(1)

from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator
from dko.data.features_3d import Enhanced3DFeatureExtractor

DATASETS = ["esol", "freesolv", "lipophilicity", "qm9_gap"]
SEEDS = [42, 123, 456]
N_CONFORMERS = 20  # Generate 20 conformers per molecule for 3D feature extraction

_FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(2048)
    return _FP_GEN.GetFingerprintAsNumPy(mol).astype(np.float64)


def extract_3d_features_for_molecule(smiles, extractor, n_conformers=N_CONFORMERS):
    """Generate conformers and extract 28 3D features per conformer.

    Returns:
        features_3d: array of shape (n_conf, 28) or None if failed
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    try:
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.numThreads = 1
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)
        if len(cids) == 0:
            # Fallback: try without ETKDG
            AllChem.EmbedMolecule(mol, randomSeed=42)
            if mol.GetNumConformers() == 0:
                return None
            cids = [0]
        # Optimize conformers
        for cid in cids:
            try:
                AllChem.MMFFOptimizeMolecule(mol, confId=cid, maxIters=200)
            except Exception:
                pass
    except Exception:
        return None

    features_list = []
    for cid in cids:
        try:
            feat = extractor.extract(mol, cid)
            features_list.append(feat.to_flat_vector())
        except Exception:
            continue

    if len(features_list) == 0:
        return None

    return np.array(features_list)


def compute_3d_stats(features_3d):
    """Compute mu (mean) and sigma invariants from 3D features.

    Args:
        features_3d: (n_conf, 28) array

    Returns:
        mu_3d: (28,) mean features
        sigma_3d: (5,) scalar invariants of covariance
    """
    mu = np.mean(features_3d, axis=0)

    if len(features_3d) < 2:
        return mu, np.zeros(5)

    centered = features_3d - mu
    variances = np.mean(centered ** 2, axis=0)

    total_var = np.sum(variances)
    max_var = np.max(variances)
    mean_var = np.mean(variances)
    sorted_var = np.sort(variances)[::-1]
    top5_var = np.sum(sorted_var[:5])
    effective_rank = float(np.sum(variances > 0.01 * total_var)) if total_var > 0 else 0

    return mu, np.array([total_var, max_var, mean_var, top5_var, effective_rank])


def precompute_all_features(dataset_name, split):
    """Precompute FP, 3D mu, and 3D sigma features for a dataset split."""
    path = Path(f"data/conformers/{dataset_name}/{split}.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)

    smiles_list = data["smiles"]
    labels = np.array([float(y) for y in data["labels"]])
    raw_features = data["features"]  # existing geometric features

    n = len(smiles_list)
    extractor = Enhanced3DFeatureExtractor()

    fp_features = np.zeros((n, 2048))
    mu_3d_features = np.zeros((n, 28))
    sigma_3d_features = np.zeros((n, 5))

    # Also compute geometric mu (from existing features) for comparison
    mu_geo_features = np.zeros((n, 256))
    sigma_geo_features = np.zeros((n, 5))

    failed = 0
    for i, smi in enumerate(smiles_list):
        # Fingerprint
        fp_features[i] = smiles_to_fingerprint(smi)

        # 3D enhanced features
        feats_3d = extract_3d_features_for_molecule(smi, extractor)
        if feats_3d is not None:
            mu_3d, sigma_3d = compute_3d_stats(feats_3d)
            mu_3d_features[i] = mu_3d
            sigma_3d_features[i] = sigma_3d
        else:
            failed += 1

        # Geometric mu/sigma from existing features (for comparison)
        mol_feats = raw_features[i]
        padded = []
        for conf_feat in mol_feats:
            cf = np.array(conf_feat).flatten()
            if len(cf) > 256:
                cf = cf[:256]
            elif len(cf) < 256:
                cf = np.pad(cf, (0, 256 - len(cf)))
            padded.append(cf)
        conformers = np.array(padded)
        mu_geo_features[i] = np.mean(conformers, axis=0)

        if len(conformers) > 1:
            centered = conformers - mu_geo_features[i]
            variances = np.mean(centered ** 2, axis=0)
            total_var = np.sum(variances)
            max_var = np.max(variances)
            mean_var = np.mean(variances)
            sorted_var = np.sort(variances)[::-1]
            top5_var = np.sum(sorted_var[:5])
            eff_rank = float(np.sum(variances > 0.01 * total_var)) if total_var > 0 else 0
            sigma_geo_features[i] = [total_var, max_var, mean_var, top5_var, eff_rank]

        if (i + 1) % 100 == 0:
            print(f"    Processed {i+1}/{n} molecules ({failed} failed 3D extraction)")

    print(f"  {split}: {n} molecules, {failed} failed 3D extraction")

    return {
        "labels": labels,
        "fp": fp_features,
        "mu_3d": mu_3d_features,
        "sigma_3d": sigma_3d_features,
        "mu_geo": mu_geo_features,
        "sigma_geo": sigma_geo_features,
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
    return rmse, mae, r2, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--output-dir", default="results/3d_descriptors")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        (["mu_3d"], "3D-only (mu)"),
        (["mu_3d", "sigma_3d"], "3D mu+sigma"),
        (["fp"], "FP-only"),
        (["fp", "mu_3d"], "FP + 3D mu"),
        (["fp", "mu_3d", "sigma_3d"], "FP + 3D mu+sigma"),
        (["fp", "mu_geo", "sigma_geo"], "FP + Geo mu+sigma (baseline)"),
        (["fp", "mu_3d", "sigma_3d", "mu_geo", "sigma_geo"], "FP + 3D + Geo (all)"),
    ]

    print("Enhanced 3D Descriptors Benchmark")
    print(f"Datasets: {args.datasets}")
    print(f"Seeds: {args.seeds}")
    print("=" * 70)

    all_results = []

    for dataset in args.datasets:
        print(f"\n=== {dataset.upper()} ===")
        try:
            print("  Precomputing features...")
            train_data = precompute_all_features(dataset, "train")
            val_data = precompute_all_features(dataset, "val")
            test_data = precompute_all_features(dataset, "test")

            # Combine train + val
            combined = {}
            for key in ["fp", "mu_3d", "sigma_3d", "mu_geo", "sigma_geo"]:
                combined[key] = np.vstack([train_data[key], val_data[key]])
            combined_y = np.concatenate([train_data["labels"], val_data["labels"]])
            test_y = test_data["labels"]

            for feat_keys, label in configs:
                rmses, maes, r2s = [], [], []
                for seed in args.seeds:
                    train_X = np.hstack([combined[k] for k in feat_keys])
                    test_X = np.hstack([test_data[k] for k in feat_keys])
                    rmse, mae, r2, _ = run_xgb(train_X, combined_y, test_X, test_y, seed)
                    rmses.append(rmse)
                    maes.append(mae)
                    r2s.append(r2)

                    all_results.append({
                        "dataset": dataset, "features": label, "seed": seed,
                        "rmse": rmse, "mae": mae, "r2": r2,
                        "feature_dim": train_X.shape[1],
                    })

                dim = all_results[-1]["feature_dim"]
                print(f"  {label:35s} (dim={dim:5d}): "
                      f"RMSE={np.mean(rmses):.4f}+/-{np.std(rmses):.4f}  "
                      f"R2={np.mean(r2s):.4f}")

        except Exception as e:
            print(f"  ERROR on {dataset}: {e}")
            traceback.print_exc()

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Dataset':<15} {'Features':<35} {'RMSE':<20} {'R2':<15}")
    print("-" * 85)

    for dataset in args.datasets:
        ds_results = [r for r in all_results if r["dataset"] == dataset]
        for label in dict.fromkeys(r["features"] for r in ds_results):
            label_results = [r for r in ds_results if r["features"] == label]
            rmses = [r["rmse"] for r in label_results]
            r2s = [r["r2"] for r in label_results]
            print(f"{dataset:<15} {label:<35} {np.mean(rmses):.4f}+/-{np.std(rmses):.4f}   {np.mean(r2s):.4f}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "datasets": args.datasets,
            "seeds": args.seeds,
            "n_conformers_3d": N_CONFORMERS,
        },
        "raw_results": all_results,
    }
    with open(output_dir / "3d_descriptors_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_dir}/3d_descriptors_results.json")


if __name__ == "__main__":
    main()
