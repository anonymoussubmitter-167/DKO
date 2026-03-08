#!/usr/bin/env python3
"""
Experiment 2: Conformer Count Ablation.

Tests sensitivity to number of conformers n in {1, 5, 10, 20, 50} on ESOL.
Runs both dko_gated (neural) and hybrid FP+mu+sigma (XGBoost) to show
diminishing returns.

The conformers in pickles are energy-ordered, so taking the first n
gives a natural ablation from lowest-energy only to full ensemble.
"""

import argparse
import json
import pickle
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import xgboost as xgb
except ImportError:
    print("ERROR: xgboost required"); sys.exit(1)

from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator

from dko.data.datasets import ConformerDataset
from dko.models.dko_variants import DKOGatedFusion
from dko.training.trainer import train_model

DATASET = "esol"
CONFORMER_COUNTS = [1, 5, 10, 20, 50]
SEEDS = [42, 123, 456]

_FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(2048)
    return _FP_GEN.GetFingerprintAsNumPy(mol).astype(np.float64)


def load_raw_data(dataset_name, split):
    """Load raw pickle data."""
    path = Path(f"data/conformers/{dataset_name}/{split}.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def compute_hybrid_features(data, max_conformers, max_dim=256):
    """Compute FP, mu, sigma features with limited conformers."""
    smiles_list = data["smiles"]
    features_list = data["features"]
    labels = np.array([float(y) for y in data["labels"]])

    n = len(smiles_list)
    fp_features = np.zeros((n, 2048))
    mu_features = np.zeros((n, max_dim))
    sigma_features = np.zeros((n, 5))

    for i, (smi, mol_feats) in enumerate(zip(smiles_list, features_list)):
        fp_features[i] = smiles_to_fingerprint(smi)

        # Subsample conformers (first n, energy-ordered)
        n_conf = min(len(mol_feats), max_conformers)
        selected_feats = mol_feats[:n_conf]

        padded = []
        for conf_feat in selected_feats:
            cf = np.array(conf_feat).flatten()
            if len(cf) > max_dim:
                cf = cf[:max_dim]
            elif len(cf) < max_dim:
                cf = np.pad(cf, (0, max_dim - len(cf)))
            padded.append(cf)

        conformers = np.array(padded)
        mu = np.mean(conformers, axis=0)
        mu_features[i] = mu

        if len(conformers) > 1:
            centered = conformers - mu
            variances = np.mean(centered ** 2, axis=0)
            total_var = np.sum(variances)
            max_var = np.max(variances)
            mean_var = np.mean(variances)
            sorted_var = np.sort(variances)[::-1]
            top5_var = np.sum(sorted_var[:5])
            eff_rank = float(np.sum(variances > 0.01 * total_var)) if total_var > 0 else 0
            sigma_features[i] = [total_var, max_var, mean_var, top5_var, eff_rank]

    return fp_features, mu_features, sigma_features, labels


def run_xgb_experiment(dataset_name, max_conformers, seed):
    """Run hybrid XGBoost with limited conformers."""
    train_data = load_raw_data(dataset_name, "train")
    val_data = load_raw_data(dataset_name, "val")
    test_data = load_raw_data(dataset_name, "test")

    train_fp, train_mu, train_sigma, train_y = compute_hybrid_features(train_data, max_conformers)
    val_fp, val_mu, val_sigma, val_y = compute_hybrid_features(val_data, max_conformers)
    test_fp, test_mu, test_sigma, test_y = compute_hybrid_features(test_data, max_conformers)

    # Combine train + val
    train_X = np.vstack([
        np.hstack([train_fp, train_mu, train_sigma]),
        np.hstack([val_fp, val_mu, val_sigma]),
    ])
    train_labels = np.concatenate([train_y, val_y])

    test_X = np.hstack([test_fp, test_mu, test_sigma])

    model = xgb.XGBRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=seed,
        n_jobs=4, verbosity=0, tree_method='hist',
    )
    model.fit(train_X, train_labels)
    pred = model.predict(test_X)

    return {
        "rmse": float(np.sqrt(mean_squared_error(test_y, pred))),
        "mae": float(mean_absolute_error(test_y, pred)),
        "r2": float(r2_score(test_y, pred)),
    }


def make_conformer_dataset(data, max_conformers, max_feature_dim=1024):
    """Create ConformerDataset with limited conformers."""
    smiles = data["smiles"]
    labels = np.array([float(y) for y in data["labels"]])
    features_list = data["features"]

    conformer_features = []
    for mol_feats in features_list:
        n_conf = min(len(mol_feats), max_conformers)
        selected = mol_feats[:n_conf]

        padded_confs = []
        for conf_feat in selected:
            cf = np.array(conf_feat).flatten()
            if len(cf) < max_feature_dim:
                cf = np.pad(cf, (0, max_feature_dim - len(cf)))
            else:
                cf = cf[:max_feature_dim]
            padded_confs.append(cf)

        conformer_features.append(torch.tensor(np.array(padded_confs), dtype=torch.float32))

    energies = None
    if "energies" in data and data["energies"] is not None:
        energies_list = []
        for e in data["energies"]:
            n_conf = min(len(e), max_conformers)
            energies_list.append(torch.tensor(e[:n_conf], dtype=torch.float32))
        energies = energies_list

    return ConformerDataset(
        smiles=smiles,
        labels=labels,
        conformer_features=conformer_features,
        conformer_energies=energies,
        max_conformers=max_conformers,
        task_type="regression",
        feature_dim=max_feature_dim,
    )


def run_neural_experiment(dataset_name, max_conformers, seed, device):
    """Run dko_gated with limited conformers."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_data = load_raw_data(dataset_name, "train")
    val_data = load_raw_data(dataset_name, "val")
    test_data = load_raw_data(dataset_name, "test")

    train_ds = make_conformer_dataset(train_data, max_conformers)
    val_ds = make_conformer_dataset(val_data, max_conformers)
    test_ds = make_conformer_dataset(test_data, max_conformers)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    feature_dim = 1024
    model = DKOGatedFusion(feature_dim=feature_dim, output_dim=1, hidden_dim=128, k=10)

    training_config = {
        "optimizer": "AdamW",
        "base_learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "max_epochs": 300,
        "early_stopping_patience": 30,
        "gradient_clip_max_norm": 1.0,
        "mixed_precision": False,
        "task_type": "regression",
    }

    model, train_results = train_model(
        model, train_loader, val_loader, training_config,
        device=device, experiment_name=f"ablation_n{max_conformers}_s{seed}",
    )

    # Evaluate
    from dko.training.evaluator import Evaluator
    evaluator = Evaluator(task_type="regression", device=device)
    test_metrics = evaluator.evaluate(model, test_loader)

    return test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--conformer-counts", nargs="+", type=int, default=CONFORMER_COUNTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="results/conformer_ablation")
    parser.add_argument("--skip-neural", action="store_true", help="Skip neural experiments (faster)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"

    print("Conformer Count Ablation")
    print(f"Dataset: {args.dataset}")
    print(f"Conformer counts: {args.conformer_counts}")
    print(f"Seeds: {args.seeds}")
    print(f"Device: {device}")
    print("=" * 70)

    all_results = []

    # XGBoost hybrid experiments
    print("\n--- Hybrid XGBoost (FP+mu+sigma) ---")
    for n_conf in args.conformer_counts:
        rmses = []
        for seed in args.seeds:
            metrics = run_xgb_experiment(args.dataset, n_conf, seed)
            rmses.append(metrics["rmse"])
            all_results.append({
                "method": "hybrid_xgb",
                "n_conformers": n_conf,
                "seed": seed,
                **metrics,
            })
        print(f"  n={n_conf:3d}: RMSE={np.mean(rmses):.4f}+/-{np.std(rmses):.4f}")

    # Neural dko_gated experiments
    if not args.skip_neural:
        print("\n--- dko_gated (Neural) ---")
        for n_conf in args.conformer_counts:
            rmses = []
            for seed in args.seeds:
                try:
                    metrics = run_neural_experiment(args.dataset, n_conf, seed, device)
                    rmse = metrics.get("rmse", metrics.get("test_rmse", float("nan")))
                    rmses.append(rmse)
                    all_results.append({
                        "method": "dko_gated",
                        "n_conformers": n_conf,
                        "seed": seed,
                        **{k: float(v) if isinstance(v, (int, float, np.floating)) else v
                           for k, v in metrics.items()},
                    })
                except Exception as e:
                    print(f"  ERROR n={n_conf} seed={seed}: {e}")
                    traceback.print_exc()

            if rmses:
                print(f"  n={n_conf:3d}: RMSE={np.mean(rmses):.4f}+/-{np.std(rmses):.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Method':<20} {'n_conf':<10} {'RMSE':<20} {'R2':<15}")
    print("-" * 65)

    for method in ["hybrid_xgb", "dko_gated"]:
        for n_conf in args.conformer_counts:
            matched = [r for r in all_results
                       if r["method"] == method and r["n_conformers"] == n_conf]
            if matched:
                rmses = [r["rmse"] for r in matched]
                r2s = [r.get("r2", float("nan")) for r in matched]
                print(f"{method:<20} {n_conf:<10} "
                      f"{np.mean(rmses):.4f}+/-{np.std(rmses):.4f}   "
                      f"{np.nanmean(r2s):.4f}")

    # Save
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "dataset": args.dataset,
            "conformer_counts": args.conformer_counts,
            "seeds": args.seeds,
        },
        "raw_results": all_results,
    }
    with open(output_dir / "conformer_ablation_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}/conformer_ablation_results.json")


if __name__ == "__main__":
    main()
