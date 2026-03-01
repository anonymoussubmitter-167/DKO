#!/usr/bin/env python3
"""
Experiment 4: Hybrid Neural Model.

Tests whether a neural MLP on concatenated [FP; mu; sigma] features
can match or beat XGBoost hybrid. This fills the gap that hybrid
FP+conformer was only tested with XGBoost, not neural networks.

Architecture: MLP 2309 -> 512 -> 256 -> 128 -> 1
Input: [FP_2048; mu_256; sigma_5] = 2309 dims

Datasets: ESOL, FreeSolv, Lipophilicity, QM9-Gap
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
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import xgboost as xgb
except ImportError:
    xgb = None

from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator

DATASETS = ["esol", "freesolv", "lipophilicity", "qm9_gap"]
SEEDS = [42, 123, 456]

_FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


class HybridMLP(nn.Module):
    """MLP for hybrid FP+mu+sigma features."""

    def __init__(self, input_dim=2309, output_dim=1, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),

            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class HybridDataset(Dataset):
    """Dataset for concatenated hybrid features."""

    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        if self.labels.dim() == 1:
            self.labels = self.labels.unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(2048)
    return _FP_GEN.GetFingerprintAsNumPy(mol).astype(np.float64)


def compute_conformer_stats(features, max_dim=256):
    """Compute mu and sigma stats from conformer features."""
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


def precompute_features(dataset_name, split, max_dim=256):
    """Precompute all feature types for a split."""
    path = Path(f"data/conformers/{dataset_name}/{split}.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)

    smiles_list = data["smiles"]
    labels = np.array([float(y) for y in data["labels"]])
    features_list = data["features"]

    n = len(smiles_list)
    fp_features = np.zeros((n, 2048))
    mu_features = np.zeros((n, max_dim))
    sigma_features = np.zeros((n, 5))

    for i, (smi, mol_feats) in enumerate(zip(smiles_list, features_list)):
        fp_features[i] = smiles_to_fingerprint(smi)
        mu, sigma = compute_conformer_stats(mol_feats, max_dim=max_dim)
        mu_features[i] = mu
        sigma_features[i] = sigma

    return {
        "labels": labels,
        "fp": fp_features,
        "mu": mu_features,
        "sigma": sigma_features,
    }


def train_mlp(model, train_loader, val_loader, device, epochs=300, patience=30, lr=1e-3):
    """Train MLP with early stopping."""
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(y)
        train_loss /= len(train_loader.dataset)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                val_loss += criterion(pred, y).item() * len(y)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_state)
    return model


def evaluate_mlp(model, loader, device):
    """Evaluate MLP and return metrics."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            pred = model(X)
            preds.extend(pred.cpu().numpy().flatten().tolist())
            targets.extend(y.numpy().flatten().tolist())

    preds = np.array(preds)
    targets = np.array(targets)

    return {
        "rmse": float(np.sqrt(mean_squared_error(targets, preds))),
        "mae": float(mean_absolute_error(targets, preds)),
        "r2": float(r2_score(targets, preds)),
    }


def run_xgb(train_X, train_y, test_X, test_y, seed):
    """Run XGBoost for comparison."""
    if xgb is None:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}

    model = xgb.XGBRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=seed,
        n_jobs=4, verbosity=0, tree_method='hist',
    )
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    return {
        "rmse": float(np.sqrt(mean_squared_error(test_y, pred))),
        "mae": float(mean_absolute_error(test_y, pred)),
        "r2": float(r2_score(test_y, pred)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="results/hybrid_neural")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"

    print("Hybrid Neural Model Experiment")
    print(f"Datasets: {args.datasets}")
    print(f"Seeds: {args.seeds}")
    print(f"Device: {device}")
    print("=" * 70)

    all_results = []

    for dataset in args.datasets:
        print(f"\n=== {dataset.upper()} ===")
        try:
            print("  Precomputing features...")
            train_data = precompute_features(dataset, "train")
            val_data = precompute_features(dataset, "val")
            test_data = precompute_features(dataset, "test")

            # Concatenate features: FP(2048) + mu(256) + sigma(5) = 2309
            def concat_features(data):
                return np.hstack([data["fp"], data["mu"], data["sigma"]])

            train_X = concat_features(train_data)
            val_X = concat_features(val_data)
            test_X = concat_features(test_data)

            input_dim = train_X.shape[1]
            print(f"  Feature dim: {input_dim}")

            # Combined train+val for XGBoost
            trainval_X = np.vstack([train_X, val_X])
            trainval_y = np.concatenate([train_data["labels"], val_data["labels"]])

            for seed in args.seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)

                # --- Neural MLP ---
                train_ds = HybridDataset(train_X, train_data["labels"])
                val_ds = HybridDataset(val_X, val_data["labels"])
                test_ds = HybridDataset(test_X, test_data["labels"])

                train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
                val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
                test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

                model = HybridMLP(input_dim=input_dim)
                model = train_mlp(model, train_loader, val_loader, device)
                mlp_metrics = evaluate_mlp(model, test_loader, device)

                all_results.append({
                    "dataset": dataset, "method": "hybrid_mlp", "seed": seed,
                    **mlp_metrics,
                })
                print(f"  MLP  seed={seed}: RMSE={mlp_metrics['rmse']:.4f}, R2={mlp_metrics['r2']:.4f}")

                # --- XGBoost ---
                xgb_metrics = run_xgb(trainval_X, trainval_y, test_X, test_data["labels"], seed)
                all_results.append({
                    "dataset": dataset, "method": "hybrid_xgb", "seed": seed,
                    **xgb_metrics,
                })
                print(f"  XGB  seed={seed}: RMSE={xgb_metrics['rmse']:.4f}, R2={xgb_metrics['r2']:.4f}")

        except Exception as e:
            print(f"  ERROR on {dataset}: {e}")
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Dataset':<15} {'Method':<15} {'RMSE':<20} {'R2':<15}")
    print("-" * 65)

    for dataset in args.datasets:
        for method in ["hybrid_mlp", "hybrid_xgb"]:
            matched = [r for r in all_results
                       if r["dataset"] == dataset and r["method"] == method]
            if matched:
                rmses = [r["rmse"] for r in matched]
                r2s = [r["r2"] for r in matched]
                print(f"{dataset:<15} {method:<15} "
                      f"{np.mean(rmses):.4f}+/-{np.std(rmses):.4f}   "
                      f"{np.mean(r2s):.4f}")

    # Save
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "datasets": args.datasets,
            "seeds": args.seeds,
            "architecture": "MLP: input -> 512 -> 256 -> 128 -> 1",
            "input_features": "FP(2048) + mu(256) + sigma(5) = 2309",
        },
        "raw_results": all_results,
    }
    with open(output_dir / "hybrid_neural_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_dir}/hybrid_neural_results.json")


if __name__ == "__main__":
    main()
