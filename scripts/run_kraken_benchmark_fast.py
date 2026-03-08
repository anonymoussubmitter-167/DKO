#!/usr/bin/env python3
"""
Run DKO models on MARCEL Kraken benchmark - FAST VERSION with PCA.

Kraken: 1552 organophosphorus ligands with 4 steric descriptor targets:
- sterimol_B5, sterimol_L, sterimol_burB5, sterimol_burL

Uses PCA to reduce fingerprint dimension from 2048 to 128 before computing covariance.
"""

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from rdkit import Chem
from rdkit.Chem import AllChem

# Import our models
from dko.models import (
    DKOGatedFusion,
    DKOScalarInvariants,
    AttentionAggregation,
)


def mol_block_to_fingerprint(mol_block: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Convert MOL block to Morgan fingerprint."""
    mol = Chem.MolFromMolBlock(mol_block, removeHs=False)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)


def precompute_kraken_features(data: Dict, max_conformers: int = 50, pca_dim: int = 128):
    """Precompute features with PCA reduction for faster covariance computation."""
    print(f"Precomputing fingerprint features for {len(data)} molecules...")

    # First pass: compute all conformer fingerprints
    all_fps = []
    mol_conf_fps = {}
    mol_conf_weights = {}

    for i, mol_id in enumerate(data.keys()):
        if i % 100 == 0:
            print(f"  Computing fingerprints: {i}/{len(data)}...", flush=True)

        smiles, targets, conformers = data[mol_id]

        conf_fps = []
        conf_weights = []

        for conf_id, (mol_block, weight, conf_targets) in list(conformers.items())[:max_conformers]:
            fp = mol_block_to_fingerprint(mol_block)
            conf_fps.append(fp)
            conf_weights.append(weight)
            all_fps.append(fp)

        mol_conf_fps[mol_id] = conf_fps
        mol_conf_weights[mol_id] = conf_weights

    # Fit PCA on all fingerprints
    print(f"  Fitting PCA ({len(all_fps)} fingerprints -> {pca_dim} dims)...")
    all_fps = np.array(all_fps)
    pca = PCA(n_components=pca_dim, random_state=42)
    pca.fit(all_fps)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # Second pass: transform and pad
    features_cache = {}
    weights_cache = {}

    for mol_id in data.keys():
        conf_fps = mol_conf_fps[mol_id]
        conf_weights = mol_conf_weights[mol_id]

        # Transform with PCA
        transformed = pca.transform(np.array(conf_fps))

        # Pad to max_conformers
        padded_features = np.zeros((max_conformers, pca_dim), dtype=np.float32)
        padded_weights = np.zeros(max_conformers, dtype=np.float32)

        n_conf = len(transformed)
        padded_features[:n_conf] = transformed
        padded_weights[:n_conf] = conf_weights

        features_cache[mol_id] = padded_features
        weights_cache[mol_id] = padded_weights

    print(f"Done precomputing features.")
    return features_cache, weights_cache, pca


class KrakenDataset(Dataset):
    """Dataset for Kraken benchmark."""

    def __init__(
        self,
        data: Dict,
        mol_ids: List[str],
        target_name: str = "sterimol_B5",
        features_cache: Dict = None,
        weights_cache: Dict = None,
    ):
        self.mol_ids = mol_ids
        self.target_name = target_name

        self.features = []
        self.targets = []
        self.weights = []

        for mol_id in mol_ids:
            smiles, targets, conformers = data[mol_id]
            target = targets[target_name]

            self.features.append(features_cache[mol_id])
            self.targets.append(target)
            self.weights.append(weights_cache[mol_id])

    def __len__(self):
        return len(self.mol_ids)

    def __getitem__(self, idx):
        return {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "weights": torch.tensor(self.weights[idx], dtype=torch.float32),
            "target": torch.tensor(self.targets[idx], dtype=torch.float32),
        }


def train_epoch(model, loader, optimizer, criterion, device, use_sigma=False, use_attention=False):
    model.train()
    total_loss = 0

    for batch in loader:
        features = batch["features"].to(device)  # (B, n_conf, D)
        targets = batch["target"].to(device)

        optimizer.zero_grad()

        # Compute mu and sigma
        mu = features.mean(dim=1)  # (B, D)

        if use_attention:
            # Attention model expects full conformer features, returns (pred, attn_weights)
            pred = model(features)[0].squeeze(-1)
        elif use_sigma:
            # Compute covariance - now D=128 so this is fast (128x128 instead of 2048x2048)
            centered = features - mu.unsqueeze(1)
            weights = batch["weights"].to(device).unsqueeze(-1)  # (B, n_conf, 1)
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
            sigma = torch.einsum('bni,bnj->bij', centered * weights, centered)  # (B, D, D)
            pred = model(mu, sigma).squeeze(-1)
        else:
            pred = model(mu).squeeze(-1)

        loss = criterion(pred, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(targets)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, device, use_sigma=False, use_attention=False):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            target = batch["target"]

            mu = features.mean(dim=1)

            if use_attention:
                pred = model(features)[0].squeeze(-1)
            elif use_sigma:
                centered = features - mu.unsqueeze(1)
                weights = batch["weights"].to(device).unsqueeze(-1)
                weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
                sigma = torch.einsum('bni,bnj->bij', centered * weights, centered)
                pred = model(mu, sigma).squeeze(-1)
            else:
                pred = model(mu).squeeze(-1)

            preds.append(pred.cpu())
            targets.append(target)

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    mae = np.mean(np.abs(preds - targets))

    return {"rmse": rmse, "mae": mae}


def run_experiment(
    data: Dict,
    target_name: str,
    model_name: str,
    seed: int,
    device: str,
    features_cache: Dict,
    weights_cache: Dict,
    feat_dim: int,
):
    """Run single experiment."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Split data
    mol_ids = list(data.keys())
    train_ids, test_ids = train_test_split(mol_ids, test_size=0.2, random_state=seed)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.125, random_state=seed)

    # Create datasets
    train_dataset = KrakenDataset(data, train_ids, target_name, features_cache, weights_cache)
    val_dataset = KrakenDataset(data, val_ids, target_name, features_cache, weights_cache)
    test_dataset = KrakenDataset(data, test_ids, target_name, features_cache, weights_cache)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Create model
    use_sigma = model_name.startswith("dko_")
    use_attention = (model_name == "attention")

    if model_name == "dko_gated":
        model = DKOGatedFusion(feature_dim=feat_dim, hidden_dim=256, output_dim=1, k=10)
    elif model_name == "dko_invariants":
        model = DKOScalarInvariants(feature_dim=feat_dim, output_dim=1)
    elif model_name == "attention":
        model = AttentionAggregation(feature_dim=feat_dim, num_heads=4, num_outputs=1)
        use_sigma = False
    elif model_name == "mean":
        # Simple MLP on mean features
        model = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )
        use_sigma = False
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 30

    for epoch in range(200):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, use_sigma, use_attention)
        val_metrics = evaluate(model, val_loader, device, use_sigma, use_attention)

        if val_metrics["rmse"] < best_val_loss:
            best_val_loss = val_metrics["rmse"]
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Load best model and evaluate
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device, use_sigma, use_attention)

    return {
        "target": target_name,
        "model": model_name,
        "seed": seed,
        "test_rmse": float(test_metrics["rmse"]),
        "test_mae": float(test_metrics["mae"]),
        "best_epoch": epoch - patience_counter,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--targets", nargs="+", default=["sterimol_B5", "sterimol_L", "sterimol_burB5", "sterimol_burL"])
    parser.add_argument("--models", nargs="+", default=["dko_gated", "dko_invariants", "attention", "mean"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--pca-dim", type=int, default=128, help="PCA dimension for fingerprints")
    parser.add_argument("--output-dir", default="results/kraken_benchmark")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Kraken data
    print("Loading Kraken dataset...")
    with open("external/MARCEL/datasets/Kraken/raw/Kraken.pickle", "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} molecules")

    # Precompute features with PCA
    features_cache, weights_cache, pca = precompute_kraken_features(data, pca_dim=args.pca_dim)
    feat_dim = args.pca_dim

    print(f"\nDevice: {device}")
    print(f"Targets: {args.targets}")
    print(f"Models: {args.models}")
    print(f"Seeds: {args.seeds}")
    print(f"PCA dimension: {args.pca_dim}")
    print("=" * 70)

    results = []

    for target in args.targets:
        print(f"\n=== {target} ===")
        for model_name in args.models:
            for seed in args.seeds:
                print(f"  {model_name} seed={seed}...", end=" ", flush=True)
                try:
                    result = run_experiment(
                        data, target, model_name, seed, device,
                        features_cache, weights_cache, feat_dim
                    )
                    results.append(result)
                    print(f"RMSE={result['test_rmse']:.4f}")
                except Exception as e:
                    print(f"ERROR: {e}")
                    import traceback
                    traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("KRAKEN BENCHMARK SUMMARY")
    print("=" * 70)

    summary = {}
    for target in args.targets:
        print(f"\n{target}:")
        summary[target] = {}
        for model_name in args.models:
            model_results = [r for r in results if r["target"] == target and r["model"] == model_name]
            if model_results:
                rmses = [r["test_rmse"] for r in model_results]
                mean_rmse = np.mean(rmses)
                std_rmse = np.std(rmses)
                summary[target][model_name] = {"mean": mean_rmse, "std": std_rmse}
                print(f"  {model_name:20s}: {mean_rmse:.4f} ± {std_rmse:.4f}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "raw_results": results,
        "summary": summary,
    }

    with open(output_dir / "kraken_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_dir / 'kraken_results.json'}")


if __name__ == "__main__":
    main()
