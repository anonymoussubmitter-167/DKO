#!/usr/bin/env python3
"""
Run DKO models on MARCEL Kraken benchmark.

Kraken: 1552 organophosphorus ligands with 4 steric descriptor targets:
- sterimol_B5, sterimol_L, sterimol_burB5, sterimol_burL

These are Boltzmann-averaged properties that explicitly depend on conformer ensembles.
"""

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

# Import our models
from dko.models import (
    DKOGatedFusion,
    DKOScalarInvariants,
    AttentionAggregation,
    MeanEnsemble,
)


def mol_block_to_fingerprint(mol_block: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Convert MOL block to Morgan fingerprint."""
    mol = Chem.MolFromMolBlock(mol_block, removeHs=False)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)


def mol_block_to_3d_features(mol_block: str, max_atoms: int = 100) -> np.ndarray:
    """Extract simple 3D features from MOL block."""
    mol = Chem.MolFromMolBlock(mol_block, removeHs=False)
    if mol is None:
        return np.zeros(max_atoms * 4)  # x, y, z, atomic_num per atom

    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()

    features = []
    for i in range(min(n_atoms, max_atoms)):
        pos = conf.GetAtomPosition(i)
        atom = mol.GetAtomWithIdx(i)
        features.extend([pos.x, pos.y, pos.z, atom.GetAtomicNum()])

    # Pad to max_atoms
    while len(features) < max_atoms * 4:
        features.append(0.0)

    return np.array(features[:max_atoms * 4], dtype=np.float32)


def precompute_kraken_features(data: Dict, feature_type: str = "fingerprint", max_conformers: int = 50):
    """Precompute features for all molecules once."""
    print(f"Precomputing {feature_type} features for {len(data)} molecules...")

    features_cache = {}
    weights_cache = {}

    for i, mol_id in enumerate(data.keys()):
        if i % 100 == 0:
            print(f"  {i}/{len(data)}...", flush=True)

        smiles, targets, conformers = data[mol_id]

        conf_features = []
        conf_weights = []

        for conf_id, (mol_block, weight, conf_targets) in list(conformers.items())[:max_conformers]:
            if feature_type == "fingerprint":
                feat = mol_block_to_fingerprint(mol_block)
            else:
                feat = mol_block_to_3d_features(mol_block)
            conf_features.append(feat)
            conf_weights.append(weight)

        # Pad conformers
        feat_dim = conf_features[0].shape[0] if conf_features else 2048
        while len(conf_features) < max_conformers:
            conf_features.append(np.zeros(feat_dim))
            conf_weights.append(0.0)

        features_cache[mol_id] = np.stack(conf_features[:max_conformers])
        weights_cache[mol_id] = np.array(conf_weights[:max_conformers])

    print(f"Done precomputing features.")
    return features_cache, weights_cache


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


def train_epoch(model, loader, optimizer, criterion, device, use_sigma=False):
    model.train()
    total_loss = 0

    for batch in loader:
        features = batch["features"].to(device)  # (B, n_conf, D)
        targets = batch["target"].to(device)

        optimizer.zero_grad()

        # Compute mu and sigma
        mu = features.mean(dim=1)  # (B, D)

        if use_sigma:
            # Compute covariance
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


def evaluate(model, loader, device, use_sigma=False):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            target = batch["target"]

            mu = features.mean(dim=1)

            if use_sigma:
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
):
    """Run single experiment."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Split data
    mol_ids = list(data.keys())
    train_ids, test_ids = train_test_split(mol_ids, test_size=0.2, random_state=seed)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.125, random_state=seed)

    # Create datasets (using precomputed features)
    train_dataset = KrakenDataset(data, train_ids, target_name, features_cache, weights_cache)
    val_dataset = KrakenDataset(data, val_ids, target_name, features_cache, weights_cache)
    test_dataset = KrakenDataset(data, test_ids, target_name, features_cache, weights_cache)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Get feature dim
    feat_dim = train_dataset.features[0].shape[1]

    # Create model
    use_sigma = model_name.startswith("dko_")

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
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, use_sigma)
        val_metrics = evaluate(model, val_loader, device, use_sigma)

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
    test_metrics = evaluate(model, test_loader, device, use_sigma)

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
    parser.add_argument("--feature-type", choices=["fingerprint", "3d"], default="fingerprint")
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

    # Precompute features once
    features_cache, weights_cache = precompute_kraken_features(data, args.feature_type)

    print(f"\nDevice: {device}")
    print(f"Targets: {args.targets}")
    print(f"Models: {args.models}")
    print(f"Seeds: {args.seeds}")
    print(f"Feature type: {args.feature_type}")
    print("=" * 70)

    results = []

    for target in args.targets:
        print(f"\n=== {target} ===")
        for model_name in args.models:
            for seed in args.seeds:
                print(f"  {model_name} seed={seed}...", end=" ", flush=True)
                try:
                    result = run_experiment(data, target, model_name, seed, device, features_cache, weights_cache)
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
