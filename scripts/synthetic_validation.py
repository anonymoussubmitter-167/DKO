#!/usr/bin/env python
"""
Experiment T — Synthetic Validation.

Generates synthetic data where the target depends on both mu and sigma:
    y = W @ mu + alpha * log(1 + trace(sigma))

Trains 4 models (dko, dko_first_order, attention, mean_ensemble)
across 4 alpha values and 3 seeds = 48 runs.

Validates that:
  - DKO variants can capture sigma signal when it exists (alpha > 0)
  - First-order models cannot capture sigma signal
  - Performance gap increases with alpha

Usage:
    python scripts/synthetic_validation.py
    python scripts/synthetic_validation.py --device cuda:0 --seeds 42 123 456
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from dko.models.dko import DKO, DKOFirstOrder
from dko.models.dko_variants import DKOEigenspectrum, DKOResidual
from dko.models.attention import AttentionAggregation
from dko.models.ensemble_baselines import MeanEnsemble
from dko.training.trainer import Trainer


# Synthetic data parameters
D = 50           # Feature dimension
N_CONF = 20      # Conformers per molecule
N_TRAIN = 1000
N_VAL = 200
N_TEST = 200

ALPHAS = [0.0, 0.1, 0.5, 1.0]


class SyntheticMolDataset(Dataset):
    """Synthetic dataset with controllable sigma dependence."""

    def __init__(self, mu, sigma, features, labels, masks):
        self.mu = torch.FloatTensor(mu)
        self.sigma = torch.FloatTensor(sigma)
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels).unsqueeze(1)
        self.masks = torch.BoolTensor(masks)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'mu': self.mu[idx],
            'sigma': self.sigma[idx],
            'features': self.features[idx],
            'mask': self.masks[idx],
            'label': self.labels[idx],
        }


def collate_fn_dko(batch):
    """Collate for DKO models — includes mu, sigma."""
    return {
        'mu': torch.stack([b['mu'] for b in batch]),
        'sigma': torch.stack([b['sigma'] for b in batch]),
        'label': torch.stack([b['label'] for b in batch]),
    }


def collate_fn_baseline(batch):
    """Collate for baseline models — includes features, mask (no mu/sigma)."""
    return {
        'features': torch.stack([b['features'] for b in batch]),
        'mask': torch.stack([b['mask'] for b in batch]),
        'label': torch.stack([b['label'] for b in batch]),
    }


def generate_synthetic_data(n_samples, alpha, seed=42):
    """Generate synthetic molecular data with controllable sigma dependence.

    Vectorized implementation for speed — avoids per-molecule QR/Cholesky loops.
    """
    rng = np.random.RandomState(seed)

    # Fixed weight vector for the linear part
    W = rng.randn(D) * 0.5

    # Batch generate means
    mus = rng.randn(n_samples, D) * 2.0

    # Generate random PSD covariance matrices in batch
    # Instead of QR per molecule (expensive), use random lower-triangular matrices
    # Sigma = L @ L^T gives PSD matrices with controlled eigenvalue spread
    L = rng.randn(n_samples, D, D) * rng.uniform(0.1, 3.0, size=(n_samples, 1, 1))
    # Zero upper triangle to make L lower-triangular
    mask_tril = np.tril(np.ones((D, D)))
    L = L * mask_tril[np.newaxis, :, :]
    # Scale diagonal to control eigenvalue spread
    diag_scale = np.abs(rng.randn(n_samples, D)) + 0.1
    for i in range(D):
        L[:, i, i] = diag_scale[:, i]
    sigmas = np.einsum('bij,bkj->bik', L, L)  # L @ L^T
    sigmas = sigmas + 1e-2 * np.eye(D)[np.newaxis, :, :]  # Regularization

    # Generate conformer features using Cholesky sampling in batch
    # For each molecule: features = mu + L @ z where z ~ N(0, I)
    # Cholesky of sigma = L @ L^T, so we can reuse L directly
    z = rng.randn(n_samples, N_CONF, D)  # Standard normal
    # features[i,j,:] = mu[i,:] + L[i,:,:] @ z[i,j,:]
    features_all = mus[:, np.newaxis, :] + np.einsum('bij,bnj->bni', L, z)

    # Compute targets: y = W @ mu + alpha * log(1 + trace(sigma)) + noise
    traces = np.trace(sigmas, axis1=1, axis2=2)  # (n_samples,)
    labels = mus @ W + alpha * np.log1p(traces) + rng.randn(n_samples) * 0.1

    masks = np.ones((n_samples, N_CONF), dtype=bool)

    return (mus, sigmas, features_all, labels, masks)


def train_and_evaluate(model, train_loader, val_loader, test_loader, device, max_epochs=100):
    """Train model and return test RMSE."""
    trainer = Trainer(
        model=model,
        task='regression',
        learning_rate=1e-3,
        weight_decay=1e-4,
        max_epochs=max_epochs,
        early_stopping_patience=15,
        use_mixed_precision=False,
        device=device,
        verbose=False,
    )

    trainer.fit(train_loader, val_loader)

    # Evaluate on test set
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            labels = batch['label']

            if 'mu' in batch and 'sigma' in batch:
                # DKO-style forward
                mu = batch['mu'].to(device)
                sigma = batch['sigma'].to(device)
                preds = model(mu, sigma)
            else:
                # Baseline model: use features
                features = batch['features'].to(device)
                mask = batch.get('mask')
                if mask is not None:
                    mask = mask.to(device)
                try:
                    preds = model(features, mask=mask)
                except TypeError:
                    preds = model(features)

            if isinstance(preds, tuple):
                preds = preds[0]

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())

    preds = np.concatenate(all_preds).flatten()
    labels = np.concatenate(all_labels).flatten()

    rmse = np.sqrt(np.mean((preds - labels) ** 2))
    mae = np.mean(np.abs(preds - labels))

    return rmse, mae


def create_model(model_name, feature_dim=D):
    """Create a model by name."""
    if model_name == 'dko':
        return DKO(feature_dim=feature_dim, output_dim=1, kernel_output_dim=32, verbose=False)
    elif model_name == 'dko_first_order':
        return DKOFirstOrder(feature_dim=feature_dim, output_dim=1, kernel_output_dim=32, verbose=False)
    elif model_name == 'dko_eigenspectrum':
        return DKOEigenspectrum(feature_dim=feature_dim, output_dim=1, k=10)
    elif model_name == 'dko_residual':
        return DKOResidual(feature_dim=feature_dim, output_dim=1, k=10)
    elif model_name == 'attention':
        return AttentionAggregation(feature_dim=feature_dim, num_outputs=1)
    elif model_name == 'mean_ensemble':
        return MeanEnsemble(feature_dim=feature_dim, num_outputs=1)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Synthetic Validation (Experiment T)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--alphas", nargs="+", type=float, default=None)
    parser.add_argument("--max-epochs", type=int, default=100)
    args = parser.parse_args()

    model_names = args.models or ['dko', 'dko_first_order', 'dko_eigenspectrum', 'dko_residual',
                                   'attention', 'mean_ensemble']
    alphas = args.alphas or ALPHAS

    print("Synthetic Validation — Experiment T")
    print(f"Device: {args.device}")
    print(f"Models: {model_names}")
    print(f"Alphas: {alphas}")
    print(f"Seeds: {args.seeds}")
    print(f"Data: D={D}, N_conf={N_CONF}, Train={N_TRAIN}, Val={N_VAL}, Test={N_TEST}")

    all_results = {}

    for alpha in alphas:
        print(f"\n{'='*70}")
        print(f"Alpha = {alpha}")
        print(f"{'='*70}")

        alpha_results = {}

        # Pre-generate data for all seeds (reuse across models)
        seed_data = {}
        for seed in args.seeds:
            print(f"  Generating data for seed={seed}...", flush=True)
            train_data = generate_synthetic_data(N_TRAIN, alpha, seed=seed)
            val_data = generate_synthetic_data(N_VAL, alpha, seed=seed + 1000)
            test_data = generate_synthetic_data(N_TEST, alpha, seed=seed + 2000)

            train_dataset = SyntheticMolDataset(*train_data)
            val_dataset = SyntheticMolDataset(*val_data)
            test_dataset = SyntheticMolDataset(*test_data)

            seed_data[seed] = (train_dataset, val_dataset, test_dataset)

        for model_name in model_names:
            seed_rmses = []
            seed_maes = []

            # DKO models use mu/sigma; baselines use features/mask
            is_dko = model_name.startswith('dko')
            collate_fn = collate_fn_dko if is_dko else collate_fn_baseline

            for seed in args.seeds:
                print(f"  {model_name} seed={seed}...", end=" ", flush=True)

                train_dataset, val_dataset, test_dataset = seed_data[seed]

                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
                val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

                # Create and train model
                torch.manual_seed(seed)
                np.random.seed(seed)

                model = create_model(model_name)
                rmse, mae = train_and_evaluate(
                    model, train_loader, val_loader, test_loader,
                    args.device, max_epochs=args.max_epochs,
                )

                seed_rmses.append(rmse)
                seed_maes.append(mae)
                print(f"RMSE={rmse:.4f}, MAE={mae:.4f}")

            alpha_results[model_name] = {
                'rmse_mean': float(np.mean(seed_rmses)),
                'rmse_std': float(np.std(seed_rmses)),
                'mae_mean': float(np.mean(seed_maes)),
                'mae_std': float(np.std(seed_maes)),
                'rmse_per_seed': [float(r) for r in seed_rmses],
            }

        all_results[f'alpha_{alpha}'] = alpha_results

    # Print summary table
    print(f"\n{'='*70}")
    print("SYNTHETIC VALIDATION SUMMARY")
    print(f"{'='*70}")

    for alpha in alphas:
        key = f'alpha_{alpha}'
        print(f"\nAlpha = {alpha} (sigma weight in target):")
        print(f"  {'Model':<25} {'RMSE':>12} {'MAE':>12}")
        print(f"  {'-'*25} {'-'*12} {'-'*12}")

        for model_name in model_names:
            if model_name in all_results[key]:
                r = all_results[key][model_name]
                print(f"  {model_name:<25} "
                      f"{r['rmse_mean']:.4f}+/-{r['rmse_std']:.4f} "
                      f"{r['mae_mean']:.4f}+/-{r['mae_std']:.4f}")

    # Analysis: does the gap grow with alpha?
    print(f"\n{'='*70}")
    print("GAP ANALYSIS: DKO vs First-Order across alphas")
    print(f"{'='*70}")

    for alpha in alphas:
        key = f'alpha_{alpha}'
        if 'dko' not in all_results[key] or 'dko_first_order' not in all_results[key]:
            continue
        dko_rmse = all_results[key]['dko']['rmse_mean']
        fo_rmse = all_results[key]['dko_first_order']['rmse_mean']
        gap = fo_rmse - dko_rmse
        pct = 100.0 * gap / fo_rmse if fo_rmse > 0 else 0.0
        print(f"  alpha={alpha}: DKO={dko_rmse:.4f}, FO={fo_rmse:.4f}, "
              f"Gap={gap:.4f} ({pct:.1f}% improvement)")

    # Save results
    output_path = project_root / 'results' / 'synthetic_validation.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
