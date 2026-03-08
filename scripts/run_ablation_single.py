#!/usr/bin/env python3
"""
Single ablation experiment runner.

Called by run_ablation.py with specific config overrides.
Patches trainer behavior at runtime to test different normalization/regularization.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dko.data.datasets import create_dataloaders_from_precomputed, DATASET_CONFIG
from dko.models import DKO, DKOFirstOrder
from dko.training.trainer import Trainer
from dko.training.evaluator import Evaluator, compute_metrics


MODEL_REGISTRY = {
    "dko": DKO,
    "dko_first_order": DKOFirstOrder,
}


def patch_compute_mu_sigma(trainer_or_evaluator, norm_dim, sigma_reg):
    """
    Monkey-patch _compute_mu_sigma to use the specified normalization and regularization.

    norm_dim: "1" for conformer-only, "1_2" for conformer+feature (broken)
    sigma_reg: float regularization value for diagonal
    """
    original_method = trainer_or_evaluator._compute_mu_sigma

    def patched_compute_mu_sigma(features, mask=None, weights=None):
        batch_size, n_conf, feat_dim = features.shape

        if torch.isnan(features).any() or torch.isinf(features).any():
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        # Apply the specified normalization
        if norm_dim == "1_2":
            # Broken: normalizes across both conformers AND features
            feat_mean = features.mean(dim=(1, 2), keepdim=True)
            feat_std = features.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        else:
            # Fixed: normalizes across conformers only
            feat_mean = features.mean(dim=1, keepdim=True)
            feat_std = features.std(dim=1, keepdim=True).clamp(min=1e-6)

        features = (features - feat_mean) / feat_std

        if mask is None:
            mask = torch.ones(batch_size, n_conf, dtype=torch.bool, device=features.device)

        if weights is None:
            valid_counts = mask.sum(dim=1, keepdim=True).float().clamp(min=1)
            weights_w = mask.float() / valid_counts
        else:
            weights_w = weights * mask.float()
            weights_w = weights_w / weights_w.sum(dim=1, keepdim=True).clamp(min=1e-8)

        weights_expanded = weights_w.unsqueeze(-1)
        mu = (features * weights_expanded).sum(dim=1)

        centered = features - mu.unsqueeze(1)
        centered = centered * mask.unsqueeze(-1).float()
        centered = torch.clamp(centered, min=-10.0, max=10.0)

        weighted_centered = centered * weights_expanded.sqrt()
        sigma = torch.bmm(weighted_centered.transpose(1, 2), weighted_centered)

        eye = torch.eye(feat_dim, device=sigma.device, dtype=sigma.dtype)
        sigma = sigma + sigma_reg * eye.unsqueeze(0)

        return mu, sigma

    # Handle different signatures (trainer has weights param, evaluator doesn't)
    if hasattr(trainer_or_evaluator, '_compute_mu_sigma'):
        trainer_or_evaluator._compute_mu_sigma = patched_compute_mu_sigma


def patch_evaluator_mu_sigma(evaluator, norm_dim, sigma_reg):
    """Patch evaluator's _compute_mu_sigma with the same override."""
    def patched(features, mask=None):
        batch_size, n_conf, feat_dim = features.shape

        if torch.isnan(features).any() or torch.isinf(features).any():
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        if norm_dim == "1_2":
            feat_mean = features.mean(dim=(1, 2), keepdim=True)
            feat_std = features.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        else:
            feat_mean = features.mean(dim=1, keepdim=True)
            feat_std = features.std(dim=1, keepdim=True).clamp(min=1e-6)

        features = (features - feat_mean) / feat_std

        if mask is None:
            mask = torch.ones(batch_size, n_conf, dtype=torch.bool, device=features.device)

        valid_counts = mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        weights = mask.float() / valid_counts
        weights_expanded = weights.unsqueeze(-1)

        mu = (features * weights_expanded).sum(dim=1)

        centered = features - mu.unsqueeze(1)
        centered = centered * mask.unsqueeze(-1).float()
        centered = torch.clamp(centered, min=-10.0, max=10.0)

        weighted_centered = centered * weights_expanded.sqrt()
        sigma = torch.bmm(weighted_centered.transpose(1, 2), weighted_centered)

        eye = torch.eye(feat_dim, device=sigma.device, dtype=sigma.dtype)
        sigma = sigma + sigma_reg * eye.unsqueeze(0)

        return mu, sigma

    evaluator._compute_mu_sigma = patched


def main():
    parser = argparse.ArgumentParser(description="Run single ablation experiment")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, default="dko")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kernel-output-dim", type=int, default=64)
    parser.add_argument("--norm-dim", type=str, default="1", choices=["1", "1_2"])
    parser.add_argument("--sigma-reg", type=float, default=1e-2)
    parser.add_argument("--mixed-precision", type=str, default="True")
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--output-dir", type=str, default="results/ablation")
    parser.add_argument("--experiment-name", type=str, default=None)
    args = parser.parse_args()

    use_mixed_precision = args.mixed_precision.lower() in ("true", "1", "yes")

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    print(f"Loading dataset: {args.dataset}")
    train_loader, val_loader, test_loader = create_dataloaders_from_precomputed(
        args.dataset, batch_size=32, num_workers=4,
    )

    # Get feature dim
    sample = next(iter(train_loader))
    feature_dim = sample["features"].shape[-1]

    # Determine task type
    task_type = "classification" if args.dataset in ["bace", "bbbp", "herg", "cyp3a4", "tox21"] else "regression"

    # Create model
    model_class = MODEL_REGISTRY[args.model]
    model = model_class(
        feature_dim=feature_dim,
        output_dim=1,
        kernel_output_dim=args.kernel_output_dim,
    )

    # Create trainer
    experiment_name = args.experiment_name or f"{args.dataset}_{args.model}_seed{args.seed}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        task=task_type,
        learning_rate=args.lr,
        weight_decay=1e-5,
        max_epochs=args.max_epochs,
        early_stopping_patience=30,
        gradient_clip_max_norm=1.0,
        device=device,
        use_mixed_precision=use_mixed_precision,
        checkpoint_dir=output_dir / "checkpoints" / experiment_name,
        log_dir=output_dir / "logs" / experiment_name,
        verbose=True,
    )

    # Patch normalization and regularization
    patch_compute_mu_sigma(trainer, args.norm_dim, args.sigma_reg)

    # Train
    print(f"Training with: lr={args.lr}, k_dim={args.kernel_output_dim}, "
          f"norm={args.norm_dim}, sigma_reg={args.sigma_reg}, mixed_precision={use_mixed_precision}")
    history = trainer.fit(train_loader, val_loader)

    # Evaluate
    evaluator = Evaluator(task_type=task_type, device=device)
    patch_evaluator_mu_sigma(evaluator, args.norm_dim, args.sigma_reg)
    test_metrics = evaluator.evaluate(model, test_loader, verbose=False)

    print(f"\nTest metrics: {test_metrics}")

    # Save results
    results = {
        "dataset": args.dataset,
        "model": args.model,
        "seed": args.seed,
        "config": {
            "lr": args.lr,
            "kernel_output_dim": args.kernel_output_dim,
            "norm_dim": args.norm_dim,
            "sigma_reg": args.sigma_reg,
            "mixed_precision": use_mixed_precision,
        },
        "test_metrics": test_metrics,
        "best_epoch": trainer.best_epoch,
        "best_val_loss": float(trainer.best_val_loss),
        "total_epochs": len(history.get("train_loss", [])),
    }

    results_path = output_dir / "logs" / experiment_name / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
