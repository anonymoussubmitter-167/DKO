#!/usr/bin/env python
"""
Experiment K — Curriculum Learning for DKO Variants.

Two-phase training:
  Phase 1 (100 epochs): Train with sigma forced to None (first-order only)
  Phase 2 (200 epochs): Load checkpoint, train with sigma at reduced LR (0.1x)

This should give a better initialization for second-order features by first
learning a good mu representation, then carefully adding sigma.

Usage:
    python scripts/run_curriculum.py --model dko_eigenspectrum --dataset esol --device cuda:0
    python scripts/run_curriculum.py --model dko_residual --dataset qm9_gap --seeds 42 123 456
"""

import argparse
import copy
import json
import pickle
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from dko.models.dko_variants import (
    DKOEigenspectrum, DKOScalarInvariants, DKOLowRank,
    DKOGatedFusion, DKOResidual, DKOCrossAttention, DKOSCCRouter,
)
from dko.training.trainer import Trainer
from dko.data.datasets import create_dataloaders_from_precomputed


MODEL_CLASSES = {
    'dko_eigenspectrum': DKOEigenspectrum,
    'dko_invariants': DKOScalarInvariants,
    'dko_lowrank': DKOLowRank,
    'dko_gated': DKOGatedFusion,
    'dko_residual': DKOResidual,
    'dko_crossattn': DKOCrossAttention,
    'dko_router': DKOSCCRouter,
}

DATASETS = ['esol', 'qm9_gap', 'qm9_lumo', 'lipophilicity']


class SigmaMaskingWrapper(nn.Module):
    """Wrapper that forces sigma=None during Phase 1 (first-order only)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, mu, sigma=None, fit_pca=False):
        return self.model(mu, sigma=None, fit_pca=fit_pca)


def run_curriculum_experiment(
    model_name: str,
    dataset_name: str,
    seed: int,
    device: str,
    phase1_epochs: int = 100,
    phase2_epochs: int = 200,
    phase2_lr_factor: float = 0.1,
    output_dir: str = "results/curriculum",
):
    """Run a single curriculum learning experiment."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    experiment_name = f"{dataset_name}_{model_name}_seed{seed}"
    print(f"\n{'='*60}")
    print(f"Curriculum: {experiment_name}")
    print(f"{'='*60}")

    # Load data
    train_loader, val_loader, test_loader = create_dataloaders_from_precomputed(
        dataset_name, batch_size=32, num_workers=4,
    )

    # Get feature dim
    sample = next(iter(train_loader))
    feature_dim = sample['features'].shape[-1]

    # Create model
    model_class = MODEL_CLASSES[model_name]
    model = model_class(feature_dim=feature_dim, output_dim=1)

    # =========================================================================
    # Phase 1: First-order only (sigma masked to None)
    # =========================================================================
    print(f"\n--- Phase 1: First-order only ({phase1_epochs} epochs) ---")

    wrapped_model = SigmaMaskingWrapper(model)
    trainer_p1 = Trainer(
        model=wrapped_model,
        task='regression',
        learning_rate=1e-4,
        weight_decay=1e-5,
        max_epochs=phase1_epochs,
        early_stopping_patience=30,
        use_mixed_precision=False,
        device=device,
        checkpoint_dir=str(output_path / experiment_name / 'phase1'),
        verbose=True,
    )

    trainer_p1.fit(train_loader, val_loader)

    # Save Phase 1 checkpoint (unwrap the model)
    phase1_ckpt_path = output_path / experiment_name / 'phase1_checkpoint.pt'
    torch.save(model.state_dict(), phase1_ckpt_path)
    print(f"Phase 1 checkpoint saved: {phase1_ckpt_path}")

    # Get Phase 1 best val loss for comparison
    phase1_best_val = trainer_p1.early_stopping.best_score

    # =========================================================================
    # Phase 2: Full model with sigma, reduced LR
    # =========================================================================
    print(f"\n--- Phase 2: Full model with sigma ({phase2_epochs} epochs, "
          f"LR factor={phase2_lr_factor}) ---")

    # Reload model from Phase 1 checkpoint
    model_p2 = model_class(feature_dim=feature_dim, output_dim=1)
    model_p2.load_state_dict(torch.load(phase1_ckpt_path, weights_only=True))

    trainer_p2 = Trainer(
        model=model_p2,
        task='regression',
        learning_rate=1e-4 * phase2_lr_factor,  # Reduced LR
        weight_decay=1e-5,
        max_epochs=phase2_epochs,
        early_stopping_patience=30,
        use_mixed_precision=False,
        device=device,
        checkpoint_dir=str(output_path / experiment_name / 'phase2'),
        verbose=True,
    )

    trainer_p2.fit(train_loader, val_loader)

    # Get Phase 2 best val loss
    phase2_best_val = trainer_p2.early_stopping.best_score

    # =========================================================================
    # Evaluate on test set
    # =========================================================================
    print(f"\n--- Test evaluation ---")
    model_p2.eval()
    model_p2.to(device)

    from dko.training.evaluator import Evaluator
    evaluator = Evaluator(task_type='regression', device=device)
    test_metrics = evaluator.evaluate(model_p2, test_loader)

    print(f"Phase 1 best val loss: {phase1_best_val:.4f}")
    print(f"Phase 2 best val loss: {phase2_best_val:.4f}")
    print(f"Test metrics: {test_metrics}")

    result = {
        'experiment': experiment_name,
        'model': model_name,
        'dataset': dataset_name,
        'seed': seed,
        'phase1_epochs': phase1_epochs,
        'phase2_epochs': phase2_epochs,
        'phase2_lr_factor': phase2_lr_factor,
        'phase1_best_val': float(phase1_best_val) if phase1_best_val is not None else None,
        'phase2_best_val': float(phase2_best_val) if phase2_best_val is not None else None,
        'test_metrics': test_metrics,
    }

    # Save result
    result_path = output_path / experiment_name / 'result.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    return result


def main():
    parser = argparse.ArgumentParser(description="Curriculum Learning (Experiment K)")
    parser.add_argument("--model", type=str, default=None,
                        choices=list(MODEL_CLASSES.keys()),
                        help="Model to train (default: dko_eigenspectrum)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Multiple models to train")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Single dataset")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Multiple datasets")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--phase1-epochs", type=int, default=100)
    parser.add_argument("--phase2-epochs", type=int, default=200)
    parser.add_argument("--phase2-lr-factor", type=float, default=0.1)
    parser.add_argument("--output-dir", type=str, default="results/curriculum")
    args = parser.parse_args()

    # Determine models and datasets
    if args.models:
        models = args.models
    elif args.model:
        models = [args.model]
    else:
        models = ['dko_eigenspectrum', 'dko_residual']  # Default: best candidates

    if args.datasets:
        datasets = args.datasets
    elif args.dataset:
        datasets = [args.dataset]
    else:
        datasets = DATASETS

    print("Curriculum Learning — Experiment K")
    print(f"Models: {models}")
    print(f"Datasets: {datasets}")
    print(f"Seeds: {args.seeds}")
    print(f"Phase 1: {args.phase1_epochs} epochs (first-order only)")
    print(f"Phase 2: {args.phase2_epochs} epochs (full, LR*{args.phase2_lr_factor})")

    all_results = []

    for model_name in models:
        for dataset_name in datasets:
            for seed in args.seeds:
                try:
                    result = run_curriculum_experiment(
                        model_name=model_name,
                        dataset_name=dataset_name,
                        seed=seed,
                        device=args.device,
                        phase1_epochs=args.phase1_epochs,
                        phase2_epochs=args.phase2_epochs,
                        phase2_lr_factor=args.phase2_lr_factor,
                        output_dir=args.output_dir,
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"ERROR in {model_name}/{dataset_name}/seed{seed}: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results.append({
                        'model': model_name,
                        'dataset': dataset_name,
                        'seed': seed,
                        'error': str(e),
                    })

    # Summary
    print(f"\n{'='*70}")
    print("CURRICULUM LEARNING SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'Dataset':<15} {'Seed':>6} {'P1 Val':>10} {'P2 Val':>10} {'Test RMSE':>10}")
    print(f"{'-'*25} {'-'*15} {'-'*6} {'-'*10} {'-'*10} {'-'*10}")

    for r in all_results:
        if 'error' in r:
            print(f"{r['model']:<25} {r['dataset']:<15} {r['seed']:>6} {'ERROR':>10}")
            continue
        p1 = r.get('phase1_best_val')
        p2 = r.get('phase2_best_val')
        rmse = r.get('test_metrics', {}).get('rmse')
        print(f"{r['model']:<25} {r['dataset']:<15} {r['seed']:>6} "
              f"{p1:>10.4f} {p2:>10.4f} "
              f"{rmse:>10.4f}" if rmse else "N/A")

    # Save all results
    output_path = Path(args.output_dir) / 'curriculum_results.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to: {output_path}")


if __name__ == "__main__":
    main()
