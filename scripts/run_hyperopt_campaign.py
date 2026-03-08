#!/usr/bin/env python
"""
Experiment J — Optuna Hyperparameter Tuning Campaign.

Uses existing dko/training/hyperopt.py (TPE sampler, MedianPruner) to
optimize hyperparameters for the top DKO variants and attention baseline.

50 trials per (model, dataset), 100 epochs per trial.
Scope: top 3-4 models from Phase 2 + attention on 4 datasets.

Run after Phase 2 identifies promising variants.

Usage:
    python scripts/run_hyperopt_campaign.py --device cuda:0
    python scripts/run_hyperopt_campaign.py --models dko_eigenspectrum dko_residual attention --datasets esol qm9_gap
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

from dko.data.datasets import create_dataloaders_from_precomputed
from dko.models.dko_variants import (
    DKOEigenspectrum, DKOScalarInvariants, DKOLowRank,
    DKOGatedFusion, DKOResidual, DKOCrossAttention, DKOSCCRouter,
)
from dko.models.attention import AttentionAggregation
from dko.models.dko import DKO, DKOFirstOrder

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


# Model classes for instantiation
MODEL_CLASSES = {
    'dko_eigenspectrum': DKOEigenspectrum,
    'dko_invariants': DKOScalarInvariants,
    'dko_lowrank': DKOLowRank,
    'dko_gated': DKOGatedFusion,
    'dko_residual': DKOResidual,
    'dko_crossattn': DKOCrossAttention,
    'dko_router': DKOSCCRouter,
    'dko': DKO,
    'dko_first_order': DKOFirstOrder,
    'attention': AttentionAggregation,
}

# Search spaces for DKO variants
DKO_VARIANT_SEARCH_SPACE = {
    'learning_rate': {'type': 'log_uniform', 'low': 1e-5, 'high': 1e-3},
    'weight_decay': {'type': 'log_uniform', 'low': 1e-6, 'high': 1e-4},
    'dropout': {'type': 'categorical', 'choices': [0.0, 0.1, 0.2, 0.3]},
    'k': {'type': 'categorical', 'choices': [5, 10, 15, 20]},
}

ATTENTION_SEARCH_SPACE = {
    'learning_rate': {'type': 'log_uniform', 'low': 1e-5, 'high': 1e-3},
    'weight_decay': {'type': 'log_uniform', 'low': 1e-6, 'high': 1e-4},
    'dropout': {'type': 'categorical', 'choices': [0.0, 0.1, 0.2]},
}

DATASETS = ['esol', 'qm9_gap', 'qm9_lumo', 'lipophilicity']


def sample_params(trial, search_space):
    """Sample hyperparameters for a trial."""
    params = {}
    for name, config in search_space.items():
        param_type = config['type']
        if param_type == 'log_uniform':
            params[name] = trial.suggest_float(name, config['low'], config['high'], log=True)
        elif param_type == 'uniform':
            params[name] = trial.suggest_float(name, config['low'], config['high'])
        elif param_type == 'int':
            params[name] = trial.suggest_int(name, config['low'], config['high'])
        elif param_type == 'categorical':
            params[name] = trial.suggest_categorical(name, config['choices'])
    return params


def create_objective(model_name, model_class, feature_dim, train_loader, val_loader, device, max_epochs):
    """Create an Optuna objective function for a given model."""

    is_variant = model_name in [
        'dko_eigenspectrum', 'dko_invariants', 'dko_lowrank',
        'dko_gated', 'dko_residual', 'dko_crossattn', 'dko_router',
    ]
    search_space = DKO_VARIANT_SEARCH_SPACE if is_variant else ATTENTION_SEARCH_SPACE

    def objective(trial):
        params = sample_params(trial, search_space)

        # Build model kwargs
        model_kwargs = {'feature_dim': feature_dim, 'output_dim': 1}
        if 'dropout' in params:
            model_kwargs['dropout'] = params['dropout']
        if 'k' in params and is_variant:
            model_kwargs['k'] = params['k']

        # Create model
        try:
            if model_name == 'attention':
                model = model_class(feature_dim=feature_dim, output_dim=1)
            elif model_name in ['dko', 'dko_first_order']:
                model = model_class(feature_dim=feature_dim, output_dim=1,
                                    kernel_output_dim=64, verbose=False)
            else:
                model = model_class(**model_kwargs)
        except Exception as e:
            raise optuna.TrialPruned(f"Model creation failed: {e}")

        model = model.to(device)

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay'],
        )

        criterion = torch.nn.MSELoss()
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(max_epochs):
            # Train
            model.train()
            train_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                features = batch['features'].to(device)
                labels = batch['label'].to(device) if 'label' in batch else batch['labels'].to(device)
                mask = batch.get('mask')
                if mask is not None:
                    mask = mask.to(device)

                # Compute mu/sigma for DKO variants
                if is_variant or model_name in ['dko', 'dko_first_order']:
                    batch_size, n_conf, feat_dim = features.shape
                    feat_mean = features.mean(dim=1, keepdim=True)
                    feat_std = features.std(dim=1, keepdim=True).clamp(min=1e-6)
                    features_norm = (features - feat_mean) / feat_std

                    if mask is None:
                        mask = torch.ones(batch_size, n_conf, dtype=torch.bool, device=device)
                    valid_counts = mask.sum(dim=1, keepdim=True).float().clamp(min=1)
                    weights = mask.float() / valid_counts
                    weights_expanded = weights.unsqueeze(-1)

                    mu = (features_norm * weights_expanded).sum(dim=1)
                    centered = features_norm - mu.unsqueeze(1)
                    centered = centered * mask.unsqueeze(-1).float()
                    centered = torch.clamp(centered, min=-10.0, max=10.0)
                    weighted_centered = centered * weights_expanded.sqrt()
                    sigma = torch.bmm(weighted_centered.transpose(1, 2), weighted_centered)
                    eye = torch.eye(feat_dim, device=device)
                    sigma = sigma + 1e-2 * eye.unsqueeze(0)

                    optimizer.zero_grad()
                    outputs = model(mu, sigma, fit_pca=(epoch == 0 and n_batches == 0))
                else:
                    optimizer.zero_grad()
                    try:
                        outputs = model(features, mask=mask)
                    except TypeError:
                        outputs = model(features)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]

                loss = criterion(outputs.squeeze(), labels.squeeze())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            # Validate
            model.eval()
            val_loss = 0.0
            n_val = 0

            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(device)
                    labels = batch['label'].to(device) if 'label' in batch else batch['labels'].to(device)
                    mask = batch.get('mask')
                    if mask is not None:
                        mask = mask.to(device)

                    if is_variant or model_name in ['dko', 'dko_first_order']:
                        batch_size, n_conf, feat_dim = features.shape
                        feat_mean = features.mean(dim=1, keepdim=True)
                        feat_std = features.std(dim=1, keepdim=True).clamp(min=1e-6)
                        features_norm = (features - feat_mean) / feat_std

                        if mask is None:
                            mask = torch.ones(batch_size, n_conf, dtype=torch.bool, device=device)
                        valid_counts = mask.sum(dim=1, keepdim=True).float().clamp(min=1)
                        weights = mask.float() / valid_counts
                        weights_expanded = weights.unsqueeze(-1)

                        mu = (features_norm * weights_expanded).sum(dim=1)
                        centered = features_norm - mu.unsqueeze(1)
                        centered = centered * mask.unsqueeze(-1).float()
                        centered = torch.clamp(centered, min=-10.0, max=10.0)
                        weighted_centered = centered * weights_expanded.sqrt()
                        sigma = torch.bmm(weighted_centered.transpose(1, 2), weighted_centered)
                        eye = torch.eye(feat_dim, device=device)
                        sigma = sigma + 1e-2 * eye.unsqueeze(0)

                        outputs = model(mu, sigma)
                    else:
                        try:
                            outputs = model(features, mask=mask)
                        except TypeError:
                            outputs = model(features)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]

                    loss = criterion(outputs.squeeze(), labels.squeeze())
                    val_loss += loss.item()
                    n_val += 1

            val_loss /= max(n_val, 1)

            # Pruning
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    break

        return best_val_loss

    return objective


def run_campaign(model_name, dataset_name, device, n_trials=50, max_epochs=100):
    """Run hyperopt campaign for one (model, dataset) pair."""
    print(f"\n{'='*60}")
    print(f"Hyperopt: {model_name} on {dataset_name}")
    print(f"{'='*60}")

    # Load data
    train_loader, val_loader, test_loader = create_dataloaders_from_precomputed(
        dataset_name, batch_size=32, num_workers=4,
    )

    # Get feature dim
    sample = next(iter(train_loader))
    feature_dim = sample['features'].shape[-1]

    model_class = MODEL_CLASSES[model_name]

    # Create study
    study_name = f"hyperopt_{model_name}_{dataset_name}"
    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=10),
    )

    objective = create_objective(
        model_name, model_class, feature_dim,
        train_loader, val_loader, device, max_epochs,
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Results
    best = study.best_trial
    print(f"\nBest trial #{best.number}: val_loss={best.value:.6f}")
    print(f"Best params: {best.params}")

    return {
        'model': model_name,
        'dataset': dataset_name,
        'best_value': best.value,
        'best_params': best.params,
        'best_trial': best.number,
        'n_trials': len(study.trials),
        'n_pruned': len([t for t in study.trials
                         if t.state == optuna.trial.TrialState.PRUNED]),
    }


def main():
    parser = argparse.ArgumentParser(description="Hyperopt Campaign (Experiment J)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to optimize")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Datasets to optimize on")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="results/hyperopt_campaign")
    args = parser.parse_args()

    if not OPTUNA_AVAILABLE:
        print("ERROR: Optuna is required. Install with: pip install optuna")
        sys.exit(1)

    models = args.models or ['dko_eigenspectrum', 'dko_residual', 'dko_gated', 'attention']
    datasets = args.datasets or DATASETS

    print("Hyperopt Campaign — Experiment J")
    print(f"Models: {models}")
    print(f"Datasets: {datasets}")
    print(f"Trials per (model, dataset): {args.n_trials}")
    print(f"Max epochs per trial: {args.max_epochs}")
    print(f"Device: {args.device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for model_name in models:
        for dataset_name in datasets:
            try:
                result = run_campaign(
                    model_name, dataset_name, args.device,
                    n_trials=args.n_trials, max_epochs=args.max_epochs,
                )
                all_results.append(result)

                # Save incrementally
                with open(output_dir / 'campaign_results.json', 'w') as f:
                    json.dump(all_results, f, indent=2)

            except Exception as e:
                print(f"ERROR: {model_name}/{dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    'model': model_name,
                    'dataset': dataset_name,
                    'error': str(e),
                })

    # Summary
    print(f"\n{'='*70}")
    print("HYPEROPT CAMPAIGN SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'Dataset':<15} {'Best Val Loss':>15} {'Trials':>8}")
    print(f"{'-'*25} {'-'*15} {'-'*15} {'-'*8}")

    for r in all_results:
        if 'error' in r:
            print(f"{r['model']:<25} {r['dataset']:<15} {'ERROR':>15}")
        else:
            print(f"{r['model']:<25} {r['dataset']:<15} "
                  f"{r['best_value']:>15.6f} {r['n_trials']:>8}")

    # Save final results
    with open(output_dir / 'campaign_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_dir / 'campaign_results.json'}")


if __name__ == "__main__":
    main()
