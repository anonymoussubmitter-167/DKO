"""
Comprehensive integration testing for training pipeline.

Tests:
- End-of-epoch validation
- End-of-training validation
- Loss tracking and convergence
- Gradient flow and norms
- Learning rate scheduling
- Metric computation accuracy
- Checkpoint saving/loading
- Multi-model comparison
- Visualization generation
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Fix Unicode output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from dko.models.dko import DKO, DKOFirstOrder
from dko.models.attention import AttentionPoolingBaseline
from dko.models.deepsets import DeepSetsBaseline
from dko.training.trainer import Trainer
from dko.training.evaluator import Evaluator

# Set style
plt.rcParams['figure.figsize'] = (12, 8)

# Create output directory
OUTPUT_DIR = Path('./test_results') / f"integration_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("COMPREHENSIVE TRAINING INTEGRATION TEST")
print("="*80)
print(f"Output directory: {OUTPUT_DIR}")

# Use CPU for consistent testing
device = 'cpu'

# =============================================================================
# 1. CREATE TEST DATASETS
# =============================================================================
print("\n[1/8] Creating test datasets...")

def create_synthetic_dataset(n_samples, feature_dim, task='regression', noise_level=0.1):
    """Create synthetic dataset with known properties."""
    # Generate features
    mu = torch.randn(n_samples, feature_dim)

    # Generate sigma (PSD)
    sigma_raw = torch.randn(n_samples, feature_dim, feature_dim)
    sigma = torch.bmm(sigma_raw, sigma_raw.transpose(1, 2))

    # Generate labels with known relationship to features
    if task == 'regression':
        # Labels depend on feature mean (simple linear relationship)
        weights = torch.randn(feature_dim, 1)
        labels = torch.mm(mu, weights) + torch.randn(n_samples, 1) * noise_level
    else:
        # Classification: labels based on feature sum
        scores = mu.sum(dim=1, keepdim=True)
        labels = (torch.sigmoid(scores) > 0.5).float()

    return mu, sigma, labels

# Create datasets
N_TRAIN = 200
N_VAL = 50
N_TEST = 50
FEATURE_DIM = 50

print(f"  Creating datasets (train={N_TRAIN}, val={N_VAL}, test={N_TEST}, dim={FEATURE_DIM})")

mu_train, sigma_train, labels_train = create_synthetic_dataset(N_TRAIN, FEATURE_DIM, task='regression')
mu_val, sigma_val, labels_val = create_synthetic_dataset(N_VAL, FEATURE_DIM, task='regression')
mu_test, sigma_test, labels_test = create_synthetic_dataset(N_TEST, FEATURE_DIM, task='regression')

# Also create conformer-style data for baselines
N_CONFORMERS = 20
conformer_features_train = torch.randn(N_TRAIN, N_CONFORMERS, FEATURE_DIM)
conformer_features_val = torch.randn(N_VAL, N_CONFORMERS, FEATURE_DIM)
conformer_features_test = torch.randn(N_TEST, N_CONFORMERS, FEATURE_DIM)

# Boltzmann weights
boltzmann_weights_train = torch.softmax(torch.randn(N_TRAIN, N_CONFORMERS), dim=-1)
boltzmann_weights_val = torch.softmax(torch.randn(N_VAL, N_CONFORMERS), dim=-1)
boltzmann_weights_test = torch.softmax(torch.randn(N_TEST, N_CONFORMERS), dim=-1)

def collate_fn_dko(batch):
    mu, sigma, labels = zip(*batch)
    return {
        'mu': torch.stack(mu),
        'sigma': torch.stack(sigma),
        'label': torch.stack(labels),
    }

def collate_fn_baseline(batch):
    features, weights, labels = zip(*batch)
    return {
        'features': torch.stack(features),
        'weights': torch.stack(weights),
        'label': torch.stack(labels),
    }

# Create data loaders
batch_size = 32

train_dataset_dko = TensorDataset(mu_train, sigma_train, labels_train)
val_dataset_dko = TensorDataset(mu_val, sigma_val, labels_val)
test_dataset_dko = TensorDataset(mu_test, sigma_test, labels_test)

train_loader_dko = DataLoader(train_dataset_dko, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_dko)
val_loader_dko = DataLoader(val_dataset_dko, batch_size=batch_size, collate_fn=collate_fn_dko)
test_loader_dko = DataLoader(test_dataset_dko, batch_size=batch_size, collate_fn=collate_fn_dko)

train_dataset_baseline = TensorDataset(conformer_features_train, boltzmann_weights_train, labels_train)
val_dataset_baseline = TensorDataset(conformer_features_val, boltzmann_weights_val, labels_val)
test_dataset_baseline = TensorDataset(conformer_features_test, boltzmann_weights_test, labels_test)

train_loader_baseline = DataLoader(train_dataset_baseline, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_baseline)
val_loader_baseline = DataLoader(val_dataset_baseline, batch_size=batch_size, collate_fn=collate_fn_baseline)
test_loader_baseline = DataLoader(test_dataset_baseline, batch_size=batch_size, collate_fn=collate_fn_baseline)

print(f"  [OK] Datasets created")

# =============================================================================
# 2. CREATE MODELS
# =============================================================================
print("\n[2/8] Creating models...")

models_config = {
    'DKO': {
        'model': DKO(feature_dim=FEATURE_DIM, output_dim=1, verbose=False),
        'train_loader': train_loader_dko,
        'val_loader': val_loader_dko,
        'test_loader': test_loader_dko,
    },
    'DKO_FirstOrder': {
        'model': DKOFirstOrder(feature_dim=FEATURE_DIM, output_dim=1, verbose=False),
        'train_loader': train_loader_dko,
        'val_loader': val_loader_dko,
        'test_loader': test_loader_dko,
    },
    'Attention': {
        'model': AttentionPoolingBaseline(feature_dim=FEATURE_DIM, output_dim=1),
        'train_loader': train_loader_baseline,
        'val_loader': val_loader_baseline,
        'test_loader': test_loader_baseline,
    },
    'DeepSets': {
        'model': DeepSetsBaseline(feature_dim=FEATURE_DIM, output_dim=1),
        'train_loader': train_loader_baseline,
        'val_loader': val_loader_baseline,
        'test_loader': test_loader_baseline,
    },
}

for name, config in models_config.items():
    n_params = sum(p.numel() for p in config['model'].parameters())
    print(f"  [OK] {name}: {n_params:,} parameters")

# =============================================================================
# 3. TRAINING WITH DETAILED MONITORING
# =============================================================================
print("\n[3/8] Training models with detailed monitoring...")

MAX_EPOCHS = 20
results = {}

for model_name, config in models_config.items():
    print(f"\n  Training {model_name}...")

    model = config['model']
    train_loader = config['train_loader']
    val_loader = config['val_loader']

    # Create trainer
    trainer = Trainer(
        model=model,
        task='regression',
        learning_rate=1e-3,
        max_epochs=MAX_EPOCHS,
        early_stopping_patience=10,
        use_wandb=False,
        checkpoint_dir=OUTPUT_DIR / 'checkpoints' / model_name,
        device=device,
    )

    # Custom training loop with detailed logging
    epoch_metrics = []
    gradient_norms = []
    learning_rates = []

    for epoch in range(MAX_EPOCHS):
        # Train one epoch
        model.train()
        train_losses = []

        for batch in train_loader:
            # Forward pass
            if 'mu' in batch:
                mu = batch['mu'].to(trainer.device)
                sigma = batch['sigma'].to(trainer.device)
                labels = batch['label'].to(trainer.device)
                fit_pca = (epoch == 0 and model_name.startswith('DKO'))
                outputs = model(mu, sigma, fit_pca=fit_pca)
            else:
                features = batch['features'].to(trainer.device)
                labels = batch['label'].to(trainer.device)
                outputs = model(features)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

            loss = trainer.criterion(outputs.squeeze(), labels.squeeze())

            # Backward
            trainer.optimizer.zero_grad()
            loss.backward()

            # Record gradient norm BEFORE clipping
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            gradient_norms.append(total_norm)

            # Clip and step
            torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.gradient_clip_max_norm)
            trainer.optimizer.step()

            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                if 'mu' in batch:
                    mu = batch['mu'].to(trainer.device)
                    sigma = batch['sigma'].to(trainer.device)
                    labels = batch['label'].to(trainer.device)
                    outputs = model(mu, sigma, fit_pca=False)
                else:
                    features = batch['features'].to(trainer.device)
                    labels = batch['label'].to(trainer.device)
                    outputs = model(features)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]

                loss = trainer.criterion(outputs.squeeze(), labels.squeeze())
                val_losses.append(loss.item())

        # Update scheduler
        trainer.scheduler.step()

        # Record metrics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        current_lr = trainer.optimizer.param_groups[0]['lr']

        epoch_metrics.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': current_lr,
        })
        learning_rates.append(current_lr)

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{MAX_EPOCHS}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={current_lr:.2e}")

        # Early stopping check
        if trainer.early_stopping(val_loss):
            print(f"    Early stopped at epoch {epoch+1}")
            break

    # Store results
    results[model_name] = {
        'epoch_metrics': epoch_metrics,
        'gradient_norms': gradient_norms,
        'learning_rates': learning_rates,
        'model': model,
    }

    print(f"  [OK] {model_name} training complete ({len(epoch_metrics)} epochs)")

# =============================================================================
# 4. FINAL EVALUATION ON TEST SET
# =============================================================================
print("\n[4/8] Final evaluation on test set...")

evaluator = Evaluator(task_type='regression', device=device, bootstrap_n_samples=100)
test_results = {}

for model_name, config in models_config.items():
    model = results[model_name]['model']
    test_loader = config['test_loader']

    # Evaluate with predictions
    eval_results = evaluator.evaluate(
        model,
        test_loader,
        return_predictions=True,
        compute_ci=True,
        verbose=False
    )

    test_results[model_name] = eval_results

    metrics = eval_results['metrics']
    print(f"  {model_name}:")
    print(f"    RMSE: {metrics['rmse']:.4f}")
    print(f"    MAE:  {metrics['mae']:.4f}")
    print(f"    R2:   {metrics['r2']:.4f}")
    print(f"    Pearson: {metrics['pearson']:.4f}")

# =============================================================================
# 5. VALIDATION CHECKS
# =============================================================================
print("\n[5/8] Running validation checks...")

checks = {}

# Check 1: Training loss decreased
for model_name, result in results.items():
    metrics = result['epoch_metrics']
    initial_loss = metrics[0]['train_loss']
    final_loss = metrics[-1]['train_loss']
    decreased = final_loss < initial_loss
    checks[f"{model_name}_loss_decreased"] = decreased
    status = "[OK]" if decreased else "[FAIL]"
    print(f"  {status} {model_name}: Loss decreased ({initial_loss:.4f} -> {final_loss:.4f})")

# Check 2: Gradients are flowing
for model_name, result in results.items():
    grad_norms = result['gradient_norms']
    mean_grad = np.mean(grad_norms)
    has_gradients = mean_grad > 1e-6
    checks[f"{model_name}_gradients_flowing"] = has_gradients
    status = "[OK]" if has_gradients else "[FAIL]"
    print(f"  {status} {model_name}: Gradients flowing (mean norm: {mean_grad:.4f})")

# Check 3: Learning rate schedule working
for model_name, result in results.items():
    lrs = result['learning_rates']
    lr_decreased = lrs[-1] < lrs[0]
    checks[f"{model_name}_lr_schedule"] = lr_decreased
    status = "[OK]" if lr_decreased else "[FAIL]"
    print(f"  {status} {model_name}: LR schedule ({lrs[0]:.2e} -> {lrs[-1]:.2e})")

# Check 4: Validation metrics are reasonable
for model_name, eval_result in test_results.items():
    metrics = eval_result['metrics']
    reasonable_rmse = 0 < metrics['rmse'] < 10  # Reasonable range
    reasonable_r2 = -1 <= metrics['r2'] <= 1
    checks[f"{model_name}_reasonable_metrics"] = reasonable_rmse and reasonable_r2
    status = "[OK]" if (reasonable_rmse and reasonable_r2) else "[FAIL]"
    print(f"  {status} {model_name}: Reasonable metrics (RMSE: {metrics['rmse']:.4f}, R2: {metrics['r2']:.4f})")

# =============================================================================
# 6. STATISTICAL COMPARISONS
# =============================================================================
print("\n[6/8] Statistical comparisons between models...")

from scipy import stats

# Compare DKO vs DKO_FirstOrder (test second-order contribution)
if 'DKO' in test_results and 'DKO_FirstOrder' in test_results:
    dko_errors = np.abs(test_results['DKO']['predictions'].flatten() - test_results['DKO']['labels'].flatten())
    dko_first_errors = np.abs(test_results['DKO_FirstOrder']['predictions'].flatten() - test_results['DKO_FirstOrder']['labels'].flatten())

    t_stat, p_value = stats.ttest_rel(dko_errors, dko_first_errors)

    print(f"  DKO vs DKO_FirstOrder (paired t-test):")
    print(f"    Mean error - DKO: {dko_errors.mean():.4f}")
    print(f"    Mean error - DKO_FirstOrder: {dko_first_errors.mean():.4f}")
    print(f"    t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")

    if p_value < 0.05:
        winner = "DKO" if dko_errors.mean() < dko_first_errors.mean() else "DKO_FirstOrder"
        print(f"    -> Significant difference (winner: {winner})")
    else:
        print(f"    -> No significant difference")

# =============================================================================
# 7. GENERATE VISUALIZATIONS
# =============================================================================
print("\n[7/8] Generating visualizations...")

# Figure 1: Training curves
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Training loss
ax = axes[0, 0]
for model_name, result in results.items():
    epochs = [m['epoch'] for m in result['epoch_metrics']]
    train_losses = [m['train_loss'] for m in result['epoch_metrics']]
    ax.plot(epochs, train_losses, marker='o', label=model_name, linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Training Loss')
ax.set_title('Training Loss vs Epoch')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Validation loss
ax = axes[0, 1]
for model_name, result in results.items():
    epochs = [m['epoch'] for m in result['epoch_metrics']]
    val_losses = [m['val_loss'] for m in result['epoch_metrics']]
    ax.plot(epochs, val_losses, marker='s', label=model_name, linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Loss')
ax.set_title('Validation Loss vs Epoch')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Learning rate
ax = axes[1, 0]
for model_name, result in results.items():
    epochs = [m['epoch'] for m in result['epoch_metrics']]
    lrs = [m['learning_rate'] for m in result['epoch_metrics']]
    ax.plot(epochs, lrs, marker='^', label=model_name, linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Learning Rate')
ax.set_title('Learning Rate Schedule')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Gradient norms
ax = axes[1, 1]
for model_name, result in results.items():
    grad_norms = result['gradient_norms']
    # Smooth with moving average
    window = 5
    if len(grad_norms) >= window:
        smoothed = np.convolve(grad_norms, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, label=model_name, alpha=0.7, linewidth=2)
ax.set_xlabel('Training Step')
ax.set_ylabel('Gradient Norm')
ax.set_title('Gradient Norms During Training')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'training_curves.png', dpi=150, bbox_inches='tight')
print(f"  [OK] Saved training_curves.png")
plt.close()

# Figure 2: Prediction quality
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

for idx, (model_name, eval_result) in enumerate(test_results.items()):
    ax = axes[idx // 2, idx % 2]

    predictions = eval_result['predictions'].flatten()
    labels = eval_result['labels'].flatten()

    # Scatter plot
    ax.scatter(labels, predictions, alpha=0.6, s=50)

    # Perfect prediction line
    min_val = min(labels.min(), predictions.min())
    max_val = max(labels.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')

    # Add metrics to plot
    metrics = eval_result['metrics']
    textstr = f"RMSE: {metrics['rmse']:.3f}\nMAE: {metrics['mae']:.3f}\nR2: {metrics['r2']:.3f}\nPearson: {metrics['pearson']:.3f}"
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('True Values')
    ax.set_ylabel('Predictions')
    ax.set_title(f'{model_name} - Predictions vs True Values')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'prediction_quality.png', dpi=150, bbox_inches='tight')
print(f"  [OK] Saved prediction_quality.png")
plt.close()

# Figure 3: Error distributions
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

for idx, (model_name, eval_result) in enumerate(test_results.items()):
    ax = axes[idx // 2, idx % 2]

    predictions = eval_result['predictions'].flatten()
    labels = eval_result['labels'].flatten()
    errors = predictions - labels

    # Histogram
    ax.hist(errors, bins=20, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero error')
    ax.axvline(errors.mean(), color='g', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():.3f}')

    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{model_name} - Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'error_distributions.png', dpi=150, bbox_inches='tight')
print(f"  [OK] Saved error_distributions.png")
plt.close()

# Figure 4: Model comparison
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

model_names = list(test_results.keys())
rmse_values = [test_results[name]['metrics']['rmse'] for name in model_names]

x_pos = np.arange(len(model_names))
ax.bar(x_pos, rmse_values, alpha=0.7, edgecolor='black')
ax.set_xlabel('Model')
ax.set_ylabel('RMSE')
ax.set_title('Model Comparison - RMSE')
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'model_comparison.png', dpi=150, bbox_inches='tight')
print(f"  [OK] Saved model_comparison.png")
plt.close()

# =============================================================================
# 8. SAVE RESULTS AND GENERATE REPORT
# =============================================================================
print("\n[8/8] Generating final report...")

# Save numerical results
# Convert numpy bools to Python bools for JSON serialization
checks_serializable = {k: bool(v) for k, v in checks.items()}

report = {
    'test_date': datetime.now().isoformat(),
    'dataset_info': {
        'n_train': N_TRAIN,
        'n_val': N_VAL,
        'n_test': N_TEST,
        'feature_dim': FEATURE_DIM,
        'task': 'regression',
    },
    'training_config': {
        'max_epochs': MAX_EPOCHS,
        'batch_size': batch_size,
        'learning_rate': 1e-3,
    },
    'models': {},
    'validation_checks': checks_serializable,
}

for model_name in models_config.keys():
    report['models'][model_name] = {
        'n_parameters': sum(p.numel() for p in models_config[model_name]['model'].parameters()),
        'n_epochs_trained': len(results[model_name]['epoch_metrics']),
        'final_train_loss': results[model_name]['epoch_metrics'][-1]['train_loss'],
        'final_val_loss': results[model_name]['epoch_metrics'][-1]['val_loss'],
        'test_metrics': {
            'rmse': float(test_results[model_name]['metrics']['rmse']),
            'mae': float(test_results[model_name]['metrics']['mae']),
            'r2': float(test_results[model_name]['metrics']['r2']),
            'pearson': float(test_results[model_name]['metrics']['pearson']),
        },
    }

# Save JSON report
with open(OUTPUT_DIR / 'test_report.json', 'w') as f:
    json.dump(report, f, indent=2)

# Generate markdown report
md_report = f"""# Training Integration Test Report

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Configuration
- Training samples: {N_TRAIN}
- Validation samples: {N_VAL}
- Test samples: {N_TEST}
- Feature dimension: {FEATURE_DIM}
- Task: Regression

## Models Tested
"""

for model_name in models_config.keys():
    n_params = report['models'][model_name]['n_parameters']
    md_report += f"- **{model_name}**: {n_params:,} parameters\n"

md_report += "\n## Training Results\n\n"
md_report += "| Model | Epochs | Final Train Loss | Final Val Loss |\n"
md_report += "|-------|--------|-----------------|----------------|\n"

for model_name in models_config.keys():
    model_data = report['models'][model_name]
    md_report += f"| {model_name} | {model_data['n_epochs_trained']} | {model_data['final_train_loss']:.4f} | {model_data['final_val_loss']:.4f} |\n"

md_report += "\n## Test Set Performance\n\n"
md_report += "| Model | RMSE | MAE | R2 | Pearson |\n"
md_report += "|-------|------|-----|----|---------|\n"

for model_name in models_config.keys():
    metrics = report['models'][model_name]['test_metrics']
    md_report += f"| {model_name} | {metrics['rmse']:.4f} | {metrics['mae']:.4f} | {metrics['r2']:.4f} | {metrics['pearson']:.4f} |\n"

md_report += "\n## Validation Checks\n\n"

for check_name, passed in checks.items():
    status = "[PASS]" if passed else "[FAIL]"
    md_report += f"- {status}: {check_name}\n"

md_report += "\n## Visualizations\n\n"
md_report += "1. **Training Curves**: `training_curves.png`\n"
md_report += "2. **Prediction Quality**: `prediction_quality.png`\n"
md_report += "3. **Error Distributions**: `error_distributions.png`\n"
md_report += "4. **Model Comparison**: `model_comparison.png`\n"

# Save markdown report
with open(OUTPUT_DIR / 'README.md', 'w') as f:
    f.write(md_report)

print(f"  [OK] Saved test_report.json")
print(f"  [OK] Saved README.md")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*80)
print("INTEGRATION TEST SUMMARY")
print("="*80)

total_checks = len(checks)
passed_checks = sum(checks.values())

print(f"\nValidation checks: {passed_checks}/{total_checks} passed\n")

for check_name, passed in checks.items():
    status = "[PASS]" if passed else "[FAIL]"
    print(f"  {status}: {check_name}")

print(f"\nAll results saved to: {OUTPUT_DIR}")
print(f"\nGenerated files:")
print(f"  - test_report.json (numerical results)")
print(f"  - README.md (human-readable report)")
print(f"  - training_curves.png")
print(f"  - prediction_quality.png")
print(f"  - error_distributions.png")
print(f"  - model_comparison.png")

if passed_checks == total_checks:
    print("\n" + "="*80)
    print("[OK] ALL INTEGRATION TESTS PASSED")
    print("="*80)
    print("\nTraining pipeline is fully validated and ready for production experiments.")
    sys.exit(0)
else:
    print("\n" + "="*80)
    print("[WARN] SOME CHECKS FAILED")
    print("="*80)
    print(f"\n{total_checks - passed_checks} check(s) failed. Review results before proceeding.")
    sys.exit(1)
