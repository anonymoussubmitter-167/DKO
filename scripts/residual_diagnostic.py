"""
Residual Analysis Diagnostic for Second-Order DKO.

This diagnostic answers: "Does first-order systematically fail on high-SCC molecules?"

If residuals correlate with SCC, second-order features might help.
This is more informative than raw SCC-label correlation which is confounded.
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import numpy as np
import pickle
from scipy import stats
from torch.utils.data import Dataset, DataLoader

from dko.models.dko import DKOFirstOrder
from dko.training.trainer import Trainer
from dko.data.features import AugmentedBasisConstructor

print('=' * 70)
print('RESIDUAL ANALYSIS DIAGNOSTIC')
print('=' * 70)
print('Testing: Do first-order residuals correlate with conformational complexity?')
print('If yes → second-order might capture what first-order misses')

data_dir = project_root / 'data' / 'conformers'
FIXED_DIM = 256
basis_constructor = AugmentedBasisConstructor()

class MolDataset(Dataset):
    def __init__(self, mu, sigma, labels):
        self.mu = torch.FloatTensor(mu)
        self.sigma = torch.FloatTensor(sigma)
        self.labels = torch.FloatTensor(labels).unsqueeze(1)

    def __len__(self):
        return len(self.mu)

    def __getitem__(self, idx):
        return {'mu': self.mu[idx], 'sigma': self.sigma[idx], 'label': self.labels[idx]}

def collate_fn(batch):
    return {
        'mu': torch.stack([b['mu'] for b in batch]),
        'sigma': torch.stack([b['sigma'] for b in batch]),
        'label': torch.stack([b['label'] for b in batch]),
    }

def prepare_data_with_scc(data_dict, fixed_dim=256):
    """Prepare data and compute SCC for each molecule."""
    features_list = data_dict['features']
    labels_list = data_dict['labels']
    weights_list = data_dict.get('boltzmann_weights', [None] * len(features_list))

    mus, sigmas, labels, scc_values = [], [], [], []

    for i in range(len(features_list)):
        mol_features = features_list[i]
        label = labels_list[i]
        weights = weights_list[i] if weights_list[i] is not None else None

        if len(mol_features) < 1:
            continue

        # Pad/truncate conformers
        fixed_features = []
        for conf_feat in mol_features:
            conf_feat = np.array(conf_feat).flatten()
            if len(conf_feat) >= fixed_dim:
                fixed_features.append(conf_feat[:fixed_dim])
            else:
                padded = np.zeros(fixed_dim)
                padded[:len(conf_feat)] = conf_feat
                fixed_features.append(padded)

        features = np.array(fixed_features)

        if weights is None:
            weights = np.ones(len(features)) / len(features)
        else:
            weights = np.array(weights)
            weights = weights / weights.sum()

        # Compute SCC (sum of weighted variances)
        if len(features) >= 2:
            mean = np.sum(weights[:, np.newaxis] * features, axis=0)
            variances = np.sum(weights[:, np.newaxis] * (features - mean) ** 2, axis=0)
            scc = variances.sum()
        else:
            scc = 0.0

        # Construct basis for DKO
        try:
            basis = basis_constructor.construct([f for f in features], weights)
            mus.append(basis.mean)
            sigmas.append(basis.second_order)
            labels.append(float(label))
            scc_values.append(scc)
        except Exception:
            continue

    return np.array(mus), np.array(sigmas), np.array(labels), np.array(scc_values)

def run_residual_diagnostic(dataset_name):
    """Run residual diagnostic on a dataset."""
    dataset_path = data_dir / dataset_name

    if not (dataset_path / 'train.pkl').exists():
        return None

    # Load data
    with open(dataset_path / 'train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open(dataset_path / 'test.pkl', 'rb') as f:
        test_data = pickle.load(f)

    # Check task type
    unique_labels = set(train_data['labels'])
    is_classification = len(unique_labels) <= 2 and all(l in [0, 1, 0.0, 1.0] for l in unique_labels)

    if is_classification:
        return {'skip': True, 'reason': 'Classification task - residual analysis less meaningful'}

    # Prepare data with SCC
    train_mu, train_sigma, train_y, train_scc = prepare_data_with_scc(train_data, FIXED_DIM)
    test_mu, test_sigma, test_y, test_scc = prepare_data_with_scc(test_data, FIXED_DIM)

    if len(train_mu) < 50 or len(test_mu) < 20:
        return {'skip': True, 'reason': 'Insufficient data'}

    D = train_mu.shape[1]

    # Normalize
    mu_mean = train_mu.mean(axis=0)
    mu_std = train_mu.std(axis=0) + 1e-8
    train_mu_norm = (train_mu - mu_mean) / mu_std
    test_mu_norm = (test_mu - mu_mean) / mu_std

    y_mean = train_y.mean()
    y_std = train_y.std() + 1e-8
    train_y_norm = (train_y - y_mean) / y_std
    test_y_norm = (test_y - y_mean) / y_std

    # Create data loaders
    train_loader = DataLoader(
        MolDataset(train_mu_norm, train_sigma, train_y_norm),
        batch_size=32, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        MolDataset(test_mu_norm, test_sigma, test_y_norm),
        batch_size=32, shuffle=False, collate_fn=collate_fn
    )

    # Train first-order model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DKOFirstOrder(feature_dim=D, output_dim=1, verbose=False)

    trainer = Trainer(
        model=model,
        task='regression',
        learning_rate=1e-4,
        weight_decay=1e-4,
        max_epochs=50,
        early_stopping_patience=10,
        use_wandb=False,
        device=device,
        verbose=False,
    )

    trainer.fit(train_loader, test_loader)

    # Get predictions on test set
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            mu = batch['mu'].to(device)
            sigma = batch['sigma'].to(device)
            preds = model(mu, sigma)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch['label'].numpy())

    preds = np.concatenate(all_preds).flatten()
    labels = np.concatenate(all_labels).flatten()

    # Compute residuals (denormalized)
    preds_denorm = preds * y_std + y_mean
    labels_denorm = labels * y_std + y_mean
    residuals = np.abs(labels_denorm - preds_denorm)

    # Correlate residuals with SCC
    # Use only test samples we have SCC for
    n_test = min(len(residuals), len(test_scc))
    residuals = residuals[:n_test]
    scc = test_scc[:n_test]

    # Filter out zero-SCC molecules (single conformer)
    mask = scc > 0
    if mask.sum() < 10:
        return {'skip': True, 'reason': 'Too few multi-conformer molecules'}

    residuals_filtered = residuals[mask]
    scc_filtered = scc[mask]

    # Compute correlations
    pearson_r, pearson_p = stats.pearsonr(residuals_filtered, scc_filtered)
    spearman_r, spearman_p = stats.spearmanr(residuals_filtered, scc_filtered)

    # Also check: do high-SCC molecules have larger residuals?
    median_scc = np.median(scc_filtered)
    high_scc_residuals = residuals_filtered[scc_filtered > median_scc]
    low_scc_residuals = residuals_filtered[scc_filtered <= median_scc]

    # T-test for difference
    t_stat, t_pvalue = stats.ttest_ind(high_scc_residuals, low_scc_residuals)
    high_scc_mean_residual = high_scc_residuals.mean()
    low_scc_mean_residual = low_scc_residuals.mean()

    # Recommendation
    if pearson_r > 0.2 and pearson_p < 0.05:
        recommendation = "LIKELY_IMPROVEMENT"
        reason = "First-order errors correlate with SCC - second-order may help"
    elif pearson_r > 0.1:
        recommendation = "POSSIBLE_IMPROVEMENT"
        reason = "Weak correlation between errors and SCC"
    else:
        recommendation = "UNLIKELY_IMPROVEMENT"
        reason = "First-order errors don't correlate with SCC"

    return {
        'skip': False,
        'n_test': n_test,
        'n_multiconf': int(mask.sum()),
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'high_scc_mean_residual': float(high_scc_mean_residual),
        'low_scc_mean_residual': float(low_scc_mean_residual),
        'residual_ratio': float(high_scc_mean_residual / low_scc_mean_residual) if low_scc_mean_residual > 0 else float('inf'),
        't_stat': float(t_stat),
        't_pvalue': float(t_pvalue),
        'recommendation': recommendation,
        'reason': reason,
    }

# Run on all datasets
datasets = ['esol', 'freesolv', 'lipophilicity', 'qm9_gap', 'qm9_homo', 'qm9_lumo']
results = {}

for dataset in datasets:
    print(f"\n--- {dataset.upper()} ---")
    result = run_residual_diagnostic(dataset)

    if result is None:
        print("  Dataset not found")
        continue

    if result.get('skip'):
        print(f"  Skipped: {result.get('reason')}")
        continue

    results[dataset] = result
    print(f"  Test samples: {result['n_test']} ({result['n_multiconf']} multi-conformer)")
    print(f"  Residual-SCC correlation: r={result['pearson_r']:.4f} (p={result['pearson_p']:.4f})")
    print(f"  High-SCC mean residual: {result['high_scc_mean_residual']:.4f}")
    print(f"  Low-SCC mean residual:  {result['low_scc_mean_residual']:.4f}")
    print(f"  Ratio (high/low): {result['residual_ratio']:.2f}x")
    print(f"  --> {result['recommendation']}: {result['reason']}")

# Summary
print('\n' + '=' * 70)
print('RESIDUAL DIAGNOSTIC SUMMARY')
print('=' * 70)
print(f"{'Dataset':<15} {'Resid-SCC r':>12} {'High/Low Ratio':>15} {'Recommendation':>20}")
print('-' * 70)

for name, r in sorted(results.items()):
    print(f"{name:<15} {r['pearson_r']:>12.4f} {r['residual_ratio']:>15.2f}x {r['recommendation']:>20}")

print('-' * 70)

# Interpretation
likely = [n for n, r in results.items() if r['recommendation'] == 'LIKELY_IMPROVEMENT']
possible = [n for n, r in results.items() if r['recommendation'] == 'POSSIBLE_IMPROVEMENT']
unlikely = [n for n, r in results.items() if r['recommendation'] == 'UNLIKELY_IMPROVEMENT']

print(f"\nLIKELY_IMPROVEMENT: {likely}")
print(f"POSSIBLE_IMPROVEMENT: {possible}")
print(f"UNLIKELY_IMPROVEMENT: {unlikely}")

print("\n" + "=" * 70)
print("INTERPRETATION:")
print("=" * 70)
print("""
If residuals correlate with SCC:
  → First-order model makes larger errors on high-variance molecules
  → Second-order features might capture what first-order misses
  → Worth trying DKO with second-order

If residuals DON'T correlate with SCC:
  → First-order errors are independent of conformational complexity
  → Second-order features unlikely to help
  → Stick with first-order (simpler, faster)
""")
