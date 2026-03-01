"""
Run 80/20 decomposition study using precomputed conformers.

Computes contribution of:
- First-order (μ only): DKO-FirstOrder
- Second-order (μ + σ): DKO-Full

Shows: percentage of signal from first-order vs second-order.
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
import time
from torch.utils.data import Dataset, DataLoader

from dko.models.dko import DKO, DKOFirstOrder
from dko.training.trainer import Trainer
from dko.training.evaluator import Evaluator
from dko.data.features import AugmentedBasisConstructor

print('=' * 70)
print('80/20 DECOMPOSITION STUDY')
print('=' * 70)
print('Testing: How much signal comes from first-order (μ) vs second-order (σ)?')

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

def prepare_data(data_dict, fixed_dim=256):
    """Convert conformer data to mu/sigma/labels."""
    features_list = data_dict['features']
    labels_list = data_dict['labels']
    weights_list = data_dict.get('boltzmann_weights', [None] * len(features_list))

    mus, sigmas, labels = [], [], []

    for i in range(len(features_list)):
        mol_features = features_list[i]
        label = labels_list[i]
        weights = weights_list[i] if weights_list[i] is not None else None

        if len(mol_features) < 1:
            continue

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

        try:
            basis = basis_constructor.construct([f for f in features], weights)
            mus.append(basis.mean)
            sigmas.append(basis.second_order)
            labels.append(float(label))
        except Exception:
            continue

    return np.array(mus), np.array(sigmas), np.array(labels)

def run_decomposition(dataset_name):
    """Run decomposition for a single dataset."""
    dataset_path = data_dir / dataset_name

    if not (dataset_path / 'train.pkl').exists():
        return None

    # Load data
    with open(dataset_path / 'train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open(dataset_path / 'val.pkl', 'rb') as f:
        val_data = pickle.load(f)
    with open(dataset_path / 'test.pkl', 'rb') as f:
        test_data = pickle.load(f)

    # Check task type
    unique_labels = set(train_data['labels'])
    is_classification = len(unique_labels) <= 2 and all(l in [0, 1, 0.0, 1.0] for l in unique_labels)
    task_type = 'classification' if is_classification else 'regression'

    if is_classification:
        return {'skip': True, 'reason': 'Classification - decomposition less meaningful'}

    # Prepare data
    train_mu, train_sigma, train_y = prepare_data(train_data, FIXED_DIM)
    val_mu, val_sigma, val_y = prepare_data(val_data, FIXED_DIM)
    test_mu, test_sigma, test_y = prepare_data(test_data, FIXED_DIM)

    if len(train_mu) < 50:
        return {'skip': True, 'reason': 'Insufficient data'}

    D = train_mu.shape[1]

    # Normalize
    mu_mean = train_mu.mean(axis=0)
    mu_std = train_mu.std(axis=0) + 1e-8
    train_mu = (train_mu - mu_mean) / mu_std
    val_mu = (val_mu - mu_mean) / mu_std
    test_mu = (test_mu - mu_mean) / mu_std

    y_mean = train_y.mean()
    y_std = train_y.std() + 1e-8
    train_y_norm = (train_y - y_mean) / y_std
    val_y_norm = (val_y - y_mean) / y_std
    test_y_norm = (test_y - y_mean) / y_std

    train_loader = DataLoader(MolDataset(train_mu, train_sigma, train_y_norm),
                              batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(MolDataset(val_mu, val_sigma, val_y_norm),
                            batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(MolDataset(test_mu, test_sigma, test_y_norm),
                             batch_size=32, shuffle=False, collate_fn=collate_fn)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    results = {}

    # Train DKO-FirstOrder
    model_fo = DKOFirstOrder(feature_dim=D, output_dim=1, verbose=False)
    trainer_fo = Trainer(
        model=model_fo, task='regression', learning_rate=1e-4,
        weight_decay=1e-4, max_epochs=100, early_stopping_patience=20,
        use_wandb=False, device=device, verbose=False,
    )
    trainer_fo.fit(train_loader, val_loader)
    evaluator = Evaluator(task_type='regression', device=device)
    metrics_fo = evaluator.evaluate(model_fo, test_loader, verbose=False)
    results['first_order'] = {
        'rmse': metrics_fo['rmse'] * y_std,
        'r2': metrics_fo['r2'],
    }

    # Train DKO-Full
    model_full = DKO(feature_dim=D, output_dim=1, use_second_order=True,
                     use_diagonal_sigma=True, verbose=False)
    trainer_full = Trainer(
        model=model_full, task='regression', learning_rate=1e-5,
        weight_decay=1e-4, max_epochs=100, early_stopping_patience=20,
        use_wandb=False, device=device, verbose=False,
    )
    trainer_full.fit(train_loader, val_loader)
    metrics_full = evaluator.evaluate(model_full, test_loader, verbose=False)
    results['full'] = {
        'rmse': metrics_full['rmse'] * y_std,
        'r2': metrics_full['r2'],
    }

    # Compute decomposition
    # Baseline: first-order RMSE
    # Improvement: first_order_rmse - full_rmse
    fo_rmse = results['first_order']['rmse']
    full_rmse = results['full']['rmse']

    improvement = fo_rmse - full_rmse
    if fo_rmse > 0:
        improvement_pct = (improvement / fo_rmse) * 100
    else:
        improvement_pct = 0

    # First-order captures what % of the signal?
    # If full_rmse = 0, first-order = 100%
    # Approximation: first_order_r2 / full_r2
    fo_r2 = max(0, results['first_order']['r2'])
    full_r2 = max(0.001, results['full']['r2'])  # Avoid division by zero

    if full_r2 > 0:
        first_order_pct = (fo_r2 / full_r2) * 100
    else:
        first_order_pct = 100

    # Clamp to reasonable range
    first_order_pct = min(100, max(0, first_order_pct))
    second_order_pct = 100 - first_order_pct

    return {
        'skip': False,
        'first_order_rmse': fo_rmse,
        'full_rmse': full_rmse,
        'first_order_r2': results['first_order']['r2'],
        'full_r2': results['full']['r2'],
        'improvement': improvement,
        'improvement_pct': improvement_pct,
        'first_order_contribution': first_order_pct,
        'second_order_contribution': second_order_pct,
    }

# Datasets to analyze
datasets = ['esol', 'freesolv', 'lipophilicity', 'qm9_gap', 'qm9_homo', 'qm9_lumo']

print(f"\nAnalyzing {len(datasets)} datasets...")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

results = {}

for dataset in datasets:
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset.upper()}")
    print('='*60)

    result = run_decomposition(dataset)

    if result is None:
        print("  Dataset not found")
        continue

    if result.get('skip'):
        print(f"  Skipped: {result.get('reason')}")
        continue

    results[dataset] = result

    print(f"  First-order RMSE: {result['first_order_rmse']:.4f}")
    print(f"  Full RMSE:        {result['full_rmse']:.4f}")
    print(f"  Improvement:      {result['improvement']:.4f} ({result['improvement_pct']:.1f}%)")
    print(f"  First-order R²:   {result['first_order_r2']:.4f}")
    print(f"  Full R²:          {result['full_r2']:.4f}")
    print(f"  ---")
    print(f"  First-order contribution: {result['first_order_contribution']:.1f}%")
    print(f"  Second-order contribution: {result['second_order_contribution']:.1f}%")

# Summary
print('\n' + '=' * 70)
print('DECOMPOSITION SUMMARY')
print('=' * 70)
print(f"{'Dataset':<15} {'FO RMSE':>10} {'Full RMSE':>10} {'Improv':>8} {'FO %':>8} {'SO %':>8}")
print('-' * 70)

fo_contributions = []
so_contributions = []

for name, r in results.items():
    fo_pct = r['first_order_contribution']
    so_pct = r['second_order_contribution']
    fo_contributions.append(fo_pct)
    so_contributions.append(so_pct)

    print(f"{name:<15} {r['first_order_rmse']:>10.4f} {r['full_rmse']:>10.4f} "
          f"{r['improvement_pct']:>7.1f}% {fo_pct:>7.1f}% {so_pct:>7.1f}%")

print('-' * 70)

if fo_contributions:
    avg_fo = np.mean(fo_contributions)
    avg_so = np.mean(so_contributions)
    print(f"{'AVERAGE':<15} {'':>10} {'':>10} {'':>8} {avg_fo:>7.1f}% {avg_so:>7.1f}%")

print('\n' + '=' * 70)
print('80/20 HYPOTHESIS VALIDATION')
print('=' * 70)

if fo_contributions:
    above_80 = sum(1 for x in fo_contributions if x >= 80)
    above_70 = sum(1 for x in fo_contributions if x >= 70)

    print(f"Datasets where first-order >= 80%: {above_80}/{len(fo_contributions)}")
    print(f"Datasets where first-order >= 70%: {above_70}/{len(fo_contributions)}")
    print(f"Average first-order contribution: {avg_fo:.1f}%")

    if avg_fo >= 80:
        print("\n*** 80/20 HYPOTHESIS CONFIRMED ***")
        print("On average, ~80% of signal comes from first-order features (μ).")
    elif avg_fo >= 70:
        print("\n*** 70/30 PATTERN OBSERVED ***")
        print("First-order captures majority but not quite 80%.")
    else:
        print("\n*** HYPOTHESIS NOT STRONGLY SUPPORTED ***")
        print("Second-order may contribute more than expected.")

# Per-dataset interpretation
print('\n' + '=' * 70)
print('PER-DATASET INTERPRETATION')
print('=' * 70)

for name, r in results.items():
    fo_pct = r['first_order_contribution']
    so_pct = r['second_order_contribution']

    if fo_pct >= 95:
        status = "NEGATIVE CONTROL - second-order adds nothing"
    elif fo_pct >= 80:
        status = "80/20 confirmed - first-order dominant"
    elif so_pct > 10:
        status = "POSSIBLE POSITIVE - second-order contributes"
    else:
        status = "First-order sufficient"

    print(f"  {name}: {status}")
