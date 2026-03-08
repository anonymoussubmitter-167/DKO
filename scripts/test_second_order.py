"""Quick test to verify second-order DKO fixes."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time

from dko.models.dko import DKO, DKOFirstOrder
from dko.training.trainer import Trainer
from dko.training.evaluator import Evaluator

print('=' * 70)
print('QUICK DKO SECOND-ORDER FIX TEST')
print('=' * 70)

# Create synthetic data with real-ish properties
np.random.seed(42)
torch.manual_seed(42)

N_TRAIN = 200
N_TEST = 50
D = 100  # Feature dimension

print(f'\nDataset: {N_TRAIN} train, {N_TEST} test, D={D}')

# Create data where second-order (covariance) features are actually predictive
def make_data(n_samples):
    mu = np.random.randn(n_samples, D) * 0.5
    sigma = np.zeros((n_samples, D, D))

    for i in range(n_samples):
        # Random covariance with varying trace
        scale = np.random.uniform(0.5, 2.0)
        raw = np.random.randn(D, D) * 0.1 * scale
        sigma[i] = raw @ raw.T + 0.01 * np.eye(D)

    # Label depends on both mu AND sigma
    mu_signal = mu.mean(axis=1)  # First-order signal
    sigma_signal = np.array([np.trace(s) / D for s in sigma])  # Second-order signal

    # Mix them with noise
    labels = 2.0 * mu_signal + 0.5 * sigma_signal + np.random.randn(n_samples) * 0.1

    return mu, sigma, labels

train_mu, train_sigma, train_y = make_data(N_TRAIN)
test_mu, test_sigma, test_y = make_data(N_TEST)

print(f'Label range: [{train_y.min():.2f}, {train_y.max():.2f}]')
print(f'Sigma trace range: [{train_sigma.trace(axis1=1, axis2=2).min():.2f}, {train_sigma.trace(axis1=1, axis2=2).max():.2f}]')

# Dataset class
class SimpleDataset(Dataset):
    def __init__(self, mu, sigma, labels):
        self.mu = torch.FloatTensor(mu)
        self.sigma = torch.FloatTensor(sigma)
        self.labels = torch.FloatTensor(labels).unsqueeze(1)

    def __len__(self):
        return len(self.mu)

    def __getitem__(self, idx):
        return {
            'mu': self.mu[idx],
            'sigma': self.sigma[idx],
            'label': self.labels[idx],
        }

def collate_fn(batch):
    return {
        'mu': torch.stack([b['mu'] for b in batch]),
        'sigma': torch.stack([b['sigma'] for b in batch]),
        'label': torch.stack([b['label'] for b in batch]),
    }

train_loader = DataLoader(SimpleDataset(train_mu, train_sigma, train_y),
                          batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(SimpleDataset(test_mu, test_sigma, test_y),
                         batch_size=32, shuffle=False, collate_fn=collate_fn)

# Models to test
models = {
    'DKO_FirstOrder': DKOFirstOrder(feature_dim=D, output_dim=1, verbose=False),
    'DKO_DiagonalSigma': DKO(feature_dim=D, output_dim=1, use_second_order=True,
                              use_diagonal_sigma=True, verbose=False),
    'DKO_SeparateNets': DKO(feature_dim=D, output_dim=1, use_second_order=True,
                             separate_mu_sigma_nets=True, verbose=False),
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'\nDevice: {device}')

print('\nTraining models (50 epochs each)...')
results = {}

for name, model in models.items():
    print(f'\n  {name}...', end=' ', flush=True)

    trainer = Trainer(
        model=model,
        task='regression',
        learning_rate=1e-4 if 'SecondOrder' in name else 1e-3,
        weight_decay=1e-4,
        max_epochs=50,
        early_stopping_patience=15,
        use_wandb=False,
        device=device,
        verbose=False,
    )

    start = time.time()
    history = trainer.fit(train_loader, test_loader)
    train_time = time.time() - start

    evaluator = Evaluator(task_type='regression', device=device)
    metrics = evaluator.evaluate(model, test_loader, verbose=False)

    results[name] = {
        'rmse': metrics.get('rmse', float('nan')),
        'r2': metrics.get('r2', float('nan')),
        'time': train_time,
        'epochs': len(history['train_loss']),
    }

    print(f'RMSE={results[name]["rmse"]:.4f}, R2={results[name]["r2"]:.4f}')

print('\n' + '=' * 70)
print('RESULTS')
print('=' * 70)
print(f"{'Model':<30} {'RMSE':>10} {'R2':>10} {'Epochs':>8}")
print('-' * 70)
for name, r in results.items():
    print(f"{name:<30} {r['rmse']:>10.4f} {r['r2']:>10.4f} {r['epochs']:>8}")
print('-' * 70)

# Check if second-order models beat first-order
first_order_rmse = results['DKO_FirstOrder']['rmse']
for name, r in results.items():
    if 'SecondOrder' in name:
        if r['rmse'] < first_order_rmse:
            print(f'{name} beats FirstOrder! (lower RMSE is better)')
        else:
            print(f'{name} does NOT beat FirstOrder yet')

if 'DKO_DiagonalSigma' in results and results['DKO_DiagonalSigma']['r2'] > 0:
    print('\nDiagonal sigma has POSITIVE R2!')
