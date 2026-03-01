"""
Test DKO variants on FreeSolv - the dataset showing POSSIBLE_IMPROVEMENT.

FreeSolv (hydration free energy) showed:
- Residual-SCC correlation: r=0.26
- High/Low SCC residual ratio: 1.44x

This suggests second-order might help. Let's test it.
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
print('FREESOLV EXPERIMENT: Testing Second-Order DKO')
print('=' * 70)
print('Hypothesis: Second-order should help on FreeSolv (hydration free energy)')
print('Rationale: Residual diagnostic showed r=0.26, ratio=1.44x')

# Load data
data_dir = project_root / 'data' / 'conformers' / 'freesolv'

with open(data_dir / 'train.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open(data_dir / 'val.pkl', 'rb') as f:
    val_data = pickle.load(f)
with open(data_dir / 'test.pkl', 'rb') as f:
    test_data = pickle.load(f)

print(f"\nTrain: {len(train_data['features'])} molecules")
print(f"Val: {len(val_data['features'])} molecules")
print(f"Test: {len(test_data['features'])} molecules")

# Check conformer counts
n_conf = [len(f) for f in train_data['features']]
print(f"Conformers/mol: min={min(n_conf)}, max={max(n_conf)}, mean={np.mean(n_conf):.1f}")

FIXED_DIM = 256
basis_constructor = AugmentedBasisConstructor()

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

        # Pad/truncate
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

print("\nPreparing data...")
train_mu, train_sigma, train_y = prepare_data(train_data, FIXED_DIM)
val_mu, val_sigma, val_y = prepare_data(val_data, FIXED_DIM)
test_mu, test_sigma, test_y = prepare_data(test_data, FIXED_DIM)

print(f"Train: {len(train_mu)} samples")
print(f"Val: {len(val_mu)} samples")
print(f"Test: {len(test_mu)} samples")
print(f"Label range: [{train_y.min():.2f}, {train_y.max():.2f}] kcal/mol")

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

train_loader = DataLoader(MolDataset(train_mu, train_sigma, train_y_norm),
                          batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(MolDataset(val_mu, val_sigma, val_y_norm),
                        batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(MolDataset(test_mu, test_sigma, test_y_norm),
                         batch_size=32, shuffle=False, collate_fn=collate_fn)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nDevice: {device}")

# Models to compare
models = {
    'DKO_FirstOrder': DKOFirstOrder(feature_dim=D, output_dim=1, verbose=False),
    'DKO_Diagonal': DKO(feature_dim=D, output_dim=1, use_second_order=True,
                        use_diagonal_sigma=True, verbose=False),
    'DKO_Full': DKO(feature_dim=D, output_dim=1, use_second_order=True,
                    use_diagonal_sigma=False, separate_mu_sigma_nets=False, verbose=False),
}

print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)

results = {}

for name, model in models.items():
    print(f"\n--- {name} ---")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Lower LR for second-order
    lr = 1e-5 if 'Full' in name or 'Diagonal' in name else 1e-4

    trainer = Trainer(
        model=model,
        task='regression',
        learning_rate=lr,
        weight_decay=1e-4,
        max_epochs=100,
        early_stopping_patience=20,
        use_wandb=False,
        device=device,
        verbose=False,
    )

    start = time.time()
    history = trainer.fit(train_loader, val_loader)
    train_time = time.time() - start

    evaluator = Evaluator(task_type='regression', device=device)
    test_metrics = evaluator.evaluate(model, test_loader, verbose=False)

    # Denormalize
    rmse = test_metrics.get('rmse', float('nan')) * y_std
    mae = test_metrics.get('mae', float('nan')) * y_std
    r2 = test_metrics.get('r2', float('nan'))

    results[name] = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'time': train_time,
        'epochs': len(history['train_loss']),
    }

    print(f"  Epochs: {results[name]['epochs']}")
    print(f"  Test RMSE: {rmse:.4f} kcal/mol")
    print(f"  Test MAE: {mae:.4f} kcal/mol")
    print(f"  Test R2: {r2:.4f}")

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print(f"{'Model':<20} {'RMSE':>12} {'MAE':>12} {'R2':>10}")
print("-" * 60)
for name, r in results.items():
    print(f"{name:<20} {r['rmse']:>12.4f} {r['mae']:>12.4f} {r['r2']:>10.4f}")
print("-" * 60)

# Analysis
fo_rmse = results['DKO_FirstOrder']['rmse']
fo_r2 = results['DKO_FirstOrder']['r2']

print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

for name in ['DKO_Diagonal', 'DKO_Full']:
    rmse = results[name]['rmse']
    r2 = results[name]['r2']

    rmse_diff = (fo_rmse - rmse) / fo_rmse * 100
    r2_diff = r2 - fo_r2

    if rmse < fo_rmse:
        print(f"\n{name} BEATS FirstOrder!")
        print(f"  RMSE improvement: {rmse_diff:.1f}%")
        print(f"  R2 improvement: {r2_diff:+.4f}")
    else:
        print(f"\n{name} does NOT beat FirstOrder")
        print(f"  RMSE difference: {rmse_diff:.1f}%")
        print(f"  R2 difference: {r2_diff:+.4f}")

# Final verdict
best_model = min(results.keys(), key=lambda k: results[k]['rmse'])
print(f"\nBest model: {best_model} (RMSE={results[best_model]['rmse']:.4f})")

if 'FirstOrder' not in best_model:
    print("\n*** HYPOTHESIS CONFIRMED: Second-order helps on FreeSolv! ***")
    print("This validates the residual diagnostic prediction.")
else:
    print("\n*** HYPOTHESIS NOT CONFIRMED: First-order still best ***")
    print("Second-order doesn't help even on FreeSolv.")
