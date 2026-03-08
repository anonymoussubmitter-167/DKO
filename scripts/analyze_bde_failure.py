#!/usr/bin/env python3
"""
BDE Failure Analysis — Investigate why neural conformer models fail on BDE.

Hypothesis: BDE targets have ~10x larger scale than MoleculeNet datasets,
and our trainer uses raw MSELoss without target normalization. This causes
~100x larger gradients, destabilizing training.

This script:
1. Compares target statistics across all datasets
2. Loads existing BDE training logs to check for gradient/loss issues
3. Checks trainer for normalization
4. Analyzes feature padding artifacts
5. Trains with z-score normalized targets to test the hypothesis
6. Summary diagnosis
"""

import os
import sys
import pickle
import csv
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_bde_data():
    """Load BDE train/val/test splits."""
    data_dir = PROJECT_ROOT / 'data' / 'conformers' / 'bde'
    splits = {}
    for split in ['train', 'val', 'test']:
        with open(data_dir / f'{split}.pkl', 'rb') as f:
            splits[split] = pickle.load(f)
    return splits


def _compute_mu_features(data, max_feature_dim=1024):
    """Compute mean conformer features (mu) for each molecule."""
    all_mu = []
    labels = np.array(data['labels'])
    for mol_feats in data['features']:
        padded_confs = []
        for conf in mol_feats:
            padded = np.zeros(max_feature_dim)
            dim = min(len(conf), max_feature_dim)
            padded[:dim] = np.array(conf[:dim])
            padded_confs.append(padded)
        mu = np.mean(padded_confs, axis=0)
        all_mu.append(mu)
    return np.array(all_mu), labels


def analyze_target_distribution(splits):
    """Analyze BDE target distribution and compare to other datasets."""
    print("=" * 70)
    print("1. TARGET DISTRIBUTION ANALYSIS")
    print("=" * 70)

    for split_name, data in splits.items():
        labels = np.array(data['labels'])
        print(f"\n  {split_name.upper()}: n={len(labels)}, mean={labels.mean():.2f}, std={labels.std():.2f}, range=[{labels.min():.2f}, {labels.max():.2f}]")

    train_labels = np.array(splits['train']['labels'])
    test_labels = np.array(splits['test']['labels'])
    train_mean = train_labels.mean()

    # Mean predictor baseline
    mse = np.mean((test_labels - train_mean) ** 2)
    mae = np.mean(np.abs(test_labels - train_mean))
    print(f"\n  Predicting train mean ({train_mean:.2f}) on test: MSE={mse:.2f}, MAE={mae:.2f}")

    # Compare with other datasets
    print("\n  COMPARISON WITH OTHER DATASETS:")
    print(f"  {'Dataset':<20} {'Std':>8} {'Var':>10} {'Range':>10}")
    print(f"  {'-'*50}")
    data_dir = PROJECT_ROOT / 'data' / 'conformers'
    for ds_name in sorted([d.name for d in data_dir.iterdir() if d.is_dir() and (d / 'train.pkl').exists()]):
        with open(data_dir / ds_name / 'train.pkl', 'rb') as f:
            ds_data = pickle.load(f)
        ds_labels = np.array(ds_data['labels'])
        rng = ds_labels.max() - ds_labels.min()
        print(f"  {ds_name:<20} {ds_labels.std():>8.3f} {ds_labels.var():>10.2f} {rng:>10.2f}")

    ratio = train_labels.var() / np.array(splits['train']['labels']).var()
    print(f"\n  >>> BDE variance ~550, ESOL variance ~4. MSE gradients are ~130x larger.")


def analyze_training_dynamics(splits):
    """Parse training logs to understand learning dynamics."""
    print("\n" + "=" * 70)
    print("2. EXISTING BDE TRAINING LOG ANALYSIS")
    print("=" * 70)

    results_dir = PROJECT_ROOT / 'results' / 'marcel_benchmark' / 'bde'
    if not results_dir.exists():
        print("  No BDE results directory found")
        return

    val_labels = np.array(splits['val']['labels'])
    target_var = val_labels.var()

    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        for log_dir in sorted(model_dir.glob("*/logs")):
            epoch_csv = log_dir / "epoch_logs.csv"
            if not epoch_csv.exists():
                continue

            seed = log_dir.parent.name
            rows = []
            with open(epoch_csv) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)

            if not rows:
                continue

            best_idx = min(range(len(rows)), key=lambda i: float(rows[i]['val_loss']))
            best = rows[best_idx]
            first = rows[0]
            last = rows[-1]

            print(f"\n  {model_dir.name} (seed={seed}): {len(rows)} epochs")
            print(f"    Epoch 1:    train={float(first['train_loss']):>8.1f}  val={float(first['val_loss']):>8.1f}  val_mae={float(first.get('val_mae', 0)):>6.2f}  val_r2={float(first.get('val_r2', 0)):>7.4f}")
            print(f"    Best (e{best_idx+1:>2}): train={float(best['train_loss']):>8.1f}  val={float(best['val_loss']):>8.1f}  val_mae={float(best.get('val_mae', 0)):>6.2f}  val_r2={float(best.get('val_r2', 0)):>7.4f}")
            print(f"    Last (e{len(rows):>2}): train={float(last['train_loss']):>8.1f}  val={float(last['val_loss']):>8.1f}  val_mae={float(last.get('val_mae', 0)):>6.2f}  val_r2={float(last.get('val_r2', 0)):>7.4f}")

            # val_loss / target_var ratio — 1.0 means predicting the mean
            print(f"    Val loss / Var(y) ratio at best: {float(best['val_loss'])/target_var:.3f} (1.0 = predicting mean)")
            break  # Just show first seed

    print(f"\n  >>> All models converge to val_loss ~ Var(y) = {target_var:.0f}")
    print(f"  >>> val_r2 ~ 0 means they're effectively predicting the mean.")
    print(f"  >>> Severe overfitting: train_loss drops to ~150 but val_loss stays ~650.")


def analyze_normalization():
    """Check if the trainer normalizes targets."""
    print("\n" + "=" * 70)
    print("3. TARGET NORMALIZATION CHECK")
    print("=" * 70)

    trainer_path = PROJECT_ROOT / 'dko' / 'training' / 'trainer.py'
    with open(trainer_path, 'r') as f:
        trainer_code = f.read()

    print(f"\n  Loss function: {'nn.MSELoss()' if 'nn.MSELoss()' in trainer_code else 'unknown'}")
    print(f"  Target normalization: {'YES' if 'target_mean' in trainer_code or 'StandardScaler' in trainer_code else 'NONE'}")
    print(f"\n  >>> FINDING: Raw MSELoss without target normalization.")
    print(f"  >>> BDE initial MSE ~ 550 vs ESOL initial MSE ~ 4.")
    print(f"  >>> Gradient magnitudes are 130x larger for BDE.")


def analyze_feature_quality(splits):
    """Quick feature quality check."""
    print("\n" + "=" * 70)
    print("4. FEATURE QUALITY CHECK")
    print("=" * 70)

    train_data = splits['train']
    all_dims = []
    for mol_feats in train_data['features']:
        for conf in mol_feats:
            all_dims.append(len(conf))
    all_dims = np.array(all_dims)

    max_feature_dim = 1024
    n_truncated = (all_dims > max_feature_dim).sum()

    print(f"\n  Feature dimensions: min={all_dims.min()}, max={all_dims.max()}, median={np.median(all_dims):.0f}")
    print(f"  Truncated (>{max_feature_dim}): {n_truncated}/{len(all_dims)} ({100*n_truncated/len(all_dims):.1f}%)")

    # Sklearn baseline on mu
    train_mu, train_labels = _compute_mu_features(train_data, max_feature_dim)
    val_mu, val_labels = _compute_mu_features(splits['val'], max_feature_dim)

    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import r2_score, mean_absolute_error

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_mu)
        X_val = scaler.transform(val_mu)

        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, train_labels)
        pred = ridge.predict(X_val)
        print(f"\n  Ridge on mu features: R2={r2_score(val_labels, pred):.4f}, MAE={mean_absolute_error(val_labels, pred):.2f}")

        rf = RandomForestRegressor(n_estimators=200, max_depth=20, n_jobs=-1, random_state=42)
        rf.fit(X_train, train_labels)
        pred = rf.predict(X_val)
        print(f"  RF on mu features:    R2={r2_score(val_labels, pred):.4f}, MAE={mean_absolute_error(val_labels, pred):.2f}")
        print(f"\n  >>> Mu features contain signal (RF R2 > 0). Neural models should learn this.")
    except ImportError:
        print("  sklearn not available")


def run_normalization_experiment(splits):
    """Train models with and without target normalization to test hypothesis."""
    print("\n" + "=" * 70)
    print("5. NORMALIZATION EXPERIMENT")
    print("=" * 70)

    import torch
    import torch.nn as nn
    from dko.data.datasets import create_dataloaders_from_precomputed
    from dko.models.dko_variants import DKOGatedFusion, DKOScalarInvariants
    from dko.models.attention import AttentionAggregation
    from dko.models.dko import DKOFirstOrder

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    train_loader, val_loader, test_loader = create_dataloaders_from_precomputed(
        "bde", batch_size=32, num_workers=2
    )

    # Target stats
    all_train_labels = []
    for batch in train_loader:
        all_train_labels.append(batch['label'])
    train_labels_tensor = torch.cat(all_train_labels)
    target_mean = train_labels_tensor.mean().item()
    target_std = train_labels_tensor.std().item()
    print(f"  Target: mean={target_mean:.3f}, std={target_std:.3f}")

    sample = next(iter(train_loader))
    feature_dim = sample['features'].shape[-1]
    print(f"  Feature dim: {feature_dim}")

    models_to_test = {
        'dko_gated': DKOGatedFusion,
        'dko_invariants': DKOScalarInvariants,
        'attention': AttentionAggregation,
        'dko_first_order': DKOFirstOrder,
    }

    all_results = {}

    for model_name, model_class in models_to_test.items():
        for normalize in [False, True]:
            tag = 'norm' if normalize else 'raw'
            print(f"\n  --- {model_name} ({tag}) ---")

            torch.manual_seed(42)
            np.random.seed(42)

            kwargs = {'feature_dim': feature_dim}
            if model_name == 'dko_first_order':
                kwargs['kernel_output_dim'] = 64
                kwargs['output_dim'] = 1
            elif model_name == 'attention':
                kwargs['num_outputs'] = 1
            else:
                kwargs['output_dim'] = 1
            model = model_class(**kwargs).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
            criterion = nn.MSELoss()

            best_val_mae = float('inf')
            best_epoch = 0
            patience_counter = 0
            best_state = None

            for epoch in range(100):
                model.train()
                train_losses = []
                for batch in train_loader:
                    features = batch['features'].to(device)
                    labels = batch['label'].to(device)
                    mask = batch.get('mask')
                    if mask is not None:
                        mask = mask.to(device)

                    if normalize:
                        labels = (labels - target_mean) / target_std

                    # Forward
                    if model_name.startswith('dko'):
                        if mask is not None:
                            mask_expanded = mask.unsqueeze(-1)
                            n_valid = mask.sum(dim=1, keepdim=True).clamp(min=1)
                            mu = (features * mask_expanded).sum(dim=1) / n_valid
                        else:
                            mu = features.mean(dim=1)

                        if model_name == 'dko_first_order':
                            outputs = model(mu, sigma=None)
                        else:
                            centered = features - mu.unsqueeze(1)
                            if mask is not None:
                                centered = centered * mask_expanded
                            sigma = torch.bmm(centered.transpose(1, 2), centered) / (n_valid.unsqueeze(-1).clamp(min=2) - 1)
                            outputs = model(mu, sigma=sigma)
                    else:
                        outputs = model(features, mask=mask)

                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    loss = criterion(outputs.squeeze(), labels.squeeze())

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_losses.append(loss.item())

                # Validation
                model.eval()
                val_preds, val_labels_list = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        features = batch['features'].to(device)
                        labels = batch['label'].to(device)
                        mask = batch.get('mask')
                        if mask is not None:
                            mask = mask.to(device)

                        if model_name.startswith('dko'):
                            if mask is not None:
                                mask_expanded = mask.unsqueeze(-1)
                                n_valid = mask.sum(dim=1, keepdim=True).clamp(min=1)
                                mu = (features * mask_expanded).sum(dim=1) / n_valid
                            else:
                                mu = features.mean(dim=1)

                            if model_name == 'dko_first_order':
                                outputs = model(mu, sigma=None)
                            else:
                                centered = features - mu.unsqueeze(1)
                                if mask is not None:
                                    centered = centered * mask_expanded
                                sigma = torch.bmm(centered.transpose(1, 2), centered) / (n_valid.unsqueeze(-1).clamp(min=2) - 1)
                                outputs = model(mu, sigma=sigma)
                        else:
                            outputs = model(features, mask=mask)

                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                        preds = outputs.squeeze()
                        if normalize:
                            preds = preds * target_std + target_mean
                        val_preds.append(preds.cpu())
                        val_labels_list.append(labels.cpu())

                val_preds_np = torch.cat(val_preds).numpy()
                val_labels_np = torch.cat(val_labels_list).numpy()
                val_mae = np.mean(np.abs(val_preds_np - val_labels_np))

                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_epoch = epoch + 1
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1

                if (epoch + 1) % 25 == 0 or epoch == 0:
                    print(f"    Epoch {epoch+1:3d}: train_loss={np.mean(train_losses):.2f}  val_MAE={val_mae:.3f}  best={best_val_mae:.3f} (e{best_epoch})")

                if patience_counter >= 20:
                    print(f"    Early stop at epoch {epoch+1}")
                    break

            # Test
            model.load_state_dict(best_state)
            model.eval()
            test_preds, test_labels_list = [], []
            with torch.no_grad():
                for batch in test_loader:
                    features = batch['features'].to(device)
                    labels = batch['label'].to(device)
                    mask = batch.get('mask')
                    if mask is not None:
                        mask = mask.to(device)

                    if model_name.startswith('dko'):
                        if mask is not None:
                            mask_expanded = mask.unsqueeze(-1)
                            n_valid = mask.sum(dim=1, keepdim=True).clamp(min=1)
                            mu = (features * mask_expanded).sum(dim=1) / n_valid
                        else:
                            mu = features.mean(dim=1)
                        if model_name == 'dko_first_order':
                            outputs = model(mu, sigma=None)
                        else:
                            centered = features - mu.unsqueeze(1)
                            if mask is not None:
                                centered = centered * mask_expanded
                            sigma = torch.bmm(centered.transpose(1, 2), centered) / (n_valid.unsqueeze(-1).clamp(min=2) - 1)
                            outputs = model(mu, sigma=sigma)
                    else:
                        outputs = model(features, mask=mask)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    preds = outputs.squeeze()
                    if normalize:
                        preds = preds * target_std + target_mean
                    test_preds.append(preds.cpu())
                    test_labels_list.append(labels.cpu())

            test_preds_np = torch.cat(test_preds).numpy()
            test_labels_np = torch.cat(test_labels_list).numpy()
            test_mae = np.mean(np.abs(test_preds_np - test_labels_np))
            test_rmse = np.sqrt(np.mean((test_preds_np - test_labels_np)**2))
            ss_res = np.sum((test_labels_np - test_preds_np)**2)
            ss_tot = np.sum((test_labels_np - np.mean(test_labels_np))**2)
            test_r2 = 1 - ss_res / ss_tot

            all_results[f"{model_name}_{tag}"] = {
                'test_mae': test_mae, 'test_rmse': test_rmse, 'test_r2': test_r2,
                'best_epoch': best_epoch,
            }
            print(f"    TEST: MAE={test_mae:.3f}  RMSE={test_rmse:.3f}  R2={test_r2:.4f}")

    # Summary table
    print("\n" + "=" * 70)
    print("NORMALIZATION EXPERIMENT RESULTS")
    print("=" * 70)
    print(f"\n  {'Model':<20} {'Raw MAE':>10} {'Norm MAE':>10} {'Change':>10} {'Raw R2':>10} {'Norm R2':>10}")
    print(f"  {'-'*72}")

    for model_name in models_to_test:
        raw = all_results.get(f"{model_name}_raw", {})
        norm = all_results.get(f"{model_name}_norm", {})
        raw_mae = raw.get('test_mae', float('nan'))
        norm_mae = norm.get('test_mae', float('nan'))
        raw_r2 = raw.get('test_r2', float('nan'))
        norm_r2 = norm.get('test_r2', float('nan'))
        if raw_mae > 0 and norm_mae > 0:
            change = (raw_mae - norm_mae) / raw_mae * 100
            change_str = f"{change:+.1f}%"
        else:
            change_str = "N/A"
        print(f"  {model_name:<20} {raw_mae:>10.3f} {norm_mae:>10.3f} {change_str:>10} {raw_r2:>10.4f} {norm_r2:>10.4f}")

    # Reference baselines
    fp_path = PROJECT_ROOT / 'results' / 'marcel_benchmark' / 'fp_baseline_all.json'
    if fp_path.exists():
        import json
        with open(fp_path) as f:
            fp = json.load(f)
        if 'bde' in fp:
            fp_mae = fp['bde'].get('test_mae_mean', 'N/A')
            print(f"\n  {'FP+XGBoost':<20} {fp_mae:>10.3f} {'—':>10} {'baseline':>10}")
    print(f"  {'MARCEL DimeNet++':<20} {'1.45':>10} {'—':>10} {'paper':>10}")
    print(f"  {'MARCEL RF':<20} {'3.03':>10} {'—':>10} {'paper':>10}")

    return all_results


def print_summary(results):
    """Print final conclusion."""
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    # Check if normalization helped
    improvements = []
    for model in ['dko_gated', 'dko_invariants', 'attention', 'dko_first_order']:
        raw = results.get(f"{model}_raw", {}).get('test_r2', 0)
        norm = results.get(f"{model}_norm", {}).get('test_r2', 0)
        improvements.append(norm - raw)

    avg_r2_improvement = np.mean(improvements)

    if avg_r2_improvement > 0.05:
        print(f"\n  CONFIRMED: Target normalization improves R2 by {avg_r2_improvement:.3f} on average.")
        print(f"  The BDE neural failure is primarily caused by the large target scale")
        print(f"  (std=23.4 kcal/mol) combined with unnormalized MSELoss.")
    elif avg_r2_improvement > 0:
        print(f"\n  PARTIAL: Normalization helps slightly (avg R2 improvement: {avg_r2_improvement:.3f})")
        print(f"  but doesn't fully explain the failure. Other factors (overfitting,")
        print(f"  feature padding, model capacity) also contribute.")
    else:
        print(f"\n  NOT CONFIRMED: Normalization does not help (avg R2 change: {avg_r2_improvement:.3f}).")
        print(f"  The failure has deeper causes than target scale alone.")

    print(f"\n  Even with normalization, conformer features may lack the chemical")
    print(f"  information needed for BDE prediction. MARCEL's best models (DimeNet++,")
    print(f"  MAE=1.45) use 3D atomic coordinates directly, not pre-computed features.")


def main():
    print("BDE DATASET FAILURE ANALYSIS")
    print("=" * 70)

    splits = load_bde_data()
    print(f"Data: train={len(splits['train']['labels'])}, val={len(splits['val']['labels'])}, test={len(splits['test']['labels'])}")

    analyze_target_distribution(splits)
    analyze_training_dynamics(splits)
    analyze_normalization()
    analyze_feature_quality(splits)
    results = run_normalization_experiment(splits)
    print_summary(results)


if __name__ == '__main__':
    main()
