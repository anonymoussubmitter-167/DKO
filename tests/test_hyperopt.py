"""
Comprehensive tests for the HyperparameterOptimizer module.

Tests search space configuration, Optuna integration, and optimization.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict
import tempfile
from pathlib import Path

# Check if optuna is available
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Import from hyperopt module
from dko.training.hyperopt import (
    HyperparameterOptimizer,
    run_hyperopt,
    create_optuna_study,
    get_search_space,
    DKO_SEARCH_SPACE,
    ATTENTION_SEARCH_SPACE,
    DEEPSETS_SEARCH_SPACE,
    check_optuna,
    OPTUNA_AVAILABLE as MODULE_OPTUNA_AVAILABLE,
)


# =============================================================================
# Test Fixtures
# =============================================================================


def create_test_loaders(
    n_train: int = 64,
    n_val: int = 32,
    feature_dim: int = 50,
    task: str = "regression",
    batch_size: int = 16,
):
    """Create train and validation data loaders for testing."""
    # Training data
    mu_train = torch.randn(n_train, feature_dim)
    sigma_train = torch.randn(n_train, feature_dim, feature_dim)
    sigma_train = torch.bmm(sigma_train, sigma_train.transpose(1, 2))

    if task == "regression":
        labels_train = torch.randn(n_train, 1)
    else:
        labels_train = torch.randint(0, 2, (n_train, 1)).float()

    # Validation data
    mu_val = torch.randn(n_val, feature_dim)
    sigma_val = torch.randn(n_val, feature_dim, feature_dim)
    sigma_val = torch.bmm(sigma_val, sigma_val.transpose(1, 2))

    if task == "regression":
        labels_val = torch.randn(n_val, 1)
    else:
        labels_val = torch.randint(0, 2, (n_val, 1)).float()

    def collate_fn(batch):
        mu, sigma, labels = zip(*batch)
        return {
            'mu': torch.stack(mu),
            'sigma': torch.stack(sigma),
            'label': torch.stack(labels),
        }

    train_dataset = TensorDataset(mu_train, sigma_train, labels_train)
    val_dataset = TensorDataset(mu_val, sigma_val, labels_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

    return train_loader, val_loader


def create_baseline_loaders(
    n_train: int = 64,
    n_val: int = 32,
    feature_dim: int = 50,
    n_conformers: int = 10,
    task: str = "regression",
    batch_size: int = 16,
):
    """Create train and validation loaders with baseline format (features, mask)."""
    # Training data
    features_train = torch.randn(n_train, n_conformers, feature_dim)
    mask_train = torch.ones(n_train, n_conformers)

    if task == "regression":
        labels_train = torch.randn(n_train, 1)
    else:
        labels_train = torch.randint(0, 2, (n_train, 1)).float()

    # Validation data
    features_val = torch.randn(n_val, n_conformers, feature_dim)
    mask_val = torch.ones(n_val, n_conformers)

    if task == "regression":
        labels_val = torch.randn(n_val, 1)
    else:
        labels_val = torch.randint(0, 2, (n_val, 1)).float()

    def collate_fn(batch):
        features, mask, labels = zip(*batch)
        return {
            'features': torch.stack(features),
            'mask': torch.stack(mask),
            'label': torch.stack(labels),
        }

    train_dataset = TensorDataset(features_train, mask_train, labels_train)
    val_dataset = TensorDataset(features_val, mask_val, labels_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

    return train_loader, val_loader


class SimpleModel(nn.Module):
    """Simple model for testing hyperopt."""

    def __init__(self, feature_dim: int, output_dim: int = 1, **kwargs):
        super().__init__()
        self.fc = nn.Linear(feature_dim, output_dim)

    def forward(self, mu, sigma=None, fit_pca=False):
        return self.fc(mu)


class SimpleBaselineModel(nn.Module):
    """Simple baseline model with mean pooling."""

    def __init__(self, feature_dim: int, output_dim: int = 1, **kwargs):
        super().__init__()
        self.fc = nn.Linear(feature_dim, output_dim)

    def forward(self, features, weights=None, mask=None):
        # Mean pooling
        if mask is not None:
            features = features * mask.unsqueeze(-1)
            pooled = features.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            pooled = features.mean(dim=1)
        return self.fc(pooled)


# =============================================================================
# Test Search Spaces
# =============================================================================


class TestSearchSpaces:
    """Tests for search space definitions."""

    def test_dko_search_space_structure(self):
        """Test DKO search space has correct structure."""
        assert 'learning_rate' in DKO_SEARCH_SPACE
        assert 'weight_decay' in DKO_SEARCH_SPACE
        assert 'dropout' in DKO_SEARCH_SPACE
        assert 'pca_variance' in DKO_SEARCH_SPACE
        assert 'kernel_output_dim' in DKO_SEARCH_SPACE
        assert 'branch_hidden_dim' in DKO_SEARCH_SPACE

        # Check types
        assert DKO_SEARCH_SPACE['learning_rate']['type'] == 'log_uniform'
        assert DKO_SEARCH_SPACE['dropout']['type'] == 'categorical'

    def test_attention_search_space_structure(self):
        """Test Attention search space has correct structure."""
        assert 'learning_rate' in ATTENTION_SEARCH_SPACE
        assert 'embed_dim' in ATTENTION_SEARCH_SPACE
        assert 'num_heads' in ATTENTION_SEARCH_SPACE
        assert 'num_attention_layers' in ATTENTION_SEARCH_SPACE

    def test_deepsets_search_space_structure(self):
        """Test DeepSets search space has correct structure."""
        assert 'learning_rate' in DEEPSETS_SEARCH_SPACE
        assert 'encoder_hidden_dim' in DEEPSETS_SEARCH_SPACE
        assert 'decoder_hidden_dim' in DEEPSETS_SEARCH_SPACE

    def test_get_search_space(self):
        """Test get_search_space function."""
        # DKO variants
        assert get_search_space('dko') == DKO_SEARCH_SPACE
        assert get_search_space('DKO') == DKO_SEARCH_SPACE

        # Attention variants
        assert get_search_space('attention') == ATTENTION_SEARCH_SPACE
        assert get_search_space('Attention') == ATTENTION_SEARCH_SPACE
        assert get_search_space('AttentionPoolingBaseline') == ATTENTION_SEARCH_SPACE

        # DeepSets variants
        assert get_search_space('deepsets') == DEEPSETS_SEARCH_SPACE
        assert get_search_space('DeepSets') == DEEPSETS_SEARCH_SPACE
        assert get_search_space('DeepSetsBaseline') == DEEPSETS_SEARCH_SPACE

        # Unknown model defaults to DKO
        assert get_search_space('unknown_model') == DKO_SEARCH_SPACE


# =============================================================================
# Test Optuna Availability Check
# =============================================================================


class TestOptunaAvailability:
    """Tests for Optuna availability checking."""

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
    def test_optuna_available(self):
        """Test that Optuna is detected as available."""
        assert MODULE_OPTUNA_AVAILABLE is True

    @pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
    def test_check_optuna_passes(self):
        """Test check_optuna doesn't raise when Optuna is available."""
        check_optuna()  # Should not raise


# =============================================================================
# Test HyperparameterOptimizer Initialization
# =============================================================================


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestHyperparameterOptimizerInit:
    """Tests for HyperparameterOptimizer initialization."""

    def test_basic_initialization(self):
        """Test basic initialization."""
        train_loader, val_loader = create_test_loaders(n_train=32, n_val=16)

        optimizer = HyperparameterOptimizer(
            model_class=SimpleModel,
            model_name='test',
            task='regression',
            feature_dim=50,
            output_dim=1,
            train_loader=train_loader,
            val_loader=val_loader,
            n_trials=2,
            verbose=False,
        )

        assert optimizer.model_name == 'test'
        assert optimizer.task == 'regression'
        assert optimizer.n_trials == 2
        assert optimizer.feature_dim == 50
        assert optimizer.output_dim == 1

    def test_custom_search_space(self):
        """Test initialization with custom search space."""
        train_loader, val_loader = create_test_loaders(n_train=32, n_val=16)

        custom_space = {
            'learning_rate': {'type': 'log_uniform', 'low': 1e-4, 'high': 1e-2},
        }

        optimizer = HyperparameterOptimizer(
            model_class=SimpleModel,
            model_name='test',
            task='regression',
            feature_dim=50,
            output_dim=1,
            train_loader=train_loader,
            val_loader=val_loader,
            n_trials=2,
            search_space=custom_space,
            verbose=False,
        )

        assert optimizer.search_space == custom_space

    def test_study_creation(self):
        """Test Optuna study is created correctly."""
        train_loader, val_loader = create_test_loaders(n_train=32, n_val=16)

        optimizer = HyperparameterOptimizer(
            model_class=SimpleModel,
            model_name='test',
            task='regression',
            feature_dim=50,
            output_dim=1,
            train_loader=train_loader,
            val_loader=val_loader,
            n_trials=2,
            verbose=False,
        )

        assert optimizer.study is not None
        assert optimizer.study.direction == optuna.study.StudyDirection.MINIMIZE


# =============================================================================
# Test Parameter Sampling
# =============================================================================


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestParameterSampling:
    """Tests for hyperparameter sampling."""

    def test_sample_log_uniform(self):
        """Test log-uniform parameter sampling."""
        train_loader, val_loader = create_test_loaders(n_train=32, n_val=16)

        optimizer = HyperparameterOptimizer(
            model_class=SimpleModel,
            model_name='test',
            task='regression',
            feature_dim=50,
            output_dim=1,
            train_loader=train_loader,
            val_loader=val_loader,
            n_trials=1,
            search_space={
                'learning_rate': {'type': 'log_uniform', 'low': 1e-5, 'high': 1e-3},
            },
            verbose=False,
        )

        # Sample parameters
        trial = optimizer.study.ask()
        params = optimizer._sample_params(trial)

        assert 'learning_rate' in params
        assert 1e-5 <= params['learning_rate'] <= 1e-3

    def test_sample_categorical(self):
        """Test categorical parameter sampling."""
        train_loader, val_loader = create_test_loaders(n_train=32, n_val=16)

        optimizer = HyperparameterOptimizer(
            model_class=SimpleModel,
            model_name='test',
            task='regression',
            feature_dim=50,
            output_dim=1,
            train_loader=train_loader,
            val_loader=val_loader,
            n_trials=1,
            search_space={
                'dropout': {'type': 'categorical', 'choices': [0.0, 0.1, 0.2]},
            },
            verbose=False,
        )

        trial = optimizer.study.ask()
        params = optimizer._sample_params(trial)

        assert 'dropout' in params
        assert params['dropout'] in [0.0, 0.1, 0.2]

    def test_sample_int(self):
        """Test integer parameter sampling."""
        train_loader, val_loader = create_test_loaders(n_train=32, n_val=16)

        optimizer = HyperparameterOptimizer(
            model_class=SimpleModel,
            model_name='test',
            task='regression',
            feature_dim=50,
            output_dim=1,
            train_loader=train_loader,
            val_loader=val_loader,
            n_trials=1,
            search_space={
                'num_layers': {'type': 'int', 'low': 1, 'high': 5},
            },
            verbose=False,
        )

        trial = optimizer.study.ask()
        params = optimizer._sample_params(trial)

        assert 'num_layers' in params
        assert isinstance(params['num_layers'], int)
        assert 1 <= params['num_layers'] <= 5


# =============================================================================
# Test Optimization
# =============================================================================


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestOptimization:
    """Tests for the optimization process."""

    def test_basic_optimization(self):
        """Test basic optimization run."""
        train_loader, val_loader = create_test_loaders(n_train=32, n_val=16)

        optimizer = HyperparameterOptimizer(
            model_class=SimpleModel,
            model_name='test',
            task='regression',
            feature_dim=50,
            output_dim=1,
            train_loader=train_loader,
            val_loader=val_loader,
            n_trials=2,
            max_epochs=2,
            search_space={
                'learning_rate': {'type': 'log_uniform', 'low': 1e-4, 'high': 1e-3},
            },
            verbose=False,
        )

        results = optimizer.optimize()

        assert 'best_params' in results
        assert 'best_value' in results
        assert 'n_trials' in results
        assert results['n_trials'] == 2

    def test_optimization_with_classification(self):
        """Test optimization for classification task."""
        train_loader, val_loader = create_test_loaders(
            n_train=32, n_val=16, task='classification'
        )

        optimizer = HyperparameterOptimizer(
            model_class=SimpleModel,
            model_name='test',
            task='classification',
            feature_dim=50,
            output_dim=1,
            train_loader=train_loader,
            val_loader=val_loader,
            n_trials=2,
            max_epochs=2,
            search_space={
                'learning_rate': {'type': 'log_uniform', 'low': 1e-4, 'high': 1e-3},
            },
            verbose=False,
        )

        results = optimizer.optimize()

        assert 'best_value' in results
        assert results['best_value'] > 0  # BCE loss should be positive

    def test_early_stopping(self):
        """Test that early stopping works."""
        train_loader, val_loader = create_test_loaders(n_train=32, n_val=16)

        optimizer = HyperparameterOptimizer(
            model_class=SimpleModel,
            model_name='test',
            task='regression',
            feature_dim=50,
            output_dim=1,
            train_loader=train_loader,
            val_loader=val_loader,
            n_trials=2,
            max_epochs=100,  # Many epochs
            early_stopping_patience=2,  # Short patience
            search_space={
                'learning_rate': {'type': 'log_uniform', 'low': 1e-4, 'high': 1e-3},
            },
            verbose=False,
        )

        # Should complete without running all 100 epochs due to early stopping
        results = optimizer.optimize()
        assert 'best_value' in results


# =============================================================================
# Test run_hyperopt Function
# =============================================================================


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestRunHyperopt:
    """Tests for the run_hyperopt convenience function."""

    def test_run_hyperopt_basic(self):
        """Test basic usage of run_hyperopt."""
        train_loader, val_loader = create_test_loaders(n_train=32, n_val=16)

        results = run_hyperopt(
            model_class=SimpleModel,
            model_name='test',
            task='regression',
            feature_dim=50,
            output_dim=1,
            train_loader=train_loader,
            val_loader=val_loader,
            n_trials=2,
            max_epochs=2,
            search_space={
                'learning_rate': {'type': 'log_uniform', 'low': 1e-4, 'high': 1e-3},
            },
            verbose=False,
        )

        assert 'best_params' in results
        assert 'best_value' in results


# =============================================================================
# Test create_optuna_study Function
# =============================================================================


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestCreateOptunaStudy:
    """Tests for the create_optuna_study function."""

    def test_create_study_minimize(self):
        """Test creating a minimization study."""
        study = create_optuna_study(
            study_name='test_study',
            direction='minimize',
        )

        assert study is not None
        assert study.direction == optuna.study.StudyDirection.MINIMIZE

    def test_create_study_maximize(self):
        """Test creating a maximization study."""
        study = create_optuna_study(
            study_name='test_study_max',
            direction='maximize',
        )

        assert study.direction == optuna.study.StudyDirection.MAXIMIZE

    def test_create_study_with_tpe_sampler(self):
        """Test creating a study with TPE sampler."""
        study = create_optuna_study(
            study_name='test_tpe',
            sampler='TPE',
        )

        assert isinstance(study.sampler, optuna.samplers.TPESampler)

    def test_create_study_with_random_sampler(self):
        """Test creating a study with Random sampler."""
        study = create_optuna_study(
            study_name='test_random',
            sampler='Random',
        )

        assert isinstance(study.sampler, optuna.samplers.RandomSampler)

    def test_create_study_with_median_pruner(self):
        """Test creating a study with MedianPruner."""
        study = create_optuna_study(
            study_name='test_median',
            pruner='MedianPruner',
        )

        assert isinstance(study.pruner, optuna.pruners.MedianPruner)

    def test_create_study_with_hyperband_pruner(self):
        """Test creating a study with HyperbandPruner."""
        study = create_optuna_study(
            study_name='test_hyperband',
            pruner='HyperbandPruner',
        )

        assert isinstance(study.pruner, optuna.pruners.HyperbandPruner)


# =============================================================================
# Test Results Saving
# =============================================================================


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestResultsSaving:
    """Tests for saving optimization results."""

    def test_save_results(self):
        """Test saving optimization results."""
        train_loader, val_loader = create_test_loaders(n_train=32, n_val=16)

        optimizer = HyperparameterOptimizer(
            model_class=SimpleModel,
            model_name='test_save',
            task='regression',
            feature_dim=50,
            output_dim=1,
            train_loader=train_loader,
            val_loader=val_loader,
            n_trials=2,
            max_epochs=2,
            search_space={
                'learning_rate': {'type': 'log_uniform', 'low': 1e-4, 'high': 1e-3},
            },
            verbose=False,
        )

        optimizer.optimize()

        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer.save_results(tmpdir)

            # Check files were created
            json_file = Path(tmpdir) / 'test_save_optimization_stats.json'
            assert json_file.exists()

    def test_get_importance(self):
        """Test getting parameter importance."""
        train_loader, val_loader = create_test_loaders(n_train=32, n_val=16)

        optimizer = HyperparameterOptimizer(
            model_class=SimpleModel,
            model_name='test',
            task='regression',
            feature_dim=50,
            output_dim=1,
            train_loader=train_loader,
            val_loader=val_loader,
            n_trials=3,
            max_epochs=2,
            search_space={
                'learning_rate': {'type': 'log_uniform', 'low': 1e-4, 'high': 1e-3},
            },
            verbose=False,
        )

        optimizer.optimize()

        importance = optimizer.get_importance()

        # Should return a dictionary (may be empty with few trials)
        assert isinstance(importance, dict)


# =============================================================================
# Test Batch Data Handling
# =============================================================================


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestBatchDataHandling:
    """Tests for batch data handling in different formats."""

    def test_dko_format_handling(self):
        """Test handling of DKO format (mu, sigma)."""
        train_loader, val_loader = create_test_loaders(n_train=32, n_val=16)

        optimizer = HyperparameterOptimizer(
            model_class=SimpleModel,
            model_name='test',
            task='regression',
            feature_dim=50,
            output_dim=1,
            train_loader=train_loader,
            val_loader=val_loader,
            n_trials=1,
            max_epochs=1,
            search_space={
                'learning_rate': {'type': 'log_uniform', 'low': 1e-4, 'high': 1e-3},
            },
            verbose=False,
        )

        # Test _get_batch_data
        for batch in train_loader:
            data, labels = optimizer._get_batch_data(batch)

            assert data['format'] == 'dko'
            assert 'mu' in data
            assert 'sigma' in data
            break

    def test_baseline_format_handling(self):
        """Test handling of baseline format (features, mask)."""
        train_loader, val_loader = create_baseline_loaders(n_train=32, n_val=16)

        optimizer = HyperparameterOptimizer(
            model_class=SimpleBaselineModel,
            model_name='test',
            task='regression',
            feature_dim=50,
            output_dim=1,
            train_loader=train_loader,
            val_loader=val_loader,
            n_trials=1,
            max_epochs=1,
            search_space={
                'learning_rate': {'type': 'log_uniform', 'low': 1e-4, 'high': 1e-3},
            },
            verbose=False,
        )

        # Test _get_batch_data
        for batch in train_loader:
            data, labels = optimizer._get_batch_data(batch)

            assert data['format'] == 'baseline'
            assert 'features' in data
            assert 'mask' in data
            break


# =============================================================================
# Test Edge Cases
# =============================================================================


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_trial(self):
        """Test optimization with single trial."""
        train_loader, val_loader = create_test_loaders(n_train=32, n_val=16)

        optimizer = HyperparameterOptimizer(
            model_class=SimpleModel,
            model_name='test',
            task='regression',
            feature_dim=50,
            output_dim=1,
            train_loader=train_loader,
            val_loader=val_loader,
            n_trials=1,
            max_epochs=1,
            search_space={
                'learning_rate': {'type': 'log_uniform', 'low': 1e-4, 'high': 1e-3},
            },
            verbose=False,
        )

        results = optimizer.optimize()

        assert results['n_trials'] == 1
        assert 'best_value' in results

    def test_very_small_dataset(self):
        """Test with very small dataset."""
        train_loader, val_loader = create_test_loaders(n_train=8, n_val=4, batch_size=4)

        optimizer = HyperparameterOptimizer(
            model_class=SimpleModel,
            model_name='test',
            task='regression',
            feature_dim=50,
            output_dim=1,
            train_loader=train_loader,
            val_loader=val_loader,
            n_trials=1,
            max_epochs=1,
            search_space={
                'learning_rate': {'type': 'log_uniform', 'low': 1e-4, 'high': 1e-3},
            },
            verbose=False,
        )

        results = optimizer.optimize()

        assert 'best_value' in results


# =============================================================================
# Run Tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
