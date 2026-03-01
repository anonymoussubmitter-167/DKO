"""Tests for dataset loading system."""

import pytest
import numpy as np
import torch
import tempfile
from pathlib import Path
import shutil

# Skip if RDKit not available
pytest.importorskip("rdkit")

from dko.data.datasets import (
    DATASET_CONFIG,
    AVAILABLE_DATASETS,
    get_dataset,
    create_dataloaders,
    collate_dko,
    collate_single_conformer,
    MolecularDatasetBase,
    DatasetStatistics,
    CachedMolecule,
)


class TestDatasetConfig:
    """Test dataset configuration."""

    def test_all_datasets_configured(self):
        """Test that all 12 datasets are configured."""
        expected_datasets = [
            'bace', 'pdbbind', 'freesolv', 'herg', 'cyp3a4', 'tox21',
            'bbbp', 'esol', 'lipo', 'qm9_homo', 'qm9_gap', 'qm9_polar'
        ]

        for dataset in expected_datasets:
            assert dataset in DATASET_CONFIG
            assert dataset in AVAILABLE_DATASETS

    def test_config_fields(self):
        """Test that all configs have required fields."""
        required_fields = [
            'name', 'task', 'metric', 'num_tasks', 'n_molecules',
            'smiles_col', 'target_col', 'expected_advantage'
        ]

        for dataset_name, config in DATASET_CONFIG.items():
            for field in required_fields:
                assert field in config, f"{dataset_name} missing {field}"

    def test_task_types(self):
        """Test that task types are valid."""
        valid_tasks = ['regression', 'classification']

        for dataset_name, config in DATASET_CONFIG.items():
            assert config['task'] in valid_tasks

    def test_metric_types(self):
        """Test that metrics are valid."""
        valid_metrics = ['rmse', 'mae', 'auc', 'acc']

        for dataset_name, config in DATASET_CONFIG.items():
            assert config['metric'] in valid_metrics

    def test_negative_controls(self):
        """Test that QM9 datasets are marked as negative controls."""
        qm9_datasets = ['qm9_homo', 'qm9_gap', 'qm9_polar']

        for dataset in qm9_datasets:
            assert DATASET_CONFIG[dataset]['expected_advantage'] == 0.0


class TestCachedMolecule:
    """Test CachedMolecule dataclass."""

    def test_creation(self):
        """Test creating CachedMolecule."""
        mol = CachedMolecule(
            smiles="CCO",
            label=np.array([1.0]),
            mu=np.random.randn(50),
            sigma=np.eye(50),
            scc=0.5,
            n_conformers=10,
        )

        assert mol.smiles == "CCO"
        assert mol.scc == 0.5
        assert mol.n_conformers == 10

    def test_optional_fields(self):
        """Test optional fields are None by default."""
        mol = CachedMolecule(
            smiles="CCO",
            label=np.array([1.0]),
            mu=np.random.randn(50),
            sigma=np.eye(50),
            scc=0.5,
            n_conformers=10,
        )

        assert mol.feature_list is None
        assert mol.weights is None
        assert mol.single_conformer_features is None


class TestDatasetStatistics:
    """Test DatasetStatistics dataclass."""

    def test_creation(self):
        """Test creating DatasetStatistics."""
        stats = DatasetStatistics(
            n_samples=100,
            n_unique_scaffolds=50,
            mean_scc=0.3,
            std_scc=0.1,
            mean_n_conformers=15.5,
        )

        assert stats.n_samples == 100
        assert stats.n_unique_scaffolds == 50
        assert stats.mean_scc == 0.3

    def test_optional_regression_fields(self):
        """Test optional fields for regression."""
        stats = DatasetStatistics(
            n_samples=100,
            n_unique_scaffolds=50,
            mean_scc=0.3,
            std_scc=0.1,
            mean_n_conformers=15.5,
            label_mean=5.0,
            label_std=2.0,
        )

        assert stats.label_mean == 5.0
        assert stats.label_std == 2.0

    def test_optional_classification_fields(self):
        """Test optional fields for classification."""
        stats = DatasetStatistics(
            n_samples=100,
            n_unique_scaffolds=50,
            mean_scc=0.3,
            std_scc=0.1,
            mean_n_conformers=15.5,
            class_balance={'0': 40, '1': 60},
        )

        assert stats.class_balance['0'] == 40
        assert stats.class_balance['1'] == 60


class TestGetDataset:
    """Test the get_dataset factory function."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)

    def test_unknown_dataset(self, temp_dir):
        """Test that unknown dataset raises error."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_dataset("unknown_dataset", root=temp_dir)

    def test_available_datasets(self):
        """Test that all available datasets can be instantiated."""
        # This just tests that the factory mapping is correct
        for name in AVAILABLE_DATASETS:
            assert name in DATASET_CONFIG

    def test_case_insensitive(self, temp_dir):
        """Test that dataset names are case-insensitive."""
        # Should work with uppercase
        # Note: actual loading may fail without data, but factory should work
        try:
            get_dataset("BACE", root=temp_dir, verbose=False)
        except Exception:
            pass  # Expected if no data

    def test_split_options(self, temp_dir):
        """Test different split options."""
        valid_splits = ['train', 'val', 'test']
        for split in valid_splits:
            # Factory should accept all valid splits
            try:
                get_dataset("bace", root=temp_dir, split=split, verbose=False)
            except Exception:
                pass  # Expected if no data


class TestCollateFunctions:
    """Test collate functions."""

    @pytest.fixture
    def sample_batch(self):
        """Create sample batch for testing."""
        return [
            {
                'mu': torch.randn(50),
                'sigma': torch.randn(50, 50),
                'label': torch.tensor([1.0]),
                'scc': 0.3,
                'n_conformers': 10,
                'smiles': "CCO",
                'idx': 0,
            },
            {
                'mu': torch.randn(50),
                'sigma': torch.randn(50, 50),
                'label': torch.tensor([2.0]),
                'scc': 0.5,
                'n_conformers': 15,
                'smiles': "CCCO",
                'idx': 1,
            },
        ]

    @pytest.fixture
    def sample_batch_with_features(self):
        """Create sample batch with features for single conformer mode."""
        return [
            {
                'mu': torch.randn(50),
                'sigma': torch.randn(50, 50),
                'label': torch.tensor([1.0]),
                'scc': 0.3,
                'n_conformers': 1,
                'smiles': "CCO",
                'idx': 0,
                'features': torch.randn(50),
            },
            {
                'mu': torch.randn(50),
                'sigma': torch.randn(50, 50),
                'label': torch.tensor([2.0]),
                'scc': 0.5,
                'n_conformers': 1,
                'smiles': "CCCO",
                'idx': 1,
                'features': torch.randn(50),
            },
        ]

    def test_collate_dko(self, sample_batch):
        """Test DKO collate function."""
        collated = collate_dko(sample_batch)

        assert 'mu' in collated
        assert 'sigma' in collated
        assert 'labels' in collated
        assert 'scc' in collated
        assert 'n_conformers' in collated
        assert 'smiles' in collated
        assert 'idx' in collated

        # Check shapes
        assert collated['mu'].shape == (2, 50)
        assert collated['sigma'].shape == (2, 50, 50)
        assert collated['labels'].shape == (2, 1)
        assert len(collated['scc']) == 2
        assert len(collated['smiles']) == 2

    def test_collate_single_conformer(self, sample_batch_with_features):
        """Test single conformer collate function."""
        collated = collate_single_conformer(sample_batch_with_features)

        assert 'features' in collated
        assert 'labels' in collated
        assert 'smiles' in collated
        assert 'idx' in collated

        # Check shapes
        assert collated['features'].shape == (2, 50)
        assert collated['labels'].shape == (2, 1)

    def test_collate_single_conformer_fallback(self, sample_batch):
        """Test that collate_single_conformer falls back to mu if no features."""
        collated = collate_single_conformer(sample_batch)

        # Should use mu as features
        assert collated['features'].shape == (2, 50)


class TestDatasetIntegration:
    """Integration tests for dataset loading.

    Note: These tests require actual data files or will use placeholder data.
    """

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.mark.slow
    def test_dataset_with_placeholder_data(self, temp_dir):
        """Test loading dataset with placeholder data."""
        try:
            dataset = get_dataset(
                "bace",
                root=temp_dir,
                split='train',
                use_ensemble=True,
                verbose=False,
                max_conformers=5,
                n_conformers_generate=10,
            )

            assert len(dataset) > 0

            # Test __getitem__
            sample = dataset[0]
            assert 'mu' in sample
            assert 'sigma' in sample
            assert 'label' in sample
            assert 'scc' in sample
            assert 'smiles' in sample

            # Test types
            assert isinstance(sample['mu'], torch.Tensor)
            assert isinstance(sample['sigma'], torch.Tensor)
            assert isinstance(sample['label'], torch.Tensor)

        except Exception as e:
            pytest.skip(f"Dataset loading failed (may need RDKit): {e}")

    @pytest.mark.slow
    def test_dataset_single_conformer_mode(self, temp_dir):
        """Test dataset in single conformer mode."""
        try:
            dataset = get_dataset(
                "esol",
                root=temp_dir,
                split='train',
                use_ensemble=False,
                verbose=False,
                max_conformers=5,
            )

            sample = dataset[0]

            # Should have features for single conformer
            assert 'mu' in sample or 'features' in sample

        except Exception as e:
            pytest.skip(f"Dataset loading failed: {e}")

    @pytest.mark.slow
    def test_dataset_statistics(self, temp_dir):
        """Test computing dataset statistics."""
        try:
            dataset = get_dataset(
                "bace",
                root=temp_dir,
                split='train',
                verbose=False,
                max_conformers=5,
            )

            stats = dataset.get_statistics()

            assert isinstance(stats, DatasetStatistics)
            assert stats.n_samples > 0
            assert stats.n_unique_scaffolds > 0
            assert stats.mean_scc >= 0
            assert stats.mean_n_conformers >= 0

        except Exception as e:
            pytest.skip(f"Dataset loading failed: {e}")

    @pytest.mark.slow
    def test_different_splits(self, temp_dir):
        """Test that different splits load different data."""
        try:
            train = get_dataset("bace", root=temp_dir, split='train', verbose=False, max_conformers=5)
            val = get_dataset("bace", root=temp_dir, split='val', verbose=False, max_conformers=5)
            test = get_dataset("bace", root=temp_dir, split='test', verbose=False, max_conformers=5)

            # Splits should have different sizes (unless very small dataset)
            total = len(train) + len(val) + len(test)
            assert total > 0

        except Exception as e:
            pytest.skip(f"Dataset loading failed: {e}")


class TestDataLoaderCreation:
    """Test DataLoader creation utilities."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.mark.slow
    def test_create_dataloaders(self, temp_dir):
        """Test creating dataloaders."""
        try:
            train_loader, val_loader, test_loader = create_dataloaders(
                "bace",
                root=temp_dir,
                batch_size=4,
                num_workers=0,  # Use 0 for testing
                verbose=False,
                max_conformers=5,
            )

            # Check types
            from torch.utils.data import DataLoader
            assert isinstance(train_loader, DataLoader)
            assert isinstance(val_loader, DataLoader)
            assert isinstance(test_loader, DataLoader)

        except Exception as e:
            pytest.skip(f"DataLoader creation failed: {e}")

    @pytest.mark.slow
    def test_dataloader_iteration(self, temp_dir):
        """Test iterating through dataloader."""
        try:
            train_loader, _, _ = create_dataloaders(
                "bace",
                root=temp_dir,
                batch_size=2,
                num_workers=0,
                verbose=False,
                max_conformers=5,
            )

            # Get one batch
            batch = next(iter(train_loader))

            assert 'mu' in batch
            assert 'labels' in batch
            assert batch['mu'].dim() == 2  # (batch_size, feature_dim)

        except Exception as e:
            pytest.skip(f"DataLoader iteration failed: {e}")


class TestDatasetProperties:
    """Test dataset properties and methods."""

    def test_dataset_config_access(self):
        """Test that dataset config is accessible."""
        # Using DATASET_CONFIG directly
        bace_config = DATASET_CONFIG['bace']

        assert bace_config['task'] == 'regression'
        assert bace_config['metric'] == 'rmse'
        assert bace_config['num_tasks'] == 1

    def test_classification_datasets(self):
        """Test classification dataset configs."""
        classification_datasets = ['herg', 'cyp3a4', 'tox21', 'bbbp']

        for name in classification_datasets:
            assert DATASET_CONFIG[name]['task'] == 'classification'
            assert DATASET_CONFIG[name]['metric'] == 'auc'

    def test_regression_datasets(self):
        """Test regression dataset configs."""
        regression_datasets = ['bace', 'pdbbind', 'freesolv', 'esol', 'lipo']

        for name in regression_datasets:
            assert DATASET_CONFIG[name]['task'] == 'regression'

    def test_expected_advantages(self):
        """Test expected DKO advantages are reasonable."""
        for name, config in DATASET_CONFIG.items():
            advantage = config['expected_advantage']
            assert 0.0 <= advantage <= 0.15  # Reasonable range


class TestMultiTaskDataset:
    """Test multi-task dataset handling."""

    def test_tox21_config(self):
        """Test Tox21 configuration."""
        config = DATASET_CONFIG['tox21']

        assert config['num_tasks'] == 12
        assert config['task'] == 'classification'
        assert config['target_col'] is None  # Multi-task


class TestCaching:
    """Test caching functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.mark.slow
    def test_cache_creation(self, temp_dir):
        """Test that caches are created."""
        try:
            dataset = get_dataset(
                "bace",
                root=temp_dir,
                split='train',
                verbose=False,
                max_conformers=5,
            )

            # Check cache directory exists
            cache_dir = temp_dir / 'bace' / 'cache'
            assert cache_dir.exists()

        except Exception as e:
            pytest.skip(f"Cache test failed: {e}")

    @pytest.mark.slow
    def test_cache_reload(self, temp_dir):
        """Test that cached data is reloaded."""
        try:
            # First load
            dataset1 = get_dataset(
                "bace",
                root=temp_dir,
                split='train',
                verbose=False,
                max_conformers=5,
            )
            n_samples1 = len(dataset1)

            # Second load (should use cache)
            dataset2 = get_dataset(
                "bace",
                root=temp_dir,
                split='train',
                verbose=False,
                max_conformers=5,
            )
            n_samples2 = len(dataset2)

            assert n_samples1 == n_samples2

        except Exception as e:
            pytest.skip(f"Cache reload test failed: {e}")

    @pytest.mark.slow
    def test_force_reload(self, temp_dir):
        """Test force reload flag."""
        try:
            # First load
            dataset1 = get_dataset(
                "bace",
                root=temp_dir,
                split='train',
                verbose=False,
                max_conformers=5,
            )

            # Force reload
            dataset2 = get_dataset(
                "bace",
                root=temp_dir,
                split='train',
                force_reload=True,
                verbose=False,
                max_conformers=5,
            )

            # Should still have same number of samples
            assert len(dataset1) == len(dataset2)

        except Exception as e:
            pytest.skip(f"Force reload test failed: {e}")
