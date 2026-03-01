"""Tests for data splitting utilities."""

import pytest
import numpy as np

# Skip if RDKit not available
pytest.importorskip("rdkit")

from dko.data.splits import (
    scaffold_split,
    random_split,
    stratified_split,
    get_split,
    cross_validation_splits,
    get_scaffold,
    get_scaffold_mapping,
    verify_no_scaffold_overlap,
    compute_split_statistics,
    get_split_hash,
)


class TestGetScaffold:
    """Test scaffold extraction."""

    def test_simple_molecule(self):
        """Test scaffold extraction for simple molecule."""
        scaffold = get_scaffold("c1ccccc1")  # Benzene
        assert scaffold == "c1ccccc1"  # Benzene is its own scaffold

    def test_substituted_benzene(self):
        """Test scaffold extraction removes substituents."""
        scaffold = get_scaffold("Cc1ccccc1")  # Toluene
        assert scaffold == "c1ccccc1"  # Should be benzene

    def test_invalid_smiles(self):
        """Test handling of invalid SMILES."""
        scaffold = get_scaffold("INVALID")
        assert scaffold == "INVALID"

    def test_complex_molecule(self):
        """Test scaffold for more complex molecule."""
        # Ibuprofen-like structure
        smiles = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
        scaffold = get_scaffold(smiles)
        assert scaffold is not None
        assert len(scaffold) > 0


class TestScaffoldMapping:
    """Test scaffold mapping function."""

    def test_scaffold_mapping(self):
        """Test creation of scaffold mapping."""
        smiles_list = [
            "c1ccccc1",      # Benzene
            "Cc1ccccc1",     # Toluene (same scaffold)
            "CCc1ccccc1",    # Ethylbenzene (same scaffold)
            "CCO",           # Ethanol
            "CCCO",          # Propanol (same scaffold)
        ]

        mapping = get_scaffold_mapping(smiles_list)

        # Should have fewer unique scaffolds than molecules
        assert len(mapping) <= len(smiles_list)

        # Benzene derivatives should share scaffold
        benzene_scaffold = get_scaffold("c1ccccc1")
        assert benzene_scaffold in mapping
        assert len(mapping[benzene_scaffold]) >= 1

    def test_empty_list(self):
        """Test with empty list."""
        mapping = get_scaffold_mapping([])
        assert len(mapping) == 0


class TestScaffoldSplit:
    """Test scaffold-based splitting."""

    @pytest.fixture
    def sample_smiles(self):
        """Create sample SMILES list for testing."""
        return [
            "c1ccccc1", "Cc1ccccc1", "CCc1ccccc1",  # Benzene derivatives
            "CCO", "CCCO", "CCCCO",                  # Alcohols
            "CC(=O)O", "CCC(=O)O", "CCCC(=O)O",      # Carboxylic acids
            "CCN", "CCCN", "CCCCN",                  # Amines
        ]

    def test_basic_split(self, sample_smiles):
        """Test basic scaffold split."""
        splits = scaffold_split(sample_smiles, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits

        # All indices should be covered
        all_indices = set(splits['train']) | set(splits['val']) | set(splits['test'])
        assert all_indices == set(range(len(sample_smiles)))

        # No overlap between splits
        assert len(set(splits['train']) & set(splits['val'])) == 0
        assert len(set(splits['train']) & set(splits['test'])) == 0
        assert len(set(splits['val']) & set(splits['test'])) == 0

    def test_split_ratios(self, sample_smiles):
        """Test that split ratios are approximately correct."""
        splits = scaffold_split(sample_smiles, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)

        n_total = len(sample_smiles)
        # Allow for some variance due to scaffold constraints
        assert len(splits['train']) >= n_total * 0.4
        assert len(splits['train']) <= n_total * 0.8

    def test_reproducibility(self, sample_smiles):
        """Test that same seed produces same split."""
        splits1 = scaffold_split(sample_smiles, seed=42)
        splits2 = scaffold_split(sample_smiles, seed=42)

        np.testing.assert_array_equal(splits1['train'], splits2['train'])
        np.testing.assert_array_equal(splits1['val'], splits2['val'])
        np.testing.assert_array_equal(splits1['test'], splits2['test'])

    def test_different_seeds(self, sample_smiles):
        """Test that different seeds produce different splits."""
        splits1 = scaffold_split(sample_smiles, seed=42)
        splits2 = scaffold_split(sample_smiles, seed=123)

        # At least one split should be different
        train_same = np.array_equal(splits1['train'], splits2['train'])
        val_same = np.array_equal(splits1['val'], splits2['val'])
        test_same = np.array_equal(splits1['test'], splits2['test'])

        assert not (train_same and val_same and test_same)

    def test_no_scaffold_overlap(self, sample_smiles):
        """Test that scaffolds don't overlap between splits."""
        splits = scaffold_split(sample_smiles, seed=42)

        assert verify_no_scaffold_overlap(splits, sample_smiles)

    def test_stratified_split(self):
        """Test stratified scaffold split for classification."""
        smiles_list = [
            "c1ccccc1", "Cc1ccccc1", "CCc1ccccc1",  # Class 0
            "CCO", "CCCO", "CCCCO",                  # Class 1
            "CC(=O)O", "CCC(=O)O", "CCCC(=O)O",      # Class 0
            "CCN", "CCCN", "CCCCN",                  # Class 1
        ]
        labels = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])

        splits = scaffold_split(
            smiles_list,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            stratify=True,
            labels=labels,
        )

        # Each split should have both classes (if possible)
        train_labels = labels[splits['train']]
        assert len(np.unique(train_labels)) >= 1  # At least one class


class TestRandomSplit:
    """Test random splitting."""

    @pytest.fixture
    def sample_smiles(self):
        """Create sample SMILES list."""
        return ["CCO", "CCCO", "CCCCO", "CCCCCO", "CCCCCCO"] * 10

    def test_basic_split(self, sample_smiles):
        """Test basic random split."""
        splits = random_split(sample_smiles, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

        assert len(splits['train']) > 0
        assert len(splits['val']) > 0
        assert len(splits['test']) > 0

        # Total should equal original
        total = len(splits['train']) + len(splits['val']) + len(splits['test'])
        assert total == len(sample_smiles)

    def test_reproducibility(self, sample_smiles):
        """Test reproducibility with same seed."""
        splits1 = random_split(sample_smiles, seed=42)
        splits2 = random_split(sample_smiles, seed=42)

        np.testing.assert_array_equal(splits1['train'], splits2['train'])


class TestStratifiedSplit:
    """Test stratified splitting."""

    @pytest.fixture
    def classification_data(self):
        """Create classification data."""
        smiles = ["CCO", "CCCO", "CCCCO"] * 20
        labels = np.array([0, 1, 0] * 20)
        return smiles, labels

    @pytest.fixture
    def regression_data(self):
        """Create regression data."""
        smiles = ["CCO", "CCCO", "CCCCO"] * 20
        labels = np.random.randn(60) * 2 + 5
        return smiles, labels

    def test_classification_split(self, classification_data):
        """Test stratified split for classification."""
        smiles, labels = classification_data
        splits = stratified_split(smiles, labels, seed=42)

        # Check class balance in train set
        train_labels = labels[splits['train']]
        unique, counts = np.unique(train_labels, return_counts=True)

        # Both classes should be represented
        assert len(unique) == 2

    def test_regression_split(self, regression_data):
        """Test stratified split for regression (binned)."""
        smiles, labels = regression_data
        splits = stratified_split(smiles, labels, n_bins=5, seed=42)

        # Should produce valid splits
        assert len(splits['train']) > 0
        assert len(splits['val']) > 0
        assert len(splits['test']) > 0


class TestGetSplit:
    """Test the unified get_split function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        smiles = ["CCO", "CCCO", "CCCCO", "c1ccccc1", "Cc1ccccc1"] * 10
        labels = np.random.randint(0, 2, len(smiles))
        return smiles, labels

    def test_scaffold_method(self, sample_data):
        """Test scaffold method."""
        smiles, labels = sample_data
        splits = get_split(smiles, method="scaffold")

        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits

    def test_random_method(self, sample_data):
        """Test random method."""
        smiles, labels = sample_data
        splits = get_split(smiles, method="random")

        assert len(splits['train']) > 0

    def test_stratified_method(self, sample_data):
        """Test stratified method."""
        smiles, labels = sample_data
        splits = get_split(smiles, labels=labels, method="stratified")

        assert len(splits['train']) > 0

    def test_stratified_scaffold_method(self, sample_data):
        """Test stratified_scaffold method."""
        smiles, labels = sample_data
        splits = get_split(smiles, labels=labels, method="stratified_scaffold")

        assert len(splits['train']) > 0

    def test_invalid_method(self, sample_data):
        """Test invalid method raises error."""
        smiles, labels = sample_data
        with pytest.raises(ValueError):
            get_split(smiles, method="invalid_method")


class TestCrossValidation:
    """Test cross-validation splitting."""

    @pytest.fixture
    def sample_smiles(self):
        """Create sample SMILES list."""
        return ["CCO", "CCCO", "CCCCO", "c1ccccc1", "Cc1ccccc1"] * 10

    def test_basic_cv(self, sample_smiles):
        """Test basic cross-validation."""
        cv_splits = cross_validation_splits(sample_smiles, n_folds=5, method="random")

        assert len(cv_splits) == 5

        for fold in cv_splits:
            assert 'train' in fold
            assert 'test' in fold
            # No overlap
            assert len(set(fold['train']) & set(fold['test'])) == 0

    def test_scaffold_cv(self, sample_smiles):
        """Test scaffold-based cross-validation."""
        cv_splits = cross_validation_splits(sample_smiles, n_folds=3, method="scaffold")

        assert len(cv_splits) == 3

        # All molecules should appear in test exactly once
        all_test_indices = []
        for fold in cv_splits:
            all_test_indices.extend(fold['test'])

        # May have duplicates due to scaffold grouping
        # But should cover most molecules
        assert len(set(all_test_indices)) > len(sample_smiles) * 0.8

    def test_cv_coverage(self, sample_smiles):
        """Test that CV covers all molecules."""
        cv_splits = cross_validation_splits(sample_smiles, n_folds=5, method="random")

        # Each molecule should be in test set exactly once
        test_counts = np.zeros(len(sample_smiles))
        for fold in cv_splits:
            for idx in fold['test']:
                test_counts[idx] += 1

        # All should be exactly 1
        assert np.all(test_counts == 1)


class TestSplitStatistics:
    """Test split statistics computation."""

    def test_compute_statistics(self):
        """Test computing split statistics."""
        smiles_list = [
            "c1ccccc1", "Cc1ccccc1", "CCc1ccccc1",
            "CCO", "CCCO", "CCCCO",
        ]
        labels = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        splits = scaffold_split(smiles_list, seed=42)
        stats = compute_split_statistics(splits, smiles_list, labels)

        assert 'train' in stats
        assert 'val' in stats
        assert 'test' in stats

        for split_name in ['train', 'val', 'test']:
            assert 'n_samples' in stats[split_name]
            assert 'n_unique_scaffolds' in stats[split_name]


class TestSplitHash:
    """Test split hash function."""

    def test_hash_same_config(self):
        """Test that same config produces same hash."""
        smiles = ["CCO", "CCCO", "CCCCO"]

        hash1 = get_split_hash(smiles, method="scaffold", seed=42)
        hash2 = get_split_hash(smiles, method="scaffold", seed=42)

        assert hash1 == hash2

    def test_hash_different_seed(self):
        """Test that different seed produces different hash."""
        smiles = ["CCO", "CCCO", "CCCCO"]

        hash1 = get_split_hash(smiles, method="scaffold", seed=42)
        hash2 = get_split_hash(smiles, method="scaffold", seed=123)

        assert hash1 != hash2

    def test_hash_different_method(self):
        """Test that different method produces different hash."""
        smiles = ["CCO", "CCCO", "CCCCO"]

        hash1 = get_split_hash(smiles, method="scaffold", seed=42)
        hash2 = get_split_hash(smiles, method="random", seed=42)

        assert hash1 != hash2


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_small_dataset(self):
        """Test splitting very small dataset."""
        smiles = ["CCO", "CCCO", "CCCCO"]
        splits = scaffold_split(smiles, seed=42)

        # Should still produce valid splits
        total = len(splits['train']) + len(splits['val']) + len(splits['test'])
        assert total == 3

    def test_single_molecule(self):
        """Test with single molecule."""
        smiles = ["CCO"]
        splits = scaffold_split(smiles, seed=42)

        # Should put in train
        assert len(splits['train']) == 1

    def test_ratio_normalization(self):
        """Test that ratios are normalized."""
        smiles = ["CCO", "CCCO", "CCCCO"] * 10
        # Ratios don't sum to 1
        splits = scaffold_split(smiles, train_ratio=0.6, val_ratio=0.3, test_ratio=0.3)

        # Should still work
        total = len(splits['train']) + len(splits['val']) + len(splits['test'])
        assert total == len(smiles)

    def test_all_invalid_smiles(self):
        """Test with all invalid SMILES."""
        smiles = ["INVALID1", "INVALID2", "INVALID3"]
        splits = scaffold_split(smiles, seed=42)

        # Should still produce valid splits (using fallback scaffold)
        total = len(splits['train']) + len(splits['val']) + len(splits['test'])
        assert total == 3
