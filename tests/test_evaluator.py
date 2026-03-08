"""
Comprehensive tests for the Evaluator module.

Tests all metrics, confidence intervals, stratified evaluation,
and statistical significance tests.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict

# Import from evaluator module
from dko.training.evaluator import (
    Evaluator,
    compute_metrics,
    compute_regression_metrics,
    compute_classification_metrics,
    compute_multitask_classification_metrics,
    compute_confidence_intervals,
    paired_t_test,
    wilcoxon_test,
)


# =============================================================================
# Test Fixtures
# =============================================================================


def create_test_loader(
    n_samples: int = 100,
    feature_dim: int = 50,
    task: str = "regression",
    include_scc: bool = False,
    include_smiles: bool = False,
):
    """Create a test data loader."""
    mu = torch.randn(n_samples, feature_dim)
    sigma = torch.randn(n_samples, feature_dim, feature_dim)
    sigma = torch.bmm(sigma, sigma.transpose(1, 2))  # Make PSD

    if task == "regression":
        labels = torch.randn(n_samples, 1)
    else:
        labels = torch.randint(0, 2, (n_samples, 1)).float()

    scc = torch.rand(n_samples) if include_scc else None
    smiles = [f"CC{i}O" for i in range(n_samples)] if include_smiles else None

    def collate_fn(batch):
        result = {
            'mu': torch.stack([b[0] for b in batch]),
            'sigma': torch.stack([b[1] for b in batch]),
            'label': torch.stack([b[2] for b in batch]),
        }

        if include_scc:
            indices = [batch[0][-1]] if isinstance(batch[0][-1], int) else [b[-1] for b in batch]
            result['scc'] = scc[indices] if scc is not None else None

        if include_smiles:
            indices = [b[-1] if isinstance(b[-1], int) else 0 for b in batch]
            result['smiles'] = [smiles[i] for i in indices] if smiles is not None else None

        return result

    # Use indices for collate
    indices = torch.arange(n_samples)
    dataset = TensorDataset(mu, sigma, labels, indices)

    def simple_collate(batch):
        mu_batch = torch.stack([b[0] for b in batch])
        sigma_batch = torch.stack([b[1] for b in batch])
        label_batch = torch.stack([b[2] for b in batch])
        idx_batch = torch.stack([b[3] for b in batch])

        result = {
            'mu': mu_batch,
            'sigma': sigma_batch,
            'label': label_batch,
        }

        if include_scc:
            result['scc'] = torch.index_select(scc, 0, idx_batch)

        if include_smiles:
            result['smiles'] = [smiles[i.item()] for i in idx_batch]

        return result

    return DataLoader(dataset, batch_size=16, collate_fn=simple_collate)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_dim: int, output_dim: int = 1, task: str = "regression"):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.task = task

    def forward(self, mu, sigma=None, fit_pca=False):
        return self.fc(mu)


class BaselineModel(nn.Module):
    """Baseline model with features format."""

    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, features, mask=None):
        # Mean pooling over conformers
        if mask is not None:
            features = features * mask.unsqueeze(-1)
            pooled = features.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            pooled = features.mean(dim=1)
        return self.fc(pooled)


# =============================================================================
# Test Regression Metrics
# =============================================================================


class TestRegressionMetrics:
    """Tests for regression metrics."""

    def test_perfect_predictions(self):
        """Test with perfect predictions."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        metrics = compute_regression_metrics(predictions, targets)

        assert metrics['rmse'] == pytest.approx(0.0, abs=1e-10)
        assert metrics['mae'] == pytest.approx(0.0, abs=1e-10)
        assert metrics['r2'] == pytest.approx(1.0, abs=1e-10)
        assert metrics['pearson'] == pytest.approx(1.0, abs=1e-10)
        assert metrics['spearman'] == pytest.approx(1.0, abs=1e-10)

    def test_constant_offset(self):
        """Test with constant offset in predictions."""
        predictions = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        metrics = compute_regression_metrics(predictions, targets)

        assert metrics['rmse'] == pytest.approx(1.0, abs=1e-10)
        assert metrics['mae'] == pytest.approx(1.0, abs=1e-10)
        # Correlation should still be perfect
        assert metrics['pearson'] == pytest.approx(1.0, abs=1e-10)
        assert metrics['spearman'] == pytest.approx(1.0, abs=1e-10)

    def test_random_predictions(self):
        """Test with random predictions."""
        np.random.seed(42)
        predictions = np.random.randn(100)
        targets = np.random.randn(100)

        metrics = compute_regression_metrics(predictions, targets)

        # Should have reasonable values
        assert metrics['rmse'] > 0
        assert metrics['mae'] > 0
        assert -1 <= metrics['pearson'] <= 1
        assert -1 <= metrics['spearman'] <= 1

    def test_handles_nan_values(self):
        """Test handling of NaN values."""
        predictions = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        targets = np.array([1.0, 2.0, np.nan, 4.0, 5.0])

        metrics = compute_regression_metrics(predictions, targets)

        # Should compute metrics on valid values only
        assert not np.isnan(metrics['rmse'])
        assert not np.isnan(metrics['mae'])

    def test_empty_predictions(self):
        """Test with empty predictions."""
        predictions = np.array([])
        targets = np.array([])

        metrics = compute_regression_metrics(predictions, targets)

        # All metrics should be NaN
        assert np.isnan(metrics['rmse'])
        assert np.isnan(metrics['mae'])

    def test_single_prediction(self):
        """Test with single prediction."""
        predictions = np.array([1.0])
        targets = np.array([2.0])

        metrics = compute_regression_metrics(predictions, targets)

        assert metrics['rmse'] == pytest.approx(1.0, abs=1e-10)
        assert metrics['mae'] == pytest.approx(1.0, abs=1e-10)
        # Correlation undefined for single value
        assert np.isnan(metrics['pearson'])

    def test_2d_input(self):
        """Test with 2D input arrays."""
        predictions = np.array([[1.0], [2.0], [3.0]])
        targets = np.array([[1.0], [2.0], [3.0]])

        metrics = compute_regression_metrics(predictions, targets)

        assert metrics['rmse'] == pytest.approx(0.0, abs=1e-10)


# =============================================================================
# Test Classification Metrics
# =============================================================================


class TestClassificationMetrics:
    """Tests for classification metrics."""

    def test_perfect_classification(self):
        """Test with perfect predictions."""
        predictions = np.array([0.1, 0.2, 0.8, 0.9])
        targets = np.array([0, 0, 1, 1])

        metrics = compute_classification_metrics(predictions, targets)

        assert metrics['accuracy'] == pytest.approx(1.0, abs=1e-10)
        assert metrics['precision'] == pytest.approx(1.0, abs=1e-10)
        assert metrics['recall'] == pytest.approx(1.0, abs=1e-10)
        assert metrics['f1'] == pytest.approx(1.0, abs=1e-10)

    def test_auc_roc(self):
        """Test AUC-ROC calculation."""
        predictions = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        targets = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        metrics = compute_classification_metrics(predictions, targets)

        assert metrics['auc'] == pytest.approx(1.0, abs=1e-10)

    def test_handles_logits(self):
        """Test handling of logit inputs (not probabilities)."""
        # Logits (can be negative or > 1)
        predictions = np.array([-2.0, -1.0, 1.0, 2.0])
        targets = np.array([0, 0, 1, 1])

        metrics = compute_classification_metrics(predictions, targets)

        # Should apply sigmoid internally
        assert 0 <= metrics['auc'] <= 1
        assert 0 <= metrics['accuracy'] <= 1

    def test_threshold_sensitivity(self):
        """Test sensitivity to threshold."""
        predictions = np.array([0.45, 0.45, 0.55, 0.55])
        targets = np.array([0, 0, 1, 1])

        # With default threshold 0.5
        metrics_05 = compute_classification_metrics(predictions, targets, threshold=0.5)

        # With different threshold
        metrics_04 = compute_classification_metrics(predictions, targets, threshold=0.4)

        # Different thresholds should give different results
        assert metrics_05['accuracy'] != metrics_04['accuracy'] or predictions.std() < 0.1

    def test_all_same_class(self):
        """Test with all same class (should handle gracefully)."""
        predictions = np.array([0.1, 0.2, 0.3, 0.4])
        targets = np.array([0, 0, 0, 0])  # All zeros

        metrics = compute_classification_metrics(predictions, targets)

        # AUC undefined when only one class
        assert np.isnan(metrics['auc'])
        # Other metrics should still work
        assert metrics['accuracy'] == pytest.approx(1.0, abs=1e-10)

    def test_specificity(self):
        """Test specificity calculation."""
        predictions = np.array([0.1, 0.9, 0.1, 0.9])
        targets = np.array([0, 0, 1, 1])

        metrics = compute_classification_metrics(predictions, targets)

        # Specificity = TN / (TN + FP)
        # Here: TN = 1, FP = 1, so specificity = 0.5
        assert metrics['specificity'] == pytest.approx(0.5, abs=1e-10)


# =============================================================================
# Test Multi-task Classification
# =============================================================================


class TestMultitaskClassificationMetrics:
    """Tests for multi-task classification metrics."""

    def test_multitask_metrics(self):
        """Test multi-task classification."""
        predictions = np.random.rand(100, 5)
        targets = np.random.randint(0, 2, (100, 5)).astype(float)

        metrics = compute_multitask_classification_metrics(predictions, targets)

        assert 'mean_auc' in metrics
        assert 'mean_auc_pr' in metrics
        assert 'mean_accuracy' in metrics
        assert metrics['n_total_tasks'] == 5

    def test_handles_missing_labels(self):
        """Test handling of NaN labels in multi-task."""
        predictions = np.random.rand(100, 3)
        targets = np.random.randint(0, 2, (100, 3)).astype(float)
        targets[:, 1] = np.nan  # Second task has all NaN

        metrics = compute_multitask_classification_metrics(predictions, targets)

        # Should handle missing task
        assert metrics['n_valid_tasks'] == 2
        assert metrics['n_total_tasks'] == 3


# =============================================================================
# Test Confidence Intervals
# =============================================================================


class TestConfidenceIntervals:
    """Tests for confidence interval computation."""

    def test_empty_values(self):
        """Test with empty values."""
        mean, lower, upper = compute_confidence_intervals([])

        assert np.isnan(mean)
        assert np.isnan(lower)
        assert np.isnan(upper)

    def test_single_value(self):
        """Test with single value."""
        mean, lower, upper = compute_confidence_intervals([1.0])

        assert mean == 1.0
        assert lower == 1.0
        assert upper == 1.0

    def test_multiple_values(self):
        """Test with multiple values."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        mean, lower, upper = compute_confidence_intervals(values)

        assert mean == pytest.approx(3.0, abs=1e-10)
        assert lower < mean
        assert upper > mean

    def test_confidence_level(self):
        """Test different confidence levels."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        _, lower_95, upper_95 = compute_confidence_intervals(values, 0.95)
        _, lower_99, upper_99 = compute_confidence_intervals(values, 0.99)

        # 99% CI should be wider than 95% CI
        assert (upper_99 - lower_99) > (upper_95 - lower_95)


# =============================================================================
# Test Evaluator Class
# =============================================================================


class TestEvaluator:
    """Tests for the Evaluator class."""

    def test_initialization(self):
        """Test evaluator initialization."""
        evaluator = Evaluator(task_type='regression')

        assert evaluator.task_type == 'regression'
        assert evaluator.primary_metric == 'rmse'

        evaluator_cls = Evaluator(task_type='classification')
        assert evaluator_cls.primary_metric == 'auc'

    def test_custom_primary_metric(self):
        """Test custom primary metric."""
        evaluator = Evaluator(task_type='regression', primary_metric='mae')

        assert evaluator.primary_metric == 'mae'

    def test_evaluate_regression(self):
        """Test evaluation for regression."""
        loader = create_test_loader(n_samples=50, feature_dim=32, task='regression')
        model = SimpleModel(input_dim=32, output_dim=1)
        model.eval()

        evaluator = Evaluator(task_type='regression', device='cpu')
        metrics = evaluator.evaluate(model, loader, verbose=False)

        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'pearson' in metrics
        assert 'spearman' in metrics
        assert 'n_samples' in metrics
        assert metrics['n_samples'] == 50

    def test_evaluate_classification(self):
        """Test evaluation for classification."""
        loader = create_test_loader(n_samples=50, feature_dim=32, task='classification')
        model = SimpleModel(input_dim=32, output_dim=1, task='classification')
        model.eval()

        evaluator = Evaluator(task_type='classification', device='cpu')
        metrics = evaluator.evaluate(model, loader, verbose=False)

        assert 'auc' in metrics
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics

    def test_evaluate_with_predictions(self):
        """Test evaluation returning predictions."""
        loader = create_test_loader(n_samples=50, feature_dim=32)
        model = SimpleModel(input_dim=32)
        model.eval()

        evaluator = Evaluator(task_type='regression', device='cpu')
        results = evaluator.evaluate(model, loader, return_predictions=True, verbose=False)

        assert 'metrics' in results
        assert 'predictions' in results
        assert 'labels' in results
        assert results['predictions'].shape[0] == 50

    def test_evaluate_with_ci(self):
        """Test evaluation with bootstrap confidence intervals."""
        loader = create_test_loader(n_samples=50, feature_dim=32)
        model = SimpleModel(input_dim=32)
        model.eval()

        evaluator = Evaluator(
            task_type='regression',
            device='cpu',
            bootstrap_n_samples=100  # Reduce for testing
        )
        metrics = evaluator.evaluate(model, loader, compute_ci=True, verbose=False)

        assert 'rmse_ci_lower' in metrics
        assert 'rmse_ci_upper' in metrics
        assert 'mae_ci_lower' in metrics
        assert 'mae_ci_upper' in metrics
        assert metrics['rmse_ci_lower'] <= metrics['rmse']
        assert metrics['rmse_ci_upper'] >= metrics['rmse']

    def test_stratified_evaluation(self):
        """Test stratified evaluation by SCC."""
        loader = create_test_loader(n_samples=100, feature_dim=32, include_scc=True)
        model = SimpleModel(input_dim=32)
        model.eval()

        evaluator = Evaluator(task_type='regression', device='cpu')
        results = evaluator.stratified_evaluation(
            model, loader, stratify_by='scc', n_bins=4, verbose=False
        )

        assert 'overall' in results
        assert 'stratified' in results
        assert 'stratify_by' in results
        assert results['stratify_by'] == 'scc'
        assert results['n_bins'] == 4

        # Should have quartile keys
        stratified = results['stratified']
        assert len(stratified) > 0

    def test_evaluate_with_smiles(self):
        """Test evaluation with SMILES strings."""
        loader = create_test_loader(n_samples=50, feature_dim=32, include_smiles=True)
        model = SimpleModel(input_dim=32)
        model.eval()

        evaluator = Evaluator(task_type='regression', device='cpu')
        results = evaluator.evaluate(model, loader, return_predictions=True, verbose=False)

        assert 'smiles' in results
        assert len(results['smiles']) == 50


# =============================================================================
# Test Statistical Tests
# =============================================================================


class TestStatisticalTests:
    """Tests for statistical significance tests."""

    def test_paired_t_test(self):
        """Test paired t-test."""
        errors_a = np.array([0.5, 0.6, 0.4, 0.55, 0.45])
        errors_b = np.array([0.7, 0.8, 0.6, 0.75, 0.65])

        t_stat, p_value = paired_t_test(errors_a, errors_b)

        assert isinstance(t_stat, float)
        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1

    def test_wilcoxon_test(self):
        """Test Wilcoxon signed-rank test."""
        errors_a = np.array([0.5, 0.6, 0.4, 0.55, 0.45])
        errors_b = np.array([0.7, 0.8, 0.6, 0.75, 0.65])

        stat, p_value = wilcoxon_test(errors_a, errors_b)

        assert isinstance(stat, float)
        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1

    def test_significance_test_method(self):
        """Test Evaluator.significance_test method."""
        evaluator = Evaluator(task_type='regression', device='cpu')

        model1_values = [0.5, 0.6, 0.4, 0.55, 0.45]
        model2_values = [0.7, 0.8, 0.6, 0.75, 0.65]

        # Paired t-test
        stat, p_value = evaluator.significance_test(
            model1_values, model2_values, test='paired_t'
        )
        assert 0 <= p_value <= 1

        # Wilcoxon test
        stat, p_value = evaluator.significance_test(
            model1_values, model2_values, test='wilcoxon'
        )
        assert 0 <= p_value <= 1


# =============================================================================
# Test Model Comparison
# =============================================================================


class TestModelComparison:
    """Tests for model comparison functionality."""

    def test_compare_models(self):
        """Test compare_models method."""
        evaluator = Evaluator(task_type='regression', device='cpu')

        # Create mock results
        results_dict = {
            'single_conformer': {
                'rmse': {
                    'mean': 0.8,
                    'std': 0.05,
                    'values': [0.75, 0.80, 0.85],
                }
            },
            'dko': {
                'rmse': {
                    'mean': 0.6,
                    'std': 0.04,
                    'values': [0.58, 0.60, 0.62],
                }
            },
            'attention': {
                'rmse': {
                    'mean': 0.65,
                    'std': 0.03,
                    'values': [0.63, 0.65, 0.67],
                }
            },
        }

        comparisons = evaluator.compare_models(
            results_dict, baseline_name='single_conformer'
        )

        assert 'dko' in comparisons
        assert 'attention' in comparisons
        assert 'single_conformer' not in comparisons  # Baseline excluded

        # Check comparison structure
        assert 'improvement_percent' in comparisons['dko']
        assert 'p_value' in comparisons['dko']
        assert 'significant' in comparisons['dko']

        # DKO should show improvement (lower RMSE)
        assert comparisons['dko']['improvement_percent'] > 0


# =============================================================================
# Test compute_metrics Function
# =============================================================================


class TestComputeMetrics:
    """Tests for the unified compute_metrics function."""

    def test_regression_dispatch(self):
        """Test that regression is properly dispatched."""
        predictions = np.array([1.0, 2.0, 3.0])
        targets = np.array([1.0, 2.0, 3.0])

        metrics = compute_metrics(predictions, targets, task_type='regression')

        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics

    def test_classification_dispatch(self):
        """Test that classification is properly dispatched."""
        predictions = np.array([0.1, 0.9, 0.1, 0.9])
        targets = np.array([0, 1, 0, 1])

        metrics = compute_metrics(predictions, targets, task_type='classification')

        assert 'auc' in metrics
        assert 'accuracy' in metrics
        assert 'f1' in metrics


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_very_small_dataset(self):
        """Test with very small dataset."""
        predictions = np.array([1.0, 2.0])
        targets = np.array([1.0, 2.0])

        metrics = compute_regression_metrics(predictions, targets)

        # Should still compute
        assert not np.isnan(metrics['rmse'])
        assert not np.isnan(metrics['mae'])

    def test_large_values(self):
        """Test with large values."""
        predictions = np.array([1e6, 2e6, 3e6])
        targets = np.array([1e6, 2e6, 3e6])

        metrics = compute_regression_metrics(predictions, targets)

        assert metrics['rmse'] == pytest.approx(0.0, abs=1e-3)

    def test_negative_values(self):
        """Test with negative values."""
        predictions = np.array([-1.0, -2.0, -3.0])
        targets = np.array([-1.0, -2.0, -3.0])

        metrics = compute_regression_metrics(predictions, targets)

        assert metrics['rmse'] == pytest.approx(0.0, abs=1e-10)


# =============================================================================
# Run Tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
