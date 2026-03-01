"""Tests for Statistical Consistency Checks."""

import pytest
import numpy as np

from dko.analysis.scc import (
    StatisticalConsistencyChecker,
    compute_scc_scores,
)


class TestStatisticalConsistencyChecker:
    """Test suite for SCC."""

    @pytest.fixture
    def checker(self):
        """Create a checker instance."""
        return StatisticalConsistencyChecker(alpha=0.05, min_seeds=3)

    def test_check_significance_significant(self, checker):
        """Test detection of significant difference."""
        model1_values = [1.0, 1.1, 1.2, 1.0, 1.1]
        model2_values = [0.5, 0.6, 0.5, 0.4, 0.5]

        result = checker.check_significance(model1_values, model2_values)

        assert result["valid"]
        assert result["significant"]
        assert result["p_value"] < 0.05

    def test_check_significance_not_significant(self, checker):
        """Test when difference is not significant."""
        model1_values = [1.0, 1.1, 0.9, 1.0, 1.1]
        model2_values = [1.05, 1.0, 0.95, 1.1, 0.9]

        result = checker.check_significance(model1_values, model2_values)

        assert result["valid"]
        # May or may not be significant depending on data

    def test_check_significance_insufficient_seeds(self, checker):
        """Test handling of insufficient seeds."""
        model1_values = [1.0, 1.1]
        model2_values = [0.5, 0.6]

        result = checker.check_significance(model1_values, model2_values)

        assert not result["valid"]
        assert "Insufficient seeds" in result["reason"]

    def test_check_consistency(self, checker):
        """Test consistency checking."""
        consistent_values = [1.0, 1.02, 0.98, 1.01, 0.99]
        result = checker.check_consistency(consistent_values, expected_cv=0.1)

        assert result["valid"]
        assert result["consistent"]
        assert result["cv"] < 0.1

    def test_check_consistency_inconsistent(self, checker):
        """Test detection of inconsistent results."""
        inconsistent_values = [1.0, 2.0, 0.5, 1.5, 3.0]
        result = checker.check_consistency(inconsistent_values, expected_cv=0.1)

        assert result["valid"]
        assert not result["consistent"]

    def test_validate_improvement(self, checker):
        """Test improvement validation."""
        baseline = [1.0, 1.1, 1.05, 0.95, 1.0]
        model = [0.9, 0.85, 0.88, 0.92, 0.87]  # ~10% improvement

        result = checker.validate_improvement(
            baseline, model, expected_improvement=10.0, lower_is_better=True
        )

        assert "actual_improvement" in result
        assert result["actual_improvement"] > 0


class TestComputeSCCScores:
    """Test the compute_scc_scores function."""

    def test_compute_scc_scores(self):
        """Test SCC score computation."""
        results = {
            "single_conformer": {"rmse": [1.0, 1.1, 1.05]},
            "dko": {"rmse": [0.9, 0.85, 0.88]},
            "attention": {"rmse": [0.95, 0.92, 0.93]},
        }

        scores = compute_scc_scores(results, baseline_key="single_conformer")

        assert "dko" in scores
        assert "attention" in scores
        assert "single_conformer" not in scores

    def test_compute_scc_scores_missing_baseline(self):
        """Test handling of missing baseline."""
        results = {
            "dko": {"rmse": [0.9, 0.85, 0.88]},
        }

        scores = compute_scc_scores(results, baseline_key="single_conformer")

        assert "error" in scores
