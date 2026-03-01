"""
Statistical Consistency Check (SCC) validation for DKO.

Implements the SCC framework to validate that model improvements
are statistically meaningful.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats
from dko.utils.logging_utils import get_logger

logger = get_logger("scc_validation")


def run_scc_validation(
    baseline_results: Dict,
    model_results: Dict,
    alpha: float = 0.05,
) -> Dict:
    """
    Run Statistical Consistency Check validation.

    Args:
        baseline_results: Dictionary of baseline metrics across seeds
        model_results: Dictionary of model metrics across seeds
        alpha: Significance level

    Returns:
        Dictionary with SCC results
    """
    logger.info("Running SCC validation")

    # Extract metric values
    baseline_values = baseline_results.get("values", [])
    model_values = model_results.get("values", [])

    if not baseline_values or not model_values:
        return {"error": "Missing values"}

    # Compute statistics
    baseline_mean = np.mean(baseline_values)
    model_mean = np.mean(model_values)
    improvement = baseline_mean - model_mean  # For RMSE (lower is better)

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(baseline_values, model_values)

    # Effect size (Cohen's d)
    diff = np.array(baseline_values) - np.array(model_values)
    cohens_d = np.mean(diff) / np.std(diff, ddof=1)

    # Confidence interval for difference
    n = len(diff)
    ci_margin = stats.t.ppf(1 - alpha/2, df=n-1) * np.std(diff, ddof=1) / np.sqrt(n)
    ci_lower = np.mean(diff) - ci_margin
    ci_upper = np.mean(diff) + ci_margin

    return {
        "baseline_mean": baseline_mean,
        "model_mean": model_mean,
        "improvement": improvement,
        "improvement_percent": (improvement / baseline_mean) * 100,
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": p_value < alpha,
        "cohens_d": cohens_d,
        "effect_size": _interpret_effect_size(cohens_d),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def _interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def validate_improvement_claims(
    all_results: Dict,
    expected_improvements: Dict,
) -> Dict:
    """
    Validate claimed improvements against expected values.

    Args:
        all_results: Results for all datasets and models
        expected_improvements: Expected improvement percentages

    Returns:
        Validation summary
    """
    validation = {}

    for dataset, expected in expected_improvements.items():
        if dataset not in all_results:
            validation[dataset] = {"status": "missing"}
            continue

        # Get DKO and baseline results
        dko_results = all_results[dataset].get("dko", {})
        baseline_results = all_results[dataset].get("single_conformer", {})

        if not dko_results or not baseline_results:
            validation[dataset] = {"status": "incomplete"}
            continue

        # Run SCC
        scc_result = run_scc_validation(baseline_results, dko_results)

        # Check if improvement meets expectations
        actual_improvement = scc_result["improvement_percent"]
        meets_expectation = actual_improvement >= expected * 0.8  # Allow 20% margin

        validation[dataset] = {
            "expected_improvement": expected,
            "actual_improvement": actual_improvement,
            "meets_expectation": meets_expectation,
            "significant": scc_result["significant"],
            "p_value": scc_result["p_value"],
        }

    return validation
