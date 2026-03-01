"""
Statistical analysis utilities.

This module provides functions for computing confidence intervals,
significance tests, and bootstrap statistics.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats


def compute_confidence_intervals(
    values: Union[List[float], np.ndarray],
    confidence_level: float = 0.95,
    method: str = "t",
) -> Tuple[float, float, float]:
    """
    Compute confidence interval for a set of values.

    Args:
        values: Sample values
        confidence_level: Confidence level (default 0.95)
        method: 't' for t-distribution, 'bootstrap' for bootstrap

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    values = np.array(values)
    n = len(values)
    mean = np.mean(values)

    if n < 2:
        return mean, mean, mean

    if method == "t":
        std = np.std(values, ddof=1)
        alpha = 1 - confidence_level
        t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
        margin = t_crit * std / np.sqrt(n)
        return mean, mean - margin, mean + margin

    elif method == "bootstrap":
        lower, upper = bootstrap_ci(values, confidence_level=confidence_level)
        return mean, lower, upper

    else:
        raise ValueError(f"Unknown method: {method}")


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable = np.mean,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval.

    Args:
        values: Sample values
        statistic: Statistic function (default: mean)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    values = np.array(values)
    n = len(values)

    # Generate bootstrap samples
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=n, replace=True)
        bootstrap_stats.append(statistic(sample))

    bootstrap_stats = np.array(bootstrap_stats)

    # Compute percentiles
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)

    return lower, upper


def bootstrap_statistics(
    values: Union[List[float], np.ndarray],
    statistics: List[str] = ["mean", "std", "median"],
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
) -> Dict[str, Dict]:
    """
    Compute multiple bootstrap statistics.

    Args:
        values: Sample values
        statistics: List of statistics to compute
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level

    Returns:
        Dictionary mapping statistic name to (value, ci_lower, ci_upper)
    """
    values = np.array(values)

    stat_funcs = {
        "mean": np.mean,
        "std": lambda x: np.std(x, ddof=1),
        "median": np.median,
        "min": np.min,
        "max": np.max,
    }

    results = {}

    for stat_name in statistics:
        if stat_name not in stat_funcs:
            continue

        func = stat_funcs[stat_name]
        value = func(values)
        lower, upper = bootstrap_ci(values, func, n_bootstrap, confidence_level)

        results[stat_name] = {
            "value": value,
            "ci_lower": lower,
            "ci_upper": upper,
        }

    return results


def perform_significance_tests(
    model1_values: Union[List[float], np.ndarray],
    model2_values: Union[List[float], np.ndarray],
    tests: List[str] = ["paired_t", "wilcoxon"],
) -> Dict[str, Dict]:
    """
    Perform multiple significance tests.

    Args:
        model1_values: Values from model 1
        model2_values: Values from model 2
        tests: List of test names

    Returns:
        Dictionary mapping test name to results
    """
    model1_values = np.array(model1_values)
    model2_values = np.array(model2_values)

    results = {}

    for test_name in tests:
        if test_name == "paired_t":
            stat, p_value = stats.ttest_rel(model1_values, model2_values)
            results[test_name] = {
                "statistic": stat,
                "p_value": p_value,
                "significant_05": p_value < 0.05,
                "significant_01": p_value < 0.01,
            }

        elif test_name == "wilcoxon":
            try:
                stat, p_value = stats.wilcoxon(model1_values, model2_values)
                results[test_name] = {
                    "statistic": stat,
                    "p_value": p_value,
                    "significant_05": p_value < 0.05,
                    "significant_01": p_value < 0.01,
                }
            except ValueError as e:
                results[test_name] = {"error": str(e)}

        elif test_name == "mann_whitney":
            stat, p_value = stats.mannwhitneyu(model1_values, model2_values)
            results[test_name] = {
                "statistic": stat,
                "p_value": p_value,
                "significant_05": p_value < 0.05,
            }

        elif test_name == "bootstrap":
            stat, p_value = _bootstrap_test(model1_values, model2_values)
            results[test_name] = {
                "statistic": stat,
                "p_value": p_value,
                "significant_05": p_value < 0.05,
            }

    return results


def _bootstrap_test(
    values1: np.ndarray,
    values2: np.ndarray,
    n_bootstrap: int = 10000,
) -> Tuple[float, float]:
    """Permutation-based bootstrap test."""
    observed_diff = np.mean(values1) - np.mean(values2)

    combined = np.concatenate([values1, values2])
    n1 = len(values1)

    count = 0
    for _ in range(n_bootstrap):
        np.random.shuffle(combined)
        boot_diff = np.mean(combined[:n1]) - np.mean(combined[n1:])
        if abs(boot_diff) >= abs(observed_diff):
            count += 1

    p_value = (count + 1) / (n_bootstrap + 1)
    return observed_diff, p_value


def compute_effect_size(
    model1_values: Union[List[float], np.ndarray],
    model2_values: Union[List[float], np.ndarray],
    paired: bool = True,
) -> Dict:
    """
    Compute various effect size measures.

    Args:
        model1_values: Values from model 1
        model2_values: Values from model 2
        paired: Whether the comparison is paired

    Returns:
        Dictionary with effect size measures
    """
    model1_values = np.array(model1_values)
    model2_values = np.array(model2_values)

    mean1 = np.mean(model1_values)
    mean2 = np.mean(model2_values)
    diff = model1_values - model2_values

    results = {
        "mean_difference": np.mean(diff),
        "relative_difference": (mean1 - mean2) / mean2 if mean2 != 0 else float("inf"),
    }

    # Cohen's d
    if paired:
        cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0
    else:
        pooled_std = np.sqrt(
            (np.var(model1_values, ddof=1) + np.var(model2_values, ddof=1)) / 2
        )
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0

    results["cohens_d"] = cohens_d
    results["cohens_d_interpretation"] = _interpret_cohens_d(cohens_d)

    # Hedge's g (corrected for small samples)
    n = len(model1_values)
    correction = 1 - 3 / (4 * (2 * n - 2) - 1)
    results["hedges_g"] = cohens_d * correction

    return results


def _interpret_cohens_d(d: float) -> str:
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


# =============================================================================
# Multiple Comparison Corrections
# =============================================================================

def bonferroni_correction(
    p_values: Union[List[float], np.ndarray],
    alpha: float = 0.05,
) -> Dict[str, Union[np.ndarray, List[bool], int]]:
    """
    Apply Bonferroni correction for multiple hypothesis testing.

    The Bonferroni correction is a conservative method that controls the
    family-wise error rate (FWER) by adjusting the significance threshold.
    Each p-value is multiplied by the number of tests, or equivalently,
    the significance level is divided by the number of tests.

    Args:
        p_values: Array of p-values from multiple hypothesis tests
        alpha: Original significance level (default: 0.05)

    Returns:
        Dictionary containing:
            - adjusted_p_values: Bonferroni-adjusted p-values (capped at 1.0)
            - significant: Boolean array indicating significant results
            - adjusted_alpha: The corrected significance threshold
            - n_tests: Number of tests performed
            - n_significant: Number of significant results after correction

    Example:
        >>> p_values = [0.01, 0.04, 0.02, 0.15, 0.03]
        >>> result = bonferroni_correction(p_values, alpha=0.05)
        >>> print(result['significant'])  # Which tests remain significant
        >>> print(result['adjusted_p_values'])  # Adjusted p-values
    """
    p_values = np.array(p_values)
    n_tests = len(p_values)

    # Adjust p-values (cap at 1.0)
    adjusted_p = np.minimum(p_values * n_tests, 1.0)

    # Determine significance
    adjusted_alpha = alpha / n_tests
    significant = p_values < adjusted_alpha

    return {
        "adjusted_p_values": adjusted_p,
        "significant": significant.tolist(),
        "adjusted_alpha": adjusted_alpha,
        "n_tests": n_tests,
        "n_significant": int(np.sum(significant)),
    }


def holm_bonferroni_correction(
    p_values: Union[List[float], np.ndarray],
    alpha: float = 0.05,
) -> Dict[str, Union[np.ndarray, List[bool], int]]:
    """
    Apply Holm-Bonferroni (step-down) correction for multiple testing.

    The Holm-Bonferroni method is less conservative than Bonferroni while
    still controlling FWER. It tests p-values in ascending order with
    progressively less stringent thresholds.

    Args:
        p_values: Array of p-values from multiple hypothesis tests
        alpha: Original significance level (default: 0.05)

    Returns:
        Dictionary containing:
            - adjusted_p_values: Holm-Bonferroni adjusted p-values
            - significant: Boolean array indicating significant results
            - n_tests: Number of tests performed
            - n_significant: Number of significant results after correction
    """
    p_values = np.array(p_values)
    n_tests = len(p_values)

    # Get sorted order and reverse mapping
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    # Calculate adjusted p-values
    adjusted_p = np.zeros(n_tests)
    for i, idx in enumerate(sorted_idx):
        # Multiplier decreases as we go through sorted p-values
        multiplier = n_tests - i
        adjusted_p[idx] = sorted_p[i] * multiplier

    # Enforce monotonicity (cumulative maximum)
    adjusted_sorted = adjusted_p[sorted_idx]
    for i in range(1, n_tests):
        adjusted_sorted[i] = max(adjusted_sorted[i], adjusted_sorted[i - 1])
    adjusted_p[sorted_idx] = adjusted_sorted

    # Cap at 1.0
    adjusted_p = np.minimum(adjusted_p, 1.0)

    # Determine significance
    significant = adjusted_p < alpha

    return {
        "adjusted_p_values": adjusted_p,
        "significant": significant.tolist(),
        "n_tests": n_tests,
        "n_significant": int(np.sum(significant)),
    }


def benjamini_hochberg_correction(
    p_values: Union[List[float], np.ndarray],
    alpha: float = 0.05,
) -> Dict[str, Union[np.ndarray, List[bool], float, int]]:
    """
    Apply Benjamini-Hochberg correction for controlling False Discovery Rate.

    The BH procedure controls FDR rather than FWER, making it less conservative
    than Bonferroni. It's particularly useful when performing many tests and
    some false positives are acceptable.

    Args:
        p_values: Array of p-values from multiple hypothesis tests
        alpha: Desired FDR level (default: 0.05)

    Returns:
        Dictionary containing:
            - adjusted_p_values: BH-adjusted p-values (q-values)
            - significant: Boolean array indicating significant results
            - fdr_threshold: The critical value for significance
            - n_tests: Number of tests performed
            - n_significant: Number of significant results (discoveries)
    """
    p_values = np.array(p_values)
    n_tests = len(p_values)

    # Sort p-values
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    # Calculate BH critical values
    ranks = np.arange(1, n_tests + 1)
    critical_values = alpha * ranks / n_tests

    # Find the largest k where p[k] <= critical_value[k]
    below_threshold = sorted_p <= critical_values
    if np.any(below_threshold):
        max_k = np.max(np.where(below_threshold)[0])
        fdr_threshold = critical_values[max_k]
    else:
        max_k = -1
        fdr_threshold = 0.0

    # Calculate adjusted p-values (q-values)
    adjusted_p = np.zeros(n_tests)
    adjusted_p[sorted_idx[-1]] = sorted_p[-1]
    for i in range(n_tests - 2, -1, -1):
        adjusted_p[sorted_idx[i]] = min(
            sorted_p[i] * n_tests / (i + 1),
            adjusted_p[sorted_idx[i + 1]]
        )
    adjusted_p = np.minimum(adjusted_p, 1.0)

    # Determine significance
    significant = np.zeros(n_tests, dtype=bool)
    significant[sorted_idx[:max_k + 1]] = True

    return {
        "adjusted_p_values": adjusted_p,
        "significant": significant.tolist(),
        "fdr_threshold": fdr_threshold,
        "n_tests": n_tests,
        "n_significant": int(np.sum(significant)),
    }


def multiple_comparison_summary(
    p_values: Union[List[float], np.ndarray],
    labels: Optional[List[str]] = None,
    alpha: float = 0.05,
) -> Dict:
    """
    Apply multiple comparison corrections and summarize results.

    Applies Bonferroni, Holm-Bonferroni, and Benjamini-Hochberg corrections
    and provides a comprehensive summary for easy comparison.

    Args:
        p_values: Array of p-values from multiple hypothesis tests
        labels: Optional labels for each test
        alpha: Significance level (default: 0.05)

    Returns:
        Dictionary with results from all correction methods and summary table
    """
    p_values = np.array(p_values)
    n_tests = len(p_values)

    if labels is None:
        labels = [f"Test_{i + 1}" for i in range(n_tests)]

    # Apply all corrections
    bonf = bonferroni_correction(p_values, alpha)
    holm = holm_bonferroni_correction(p_values, alpha)
    bh = benjamini_hochberg_correction(p_values, alpha)

    # Build summary table
    summary = []
    for i in range(n_tests):
        summary.append({
            "label": labels[i],
            "p_value": p_values[i],
            "bonferroni_adj_p": bonf["adjusted_p_values"][i],
            "bonferroni_sig": bonf["significant"][i],
            "holm_adj_p": holm["adjusted_p_values"][i],
            "holm_sig": holm["significant"][i],
            "bh_adj_p": bh["adjusted_p_values"][i],
            "bh_sig": bh["significant"][i],
        })

    return {
        "bonferroni": bonf,
        "holm_bonferroni": holm,
        "benjamini_hochberg": bh,
        "summary": summary,
        "recommendation": _recommend_correction(n_tests, alpha),
    }


def _recommend_correction(n_tests: int, alpha: float) -> str:
    """Provide recommendation on which correction to use."""
    if n_tests <= 5:
        return (
            "With few tests, Bonferroni is appropriate and easy to interpret. "
            "All methods will give similar results."
        )
    elif n_tests <= 20:
        return (
            "Holm-Bonferroni is recommended as it maintains FWER control while "
            "being less conservative than Bonferroni."
        )
    else:
        return (
            "With many tests, Benjamini-Hochberg (FDR control) is recommended "
            "unless strict FWER control is required. BH allows more discoveries "
            "while controlling the expected proportion of false positives."
        )


def paired_comparisons_with_correction(
    baseline_results: Dict[str, List[float]],
    method_results: Dict[str, List[float]],
    alpha: float = 0.05,
    test: str = "paired_t",
) -> Dict:
    """
    Perform pairwise comparisons with multiple testing correction.

    Useful for comparing a method against multiple baselines across datasets.

    Args:
        baseline_results: Dict mapping dataset names to baseline metric values
        method_results: Dict mapping dataset names to method metric values
        alpha: Significance level
        test: Statistical test to use ('paired_t' or 'wilcoxon')

    Returns:
        Dictionary with raw p-values, corrected p-values, and significance
    """
    datasets = list(baseline_results.keys())
    p_values = []
    test_results = []

    for dataset in datasets:
        baseline = np.array(baseline_results[dataset])
        method = np.array(method_results[dataset])

        if test == "paired_t":
            stat, p = stats.ttest_rel(method, baseline)
        elif test == "wilcoxon":
            try:
                stat, p = stats.wilcoxon(method, baseline)
            except ValueError:
                stat, p = 0.0, 1.0
        else:
            raise ValueError(f"Unknown test: {test}")

        p_values.append(p)
        test_results.append({
            "dataset": dataset,
            "statistic": stat,
            "p_value": p,
            "mean_diff": np.mean(method) - np.mean(baseline),
        })

    # Apply corrections
    corrections = multiple_comparison_summary(p_values, datasets, alpha)

    # Combine results
    for i, result in enumerate(test_results):
        result["bonferroni_sig"] = corrections["bonferroni"]["significant"][i]
        result["holm_sig"] = corrections["holm_bonferroni"]["significant"][i]
        result["bh_sig"] = corrections["benjamini_hochberg"]["significant"][i]

    return {
        "test_results": test_results,
        "corrections": corrections,
        "n_significant_bonferroni": corrections["bonferroni"]["n_significant"],
        "n_significant_holm": corrections["holm_bonferroni"]["n_significant"],
        "n_significant_bh": corrections["benjamini_hochberg"]["n_significant"],
    }
