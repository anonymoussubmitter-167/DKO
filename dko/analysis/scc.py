"""
Structural Conformational Complexity (SCC) and Diagnostic Utilities.

This module provides tools for:
1. Statistical Consistency Checks (SCC) for model comparison
2. Structural Conformational Complexity (SCC) metric for decision rules
3. Diagnostics to predict when second-order DKO will help

Key insight from Theorem 1: SCC is NECESSARY but NOT SUFFICIENT for
second-order improvement. High SCC indicates conformational variability
exists, but whether the target property depends on this variability
must be determined empirically.

VALIDATED FINDINGS (2026-01)
============================

1. SCC-Label Correlation is CONFOUNDED:
   - Larger molecules have both more conformational variance AND different properties
   - High SCC-label correlation does NOT reliably predict second-order benefit
   - This diagnostic can identify NEGATIVE controls but not POSITIVE controls

2. Residual Diagnostic is MORE RELIABLE:
   - Trains first-order model, then tests: do errors correlate with SCC?
   - Directly answers: "Does first-order fail on conformationally complex molecules?"
   - If yes -> second-order may capture what first-order misses

3. Dataset Classifications (validated experimentally):
   - FreeSolv: POSITIVE CONTROL - second-order improves RMSE by ~3%
     (Residual diagnostic: r=0.26, high/low ratio=1.44x)
   - ESOL, Lipophilicity, QM9: NEGATIVE CONTROLS - second-order doesn't help
     (Residual diagnostic: r < 0.15 for all)
   - BACE, BBBP: Classification tasks - decomposition less meaningful

4. Theoretical Basis:
   - Theorem 1: SCC bounds potential benefit, but benefit requires property-SCC coupling
   - Theorem 4: Conformational entropy (relevant for binding/solvation) is recoverable
     from covariance - explains why FreeSolv benefits from second-order

Recommended diagnostic workflow:
1. compute_sigma_label_correlation() - Quick check, but confounded by molecular size
2. diagnose_dataset_for_second_order() - Comprehensive, but still confounded
3. run_residual_diagnostic() - BEST: directly tests if first-order fails on high-SCC molecules

The residual diagnostic answers: "Does first-order systematically fail on
conformationally complex molecules?" If yes, second-order might help.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats


class StatisticalConsistencyChecker:
    """
    Statistical Consistency Checker for model comparison.

    Provides methods for:
    - Testing significance of improvements
    - Computing effect sizes
    - Validating result consistency across seeds
    """

    def __init__(
        self,
        alpha: float = 0.05,
        min_seeds: int = 3,
        effect_size_threshold: float = 0.2,
    ):
        """
        Initialize SCC.

        Args:
            alpha: Significance level
            min_seeds: Minimum number of seeds for valid comparison
            effect_size_threshold: Minimum effect size for practical significance
        """
        self.alpha = alpha
        self.min_seeds = min_seeds
        self.effect_size_threshold = effect_size_threshold

    def check_significance(
        self,
        model1_values: List[float],
        model2_values: List[float],
        paired: bool = True,
    ) -> Dict:
        """
        Check statistical significance of difference.

        Args:
            model1_values: Values from model 1 across seeds
            model2_values: Values from model 2 across seeds
            paired: Whether to use paired test

        Returns:
            Dictionary with test results
        """
        model1_values = np.array(model1_values)
        model2_values = np.array(model2_values)

        n = len(model1_values)

        if n < self.min_seeds:
            return {
                "valid": False,
                "reason": f"Insufficient seeds ({n} < {self.min_seeds})",
            }

        # Compute means and difference
        mean1 = np.mean(model1_values)
        mean2 = np.mean(model2_values)
        diff = model1_values - model2_values

        # Statistical test
        if paired:
            t_stat, p_value = stats.ttest_rel(model1_values, model2_values)
        else:
            t_stat, p_value = stats.ttest_ind(model1_values, model2_values)

        # Effect size (Cohen's d for paired)
        if paired:
            cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0
        else:
            pooled_std = np.sqrt(
                (np.var(model1_values) + np.var(model2_values)) / 2
            )
            cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0

        # Confidence interval
        ci_margin = stats.t.ppf(1 - self.alpha / 2, df=n - 1) * np.std(diff, ddof=1) / np.sqrt(n)

        return {
            "valid": True,
            "mean_model1": mean1,
            "mean_model2": mean2,
            "mean_difference": np.mean(diff),
            "std_difference": np.std(diff, ddof=1),
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < self.alpha,
            "cohens_d": cohens_d,
            "effect_size": self._interpret_effect_size(cohens_d),
            "practically_significant": abs(cohens_d) >= self.effect_size_threshold,
            "ci_lower": np.mean(diff) - ci_margin,
            "ci_upper": np.mean(diff) + ci_margin,
            "n_seeds": n,
        }

    def _interpret_effect_size(self, d: float) -> str:
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

    def check_consistency(
        self,
        values: List[float],
        expected_cv: float = 0.1,
    ) -> Dict:
        """
        Check if results are consistent across seeds.

        Args:
            values: Metric values across seeds
            expected_cv: Expected coefficient of variation

        Returns:
            Consistency check results
        """
        values = np.array(values)
        n = len(values)

        if n < 2:
            return {"valid": False, "reason": "Need at least 2 values"}

        mean = np.mean(values)
        std = np.std(values, ddof=1)
        cv = std / abs(mean) if mean != 0 else float("inf")

        # Check for outliers using IQR
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        outliers = np.sum((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr))

        return {
            "valid": True,
            "mean": mean,
            "std": std,
            "cv": cv,
            "consistent": cv <= expected_cv,
            "n_outliers": int(outliers),
            "range": np.max(values) - np.min(values),
        }

    def validate_improvement(
        self,
        baseline_values: List[float],
        model_values: List[float],
        expected_improvement: float,
        lower_is_better: bool = True,
    ) -> Dict:
        """
        Validate that improvement meets expectations.

        Args:
            baseline_values: Baseline metric values
            model_values: Model metric values
            expected_improvement: Expected improvement percentage
            lower_is_better: Whether lower metric values are better

        Returns:
            Validation results
        """
        baseline_values = np.array(baseline_values)
        model_values = np.array(model_values)

        baseline_mean = np.mean(baseline_values)
        model_mean = np.mean(model_values)

        # Compute actual improvement
        if lower_is_better:
            actual_improvement = (baseline_mean - model_mean) / baseline_mean * 100
        else:
            actual_improvement = (model_mean - baseline_mean) / baseline_mean * 100

        # Check significance
        sig_result = self.check_significance(
            baseline_values if lower_is_better else model_values,
            model_values if lower_is_better else baseline_values,
        )

        # Check if improvement meets expectation (with some margin)
        margin = 0.2  # 20% margin
        meets_expectation = actual_improvement >= expected_improvement * (1 - margin)

        return {
            "expected_improvement": expected_improvement,
            "actual_improvement": actual_improvement,
            "meets_expectation": meets_expectation,
            "improvement_ratio": actual_improvement / expected_improvement if expected_improvement > 0 else float("inf"),
            **sig_result,
        }


class StructuralConformationalComplexity:
    """
    Structural Conformational Complexity (SCC) metric.

    SCC measures conformational variability as the total variance of geometric
    features across the conformer ensemble. This metric is used to predict
    a priori whether ensemble methods will provide benefit.

    From the research plan:
    - SCC = sum of variance of each geometric feature across conformers
    - Low SCC → single-conformer methods suffice
    - High SCC → ensemble methods may help

    Theorem 1 (SCC Upper Bound):
    For any Lipschitz-continuous property function, the ensemble advantage
    is bounded by: L * sqrt(SCC), where L is the Lipschitz constant.
    """

    def __init__(
        self,
        threshold: float = 1.0,
        use_boltzmann_weights: bool = True,
    ):
        """
        Initialize SCC calculator.

        Args:
            threshold: SCC threshold for decision rule (calibrated on validation)
            use_boltzmann_weights: Whether to use Boltzmann weights for variance calc
        """
        self.threshold = threshold
        self.use_boltzmann_weights = use_boltzmann_weights
        self._calibrated = False

    def compute(
        self,
        features_list: List[np.ndarray],
        weights: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute SCC as total variance of geometric features.

        SCC = Σ_i Var(f_i) = Σ_i E[(f_i - μ_i)²]

        where f_i is the i-th geometric feature and expectation is over conformers.

        Args:
            features_list: List of feature vectors, one per conformer
            weights: Optional Boltzmann weights (uniform if None)

        Returns:
            SCC score (sum of feature variances)
        """
        if len(features_list) < 2:
            return 0.0

        # Handle variable-length feature vectors by padding
        max_len = max(len(f) for f in features_list)
        padded_features = []
        for f in features_list:
            if len(f) < max_len:
                padded = np.pad(f, (0, max_len - len(f)), mode='constant')
                padded_features.append(padded)
            else:
                padded_features.append(f)

        features = np.stack(padded_features, axis=0)  # (n_conformers, feature_dim)
        n_conformers, feature_dim = features.shape

        # Handle weights
        if weights is None or not self.use_boltzmann_weights:
            weights = np.ones(n_conformers) / n_conformers
        else:
            weights = np.asarray(weights)
            weights = weights / weights.sum()

        # Weighted mean: μ_i = Σ_j w_j * f_ji
        mean = np.sum(weights[:, np.newaxis] * features, axis=0)

        # Weighted variance for each feature: Var(f_i) = Σ_j w_j * (f_ji - μ_i)²
        centered = features - mean[np.newaxis, :]
        variances = np.sum(weights[:, np.newaxis] * centered ** 2, axis=0)

        # SCC = total variance = sum of all feature variances
        scc = float(np.sum(variances))

        return scc

    def compute_from_ensemble(
        self,
        mol,
        conformer_ids: Optional[List[int]] = None,
        weights: Optional[np.ndarray] = None,
        extractor=None,
    ) -> float:
        """
        Compute SCC directly from RDKit molecule with conformers.

        Args:
            mol: RDKit Mol object with conformers
            conformer_ids: Conformer IDs to use (None for all)
            weights: Boltzmann weights
            extractor: GeometricFeatureExtractor instance

        Returns:
            SCC score
        """
        from dko.data.features import GeometricFeatureExtractor

        if extractor is None:
            extractor = GeometricFeatureExtractor()

        if conformer_ids is None:
            conformer_ids = list(range(mol.GetNumConformers()))

        if len(conformer_ids) < 2:
            return 0.0

        # Extract features for all conformers
        features_list = []
        for conf_id in conformer_ids:
            geo_features = extractor.extract(mol, conf_id)
            features_list.append(geo_features.to_flat_vector())

        return self.compute(features_list, weights)

    def predict_ensemble_benefit(self, scc: float) -> bool:
        """
        Decision rule: should we use ensemble methods?

        Args:
            scc: SCC score for a molecule/dataset

        Returns:
            True if ensemble methods recommended, False otherwise
        """
        return scc > self.threshold

    def calibrate_threshold(
        self,
        scc_values: List[float],
        ensemble_advantages: List[float],
        target_accuracy: float = 0.85,
    ) -> float:
        """
        Calibrate the SCC threshold to maximize decision accuracy.

        Args:
            scc_values: SCC scores for validation molecules
            ensemble_advantages: Actual ensemble vs single-conformer improvement
            target_accuracy: Target decision accuracy

        Returns:
            Calibrated threshold
        """
        scc_values = np.array(scc_values)
        advantages = np.array(ensemble_advantages)

        # True labels: ensemble helps if advantage > 0
        true_labels = advantages > 0

        # Search for optimal threshold
        best_threshold = 0.0
        best_accuracy = 0.0

        # Try percentiles as candidate thresholds
        for percentile in range(5, 96, 5):
            threshold = np.percentile(scc_values, percentile)
            predictions = scc_values > threshold
            accuracy = np.mean(predictions == true_labels)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold

        self.threshold = best_threshold
        self._calibrated = True

        return best_threshold

    def compute_regret(
        self,
        scc_values: List[float],
        ensemble_advantages: List[float],
    ) -> Dict:
        """
        Compute regret from following SCC decision rule vs oracle.

        Regret = performance loss from following SCC rule instead of
        always choosing the better method (oracle).

        Args:
            scc_values: SCC scores
            ensemble_advantages: Actual improvements from ensemble

        Returns:
            Dict with regret statistics
        """
        scc_values = np.array(scc_values)
        advantages = np.array(ensemble_advantages)

        # Predictions from SCC rule
        predictions = scc_values > self.threshold

        # Oracle always chooses correctly
        oracle_choices = advantages > 0

        # Compute regret
        # When we recommend single but ensemble was better: we lose the advantage
        # When we recommend ensemble but single was better: we lose |advantage|
        regret = np.zeros_like(advantages)

        # False negatives (missed ensemble benefit)
        fn_mask = (~predictions) & oracle_choices
        regret[fn_mask] = advantages[fn_mask]

        # False positives (unnecessary ensemble)
        fp_mask = predictions & (~oracle_choices)
        regret[fp_mask] = -advantages[fp_mask]  # advantage is negative here

        return {
            "mean_regret": float(np.mean(regret)),
            "max_regret": float(np.max(regret)),
            "total_regret": float(np.sum(regret)),
            "accuracy": float(np.mean(predictions == oracle_choices)),
            "false_positive_rate": float(np.mean(fp_mask)),
            "false_negative_rate": float(np.mean(fn_mask)),
            "threshold": self.threshold,
        }


def compute_dataset_scc(
    smiles_list: List[str],
    n_conformers: int = 50,
    show_progress: bool = True,
) -> Dict:
    """
    Compute SCC statistics for a dataset.

    Args:
        smiles_list: List of SMILES strings
        n_conformers: Number of conformers per molecule
        show_progress: Whether to show progress bar

    Returns:
        Dict with SCC statistics
    """
    from dko.data.conformers import ConformerGenerator
    from dko.data.features import GeometricFeatureExtractor

    generator = ConformerGenerator(max_conformers=n_conformers)
    extractor = GeometricFeatureExtractor()
    scc_calc = StructuralConformationalComplexity()

    scc_values = []

    iterator = smiles_list
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(smiles_list, desc="Computing SCC")
        except ImportError:
            pass

    for smiles in iterator:
        try:
            ensemble = generator.generate_from_smiles(smiles)
            if ensemble.n_conformers >= 2:
                scc = scc_calc.compute_from_ensemble(
                    ensemble.mol,
                    ensemble.conformer_ids,
                    ensemble.boltzmann_weights,
                    extractor,
                )
                scc_values.append(scc)
        except Exception:
            continue

    scc_array = np.array(scc_values)

    return {
        "mean": float(np.mean(scc_array)) if len(scc_array) > 0 else 0.0,
        "std": float(np.std(scc_array)) if len(scc_array) > 0 else 0.0,
        "median": float(np.median(scc_array)) if len(scc_array) > 0 else 0.0,
        "min": float(np.min(scc_array)) if len(scc_array) > 0 else 0.0,
        "max": float(np.max(scc_array)) if len(scc_array) > 0 else 0.0,
        "quartiles": {
            "q1": float(np.percentile(scc_array, 25)) if len(scc_array) > 0 else 0.0,
            "q2": float(np.percentile(scc_array, 50)) if len(scc_array) > 0 else 0.0,
            "q3": float(np.percentile(scc_array, 75)) if len(scc_array) > 0 else 0.0,
        },
        "n_molecules": len(scc_values),
        "values": scc_values,
    }


def compute_scc_scores(
    results: Dict[str, Dict[str, List[float]]],
    baseline_key: str = "single_conformer",
    metric_key: str = "rmse",
) -> Dict[str, Dict]:
    """
    Compute SCC scores for all model comparisons.

    Args:
        results: Dictionary of model name -> metric name -> values
        baseline_key: Key for baseline model
        metric_key: Metric to compare

    Returns:
        Dictionary of model name -> SCC results
    """
    scc = StatisticalConsistencyChecker()
    scores = {}

    baseline_values = results.get(baseline_key, {}).get(metric_key, [])

    if not baseline_values:
        return {"error": "Baseline values not found"}

    for model_name, metrics in results.items():
        if model_name == baseline_key:
            continue

        model_values = metrics.get(metric_key, [])
        if model_values:
            scores[model_name] = scc.check_significance(baseline_values, model_values)

    return scores


def validate_scc(
    results: Dict,
    expected_improvements: Dict[str, float],
    metric_key: str = "rmse",
) -> Dict:
    """
    Validate results against expected improvements.

    Args:
        results: Experimental results
        expected_improvements: Expected improvement per dataset
        metric_key: Metric to validate

    Returns:
        Validation summary
    """
    scc = StatisticalConsistencyChecker()
    validation = {}

    for dataset, expected_imp in expected_improvements.items():
        dataset_results = results.get(dataset, {})

        baseline = dataset_results.get("single_conformer", {}).get(metric_key, [])
        model = dataset_results.get("dko", {}).get(metric_key, [])

        if baseline and model:
            validation[dataset] = scc.validate_improvement(
                baseline, model, expected_imp, lower_is_better=True
            )
        else:
            validation[dataset] = {"valid": False, "reason": "Missing data"}

    return validation


def compute_sigma_label_correlation(
    sigmas: np.ndarray,
    labels: np.ndarray,
    method: str = "trace",
) -> Dict:
    """
    Diagnostic: Compute correlation between sigma (covariance) features and labels.

    This determines whether second-order features contain predictive signal
    BEFORE running full experiments. Low correlation suggests the dataset
    is a negative control for second-order DKO.

    Args:
        sigmas: (N, D, D) covariance matrices or (N, D) diagonal features
        labels: (N,) target values
        method: How to summarize sigma - "trace", "frobenius", "diagonal_sum", "max_var"

    Returns:
        Dictionary with correlation metrics and recommendation
    """
    sigmas = np.asarray(sigmas)
    labels = np.asarray(labels).flatten()

    n_samples = len(labels)

    # Extract scalar summary of sigma for each sample
    if sigmas.ndim == 3:
        # Full covariance matrix (N, D, D)
        if method == "trace":
            sigma_summary = np.trace(sigmas, axis1=1, axis2=2)
        elif method == "frobenius":
            sigma_summary = np.linalg.norm(sigmas.reshape(n_samples, -1), axis=1)
        elif method == "diagonal_sum":
            sigma_summary = np.diagonal(sigmas, axis1=1, axis2=2).sum(axis=1)
        elif method == "max_var":
            sigma_summary = np.diagonal(sigmas, axis1=1, axis2=2).max(axis=1)
        else:
            sigma_summary = np.trace(sigmas, axis1=1, axis2=2)
    elif sigmas.ndim == 2:
        # Already flattened or diagonal (N, D)
        sigma_summary = sigmas.sum(axis=1)
    else:
        raise ValueError(f"sigmas must be 2D or 3D, got shape {sigmas.shape}")

    # Compute correlations
    pearson_r, pearson_p = stats.pearsonr(sigma_summary, labels)
    spearman_r, spearman_p = stats.spearmanr(sigma_summary, labels)

    # Also compute correlation with individual diagonal elements
    if sigmas.ndim == 3:
        diag = np.diagonal(sigmas, axis1=1, axis2=2)  # (N, D)
    else:
        diag = sigmas  # (N, D)

    # Find the diagonal element most correlated with labels
    max_diag_corr = 0.0
    max_diag_idx = 0
    for i in range(diag.shape[1]):
        r, _ = stats.pearsonr(diag[:, i], labels)
        if abs(r) > abs(max_diag_corr):
            max_diag_corr = r
            max_diag_idx = i

    # Recommendation based on correlation strength
    abs_corr = max(abs(pearson_r), abs(spearman_r))
    if abs_corr < 0.1:
        recommendation = "NEGATIVE_CONTROL"
        reason = "Sigma features show no correlation with labels"
    elif abs_corr < 0.2:
        recommendation = "LIKELY_NEGATIVE_CONTROL"
        reason = "Sigma features show weak correlation with labels"
    elif abs_corr < 0.3:
        recommendation = "UNCERTAIN"
        reason = "Sigma features show moderate correlation - test empirically"
    else:
        recommendation = "EXPECT_IMPROVEMENT"
        reason = "Sigma features show meaningful correlation with labels"

    return {
        "method": method,
        "n_samples": n_samples,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "max_diagonal_correlation": float(max_diag_corr),
        "max_diagonal_index": int(max_diag_idx),
        "recommendation": recommendation,
        "reason": reason,
    }


def diagnose_dataset_for_second_order(
    features_list: List[List[np.ndarray]],
    labels: List[float],
    weights_list: Optional[List[np.ndarray]] = None,
) -> Dict:
    """
    Quick diagnostic for whether a dataset will benefit from second-order DKO.

    LIMITATION: This diagnostic uses SCC-label correlation, which is CONFOUNDED
    by molecular size (larger molecules have both more variance AND different
    properties). It can reliably identify NEGATIVE controls (low correlation),
    but cannot reliably identify POSITIVE controls (high correlation may be
    confounded).

    For a more reliable diagnostic, use run_residual_diagnostic() which directly
    tests whether first-order model errors correlate with conformational complexity.

    Classifications:
    - EXPECT_IMPROVEMENT: High SCC-label correlation (but may be confounded!)
    - NEGATIVE_CONTROL: Low SCC-label correlation (reliable)

    Args:
        features_list: List of conformer feature lists, one per molecule
        labels: Target values for each molecule
        weights_list: Optional Boltzmann weights for each molecule

    Returns:
        Dictionary with SCC stats, sigma-label correlation, and recommendation

    See Also:
        run_residual_diagnostic: More reliable diagnostic (trains first-order model)
    """
    from dko.data.features import AugmentedBasisConstructor

    scc_calculator = StructuralConformationalComplexity()
    basis_constructor = AugmentedBasisConstructor()

    # Compute SCC and sigma for each molecule
    scc_values = []
    sigmas = []

    for i, mol_features in enumerate(features_list):
        if len(mol_features) < 2:
            continue

        weights = weights_list[i] if weights_list else None

        # Compute SCC
        scc = scc_calculator.compute(mol_features, weights)
        scc_values.append(scc)

        # Compute sigma (covariance)
        try:
            basis = basis_constructor.construct(mol_features, weights)
            sigmas.append(basis.second_order)
        except Exception:
            continue

    if len(sigmas) == 0:
        return {"valid": False, "reason": "Could not compute sigma for any molecules"}

    sigmas = np.array(sigmas)
    labels_arr = np.array(labels[:len(sigmas)])
    scc_values = np.array(scc_values)

    # SCC statistics
    scc_stats = {
        "mean": float(np.mean(scc_values)),
        "std": float(np.std(scc_values)),
        "median": float(np.median(scc_values)),
        "min": float(np.min(scc_values)),
        "max": float(np.max(scc_values)),
    }

    # Sigma-label correlation
    sigma_corr = compute_sigma_label_correlation(sigmas, labels_arr)

    # Also check SCC-label correlation (molecules with more variance = different labels?)
    scc_label_r, scc_label_p = stats.pearsonr(scc_values, labels_arr)

    # Final recommendation
    sigma_predictive = abs(sigma_corr["pearson_r"]) > 0.15
    scc_predictive = abs(scc_label_r) > 0.15

    if sigma_predictive or scc_predictive:
        final_recommendation = "EXPECT_IMPROVEMENT"
        final_reason = "Second-order features show correlation with target property"
    else:
        final_recommendation = "NEGATIVE_CONTROL"
        final_reason = "Second-order features show no correlation with target - use first-order only"

    return {
        "valid": True,
        "n_molecules": len(sigmas),
        "scc_stats": scc_stats,
        "scc_label_correlation": float(scc_label_r),
        "scc_label_p_value": float(scc_label_p),
        "sigma_label_analysis": sigma_corr,
        "recommendation": final_recommendation,
        "reason": final_reason,
    }


def run_residual_diagnostic(
    train_features: List[List[np.ndarray]],
    train_labels: List[float],
    test_features: List[List[np.ndarray]],
    test_labels: List[float],
    train_weights: Optional[List[np.ndarray]] = None,
    test_weights: Optional[List[np.ndarray]] = None,
    feature_dim: int = 256,
    max_epochs: int = 50,
    device: str = "auto",
) -> Dict:
    """
    Residual Analysis Diagnostic for second-order DKO.

    This diagnostic answers: "Does first-order systematically fail on high-SCC molecules?"

    Unlike SCC-label correlation (which is confounded by molecular size), this directly
    tests whether first-order model errors correlate with conformational complexity.
    If they do, second-order features might capture what first-order misses.

    Based on Theorem 1: SCC is necessary but not sufficient for second-order benefit.
    This diagnostic empirically tests the "sufficient" part.

    Args:
        train_features: List of conformer feature lists for training molecules
        train_labels: Training labels
        test_features: List of conformer feature lists for test molecules
        test_labels: Test labels
        train_weights: Optional Boltzmann weights for training molecules
        test_weights: Optional Boltzmann weights for test molecules
        feature_dim: Fixed feature dimension for padding/truncation
        max_epochs: Maximum training epochs for first-order model
        device: Device to train on ("auto", "cuda", or "cpu")

    Returns:
        Dictionary with:
        - residual_scc_correlation: Pearson r between |residuals| and SCC
        - high_low_ratio: Ratio of mean residual for high-SCC vs low-SCC molecules
        - recommendation: LIKELY_IMPROVEMENT, POSSIBLE_IMPROVEMENT, or UNLIKELY_IMPROVEMENT
        - reason: Explanation of recommendation
    """
    import torch
    from torch.utils.data import Dataset, DataLoader

    from dko.models.dko import DKOFirstOrder
    from dko.training.trainer import Trainer
    from dko.data.features import AugmentedBasisConstructor

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    basis_constructor = AugmentedBasisConstructor()

    def prepare_data_with_scc(features_list, labels_list, weights_list, fixed_dim):
        """Prepare data and compute SCC for each molecule."""
        if weights_list is None:
            weights_list = [None] * len(features_list)

        mus, sigmas, labels, scc_values = [], [], [], []

        for i in range(len(features_list)):
            mol_features = features_list[i]
            label = labels_list[i]
            weights = weights_list[i] if weights_list[i] is not None else None

            if len(mol_features) < 1:
                continue

            # Pad/truncate conformers
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
                w = np.ones(len(features)) / len(features)
            else:
                w = np.array(weights)
                w = w / w.sum()

            # Compute SCC (sum of weighted variances)
            if len(features) >= 2:
                mean = np.sum(w[:, np.newaxis] * features, axis=0)
                variances = np.sum(w[:, np.newaxis] * (features - mean) ** 2, axis=0)
                scc = variances.sum()
            else:
                scc = 0.0

            # Construct basis for DKO
            try:
                basis = basis_constructor.construct([f for f in features], w)
                mus.append(basis.mean)
                sigmas.append(basis.second_order)
                labels.append(float(label))
                scc_values.append(scc)
            except Exception:
                continue

        return np.array(mus), np.array(sigmas), np.array(labels), np.array(scc_values)

    # Prepare data
    train_mu, train_sigma, train_y, train_scc = prepare_data_with_scc(
        train_features, train_labels, train_weights, feature_dim
    )
    test_mu, test_sigma, test_y, test_scc = prepare_data_with_scc(
        test_features, test_labels, test_weights, feature_dim
    )

    if len(train_mu) < 50 or len(test_mu) < 20:
        return {
            "valid": False,
            "reason": "Insufficient data for residual diagnostic",
        }

    D = train_mu.shape[1]

    # Normalize
    mu_mean = train_mu.mean(axis=0)
    mu_std = train_mu.std(axis=0) + 1e-8
    train_mu_norm = (train_mu - mu_mean) / mu_std
    test_mu_norm = (test_mu - mu_mean) / mu_std

    y_mean = train_y.mean()
    y_std = train_y.std() + 1e-8
    train_y_norm = (train_y - y_mean) / y_std
    test_y_norm = (test_y - y_mean) / y_std

    # Simple dataset class
    class SimpleDataset(Dataset):
        def __init__(self, mu, sigma, labels):
            self.mu = torch.FloatTensor(mu)
            self.sigma = torch.FloatTensor(sigma)
            self.labels = torch.FloatTensor(labels).unsqueeze(1)

        def __len__(self):
            return len(self.mu)

        def __getitem__(self, idx):
            return {"mu": self.mu[idx], "sigma": self.sigma[idx], "label": self.labels[idx]}

    def collate_fn(batch):
        return {
            "mu": torch.stack([b["mu"] for b in batch]),
            "sigma": torch.stack([b["sigma"] for b in batch]),
            "label": torch.stack([b["label"] for b in batch]),
        }

    train_loader = DataLoader(
        SimpleDataset(train_mu_norm, train_sigma, train_y_norm),
        batch_size=32, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        SimpleDataset(test_mu_norm, test_sigma, test_y_norm),
        batch_size=32, shuffle=False, collate_fn=collate_fn
    )

    # Train first-order model
    model = DKOFirstOrder(feature_dim=D, output_dim=1, verbose=False)

    trainer = Trainer(
        model=model,
        task="regression",
        learning_rate=1e-4,
        weight_decay=1e-4,
        max_epochs=max_epochs,
        early_stopping_patience=10,
        use_wandb=False,
        device=device,
        verbose=False,
    )

    trainer.fit(train_loader, test_loader)

    # Get predictions on test set
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            mu = batch["mu"].to(device)
            sigma = batch["sigma"].to(device)
            preds = model(mu, sigma)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch["label"].numpy())

    preds = np.concatenate(all_preds).flatten()
    labels = np.concatenate(all_labels).flatten()

    # Compute residuals (denormalized)
    preds_denorm = preds * y_std + y_mean
    labels_denorm = labels * y_std + y_mean
    residuals = np.abs(labels_denorm - preds_denorm)

    # Correlate residuals with SCC
    n_test = min(len(residuals), len(test_scc))
    residuals = residuals[:n_test]
    scc = test_scc[:n_test]

    # Filter out zero-SCC molecules (single conformer)
    mask = scc > 0
    if mask.sum() < 10:
        return {
            "valid": False,
            "reason": "Too few multi-conformer molecules for residual analysis",
        }

    residuals_filtered = residuals[mask]
    scc_filtered = scc[mask]

    # Compute correlations
    pearson_r, pearson_p = stats.pearsonr(residuals_filtered, scc_filtered)
    spearman_r, spearman_p = stats.spearmanr(residuals_filtered, scc_filtered)

    # Check: do high-SCC molecules have larger residuals?
    median_scc = np.median(scc_filtered)
    high_scc_residuals = residuals_filtered[scc_filtered > median_scc]
    low_scc_residuals = residuals_filtered[scc_filtered <= median_scc]

    t_stat, t_pvalue = stats.ttest_ind(high_scc_residuals, low_scc_residuals)
    high_scc_mean = high_scc_residuals.mean()
    low_scc_mean = low_scc_residuals.mean()
    ratio = high_scc_mean / low_scc_mean if low_scc_mean > 0 else float("inf")

    # Recommendation based on correlation
    if pearson_r > 0.2 and pearson_p < 0.1:
        recommendation = "LIKELY_IMPROVEMENT"
        reason = "First-order errors correlate with SCC - second-order may help"
    elif pearson_r > 0.1 or ratio > 1.2:
        recommendation = "POSSIBLE_IMPROVEMENT"
        reason = "Weak signal that high-SCC molecules have larger first-order errors"
    else:
        recommendation = "UNLIKELY_IMPROVEMENT"
        reason = "First-order errors don't correlate with SCC - second-order unlikely to help"

    return {
        "valid": True,
        "n_test": n_test,
        "n_multiconf": int(mask.sum()),
        "residual_scc_pearson_r": float(pearson_r),
        "residual_scc_pearson_p": float(pearson_p),
        "residual_scc_spearman_r": float(spearman_r),
        "residual_scc_spearman_p": float(spearman_p),
        "high_scc_mean_residual": float(high_scc_mean),
        "low_scc_mean_residual": float(low_scc_mean),
        "high_low_ratio": float(ratio),
        "t_statistic": float(t_stat),
        "t_pvalue": float(t_pvalue),
        "recommendation": recommendation,
        "reason": reason,
    }
