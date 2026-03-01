"""
Decision Rule Calibration for DKO.

This module implements SCC-based decision rules for selecting between
single-conformer and ensemble methods.

Key insight from research plan: SCC (Structural Conformational Complexity)
can predict when ensemble methods like DKO will outperform single-conformer
approaches. This enables practical deployment decisions.

Decision Rule:
    IF SCC(molecule) > threshold THEN use DKO
    ELSE use single-conformer baseline

Calibration:
    1. Compute SCC for all molecules in dataset
    2. Compute DKO advantage for each molecule
    3. Search for optimal threshold that maximizes decision accuracy
    4. Compute regret (cost of suboptimal decisions)
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
import json
from scipy import stats as scipy_stats

from dko.utils.logging_utils import get_logger

logger = get_logger("decision_rule")


@dataclass
class DecisionRuleResult:
    """Results from decision rule calibration."""

    # Threshold search
    optimal_threshold: float = 0.0
    threshold_accuracy: float = 0.0
    thresholds_tested: List[float] = field(default_factory=list)
    accuracies: List[float] = field(default_factory=list)

    # Regret analysis
    mean_regret: float = 0.0
    max_regret: float = 0.0
    regret_distribution: Dict = field(default_factory=dict)

    # Validation
    cross_val_accuracy: float = 0.0
    cross_val_std: float = 0.0

    # Dataset info
    n_molecules: int = 0
    n_high_scc: int = 0
    n_low_scc: int = 0

    def to_dict(self) -> Dict:
        return {
            "threshold": {
                "optimal": self.optimal_threshold,
                "accuracy": self.threshold_accuracy,
                "tested": self.thresholds_tested,
                "accuracies": self.accuracies,
            },
            "regret": {
                "mean": self.mean_regret,
                "max": self.max_regret,
                "distribution": self.regret_distribution,
            },
            "validation": {
                "cross_val_accuracy": self.cross_val_accuracy,
                "cross_val_std": self.cross_val_std,
            },
            "dataset": {
                "n_molecules": self.n_molecules,
                "n_high_scc": self.n_high_scc,
                "n_low_scc": self.n_low_scc,
            },
        }


def run_decision_rule_experiment(
    all_results: Optional[Dict] = None,
    dataset_properties: Optional[Dict] = None,
    datasets: Optional[List[str]] = None,
    output_dir: str = "results/decision_rule",
    seeds: List[int] = [42, 123, 456],
    device: str = "cuda",
    n_threshold_points: int = 20,
) -> DecisionRuleResult:
    """
    Calibrate SCC-based decision rule.

    Finds optimal SCC threshold for deciding between DKO and single-conformer.

    Args:
        all_results: Pre-computed results (optional)
        dataset_properties: Dataset property annotations
        datasets: Datasets to use for calibration
        output_dir: Output directory
        seeds: Random seeds
        device: Device for training
        n_threshold_points: Number of threshold values to test

    Returns:
        DecisionRuleResult with calibrated threshold and analysis
    """
    from dko.analysis.scc import StructuralConformationalComplexity

    logger.info("=" * 70)
    logger.info("DECISION RULE CALIBRATION")
    logger.info("=" * 70)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # If no pre-computed results, run experiments
    if all_results is None:
        all_results, scc_values, advantages = compute_scc_and_advantages(
            datasets=datasets,
            seeds=seeds,
            device=device,
        )
    else:
        # Extract SCC values and advantages from results
        scc_values, advantages = extract_scc_advantages(all_results, dataset_properties)

    if len(scc_values) < 5:
        logger.warning("Insufficient data for calibration")
        return DecisionRuleResult()

    # Initialize result
    result = DecisionRuleResult(n_molecules=len(scc_values))

    # Grid search for optimal threshold
    result = calibrate_threshold_grid_search(
        scc_values=np.array(scc_values),
        advantages=np.array(advantages),
        result=result,
        n_points=n_threshold_points,
    )

    # Compute regret
    result = compute_regret_analysis(
        scc_values=np.array(scc_values),
        advantages=np.array(advantages),
        threshold=result.optimal_threshold,
        result=result,
    )

    # Cross-validation
    result = cross_validate_threshold(
        scc_values=np.array(scc_values),
        advantages=np.array(advantages),
        result=result,
        n_folds=5,
    )

    # Log results
    logger.info("\n" + "=" * 70)
    logger.info("CALIBRATION RESULTS")
    logger.info("=" * 70)
    logger.info(f"Optimal SCC threshold: {result.optimal_threshold:.4f}")
    logger.info(f"Decision accuracy: {result.threshold_accuracy:.1%}")
    logger.info(f"Cross-validation accuracy: {result.cross_val_accuracy:.1%} +/- {result.cross_val_std:.1%}")
    logger.info("-" * 70)
    logger.info(f"Mean regret: {result.mean_regret:.4f}")
    logger.info(f"Max regret: {result.max_regret:.4f}")
    logger.info("-" * 70)
    logger.info(f"High-SCC molecules: {result.n_high_scc} ({result.n_high_scc/result.n_molecules:.1%})")
    logger.info(f"Low-SCC molecules: {result.n_low_scc} ({result.n_low_scc/result.n_molecules:.1%})")
    logger.info("=" * 70)

    # Decision rule summary
    logger.info("\nDECISION RULE:")
    logger.info(f"  IF SCC > {result.optimal_threshold:.4f}:")
    logger.info("    USE DKO (ensemble method)")
    logger.info("  ELSE:")
    logger.info("    USE single-conformer baseline")
    logger.info(f"\n  Expected accuracy: {result.cross_val_accuracy:.1%}")

    # Save results
    output_file = output_path / "decision_rule_calibration.json"
    with open(output_file, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    logger.info(f"\nResults saved to {output_file}")

    return result


def compute_scc_and_advantages(
    datasets: Optional[List[str]] = None,
    seeds: List[int] = [42, 123, 456],
    device: str = "cuda",
    n_epochs: int = 50,
) -> Tuple[Dict, List[float], List[float]]:
    """
    Compute SCC values and DKO advantages for molecules.

    Args:
        datasets: Datasets to use
        seeds: Random seeds
        device: Device
        n_epochs: Training epochs

    Returns:
        Tuple of (results_dict, scc_values, advantages)
    """
    from dko.data.datasets import load_dataset
    from dko.models.dko import DKO
    from dko.models.ensemble_baselines import SingleConformer
    from dko.training.trainer import Trainer
    from dko.analysis.scc import StructuralConformationalComplexity

    if datasets is None:
        datasets = ["esol", "freesolv", "bace", "lipophilicity"]

    scc_calculator = StructuralConformationalComplexity()
    all_scc = []
    all_advantages = []
    all_results = {}

    for dataset_name in datasets:
        logger.info(f"\nDataset: {dataset_name}")

        try:
            dataset = load_dataset(dataset_name)
            feature_dim = dataset.feature_dim
            task = dataset.task

            metric = "rmse" if task == "regression" else "auroc"
            lower_is_better = task == "regression"

            # Compute SCC for molecules
            dataset_scc = []
            for batch in dataset.train_loader:
                features = batch.get("features", None)
                if features is not None:
                    for i in range(features.shape[0]):
                        mol_features = features[i].numpy()
                        # Convert to list of conformer features
                        features_list = [mol_features[j] for j in range(mol_features.shape[0])]
                        scc = scc_calculator.compute(features_list)
                        dataset_scc.append(scc)

            mean_scc = np.mean(dataset_scc) if dataset_scc else 0.0
            logger.info(f"  Mean SCC: {mean_scc:.4f}")

            # Train models
            dko_perfs = []
            sc_perfs = []

            for seed in seeds:
                # DKO
                dko = DKO(
                    feature_dim=feature_dim,
                    output_dim=1,
                    task=task,
                    use_second_order=True,
                    verbose=False,
                )
                trainer = Trainer(
                    model=dko,
                    config={"lr": 1e-4, "n_epochs": n_epochs, "seed": seed},
                    device=device,
                )
                trainer.fit(dataset.train_loader, dataset.val_loader)
                test_metrics = trainer.evaluate(dataset.test_loader)
                dko_perfs.append(test_metrics[metric])

                # Single conformer
                sc = SingleConformer(feature_dim=feature_dim, num_outputs=1)
                trainer = Trainer(
                    model=sc,
                    config={"lr": 1e-4, "n_epochs": n_epochs, "seed": seed},
                    device=device,
                )
                trainer.fit(dataset.train_loader, dataset.val_loader)
                test_metrics = trainer.evaluate(dataset.test_loader)
                sc_perfs.append(test_metrics[metric])

            dko_mean = np.mean(dko_perfs)
            sc_mean = np.mean(sc_perfs)

            # Compute advantage (positive = DKO better)
            if lower_is_better:
                advantage = (sc_mean - dko_mean) / sc_mean * 100
            else:
                advantage = (dko_mean - sc_mean) / dko_mean * 100

            logger.info(f"  DKO: {dko_mean:.4f}, Single: {sc_mean:.4f}")
            logger.info(f"  Advantage: {advantage:.2f}%")

            all_scc.append(mean_scc)
            all_advantages.append(advantage)

            all_results[dataset_name] = {
                "scc": mean_scc,
                "dko": dko_mean,
                "single_conformer": sc_mean,
                "advantage": advantage,
            }

        except Exception as e:
            logger.error(f"Failed on {dataset_name}: {e}")
            continue

    return all_results, all_scc, all_advantages


def extract_scc_advantages(
    all_results: Dict,
    dataset_properties: Optional[Dict],
) -> Tuple[List[float], List[float]]:
    """Extract SCC values and advantages from results dict."""
    scc_values = []
    advantages = []

    for dataset, results in all_results.items():
        scc = results.get("scc", 0)
        if scc == 0 and dataset_properties and dataset in dataset_properties:
            scc = dataset_properties[dataset].get("scc", 0)

        dko = results.get("dko", {})
        baseline = results.get("single_conformer", {})

        if isinstance(dko, dict):
            dko_metric = dko.get("rmse", {}).get("mean", dko.get("mean", 0))
        else:
            dko_metric = dko

        if isinstance(baseline, dict):
            baseline_metric = baseline.get("rmse", {}).get("mean", baseline.get("mean", 0))
        else:
            baseline_metric = baseline

        if baseline_metric > 0:
            advantage = (baseline_metric - dko_metric) / baseline_metric * 100
            scc_values.append(scc)
            advantages.append(advantage)

    return scc_values, advantages


def calibrate_threshold_grid_search(
    scc_values: np.ndarray,
    advantages: np.ndarray,
    result: DecisionRuleResult,
    n_points: int = 20,
) -> DecisionRuleResult:
    """
    Find optimal threshold via grid search.

    Decision rule accuracy: fraction of correct predictions where
    - SCC > threshold AND advantage > 0 (correctly predicted DKO benefit)
    - SCC <= threshold AND advantage <= 0 (correctly predicted no benefit)
    """
    # Generate threshold grid
    scc_min, scc_max = scc_values.min(), scc_values.max()
    thresholds = np.linspace(scc_min, scc_max, n_points)

    best_accuracy = 0
    best_threshold = thresholds[n_points // 2]  # Default to median
    accuracies = []

    for threshold in thresholds:
        # Predictions: use DKO if SCC > threshold
        predictions = scc_values > threshold

        # Ground truth: DKO is better if advantage > 0
        ground_truth = advantages > 0

        # Accuracy
        correct = (predictions == ground_truth).sum()
        accuracy = correct / len(scc_values)
        accuracies.append(accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    result.optimal_threshold = float(best_threshold)
    result.threshold_accuracy = float(best_accuracy)
    result.thresholds_tested = thresholds.tolist()
    result.accuracies = accuracies

    # Count high/low SCC
    result.n_high_scc = int((scc_values > best_threshold).sum())
    result.n_low_scc = int((scc_values <= best_threshold).sum())

    return result


def compute_regret_analysis(
    scc_values: np.ndarray,
    advantages: np.ndarray,
    threshold: float,
    result: DecisionRuleResult,
) -> DecisionRuleResult:
    """
    Compute regret from following the decision rule.

    Regret = performance loss from following decision rule vs oracle.

    For each molecule:
    - If rule says use DKO but single was better: regret = |advantage| (negative advantage)
    - If rule says use single but DKO was better: regret = advantage (positive advantage)
    """
    predictions = scc_values > threshold
    ground_truth = advantages > 0

    regrets = []
    for pred, truth, adv in zip(predictions, ground_truth, advantages):
        if pred != truth:
            # Wrong decision
            regrets.append(abs(adv))
        else:
            # Correct decision
            regrets.append(0)

    regrets = np.array(regrets)

    result.mean_regret = float(regrets.mean())
    result.max_regret = float(regrets.max())
    result.regret_distribution = {
        "percentiles": {
            "50": float(np.percentile(regrets, 50)),
            "90": float(np.percentile(regrets, 90)),
            "95": float(np.percentile(regrets, 95)),
            "99": float(np.percentile(regrets, 99)),
        },
        "mean": float(regrets.mean()),
        "std": float(regrets.std()),
    }

    return result


def cross_validate_threshold(
    scc_values: np.ndarray,
    advantages: np.ndarray,
    result: DecisionRuleResult,
    n_folds: int = 5,
) -> DecisionRuleResult:
    """
    Cross-validate threshold calibration.

    Uses k-fold CV to estimate out-of-sample accuracy.
    """
    from sklearn.model_selection import KFold

    n = len(scc_values)
    if n < n_folds:
        result.cross_val_accuracy = result.threshold_accuracy
        result.cross_val_std = 0.0
        return result

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_accuracies = []

    for train_idx, test_idx in kf.split(scc_values):
        # Calibrate on train fold
        train_scc = scc_values[train_idx]
        train_adv = advantages[train_idx]

        # Simple grid search on train
        thresholds = np.linspace(train_scc.min(), train_scc.max(), 10)
        best_thresh = thresholds[5]
        best_acc = 0

        for t in thresholds:
            pred = train_scc > t
            truth = train_adv > 0
            acc = (pred == truth).mean()
            if acc > best_acc:
                best_acc = acc
                best_thresh = t

        # Evaluate on test fold
        test_scc = scc_values[test_idx]
        test_adv = advantages[test_idx]
        test_pred = test_scc > best_thresh
        test_truth = test_adv > 0
        test_acc = (test_pred == test_truth).mean()
        fold_accuracies.append(test_acc)

    result.cross_val_accuracy = float(np.mean(fold_accuracies))
    result.cross_val_std = float(np.std(fold_accuracies))

    return result


def apply_decision_rule(
    scc: float,
    threshold: float,
) -> str:
    """
    Apply calibrated decision rule.

    Args:
        scc: SCC value for molecule
        threshold: Calibrated threshold

    Returns:
        "dko" or "single_conformer"
    """
    return "dko" if scc > threshold else "single_conformer"


def batch_apply_decision_rule(
    scc_values: List[float],
    threshold: float,
) -> List[str]:
    """Apply decision rule to batch of molecules."""
    return [apply_decision_rule(scc, threshold) for scc in scc_values]
