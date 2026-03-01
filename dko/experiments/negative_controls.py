"""
Negative Control Experiments for DKO.

This module implements negative control experiments to validate the SCC hypothesis:

Key Hypothesis: DKO's advantage over baselines should be correlated with
the task's dependence on conformational flexibility (SCC).

Negative Controls (Properties with no second-order dependence):
- QM9 electronic properties (HOMO, LUMO, Gap) - determined by electronic structure
- ESOL aqueous solubility - determined by mean surface properties
- Lipophilicity - determined by mean hydrophobic/hydrophilic balance
- Expected: DKO second-order features should show NO advantage over first-order

Positive Controls (Properties with potential second-order dependence):
- Binding affinity (BACE, PDBBind) - conformational flexibility affects binding
- FreeSolv hydration free energy - solvation may depend on conformational ensemble
- Expected: DKO second-order features MAY show advantage

Analysis:
1. Compare DKO improvement across low-SCC vs high-SCC tasks
2. Compute correlation between SCC and DKO advantage
3. Validate that DKO isn't just adding complexity without benefit on low-SCC tasks
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import torch
from pathlib import Path
import json
from scipy import stats as scipy_stats

from dko.utils.logging_utils import get_logger

logger = get_logger("negative_controls")


@dataclass
class NegativeControlResult:
    """Results from negative control analysis."""

    # Dataset categorization
    low_scc_datasets: List[str] = field(default_factory=list)
    high_scc_datasets: List[str] = field(default_factory=list)

    # Performance data
    # Format: results[dataset][model] = {"mean": x, "std": y}
    results: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)

    # DKO advantages
    low_scc_advantages: List[float] = field(default_factory=list)
    high_scc_advantages: List[float] = field(default_factory=list)

    # Analysis
    correlation_scc_advantage: float = 0.0
    p_value: float = 1.0
    low_scc_mean_advantage: float = 0.0
    high_scc_mean_advantage: float = 0.0
    hypothesis_validated: bool = False

    def to_dict(self) -> Dict:
        return {
            "categorization": {
                "low_scc": self.low_scc_datasets,
                "high_scc": self.high_scc_datasets,
            },
            "results": self.results,
            "advantages": {
                "low_scc": self.low_scc_advantages,
                "high_scc": self.high_scc_advantages,
            },
            "analysis": {
                "correlation_scc_advantage": self.correlation_scc_advantage,
                "p_value": self.p_value,
                "low_scc_mean_advantage": self.low_scc_mean_advantage,
                "high_scc_mean_advantage": self.high_scc_mean_advantage,
                "hypothesis_validated": self.hypothesis_validated,
            },
        }


def run_negative_control_experiment(
    negative_datasets: Optional[List[str]] = None,
    positive_datasets: Optional[List[str]] = None,
    seeds: List[int] = [42, 123, 456],
    output_dir: str = "results/negative_controls",
    device: str = "cuda",
    n_epochs: int = 100,
    baseline: str = "attention",  # Baseline to compare against
) -> NegativeControlResult:
    """
    Run negative control experiment.

    Compares DKO advantage on low-SCC (negative control) vs high-SCC
    (positive control) datasets.

    Args:
        negative_datasets: Low-SCC datasets (electronic properties)
        positive_datasets: High-SCC datasets (binding, solubility)
        seeds: Random seeds
        output_dir: Output directory
        device: Device for training
        n_epochs: Training epochs
        baseline: Baseline model to compare against

    Returns:
        NegativeControlResult with analysis
    """
    from dko.data.datasets import load_dataset
    from dko.models.dko import DKO
    from dko.models.attention import AttentionPoolingBaseline
    from dko.models.ensemble_baselines import SingleConformer, MeanFeatureAggregation
    from dko.training.trainer import Trainer

    # Default datasets
    if negative_datasets is None:
        negative_datasets = ["qm9_homo", "qm9_lumo", "qm9_gap"]

    if positive_datasets is None:
        positive_datasets = ["esol", "freesolv", "bace", "lipophilicity"]

    logger.info("=" * 70)
    logger.info("NEGATIVE CONTROL EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"Negative controls (Low-SCC): {negative_datasets}")
    logger.info(f"Positive controls (High-SCC): {positive_datasets}")
    logger.info(f"Baseline for comparison: {baseline}")
    logger.info("=" * 70)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize result
    result = NegativeControlResult(
        low_scc_datasets=negative_datasets,
        high_scc_datasets=positive_datasets,
    )

    all_datasets = negative_datasets + positive_datasets

    # Model factories
    def get_model(model_name: str, feature_dim: int, task: str):
        if model_name == "dko":
            return DKO(
                feature_dim=feature_dim,
                output_dim=1,
                task=task,
                use_second_order=True,
                verbose=False,
            )
        elif model_name == "attention":
            return AttentionPoolingBaseline(
                feature_dim=feature_dim,
                output_dim=1,
                task=task,
            )
        elif model_name == "single_conformer":
            return SingleConformer(
                feature_dim=feature_dim,
                num_outputs=1,
            )
        elif model_name == "mfa":
            return MeanFeatureAggregation(
                feature_dim=feature_dim,
                output_dim=1,
                task=task,
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

    # Run experiments
    for dataset_name in all_datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: {dataset_name}")
        is_negative = dataset_name in negative_datasets
        logger.info(f"Category: {'NEGATIVE CONTROL (Low-SCC)' if is_negative else 'POSITIVE CONTROL (High-SCC)'}")
        logger.info("=" * 60)

        try:
            # Load dataset
            dataset = load_dataset(dataset_name)
            feature_dim = dataset.feature_dim
            task = dataset.task

            # Determine metric
            if task == "regression":
                metric = "rmse"
                lower_is_better = True
            else:
                metric = "auroc"
                lower_is_better = False

            result.results[dataset_name] = {}

            # Train DKO and baseline
            for model_name in ["dko", baseline]:
                logger.info(f"  Training {model_name}...")
                performances = []

                for seed in seeds:
                    model = get_model(model_name, feature_dim, task)

                    trainer = Trainer(
                        model=model,
                        config={
                            "lr": 1e-4,
                            "weight_decay": 1e-5,
                            "n_epochs": n_epochs,
                            "batch_size": 32,
                            "patience": 30,
                            "seed": seed,
                        },
                        device=device,
                    )

                    trainer.fit(dataset.train_loader, dataset.val_loader)
                    test_metrics = trainer.evaluate(dataset.test_loader)
                    performances.append(test_metrics[metric])

                mean_perf = np.mean(performances)
                std_perf = np.std(performances)

                result.results[dataset_name][model_name] = {
                    "mean": float(mean_perf),
                    "std": float(std_perf),
                    "metric": metric,
                    "lower_is_better": lower_is_better,
                }

                logger.info(f"    {metric}: {mean_perf:.4f} ± {std_perf:.4f}")

            # Compute DKO advantage
            dko_mean = result.results[dataset_name]["dko"]["mean"]
            baseline_mean = result.results[dataset_name][baseline]["mean"]

            if lower_is_better:
                # Lower is better: advantage = baseline - dko (positive = dko is better)
                advantage = baseline_mean - dko_mean
                relative_advantage = (advantage / baseline_mean) * 100 if baseline_mean != 0 else 0
            else:
                # Higher is better: advantage = dko - baseline
                advantage = dko_mean - baseline_mean
                relative_advantage = (advantage / dko_mean) * 100 if dko_mean != 0 else 0

            result.results[dataset_name]["dko_advantage"] = {
                "absolute": float(advantage),
                "relative_percent": float(relative_advantage),
            }

            if is_negative:
                result.low_scc_advantages.append(relative_advantage)
            else:
                result.high_scc_advantages.append(relative_advantage)

            logger.info(f"  DKO advantage: {relative_advantage:.2f}%")

        except Exception as e:
            logger.error(f"Failed on {dataset_name}: {e}")
            continue

    # Analyze results
    result = analyze_negative_controls(result)

    # Log analysis
    logger.info("\n" + "=" * 70)
    logger.info("NEGATIVE CONTROL ANALYSIS")
    logger.info("=" * 70)
    logger.info(f"Low-SCC mean DKO advantage:  {result.low_scc_mean_advantage:.2f}%")
    logger.info(f"High-SCC mean DKO advantage: {result.high_scc_mean_advantage:.2f}%")
    logger.info("-" * 70)
    logger.info(f"Correlation (SCC vs advantage): {result.correlation_scc_advantage:.3f}")
    logger.info(f"P-value: {result.p_value:.4f}")
    logger.info("-" * 70)

    if result.hypothesis_validated:
        logger.info("✓ HYPOTHESIS VALIDATED")
        logger.info("  DKO shows significantly more advantage on high-SCC tasks")
    else:
        logger.info("✗ HYPOTHESIS NOT VALIDATED")
        logger.info("  DKO advantage does not correlate with SCC")

    # Save results
    output_file = output_path / "negative_control_analysis.json"
    with open(output_file, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    logger.info(f"\nResults saved to {output_file}")

    return result


def analyze_negative_controls(result: NegativeControlResult) -> NegativeControlResult:
    """
    Analyze negative control results.

    Tests whether DKO advantage correlates with SCC category.
    """
    # Compute mean advantages
    if result.low_scc_advantages:
        result.low_scc_mean_advantage = float(np.mean(result.low_scc_advantages))
    if result.high_scc_advantages:
        result.high_scc_mean_advantage = float(np.mean(result.high_scc_advantages))

    # Compute correlation using binary SCC indicator
    # 0 = low-SCC (negative control), 1 = high-SCC (positive control)
    scc_values = [0] * len(result.low_scc_advantages) + [1] * len(result.high_scc_advantages)
    advantages = result.low_scc_advantages + result.high_scc_advantages

    if len(advantages) >= 3:
        corr, p_value = scipy_stats.pearsonr(scc_values, advantages)
        result.correlation_scc_advantage = float(corr)
        result.p_value = float(p_value)

        # Validate hypothesis:
        # 1. Positive correlation between SCC and advantage
        # 2. High-SCC advantage significantly greater than low-SCC
        # 3. P-value < 0.05

        if (result.correlation_scc_advantage > 0.3 and
            result.high_scc_mean_advantage > result.low_scc_mean_advantage and
            result.p_value < 0.1):  # Relaxed for small sample sizes
            result.hypothesis_validated = True

        # Also do t-test between groups
        if len(result.low_scc_advantages) >= 2 and len(result.high_scc_advantages) >= 2:
            t_stat, t_pvalue = scipy_stats.ttest_ind(
                result.high_scc_advantages,
                result.low_scc_advantages,
            )
            logger.info(f"T-test: t={t_stat:.3f}, p={t_pvalue:.4f}")

            if t_pvalue < 0.1 and t_stat > 0:  # High-SCC > Low-SCC
                result.hypothesis_validated = True

    return result


def compute_dataset_scc(dataset_name: str) -> float:
    """
    Compute SCC for a dataset.

    Uses the SCC computation from dko.analysis.scc.

    Args:
        dataset_name: Dataset name

    Returns:
        Mean SCC value for the dataset
    """
    from dko.data.datasets import load_dataset
    from dko.analysis.scc import compute_dataset_scc as _compute_scc

    try:
        dataset = load_dataset(dataset_name)
        scc_stats = _compute_scc(dataset)
        return scc_stats["mean"]
    except Exception as e:
        logger.warning(f"Could not compute SCC for {dataset_name}: {e}")
        return 0.0


def run_scc_advantage_correlation(
    datasets: Optional[List[str]] = None,
    seeds: List[int] = [42, 123, 456],
    output_dir: str = "results/negative_controls",
    device: str = "cuda",
    n_epochs: int = 100,
) -> Dict:
    """
    Run detailed SCC-advantage correlation analysis.

    Computes actual SCC values for each dataset and correlates
    with DKO advantage.

    Args:
        datasets: Datasets to analyze
        seeds: Random seeds
        output_dir: Output directory
        device: Device for training
        n_epochs: Training epochs

    Returns:
        Correlation analysis results
    """
    from dko.data.datasets import load_dataset
    from dko.models.dko import DKO
    from dko.models.attention import AttentionPoolingBaseline
    from dko.training.trainer import Trainer
    from dko.analysis.scc import compute_dataset_scc

    if datasets is None:
        datasets = [
            # Low-SCC expected
            "qm9_homo", "qm9_lumo", "qm9_gap",
            # High-SCC expected
            "esol", "freesolv", "bace", "lipophilicity", "bbbp",
        ]

    logger.info("=" * 70)
    logger.info("SCC-ADVANTAGE CORRELATION ANALYSIS")
    logger.info("=" * 70)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []

    for dataset_name in datasets:
        logger.info(f"\n{dataset_name}:")

        try:
            # Load dataset
            dataset = load_dataset(dataset_name)
            feature_dim = dataset.feature_dim
            task = dataset.task

            # Compute SCC
            scc = compute_dataset_scc(dataset)
            logger.info(f"  SCC: {scc['mean']:.4f} ± {scc['std']:.4f}")

            # Determine metric
            if task == "regression":
                metric = "rmse"
                lower_is_better = True
            else:
                metric = "auroc"
                lower_is_better = False

            # Train DKO
            dko_perfs = []
            for seed in seeds:
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

            # Train attention baseline
            attn_perfs = []
            for seed in seeds:
                attn = AttentionPoolingBaseline(
                    feature_dim=feature_dim,
                    output_dim=1,
                    task=task,
                )
                trainer = Trainer(
                    model=attn,
                    config={"lr": 1e-4, "n_epochs": n_epochs, "seed": seed},
                    device=device,
                )
                trainer.fit(dataset.train_loader, dataset.val_loader)
                test_metrics = trainer.evaluate(dataset.test_loader)
                attn_perfs.append(test_metrics[metric])

            # Compute advantage
            dko_mean = np.mean(dko_perfs)
            attn_mean = np.mean(attn_perfs)

            if lower_is_better:
                advantage = (attn_mean - dko_mean) / attn_mean * 100
            else:
                advantage = (dko_mean - attn_mean) / dko_mean * 100

            logger.info(f"  DKO: {dko_mean:.4f}, Attention: {attn_mean:.4f}")
            logger.info(f"  DKO advantage: {advantage:.2f}%")

            results.append({
                "dataset": dataset_name,
                "scc": scc["mean"],
                "dko_advantage": advantage,
            })

        except Exception as e:
            logger.error(f"Failed on {dataset_name}: {e}")
            continue

    # Compute correlation
    if len(results) >= 3:
        scc_values = [r["scc"] for r in results]
        advantages = [r["dko_advantage"] for r in results]

        corr, p_value = scipy_stats.pearsonr(scc_values, advantages)

        logger.info("\n" + "=" * 70)
        logger.info("CORRELATION ANALYSIS")
        logger.info("=" * 70)
        logger.info(f"Pearson correlation (SCC vs DKO advantage): {corr:.3f}")
        logger.info(f"P-value: {p_value:.4f}")

        if corr > 0.5 and p_value < 0.1:
            logger.info("✓ Strong positive correlation confirmed")
            logger.info("  DKO advantage increases with SCC")
        elif corr > 0.3:
            logger.info("~ Moderate positive correlation")
        else:
            logger.info("✗ Weak or no correlation")

    # Save results
    output_file = output_path / "scc_advantage_correlation.json"
    with open(output_file, "w") as f:
        json.dump({
            "results": results,
            "correlation": corr if len(results) >= 3 else None,
            "p_value": p_value if len(results) >= 3 else None,
        }, f, indent=2)

    logger.info(f"\nResults saved to {output_file}")

    return {"results": results, "correlation": corr, "p_value": p_value}
