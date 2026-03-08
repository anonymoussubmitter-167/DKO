"""
80/20 Decomposition Study for DKO.

Analyzes the contribution of different components to ensemble method improvements:
- Mean contribution: MFA - SingleConformer (better mean estimation)
- Kernel contribution: DKO-FirstOrder - MFA (kernel structure value)
- Covariance contribution: DKO-Full - DKO-FirstOrder (second-order features)

Hypothesis: ~80% of ensemble improvements come from better mean estimation,
not from capturing molecular flexibility (covariance).

This decomposition is only possible because DKO explicitly separates first-order
(mean) and second-order (covariance) features - attention-based methods cannot
be decomposed this way.

## Deconfounding Sampling vs Aggregation Effects

The standard decomposition (MFA - SingleConformer) conflates two effects:
1. Using 50 conformers vs 1 conformer (sampling effect)
2. Averaging vs single selection (aggregation effect)

To separate these, use SingleConformer selection variants:

    SingleConformer(lowest_energy) - standard baseline
    SingleConformer(random)        - controls for sampling bias
    SingleConformer(centroid)      - controls for outlier effects

Analysis framework:
    MFA - Single(lowest)   = sampling + aggregation + energy bias
    MFA - Single(random)   = aggregation (no energy bias)
    MFA - Single(centroid) = outlier reduction + aggregation

Use single_conformer_random and single_conformer_centroid from MODEL_REGISTRY
to run the deconfounded analysis.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import torch
from pathlib import Path
import json

from dko.utils.logging_utils import get_logger

logger = get_logger("decomposition")


@dataclass
class DecompositionResult:
    """Results from 80/20 decomposition analysis."""

    # Raw performance values
    single_conformer: float
    mfa: float
    dko_first_order: float
    dko_full: float

    # Decomposed contributions
    mean_contribution: float  # MFA - SingleConf
    kernel_contribution: float  # DKO-First - MFA
    covariance_contribution: float  # DKO-Full - DKO-First
    total_improvement: float  # DKO-Full - SingleConf

    # Percentages
    mean_percentage: float
    kernel_percentage: float
    covariance_percentage: float

    # Metadata
    dataset: str
    metric: str
    lower_is_better: bool
    n_seeds: int

    def to_dict(self) -> Dict:
        return {
            "performance": {
                "single_conformer": self.single_conformer,
                "mfa": self.mfa,
                "dko_first_order": self.dko_first_order,
                "dko_full": self.dko_full,
            },
            "contributions": {
                "mean": self.mean_contribution,
                "kernel": self.kernel_contribution,
                "covariance": self.covariance_contribution,
                "total": self.total_improvement,
            },
            "percentages": {
                "mean": self.mean_percentage,
                "kernel": self.kernel_percentage,
                "covariance": self.covariance_percentage,
            },
            "metadata": {
                "dataset": self.dataset,
                "metric": self.metric,
                "lower_is_better": self.lower_is_better,
                "n_seeds": self.n_seeds,
            }
        }


def compute_decomposition(
    single_conformer_values: List[float],
    mfa_values: List[float],
    dko_first_values: List[float],
    dko_full_values: List[float],
    dataset: str = "unknown",
    metric: str = "rmse",
    lower_is_better: bool = True,
) -> DecompositionResult:
    """
    Compute 80/20 decomposition from experimental results.

    The decomposition breaks down the total improvement into three components:
    1. Mean contribution: improvement from better mean estimation (MFA vs Single)
    2. Kernel contribution: improvement from kernel structure (DKO-First vs MFA)
    3. Covariance contribution: improvement from second-order features (DKO-Full vs DKO-First)

    Args:
        single_conformer_values: Metric values for single conformer baseline (per seed)
        mfa_values: Metric values for Mean Feature Aggregation (per seed)
        dko_first_values: Metric values for DKO first-order only (per seed)
        dko_full_values: Metric values for full DKO (per seed)
        dataset: Dataset name
        metric: Metric name
        lower_is_better: Whether lower metric values indicate better performance

    Returns:
        DecompositionResult with all computed values
    """
    # Compute means
    single_mean = np.mean(single_conformer_values)
    mfa_mean = np.mean(mfa_values)
    dko_first_mean = np.mean(dko_first_values)
    dko_full_mean = np.mean(dko_full_values)

    # Compute contributions (sign depends on metric direction)
    if lower_is_better:
        # Lower is better: improvement = baseline - model (positive = good)
        mean_contrib = single_mean - mfa_mean
        kernel_contrib = mfa_mean - dko_first_mean
        cov_contrib = dko_first_mean - dko_full_mean
        total = single_mean - dko_full_mean
    else:
        # Higher is better: improvement = model - baseline (positive = good)
        mean_contrib = mfa_mean - single_mean
        kernel_contrib = dko_first_mean - mfa_mean
        cov_contrib = dko_full_mean - dko_first_mean
        total = dko_full_mean - single_mean

    # Compute percentages (handle zero total case)
    if abs(total) > 1e-8:
        mean_pct = (mean_contrib / total) * 100
        kernel_pct = (kernel_contrib / total) * 100
        cov_pct = (cov_contrib / total) * 100
    else:
        # No improvement - assign 0%
        mean_pct = 0.0
        kernel_pct = 0.0
        cov_pct = 0.0

    return DecompositionResult(
        single_conformer=single_mean,
        mfa=mfa_mean,
        dko_first_order=dko_first_mean,
        dko_full=dko_full_mean,
        mean_contribution=mean_contrib,
        kernel_contribution=kernel_contrib,
        covariance_contribution=cov_contrib,
        total_improvement=total,
        mean_percentage=mean_pct,
        kernel_percentage=kernel_pct,
        covariance_percentage=cov_pct,
        dataset=dataset,
        metric=metric,
        lower_is_better=lower_is_better,
        n_seeds=len(single_conformer_values),
    )


def run_decomposition_study(
    dataset_name: str,
    config_path: Optional[str] = None,
    output_dir: str = "results/decomposition",
    device: str = "cuda",
    seeds: List[int] = [42, 123, 456],
    n_epochs: int = 100,
) -> Dict:
    """
    Run full decomposition study for a dataset.

    Trains four models (SingleConformer, MFA, DKO-FirstOrder, DKO-Full)
    and computes the 80/20 decomposition.

    Args:
        dataset_name: Dataset to use
        config_path: Path to config (optional)
        output_dir: Output directory
        device: Device for training
        seeds: Random seeds for multiple runs
        n_epochs: Number of training epochs

    Returns:
        Dictionary with decomposition results
    """
    from dko.data.datasets import load_dataset
    from dko.models.dko import DKO, DKOFirstOrder
    from dko.models.ensemble_baselines import MeanFeatureAggregation, SingleConformer
    from dko.training.trainer import Trainer
    from dko.utils.config import load_config

    logger.info(f"Running decomposition study on {dataset_name}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load config if provided
    if config_path:
        config = load_config(config_path)
    else:
        config = {}

    # Load dataset
    dataset = load_dataset(dataset_name)
    feature_dim = dataset.feature_dim
    task = dataset.task

    # Determine metric based on task
    if task == "regression":
        metric = "rmse"
        lower_is_better = True
    else:
        metric = "auroc"
        lower_is_better = False

    # Results storage
    results = {
        "single_conformer": [],
        "mfa": [],
        "dko_first_order": [],
        "dko_full": [],
    }

    # Model factories
    model_configs = {
        "single_conformer": lambda: SingleConformer(
            feature_dim=feature_dim,
            num_outputs=1,
        ),
        "mfa": lambda: MeanFeatureAggregation(
            feature_dim=feature_dim,
            output_dim=1,
            task=task,
        ),
        "dko_first_order": lambda: DKOFirstOrder(
            feature_dim=feature_dim,
            output_dim=1,
            task=task,
            verbose=False,
        ),
        "dko_full": lambda: DKO(
            feature_dim=feature_dim,
            output_dim=1,
            task=task,
            use_second_order=True,
            verbose=False,
        ),
    }

    # Train each model with each seed
    for seed in seeds:
        logger.info(f"Seed {seed}")

        for model_name, model_factory in model_configs.items():
            logger.info(f"  Training {model_name}...")

            # Create model
            model = model_factory()

            # Create trainer
            trainer = Trainer(
                model=model,
                config={
                    "lr": config.get("lr", 1e-4),
                    "weight_decay": config.get("weight_decay", 1e-5),
                    "n_epochs": n_epochs,
                    "batch_size": config.get("batch_size", 32),
                    "patience": config.get("patience", 30),
                    "seed": seed,
                },
                device=device,
            )

            # Train
            trainer.fit(dataset.train_loader, dataset.val_loader)

            # Evaluate
            test_metrics = trainer.evaluate(dataset.test_loader)

            # Store result
            results[model_name].append(test_metrics[metric])

            logger.info(f"    {metric}: {test_metrics[metric]:.4f}")

    # Compute decomposition
    decomposition = compute_decomposition(
        single_conformer_values=results["single_conformer"],
        mfa_values=results["mfa"],
        dko_first_values=results["dko_first_order"],
        dko_full_values=results["dko_full"],
        dataset=dataset_name,
        metric=metric,
        lower_is_better=lower_is_better,
    )

    # Log summary
    logger.info("\n" + "=" * 60)
    logger.info("80/20 Decomposition Results")
    logger.info("=" * 60)
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Metric: {metric} ({'lower' if lower_is_better else 'higher'} is better)")
    logger.info(f"Seeds: {seeds}")
    logger.info("-" * 60)
    logger.info(f"Single Conformer: {decomposition.single_conformer:.4f}")
    logger.info(f"MFA:              {decomposition.mfa:.4f}")
    logger.info(f"DKO First-Order:  {decomposition.dko_first_order:.4f}")
    logger.info(f"DKO Full:         {decomposition.dko_full:.4f}")
    logger.info("-" * 60)
    logger.info(f"Total Improvement: {decomposition.total_improvement:.4f}")
    logger.info(f"Mean Contribution:     {decomposition.mean_contribution:.4f} ({decomposition.mean_percentage:.1f}%)")
    logger.info(f"Kernel Contribution:   {decomposition.kernel_contribution:.4f} ({decomposition.kernel_percentage:.1f}%)")
    logger.info(f"Covariance Contribution: {decomposition.covariance_contribution:.4f} ({decomposition.covariance_percentage:.1f}%)")
    logger.info("=" * 60)

    # Validate 80/20 hypothesis
    first_order_pct = decomposition.mean_percentage + decomposition.kernel_percentage
    logger.info(f"\n80/20 Hypothesis Check:")
    logger.info(f"  First-order (mean + kernel): {first_order_pct:.1f}%")
    logger.info(f"  Second-order (covariance):   {decomposition.covariance_percentage:.1f}%")
    if first_order_pct >= 70:
        logger.info("  ✓ Hypothesis SUPPORTED (first-order ≥ 70%)")
    else:
        logger.info("  ✗ Hypothesis NOT supported (first-order < 70%)")

    # Save results
    output_file = output_path / f"{dataset_name}_decomposition.json"
    with open(output_file, "w") as f:
        json.dump({
            "decomposition": decomposition.to_dict(),
            "raw_results": results,
        }, f, indent=2)

    logger.info(f"\nResults saved to {output_file}")

    return {
        "decomposition": decomposition,
        "raw_results": results,
    }


def aggregate_decomposition_results(
    results: List[DecompositionResult],
) -> Dict:
    """
    Aggregate decomposition results across multiple datasets.

    Args:
        results: List of DecompositionResult objects

    Returns:
        Aggregated statistics
    """
    mean_pcts = [r.mean_percentage for r in results]
    kernel_pcts = [r.kernel_percentage for r in results]
    cov_pcts = [r.covariance_percentage for r in results]

    first_order_pcts = [m + k for m, k in zip(mean_pcts, kernel_pcts)]

    return {
        "mean_contribution": {
            "mean": np.mean(mean_pcts),
            "std": np.std(mean_pcts),
            "min": np.min(mean_pcts),
            "max": np.max(mean_pcts),
        },
        "kernel_contribution": {
            "mean": np.mean(kernel_pcts),
            "std": np.std(kernel_pcts),
            "min": np.min(kernel_pcts),
            "max": np.max(kernel_pcts),
        },
        "covariance_contribution": {
            "mean": np.mean(cov_pcts),
            "std": np.std(cov_pcts),
            "min": np.min(cov_pcts),
            "max": np.max(cov_pcts),
        },
        "first_order_total": {
            "mean": np.mean(first_order_pcts),
            "std": np.std(first_order_pcts),
            "min": np.min(first_order_pcts),
            "max": np.max(first_order_pcts),
        },
        "hypothesis_support_rate": np.mean([p >= 70 for p in first_order_pcts]),
        "n_datasets": len(results),
        "datasets": [r.dataset for r in results],
    }


def run_full_decomposition_study(
    datasets: Optional[List[str]] = None,
    output_dir: str = "results/decomposition",
    device: str = "cuda",
    seeds: List[int] = [42, 123, 456],
) -> Dict:
    """
    Run decomposition study across all datasets.

    Args:
        datasets: List of dataset names (None for all standard datasets)
        output_dir: Output directory
        device: Device for training
        seeds: Random seeds

    Returns:
        Aggregated results
    """
    if datasets is None:
        # Standard benchmark datasets
        datasets = [
            "esol", "freesolv", "lipophilicity",  # Solubility
            "bace", "bbbp",  # Binding/ADMET
            "qm9_homo", "qm9_lumo", "qm9_gap",  # Electronic (negative controls)
        ]

    logger.info(f"Running decomposition study on {len(datasets)} datasets")

    results = []
    for dataset in datasets:
        try:
            result = run_decomposition_study(
                dataset_name=dataset,
                output_dir=output_dir,
                device=device,
                seeds=seeds,
            )
            results.append(result["decomposition"])
        except Exception as e:
            logger.error(f"Failed on {dataset}: {e}")
            continue

    # Aggregate
    aggregated = aggregate_decomposition_results(results)

    # Save aggregated results
    output_path = Path(output_dir)
    with open(output_path / "aggregated_decomposition.json", "w") as f:
        json.dump(aggregated, f, indent=2)

    # Log summary
    logger.info("\n" + "=" * 60)
    logger.info("AGGREGATED 80/20 DECOMPOSITION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Datasets analyzed: {aggregated['n_datasets']}")
    logger.info("-" * 60)
    logger.info(f"Mean Contribution:       {aggregated['mean_contribution']['mean']:.1f}% ± {aggregated['mean_contribution']['std']:.1f}%")
    logger.info(f"Kernel Contribution:     {aggregated['kernel_contribution']['mean']:.1f}% ± {aggregated['kernel_contribution']['std']:.1f}%")
    logger.info(f"Covariance Contribution: {aggregated['covariance_contribution']['mean']:.1f}% ± {aggregated['covariance_contribution']['std']:.1f}%")
    logger.info("-" * 60)
    logger.info(f"First-order Total:       {aggregated['first_order_total']['mean']:.1f}% ± {aggregated['first_order_total']['std']:.1f}%")
    logger.info(f"Hypothesis Support Rate: {aggregated['hypothesis_support_rate']*100:.0f}%")
    logger.info("=" * 60)

    return {
        "individual_results": results,
        "aggregated": aggregated,
    }
