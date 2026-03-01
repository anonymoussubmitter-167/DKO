"""
Sample Efficiency Experiments for DKO.

This module implements two key experiments from the research plan:

1. Data Fraction Experiment:
   - Train on varying fractions of training data (10%, 25%, 50%, 75%, 100%)
   - Compare DKO vs baselines at each fraction
   - Test hypothesis: DKO shows better sample efficiency for high-SCC tasks

2. Conformer Count Experiment:
   - Vary number of conformers per molecule: [5, 10, 20, 30, 50]
   - Measure performance improvement as conformer count increases
   - Test hypothesis: DKO extracts more information per conformer

Key insight: Sample efficiency differences should be most pronounced on
high-SCC (high structural conformational complexity) tasks where molecular
flexibility impacts properties.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import torch
from pathlib import Path
import json
import copy

from dko.utils.logging_utils import get_logger

logger = get_logger("sample_efficiency")


@dataclass
class SampleEfficiencyResult:
    """Results from sample efficiency experiment."""

    dataset: str
    experiment_type: str  # 'data_fraction' or 'conformer_count'
    x_values: List[float]  # fractions or conformer counts
    x_label: str  # 'Data Fraction' or 'Conformer Count'
    model_results: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    # model_results[model_name][metric] = [values for each x]

    def to_dict(self) -> Dict:
        return {
            "dataset": self.dataset,
            "experiment_type": self.experiment_type,
            "x_values": self.x_values,
            "x_label": self.x_label,
            "model_results": self.model_results,
        }


def run_sample_efficiency_experiment(
    dataset_name: str,
    fractions: List[float] = [0.1, 0.25, 0.5, 0.75, 1.0],
    models: Optional[List[str]] = None,
    seeds: List[int] = [42, 123, 456],
    config_path: Optional[str] = None,
    output_dir: str = "results/sample_efficiency",
    device: str = "cuda",
    n_epochs: int = 100,
) -> SampleEfficiencyResult:
    """
    Run data fraction sample efficiency experiment.

    Trains models on varying fractions of the training data
    to evaluate sample efficiency.

    Args:
        dataset_name: Dataset to use
        fractions: Training data fractions to test
        models: Models to compare (default: all standard models)
        seeds: Random seeds for multiple runs
        config_path: Path to config (optional)
        output_dir: Output directory
        device: Device for training
        n_epochs: Number of training epochs

    Returns:
        SampleEfficiencyResult with all metrics
    """
    from dko.data.datasets import load_dataset
    from dko.models.dko import DKO
    from dko.models.ensemble_baselines import (
        SingleConformer,
        MeanFeatureAggregation,
        MeanEnsemble,
        BoltzmannEnsemble,
        MultiInstanceLearning,
    )
    from dko.models.attention import AttentionPoolingBaseline
    from dko.models.deepsets import DeepSetsBaseline
    from dko.training.trainer import Trainer
    from dko.utils.config import load_config

    logger.info(f"Running data fraction experiment on {dataset_name}")
    logger.info(f"Fractions: {fractions}")
    logger.info(f"Seeds: {seeds}")

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

    # Default models
    if models is None:
        models = ["dko", "attention", "deepsets", "single_conformer", "mfa", "mil"]

    # Model factories
    def get_model(model_name: str):
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
        elif model_name == "deepsets":
            return DeepSetsBaseline(
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
        elif model_name == "mil":
            return MultiInstanceLearning(
                feature_dim=feature_dim,
                output_dim=1,
                task=task,
                pooling="attention",
            )
        elif model_name == "mean_ensemble":
            return MeanEnsemble(
                feature_dim=feature_dim,
                num_outputs=1,
            )
        elif model_name == "boltzmann_ensemble":
            return BoltzmannEnsemble(
                feature_dim=feature_dim,
                num_outputs=1,
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

    # Results storage
    results = SampleEfficiencyResult(
        dataset=dataset_name,
        experiment_type="data_fraction",
        x_values=fractions,
        x_label="Data Fraction",
    )

    for model_name in models:
        results.model_results[model_name] = {
            f"{metric}_mean": [],
            f"{metric}_std": [],
        }

    # Run experiment for each fraction
    for fraction in fractions:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training with {fraction*100:.0f}% of data")
        logger.info("=" * 60)

        # Subsample training data
        train_loader_subset = subsample_dataloader(dataset.train_loader, fraction)

        for model_name in models:
            logger.info(f"  {model_name}...")
            performances = []

            for seed in seeds:
                # Create fresh model
                model = get_model(model_name)

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

                # Train on subset
                trainer.fit(train_loader_subset, dataset.val_loader)

                # Evaluate on full test set
                test_metrics = trainer.evaluate(dataset.test_loader)
                performances.append(test_metrics[metric])

            mean_perf = np.mean(performances)
            std_perf = np.std(performances)

            results.model_results[model_name][f"{metric}_mean"].append(mean_perf)
            results.model_results[model_name][f"{metric}_std"].append(std_perf)

            logger.info(f"    {metric}: {mean_perf:.4f} ± {std_perf:.4f}")

    # Log summary
    log_sample_efficiency_summary(results, metric, lower_is_better)

    # Save results
    output_file = output_path / f"{dataset_name}_data_fraction.json"
    with open(output_file, "w") as f:
        json.dump(results.to_dict(), f, indent=2)

    logger.info(f"\nResults saved to {output_file}")

    return results


def run_conformer_count_experiment(
    dataset_name: str,
    conformer_counts: List[int] = [5, 10, 20, 30, 50],
    models: Optional[List[str]] = None,
    seeds: List[int] = [42, 123, 456],
    config_path: Optional[str] = None,
    output_dir: str = "results/sample_efficiency",
    device: str = "cuda",
    n_epochs: int = 100,
) -> SampleEfficiencyResult:
    """
    Run conformer count experiment.

    Tests how performance scales with number of conformers per molecule.
    This directly tests whether ensemble methods can extract more information
    from additional conformational samples.

    Key hypothesis: DKO should show steeper improvement curves on high-SCC
    tasks because it captures distributional (covariance) information.

    Args:
        dataset_name: Dataset to use
        conformer_counts: Number of conformers to test
        models: Models to compare
        seeds: Random seeds
        config_path: Path to config
        output_dir: Output directory
        device: Device for training
        n_epochs: Training epochs

    Returns:
        SampleEfficiencyResult with conformer scaling results
    """
    from dko.data.datasets import load_dataset
    from dko.models.dko import DKO
    from dko.models.ensemble_baselines import (
        SingleConformer,
        MeanFeatureAggregation,
        MeanEnsemble,
        MultiInstanceLearning,
    )
    from dko.models.attention import AttentionPoolingBaseline
    from dko.models.deepsets import DeepSetsBaseline
    from dko.training.trainer import Trainer
    from dko.utils.config import load_config

    logger.info(f"Running conformer count experiment on {dataset_name}")
    logger.info(f"Conformer counts: {conformer_counts}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load config if provided
    if config_path:
        config = load_config(config_path)
    else:
        config = {}

    # Load dataset info (we'll reload with different conformer limits)
    base_dataset = load_dataset(dataset_name)
    feature_dim = base_dataset.feature_dim
    task = base_dataset.task

    # Determine metric
    if task == "regression":
        metric = "rmse"
        lower_is_better = True
    else:
        metric = "auroc"
        lower_is_better = False

    # Default models (single_conformer is special - doesn't benefit from more conformers)
    if models is None:
        models = ["dko", "attention", "mfa", "mil", "single_conformer"]

    # Model factories
    def get_model(model_name: str):
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
        elif model_name == "mfa":
            return MeanFeatureAggregation(
                feature_dim=feature_dim,
                output_dim=1,
                task=task,
            )
        elif model_name == "mil":
            return MultiInstanceLearning(
                feature_dim=feature_dim,
                output_dim=1,
                task=task,
                pooling="attention",
            )
        elif model_name == "single_conformer":
            return SingleConformer(
                feature_dim=feature_dim,
                num_outputs=1,
            )
        elif model_name == "deepsets":
            return DeepSetsBaseline(
                feature_dim=feature_dim,
                output_dim=1,
                task=task,
            )
        elif model_name == "mean_ensemble":
            return MeanEnsemble(
                feature_dim=feature_dim,
                num_outputs=1,
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

    # Results storage
    results = SampleEfficiencyResult(
        dataset=dataset_name,
        experiment_type="conformer_count",
        x_values=[float(c) for c in conformer_counts],
        x_label="Conformer Count",
    )

    for model_name in models:
        results.model_results[model_name] = {
            f"{metric}_mean": [],
            f"{metric}_std": [],
        }

    # Run experiment for each conformer count
    for n_conf in conformer_counts:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training with {n_conf} conformers per molecule")
        logger.info("=" * 60)

        # Load dataset with conformer limit
        dataset = load_dataset(dataset_name, n_conformers=n_conf)

        for model_name in models:
            # Single conformer doesn't benefit from more conformers
            # but we still run it for comparison
            logger.info(f"  {model_name}...")
            performances = []

            for seed in seeds:
                # Create fresh model
                model = get_model(model_name)

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
                performances.append(test_metrics[metric])

            mean_perf = np.mean(performances)
            std_perf = np.std(performances)

            results.model_results[model_name][f"{metric}_mean"].append(mean_perf)
            results.model_results[model_name][f"{metric}_std"].append(std_perf)

            logger.info(f"    {metric}: {mean_perf:.4f} ± {std_perf:.4f}")

    # Log summary
    log_conformer_scaling_summary(results, metric, lower_is_better)

    # Save results
    output_file = output_path / f"{dataset_name}_conformer_count.json"
    with open(output_file, "w") as f:
        json.dump(results.to_dict(), f, indent=2)

    logger.info(f"\nResults saved to {output_file}")

    return results


def subsample_dataloader(dataloader, fraction: float):
    """
    Create a subsampled version of a dataloader.

    Args:
        dataloader: Original dataloader
        fraction: Fraction of data to keep (0-1)

    Returns:
        New dataloader with subsampled data
    """
    from torch.utils.data import DataLoader, Subset

    dataset = dataloader.dataset
    n_samples = len(dataset)
    n_keep = max(1, int(n_samples * fraction))

    # Random subset
    indices = np.random.permutation(n_samples)[:n_keep]
    subset = Subset(dataset, indices)

    # Create new dataloader with same settings
    return DataLoader(
        subset,
        batch_size=dataloader.batch_size,
        shuffle=True,
        num_workers=getattr(dataloader, "num_workers", 0),
        collate_fn=getattr(dataloader, "collate_fn", None),
    )


def log_sample_efficiency_summary(
    results: SampleEfficiencyResult,
    metric: str,
    lower_is_better: bool,
) -> None:
    """Log summary of data fraction experiment."""
    logger.info("\n" + "=" * 70)
    logger.info("SAMPLE EFFICIENCY SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Dataset: {results.dataset}")
    logger.info(f"Metric: {metric} ({'lower' if lower_is_better else 'higher'} is better)")
    logger.info("-" * 70)

    # Header
    header = f"{'Fraction':<12}"
    for model in results.model_results.keys():
        header += f"{model:<18}"
    logger.info(header)
    logger.info("-" * 70)

    # Data rows
    for i, fraction in enumerate(results.x_values):
        row = f"{fraction:<12.2f}"
        for model, metrics in results.model_results.items():
            mean = metrics[f"{metric}_mean"][i]
            std = metrics[f"{metric}_std"][i]
            row += f"{mean:.4f}±{std:.4f}  "
        logger.info(row)

    logger.info("=" * 70)

    # Compute and report sample efficiency (improvement rate)
    logger.info("\nSample Efficiency Analysis:")
    for model, metrics in results.model_results.items():
        means = metrics[f"{metric}_mean"]
        if len(means) >= 2:
            # Improvement from min to max fraction
            if lower_is_better:
                improvement = means[0] - means[-1]  # How much it decreased
                relative = improvement / means[0] * 100 if means[0] != 0 else 0
            else:
                improvement = means[-1] - means[0]  # How much it increased
                relative = improvement / means[-1] * 100 if means[-1] != 0 else 0

            logger.info(
                f"  {model}: {relative:.1f}% improvement from "
                f"{results.x_values[0]*100:.0f}% to {results.x_values[-1]*100:.0f}% data"
            )


def log_conformer_scaling_summary(
    results: SampleEfficiencyResult,
    metric: str,
    lower_is_better: bool,
) -> None:
    """Log summary of conformer count experiment."""
    logger.info("\n" + "=" * 70)
    logger.info("CONFORMER SCALING SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Dataset: {results.dataset}")
    logger.info(f"Metric: {metric} ({'lower' if lower_is_better else 'higher'} is better)")
    logger.info("-" * 70)

    # Header
    header = f"{'#Conf':<8}"
    for model in results.model_results.keys():
        header += f"{model:<18}"
    logger.info(header)
    logger.info("-" * 70)

    # Data rows
    for i, n_conf in enumerate(results.x_values):
        row = f"{int(n_conf):<8}"
        for model, metrics in results.model_results.items():
            mean = metrics[f"{metric}_mean"][i]
            std = metrics[f"{metric}_std"][i]
            row += f"{mean:.4f}±{std:.4f}  "
        logger.info(row)

    logger.info("=" * 70)

    # Compute and report scaling factors
    logger.info("\nConformer Scaling Analysis:")
    logger.info("(Improvement from min to max conformers)")

    best_improvement = 0
    best_model = None

    for model, metrics in results.model_results.items():
        means = metrics[f"{metric}_mean"]
        if len(means) >= 2:
            if lower_is_better:
                improvement = means[0] - means[-1]
                relative = improvement / means[0] * 100 if means[0] != 0 else 0
            else:
                improvement = means[-1] - means[0]
                relative = improvement / means[-1] * 100 if means[-1] != 0 else 0

            logger.info(
                f"  {model}: {relative:.1f}% improvement "
                f"({int(results.x_values[0])} -> {int(results.x_values[-1])} conformers)"
            )

            if relative > best_improvement:
                best_improvement = relative
                best_model = model

    if best_model:
        logger.info(f"\nBest scaling: {best_model} ({best_improvement:.1f}% improvement)")
        if best_model == "dko":
            logger.info("✓ DKO shows best conformer utilization")


def run_full_sample_efficiency_study(
    datasets: Optional[List[str]] = None,
    output_dir: str = "results/sample_efficiency",
    device: str = "cuda",
    seeds: List[int] = [42, 123, 456],
) -> Dict:
    """
    Run full sample efficiency study across datasets.

    Runs both data fraction and conformer count experiments
    on multiple datasets.

    Args:
        datasets: Datasets to test (default: standard benchmarks)
        output_dir: Output directory
        device: Device for training
        seeds: Random seeds

    Returns:
        Dictionary with all results
    """
    if datasets is None:
        # Standard benchmark datasets
        datasets = [
            "esol",
            "freesolv",
            "lipophilicity",
            "bace",
            "bbbp",
        ]

    logger.info(f"Running full sample efficiency study on {len(datasets)} datasets")

    all_results = {
        "data_fraction": {},
        "conformer_count": {},
    }

    for dataset in datasets:
        logger.info(f"\n{'#'*70}")
        logger.info(f"# Dataset: {dataset}")
        logger.info("#" * 70)

        try:
            # Data fraction experiment
            logger.info("\n--- Data Fraction Experiment ---")
            df_result = run_sample_efficiency_experiment(
                dataset_name=dataset,
                output_dir=output_dir,
                device=device,
                seeds=seeds,
            )
            all_results["data_fraction"][dataset] = df_result.to_dict()

            # Conformer count experiment
            logger.info("\n--- Conformer Count Experiment ---")
            cc_result = run_conformer_count_experiment(
                dataset_name=dataset,
                output_dir=output_dir,
                device=device,
                seeds=seeds,
            )
            all_results["conformer_count"][dataset] = cc_result.to_dict()

        except Exception as e:
            logger.error(f"Failed on {dataset}: {e}")
            continue

    # Save aggregated results
    output_path = Path(output_dir)
    with open(output_path / "full_study_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nFull study results saved to {output_path / 'full_study_results.json'}")

    return all_results
