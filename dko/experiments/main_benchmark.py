"""
Main benchmark experiment for DKO.

This script runs the main comparative benchmark across all datasets
and models, evaluating DKO against baselines.

Key Insight: The primary value of DKO is enabling DECOMPOSITION ANALYSIS,
not necessarily achieving best performance. DKO's explicit separation of
first-order (mean) and second-order (covariance) features allows us to:

1. Quantify the 80/20 hypothesis: How much of ensemble improvement comes
   from better mean estimation vs capturing conformational flexibility?

2. Validate negative controls: Confirm that second-order features don't
   help on properties that shouldn't depend on conformational variance
   (ESOL, Lipophilicity, QM9 electronic properties).

3. Identify when ensembles matter: Only binding affinity and potentially
   FreeSolv show meaningful second-order dependence.

DKO-FirstOrder should match or beat attention on most tasks.
DKO-Full (with covariance) should only beat DKO-FirstOrder on tasks
where conformational flexibility affects the property.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional
import json
import numpy as np
import torch

from dko.utils.config import Config, create_experiment_config
from dko.utils.logging_utils import ExperimentTracker, get_logger
from dko.data.datasets import create_dataloaders, create_dataloaders_from_precomputed, AVAILABLE_DATASETS
from dko.models import (
    DKO,
    DKOFirstOrder,
    DKOEigenspectrum,
    DKOScalarInvariants,
    DKOLowRank,
    DKOGatedFusion,
    DKOResidual,
    DKOCrossAttention,
    DKOSCCRouter,
    AttentionAggregation,
    AttentionAugmented,
    DeepSets,
    DeepSetsAugmented,
    SingleConformer,
    MeanEnsemble,
    BoltzmannEnsemble,
    MeanFeatureAggregation,
    MultiInstanceLearning,
    SchNet,
    DimeNetPP,
    SphereNet,
    ThreeDInfomax,
    GEM,
)
from dko.training.trainer import train_model
from dko.training.evaluator import Evaluator


logger = get_logger("benchmark")


# Model registry
MODEL_REGISTRY = {
    # DKO variants
    "dko": DKO,
    "dko_first_order": DKOFirstOrder,
    "dko_diagonal": lambda **kwargs: DKO(use_diagonal_sigma=True, **kwargs),
    "dko_separate_nets": lambda **kwargs: DKO(separate_mu_sigma_nets=True, **kwargs),
    # DKO eigendecomposition variants
    "dko_eigenspectrum": DKOEigenspectrum,
    "dko_invariants": DKOScalarInvariants,
    "dko_lowrank": DKOLowRank,
    "dko_gated": DKOGatedFusion,
    "dko_residual": DKOResidual,
    "dko_crossattn": DKOCrossAttention,
    "dko_router": DKOSCCRouter,
    # Attention-based
    "attention": AttentionAggregation,
    "attention_augmented": AttentionAugmented,
    # DeepSets-based
    "deepsets": DeepSets,
    "deepsets_augmented": DeepSetsAugmented,
    # Ensemble baselines
    "single_conformer": SingleConformer,  # Default: lowest_energy
    "single_conformer_random": lambda **kwargs: SingleConformer(selection_method="random", **kwargs),
    "single_conformer_centroid": lambda **kwargs: SingleConformer(selection_method="centroid", **kwargs),
    "mean_ensemble": MeanEnsemble,
    "boltzmann_ensemble": BoltzmannEnsemble,
    "mfa": MeanFeatureAggregation,
    "mil": MultiInstanceLearning,
    # GNN baselines
    "schnet": SchNet,
    "dimenet++": DimeNetPP,
    "dimenetpp": DimeNetPP,  # Alias without special chars
    "spherenet": SphereNet,
    "3d_infomax": ThreeDInfomax,
    "3dinfomax": ThreeDInfomax,  # Alias without underscore
    "gem": GEM,
}


def run_single_experiment(
    dataset_name: str,
    model_name: str,
    config: Config,
    seed: int = 42,
    device: str = "cuda",
    output_dir: str = "results/benchmark",
) -> Dict:
    """
    Run a single experiment (one dataset, one model, one seed).

    Args:
        dataset_name: Name of the dataset
        model_name: Name of the model
        config: Configuration
        seed: Random seed
        device: Device to use

    Returns:
        Dictionary of results
    """
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data from precomputed conformers
    logger.info(f"Loading dataset: {dataset_name}")
    try:
        # Try precomputed first (much faster)
        train_loader, val_loader, test_loader = create_dataloaders_from_precomputed(
            dataset_name,
            batch_size=config.get("training.batch_size", 32),
            num_workers=config.get("project.num_workers", 4),
        )
        logger.info(f"Loaded precomputed conformers for {dataset_name}")
    except FileNotFoundError as e:
        # Fall back to on-the-fly generation
        logger.warning(f"Precomputed conformers not found, generating on-the-fly: {e}")
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset_name,
            batch_size=config.get("training.batch_size", 32),
            num_workers=config.get("project.num_workers", 4),
        )

    # Get feature dimension from data
    sample = next(iter(train_loader))
    feature_dim = sample["features"].shape[-1]

    # Create model
    logger.info(f"Creating model: {model_name}")
    model_class = MODEL_REGISTRY[model_name]
    model_config = config.get("model", {})
    model_config["feature_dim"] = feature_dim

    # Set number of outputs (different models use different param names)
    if dataset_name in ["tox21"]:
        n_outputs = 12  # Multi-task
    else:
        n_outputs = 1

    # DKO-family uses output_dim, others use num_outputs
    model_config["output_dim"] = n_outputs
    model_config["num_outputs"] = n_outputs  # For legacy models

    # DKO-specific config
    is_dko = model_name.startswith("dko")
    is_dko_variant = model_name in [
        "dko_eigenspectrum", "dko_invariants", "dko_lowrank",
        "dko_gated", "dko_residual", "dko_crossattn", "dko_router",
    ]
    if is_dko and not is_dko_variant:
        model_config["kernel_output_dim"] = 64  # Full PSD feature dimension (L-matrix scaling handles stability)

    # Clean up conflicting args before passing to model
    try:
        model = model_class(**model_config)
    except TypeError as e:
        # Try removing one of the output params
        if "num_outputs" in str(e):
            del model_config["num_outputs"]
            model = model_class(**model_config)
        elif "output_dim" in str(e):
            del model_config["output_dim"]
            model = model_class(**model_config)
        else:
            raise

    # Determine task type
    task_type = "classification" if dataset_name in ["bace", "herg", "cyp3a4", "tox21", "bbbp"] else "regression"

    # Training config - uniform learning rate for all models.
    # Gradient clipping (max_norm=1.0) + L-matrix scaling (1/sqrt(k_dim)) handle DKO stability.
    base_lr = config.get("training.base_learning_rate", 1e-4)

    # Mixed precision: disabled for full DKO (ablation shows RMSE 2.685 vs 2.056 with mp=False,
    # and 10x higher variance). L-matrix scaling is insufficient for fp16 covariance matrices.
    # First-order and baselines are fine with mixed precision.
    is_full_dko = model_name in [
        "dko", "dko_diagonal", "dko_separate_nets",
        "dko_eigenspectrum", "dko_invariants", "dko_lowrank",
        "dko_gated", "dko_residual", "dko_crossattn", "dko_router",
    ]
    use_mixed_precision = False if is_full_dko else config.get("training.mixed_precision", True)

    training_config = {
        "optimizer": config.get("training.optimizer", "AdamW"),
        "base_learning_rate": base_lr,
        "weight_decay": config.get("training.weight_decay", 1e-5),
        "max_epochs": config.get("training.max_epochs", 300),
        "early_stopping_patience": config.get("training.early_stopping_patience", 30),
        "gradient_clip_max_norm": config.get("training.gradient_clip_max_norm", 1.0),
        "mixed_precision": use_mixed_precision,
        "task_type": task_type,
        "scheduler": config.get("training.scheduler", {}),
    }

    # Train
    experiment_name = f"{dataset_name}_{model_name}_seed{seed}"
    model, train_results = train_model(
        model,
        train_loader,
        val_loader,
        training_config,
        device=device,
        experiment_name=experiment_name,
        output_dir=output_dir,
    )

    # Evaluate on test set
    evaluator = Evaluator(task_type=task_type, device=device)
    test_metrics = evaluator.evaluate(model, test_loader)

    logger.info(f"Test metrics: {test_metrics}")

    return {
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "test_metrics": test_metrics,
        "train_results": train_results,
    }


def run_main_benchmark(
    datasets: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    seeds: List[int] = [42, 123, 456],
    config_path: Optional[str] = None,
    output_dir: str = "results/benchmark",
    device: str = "cuda",
) -> Dict:
    """
    Run the main benchmark across datasets and models.

    Args:
        datasets: List of datasets (default: all)
        models: List of models (default: all)
        seeds: Random seeds for multiple runs
        config_path: Path to config file
        output_dir: Output directory for results
        device: Device to use

    Returns:
        Dictionary of all results
    """
    datasets = datasets or AVAILABLE_DATASETS
    models = models or list(MODEL_REGISTRY.keys())

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    if config_path:
        config = Config(base_config_path=config_path)
    else:
        config = Config()

    # Run experiments
    all_results = {}

    # Load partial results if they exist (resume support)
    results_path = output_dir / "benchmark_results.json"
    if results_path.exists():
        try:
            with open(results_path, "r") as f:
                all_results = json.load(f)
            logger.info(f"Loaded existing partial results from {results_path}")
        except Exception:
            all_results = {}

    for dataset in datasets:
        logger.info(f"=== Dataset: {dataset} ===")
        if dataset not in all_results:
            all_results[dataset] = {}

        for model_name in models:
            logger.info(f"--- Model: {model_name} ---")
            if model_name not in all_results[dataset]:
                all_results[dataset][model_name] = []

            # Check which seeds are already done
            done_seeds = set()
            for r in all_results[dataset][model_name]:
                if isinstance(r, dict) and 'seed' in r:
                    done_seeds.add(r['seed'])

            for seed in seeds:
                if seed in done_seeds:
                    logger.info(f"Skipping {dataset}/{model_name}/seed{seed} (already done)")
                    continue

                try:
                    result = run_single_experiment(
                        dataset, model_name, config, seed, device,
                        output_dir=str(output_dir),
                    )
                    all_results[dataset][model_name].append(result)
                except Exception as e:
                    logger.error(f"Error in {dataset}/{model_name}/seed{seed}: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results[dataset][model_name].append({
                        "error": str(e),
                        "seed": seed,
                    })

                # Save results incrementally after each experiment
                with open(results_path, "w") as f:
                    json.dump(all_results, f, indent=2, default=str)

                # Save incremental summary too
                summary = aggregate_results(all_results)
                summary_path = output_dir / "benchmark_summary.json"
                with open(summary_path, "w") as f:
                    json.dump(summary, f, indent=2)

                logger.info(f"Results saved incrementally to {output_dir}")

    # Final summary
    summary = aggregate_results(all_results)
    summary_path = output_dir / "benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"All results saved to {output_dir}")

    return {"results": all_results, "summary": summary}


def aggregate_results(all_results: Dict) -> Dict:
    """
    Aggregate results across seeds.

    Args:
        all_results: Raw results dictionary

    Returns:
        Summary statistics
    """
    summary = {}

    for dataset, model_results in all_results.items():
        summary[dataset] = {}

        for model_name, seed_results in model_results.items():
            # Extract metric values
            metrics_per_seed = {}

            for result in seed_results:
                if "error" in result:
                    continue

                for metric, value in result.get("test_metrics", {}).items():
                    if metric not in metrics_per_seed:
                        metrics_per_seed[metric] = []
                    metrics_per_seed[metric].append(value)

            # Compute statistics
            summary[dataset][model_name] = {}
            for metric, values in metrics_per_seed.items():
                if values:
                    summary[dataset][model_name][metric] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                        "n": int(len(values)),
                    }

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DKO benchmark")
    parser.add_argument("--datasets", nargs="+", default=None, help="Datasets to evaluate")
    parser.add_argument("--models", nargs="+", default=None, help="Models to evaluate")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument("--output-dir", type=str, default="results/benchmark")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    run_main_benchmark(
        datasets=args.datasets,
        models=args.models,
        seeds=args.seeds,
        config_path=args.config,
        output_dir=args.output_dir,
        device=args.device,
    )
