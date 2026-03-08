"""
Attention Analysis and Scaling Experiments for DKO.

This module provides:
1. Attention weight analysis - comparing learned attention to Boltzmann weights
2. Attention scaling experiment - testing Conjecture 3.3

Conjecture 3.3 (Attention Sample Complexity Scaling):
Attention trained end-to-end on covariance-dependent properties requires
sample complexity that scales as: n_cov_params / hidden_dim

where n_cov_params = D*(D+1)/2 is the number of unique covariance entries.

Experimental design from research plan:
- Fix task to high-SCC subset of dataset (e.g., BACE)
- Vary attention hidden dimension: 1/4, 1/2, 1, 2, 4 × feature_dim
- Measure conformers required for convergence at each hidden dim
- Fit log-log relationship between sample requirement and cov_params/hidden_dim
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
from scipy import stats as scipy_stats

from dko.utils.logging_utils import get_logger

logger = get_logger("attention_analysis")


@dataclass
class ScalingExperimentResult:
    """Results from attention scaling experiment."""

    hidden_dims: List[int]
    convergence_conformers: List[int]
    final_performance: List[float]
    scaling_exponent: float
    r_squared: float
    feature_dim: int
    cov_params: int
    dataset: str


def run_attention_analysis(
    dataset_name: str,
    model_path: str,
    output_dir: str = "results/attention",
    device: str = "cuda",
    n_samples: int = 100,
) -> Dict:
    """
    Analyze attention patterns in trained model.

    Compares learned attention weights to ground-truth Boltzmann weights
    to test whether attention learns physically meaningful aggregation.

    Args:
        dataset_name: Dataset name
        model_path: Path to trained model checkpoint
        output_dir: Output directory for visualizations
        device: Device
        n_samples: Number of samples to analyze

    Returns:
        Dictionary with attention statistics
    """
    from dko.models.baselines import AttentionPoolingBaseline
    from dko.data.datasets import load_dataset

    logger.info(f"Analyzing attention patterns for {dataset_name}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint.get("config", {})

    # Load dataset
    dataset = load_dataset(dataset_name)

    # Create model
    model = AttentionPoolingBaseline(
        feature_dim=dataset.feature_dim,
        output_dim=1,
        **model_config.get("model", {}),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Collect attention weights and Boltzmann weights
    attention_weights_all = []
    boltzmann_weights_all = []
    correlations = []

    with torch.no_grad():
        for i, batch in enumerate(dataset.test_loader):
            if i >= n_samples:
                break

            # Get attention weights from model
            features = batch["features"].to(device)
            mask = batch.get("mask", None)
            if mask is not None:
                mask = mask.to(device)

            # Forward pass to get attention weights
            _, attn_weights = model(features, mask, return_attention=True)

            # Get Boltzmann weights
            boltz_weights = batch.get("boltzmann_weights", None)

            if boltz_weights is not None and attn_weights is not None:
                # Compute correlation
                for j in range(attn_weights.size(0)):
                    attn = attn_weights[j].cpu().numpy()
                    boltz = boltz_weights[j].cpu().numpy()

                    # Truncate to same length
                    min_len = min(len(attn), len(boltz))
                    if min_len > 1:
                        corr, _ = scipy_stats.pearsonr(attn[:min_len], boltz[:min_len])
                        if not np.isnan(corr):
                            correlations.append(corr)

                attention_weights_all.append(attn_weights.cpu())
                boltzmann_weights_all.append(boltz_weights.cpu())

    # Compute statistics
    if attention_weights_all:
        all_attn = torch.cat(attention_weights_all, dim=0)
        stats = compute_attention_statistics(all_attn)
    else:
        stats = {"mean_entropy": 0.0, "mean_concentration": 0.0}

    # Add correlation statistics
    if correlations:
        stats["boltzmann_correlation"] = {
            "mean": float(np.mean(correlations)),
            "std": float(np.std(correlations)),
            "min": float(np.min(correlations)),
            "max": float(np.max(correlations)),
        }
    else:
        stats["boltzmann_correlation"] = {"mean": 0.0, "std": 0.0}

    # Hypothesis from research plan: low correlation (<0.3) expected on high-SCC tasks
    # where DKO outperforms attention
    logger.info(f"Attention-Boltzmann correlation: {stats['boltzmann_correlation']['mean']:.3f}")

    # Save results
    with open(output_path / f"{dataset_name}_attention_analysis.json", "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def compute_attention_statistics(
    attention_weights: torch.Tensor,
    energies: Optional[torch.Tensor] = None,
) -> Dict:
    """
    Compute statistics from attention weights.

    Args:
        attention_weights: Attention weights (batch, n_conformers)
        energies: Optional conformer energies

    Returns:
        Dictionary of statistics
    """
    # Entropy: H = -sum(p * log(p))
    eps = 1e-8
    entropy = -(attention_weights * torch.log(attention_weights + eps)).sum(dim=-1)
    mean_entropy = entropy.mean().item()
    std_entropy = entropy.std().item()

    # Concentration (max weight)
    max_weights = attention_weights.max(dim=-1)[0]
    mean_concentration = max_weights.mean().item()

    # Effective number of conformers (exp of entropy)
    effective_n = torch.exp(entropy).mean().item()

    # Uniformity (distance from uniform distribution)
    n_conformers = attention_weights.size(-1)
    uniform = torch.ones_like(attention_weights) / n_conformers
    kl_from_uniform = (attention_weights * torch.log((attention_weights + eps) / uniform)).sum(dim=-1)
    mean_kl = kl_from_uniform.mean().item()

    stats = {
        "mean_entropy": mean_entropy,
        "std_entropy": std_entropy,
        "mean_concentration": mean_concentration,
        "effective_conformers": effective_n,
        "kl_from_uniform": mean_kl,
        "n_conformers": n_conformers,
    }

    return stats


def run_attention_scaling_experiment(
    dataset_name: str,
    hidden_dim_multipliers: List[float] = [0.25, 0.5, 1.0, 2.0, 4.0],
    conformer_counts: List[int] = [5, 10, 20, 30, 50],
    output_dir: str = "results/scaling",
    device: str = "cuda",
    seeds: List[int] = [42, 123, 456],
    n_epochs: int = 100,
    convergence_threshold: float = 0.01,
) -> ScalingExperimentResult:
    """
    Run attention scaling experiment to test Conjecture 3.3.

    Conjecture: Sample complexity scales as cov_params / hidden_dim

    Experimental procedure:
    1. For each hidden_dim setting:
       - Train attention model with varying conformer counts
       - Find minimum conformers needed for convergence
    2. Fit log-log relationship between:
       - x = cov_params / hidden_dim (theoretical complexity ratio)
       - y = convergence conformer count
    3. Expected: scaling exponent between 0.7-1.3

    Args:
        dataset_name: Dataset to use (should be high-SCC task like BACE)
        hidden_dim_multipliers: Multipliers for hidden dim relative to feature_dim
        conformer_counts: Conformer counts to test
        output_dir: Output directory
        device: Device for training
        seeds: Random seeds
        n_epochs: Training epochs
        convergence_threshold: Relative improvement threshold for convergence

    Returns:
        ScalingExperimentResult with fitted scaling relationship
    """
    from dko.data.datasets import load_dataset
    from dko.models.baselines import AttentionPoolingBaseline
    from dko.training.trainer import Trainer

    logger.info(f"Running attention scaling experiment on {dataset_name}")
    logger.info(f"Hidden dim multipliers: {hidden_dim_multipliers}")
    logger.info(f"Conformer counts: {conformer_counts}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load dataset info
    dataset = load_dataset(dataset_name)
    feature_dim = dataset.feature_dim
    task = dataset.task

    # Compute covariance parameters
    cov_params = feature_dim * (feature_dim + 1) // 2

    logger.info(f"Feature dimension: {feature_dim}")
    logger.info(f"Covariance parameters: {cov_params}")

    # Results storage
    hidden_dims = []
    convergence_conformers = []
    final_performances = []

    for multiplier in hidden_dim_multipliers:
        hidden_dim = max(16, int(feature_dim * multiplier))
        hidden_dims.append(hidden_dim)

        logger.info(f"\n{'='*60}")
        logger.info(f"Testing hidden_dim = {hidden_dim} ({multiplier}x feature_dim)")
        logger.info(f"Complexity ratio (cov_params/hidden_dim): {cov_params/hidden_dim:.1f}")
        logger.info("="*60)

        # Find convergence point
        prev_performance = None
        converged_at = conformer_counts[-1]  # Default to max
        best_perf = None

        for n_conf in conformer_counts:
            logger.info(f"\n  Conformers: {n_conf}")

            # Train with this conformer count
            performances = []

            for seed in seeds:
                # Create model
                model = AttentionPoolingBaseline(
                    feature_dim=feature_dim,
                    hidden_dim=hidden_dim,
                    output_dim=1,
                    task=task,
                    n_heads=max(1, hidden_dim // 32),
                )

                # Create trainer
                trainer = Trainer(
                    model=model,
                    config={
                        "lr": 1e-4,
                        "n_epochs": n_epochs,
                        "patience": 30,
                        "seed": seed,
                        "n_conformers": n_conf,  # Limit conformers
                    },
                    device=device,
                )

                # Train
                trainer.fit(dataset.train_loader, dataset.val_loader)

                # Evaluate
                test_metrics = trainer.evaluate(dataset.test_loader)
                metric_value = test_metrics.get("rmse", test_metrics.get("auroc", 0))
                performances.append(metric_value)

            mean_perf = np.mean(performances)
            std_perf = np.std(performances)
            logger.info(f"    Performance: {mean_perf:.4f} ± {std_perf:.4f}")

            # Check convergence
            if prev_performance is not None:
                rel_improvement = abs(prev_performance - mean_perf) / (abs(prev_performance) + 1e-8)
                if rel_improvement < convergence_threshold:
                    converged_at = n_conf
                    best_perf = mean_perf
                    logger.info(f"    Converged! (improvement {rel_improvement:.4f} < {convergence_threshold})")
                    break

            prev_performance = mean_perf
            best_perf = mean_perf

        convergence_conformers.append(converged_at)
        final_performances.append(best_perf if best_perf else prev_performance)

        logger.info(f"\n  Convergence at {converged_at} conformers")

    # Fit scaling relationship
    # y = a * x^b  =>  log(y) = log(a) + b * log(x)
    # x = cov_params / hidden_dim
    # y = convergence conformers

    x_values = np.array([cov_params / h for h in hidden_dims])
    y_values = np.array(convergence_conformers)

    # Log-log regression
    log_x = np.log(x_values)
    log_y = np.log(y_values)

    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(log_x, log_y)
    scaling_exponent = slope
    r_squared = r_value ** 2

    logger.info("\n" + "=" * 60)
    logger.info("SCALING ANALYSIS RESULTS")
    logger.info("=" * 60)
    logger.info(f"Scaling exponent (b in y = a*x^b): {scaling_exponent:.3f}")
    logger.info(f"R-squared: {r_squared:.3f}")
    logger.info(f"Expected range for Conjecture 3.3: 0.7 - 1.3")

    if 0.7 <= scaling_exponent <= 1.3:
        logger.info("✓ Conjecture 3.3 SUPPORTED")
    else:
        logger.info("✗ Conjecture 3.3 NOT supported")

    # Create result object
    result = ScalingExperimentResult(
        hidden_dims=hidden_dims,
        convergence_conformers=convergence_conformers,
        final_performance=final_performances,
        scaling_exponent=scaling_exponent,
        r_squared=r_squared,
        feature_dim=feature_dim,
        cov_params=cov_params,
        dataset=dataset_name,
    )

    # Save results
    with open(output_path / f"{dataset_name}_scaling.json", "w") as f:
        json.dump({
            "hidden_dims": hidden_dims,
            "convergence_conformers": convergence_conformers,
            "final_performance": final_performances,
            "complexity_ratios": x_values.tolist(),
            "scaling_exponent": scaling_exponent,
            "r_squared": r_squared,
            "feature_dim": feature_dim,
            "cov_params": cov_params,
            "conjecture_supported": 0.7 <= scaling_exponent <= 1.3,
        }, f, indent=2)

    logger.info(f"\nResults saved to {output_path / f'{dataset_name}_scaling.json'}")

    return result


def run_control_scaling_experiment(
    control_dataset: str = "qm9_homo",
    **kwargs,
) -> ScalingExperimentResult:
    """
    Run scaling experiment on a control (low-SCC) dataset.

    For control datasets (e.g., QM9 electronic properties), we expect
    scaling exponent near zero because the property doesn't depend on
    covariance information.

    Args:
        control_dataset: Low-SCC dataset name
        **kwargs: Arguments passed to run_attention_scaling_experiment

    Returns:
        ScalingExperimentResult for control
    """
    logger.info("\n" + "=" * 60)
    logger.info("CONTROL EXPERIMENT (Low-SCC dataset)")
    logger.info("=" * 60)
    logger.info(f"Dataset: {control_dataset}")
    logger.info("Expected: scaling exponent near 0 (no covariance dependence)")

    result = run_attention_scaling_experiment(
        dataset_name=control_dataset,
        output_dir=kwargs.get("output_dir", "results/scaling") + "/control",
        **{k: v for k, v in kwargs.items() if k != "output_dir"}
    )

    if abs(result.scaling_exponent) < 0.3:
        logger.info("✓ Control validated (exponent near 0)")
    else:
        logger.info("⚠ Control shows unexpected scaling - investigate confounds")

    return result


def visualize_attention_weights(
    model: torch.nn.Module,
    sample: Dict,
    output_path: str,
) -> None:
    """
    Visualize attention weights for a single molecule.

    Creates a bar plot comparing attention weights to Boltzmann weights.

    Args:
        model: Trained model
        sample: Sample dictionary with conformer features
        output_path: Path to save visualization
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for visualization")
        return

    model.eval()
    with torch.no_grad():
        features = sample["features"].unsqueeze(0)
        _, attn_weights = model(features, return_attention=True)
        attn_weights = attn_weights[0].cpu().numpy()

    boltz_weights = sample.get("boltzmann_weights", None)
    if boltz_weights is not None:
        boltz_weights = boltz_weights.cpu().numpy()

    # Create visualization
    n_conf = len(attn_weights)
    x = np.arange(n_conf)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - width/2, attn_weights, width, label='Attention', alpha=0.8)
    if boltz_weights is not None:
        ax.bar(x + width/2, boltz_weights[:n_conf], width, label='Boltzmann', alpha=0.8)

    ax.set_xlabel('Conformer Index')
    ax.set_ylabel('Weight')
    ax.set_title('Attention vs Boltzmann Weights')
    ax.legend()
    ax.set_xticks(x)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"Saved visualization to {output_path}")
