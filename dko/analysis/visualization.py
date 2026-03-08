"""
Visualization utilities for DKO experiments.

This module provides functions for creating plots and tables
for experiment results.
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def check_matplotlib():
    """Check if matplotlib is available."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib and seaborn required for visualization")


def plot_learning_curves(
    history: List[Dict],
    metrics: List[str] = ["loss"],
    title: str = "Learning Curves",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    Plot training and validation learning curves.

    Args:
        history: List of epoch metrics dictionaries
        metrics: Metrics to plot
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    check_matplotlib()

    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        train_key = f"train_{metric}"
        val_key = f"val_{metric}"

        epochs = [h["epoch"] for h in history]
        train_values = [h.get(train_key, h.get(metric)) for h in history]
        val_values = [h.get(val_key) for h in history if val_key in h]

        ax.plot(epochs, train_values, label="Train", linewidth=2)
        if val_values:
            ax.plot(epochs[:len(val_values)], val_values, label="Validation", linewidth=2)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} vs Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_attention_weights(
    attention_weights: np.ndarray,
    conformer_energies: Optional[np.ndarray] = None,
    title: str = "Attention Weights",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> None:
    """
    Plot attention weights over conformers.

    Args:
        attention_weights: Attention weights (n_conformers,)
        conformer_energies: Optional conformer energies
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    check_matplotlib()

    fig, axes = plt.subplots(1, 2 if conformer_energies is not None else 1, figsize=figsize)

    if conformer_energies is None:
        axes = [axes]

    # Plot attention weights
    ax = axes[0]
    n_conf = len(attention_weights)
    bars = ax.bar(range(n_conf), attention_weights, color="steelblue", alpha=0.7)
    ax.set_xlabel("Conformer Index")
    ax.set_ylabel("Attention Weight")
    ax.set_title("Attention Distribution")

    # Highlight max
    max_idx = np.argmax(attention_weights)
    bars[max_idx].set_color("crimson")

    # Plot correlation with energy if available
    if conformer_energies is not None:
        ax = axes[1]
        ax.scatter(conformer_energies, attention_weights, alpha=0.7)
        ax.set_xlabel("Energy (kcal/mol)")
        ax.set_ylabel("Attention Weight")
        ax.set_title("Attention vs Energy")

        # Add correlation
        corr = np.corrcoef(conformer_energies, attention_weights)[0, 1]
        ax.annotate(f"r = {corr:.3f}", xy=(0.05, 0.95), xycoords="axes fraction")

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_conformer_distributions(
    conformer_counts: List[int],
    energies: Optional[List[List[float]]] = None,
    title: str = "Conformer Distributions",
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot distribution of conformer counts and energies.

    Args:
        conformer_counts: Number of conformers per molecule
        energies: Optional list of energy lists per molecule
        title: Plot title
        save_path: Path to save figure
    """
    check_matplotlib()

    fig, axes = plt.subplots(1, 2 if energies else 1, figsize=(12, 5))

    if energies is None:
        axes = [axes]

    # Conformer count histogram
    ax = axes[0]
    ax.hist(conformer_counts, bins=20, color="steelblue", alpha=0.7, edgecolor="black")
    ax.set_xlabel("Number of Conformers")
    ax.set_ylabel("Frequency")
    ax.set_title("Conformer Count Distribution")
    ax.axvline(np.mean(conformer_counts), color="red", linestyle="--",
               label=f"Mean: {np.mean(conformer_counts):.1f}")
    ax.legend()

    # Energy distribution
    if energies:
        ax = axes[1]
        all_energies = [e for energy_list in energies for e in energy_list]
        ax.hist(all_energies, bins=50, color="coral", alpha=0.7, edgecolor="black")
        ax.set_xlabel("Energy (kcal/mol)")
        ax.set_ylabel("Frequency")
        ax.set_title("Conformer Energy Distribution")

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_performance_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = "rmse",
    title: str = "Model Comparison",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    Plot performance comparison across models.

    Args:
        results: Dictionary mapping model name to metrics
        metric: Metric to plot
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    check_matplotlib()

    models = list(results.keys())
    means = [results[m].get(metric, {}).get("mean", 0) for m in models]
    stds = [results[m].get(metric, {}).get("std", 0) for m in models]

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(models))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color="steelblue", alpha=0.7)

    # Highlight best model
    best_idx = np.argmin(means) if metric in ["rmse", "mae", "loss"] else np.argmax(means)
    bars[best_idx].set_color("green")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def create_summary_table(
    results: Dict[str, Dict[str, Dict]],
    metrics: List[str] = ["rmse", "mae"],
    format: str = "markdown",
) -> str:
    """
    Create a summary table of results.

    Args:
        results: Dictionary of dataset -> model -> metrics
        metrics: Metrics to include
        format: Output format ('markdown', 'latex', 'csv')

    Returns:
        Formatted table string
    """
    # Get all datasets and models
    datasets = list(results.keys())
    models = set()
    for dataset_results in results.values():
        models.update(dataset_results.keys())
    models = sorted(models)

    # Build table
    if format == "markdown":
        # Header
        header = "| Dataset | " + " | ".join(models) + " |"
        separator = "|" + "|".join(["---"] * (len(models) + 1)) + "|"

        rows = [header, separator]

        for dataset in datasets:
            row_values = [dataset]
            for model in models:
                model_results = results[dataset].get(model, {})
                if model_results:
                    metric = metrics[0]  # Primary metric
                    mean = model_results.get(metric, {}).get("mean", 0)
                    std = model_results.get(metric, {}).get("std", 0)
                    row_values.append(f"{mean:.4f} ({std:.4f})")
                else:
                    row_values.append("-")
            rows.append("| " + " | ".join(row_values) + " |")

        return "\n".join(rows)

    elif format == "latex":
        # LaTeX table
        lines = [
            "\\begin{table}[h]",
            "\\centering",
            "\\begin{tabular}{l" + "c" * len(models) + "}",
            "\\toprule",
            "Dataset & " + " & ".join(models) + " \\\\",
            "\\midrule",
        ]

        for dataset in datasets:
            row_values = [dataset]
            for model in models:
                model_results = results[dataset].get(model, {})
                if model_results:
                    metric = metrics[0]
                    mean = model_results.get(metric, {}).get("mean", 0)
                    std = model_results.get(metric, {}).get("std", 0)
                    row_values.append(f"${mean:.4f} \\pm {std:.4f}$")
                else:
                    row_values.append("-")
            lines.append(" & ".join(row_values) + " \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])

        return "\n".join(lines)

    elif format == "csv":
        # CSV format
        header = "dataset," + ",".join(f"{m}_mean,{m}_std" for m in metrics for _ in models)
        lines = [header]

        for dataset in datasets:
            row_values = [dataset]
            for model in models:
                for metric in metrics:
                    model_results = results[dataset].get(model, {})
                    mean = model_results.get(metric, {}).get("mean", "")
                    std = model_results.get(metric, {}).get("std", "")
                    row_values.extend([str(mean), str(std)])
            lines.append(",".join(row_values))

        return "\n".join(lines)

    else:
        raise ValueError(f"Unknown format: {format}")
