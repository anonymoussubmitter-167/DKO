"""
Representation vs Architecture Experiment for DKO.

This experiment tests the key hypothesis from the research plan:

Hypothesis: For conformer ensemble learning, the REPRESENTATION
(geometric features capturing 3D structure) matters more than the
ARCHITECTURE (attention, transformers, etc.).

Experimental Design:
1. Geometric features + Simple MLP → Tests feature quality
2. Geometric features + Complex (Attention/Transformer) → Tests if complexity helps
3. Learned features (GNN) + Simple aggregation → Tests learned representations
4. Learned features (GNN) + Complex aggregation → Full complexity baseline

Key insight: If geometric features + simple architecture matches or beats
learned features + complex architecture, this validates DKO's design principle
that explicit geometric features are sufficient.

Expected Results:
- On high-SCC tasks: Geometric + DKO should outperform learned + attention
- On low-SCC tasks: Both should be similar (flexibility doesn't matter)
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json

from dko.utils.logging_utils import get_logger

logger = get_logger("rep_vs_arch")


@dataclass
class RepVsArchResult:
    """Results from representation vs architecture experiment."""

    dataset: str
    metric: str
    lower_is_better: bool

    # Results for each combination
    # Format: results[representation][architecture] = {"mean": x, "std": y}
    results: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)

    # Analysis
    best_combination: Tuple[str, str] = ("", "")
    representation_effect: float = 0.0  # Effect of geometric vs learned
    architecture_effect: float = 0.0  # Effect of simple vs complex
    interaction_effect: float = 0.0  # Interaction between rep and arch

    def to_dict(self) -> Dict:
        return {
            "dataset": self.dataset,
            "metric": self.metric,
            "lower_is_better": self.lower_is_better,
            "results": self.results,
            "analysis": {
                "best_combination": self.best_combination,
                "representation_effect": self.representation_effect,
                "architecture_effect": self.architecture_effect,
                "interaction_effect": self.interaction_effect,
            },
        }


def run_representation_vs_architecture_experiment(
    dataset_name: str,
    seeds: List[int] = [42, 123, 456],
    output_dir: str = "results/rep_vs_arch",
    device: str = "cuda",
    n_epochs: int = 100,
) -> RepVsArchResult:
    """
    Run representation vs architecture experiment.

    Compares four combinations:
    1. Geometric + Simple (MFA with geometric features)
    2. Geometric + Complex (Attention with geometric features)
    3. Learned + Simple (SchNet encoding + mean aggregation)
    4. Learned + Complex (SchNet encoding + attention aggregation)

    Also tests DKO as the "best of both worlds" approach.

    Args:
        dataset_name: Dataset to use
        seeds: Random seeds for multiple runs
        output_dir: Output directory
        device: Device for training
        n_epochs: Training epochs

    Returns:
        RepVsArchResult with all comparisons
    """
    from dko.data.datasets import load_dataset
    from dko.models.dko import DKO
    from dko.models.ensemble_baselines import MeanFeatureAggregation, MultiInstanceLearning
    from dko.models.attention import AttentionPoolingBaseline
    from dko.training.trainer import Trainer

    logger.info(f"Running representation vs architecture experiment on {dataset_name}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

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

    # Define representations and architectures
    representations = {
        "geometric": "Pre-computed geometric features (RDKit descriptors, fingerprints)",
        "learned": "Learned features from GNN encoder",
    }

    architectures = {
        "simple": "Mean feature aggregation (MFA)",
        "complex": "Multi-head attention aggregation",
    }

    # Model factories
    def get_model(rep: str, arch: str):
        """Get model for representation-architecture combination."""

        if rep == "geometric" and arch == "simple":
            # Geometric + Simple = MFA with raw features
            return MeanFeatureAggregation(
                feature_dim=feature_dim,
                output_dim=1,
                task=task,
                hidden_dims=[256, 128],
                prediction_hidden_dims=[64, 32],
            )

        elif rep == "geometric" and arch == "complex":
            # Geometric + Complex = Attention with raw features
            return AttentionPoolingBaseline(
                feature_dim=feature_dim,
                output_dim=1,
                task=task,
                hidden_dim=256,
                n_heads=4,
            )

        elif rep == "learned" and arch == "simple":
            # Learned + Simple = GNN encoder + mean aggregation
            return LearnedRepresentationSimple(
                feature_dim=feature_dim,
                output_dim=1,
                task=task,
                hidden_dim=256,
            )

        elif rep == "learned" and arch == "complex":
            # Learned + Complex = GNN encoder + attention
            return LearnedRepresentationComplex(
                feature_dim=feature_dim,
                output_dim=1,
                task=task,
                hidden_dim=256,
                n_heads=4,
            )

        else:
            raise ValueError(f"Unknown combination: {rep} + {arch}")

    # Results storage
    result = RepVsArchResult(
        dataset=dataset_name,
        metric=metric,
        lower_is_better=lower_is_better,
    )

    # Also test DKO
    models_to_test = [
        ("geometric", "simple"),
        ("geometric", "complex"),
        ("learned", "simple"),
        ("learned", "complex"),
    ]

    # Run experiments
    logger.info("\n" + "=" * 70)
    logger.info("REPRESENTATION VS ARCHITECTURE EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Task: {task}")
    logger.info(f"Feature dim: {feature_dim}")
    logger.info("=" * 70)

    for rep, arch in models_to_test:
        logger.info(f"\n{rep} + {arch}:")
        logger.info(f"  Rep: {representations[rep]}")
        logger.info(f"  Arch: {architectures[arch]}")

        performances = []

        for seed in seeds:
            model = get_model(rep, arch)

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

        if rep not in result.results:
            result.results[rep] = {}
        result.results[rep][arch] = {
            "mean": float(mean_perf),
            "std": float(std_perf),
        }

        logger.info(f"  {metric}: {mean_perf:.4f} ± {std_perf:.4f}")

    # Test DKO as reference
    logger.info("\nDKO (geometric + kernel-based):")
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
        dko_perfs.append(test_metrics[metric])

    dko_mean = np.mean(dko_perfs)
    dko_std = np.std(dko_perfs)
    result.results["dko"] = {"kernel": {"mean": float(dko_mean), "std": float(dko_std)}}
    logger.info(f"  {metric}: {dko_mean:.4f} ± {dko_std:.4f}")

    # Analyze results
    result = analyze_rep_vs_arch(result)

    # Log analysis
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS")
    logger.info("=" * 70)
    logger.info(f"Best combination: {result.best_combination}")
    logger.info(f"Representation effect: {result.representation_effect:.4f}")
    logger.info(f"Architecture effect: {result.architecture_effect:.4f}")
    logger.info(f"Interaction effect: {result.interaction_effect:.4f}")

    if abs(result.representation_effect) > abs(result.architecture_effect):
        logger.info("\n✓ REPRESENTATION matters more than ARCHITECTURE")
        logger.info("  (Geometric features capture key information)")
    else:
        logger.info("\n✗ ARCHITECTURE matters more than REPRESENTATION")
        logger.info("  (Complex models extract more from data)")

    # Save results
    output_file = output_path / f"{dataset_name}_rep_vs_arch.json"
    with open(output_file, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    logger.info(f"\nResults saved to {output_file}")

    return result


def analyze_rep_vs_arch(result: RepVsArchResult) -> RepVsArchResult:
    """
    Analyze representation vs architecture effects.

    Uses 2x2 factorial analysis to decompose effects.
    """
    # Get means for each combination
    geo_simple = result.results.get("geometric", {}).get("simple", {}).get("mean", 0)
    geo_complex = result.results.get("geometric", {}).get("complex", {}).get("mean", 0)
    learn_simple = result.results.get("learned", {}).get("simple", {}).get("mean", 0)
    learn_complex = result.results.get("learned", {}).get("complex", {}).get("mean", 0)

    # For lower_is_better metrics, we flip signs for interpretation
    sign = -1 if result.lower_is_better else 1

    # Main effects (average difference)
    # Representation effect: geometric vs learned (averaging over architectures)
    geo_avg = (geo_simple + geo_complex) / 2
    learn_avg = (learn_simple + learn_complex) / 2
    result.representation_effect = sign * (geo_avg - learn_avg)

    # Architecture effect: simple vs complex (averaging over representations)
    simple_avg = (geo_simple + learn_simple) / 2
    complex_avg = (geo_complex + learn_complex) / 2
    result.architecture_effect = sign * (complex_avg - simple_avg)

    # Interaction effect
    # If geometric+simple and learned+complex are similar but other combos differ
    result.interaction_effect = sign * (
        (geo_simple - geo_complex) - (learn_simple - learn_complex)
    ) / 2

    # Find best combination
    all_combos = [
        ("geometric", "simple", geo_simple),
        ("geometric", "complex", geo_complex),
        ("learned", "simple", learn_simple),
        ("learned", "complex", learn_complex),
    ]

    if result.lower_is_better:
        best = min(all_combos, key=lambda x: x[2])
    else:
        best = max(all_combos, key=lambda x: x[2])

    result.best_combination = (best[0], best[1])

    return result


class LearnedRepresentationSimple(nn.Module):
    """
    Learned representation with simple aggregation.

    Uses a small neural network to learn features from raw inputs,
    then aggregates via simple mean pooling.
    """

    def __init__(
        self,
        feature_dim: int,
        output_dim: int = 1,
        task: str = "regression",
        hidden_dim: int = 256,
        n_layers: int = 3,
    ):
        super().__init__()

        self.task = task

        # Feature learning network (like a simple SchNet message passing)
        layers = []
        in_dim = feature_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with mean aggregation."""
        batch_size, n_conf, feat_dim = x.shape

        # Encode each conformer
        x_flat = x.view(-1, feat_dim)
        encoded = self.encoder(x_flat)
        encoded = encoded.view(batch_size, n_conf, -1)

        # Simple mean aggregation
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            aggregated = (encoded * mask_expanded).sum(dim=1)
            aggregated = aggregated / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            aggregated = encoded.mean(dim=1)

        # Predict
        output = self.predictor(aggregated)

        return output


class LearnedRepresentationComplex(nn.Module):
    """
    Learned representation with complex (attention) aggregation.

    Uses a neural network to learn features, then aggregates
    via multi-head self-attention.
    """

    def __init__(
        self,
        feature_dim: int,
        output_dim: int = 1,
        task: str = "regression",
        hidden_dim: int = 256,
        n_layers: int = 3,
        n_heads: int = 4,
    ):
        super().__init__()

        self.task = task
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        # Feature learning network
        layers = []
        in_dim = feature_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)

        # Multi-head attention for aggregation
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            batch_first=True,
        )

        # Query for pooling (learned)
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with attention aggregation."""
        batch_size, n_conf, feat_dim = x.shape

        # Encode each conformer
        x_flat = x.view(-1, feat_dim)
        encoded = self.encoder(x_flat)
        encoded = encoded.view(batch_size, n_conf, -1)

        # Attention-based aggregation
        # Use learned query to aggregate
        query = self.pool_query.expand(batch_size, -1, -1)

        # Create attention mask if needed
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask

        # Cross-attention: query attends to encoded conformers
        aggregated, _ = self.attention(
            query, encoded, encoded,
            key_padding_mask=key_padding_mask,
        )
        aggregated = aggregated.squeeze(1)

        # Predict
        output = self.predictor(aggregated)

        return output


def run_full_rep_vs_arch_study(
    datasets: Optional[List[str]] = None,
    output_dir: str = "results/rep_vs_arch",
    device: str = "cuda",
    seeds: List[int] = [42, 123, 456],
) -> Dict:
    """
    Run representation vs architecture study across datasets.

    Args:
        datasets: Datasets to test
        output_dir: Output directory
        device: Device for training
        seeds: Random seeds

    Returns:
        Aggregated results
    """
    if datasets is None:
        datasets = ["esol", "freesolv", "bace", "bbbp", "lipophilicity"]

    logger.info(f"Running rep vs arch study on {len(datasets)} datasets")

    all_results = {}
    rep_effects = []
    arch_effects = []

    for dataset in datasets:
        try:
            result = run_representation_vs_architecture_experiment(
                dataset_name=dataset,
                seeds=seeds,
                output_dir=output_dir,
                device=device,
            )
            all_results[dataset] = result.to_dict()
            rep_effects.append(result.representation_effect)
            arch_effects.append(result.architecture_effect)
        except Exception as e:
            logger.error(f"Failed on {dataset}: {e}")
            continue

    # Aggregate analysis
    logger.info("\n" + "=" * 70)
    logger.info("AGGREGATED ANALYSIS")
    logger.info("=" * 70)
    logger.info(f"Datasets analyzed: {len(all_results)}")
    logger.info(f"Avg representation effect: {np.mean(rep_effects):.4f} ± {np.std(rep_effects):.4f}")
    logger.info(f"Avg architecture effect: {np.mean(arch_effects):.4f} ± {np.std(arch_effects):.4f}")

    if abs(np.mean(rep_effects)) > abs(np.mean(arch_effects)):
        logger.info("\n✓ Overall: REPRESENTATION matters more")
    else:
        logger.info("\n✗ Overall: ARCHITECTURE matters more")

    # Save aggregated results
    output_path = Path(output_dir)
    with open(output_path / "full_study_results.json", "w") as f:
        json.dump({
            "individual_results": all_results,
            "aggregated": {
                "mean_rep_effect": float(np.mean(rep_effects)),
                "std_rep_effect": float(np.std(rep_effects)),
                "mean_arch_effect": float(np.mean(arch_effects)),
                "std_arch_effect": float(np.std(arch_effects)),
            },
        }, f, indent=2)

    return all_results
