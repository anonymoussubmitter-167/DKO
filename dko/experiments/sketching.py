"""
Sketching Experiments for DKO.

This module implements random sketching techniques for scaling DKO
to large conformer ensembles.

Key idea: Instead of computing full covariance matrix O(N^2), use
random sketching to approximate the distribution with O(k) samples
where k << N.

Sketching Methods:
1. Random sampling: Uniformly sample k conformers
2. Energy-weighted sampling: Sample proportional to Boltzmann weights
3. Coreset selection: Select diverse representative conformers
4. Feature hashing: Project features to lower dimension

The experiment measures:
- Performance degradation vs full ensemble
- Computational speedup
- Memory savings
- Optimal sketch size for different dataset sizes
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import torch
import time
from pathlib import Path
import json

from dko.utils.logging_utils import get_logger

logger = get_logger("sketching")


@dataclass
class SketchingResult:
    """Results from sketching experiment."""

    dataset: str
    full_ensemble_size: int

    # Results per sketch method and size
    # Format: results[method][size] = {metric: value}
    results: Dict[str, Dict[int, Dict[str, float]]] = field(default_factory=dict)

    # Best configuration
    best_method: str = ""
    best_size: int = 0
    best_tradeoff: float = 0.0  # Performance / cost ratio

    def to_dict(self) -> Dict:
        return {
            "dataset": self.dataset,
            "full_ensemble_size": self.full_ensemble_size,
            "results": self.results,
            "best": {
                "method": self.best_method,
                "size": self.best_size,
                "tradeoff": self.best_tradeoff,
            },
        }


def run_sketching_experiment(
    dataset_name: str,
    sketch_sizes: List[int] = [5, 10, 20, 30, 50],
    full_ensemble_size: int = 100,
    methods: Optional[List[str]] = None,
    seeds: List[int] = [42, 123, 456],
    config_path: Optional[str] = None,
    output_dir: str = "results/sketching",
    device: str = "cuda",
    n_epochs: int = 50,
) -> SketchingResult:
    """
    Run sketching experiment.

    Compares performance with different sketch sizes and methods vs full ensemble.

    Args:
        dataset_name: Dataset to use
        sketch_sizes: Sketch sizes to test
        full_ensemble_size: Full conformer ensemble size
        methods: Sketching methods to test
        seeds: Random seeds
        config_path: Path to config
        output_dir: Output directory
        device: Device
        n_epochs: Training epochs

    Returns:
        SketchingResult with all metrics
    """
    from dko.data.datasets import load_dataset
    from dko.models.dko import DKO
    from dko.training.trainer import Trainer

    logger.info("=" * 70)
    logger.info("SKETCHING EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Full ensemble size: {full_ensemble_size}")
    logger.info(f"Sketch sizes: {sketch_sizes}")
    logger.info("=" * 70)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if methods is None:
        methods = ["random", "energy_weighted", "diverse"]

    # Load dataset
    dataset = load_dataset(dataset_name)
    feature_dim = dataset.feature_dim
    task = dataset.task

    metric = "rmse" if task == "regression" else "auroc"
    lower_is_better = task == "regression"

    result = SketchingResult(
        dataset=dataset_name,
        full_ensemble_size=full_ensemble_size,
    )

    # First, train with full ensemble as baseline
    logger.info("\n--- Full Ensemble Baseline ---")
    full_results = train_and_evaluate(
        dataset=dataset,
        feature_dim=feature_dim,
        task=task,
        n_conformers=full_ensemble_size,
        seeds=seeds,
        device=device,
        n_epochs=n_epochs,
    )
    result.results["full"] = {
        full_ensemble_size: full_results,
    }
    logger.info(f"Full ensemble {metric}: {full_results[metric + '_mean']:.4f}")

    baseline_metric = full_results[metric + "_mean"]

    # Test each sketching method and size
    for method in methods:
        logger.info(f"\n--- Method: {method} ---")
        result.results[method] = {}

        for size in sketch_sizes:
            if size >= full_ensemble_size:
                continue

            logger.info(f"  Sketch size: {size}")

            # Apply sketching
            sketch_results = train_with_sketching(
                dataset=dataset,
                feature_dim=feature_dim,
                task=task,
                sketch_size=size,
                method=method,
                seeds=seeds,
                device=device,
                n_epochs=n_epochs,
            )

            # Compute relative error
            sketch_metric = sketch_results[metric + "_mean"]
            if lower_is_better:
                relative_error = (sketch_metric - baseline_metric) / baseline_metric * 100
            else:
                relative_error = (baseline_metric - sketch_metric) / baseline_metric * 100

            sketch_results["relative_error"] = relative_error
            sketch_results["speedup"] = full_ensemble_size / size

            result.results[method][size] = sketch_results

            logger.info(f"    {metric}: {sketch_metric:.4f} (error: {relative_error:+.2f}%)")
            logger.info(f"    Speedup: {sketch_results['speedup']:.1f}x")

    # Find best tradeoff (minimize error * cost)
    best_tradeoff = float("inf")
    for method, sizes in result.results.items():
        if method == "full":
            continue
        for size, metrics in sizes.items():
            # Tradeoff: error * (1 / speedup) - want low error and high speedup
            error = abs(metrics.get("relative_error", 100))
            speedup = metrics.get("speedup", 1)
            tradeoff = error / speedup  # Lower is better
            if tradeoff < best_tradeoff:
                best_tradeoff = tradeoff
                result.best_method = method
                result.best_size = size
                result.best_tradeoff = tradeoff

    # Log summary
    logger.info("\n" + "=" * 70)
    logger.info("SKETCHING SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Best method: {result.best_method}")
    logger.info(f"Best size: {result.best_size}")
    logger.info(f"Best tradeoff score: {result.best_tradeoff:.4f}")

    if result.best_method in result.results:
        best = result.results[result.best_method].get(result.best_size, {})
        logger.info(f"  Performance: {best.get(metric + '_mean', 0):.4f}")
        logger.info(f"  Relative error: {best.get('relative_error', 0):.2f}%")
        logger.info(f"  Speedup: {best.get('speedup', 1):.1f}x")
    logger.info("=" * 70)

    # Save results
    output_file = output_path / f"{dataset_name}_sketching.json"
    with open(output_file, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    logger.info(f"\nResults saved to {output_file}")

    return result


def train_and_evaluate(
    dataset,
    feature_dim: int,
    task: str,
    n_conformers: int,
    seeds: List[int],
    device: str,
    n_epochs: int,
) -> Dict[str, float]:
    """Train and evaluate DKO with specified number of conformers."""
    from dko.models.dko import DKO
    from dko.training.trainer import Trainer

    metric = "rmse" if task == "regression" else "auroc"
    performances = []
    times = []

    for seed in seeds:
        model = DKO(
            feature_dim=feature_dim,
            output_dim=1,
            task=task,
            use_second_order=True,
            verbose=False,
        )

        trainer = Trainer(
            model=model,
            config={
                "lr": 1e-4,
                "n_epochs": n_epochs,
                "seed": seed,
                "n_conformers": n_conformers,
            },
            device=device,
        )

        start_time = time.time()
        trainer.fit(dataset.train_loader, dataset.val_loader)
        train_time = time.time() - start_time

        test_metrics = trainer.evaluate(dataset.test_loader)
        performances.append(test_metrics[metric])
        times.append(train_time)

    return {
        f"{metric}_mean": float(np.mean(performances)),
        f"{metric}_std": float(np.std(performances)),
        "train_time_mean": float(np.mean(times)),
        "train_time_std": float(np.std(times)),
    }


def train_with_sketching(
    dataset,
    feature_dim: int,
    task: str,
    sketch_size: int,
    method: str,
    seeds: List[int],
    device: str,
    n_epochs: int,
) -> Dict[str, float]:
    """Train DKO with sketched conformer ensemble."""
    from dko.models.dko import DKO
    from dko.training.trainer import Trainer

    metric = "rmse" if task == "regression" else "auroc"
    performances = []
    times = []

    for seed in seeds:
        # Set seed for reproducible sketching
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Create sketched dataloader
        sketched_train = create_sketched_dataloader(
            dataset.train_loader,
            sketch_size=sketch_size,
            method=method,
        )

        model = DKO(
            feature_dim=feature_dim,
            output_dim=1,
            task=task,
            use_second_order=True,
            verbose=False,
        )

        trainer = Trainer(
            model=model,
            config={
                "lr": 1e-4,
                "n_epochs": n_epochs,
                "seed": seed,
            },
            device=device,
        )

        start_time = time.time()
        trainer.fit(sketched_train, dataset.val_loader)
        train_time = time.time() - start_time

        test_metrics = trainer.evaluate(dataset.test_loader)
        performances.append(test_metrics[metric])
        times.append(train_time)

    return {
        f"{metric}_mean": float(np.mean(performances)),
        f"{metric}_std": float(np.std(performances)),
        "train_time_mean": float(np.mean(times)),
        "train_time_std": float(np.std(times)),
    }


def create_sketched_dataloader(
    dataloader,
    sketch_size: int,
    method: str = "random",
):
    """
    Create a dataloader with sketched conformer ensembles.

    Args:
        dataloader: Original dataloader
        sketch_size: Number of conformers to sample
        method: Sketching method

    Returns:
        Sketched dataloader
    """
    from torch.utils.data import DataLoader

    # Create sketched dataset
    sketched_dataset = SketchedDataset(
        dataloader.dataset,
        sketch_size=sketch_size,
        method=method,
    )

    return DataLoader(
        sketched_dataset,
        batch_size=dataloader.batch_size,
        shuffle=True,
        num_workers=getattr(dataloader, "num_workers", 0),
        collate_fn=getattr(dataloader, "collate_fn", None),
    )


class SketchedDataset:
    """Dataset wrapper that sketches conformer ensembles."""

    def __init__(
        self,
        dataset,
        sketch_size: int,
        method: str = "random",
    ):
        self.dataset = dataset
        self.sketch_size = sketch_size
        self.method = method

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Get features
        features = item.get("features", None)
        if features is None:
            return item

        # Apply sketching
        if isinstance(features, torch.Tensor):
            n_conf = features.shape[0]
            if n_conf <= self.sketch_size:
                return item

            indices = self._select_indices(features, item.get("energies", None))
            item["features"] = features[indices]

            # Also sketch energies and other conformer-level data
            if "energies" in item and item["energies"] is not None:
                item["energies"] = item["energies"][indices]
            if "mask" in item and item["mask"] is not None:
                item["mask"] = item["mask"][indices]

        return item

    def _select_indices(
        self,
        features: torch.Tensor,
        energies: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Select conformer indices based on method."""
        n_conf = features.shape[0]

        if self.method == "random":
            # Uniform random sampling
            indices = torch.randperm(n_conf)[:self.sketch_size]

        elif self.method == "energy_weighted":
            # Sample proportional to Boltzmann weights
            if energies is not None:
                kT = 0.001987204 * 300  # kB * T at 300K
                shifted = energies - energies.min()
                weights = torch.exp(-shifted / kT)
                weights = weights / weights.sum()
                indices = torch.multinomial(weights, self.sketch_size, replacement=False)
            else:
                indices = torch.randperm(n_conf)[:self.sketch_size]

        elif self.method == "diverse":
            # Greedy diverse selection (farthest point sampling)
            indices = self._farthest_point_sampling(features)

        elif self.method == "stratified":
            # Energy-stratified sampling
            if energies is not None:
                indices = self._stratified_energy_sampling(energies)
            else:
                indices = torch.randperm(n_conf)[:self.sketch_size]

        else:
            raise ValueError(f"Unknown sketching method: {self.method}")

        return indices

    def _farthest_point_sampling(self, features: torch.Tensor) -> torch.Tensor:
        """Select diverse conformers using farthest point sampling."""
        n_conf = features.shape[0]
        feat_flat = features.view(n_conf, -1)

        # Start with random point
        selected = [torch.randint(n_conf, (1,)).item()]
        distances = torch.full((n_conf,), float("inf"))

        for _ in range(self.sketch_size - 1):
            # Update distances to nearest selected point
            last = feat_flat[selected[-1]]
            dist_to_last = torch.norm(feat_flat - last, dim=1)
            distances = torch.minimum(distances, dist_to_last)

            # Select farthest point
            distances[selected] = -1  # Exclude already selected
            next_idx = distances.argmax().item()
            selected.append(next_idx)

        return torch.tensor(selected)

    def _stratified_energy_sampling(self, energies: torch.Tensor) -> torch.Tensor:
        """Sample from energy strata."""
        n_conf = len(energies)
        n_strata = min(self.sketch_size, n_conf)
        samples_per_stratum = self.sketch_size // n_strata

        # Sort by energy
        sorted_indices = torch.argsort(energies)

        # Create strata
        stratum_size = n_conf // n_strata
        selected = []

        for i in range(n_strata):
            start = i * stratum_size
            end = start + stratum_size if i < n_strata - 1 else n_conf
            stratum = sorted_indices[start:end]

            # Random sample from stratum
            n_sample = min(samples_per_stratum, len(stratum))
            perm = torch.randperm(len(stratum))[:n_sample]
            selected.extend(stratum[perm].tolist())

        # Pad if needed
        while len(selected) < self.sketch_size:
            idx = torch.randint(n_conf, (1,)).item()
            if idx not in selected:
                selected.append(idx)

        return torch.tensor(selected[:self.sketch_size])


def analyze_sketching_tradeoffs(
    results: SketchingResult,
    error_threshold: float = 5.0,  # Max acceptable relative error %
) -> Dict:
    """
    Analyze sketching tradeoffs to find optimal configuration.

    Args:
        results: SketchingResult from experiment
        error_threshold: Maximum acceptable performance degradation

    Returns:
        Analysis with recommendations
    """
    analysis = {
        "acceptable_configs": [],
        "optimal_config": None,
        "speedup_at_threshold": 0.0,
    }

    metric_key = "rmse_mean" if "rmse_mean" in list(results.results.get("full", {}).values())[0] else "auroc_mean"

    # Find configurations under error threshold
    for method, sizes in results.results.items():
        if method == "full":
            continue

        for size, metrics in sizes.items():
            error = abs(metrics.get("relative_error", 100))
            if error <= error_threshold:
                config = {
                    "method": method,
                    "size": size,
                    "error": error,
                    "speedup": metrics.get("speedup", 1),
                    "performance": metrics.get(metric_key, 0),
                }
                analysis["acceptable_configs"].append(config)

    # Find optimal (max speedup under threshold)
    if analysis["acceptable_configs"]:
        optimal = max(analysis["acceptable_configs"], key=lambda x: x["speedup"])
        analysis["optimal_config"] = optimal
        analysis["speedup_at_threshold"] = optimal["speedup"]

    return analysis
