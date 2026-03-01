#!/usr/bin/env python3
"""
10-seed statistical validation for key model comparisons.

Tests whether dko_gated wins on ESOL and dko_invariants wins on Lipophilicity
are statistically significant vs attention baseline.

Seeds: 42, 123, 456, 789, 1000, 1111, 2222, 3333, 4444, 5555
Models: dko_gated, dko_invariants, attention
Datasets: esol, lipophilicity
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dko.data.conformer_dataset import ConformerDataset
from dko.models import (
    DKOGatedFusion,
    DKOScalarInvariants,
    AttentionAggregation,
)
from dko.training.trainer import Trainer
from dko.training.evaluator import Evaluator


SEEDS = [42, 123, 456, 789, 1000, 1111, 2222, 3333, 4444, 5555]
MODELS = ["dko_gated", "dko_invariants", "attention"]
DATASETS = ["esol", "lipophilicity"]

MODEL_CLASSES = {
    "dko_gated": DKOGatedFusion,
    "dko_invariants": DKOScalarInvariants,
    "attention": AttentionAggregation,
}


def get_model(model_name: str, input_dim: int, **kwargs):
    """Instantiate model by name."""
    cls = MODEL_CLASSES[model_name]

    if model_name == "attention":
        return cls(feature_dim=input_dim, num_heads=4, num_outputs=1)
    elif model_name == "dko_gated":
        return cls(feature_dim=input_dim, hidden_dim=256, output_dim=1, k=10)
    elif model_name == "dko_invariants":
        return cls(feature_dim=input_dim, output_dim=1)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_experiment(model_name: str, dataset_name: str, seed: int, device: str, output_dir: Path):
    """Run single experiment."""
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    data_dir = Path(f"data/conformers/{dataset_name}")
    train_dataset = ConformerDataset(data_dir / "train.pkl")
    val_dataset = ConformerDataset(data_dir / "val.pkl")
    test_dataset = ConformerDataset(data_dir / "test.pkl")

    input_dim = train_dataset[0]["conformer_features"].shape[-1]

    # Create model
    model = get_model(model_name, input_dim)

    # Training config
    config = {
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 32,
        "epochs": 200,
        "patience": 30,
        "device": device,
        "checkpoint_dir": str(output_dir / "checkpoints" / f"{dataset_name}_{model_name}_seed{seed}"),
    }

    # Determine if model uses sigma
    uses_sigma = model_name.startswith("dko_")

    # Train
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        use_sigma=uses_sigma,
    )

    train_results = trainer.train()

    # Evaluate
    evaluator = Evaluator(model=model, device=device, use_sigma=uses_sigma)
    test_metrics = evaluator.evaluate(test_dataset)

    return {
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "test_metrics": test_metrics,
        "best_epoch": train_results.get("best_epoch", -1),
        "best_val_loss": train_results.get("best_val_loss", -1),
    }


def compute_statistics(results: list) -> dict:
    """Compute mean, std, and perform t-test."""
    from scipy import stats

    # Group by model and dataset
    grouped = {}
    for r in results:
        key = (r["dataset"], r["model"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r["test_metrics"]["rmse"])

    stats_results = {}
    for (dataset, model), rmses in grouped.items():
        rmses = np.array(rmses)
        stats_results[f"{dataset}_{model}"] = {
            "rmse_mean": float(np.mean(rmses)),
            "rmse_std": float(np.std(rmses)),
            "rmse_values": rmses.tolist(),
            "n_seeds": len(rmses),
        }

    # Pairwise t-tests vs attention
    for dataset in DATASETS:
        attention_key = f"{dataset}_attention"
        if attention_key not in stats_results:
            continue
        attention_rmses = stats_results[attention_key]["rmse_values"]

        for model in ["dko_gated", "dko_invariants"]:
            model_key = f"{dataset}_{model}"
            if model_key not in stats_results:
                continue
            model_rmses = stats_results[model_key]["rmse_values"]

            # Welch's t-test (unequal variance)
            t_stat, p_value = stats.ttest_ind(model_rmses, attention_rmses, equal_var=False)

            stats_results[f"{dataset}_{model}_vs_attention"] = {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant_0.05": p_value < 0.05,
                "significant_0.01": p_value < 0.01,
                "model_better": np.mean(model_rmses) < np.mean(attention_rmses),
            }

    return stats_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--output-dir", type=str, default="results/10seed_validation")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"10-Seed Statistical Validation")
    print(f"Device: {device}")
    print(f"Models: {args.models}")
    print(f"Datasets: {args.datasets}")
    print(f"Seeds: {args.seeds}")
    print(f"Total experiments: {len(args.models) * len(args.datasets) * len(args.seeds)}")
    print("=" * 70)

    results = []

    for dataset in args.datasets:
        for model in args.models:
            for seed in args.seeds:
                print(f"\n>>> {dataset} / {model} / seed={seed}")
                try:
                    result = run_experiment(model, dataset, seed, device, output_dir)
                    results.append(result)
                    print(f"    RMSE: {result['test_metrics']['rmse']:.4f}")

                    # Save incrementally
                    with open(output_dir / "results_partial.json", "w") as f:
                        json.dump(results, f, indent=2)

                except Exception as e:
                    print(f"    ERROR: {e}")
                    import traceback
                    traceback.print_exc()

    # Compute statistics
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    stats_results = compute_statistics(results)

    # Print summary
    for dataset in args.datasets:
        print(f"\n{dataset.upper()}:")
        for model in args.models:
            key = f"{dataset}_{model}"
            if key in stats_results:
                s = stats_results[key]
                print(f"  {model:20s}: {s['rmse_mean']:.4f} +/- {s['rmse_std']:.4f} (n={s['n_seeds']})")

        # Print t-test results
        for model in ["dko_gated", "dko_invariants"]:
            ttest_key = f"{dataset}_{model}_vs_attention"
            if ttest_key in stats_results:
                t = stats_results[ttest_key]
                sig = "***" if t["significant_0.01"] else ("**" if t["significant_0.05"] else "")
                better = "BETTER" if t["model_better"] else "WORSE"
                print(f"  {model} vs attention: p={t['p_value']:.4f} {sig} ({better})")

    # Save final results
    final_output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "models": args.models,
            "datasets": args.datasets,
            "seeds": args.seeds,
        },
        "raw_results": results,
        "statistics": stats_results,
    }

    with open(output_dir / "validation_results.json", "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"\nResults saved to {output_dir / 'validation_results.json'}")


if __name__ == "__main__":
    main()
