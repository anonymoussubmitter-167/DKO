#!/usr/bin/env python3
"""
Analyze 10-seed validation results and compute statistical significance.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats


def load_results(output_dir: Path) -> dict:
    """Load all benchmark results from output directory."""
    results = defaultdict(lambda: defaultdict(list))

    for model_dir in output_dir.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        for dataset_dir in model_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            dataset_name = dataset_dir.name

            # Find benchmark_results.json
            results_file = dataset_dir / "benchmark_results.json"
            if results_file.exists():
                with open(results_file) as f:
                    data = json.load(f)

                # Extract RMSE from each seed
                for entry in data.get(dataset_name, {}).get(model_name, []):
                    rmse = entry.get("test_metrics", {}).get("rmse")
                    if rmse is not None:
                        results[dataset_name][model_name].append(rmse)

    return dict(results)


def compute_statistics(results: dict) -> dict:
    """Compute mean, std, and pairwise t-tests."""
    stats_out = {}

    for dataset, models in results.items():
        stats_out[dataset] = {}

        # Basic stats for each model
        for model, rmses in models.items():
            rmses = np.array(rmses)
            stats_out[dataset][model] = {
                "rmse_mean": float(np.mean(rmses)),
                "rmse_std": float(np.std(rmses)),
                "rmse_sem": float(np.std(rmses) / np.sqrt(len(rmses))),
                "n_seeds": len(rmses),
                "rmse_values": rmses.tolist(),
            }

        # T-tests vs attention baseline
        if "attention" in models:
            attention_rmses = np.array(models["attention"])

            for model in ["dko_gated", "dko_invariants"]:
                if model not in models:
                    continue
                model_rmses = np.array(models[model])

                # Welch's t-test (unequal variance)
                t_stat, p_value = stats.ttest_ind(model_rmses, attention_rmses, equal_var=False)

                # One-sided p-value (is model better than attention?)
                p_one_sided = p_value / 2 if t_stat < 0 else 1 - p_value / 2

                stats_out[dataset][f"{model}_vs_attention"] = {
                    "t_statistic": float(t_stat),
                    "p_value_two_sided": float(p_value),
                    "p_value_one_sided": float(p_one_sided),
                    "significant_0.05": p_one_sided < 0.05,
                    "significant_0.01": p_one_sided < 0.01,
                    "model_mean": float(np.mean(model_rmses)),
                    "attention_mean": float(np.mean(attention_rmses)),
                    "improvement": float((np.mean(attention_rmses) - np.mean(model_rmses)) / np.mean(attention_rmses) * 100),
                }

    return stats_out


def print_summary(stats_out: dict):
    """Print formatted summary."""
    print("\n" + "=" * 80)
    print("10-SEED STATISTICAL VALIDATION SUMMARY")
    print("=" * 80)

    for dataset in ["esol", "lipophilicity"]:
        if dataset not in stats_out:
            continue

        print(f"\n{dataset.upper()}")
        print("-" * 40)

        # Print model results
        for model in ["attention", "dko_gated", "dko_invariants"]:
            if model in stats_out[dataset]:
                s = stats_out[dataset][model]
                print(f"  {model:20s}: {s['rmse_mean']:.4f} ± {s['rmse_std']:.4f} (n={s['n_seeds']})")

        # Print t-test results
        print()
        for model in ["dko_gated", "dko_invariants"]:
            key = f"{model}_vs_attention"
            if key in stats_out[dataset]:
                t = stats_out[dataset][key]
                sig_stars = "***" if t["significant_0.01"] else ("*" if t["significant_0.05"] else "")
                direction = "BETTER" if t["improvement"] > 0 else "WORSE"
                print(f"  {model} vs attention:")
                print(f"    p-value (one-sided): {t['p_value_one_sided']:.4f} {sig_stars}")
                print(f"    Improvement: {t['improvement']:+.2f}% ({direction})")

    print("\n" + "=" * 80)
    print("Significance: * p<0.05, *** p<0.01 (one-sided t-test)")
    print("=" * 80)


def main():
    output_dir = Path("results/10seed_validation")

    if not output_dir.exists():
        print(f"Error: {output_dir} not found")
        sys.exit(1)

    print(f"Loading results from {output_dir}...")
    results = load_results(output_dir)

    if not results:
        print("No results found!")
        sys.exit(1)

    print(f"Found results for datasets: {list(results.keys())}")
    for dataset, models in results.items():
        print(f"  {dataset}: {[(m, len(v)) for m, v in models.items()]}")

    stats_out = compute_statistics(results)
    print_summary(stats_out)

    # Save to JSON
    output_file = output_dir / "statistical_analysis.json"
    with open(output_file, "w") as f:
        json.dump(stats_out, f, indent=2)
    print(f"\nFull results saved to {output_file}")


if __name__ == "__main__":
    main()
