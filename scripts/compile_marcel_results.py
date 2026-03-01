#!/usr/bin/env python3
"""Compile all MARCEL benchmark results into a summary table."""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def load_model_results(results_dir):
    """Load benchmark_summary.json files from a results directory."""
    results = {}
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        summary_path = model_dir / "benchmark_summary.json"
        if summary_path.exists():
            data = json.load(open(summary_path))
            for dataset, models in data.items():
                for model_name, metrics in models.items():
                    if metrics["mae"]["n"] >= 3:  # only complete results
                        results[(dataset, model_name)] = {
                            "mae": metrics["mae"]["mean"],
                            "mae_std": metrics["mae"]["std"],
                            "rmse": metrics["rmse"]["mean"],
                            "rmse_std": metrics["rmse"]["std"],
                            "r2": metrics["r2"]["mean"],
                        }
    return results


def load_fp_baseline(fp_path):
    """Load FP+XGBoost baseline results."""
    data = json.load(open(fp_path))
    by_ds = defaultdict(list)
    for r in data["results"]:
        by_ds[r["dataset"]].append(r)

    results = {}
    for ds, runs in by_ds.items():
        maes = [r["mae"] for r in runs]
        rmses = [r["rmse"] for r in runs]
        r2s = [r["r2"] for r in runs]
        results[(ds, "FP+XGBoost")] = {
            "mae": np.mean(maes),
            "mae_std": np.std(maes),
            "rmse": np.mean(rmses),
            "rmse_std": np.std(rmses),
            "r2": np.mean(r2s),
        }
    return results


def main():
    base = Path("results")

    # Collect all results
    all_results = {}

    # FP baseline
    fp_path = base / "marcel_benchmark" / "fp_baseline_all.json"
    if fp_path.exists():
        all_results.update(load_fp_baseline(fp_path))

    # Kraken DKO models
    kraken_dir = base / "kraken_benchmark"
    if kraken_dir.exists():
        all_results.update(load_model_results(kraken_dir))

    # BDE DKO models
    bde_dir = base / "marcel_benchmark" / "bde"
    if bde_dir.exists():
        all_results.update(load_model_results(bde_dir))

    # Drugs DKO models
    drugs_dir = base / "marcel_benchmark" / "drugs"
    if drugs_dir.exists():
        all_results.update(load_model_results(drugs_dir))

    # Print by dataset
    datasets = [
        "kraken_B5", "kraken_L", "kraken_burB5", "kraken_burL",
        "drugs_ip", "drugs_ea", "drugs_chi",
        "bde",
    ]
    model_order = [
        "FP+XGBoost", "dko_gated", "dko_invariants", "dko_eigenspectrum",
        "dko_residual", "dko_first_order", "attention", "mean_ensemble",
    ]

    print("=" * 100)
    print("MARCEL BENCHMARK RESULTS")
    print("=" * 100)

    for ds in datasets:
        models_with_results = [(m, all_results[(ds, m)])
                               for m in model_order
                               if (ds, m) in all_results]
        if not models_with_results:
            continue

        print(f"\n{ds}")
        print("-" * 80)
        print(f"{'Model':22s} {'MAE':>12s} {'RMSE':>12s} {'R²':>8s}")
        print("-" * 80)

        best_mae = min(r["mae"] for _, r in models_with_results)
        for model, r in models_with_results:
            marker = " *" if r["mae"] == best_mae else ""
            print(f"{model:22s} {r['mae']:7.4f}±{r['mae_std']:6.4f} "
                  f"{r['rmse']:7.4f}±{r['rmse_std']:6.4f} "
                  f"{r['r2']:7.4f}{marker}")

    # Summary table (MAE only, for paper)
    print("\n\n" + "=" * 100)
    print("SUMMARY TABLE (MAE)")
    print("=" * 100)

    # Header
    active_models = []
    for m in model_order:
        if any((ds, m) in all_results for ds in datasets):
            active_models.append(m)

    header = f"{'Dataset':15s}" + "".join(f" {m:>12s}" for m in active_models)
    print(header)
    print("-" * len(header))

    for ds in datasets:
        row = f"{ds:15s}"
        for m in active_models:
            key = (ds, m)
            if key in all_results:
                row += f" {all_results[key]['mae']:12.4f}"
            else:
                row += f" {'--':>12s}"
        print(row)

    # Save as JSON
    output = {
        "datasets": datasets,
        "models": model_order,
        "results": {f"{ds}_{m}": r for (ds, m), r in all_results.items()},
    }
    out_path = base / "marcel_benchmark" / "compiled_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
