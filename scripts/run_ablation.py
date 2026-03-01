#!/usr/bin/env python3
"""
Ablation Study (Experiment B).

Isolates which fix matters most by toggling each independently.

Configs:
  0. baseline      - Current DKO (no changes): lr=1e-5, k_dim=32, norm=dim(1,2), sigma_reg=1e-4, no mixed precision
  1. +LR           - Only fix learning rate to 1e-4
  2. +k_dim        - Only fix kernel_output_dim to 64
  3. +norm         - Only fix normalization to dim=1 + sigma_reg=1e-2
  4. +LR+k_dim     - Fix LR and kernel_dim together
  5. +LR+k_dim+norm - Fix LR, kernel_dim, and normalization
  6. all_fixes     - All fixes including mixed precision

The script patches the config per-run so we don't need to modify source files.
Instead it calls a single-experiment runner with config overrides.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ablation configurations
# Each config specifies overrides relative to a "baseline" state
ABLATION_CONFIGS = {
    "baseline": {
        "description": "Current DKO settings (no fixes)",
        "lr": 1e-5,
        "kernel_output_dim": 32,
        "norm_dim": "1_2",  # dim=(1,2) - broken normalization
        "sigma_reg": 1e-4,
        "mixed_precision": False,
    },
    "fix_lr": {
        "description": "+LR only (1e-4)",
        "lr": 1e-4,
        "kernel_output_dim": 32,
        "norm_dim": "1_2",
        "sigma_reg": 1e-4,
        "mixed_precision": False,
    },
    "fix_kdim": {
        "description": "+kernel_dim only (64)",
        "lr": 1e-5,
        "kernel_output_dim": 64,
        "norm_dim": "1_2",
        "sigma_reg": 1e-4,
        "mixed_precision": False,
    },
    "fix_norm": {
        "description": "+norm fix only (dim=1, sigma_reg=1e-2)",
        "lr": 1e-5,
        "kernel_output_dim": 32,
        "norm_dim": "1",  # dim=1 - correct normalization
        "sigma_reg": 1e-2,
        "mixed_precision": False,
    },
    "fix_lr_kdim": {
        "description": "+LR + kernel_dim",
        "lr": 1e-4,
        "kernel_output_dim": 64,
        "norm_dim": "1_2",
        "sigma_reg": 1e-4,
        "mixed_precision": False,
    },
    "fix_lr_kdim_norm": {
        "description": "+LR + kernel_dim + norm",
        "lr": 1e-4,
        "kernel_output_dim": 64,
        "norm_dim": "1",
        "sigma_reg": 1e-2,
        "mixed_precision": False,
    },
    "all_fixes": {
        "description": "All fixes (LR + kernel_dim + norm + mixed precision)",
        "lr": 1e-4,
        "kernel_output_dim": 64,
        "norm_dim": "1",
        "sigma_reg": 1e-2,
        "mixed_precision": True,
    },
}


def run_ablation_experiment(
    config_name,
    config,
    dataset,
    seed,
    gpu_id,
    output_dir,
    model="dko",
):
    """Run a single ablation experiment by calling the ablation runner."""
    experiment_name = f"ablation_{config_name}_{dataset}_seed{seed}"
    exp_output_dir = Path(output_dir) / config_name

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "run_ablation_single.py"),
        "--dataset", dataset,
        "--model", model,
        "--seed", str(seed),
        "--lr", str(config["lr"]),
        "--kernel-output-dim", str(config["kernel_output_dim"]),
        "--norm-dim", config["norm_dim"],
        "--sigma-reg", str(config["sigma_reg"]),
        "--mixed-precision", str(config["mixed_precision"]),
        "--output-dir", str(exp_output_dir),
        "--experiment-name", experiment_name,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[GPU {gpu_id}] Running {experiment_name}: {config['description']}")
    return subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def main():
    parser = argparse.ArgumentParser(description="Run DKO Ablation Study")
    parser.add_argument(
        "--dataset", type=str, default="esol", help="Dataset for ablation"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 123, 456],
        help="Random seeds"
    )
    parser.add_argument(
        "--configs", nargs="+", default=None,
        help="Specific configs to run (default: all)"
    )
    parser.add_argument(
        "--gpu-ids", nargs="+", type=int, default=[3, 4, 5, 6, 7, 8, 9],
        help="GPU IDs to use"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/ablation",
        help="Output directory"
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Run sequentially instead of parallel"
    )
    args = parser.parse_args()

    configs_to_run = args.configs or list(ABLATION_CONFIGS.keys())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save ablation config
    with open(output_dir / "ablation_configs.json", "w") as f:
        json.dump(ABLATION_CONFIGS, f, indent=2, default=str)

    if args.sequential:
        # Run one at a time
        for config_name in configs_to_run:
            config = ABLATION_CONFIGS[config_name]
            for seed in args.seeds:
                proc = run_ablation_experiment(
                    config_name, config, args.dataset, seed,
                    gpu_id=args.gpu_ids[0],
                    output_dir=str(output_dir),
                )
                stdout, stderr = proc.communicate()
                if proc.returncode != 0:
                    print(f"FAILED: {config_name}/seed{seed}")
                    print(stderr.decode())
                else:
                    print(f"DONE: {config_name}/seed{seed}")
    else:
        # Parallel: assign configs to GPUs
        processes = []
        gpu_idx = 0

        for config_name in configs_to_run:
            config = ABLATION_CONFIGS[config_name]
            for seed in args.seeds:
                gpu_id = args.gpu_ids[gpu_idx % len(args.gpu_ids)]
                proc = run_ablation_experiment(
                    config_name, config, args.dataset, seed,
                    gpu_id=gpu_id,
                    output_dir=str(output_dir),
                )
                processes.append((config_name, seed, gpu_id, proc))
                gpu_idx += 1

        # Wait for all
        print(f"\nWaiting for {len(processes)} experiments...")
        for config_name, seed, gpu_id, proc in processes:
            stdout, stderr = proc.communicate()
            status = "OK" if proc.returncode == 0 else "FAILED"
            print(f"  [{status}] {config_name}/seed{seed} (GPU {gpu_id})")
            if proc.returncode != 0:
                err_text = stderr.decode()[-500:]
                print(f"    Error: {err_text}")

    # Aggregate results
    print(f"\nAggregating results from {output_dir}...")
    aggregate_ablation_results(output_dir, configs_to_run)


def aggregate_ablation_results(output_dir, configs):
    """Aggregate results across all ablation configs."""
    output_dir = Path(output_dir)
    summary = {}

    for config_name in configs:
        config_dir = output_dir / config_name
        results_files = list(config_dir.glob("**/results.json"))

        if not results_files:
            print(f"  No results for {config_name}")
            continue

        metrics_per_seed = {}
        for rf in results_files:
            with open(rf) as f:
                result = json.load(f)
            test_metrics = result.get("test_metrics", {})
            for metric, value in test_metrics.items():
                if metric not in metrics_per_seed:
                    metrics_per_seed[metric] = []
                metrics_per_seed[metric].append(value)

        config_summary = {}
        for metric, values in metrics_per_seed.items():
            import numpy as np
            config_summary[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
            }

        summary[config_name] = {
            "description": ABLATION_CONFIGS[config_name]["description"],
            "metrics": config_summary,
            "n_seeds": len(results_files),
        }

    # Save summary
    summary_path = output_dir / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print comparison table
    print(f"\n{'Config':<25} {'RMSE':>10} {'MAE':>10} {'R²':>10} {'Pearson':>10}")
    print("-" * 70)
    for config_name, data in summary.items():
        metrics = data.get("metrics", {})
        rmse = metrics.get("rmse", {}).get("mean", float("nan"))
        mae = metrics.get("mae", {}).get("mean", float("nan"))
        r2 = metrics.get("r2", {}).get("mean", float("nan"))
        pearson = metrics.get("pearson", {}).get("mean", float("nan"))
        print(f"  {config_name:<23} {rmse:>10.4f} {mae:>10.4f} {r2:>10.4f} {pearson:>10.4f}")

    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
