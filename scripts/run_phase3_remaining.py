#!/usr/bin/env python3
"""
Run Phase 3 variants on remaining datasets (FreeSolv, QM9-HOMO, BACE, BBBP).

This completes the picture across all 8 datasets.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Phase 3 models to run
MODELS = [
    "dko_eigenspectrum",
    "dko_invariants",
    "dko_lowrank",
    "dko_residual",
    "dko_crossattn",
    "dko_gated",
    "dko_router",
]

# Remaining datasets (not in Phase 3 yet)
REMAINING_DATASETS = ["freesolv", "qm9_homo", "bace", "bbbp"]

SEEDS = [42, 123, 456]


def run_benchmark(model: str, dataset: str, seed: int, gpu: int, output_dir: Path) -> dict:
    """Run single benchmark experiment."""
    cmd = [
        "python", "-m", "dko.experiments.main_benchmark",
        "--model", model,
        "--dataset", dataset,
        "--seed", str(seed),
        "--epochs", "200",
        "--patience", "30",
        "--batch-size", "32",
        "--lr", "1e-3",
        "--device", f"cuda:{gpu}",
        "--output-dir", str(output_dir),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
    )

    return {
        "returncode": result.returncode,
        "stdout": result.stdout[-2000:] if result.stdout else "",
        "stderr": result.stderr[-2000:] if result.stderr else "",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--datasets", nargs="+", default=REMAINING_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--output-dir", type=str, default="results/phase3_remaining")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(args.models) * len(args.datasets) * len(args.seeds)
    print(f"Phase 3 Remaining Datasets")
    print(f"GPU: {args.gpu}")
    print(f"Models: {args.models}")
    print(f"Datasets: {args.datasets}")
    print(f"Seeds: {args.seeds}")
    print(f"Total experiments: {total}")
    print("=" * 70)

    completed = 0
    failed = []

    for dataset in args.datasets:
        for model in args.models:
            for seed in args.seeds:
                completed += 1
                print(f"\n[{completed}/{total}] {dataset} / {model} / seed={seed}")

                result = run_benchmark(model, dataset, seed, args.gpu, output_dir / model)

                if result["returncode"] != 0:
                    print(f"  FAILED: {result['stderr'][-500:]}")
                    failed.append(f"{dataset}/{model}/seed{seed}")
                else:
                    # Try to extract RMSE from output
                    if "test_rmse" in result["stdout"]:
                        import re
                        match = re.search(r"test_rmse['\"]?\s*[:=]\s*([\d.]+)", result["stdout"])
                        if match:
                            print(f"  RMSE: {match.group(1)}")
                        else:
                            print("  Done")
                    else:
                        print("  Done")

    print("\n" + "=" * 70)
    print(f"Completed: {completed - len(failed)}/{total}")
    if failed:
        print(f"Failed: {failed}")

    # Save completion status
    status = {
        "timestamp": datetime.now().isoformat(),
        "total": total,
        "completed": completed - len(failed),
        "failed": failed,
    }
    with open(output_dir / "completion_status.json", "w") as f:
        json.dump(status, f, indent=2)


if __name__ == "__main__":
    main()
