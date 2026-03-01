#!/usr/bin/env python
"""
Submit All Experiments Script for DKO.

Submits a comprehensive set of experiments across datasets, models, and seeds.
Supports both SLURM cluster and local execution.

Usage:
    python scripts/submit_all_experiments.py --dry-run
    python scripts/submit_all_experiments.py --datasets esol freesolv
    python scripts/submit_all_experiments.py --seeds 42 123 456 --submit
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from itertools import product

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# Default experiment configurations
DEFAULT_DATASETS = ["esol", "freesolv", "lipophilicity"]
DEFAULT_MODELS = ["dko_first_order", "dko_second_order"]
DEFAULT_SEEDS = [42, 123, 456]


class ExperimentSubmitter:
    """Submit DKO experiments to SLURM or run locally."""

    def __init__(
        self,
        experiments_dir: str = "experiments",
        configs_dir: str = "configs/experiments",
    ):
        self.experiments_dir = Path(experiments_dir)
        self.configs_dir = Path(configs_dir)
        self.slurm_available = self._check_slurm()

        # Create directories
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir.mkdir(parents=True, exist_ok=True)

    def _check_slurm(self) -> bool:
        """Check if SLURM is available."""
        try:
            subprocess.run(['squeue', '--version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def generate_experiment_configs(
        self,
        datasets: List[str],
        models: List[str],
        seeds: List[int],
        base_config: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Generate all experiment configurations."""
        experiments = []

        base_config = base_config or {}

        for dataset, model, seed in product(datasets, models, seeds):
            exp_name = f"{model}_{dataset}_seed{seed}"

            config = {
                "experiment_name": exp_name,
                "dataset": {
                    "name": dataset,
                    "n_conformers": 20,
                },
                "model": {
                    "type": model,
                    "kernel_hidden_dims": [128, 64],
                    "kernel_output_dim": 32,
                },
                "training": {
                    "max_epochs": 200,
                    "batch_size": 32,
                    "learning_rate": 1e-3,
                    "weight_decay": 1e-4,
                    "patience": 20,
                    "seed": seed,
                },
            }

            # Merge with base config
            for section, values in base_config.items():
                if section in config and isinstance(config[section], dict):
                    config[section].update(values)
                else:
                    config[section] = values

            experiments.append({
                "name": exp_name,
                "dataset": dataset,
                "model": model,
                "seed": seed,
                "config": config,
            })

        return experiments

    def save_configs(self, experiments: List[Dict]) -> List[Path]:
        """Save experiment configurations to files."""
        config_paths = []

        for exp in experiments:
            config_path = self.configs_dir / f"{exp['name']}.yaml"

            if YAML_AVAILABLE:
                with open(config_path, 'w') as f:
                    yaml.dump(exp['config'], f, default_flow_style=False)
            else:
                # Fallback to JSON
                config_path = config_path.with_suffix('.json')
                with open(config_path, 'w') as f:
                    json.dump(exp['config'], f, indent=2)

            config_paths.append(config_path)

        return config_paths

    def submit_slurm(
        self,
        experiments: List[Dict],
        config_paths: List[Path],
        script: str = "scripts/submit_hpc.sh",
        partition: Optional[str] = None,
        time_limit: Optional[str] = None,
        dry_run: bool = False,
    ) -> List[str]:
        """Submit experiments to SLURM."""
        job_ids = []

        for exp, config_path in zip(experiments, config_paths):
            cmd = ["sbatch"]

            if partition:
                cmd.extend(["--partition", partition])
            if time_limit:
                cmd.extend(["--time", time_limit])

            cmd.extend([script, str(config_path), exp['name']])

            if dry_run:
                print(f"[DRY RUN] {' '.join(cmd)}")
                job_ids.append("DRY_RUN")
            else:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    job_id = result.stdout.strip().split()[-1]
                    job_ids.append(job_id)
                    print(f"Submitted {exp['name']} as job {job_id}")
                except subprocess.CalledProcessError as e:
                    print(f"Failed to submit {exp['name']}: {e.stderr}")
                    job_ids.append(None)

        return job_ids

    def submit_array_job(
        self,
        experiments: List[Dict],
        config_paths: List[Path],
        script: str = "scripts/slurm_submit_batch.sh",
        max_concurrent: int = 10,
        dry_run: bool = False,
    ) -> Optional[str]:
        """Submit experiments as a SLURM array job."""
        n_experiments = len(experiments)

        # Create a manifest file for the array job
        manifest_path = self.configs_dir / "experiment_manifest.json"
        manifest = [
            {"index": i, "name": exp["name"], "config": str(path)}
            for i, (exp, path) in enumerate(zip(experiments, config_paths))
        ]

        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        # Build array job command
        array_spec = f"0-{n_experiments - 1}"
        if max_concurrent > 0:
            array_spec += f"%{max_concurrent}"

        cmd = [
            "sbatch",
            f"--array={array_spec}",
            script,
            str(self.configs_dir),
        ]

        if dry_run:
            print(f"[DRY RUN] {' '.join(cmd)}")
            print(f"[DRY RUN] Would submit {n_experiments} experiments")
            return "DRY_RUN"

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            job_id = result.stdout.strip().split()[-1]
            print(f"Submitted array job {job_id} with {n_experiments} tasks")
            return job_id
        except subprocess.CalledProcessError as e:
            print(f"Failed to submit array job: {e.stderr}")
            return None

    def run_local(
        self,
        experiments: List[Dict],
        config_paths: List[Path],
        sequential: bool = True,
        dry_run: bool = False,
    ) -> List[int]:
        """Run experiments locally."""
        exit_codes = []

        for exp, config_path in zip(experiments, config_paths):
            cmd = [
                sys.executable,
                "-m", "dko.scripts.train",
                "--config", str(config_path),
                "--experiment-name", exp["name"],
            ]

            if dry_run:
                print(f"[DRY RUN] {' '.join(cmd)}")
                exit_codes.append(0)
                continue

            print(f"\nRunning {exp['name']}...")

            try:
                result = subprocess.run(cmd)
                exit_codes.append(result.returncode)

                if result.returncode != 0:
                    print(f"Experiment {exp['name']} failed with code {result.returncode}")
                    if sequential:
                        response = input("Continue with remaining experiments? [y/N]: ")
                        if response.lower() != 'y':
                            break
            except KeyboardInterrupt:
                print("\nInterrupted by user")
                break

        return exit_codes

    def check_existing(self, experiments: List[Dict]) -> Dict[str, List[Dict]]:
        """Check which experiments already exist."""
        existing = []
        new = []

        for exp in experiments:
            exp_dir = self.experiments_dir / exp['name']
            results_file = exp_dir / "results.json"

            if results_file.exists():
                existing.append(exp)
            else:
                new.append(exp)

        return {"existing": existing, "new": new}


def main():
    parser = argparse.ArgumentParser(
        description="Submit all DKO experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate and view experiment configs (dry run)
    python scripts/submit_all_experiments.py --dry-run

    # Submit specific datasets
    python scripts/submit_all_experiments.py --datasets esol freesolv --submit

    # Submit with custom seeds
    python scripts/submit_all_experiments.py --seeds 42 123 456 789 --submit

    # Use array job submission
    python scripts/submit_all_experiments.py --array --max-concurrent 5 --submit

    # Run locally instead of SLURM
    python scripts/submit_all_experiments.py --local

    # Skip existing experiments
    python scripts/submit_all_experiments.py --skip-existing --submit
        """
    )

    # Experiment selection
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help=f"Datasets to run (default: {DEFAULT_DATASETS})"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"Models to run (default: {DEFAULT_MODELS})"
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help=f"Random seeds (default: {DEFAULT_SEEDS})"
    )

    # Submission options
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually submit jobs (default: dry run)"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run locally instead of SLURM"
    )
    parser.add_argument(
        "--array",
        action="store_true",
        help="Submit as SLURM array job"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Max concurrent array tasks"
    )

    # SLURM options
    parser.add_argument(
        "--partition",
        type=str,
        help="SLURM partition"
    )
    parser.add_argument(
        "--time",
        type=str,
        help="SLURM time limit (e.g., 24:00:00)"
    )

    # Other options
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip experiments that already have results"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without doing it"
    )
    parser.add_argument(
        "--base-config",
        type=str,
        help="Base config file to inherit from"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments",
        help="Output directory for experiments"
    )

    args = parser.parse_args()

    # If neither --submit nor --local, default to dry run
    if not args.submit and not args.local:
        args.dry_run = True

    print(f"\n{'='*60}")
    print("DKO Experiment Submission")
    print(f"{'='*60}")
    print(f"Datasets: {args.datasets}")
    print(f"Models: {args.models}")
    print(f"Seeds: {args.seeds}")
    print(f"Total experiments: {len(args.datasets) * len(args.models) * len(args.seeds)}")
    print(f"{'='*60}\n")

    # Create submitter
    submitter = ExperimentSubmitter(experiments_dir=args.output_dir)

    # Load base config if provided
    base_config = None
    if args.base_config:
        with open(args.base_config) as f:
            if args.base_config.endswith('.yaml') or args.base_config.endswith('.yml'):
                base_config = yaml.safe_load(f)
            else:
                base_config = json.load(f)

    # Generate experiments
    experiments = submitter.generate_experiment_configs(
        datasets=args.datasets,
        models=args.models,
        seeds=args.seeds,
        base_config=base_config,
    )

    # Check for existing
    if args.skip_existing:
        check = submitter.check_existing(experiments)
        if check["existing"]:
            print(f"Skipping {len(check['existing'])} existing experiments")
            experiments = check["new"]

    if not experiments:
        print("No experiments to submit")
        return

    # Save configs
    print(f"Generating {len(experiments)} experiment configs...")
    config_paths = submitter.save_configs(experiments)

    # Submit or run
    if args.local:
        print("\nRunning experiments locally...")
        exit_codes = submitter.run_local(
            experiments, config_paths,
            dry_run=args.dry_run
        )
        success = sum(1 for c in exit_codes if c == 0)
        print(f"\nCompleted: {success}/{len(experiments)} experiments succeeded")

    elif args.array and submitter.slurm_available:
        print("\nSubmitting as SLURM array job...")
        job_id = submitter.submit_array_job(
            experiments, config_paths,
            max_concurrent=args.max_concurrent,
            dry_run=args.dry_run
        )
        if job_id:
            print(f"\nMonitor with: squeue -j {job_id}")

    elif submitter.slurm_available:
        print("\nSubmitting individual SLURM jobs...")
        job_ids = submitter.submit_slurm(
            experiments, config_paths,
            partition=args.partition,
            time_limit=args.time,
            dry_run=args.dry_run
        )
        submitted = sum(1 for j in job_ids if j is not None)
        print(f"\nSubmitted {submitted}/{len(experiments)} jobs")

    else:
        print("\nSLURM not available. Use --local to run locally.")
        if args.dry_run:
            print("\nGenerated configs saved to:", submitter.configs_dir)


if __name__ == "__main__":
    main()
