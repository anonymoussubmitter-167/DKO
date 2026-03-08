#!/usr/bin/env python
"""
Main experiment runner for DKO.

This script provides a unified interface for running various
experiments including benchmarks, ablations, and analysis.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add project root to Python path if not installed
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dko.utils.config import Config, create_experiment_config
from dko.utils.logging_utils import setup_logging, get_logger
from dko.experiments.main_benchmark import run_main_benchmark
from dko.experiments.decomposition import run_decomposition_study
from dko.experiments.sample_efficiency import run_sample_efficiency_experiment
from dko.experiments.attention_analysis import run_attention_analysis
from dko.experiments.sketching import run_sketching_experiment
from dko.experiments.scc_validation import run_scc_validation
from dko.experiments.representation_vs_architecture import run_representation_vs_architecture_experiment
from dko.experiments.negative_controls import run_negative_control_experiment
from dko.experiments.decision_rule import run_decision_rule_experiment


logger = get_logger("experiment")


EXPERIMENTS = {
    "benchmark": run_main_benchmark,
    "decomposition": run_decomposition_study,
    "sample_efficiency": run_sample_efficiency_experiment,
    "attention": run_attention_analysis,
    "sketching": run_sketching_experiment,
    "scc_validation": run_scc_validation,
    "rep_vs_arch": run_representation_vs_architecture_experiment,
    "negative_controls": run_negative_control_experiment,
    "decision_rule": run_decision_rule_experiment,
}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run DKO experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run main benchmark on all datasets
    python run_experiment.py --experiment benchmark

    # Run benchmark on specific datasets
    python run_experiment.py --experiment benchmark --datasets bace pdbbind

    # Run with specific model
    python run_experiment.py --experiment benchmark --model dko --dataset bace

    # Run with hyperparameter optimization
    python run_experiment.py --experiment benchmark --dataset bace --hyperopt

    # Run decomposition study
    python run_experiment.py --experiment decomposition --dataset bace
        """,
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default="benchmark",
        choices=list(EXPERIMENTS.keys()),
        help="Experiment to run",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Single dataset to run on",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Multiple datasets to run on",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Single model to run",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Multiple models to run",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456],
        help="Random seeds",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Configuration file path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--hyperopt",
        action="store_true",
        help="Run hyperparameter optimization",
    )
    parser.add_argument(
        "--hyperopt-trials",
        type=int,
        default=50,
        help="Number of hyperopt trials",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )

    args = parser.parse_args()

    # Setup logging
    output_dir = Path(args.output_dir) / args.experiment
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "logs", name="experiment")

    logger.info(f"Running experiment: {args.experiment}")
    logger.info(f"Output directory: {output_dir}")

    # Load config
    if args.config:
        config = Config(base_config_path=args.config)
    else:
        config = Config()

    # Handle dataset(s) argument
    datasets = None
    if args.dataset:
        datasets = [args.dataset]
    elif args.datasets:
        datasets = args.datasets

    # Handle model(s) argument
    models = None
    if args.model:
        models = [args.model]
    elif args.models:
        models = args.models

    # Run experiment
    experiment_fn = EXPERIMENTS[args.experiment]

    try:
        if args.experiment == "benchmark":
            results = run_main_benchmark(
                datasets=datasets,
                models=models,
                seeds=args.seeds,
                config_path=args.config,
                output_dir=str(output_dir),
                device=args.device,
            )
        elif args.experiment == "decomposition":
            if not datasets or len(datasets) != 1:
                parser.error("Decomposition requires exactly one dataset (--dataset)")
            results = run_decomposition_study(
                dataset_name=datasets[0],
                config_path=args.config,
                output_dir=str(output_dir),
                device=args.device,
            )
        elif args.experiment == "sample_efficiency":
            if not datasets or len(datasets) != 1:
                parser.error("Sample efficiency requires exactly one dataset (--dataset)")
            results = run_sample_efficiency_experiment(
                dataset_name=datasets[0],
                models=models,
                seeds=args.seeds,
                config_path=args.config,
                output_dir=str(output_dir),
                device=args.device,
            )
        elif args.experiment == "attention":
            if not datasets or len(datasets) != 1:
                parser.error("Attention analysis requires exactly one dataset (--dataset)")
            results = run_attention_analysis(
                dataset_name=datasets[0],
                model_path="",  # Would need model path
                output_dir=str(output_dir),
                device=args.device,
            )
        elif args.experiment == "sketching":
            if not datasets or len(datasets) != 1:
                parser.error("Sketching requires exactly one dataset (--dataset)")
            results = run_sketching_experiment(
                dataset_name=datasets[0],
                seeds=args.seeds,
                config_path=args.config,
                output_dir=str(output_dir),
                device=args.device,
            )
        else:
            logger.error(f"Unknown experiment: {args.experiment}")
            sys.exit(1)

        logger.info("Experiment completed successfully!")
        logger.info(f"Results saved to: {output_dir}")

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        if args.debug:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
