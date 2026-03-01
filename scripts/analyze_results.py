#!/usr/bin/env python
"""
Results analysis script for DKO experiments.

This script analyzes experiment results and generates
visualizations and summary tables.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to Python path if not installed
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dko.utils.logging_utils import setup_logging, get_logger
from dko.analysis.scc import StatisticalConsistencyChecker, validate_scc
from dko.analysis.statistics import (
    compute_confidence_intervals,
    perform_significance_tests,
    compute_effect_size,
)
from dko.analysis.visualization import (
    plot_learning_curves,
    plot_performance_comparison,
    create_summary_table,
)


logger = get_logger("analyze")


def load_results(results_dir: Path) -> Dict:
    """Load experiment results from directory."""
    results_path = results_dir / "benchmark_results.json"

    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        return {}

    with open(results_path, "r") as f:
        return json.load(f)


def analyze_benchmark_results(
    results: Dict,
    output_dir: Path,
    baseline: str = "single_conformer",
) -> Dict:
    """
    Analyze benchmark results.

    Args:
        results: Raw results dictionary
        output_dir: Output directory for analysis
        baseline: Baseline model name

    Returns:
        Analysis summary
    """
    analysis = {}
    scc = StatisticalConsistencyChecker()

    for dataset, model_results in results.items():
        logger.info(f"Analyzing dataset: {dataset}")
        analysis[dataset] = {}

        baseline_values = []
        if baseline in model_results:
            for seed_result in model_results[baseline]:
                if "error" not in seed_result:
                    metric = seed_result.get("test_metrics", {})
                    if "rmse" in metric:
                        baseline_values.append(metric["rmse"])
                    elif "auc" in metric:
                        baseline_values.append(metric["auc"])

        for model_name, seed_results in model_results.items():
            if model_name == baseline:
                continue

            model_values = []
            for seed_result in seed_results:
                if "error" not in seed_result:
                    metric = seed_result.get("test_metrics", {})
                    if "rmse" in metric:
                        model_values.append(metric["rmse"])
                    elif "auc" in metric:
                        model_values.append(metric["auc"])

            if model_values and baseline_values:
                # Statistical comparison
                sig_result = scc.check_significance(baseline_values, model_values)
                effect = compute_effect_size(baseline_values, model_values)

                analysis[dataset][model_name] = {
                    "mean": sum(model_values) / len(model_values),
                    "std": (sum((v - sum(model_values)/len(model_values))**2
                               for v in model_values) / len(model_values)) ** 0.5,
                    "vs_baseline": sig_result,
                    "effect_size": effect,
                }

    return analysis


def generate_reports(
    results: Dict,
    analysis: Dict,
    output_dir: Path,
) -> None:
    """Generate analysis reports and visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary table (Markdown)
    table_md = create_summary_table(results, format="markdown")
    with open(output_dir / "summary_table.md", "w") as f:
        f.write(table_md)
    logger.info(f"Saved summary table to {output_dir / 'summary_table.md'}")

    # Summary table (LaTeX)
    table_latex = create_summary_table(results, format="latex")
    with open(output_dir / "summary_table.tex", "w") as f:
        f.write(table_latex)
    logger.info(f"Saved LaTeX table to {output_dir / 'summary_table.tex'}")

    # Performance comparison plots
    for dataset in results:
        try:
            plot_performance_comparison(
                results[dataset],
                title=f"Model Comparison - {dataset}",
                save_path=output_dir / f"comparison_{dataset}.png",
            )
            logger.info(f"Saved comparison plot for {dataset}")
        except Exception as e:
            logger.warning(f"Failed to create plot for {dataset}: {e}")

    # Analysis summary
    with open(output_dir / "analysis_summary.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    logger.info(f"Saved analysis summary to {output_dir / 'analysis_summary.json'}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze DKO experiment results")

    parser.add_argument(
        "--experiment",
        type=str,
        default="benchmark",
        help="Experiment name",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Results directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for analysis (default: results-dir/analysis)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="single_conformer",
        help="Baseline model for comparison",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="markdown",
        choices=["markdown", "latex", "csv"],
        help="Output format for tables",
    )

    args = parser.parse_args()

    # Setup paths
    results_dir = Path(args.results_dir) / args.experiment
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "analysis"

    # Setup logging
    setup_logging(output_dir / "logs", name="analyze")

    logger.info(f"Analyzing results from: {results_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Load results
    results = load_results(results_dir)

    if not results:
        logger.error("No results to analyze")
        return

    # Analyze
    analysis = analyze_benchmark_results(results, output_dir, args.baseline)

    # Generate reports
    generate_reports(results, analysis, output_dir)

    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
