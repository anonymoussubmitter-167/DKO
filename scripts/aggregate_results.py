#!/usr/bin/env python
"""
Results aggregation script for DKO experiments.

Aggregates results from multiple experiments into summary tables and plots.

Usage:
    python scripts/aggregate_results.py --experiments-dir experiments/
    python scripts/aggregate_results.py --experiments-dir experiments/ --output results/summary.csv
    python scripts/aggregate_results.py --experiments-dir experiments/ --format latex
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys
from collections import defaultdict
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


def find_experiments(experiments_dir: Path) -> List[Path]:
    """Find all experiment directories."""
    experiments = []

    for exp_dir in experiments_dir.iterdir():
        if exp_dir.is_dir():
            # Check for required files
            has_config = (exp_dir / 'config.json').exists()
            has_metrics = (exp_dir / 'metrics' / 'training_history.json').exists()

            if has_config and has_metrics:
                experiments.append(exp_dir)

    return sorted(experiments)


def load_experiment_data(exp_dir: Path) -> Optional[Dict[str, Any]]:
    """Load data from a single experiment."""
    try:
        # Load config
        with open(exp_dir / 'config.json', 'r') as f:
            config = json.load(f)

        # Load training history
        with open(exp_dir / 'metrics' / 'training_history.json', 'r') as f:
            history = json.load(f)

        # Load model info if available
        model_info = {}
        if (exp_dir / 'model_info.json').exists():
            with open(exp_dir / 'model_info.json', 'r') as f:
                model_info = json.load(f)

        # Load environment info if available
        env_info = {}
        if (exp_dir / 'environment.json').exists():
            with open(exp_dir / 'environment.json', 'r') as f:
                env_info = json.load(f)

        # Load final metrics if available
        final_metrics = {}
        if (exp_dir / 'metrics' / 'final_metrics.json').exists():
            with open(exp_dir / 'metrics' / 'final_metrics.json', 'r') as f:
                final_metrics = json.load(f)

        # Find best epoch
        val_losses = history.get('val_loss', [])
        if val_losses:
            best_epoch = int(np.argmin(val_losses)) + 1 if NUMPY_AVAILABLE else val_losses.index(min(val_losses)) + 1
            best_val_loss = min(val_losses)
        else:
            best_epoch = 0
            best_val_loss = float('inf')

        return {
            'name': exp_dir.name,
            'path': str(exp_dir),
            'config': config,
            'history': history,
            'model_info': model_info,
            'env_info': env_info,
            'final_metrics': final_metrics,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'total_epochs': len(val_losses),
        }

    except Exception as e:
        print(f"Error loading experiment {exp_dir.name}: {e}")
        return None


def extract_summary_row(exp_data: Dict) -> Dict[str, Any]:
    """Extract a summary row for the results table."""
    config = exp_data['config']
    history = exp_data['history']
    model_info = exp_data.get('model_info', {})
    final_metrics = exp_data.get('final_metrics', {})

    # Extract key information
    row = {
        'experiment': exp_data['name'],
        'model': config.get('model', 'unknown'),
        'dataset': config.get('dataset', {}).get('name', 'unknown') if isinstance(config.get('dataset'), dict) else config.get('dataset', 'unknown'),
        'task': config.get('task', 'unknown'),

        # Training info
        'learning_rate': config.get('learning_rate', config.get('base_learning_rate', 'N/A')),
        'batch_size': config.get('batch_size', 'N/A'),
        'epochs_trained': exp_data['total_epochs'],
        'best_epoch': exp_data['best_epoch'],

        # Best validation loss
        'best_val_loss': exp_data['best_val_loss'],

        # Final training loss
        'final_train_loss': history['train_loss'][-1] if history.get('train_loss') else None,

        # Model parameters
        'parameters': model_info.get('total_parameters', 'N/A'),
    }

    # Add final metrics if available
    for metric, value in final_metrics.items():
        row[f'test_{metric}'] = value

    return row


def create_summary_table(experiments: List[Dict]) -> Optional[Any]:
    """Create a summary DataFrame from experiments."""
    if not PANDAS_AVAILABLE:
        print("pandas not available, returning dict format")
        return [extract_summary_row(exp) for exp in experiments if exp]

    rows = []
    for exp_data in experiments:
        if exp_data:
            rows.append(extract_summary_row(exp_data))

    if not rows:
        return None

    df = pd.DataFrame(rows)

    # Sort by dataset and model
    df = df.sort_values(['dataset', 'model', 'experiment'])

    return df


def create_dataset_summary(df) -> Any:
    """Create per-dataset summary with mean and std across seeds."""
    if not PANDAS_AVAILABLE:
        return None

    # Group by dataset and model
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    groupby_cols = ['dataset', 'model', 'task']

    # Only include columns that exist in the dataframe
    groupby_cols = [c for c in groupby_cols if c in df.columns]

    if not groupby_cols:
        return df

    # Aggregate
    agg_funcs = {col: ['mean', 'std', 'count'] for col in numeric_cols if col in df.columns}

    summary = df.groupby(groupby_cols).agg(agg_funcs)

    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]

    return summary.reset_index()


def format_for_latex(df) -> str:
    """Format summary table for LaTeX."""
    if not PANDAS_AVAILABLE:
        return "pandas required for LaTeX formatting"

    # Select key columns
    display_cols = ['dataset', 'model', 'best_val_loss', 'parameters', 'epochs_trained']
    display_cols = [c for c in display_cols if c in df.columns]

    # Add any test metrics
    test_cols = [c for c in df.columns if c.startswith('test_')]
    display_cols.extend(test_cols)

    df_display = df[display_cols].copy()

    # Format numbers
    for col in df_display.select_dtypes(include=[float]).columns:
        df_display[col] = df_display[col].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')

    return df_display.to_latex(index=False, escape=False)


def format_for_markdown(df) -> str:
    """Format summary table for Markdown."""
    if not PANDAS_AVAILABLE:
        return "pandas required for Markdown formatting"

    return df.to_markdown(index=False)


def print_summary(experiments: List[Dict]):
    """Print a quick summary to console."""
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    # Group by dataset and model
    by_dataset_model = defaultdict(list)
    for exp in experiments:
        if exp:
            key = (exp['config'].get('dataset', 'unknown'), exp['config'].get('model', 'unknown'))
            by_dataset_model[key].append(exp)

    for (dataset, model), exps in sorted(by_dataset_model.items()):
        print(f"\n{dataset} - {model}:")
        print("-" * 40)

        val_losses = [e['best_val_loss'] for e in exps if e['best_val_loss'] < float('inf')]
        epochs = [e['total_epochs'] for e in exps]

        if val_losses and NUMPY_AVAILABLE:
            print(f"  Runs: {len(exps)}")
            print(f"  Best Val Loss: {np.mean(val_losses):.4f} +/- {np.std(val_losses):.4f}")
            print(f"  Epochs: {np.mean(epochs):.1f} +/- {np.std(epochs):.1f}")
        else:
            print(f"  Runs: {len(exps)}")
            for exp in exps:
                print(f"    {exp['name']}: val_loss={exp['best_val_loss']:.4f}, epochs={exp['total_epochs']}")


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate DKO experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--experiments-dir',
        type=str,
        default='experiments',
        help='Directory containing experiment results'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (CSV or JSON)'
    )
    parser.add_argument(
        '--format',
        choices=['csv', 'json', 'latex', 'markdown'],
        default='csv',
        help='Output format'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Create aggregated summary across seeds'
    )
    parser.add_argument(
        '--filter-dataset',
        type=str,
        help='Filter by dataset name'
    )
    parser.add_argument(
        '--filter-model',
        type=str,
        help='Filter by model type'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed output'
    )

    args = parser.parse_args()

    # Find experiments
    experiments_dir = Path(args.experiments_dir)
    if not experiments_dir.exists():
        print(f"Experiments directory not found: {experiments_dir}")
        sys.exit(1)

    exp_paths = find_experiments(experiments_dir)
    print(f"Found {len(exp_paths)} experiments")

    if not exp_paths:
        print("No valid experiments found")
        sys.exit(0)

    # Load experiment data
    experiments = []
    for exp_path in exp_paths:
        exp_data = load_experiment_data(exp_path)
        if exp_data:
            experiments.append(exp_data)

    print(f"Successfully loaded {len(experiments)} experiments")

    if not experiments:
        print("No experiments to aggregate")
        sys.exit(0)

    # Apply filters
    if args.filter_dataset:
        experiments = [e for e in experiments
                       if args.filter_dataset.lower() in str(e['config'].get('dataset', '')).lower()]

    if args.filter_model:
        experiments = [e for e in experiments
                       if args.filter_model.lower() in str(e['config'].get('model', '')).lower()]

    print(f"After filtering: {len(experiments)} experiments")

    # Print console summary
    if args.verbose:
        print_summary(experiments)

    # Create summary table
    df = create_summary_table(experiments)

    if df is None or (PANDAS_AVAILABLE and len(df) == 0):
        print("No data to output")
        sys.exit(0)

    # Create aggregated summary if requested
    if args.summary and PANDAS_AVAILABLE:
        df = create_dataset_summary(df)

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if args.format == 'csv':
            if PANDAS_AVAILABLE:
                df.to_csv(output_path, index=False)
            else:
                import csv
                with open(output_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=df[0].keys())
                    writer.writeheader()
                    writer.writerows(df)

        elif args.format == 'json':
            if PANDAS_AVAILABLE:
                df.to_json(output_path, orient='records', indent=2)
            else:
                with open(output_path, 'w') as f:
                    json.dump(df, f, indent=2)

        elif args.format == 'latex':
            with open(output_path, 'w') as f:
                f.write(format_for_latex(df))

        elif args.format == 'markdown':
            with open(output_path, 'w') as f:
                f.write(format_for_markdown(df))

        print(f"\nResults saved to: {output_path}")

    else:
        # Print to console
        if PANDAS_AVAILABLE:
            print("\n" + "=" * 80)
            print("RESULTS TABLE")
            print("=" * 80)
            print(df.to_string())
        else:
            print("\nResults (dict format):")
            for row in df:
                print(row)


if __name__ == '__main__':
    main()
