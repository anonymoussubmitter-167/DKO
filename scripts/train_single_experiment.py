#!/usr/bin/env python
"""
Single Experiment Training Script for DKO.

Wrapper script for training a single DKO experiment with full configuration support.
Provides a cleaner interface than the main train.py module.

Usage:
    python scripts/train_single_experiment.py --config configs/experiments/dko_esol.yaml
    python scripts/train_single_experiment.py --config configs/experiments/dko_esol.yaml --seed 42
    python scripts/train_single_experiment.py --dataset esol --model dko --epochs 100
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        if path.suffix in ['.yaml', '.yml']:
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML required to load YAML configs")
            return yaml.safe_load(f)
        else:
            return json.load(f)


def merge_configs(base: Dict, override: Dict) -> Dict:
    """Deep merge two config dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def create_experiment_dir(
    base_dir: str,
    experiment_name: str,
    seed: int
) -> Path:
    """Create experiment directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"{experiment_name}_seed{seed}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "metrics").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)

    return exp_dir


def save_config(config: Dict[str, Any], output_path: Path):
    """Save configuration to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)


def train_experiment(
    config: Dict[str, Any],
    experiment_dir: Path,
    seed: int,
    resume_from: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single training experiment."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for training")

    # Import training components
    from dko.scripts.train import (
        setup_experiment,
        create_dataloaders,
        create_model,
        train_model,
        evaluate_model,
    )

    # Set seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n{'='*60}")
    print(f"DKO Single Experiment Training")
    print(f"{'='*60}")
    print(f"Experiment directory: {experiment_dir}")
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"{'='*60}\n")

    # Setup experiment
    config = setup_experiment(config, seed=seed)

    # Save final config
    save_config(config, experiment_dir / "config.json")

    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(config)
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val: {len(val_loader.dataset)} samples")
    print(f"  Test: {len(test_loader.dataset)} samples")

    # Create model
    print("\nCreating model...")
    model = create_model(config, device=device)
    print(f"  Model: {config['model']['type']}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_from:
        print(f"\nResuming from: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"  Resuming from epoch {start_epoch}")

    # Train model
    print("\nStarting training...")
    training_history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        experiment_dir=experiment_dir,
        device=device,
        start_epoch=start_epoch,
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        config=config,
        device=device,
    )

    # Save results
    results = {
        "experiment_name": experiment_dir.name,
        "seed": seed,
        "device": device,
        "config": config,
        "training_history": training_history,
        "test_metrics": test_metrics,
        "completed": datetime.now().isoformat(),
    }

    results_path = experiment_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {results_path}")
    print("\nTest Metrics:")
    for key, value in test_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train a single DKO experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with config file
    python scripts/train_single_experiment.py --config configs/experiments/dko_esol.yaml

    # Train with custom seed
    python scripts/train_single_experiment.py --config configs/experiments/dko_esol.yaml --seed 123

    # Quick training with command-line options
    python scripts/train_single_experiment.py --dataset esol --epochs 50 --batch-size 32

    # Resume from checkpoint
    python scripts/train_single_experiment.py --config configs/experiments/dko_esol.yaml \\
        --resume-from experiments/dko_esol_seed42/checkpoints/last_model.pt
        """
    )

    # Config options
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to config file (YAML or JSON)"
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default="configs/base_config.yaml",
        help="Base config to inherit from"
    )

    # Dataset options
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["esol", "freesolv", "lipophilicity", "qm7", "qm9"],
        help="Dataset to use"
    )

    # Model options
    parser.add_argument(
        "--model",
        type=str,
        choices=["dko", "dko_first_order", "dko_second_order"],
        default="dko_first_order",
        help="Model type"
    )

    # Training options
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size"
    )
    parser.add_argument(
        "--learning-rate", "--lr",
        type=float,
        help="Learning rate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    # Output options
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Name for the experiment"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments",
        help="Base directory for experiment outputs"
    )

    # Resume options
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Path to checkpoint to resume from"
    )

    # Device options
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to use for training"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        help="GPU index to use"
    )

    args = parser.parse_args()

    # Load base config
    config = {}
    if args.base_config and Path(args.base_config).exists():
        config = load_config(args.base_config)

    # Override with experiment config
    if args.config:
        exp_config = load_config(args.config)
        config = merge_configs(config, exp_config)

    # Override with command-line arguments
    if args.dataset:
        config.setdefault("dataset", {})["name"] = args.dataset

    if args.model:
        config.setdefault("model", {})["type"] = args.model

    if args.epochs:
        config.setdefault("training", {})["max_epochs"] = args.epochs

    if args.batch_size:
        config.setdefault("training", {})["batch_size"] = args.batch_size

    if args.learning_rate:
        config.setdefault("training", {})["learning_rate"] = args.learning_rate

    # Validate config
    if "dataset" not in config or "name" not in config.get("dataset", {}):
        parser.error("Dataset must be specified via --config or --dataset")

    # Determine experiment name
    experiment_name = args.experiment_name
    if not experiment_name:
        dataset_name = config["dataset"]["name"]
        model_name = config.get("model", {}).get("type", "dko")
        experiment_name = f"{model_name}_{dataset_name}"

    # Setup device
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.gpu is not None and device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Create experiment directory
    exp_dir = create_experiment_dir(args.output_dir, experiment_name, args.seed)

    # Run training
    try:
        results = train_experiment(
            config=config,
            experiment_dir=exp_dir,
            seed=args.seed,
            resume_from=args.resume_from,
            device=device,
        )
        sys.exit(0)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
