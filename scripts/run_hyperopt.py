#!/usr/bin/env python
"""
Hyperparameter Optimization Script for DKO.

Uses Optuna for Bayesian hyperparameter optimization across DKO model configurations.

Usage:
    python scripts/run_hyperopt.py --dataset esol --n-trials 50
    python scripts/run_hyperopt.py --dataset freesolv --n-trials 100 --sampler tpe
    python scripts/run_hyperopt.py --dataset qm7 --n-trials 200 --pruner hyperband
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Callable

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import optuna
    from optuna.pruners import MedianPruner, HyperbandPruner
    from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class DKOHyperoptObjective:
    """Optuna objective for DKO hyperparameter optimization."""

    def __init__(
        self,
        dataset: str,
        base_config: Dict[str, Any],
        device: str = "cuda",
        max_epochs: int = 50,
        patience: int = 10,
    ):
        self.dataset = dataset
        self.base_config = base_config
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience

        # Import DKO components
        from dko.scripts.train import create_dataloaders, create_model
        from dko.training.trainer import Trainer

        self.create_dataloaders = create_dataloaders
        self.create_model = create_model
        self.Trainer = Trainer

    def __call__(self, trial: 'optuna.Trial') -> float:
        """Objective function for a single trial."""
        # Sample hyperparameters
        config = self._sample_hyperparameters(trial)

        try:
            # Create dataloaders
            train_loader, val_loader, _ = self.create_dataloaders(config)

            # Create model
            model = self.create_model(config, device=self.device)

            # Create trainer
            trainer = self.Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                device=self.device,
            )

            # Training loop with pruning
            best_val_loss = float('inf')
            epochs_without_improvement = 0

            for epoch in range(self.max_epochs):
                train_loss = trainer.train_epoch()
                val_loss = trainer.validate()

                # Report intermediate value for pruning
                trial.report(val_loss, epoch)

                # Handle pruning
                if trial.should_prune():
                    raise optuna.TrialPruned()

                # Track best
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                # Early stopping
                if epochs_without_improvement >= self.patience:
                    break

            return best_val_loss

        except Exception as e:
            print(f"Trial failed: {e}")
            return float('inf')

    def _sample_hyperparameters(self, trial: 'optuna.Trial') -> Dict[str, Any]:
        """Sample hyperparameters for a trial."""
        config = self.base_config.copy()

        # Learning rate
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        config.setdefault("training", {})["learning_rate"] = lr

        # Batch size
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        config["training"]["batch_size"] = batch_size

        # Weight decay
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        config["training"]["weight_decay"] = weight_decay

        # Model architecture
        config.setdefault("model", {})

        # Kernel hidden dimensions
        n_hidden_layers = trial.suggest_int("n_hidden_layers", 1, 4)
        hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
        config["model"]["kernel_hidden_dims"] = [hidden_dim] * n_hidden_layers

        # Kernel output dimension
        kernel_output_dim = trial.suggest_categorical("kernel_output_dim", [8, 16, 32, 64])
        config["model"]["kernel_output_dim"] = kernel_output_dim

        # Dropout
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        config["model"]["dropout"] = dropout

        # Number of conformers
        n_conformers = trial.suggest_int("n_conformers", 5, 50, step=5)
        config.setdefault("dataset", {})["n_conformers"] = n_conformers

        # Optimizer
        optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"])
        config["training"]["optimizer"] = optimizer

        if optimizer == "sgd":
            momentum = trial.suggest_float("momentum", 0.8, 0.99)
            config["training"]["momentum"] = momentum

        # Scheduler
        scheduler = trial.suggest_categorical(
            "scheduler",
            ["cosine", "step", "plateau", "none"]
        )
        config["training"]["scheduler"] = scheduler

        return config


def create_study(
    study_name: str,
    sampler: str = "tpe",
    pruner: str = "median",
    storage: Optional[str] = None,
) -> 'optuna.Study':
    """Create an Optuna study."""
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required for hyperparameter optimization")

    # Create sampler
    if sampler == "tpe":
        sampler_obj = TPESampler(seed=42)
    elif sampler == "random":
        sampler_obj = RandomSampler(seed=42)
    elif sampler == "cmaes":
        sampler_obj = CmaEsSampler(seed=42)
    else:
        raise ValueError(f"Unknown sampler: {sampler}")

    # Create pruner
    if pruner == "median":
        pruner_obj = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    elif pruner == "hyperband":
        pruner_obj = HyperbandPruner(min_resource=5, max_resource=50, reduction_factor=3)
    elif pruner == "none":
        pruner_obj = None
    else:
        raise ValueError(f"Unknown pruner: {pruner}")

    # Create study
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler_obj,
        pruner=pruner_obj,
        storage=storage,
        load_if_exists=True,
    )

    return study


def save_study_results(
    study: 'optuna.Study',
    output_dir: Path,
    dataset: str,
):
    """Save study results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Best trial
    best_trial = study.best_trial
    best_params = {
        "trial_number": best_trial.number,
        "value": best_trial.value,
        "params": best_trial.params,
        "datetime": datetime.now().isoformat(),
    }

    with open(output_dir / "best_params.json", 'w') as f:
        json.dump(best_params, f, indent=2)

    # All trials
    trials_data = []
    for trial in study.trials:
        trials_data.append({
            "number": trial.number,
            "value": trial.value if trial.value is not None else None,
            "state": trial.state.name,
            "params": trial.params,
        })

    with open(output_dir / "all_trials.json", 'w') as f:
        json.dump(trials_data, f, indent=2)

    # Parameter importance (if enough trials)
    if len(study.trials) >= 10:
        try:
            importance = optuna.importance.get_param_importances(study)
            with open(output_dir / "param_importance.json", 'w') as f:
                json.dump(importance, f, indent=2)
        except:
            pass

    # Generate best config file
    best_config = generate_best_config(best_trial.params, dataset)
    with open(output_dir / f"best_config_{dataset}.yaml", 'w') as f:
        import yaml
        yaml.dump(best_config, f, default_flow_style=False)

    print(f"\nResults saved to: {output_dir}")


def generate_best_config(params: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    """Generate a config file from best hyperparameters."""
    config = {
        "dataset": {
            "name": dataset,
            "n_conformers": params.get("n_conformers", 20),
        },
        "model": {
            "type": "dko_first_order",
            "kernel_hidden_dims": [params.get("hidden_dim", 64)] * params.get("n_hidden_layers", 2),
            "kernel_output_dim": params.get("kernel_output_dim", 16),
            "dropout": params.get("dropout", 0.1),
        },
        "training": {
            "max_epochs": 200,
            "batch_size": params.get("batch_size", 32),
            "learning_rate": params.get("learning_rate", 1e-3),
            "weight_decay": params.get("weight_decay", 1e-4),
            "optimizer": params.get("optimizer", "adamw"),
            "scheduler": params.get("scheduler", "cosine"),
            "patience": 20,
        },
    }

    if params.get("optimizer") == "sgd":
        config["training"]["momentum"] = params.get("momentum", 0.9)

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for DKO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic hyperopt with 50 trials
    python scripts/run_hyperopt.py --dataset esol --n-trials 50

    # Use TPE sampler with Hyperband pruning
    python scripts/run_hyperopt.py --dataset freesolv --n-trials 100 --sampler tpe --pruner hyperband

    # Resume a study
    python scripts/run_hyperopt.py --dataset esol --study-name my_study --n-trials 50

    # Use persistent storage
    python scripts/run_hyperopt.py --dataset qm7 --storage sqlite:///hyperopt.db
        """
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["esol", "freesolv", "lipophilicity", "qm7", "qm9"],
        help="Dataset to optimize for"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of optimization trials"
    )
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["tpe", "random", "cmaes"],
        default="tpe",
        help="Optimization sampler"
    )
    parser.add_argument(
        "--pruner",
        type=str,
        choices=["median", "hyperband", "none"],
        default="median",
        help="Trial pruning strategy"
    )
    parser.add_argument(
        "--study-name",
        type=str,
        help="Name for the study (default: auto-generated)"
    )
    parser.add_argument(
        "--storage",
        type=str,
        help="Optuna storage URL (e.g., sqlite:///hyperopt.db)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/hyperopt",
        help="Directory for output files"
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=50,
        help="Max epochs per trial"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cuda", "cpu", "auto"],
        help="Device to use"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs"
    )

    args = parser.parse_args()

    if not OPTUNA_AVAILABLE:
        print("ERROR: Optuna is required for hyperparameter optimization")
        print("Install with: pip install optuna")
        sys.exit(1)

    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch is required")
        sys.exit(1)

    # Determine device
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print("DKO Hyperparameter Optimization")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"N trials: {args.n_trials}")
    print(f"Sampler: {args.sampler}")
    print(f"Pruner: {args.pruner}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # Create study name
    study_name = args.study_name or f"dko_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create base config
    base_config = {
        "dataset": {"name": args.dataset},
        "model": {"type": "dko_first_order"},
        "training": {"max_epochs": args.max_epochs},
    }

    # Create objective
    objective = DKOHyperoptObjective(
        dataset=args.dataset,
        base_config=base_config,
        device=device,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )

    # Create study
    study = create_study(
        study_name=study_name,
        sampler=args.sampler,
        pruner=args.pruner,
        storage=args.storage,
    )

    # Run optimization
    print(f"Starting optimization with {args.n_trials} trials...")

    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
    )

    # Print results
    print(f"\n{'='*60}")
    print("Optimization Complete!")
    print(f"{'='*60}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_trial.value:.6f}")
    print("\nBest parameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

    # Save results
    output_dir = Path(args.output_dir) / study_name
    save_study_results(study, output_dir, args.dataset)

    print(f"\nTo train with best params:")
    print(f"  python scripts/train_single_experiment.py --config {output_dir}/best_config_{args.dataset}.yaml")


if __name__ == "__main__":
    main()
