"""
Bayesian hyperparameter optimization using Optuna.

Specifications from research plan:
- Method: Optuna with TPE sampler
- Budget: 50 trials per method per dataset
- Pruner: MedianPruner for early stopping bad trials
- Objective: Minimize validation loss
- Model-specific search spaces
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Optional imports
try:
    import optuna
    from optuna.pruners import MedianPruner, HyperbandPruner
    from optuna.samplers import TPESampler, RandomSampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger("hyperopt")


def check_optuna():
    """Check if Optuna is available."""
    if not OPTUNA_AVAILABLE:
        raise ImportError(
            "Optuna is required for hyperparameter optimization. "
            "Install with: pip install optuna"
        )


# =============================================================================
# Model-Specific Search Spaces (from Research Plan)
# =============================================================================

DKO_SEARCH_SPACE = {
    'learning_rate': {'type': 'log_uniform', 'low': 1e-5, 'high': 1e-3},
    'weight_decay': {'type': 'log_uniform', 'low': 1e-6, 'high': 1e-4},
    'dropout': {'type': 'categorical', 'choices': [0.0, 0.1, 0.2]},
    'pca_variance': {'type': 'categorical', 'choices': [0.90, 0.95, 0.99]},
    'kernel_output_dim': {'type': 'categorical', 'choices': [32, 64, 128]},
    'branch_hidden_dim': {'type': 'categorical', 'choices': [64, 128, 256]},
}

ATTENTION_SEARCH_SPACE = {
    'learning_rate': {'type': 'log_uniform', 'low': 1e-5, 'high': 1e-3},
    'weight_decay': {'type': 'log_uniform', 'low': 1e-6, 'high': 1e-4},
    'dropout': {'type': 'categorical', 'choices': [0.0, 0.1, 0.2]},
    'embed_dim': {'type': 'categorical', 'choices': [64, 128, 256]},
    'num_heads': {'type': 'categorical', 'choices': [2, 4, 8]},
    'num_attention_layers': {'type': 'int', 'low': 1, 'high': 3},
}

DEEPSETS_SEARCH_SPACE = {
    'learning_rate': {'type': 'log_uniform', 'low': 1e-5, 'high': 1e-3},
    'weight_decay': {'type': 'log_uniform', 'low': 1e-6, 'high': 1e-4},
    'dropout': {'type': 'categorical', 'choices': [0.0, 0.1, 0.2]},
    'encoder_hidden_dim': {'type': 'categorical', 'choices': [128, 256, 512]},
    'decoder_hidden_dim': {'type': 'categorical', 'choices': [64, 128, 256]},
}


def get_search_space(model_name: str) -> Dict[str, Dict]:
    """Get search space for a model."""
    spaces = {
        'dko': DKO_SEARCH_SPACE,
        'DKO': DKO_SEARCH_SPACE,
        'attention': ATTENTION_SEARCH_SPACE,
        'Attention': ATTENTION_SEARCH_SPACE,
        'AttentionPoolingBaseline': ATTENTION_SEARCH_SPACE,
        'deepsets': DEEPSETS_SEARCH_SPACE,
        'DeepSets': DEEPSETS_SEARCH_SPACE,
        'DeepSetsBaseline': DEEPSETS_SEARCH_SPACE,
    }
    return spaces.get(model_name, DKO_SEARCH_SPACE)


class HyperparameterOptimizer:
    """
    Bayesian hyperparameter optimization using Optuna.

    Specifications from research plan:
    - TPE sampler for Bayesian optimization
    - MedianPruner for early stopping bad trials
    - 50 trials per model per dataset
    - Model-specific search spaces
    """

    def __init__(
        self,
        model_class: type,
        model_name: str,
        task: str,
        feature_dim: int,
        output_dim: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_trials: int = 50,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        device: Optional[str] = None,
        max_epochs: int = 100,
        early_stopping_patience: int = 15,
        search_space: Optional[Dict[str, Dict]] = None,
        verbose: bool = True,
    ):
        """
        Initialize hyperparameter optimizer.

        Args:
            model_class: Model class to optimize
            model_name: Model name (for search space selection)
            task: 'regression' or 'classification'
            feature_dim: Input feature dimension
            output_dim: Output dimension
            train_loader: Training data loader
            val_loader: Validation data loader
            n_trials: Number of optimization trials (default: 50)
            study_name: Name for Optuna study
            storage: Database URL for distributed optimization
            device: Device to use
            max_epochs: Maximum epochs per trial
            early_stopping_patience: Early stopping patience
            search_space: Custom search space (uses model default if None)
            verbose: Whether to print progress
        """
        check_optuna()

        self.model_class = model_class
        self.model_name = model_name
        self.task = task
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_trials = n_trials
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.verbose = verbose

        # Get search space
        self.search_space = search_space or get_search_space(model_name)

        # Create study name
        if study_name is None:
            study_name = f"{model_name}_{task}_hyperopt"

        # Create study with TPE sampler and MedianPruner (from research plan)
        self.study = optuna.create_study(
            study_name=study_name,
            direction='minimize',  # Minimize validation loss
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=10),
            storage=storage,
            load_if_exists=True,
        )

        self.best_model = None
        self.best_params = None

    def _sample_params(self, trial: "optuna.Trial") -> Dict[str, Any]:
        """Sample hyperparameters for a trial."""
        params = {}

        for name, config in self.search_space.items():
            param_type = config["type"]

            if param_type == "log_uniform":
                params[name] = trial.suggest_float(
                    name, config["low"], config["high"], log=True
                )
            elif param_type == "uniform":
                params[name] = trial.suggest_float(
                    name, config["low"], config["high"]
                )
            elif param_type == "int":
                params[name] = trial.suggest_int(
                    name, config["low"], config["high"]
                )
            elif param_type == "categorical":
                params[name] = trial.suggest_categorical(name, config["choices"])
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")

        return params

    def _create_model(self, params: Dict[str, Any]) -> nn.Module:
        """Create model with given hyperparameters."""
        # Extract model-specific parameters
        model_params = {k: v for k, v in params.items()
                        if k not in ['learning_rate', 'weight_decay']}

        if self.model_name in ['dko', 'DKO']:
            from dko.models.dko import DKO
            model = DKO(
                feature_dim=self.feature_dim,
                output_dim=self.output_dim,
                task=self.task,
                pca_variance=model_params.get('pca_variance', 0.95),
                kernel_output_dim=model_params.get('kernel_output_dim', 64),
                branch_hidden_dim=model_params.get('branch_hidden_dim', 128),
                dropout=model_params.get('dropout', 0.1),
                verbose=False,
            )

        elif self.model_name in ['attention', 'Attention', 'AttentionPoolingBaseline']:
            from dko.models.attention import AttentionPoolingBaseline
            model = AttentionPoolingBaseline(
                feature_dim=self.feature_dim,
                output_dim=self.output_dim,
                task=self.task,
                embed_dim=model_params.get('embed_dim', 128),
                num_heads=model_params.get('num_heads', 4),
                num_attention_layers=model_params.get('num_attention_layers', 2),
                dropout=model_params.get('dropout', 0.1),
            )

        elif self.model_name in ['deepsets', 'DeepSets', 'DeepSetsBaseline']:
            from dko.models.deepsets import DeepSetsBaseline
            encoder_dim = model_params.get('encoder_hidden_dim', 256)
            model = DeepSetsBaseline(
                feature_dim=self.feature_dim,
                output_dim=self.output_dim,
                task=self.task,
                encoder_hidden_dims=[encoder_dim, encoder_dim, encoder_dim // 2],
                decoder_hidden_dim=model_params.get('decoder_hidden_dim', 128),
                dropout=model_params.get('dropout', 0.1),
            )

        else:
            # Generic model creation
            model = self.model_class(
                feature_dim=self.feature_dim,
                output_dim=self.output_dim,
                **model_params
            )

        return model.to(self.device)

    def _get_batch_data(self, batch: Dict) -> Tuple[Dict, torch.Tensor]:
        """Extract data from batch."""
        labels = batch.get('label', batch.get('labels'))
        if labels is not None:
            labels = labels.to(self.device)

        # DKO format
        if 'mu' in batch and 'sigma' in batch:
            return {
                'format': 'dko',
                'mu': batch['mu'].to(self.device),
                'sigma': batch['sigma'].to(self.device),
            }, labels

        # Baseline format
        if 'features' in batch:
            mask = batch.get('mask')
            if mask is not None:
                mask = mask.to(self.device)

            weights = batch.get('weights', batch.get('boltzmann_weights'))
            if weights is not None:
                weights = weights.to(self.device)

            return {
                'format': 'baseline',
                'features': batch['features'].to(self.device),
                'mask': mask,
                'weights': weights,
            }, labels

        raise ValueError("Unknown batch format")

    def _forward_pass(self, model: nn.Module, data: Dict, fit_pca: bool = False) -> torch.Tensor:
        """Forward pass for any model format."""
        if data['format'] == 'dko':
            outputs = model(data['mu'], data['sigma'], fit_pca=fit_pca)
        else:
            features = data['features']
            mask = data.get('mask')
            weights = data.get('weights')

            if weights is not None:
                outputs = model(features, weights, mask=mask)
            elif mask is not None:
                outputs = model(features, mask=mask)
            else:
                outputs = model(features)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

        return outputs

    def objective(self, trial: "optuna.Trial") -> float:
        """
        Objective function for Optuna optimization.

        Args:
            trial: Optuna trial

        Returns:
            Validation loss to minimize
        """
        # Sample hyperparameters
        params = self._sample_params(trial)

        if self.verbose:
            logger.info(f"Trial {trial.number}: {params}")

        # Create model
        try:
            model = self._create_model(params)
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            raise optuna.TrialPruned()

        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params.get('learning_rate', 1e-4),
            weight_decay=params.get('weight_decay', 1e-5)
        )

        # Loss function
        if self.task == 'regression':
            criterion = nn.MSELoss()
        else:
            criterion = nn.BCEWithLogitsLoss()

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.max_epochs):
            # Training
            model.train()
            train_loss = 0.0
            n_batches = 0

            for batch_idx, batch in enumerate(self.train_loader):
                data, labels = self._get_batch_data(batch)

                # Fit PCA on first batch of first epoch for DKO
                fit_pca = (epoch == 0 and batch_idx == 0)

                optimizer.zero_grad()
                outputs = self._forward_pass(model, data, fit_pca=fit_pca)
                loss = criterion(outputs.squeeze(), labels.squeeze())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            train_loss /= n_batches

            # Validation
            model.eval()
            val_loss = 0.0
            n_val_batches = 0

            with torch.no_grad():
                for batch in self.val_loader:
                    data, labels = self._get_batch_data(batch)
                    outputs = self._forward_pass(model, data, fit_pca=False)
                    loss = criterion(outputs.squeeze(), labels.squeeze())
                    val_loss += loss.item()
                    n_val_batches += 1

            val_loss /= n_val_batches

            # Report to Optuna for pruning
            trial.report(val_loss, epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    break

        return best_val_loss

    def optimize(self) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.

        Returns:
            Dictionary with best parameters and results
        """
        if self.verbose:
            print(f"\nStarting hyperparameter optimization for {self.model_name}")
            print(f"  Trials: {self.n_trials}")
            print(f"  Device: {self.device}")
            print(f"  Max epochs per trial: {self.max_epochs}")

        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=self.verbose,
        )

        # Get best trial
        best_trial = self.study.best_trial
        self.best_params = best_trial.params

        if self.verbose:
            print(f"\nOptimization completed!")
            print(f"  Best trial: {best_trial.number}")
            print(f"  Best value: {best_trial.value:.4f}")
            print(f"  Best params:")
            for key, value in best_trial.params.items():
                print(f"    {key}: {value}")

        return {
            'best_params': best_trial.params,
            'best_value': best_trial.value,
            'best_trial_number': best_trial.number,
            'n_trials': len(self.study.trials),
            'n_pruned': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        }

    def get_importance(self) -> Dict[str, float]:
        """Get hyperparameter importance scores."""
        if len(self.study.trials) < 2:
            return {}
        try:
            return optuna.importance.get_param_importances(self.study)
        except Exception:
            return {}

    def save_results(self, output_dir: Union[str, Path]):
        """Save optimization results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save best parameters as YAML
        if YAML_AVAILABLE:
            params_file = output_dir / f'{self.model_name}_best_params.yaml'
            with open(params_file, 'w') as f:
                yaml.dump(self.study.best_params, f)

        # Save statistics as JSON
        stats = {
            'best_value': self.study.best_value,
            'best_trial': self.study.best_trial.number,
            'n_trials': len(self.study.trials),
            'best_params': self.study.best_params,
            'importance': self.get_importance(),
        }

        stats_file = output_dir / f'{self.model_name}_optimization_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        if self.verbose:
            print(f"Results saved to {output_dir}")

    def plot_optimization_history(self, output_dir: Union[str, Path]):
        """Plot optimization history if plotly is available."""
        try:
            from optuna.visualization import (
                plot_optimization_history,
                plot_param_importances,
            )

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Optimization history
            fig = plot_optimization_history(self.study)
            fig.write_html(str(output_dir / f'{self.model_name}_optimization_history.html'))

            # Parameter importances
            if len(self.study.trials) >= 10:
                fig = plot_param_importances(self.study)
                fig.write_html(str(output_dir / f'{self.model_name}_param_importances.html'))

            if self.verbose:
                print(f"Plots saved to {output_dir}")

        except ImportError:
            logger.warning("Plotly not installed, skipping visualization")


def run_hyperopt(
    model_class: type,
    model_name: str,
    task: str,
    feature_dim: int,
    output_dim: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_trials: int = 50,
    device: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function for hyperparameter optimization.

    Args:
        model_class: Model class
        model_name: Model name
        task: Task type
        feature_dim: Feature dimension
        output_dim: Output dimension
        train_loader: Training loader
        val_loader: Validation loader
        n_trials: Number of trials
        device: Device
        **kwargs: Additional arguments

    Returns:
        Dictionary with optimization results
    """
    optimizer = HyperparameterOptimizer(
        model_class=model_class,
        model_name=model_name,
        task=task,
        feature_dim=feature_dim,
        output_dim=output_dim,
        train_loader=train_loader,
        val_loader=val_loader,
        n_trials=n_trials,
        device=device,
        **kwargs,
    )

    return optimizer.optimize()


def create_optuna_study(
    study_name: str,
    direction: str = "minimize",
    storage: Optional[str] = None,
    sampler: str = "TPE",
    pruner: str = "MedianPruner",
) -> "optuna.Study":
    """
    Create an Optuna study with specified configuration.

    Args:
        study_name: Name of the study
        direction: Optimization direction
        storage: Storage URL
        sampler: Sampler type ('TPE' or 'Random')
        pruner: Pruner type ('MedianPruner' or 'HyperbandPruner')

    Returns:
        Optuna Study object
    """
    check_optuna()

    if sampler == "TPE":
        sampler_obj = TPESampler(seed=42)
    elif sampler == "Random":
        sampler_obj = RandomSampler(seed=42)
    else:
        sampler_obj = TPESampler(seed=42)

    if pruner == "MedianPruner":
        pruner_obj = MedianPruner(n_startup_trials=10, n_warmup_steps=10)
    elif pruner == "HyperbandPruner":
        pruner_obj = HyperbandPruner()
    else:
        pruner_obj = MedianPruner()

    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        sampler=sampler_obj,
        pruner=pruner_obj,
        load_if_exists=True,
    )


if __name__ == "__main__":
    print("Testing HyperparameterOptimizer...")

    if not OPTUNA_AVAILABLE:
        print("Optuna not installed, skipping test")
    else:
        from torch.utils.data import TensorDataset

        # Create dummy data
        n_train = 64
        n_val = 32
        D = 50

        # Training data
        mu_train = torch.randn(n_train, D)
        sigma_train = torch.randn(n_train, D, D)
        sigma_train = torch.bmm(sigma_train, sigma_train.transpose(1, 2))
        labels_train = torch.randn(n_train, 1)

        # Validation data
        mu_val = torch.randn(n_val, D)
        sigma_val = torch.randn(n_val, D, D)
        sigma_val = torch.bmm(sigma_val, sigma_val.transpose(1, 2))
        labels_val = torch.randn(n_val, 1)

        def collate_fn(batch):
            mu, sigma, labels = zip(*batch)
            return {
                'mu': torch.stack(mu),
                'sigma': torch.stack(sigma),
                'label': torch.stack(labels),
            }

        train_dataset = TensorDataset(mu_train, sigma_train, labels_train)
        val_dataset = TensorDataset(mu_val, sigma_val, labels_val)

        train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=16, collate_fn=collate_fn)

        # Simple model for testing
        class SimpleModel(nn.Module):
            def __init__(self, feature_dim, output_dim, **kwargs):
                super().__init__()
                self.fc = nn.Linear(feature_dim, output_dim)

            def forward(self, mu, sigma=None, fit_pca=False):
                return self.fc(mu)

        # Create optimizer with minimal trials for testing
        optimizer = HyperparameterOptimizer(
            model_class=SimpleModel,
            model_name='test',
            task='regression',
            feature_dim=D,
            output_dim=1,
            train_loader=train_loader,
            val_loader=val_loader,
            n_trials=3,  # Small for testing
            max_epochs=3,
            search_space={
                'learning_rate': {'type': 'log_uniform', 'low': 1e-4, 'high': 1e-3},
            },
            verbose=False,
        )

        # Suppress Optuna logs for testing
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        results = optimizer.optimize()

        print("\n[OK] HyperparameterOptimizer test passed!")
        print(f"  Best params: {results['best_params']}")
        print(f"  Best value: {results['best_value']:.4f}")
        print(f"  Trials completed: {results['n_trials']}")
