"""
Configuration system for DKO experiments.

Provides YAML-based configuration loading with hierarchical config support,
environment variable overrides, validation, and merging utilities.
"""

import os
import copy
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field


def deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries.

    Values in override take precedence. Nested dicts are merged recursively.

    Args:
        base: Base dictionary
        override: Override dictionary with values to merge

    Returns:
        Merged dictionary
    """
    result = copy.deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result


def load_yaml(path: Union[str, Path]) -> Dict:
    """Load a YAML file and return its contents as a dictionary."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_env_overrides(config: Dict, prefix: str = "DKO") -> Dict:
    """
    Apply environment variable overrides to configuration.

    Environment variables should be formatted as:
    {PREFIX}_{SECTION}_{KEY} = value

    For example: DKO_TRAINING_BATCH_SIZE=64

    Args:
        config: Configuration dictionary to modify
        prefix: Environment variable prefix

    Returns:
        Modified configuration dictionary
    """
    result = copy.deepcopy(config)

    for env_key, env_value in os.environ.items():
        if not env_key.startswith(f"{prefix}_"):
            continue

        # Parse the key path
        parts = env_key[len(prefix) + 1:].lower().split("_")

        # Navigate to the right location in config
        current = result
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set the value with type inference
        final_key = parts[-1]
        current[final_key] = _infer_type(env_value)

    return result


def _infer_type(value: str) -> Any:
    """Infer the type of a string value."""
    # Boolean
    if value.lower() in ("true", "yes", "1"):
        return True
    if value.lower() in ("false", "no", "0"):
        return False

    # None
    if value.lower() in ("none", "null"):
        return None

    # Integer
    try:
        return int(value)
    except ValueError:
        pass

    # Float
    try:
        return float(value)
    except ValueError:
        pass

    # List (comma-separated)
    if "," in value:
        return [_infer_type(v.strip()) for v in value.split(",")]

    return value


@dataclass
class ConfigSchema:
    """Schema definition for configuration validation."""
    required_fields: List[str] = field(default_factory=list)
    field_types: Dict[str, type] = field(default_factory=dict)
    field_ranges: Dict[str, tuple] = field(default_factory=dict)
    field_choices: Dict[str, List] = field(default_factory=dict)


# Default configuration values
DEFAULT_CONFIG = {
    "project": {
        "name": "dko-conformer-ensembles",
        "seed": 42,
        "num_workers": 8,
    },
    "data": {
        "conformer_generation": {
            "method": "ETKDG",
            "max_conformers": 50,
            "energy_window": 15.0,
            "rmsd_threshold": 0.5,
            "force_field": "MMFF94",
            "temperature": 300,
            "random_seed": 42,
        },
        "splitting": {
            "method": "scaffold",
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "seed": 42,
            "stratify": True,
        },
        "features": {
            "bond_distance_cutoff": 4.0,
            "include_bond_distances": True,
            "include_bond_angles": True,
            "include_torsion_angles": True,
        },
    },
    "training": {
        "optimizer": "AdamW",
        "base_learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "max_epochs": 300,
        "early_stopping_patience": 30,
        "batch_size": 32,
        "gradient_clip_max_norm": 1.0,
        "mixed_precision": True,
        "scheduler": {
            "type": "cosine",
            "eta_min": 1e-6,
        },
        "loss": {
            "regression": "mse",
            "classification": "bce_with_logits",
        },
    },
    "evaluation": {
        "regression_metrics": ["rmse", "mae", "pearson"],
        "classification_metrics": ["auc", "accuracy"],
        "num_seeds": 3,
        "random_seeds": [42, 123, 456],
    },
    "hyperopt": {
        "method": "optuna",
        "n_trials": 50,
        "sampler": "TPE",
        "pruner": "MedianPruner",
    },
    "wandb": {
        "entity": None,
        "project": "dko-research",
        "tags": ["molecular-ml", "conformers", "dko"],
        "log_freq": 10,
    },
}


# Validation schema
VALIDATION_SCHEMA = ConfigSchema(
    required_fields=[
        "project.name",
        "project.seed",
        "training.batch_size",
        "training.max_epochs",
    ],
    field_types={
        "project.seed": int,
        "training.batch_size": int,
        "training.max_epochs": int,
        "training.base_learning_rate": float,
        "data.conformer_generation.max_conformers": int,
    },
    field_ranges={
        "training.batch_size": (1, 1024),
        "training.max_epochs": (1, 10000),
        "training.base_learning_rate": (1e-8, 1.0),
        "data.splitting.train_ratio": (0.0, 1.0),
        "data.splitting.val_ratio": (0.0, 1.0),
        "data.splitting.test_ratio": (0.0, 1.0),
    },
    field_choices={
        "data.splitting.method": ["scaffold", "random", "stratified"],
        "training.optimizer": ["Adam", "AdamW", "SGD", "RMSprop"],
        "data.conformer_generation.force_field": ["MMFF94", "UFF", "MMFF94s"],
    },
)


def get_nested_value(config: Dict, key_path: str, default: Any = None) -> Any:
    """Get a nested value from a config using dot notation."""
    keys = key_path.split(".")
    current = config

    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]

    return current


def set_nested_value(config: Dict, key_path: str, value: Any) -> None:
    """Set a nested value in a config using dot notation."""
    keys = key_path.split(".")
    current = config

    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value


def validate_config(config: Dict, schema: ConfigSchema = VALIDATION_SCHEMA) -> List[str]:
    """
    Validate a configuration against a schema.

    Args:
        config: Configuration dictionary to validate
        schema: Validation schema

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Check required fields
    for field_path in schema.required_fields:
        value = get_nested_value(config, field_path)
        if value is None:
            errors.append(f"Required field missing: {field_path}")

    # Check field types
    for field_path, expected_type in schema.field_types.items():
        value = get_nested_value(config, field_path)
        if value is not None and not isinstance(value, expected_type):
            errors.append(
                f"Type mismatch for {field_path}: expected {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )

    # Check field ranges
    for field_path, (min_val, max_val) in schema.field_ranges.items():
        value = get_nested_value(config, field_path)
        if value is not None and (value < min_val or value > max_val):
            errors.append(
                f"Value out of range for {field_path}: {value} not in [{min_val}, {max_val}]"
            )

    # Check field choices
    for field_path, choices in schema.field_choices.items():
        value = get_nested_value(config, field_path)
        if value is not None and value not in choices:
            errors.append(
                f"Invalid value for {field_path}: {value} not in {choices}"
            )

    return errors


class Config:
    """
    Configuration manager for DKO experiments.

    Supports hierarchical configuration loading with the following precedence:
    1. Environment variables (highest)
    2. Experiment config
    3. Model config
    4. Dataset config
    5. Base config
    6. Default values (lowest)
    """

    def __init__(
        self,
        base_config_path: Optional[Union[str, Path]] = None,
        dataset_config_path: Optional[Union[str, Path]] = None,
        model_config_path: Optional[Union[str, Path]] = None,
        experiment_config_path: Optional[Union[str, Path]] = None,
        overrides: Optional[Dict] = None,
        apply_env: bool = True,
    ):
        """
        Initialize configuration.

        Args:
            base_config_path: Path to base configuration file
            dataset_config_path: Path to dataset-specific configuration
            model_config_path: Path to model-specific configuration
            experiment_config_path: Path to experiment-specific configuration
            overrides: Dictionary of manual overrides
            apply_env: Whether to apply environment variable overrides
        """
        # Start with defaults
        self._config = copy.deepcopy(DEFAULT_CONFIG)

        # Load and merge configs in order of increasing precedence
        config_paths = [
            base_config_path,
            dataset_config_path,
            model_config_path,
            experiment_config_path,
        ]

        for path in config_paths:
            if path is not None:
                loaded = load_yaml(path)
                self._config = deep_merge(self._config, loaded)

        # Apply manual overrides
        if overrides:
            self._config = deep_merge(self._config, overrides)

        # Apply environment variable overrides
        if apply_env:
            self._config = apply_env_overrides(self._config)

        # Validate
        errors = validate_config(self._config)
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(errors))

    def __getitem__(self, key: str) -> Any:
        """Get a configuration value using dot notation or dictionary access."""
        if "." in key:
            return get_nested_value(self._config, key)
        return self._config.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set a configuration value using dot notation or dictionary access."""
        if "." in key:
            set_nested_value(self._config, key, value)
        else:
            self._config[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value with a default."""
        value = self[key]
        return value if value is not None else default

    def to_dict(self) -> Dict:
        """Return the full configuration as a dictionary."""
        return copy.deepcopy(self._config)

    def save(self, path: Union[str, Path]) -> None:
        """Save the configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_args(
        cls,
        config_dir: Union[str, Path] = "configs",
        dataset: Optional[str] = None,
        model: Optional[str] = None,
        experiment: Optional[str] = None,
        **overrides,
    ) -> "Config":
        """
        Create a configuration from command-line style arguments.

        Args:
            config_dir: Base directory containing config files
            dataset: Dataset name (e.g., "bace")
            model: Model name (e.g., "dko")
            experiment: Experiment name (e.g., "main_benchmark")
            **overrides: Additional overrides as keyword arguments

        Returns:
            Configured Config instance
        """
        config_dir = Path(config_dir)

        base_path = config_dir / "base_config.yaml"
        dataset_path = config_dir / "datasets" / f"{dataset}.yaml" if dataset else None
        model_path = config_dir / "models" / f"{model}.yaml" if model else None
        exp_path = config_dir / "experiments" / f"{experiment}.yaml" if experiment else None

        return cls(
            base_config_path=base_path if base_path.exists() else None,
            dataset_config_path=dataset_path if dataset_path and dataset_path.exists() else None,
            model_config_path=model_path if model_path and model_path.exists() else None,
            experiment_config_path=exp_path if exp_path and exp_path.exists() else None,
            overrides=overrides if overrides else None,
        )


def create_experiment_config(
    dataset: str,
    model: str,
    experiment: str = "default",
    config_dir: Union[str, Path] = "configs",
    **overrides,
) -> Config:
    """
    Convenience function to create an experiment configuration.

    Args:
        dataset: Dataset name
        model: Model name
        experiment: Experiment name
        config_dir: Configuration directory
        **overrides: Additional overrides

    Returns:
        Configured Config instance
    """
    return Config.from_args(
        config_dir=config_dir,
        dataset=dataset,
        model=model,
        experiment=experiment,
        **overrides,
    )
