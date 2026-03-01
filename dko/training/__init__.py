"""
DKO Training Module

Contains utilities for:
- Model training (basic and HPC-grade)
- Experiment logging
- Evaluation
- Hyperparameter optimization
"""

from dko.training.trainer import Trainer, EarlyStopping, create_trainer

__all__ = [
    # Basic Trainer
    "Trainer",
    "EarlyStopping",
    "create_trainer",
]

# HPC-grade trainer (optional, has more dependencies)
try:
    from dko.training.hpc_trainer import (
        EnhancedTrainer,
        ExperimentLogger,
        create_enhanced_trainer,
    )
    __all__.extend(["EnhancedTrainer", "ExperimentLogger", "create_enhanced_trainer"])
except ImportError:
    pass

# Evaluator (optional - may not exist yet)
try:
    from dko.training.evaluator import Evaluator, compute_metrics
    __all__.extend(["Evaluator", "compute_metrics"])
except ImportError:
    pass

# Hyperparameter optimization (optional - may not exist yet)
try:
    from dko.training.hyperopt import (
        HyperparameterOptimizer,
        run_hyperopt,
        create_optuna_study,
    )
    __all__.extend(["HyperparameterOptimizer", "run_hyperopt", "create_optuna_study"])
except ImportError:
    pass
