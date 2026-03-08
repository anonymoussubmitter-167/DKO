"""
Logging utilities for DKO experiments.

Provides Weights & Biases integration, local file logging, experiment tracking,
checkpoint management, and progress bar utilities.
"""

import os
import sys
import json
import logging
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from functools import wraps
import pickle

import torch
from tqdm import tqdm

# Conditional import for wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def get_git_info() -> Dict[str, str]:
    """Get current git commit hash and branch."""
    info = {"commit": "unknown", "branch": "unknown", "dirty": False}

    try:
        # Get commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["commit"] = result.stdout.strip()[:8]

        # Get branch name
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["branch"] = result.stdout.strip()

        # Check if dirty
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["dirty"] = len(result.stdout.strip()) > 0

    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return info


def setup_logging(
    log_dir: Union[str, Path],
    name: str = "dko",
    level: int = logging.INFO,
    log_to_console: bool = True,
    log_to_file: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Set up logging with file rotation and console output.

    Args:
        log_dir: Directory for log files
        name: Logger name
        level: Logging level
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file
        max_bytes: Maximum bytes per log file before rotation
        backup_count: Number of backup files to keep

    Returns:
        Configured logger
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler with rotation
    if log_to_file:
        from logging.handlers import RotatingFileHandler

        log_file = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class ExperimentTracker:
    """
    Experiment tracking with Weights & Biases integration.

    Provides automatic run naming, config logging, metric tracking,
    and checkpoint management.
    """

    def __init__(
        self,
        project: str = "dko-research",
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        log_dir: Union[str, Path] = "logs",
        use_wandb: bool = True,
        resume: Optional[str] = None,
    ):
        """
        Initialize experiment tracker.

        Args:
            project: W&B project name
            name: Run name (auto-generated if None)
            config: Experiment configuration
            tags: Tags for the run
            notes: Notes for the run
            log_dir: Local log directory
            use_wandb: Whether to use W&B (if available)
            resume: W&B run ID to resume
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.config = config or {}
        self.step = 0
        self._metrics_history: List[Dict] = []

        # Get git info
        git_info = get_git_info()
        self.config["git"] = git_info

        # Generate run name if not provided
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"run_{timestamp}_{git_info['commit']}"

        self.name = name
        self.run_dir = self.log_dir / self.name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Setup local logger
        self.logger = setup_logging(self.run_dir, name="experiment")

        # Initialize W&B
        self.wandb_run = None
        if self.use_wandb:
            entity = os.environ.get("WANDB_ENTITY") or self.config.get("wandb", {}).get("entity")

            self.wandb_run = wandb.init(
                project=project,
                name=name,
                config=self.config,
                tags=tags or [],
                notes=notes,
                resume="allow" if resume else None,
                id=resume,
            )

            self.logger.info(f"W&B run initialized: {self.wandb_run.url}")

        # Save config locally
        self._save_config()

        self.logger.info(f"Experiment initialized: {self.name}")
        self.logger.info(f"Git commit: {git_info['commit']} (dirty: {git_info['dirty']})")

    def _save_config(self) -> None:
        """Save configuration to local file."""
        config_path = self.run_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, default=str)

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics.

        Args:
            metrics: Dictionary of metric names to values
            step: Step number (auto-incremented if None)
        """
        if step is None:
            step = self.step
            self.step += 1

        # Add step to metrics
        metrics_with_step = {"step": step, **metrics}

        # Log to W&B
        if self.use_wandb and self.wandb_run:
            wandb.log(metrics, step=step)

        # Log locally
        self._metrics_history.append(metrics_with_step)

        # Log to file periodically
        if step % 100 == 0:
            self._save_metrics()

    def log_summary(self, summary: Dict[str, Any]) -> None:
        """Log summary metrics (final results)."""
        if self.use_wandb and self.wandb_run:
            for key, value in summary.items():
                wandb.run.summary[key] = value

        # Save to local file
        summary_path = self.run_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        self.logger.info(f"Summary logged: {summary}")

    def _save_metrics(self) -> None:
        """Save metrics history to file."""
        metrics_path = self.run_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self._metrics_history, f, indent=2, default=str)

    def save_checkpoint(
        self,
        state: Dict[str, Any],
        name: str = "checkpoint",
        is_best: bool = False,
    ) -> Path:
        """
        Save a checkpoint.

        Args:
            state: State dictionary to save
            name: Checkpoint name
            is_best: Whether this is the best checkpoint so far

        Returns:
            Path to saved checkpoint
        """
        checkpoint_dir = self.run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"{name}.pt"
        torch.save(state, checkpoint_path)

        if is_best:
            best_path = checkpoint_dir / "best.pt"
            torch.save(state, best_path)
            self.logger.info(f"New best checkpoint saved: {best_path}")

        return checkpoint_path

    def load_checkpoint(self, name: str = "best") -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint.

        Args:
            name: Checkpoint name

        Returns:
            Loaded state dictionary or None if not found
        """
        checkpoint_dir = self.run_dir / "checkpoints"
        checkpoint_path = checkpoint_dir / f"{name}.pt"

        if checkpoint_path.exists():
            state = torch.load(checkpoint_path, map_location="cpu")
            self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return state

        return None

    def finish(self) -> None:
        """Finish the experiment run."""
        self._save_metrics()

        if self.use_wandb and self.wandb_run:
            wandb.finish()

        self.logger.info(f"Experiment finished: {self.name}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()
        return False


class ResultCache:
    """
    Cache for experiment results to avoid recomputation.

    Uses content-addressable storage based on configuration hashes.
    """

    def __init__(self, cache_dir: Union[str, Path] = ".cache"):
        """
        Initialize result cache.

        Args:
            cache_dir: Directory for cached results
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_hash(self, config: Dict) -> str:
        """Compute a hash for a configuration."""
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def get(self, config: Dict, name: str = "result") -> Optional[Any]:
        """
        Get a cached result.

        Args:
            config: Configuration that generated the result
            name: Name of the cached item

        Returns:
            Cached result or None if not found
        """
        cache_hash = self._compute_hash(config)
        cache_path = self.cache_dir / f"{name}_{cache_hash}.pkl"

        if cache_path.exists():
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        return None

    def set(self, config: Dict, result: Any, name: str = "result") -> None:
        """
        Cache a result.

        Args:
            config: Configuration that generated the result
            result: Result to cache
            name: Name of the cached item
        """
        cache_hash = self._compute_hash(config)
        cache_path = self.cache_dir / f"{name}_{cache_hash}.pkl"

        with open(cache_path, "wb") as f:
            pickle.dump(result, f)

    def clear(self) -> None:
        """Clear all cached results."""
        for path in self.cache_dir.glob("*.pkl"):
            path.unlink()


def cached(cache: ResultCache, name: str = "result"):
    """
    Decorator for caching function results.

    Args:
        cache: ResultCache instance
        name: Cache name prefix
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create config from arguments
            config = {"args": args, "kwargs": kwargs, "func": func.__name__}

            # Check cache
            result = cache.get(config, name)
            if result is not None:
                return result

            # Compute and cache
            result = func(*args, **kwargs)
            cache.set(config, result, name)

            return result

        return wrapper

    return decorator


class ProgressBar:
    """
    Wrapper for tqdm with consistent styling.
    """

    def __init__(
        self,
        iterable=None,
        total: Optional[int] = None,
        desc: str = "",
        unit: str = "it",
        disable: bool = False,
        leave: bool = True,
    ):
        """
        Initialize progress bar.

        Args:
            iterable: Iterable to wrap
            total: Total number of iterations
            desc: Description
            unit: Unit name
            disable: Whether to disable the progress bar
            leave: Whether to leave the bar on screen after completion
        """
        self.pbar = tqdm(
            iterable,
            total=total,
            desc=desc,
            unit=unit,
            disable=disable,
            leave=leave,
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

    def __iter__(self):
        return iter(self.pbar)

    def __enter__(self):
        return self.pbar.__enter__()

    def __exit__(self, *args):
        return self.pbar.__exit__(*args)

    def update(self, n: int = 1) -> None:
        """Update progress bar."""
        self.pbar.update(n)

    def set_description(self, desc: str) -> None:
        """Set description."""
        self.pbar.set_description(desc)

    def set_postfix(self, **kwargs) -> None:
        """Set postfix text."""
        self.pbar.set_postfix(**kwargs)


def create_progress_bar(
    iterable=None,
    total: Optional[int] = None,
    desc: str = "",
    unit: str = "it",
    disable: bool = False,
) -> ProgressBar:
    """
    Create a progress bar.

    Args:
        iterable: Iterable to wrap
        total: Total number of iterations
        desc: Description
        unit: Unit name
        disable: Whether to disable

    Returns:
        ProgressBar instance
    """
    return ProgressBar(iterable, total, desc, unit, disable)


# Convenience functions for quick logging
_default_logger = None


def get_logger(name: str = "dko") -> logging.Logger:
    """Get or create the default logger."""
    global _default_logger
    if _default_logger is None:
        _default_logger = logging.getLogger(name)
        if not _default_logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            _default_logger.addHandler(handler)
            _default_logger.setLevel(logging.INFO)
    return _default_logger


def log_info(message: str) -> None:
    """Log an info message."""
    get_logger().info(message)


def log_warning(message: str) -> None:
    """Log a warning message."""
    get_logger().warning(message)


def log_error(message: str) -> None:
    """Log an error message."""
    get_logger().error(message)
