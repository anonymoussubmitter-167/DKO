"""
Enhanced Trainer with comprehensive HPC-grade logging.

Features:
- Complete experiment documentation
- Multi-checkpoint strategy (best, last, periodic)
- Full reproducibility tracking
- HPC-specific monitoring (SLURM, GPU memory)
- Automatic resumption from any checkpoint
- Detailed debugging logs
- Git state tracking
"""

import os
import sys
import json
import socket
import subprocess
import hashlib
import time
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

# Optional imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None

from dko.training.trainer import EarlyStopping


class ExperimentLogger:
    """
    Comprehensive experiment logging for HPC training.

    Logs everything needed for complete reproducibility:
    - Environment (hardware, software, SLURM)
    - Configuration (hyperparameters, model settings)
    - Metrics (loss, gradients, memory usage)
    - Checkpoints with metadata
    """

    def __init__(
        self,
        log_dir: Union[str, Path],
        experiment_name: str,
        verbose: bool = True,
    ):
        """
        Initialize experiment logger.

        Args:
            log_dir: Root directory for all experiments
            experiment_name: Unique experiment identifier
            verbose: Whether to print log messages
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.experiment_dir = self.log_dir / experiment_name
        self.verbose = verbose

        # Create directory structure
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.experiment_dir / 'logs').mkdir(exist_ok=True)
        (self.experiment_dir / 'metrics').mkdir(exist_ok=True)
        (self.experiment_dir / 'plots').mkdir(exist_ok=True)

        # Open log file
        self.log_file = open(self.experiment_dir / 'logs' / 'training.log', 'a')

        # Metrics storage
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_time': [],
            'gpu_memory_gb': [],
            'gradient_norm': [],
        }

        # Log experiment start
        self.log_message("=" * 80)
        self.log_message(f"EXPERIMENT: {experiment_name}")
        self.log_message(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_message(f"Log directory: {self.experiment_dir}")
        self.log_message("=" * 80)

    def log_message(self, message: str, level: str = 'INFO'):
        """Log a message with timestamp."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_line = f"[{timestamp}] [{level}] {message}"

        if self.verbose:
            print(log_line)

        self.log_file.write(log_line + '\n')
        self.log_file.flush()

    def log_environment(self) -> Dict[str, Any]:
        """Log complete environment information."""
        env_info = {
            'timestamp': datetime.now().isoformat(),
            'hostname': socket.gethostname(),
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'cpu_count': os.cpu_count(),
        }

        # Memory info
        if PSUTIL_AVAILABLE:
            env_info['total_memory_gb'] = psutil.virtual_memory().total / (1024**3)
            env_info['available_memory_gb'] = psutil.virtual_memory().available / (1024**3)

        # GPU details
        if torch.cuda.is_available():
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_info.append({
                    'id': i,
                    'name': torch.cuda.get_device_name(i),
                    'memory_total_gb': props.total_memory / (1024**3),
                    'compute_capability': f"{props.major}.{props.minor}",
                    'multi_processor_count': props.multi_processor_count,
                })
            env_info['gpus'] = gpu_info

        # SLURM info if available
        slurm_vars = [
            'SLURM_JOB_ID', 'SLURM_JOB_NAME', 'SLURM_NODELIST',
            'SLURM_NTASKS', 'SLURM_CPUS_PER_TASK', 'SLURM_MEM_PER_NODE',
            'SLURM_GPUS', 'SLURM_ARRAY_TASK_ID', 'SLURM_ARRAY_JOB_ID',
        ]
        slurm_info = {}
        for var in slurm_vars:
            if var in os.environ:
                slurm_info[var] = os.environ[var]
        if slurm_info:
            env_info['slurm'] = slurm_info

        # Git info
        env_info['git'] = self._get_git_info()

        # Save environment info
        with open(self.experiment_dir / 'environment.json', 'w') as f:
            json.dump(env_info, f, indent=2)

        # Log summary
        self.log_message("Environment logged:")
        self.log_message(f"  Hostname: {env_info['hostname']}")
        self.log_message(f"  PyTorch: {env_info['pytorch_version']}")
        self.log_message(f"  CUDA: {env_info.get('cuda_version', 'N/A')}")
        self.log_message(f"  GPUs: {env_info['gpu_count']}")

        if env_info.get('git'):
            git = env_info['git']
            self.log_message(f"  Git: {git.get('commit_hash', 'N/A')[:8]} ({git.get('branch', 'N/A')})")
            if git.get('dirty'):
                self.log_message("  WARNING: Git repository has uncommitted changes!", level='WARNING')

        if env_info.get('slurm'):
            self.log_message(f"  SLURM Job ID: {slurm_info.get('SLURM_JOB_ID', 'N/A')}")

        return env_info

    def _get_git_info(self) -> Optional[Dict[str, Any]]:
        """Get git repository information."""
        try:
            git_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode('ascii').strip()

            git_branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode('ascii').strip()

            git_status = subprocess.check_output(
                ['git', 'status', '--porcelain'],
                stderr=subprocess.DEVNULL
            ).decode('ascii').strip()

            return {
                'commit_hash': git_hash,
                'branch': git_branch,
                'dirty': len(git_status) > 0,
            }
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def log_config(self, config: Dict[str, Any]):
        """Log complete configuration."""
        # Save as JSON
        with open(self.experiment_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        # Save as YAML if available
        if YAML_AVAILABLE:
            with open(self.experiment_dir / 'config.yaml', 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

        # Create config hash for reproducibility
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

        self.log_message("Configuration logged:")
        self.log_message(f"  Config hash: {config_hash}")
        for key, value in config.items():
            if isinstance(value, dict):
                self.log_message(f"  {key}:")
                for k, v in value.items():
                    self.log_message(f"    {k}: {v}")
            else:
                self.log_message(f"  {key}: {value}")

    def log_model_architecture(self, model: nn.Module) -> Dict[str, Any]:
        """Log model architecture and parameter count."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        model_info = {
            'model_class': model.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
        }

        # Layer-wise parameter count
        layer_params = {}
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                n_params = sum(p.numel() for p in module.parameters())
                if n_params > 0:
                    layer_params[name] = n_params
        model_info['layer_parameters'] = layer_params

        # Save model architecture
        with open(self.experiment_dir / 'model_architecture.txt', 'w') as f:
            f.write(f"Model: {model.__class__.__name__}\n")
            f.write(f"Total parameters: {total_params:,}\n")
            f.write(f"Trainable parameters: {trainable_params:,}\n")
            f.write("\n" + "=" * 60 + "\n\n")
            f.write(str(model))
            f.write("\n\n" + "=" * 60 + "\n")
            f.write("\nLayer-wise parameters:\n")
            for name, count in layer_params.items():
                f.write(f"  {name}: {count:,}\n")

        with open(self.experiment_dir / 'model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)

        self.log_message(f"Model: {model.__class__.__name__}")
        self.log_message(f"  Total parameters: {total_params:,}")
        self.log_message(f"  Trainable parameters: {trainable_params:,}")

        return model_info

    def log_dataset_info(
        self,
        train_size: int,
        val_size: int,
        test_size: Optional[int] = None,
        batch_size: int = 32,
        additional_info: Optional[Dict] = None,
    ):
        """Log dataset split information."""
        dataset_info = {
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'total_size': train_size + val_size + (test_size or 0),
            'batch_size': batch_size,
            'train_batches': (train_size + batch_size - 1) // batch_size,
            'val_batches': (val_size + batch_size - 1) // batch_size,
        }

        if additional_info:
            dataset_info.update(additional_info)

        with open(self.experiment_dir / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)

        self.log_message(f"Dataset: train={train_size}, val={val_size}, test={test_size or 'N/A'}")
        self.log_message(f"  Batch size: {batch_size}")

    def log_epoch_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        learning_rate: float,
        epoch_time: float,
        additional_metrics: Optional[Dict[str, float]] = None,
    ):
        """Log metrics for an epoch."""
        # Store in history
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['val_loss'].append(val_loss)
        self.metrics_history['learning_rate'].append(learning_rate)
        self.metrics_history['epoch_time'].append(epoch_time)

        # GPU memory if available
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / (1024**3)
            self.metrics_history['gpu_memory_gb'].append(gpu_memory)
            torch.cuda.reset_peak_memory_stats()

        # Build metrics dict
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': learning_rate,
            'epoch_time': epoch_time,
        }

        if additional_metrics:
            metrics.update(additional_metrics)
            for key, value in additional_metrics.items():
                if key not in self.metrics_history:
                    self.metrics_history[key] = []
                self.metrics_history[key].append(value)

        # Append to CSV
        csv_file = self.experiment_dir / 'metrics' / 'training_metrics.csv'
        write_header = not csv_file.exists()

        with open(csv_file, 'a') as f:
            if write_header:
                f.write(','.join(metrics.keys()) + '\n')
            f.write(','.join(str(v) for v in metrics.values()) + '\n')

        # Save full history as JSON
        with open(self.experiment_dir / 'metrics' / 'training_history.json', 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

    def log_gradient_stats(self, model: nn.Module) -> Dict[str, float]:
        """Log gradient statistics."""
        total_norm = 0.0
        param_norms = []

        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                param_norms.append(param_norm)
                total_norm += param_norm ** 2

        total_norm = total_norm ** 0.5

        self.metrics_history['gradient_norm'].append(total_norm)

        return {
            'total_norm': total_norm,
            'mean_norm': np.mean(param_norms) if param_norms else 0.0,
            'max_norm': np.max(param_norms) if param_norms else 0.0,
            'min_norm': np.min(param_norms) if param_norms else 0.0,
        }

    def save_checkpoint_metadata(self, checkpoint_path: Path, metadata: Dict[str, Any]):
        """Save metadata alongside checkpoint."""
        metadata_path = checkpoint_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def create_experiment_manifest(self) -> Dict[str, Any]:
        """Create a complete experiment manifest for reproducibility."""
        manifest = {
            'experiment_name': self.experiment_name,
            'created_at': datetime.now().isoformat(),
            'files': {},
        }

        # List all files in experiment directory
        for file_path in self.experiment_dir.rglob('*'):
            if file_path.is_file():
                rel_path = file_path.relative_to(self.experiment_dir)
                manifest['files'][str(rel_path)] = {
                    'size_bytes': file_path.stat().st_size,
                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                }

        with open(self.experiment_dir / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)

        return manifest

    def close(self):
        """Close log file and finalize experiment."""
        # Create final manifest
        self.create_experiment_manifest()

        self.log_message("=" * 80)
        self.log_message(f"Experiment completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_message("=" * 80)
        self.log_file.close()


class EnhancedTrainer:
    """
    Enhanced Trainer with comprehensive HPC logging.

    Features:
    - Complete experiment documentation
    - Multi-checkpoint strategy (best, last, periodic)
    - Full reproducibility tracking
    - Resumption from any checkpoint
    - Detailed monitoring and debugging
    - SLURM-aware logging
    """

    def __init__(
        self,
        model: nn.Module,
        task: str = 'regression',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        max_epochs: int = 300,
        early_stopping_patience: int = 30,
        gradient_clip_max_norm: float = 1.0,
        device: Optional[str] = None,
        use_mixed_precision: bool = True,
        experiment_dir: Optional[Union[str, Path]] = None,
        experiment_name: Optional[str] = None,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_config: Optional[Dict] = None,
        save_every_n_epochs: int = 10,
        config: Optional[Dict[str, Any]] = None,
        resume_from: Optional[Union[str, Path]] = None,
        verbose: bool = True,
    ):
        """
        Initialize enhanced trainer.

        Args:
            model: PyTorch model to train
            task: 'regression' or 'classification'
            learning_rate: Base learning rate (default: 1e-4)
            weight_decay: Weight decay for AdamW (default: 1e-5)
            max_epochs: Maximum number of epochs (default: 300)
            early_stopping_patience: Patience for early stopping (default: 30)
            gradient_clip_max_norm: Max norm for gradient clipping (default: 1.0)
            device: Device to train on (auto-detect if None)
            use_mixed_precision: Whether to use FP16 training
            experiment_dir: Root directory for experiments
            experiment_name: Unique experiment name
            use_wandb: Whether to use W&B logging
            wandb_project: W&B project name
            wandb_config: Additional W&B config
            save_every_n_epochs: Save checkpoint every N epochs
            config: Full experiment configuration
            resume_from: Path to checkpoint to resume from
            verbose: Whether to print progress
        """
        # Auto-detect device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = model.to(device)
        self.task = task
        self.device = device
        self.max_epochs = max_epochs
        self.gradient_clip_max_norm = gradient_clip_max_norm
        self.use_mixed_precision = use_mixed_precision and device == 'cuda'
        self.save_every_n_epochs = save_every_n_epochs
        self.verbose = verbose

        # Generate experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_name = f"{model.__class__.__name__}_{task}_{timestamp}"

        # Initialize logger
        self.logger = ExperimentLogger(
            log_dir=experiment_dir or Path('./experiments'),
            experiment_name=experiment_name,
            verbose=verbose,
        )

        # Log environment
        self.logger.log_environment()

        # Build and log configuration
        if config is None:
            config = {}

        self.config = {
            'model': model.__class__.__name__,
            'task': task,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'max_epochs': max_epochs,
            'early_stopping_patience': early_stopping_patience,
            'gradient_clip_max_norm': gradient_clip_max_norm,
            'use_mixed_precision': use_mixed_precision,
            'device': device,
            **config,
        }
        self.logger.log_config(self.config)

        # Log model architecture
        self.logger.log_model_architecture(model)

        # Optimizer: AdamW as specified in research plan
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Scheduler: Cosine annealing to eta_min=1e-6
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max_epochs,
            eta_min=1e-6
        )

        # Loss function
        if task == 'regression':
            self.criterion = nn.MSELoss()
        elif task == 'classification':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown task: {task}. Use 'regression' or 'classification'")

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            mode='min'
        )

        # Mixed precision scaler
        self.scaler = GradScaler('cuda') if self.use_mixed_precision else None

        # W&B logging
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_initialized = False
        if self.use_wandb:
            try:
                full_config = {**self.config}
                if wandb_config:
                    full_config.update(wandb_config)

                wandb.init(
                    project=wandb_project or "dko-hpc",
                    name=experiment_name,
                    config=full_config,
                    dir=str(self.logger.experiment_dir),
                    reinit=True,
                )
                self.wandb_initialized = True
                self.logger.log_message("W&B initialized successfully")
            except Exception as e:
                self.logger.log_message(f"Failed to initialize W&B: {e}", level='WARNING')
                self.use_wandb = False

        # Training state
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.current_epoch = 0

        # Resume from checkpoint if provided
        if resume_from:
            self.load_checkpoint(resume_from)
            self.logger.log_message(f"Resumed from checkpoint: {resume_from}")

    def _get_batch_data(self, batch: Dict) -> tuple:
        """Extract data from batch, handling both DKO and baseline formats."""
        labels = batch.get('label', batch.get('labels'))
        if labels is not None:
            labels = labels.to(self.device)

        # Check for DKO format (mu, sigma)
        if 'mu' in batch and 'sigma' in batch:
            return {
                'format': 'dko',
                'mu': batch['mu'].to(self.device),
                'sigma': batch['sigma'].to(self.device),
            }, labels

        # Check for baseline format with conformer features
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

        raise ValueError("Unknown batch format. Expected 'mu'/'sigma' or 'features'")

    def _forward_pass(self, data: Dict, fit_pca: bool = False) -> torch.Tensor:
        """Perform forward pass for any model format."""
        if data['format'] == 'dko':
            outputs = self.model(
                data['mu'],
                data['sigma'],
                fit_pca=fit_pca
            )
        else:
            features = data['features']
            mask = data.get('mask')
            weights = data.get('weights')

            if weights is not None:
                outputs = self.model(features, weights, mask=mask)
            elif mask is not None:
                outputs = self.model(features, mask=mask)
            else:
                outputs = self.model(features)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

        return outputs

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch with detailed logging."""
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        # Only fit PCA on first batch of first epoch (fresh training)
        fit_pca = (epoch == 0 and self.start_epoch == 0)

        # Progress bar
        if self.verbose and TQDM_AVAILABLE:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.max_epochs}")
        else:
            pbar = train_loader

        for batch_idx, batch in enumerate(pbar):
            data, labels = self._get_batch_data(batch)
            do_fit_pca = fit_pca and batch_idx == 0

            self.optimizer.zero_grad()

            if self.use_mixed_precision:
                with autocast('cuda'):
                    outputs = self._forward_pass(data, fit_pca=do_fit_pca)
                    loss = self.criterion(outputs.squeeze(), labels.squeeze())

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_max_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self._forward_pass(data, fit_pca=do_fit_pca)
                loss = self.criterion(outputs.squeeze(), labels.squeeze())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_max_norm
                )
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if self.verbose and TQDM_AVAILABLE:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Log gradient statistics
        grad_stats = self.logger.log_gradient_stats(self.model)

        avg_loss = total_loss / max(num_batches, 1)
        return {
            'loss': avg_loss,
            'gradient_norm': grad_stats['total_norm'],
        }

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate with detailed metrics."""
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            data, labels = self._get_batch_data(batch)
            outputs = self._forward_pass(data, fit_pca=False)
            loss = self.criterion(outputs.squeeze(), labels.squeeze())

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return {'loss': avg_loss}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, Any]:
        """Full training loop with comprehensive logging."""
        # Log dataset info
        batch_size = train_loader.batch_size or 32
        self.logger.log_dataset_info(
            train_size=len(train_loader.dataset),
            val_size=len(val_loader.dataset),
            batch_size=batch_size,
        )

        self.logger.log_message(f"\nStarting training from epoch {self.start_epoch + 1}")
        self.logger.log_message(f"Device: {self.device}")
        self.logger.log_message(f"Mixed precision: {self.use_mixed_precision}")

        # Reset early stopping if resuming
        if self.start_epoch > 0:
            self.early_stopping.reset()

        for epoch in range(self.start_epoch, self.max_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = self.validate(val_loader)

            # Update scheduler
            self.scheduler.step()

            # Metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start

            # Log metrics
            self.logger.log_epoch_metrics(
                epoch=epoch + 1,
                train_loss=train_metrics['loss'],
                val_loss=val_metrics['loss'],
                learning_rate=current_lr,
                epoch_time=epoch_time,
                additional_metrics={'gradient_norm': train_metrics['gradient_norm']},
            )

            self.logger.log_message(
                f"Epoch {epoch + 1}/{self.max_epochs} ({epoch_time:.1f}s) - "
                f"train_loss: {train_metrics['loss']:.4f}, "
                f"val_loss: {val_metrics['loss']:.4f}, "
                f"lr: {current_lr:.2e}, "
                f"grad_norm: {train_metrics['gradient_norm']:.4f}"
            )

            # W&B logging
            if self.use_wandb and self.wandb_initialized:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'learning_rate': current_lr,
                    'gradient_norm': train_metrics['gradient_norm'],
                    'epoch_time': epoch_time,
                })

            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_epoch = epoch + 1
                self.save_checkpoint('best_model.pt', epoch, val_metrics['loss'], is_best=True)
                self.logger.log_message(f"  -> New best model (val_loss: {self.best_val_loss:.4f})")

            # Save periodic checkpoint
            if (epoch + 1) % self.save_every_n_epochs == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt', epoch, val_metrics['loss'])
                self.logger.log_message(f"  -> Saved checkpoint at epoch {epoch + 1}")

            # Always save last checkpoint
            self.save_checkpoint('last_model.pt', epoch, val_metrics['loss'])

            # Early stopping
            if self.early_stopping(val_metrics['loss']):
                self.logger.log_message(f"\nEarly stopping at epoch {epoch + 1}")
                self.logger.log_message(f"Best epoch: {self.best_epoch}, Best val_loss: {self.best_val_loss:.4f}")
                break

        # Load best model
        best_checkpoint = self.logger.experiment_dir / 'checkpoints' / 'best_model.pt'
        if best_checkpoint.exists():
            self.load_checkpoint(best_checkpoint)
            self.logger.log_message(f"Loaded best model from epoch {self.best_epoch}")

        # Finalize
        if self.use_wandb and self.wandb_initialized:
            wandb.finish()

        self.logger.close()

        return self.logger.metrics_history

    def save_checkpoint(
        self,
        filename: str,
        epoch: int,
        val_loss: float,
        is_best: bool = False,
    ):
        """Save checkpoint with complete metadata."""
        checkpoint_path = self.logger.experiment_dir / 'checkpoints' / filename

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'config': self.config,
            'early_stopping_counter': self.early_stopping.counter,
            'early_stopping_best_score': self.early_stopping.best_score,
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)

        # Save metadata
        metadata = {
            'epoch': epoch + 1,
            'val_loss': val_loss,
            'is_best': is_best,
            'timestamp': datetime.now().isoformat(),
            'checkpoint_file': filename,
        }
        self.logger.save_checkpoint_metadata(checkpoint_path, metadata)

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> bool:
        """Load checkpoint and resume training."""
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            self.logger.log_message(f"Checkpoint not found: {checkpoint_path}", level='WARNING')
            return False

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_epoch = checkpoint['best_epoch']
        self.start_epoch = checkpoint['epoch']

        # Restore early stopping state
        if 'early_stopping_counter' in checkpoint:
            self.early_stopping.counter = checkpoint['early_stopping_counter']
            self.early_stopping.best_score = checkpoint['early_stopping_best_score']

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.logger.log_message(f"Loaded checkpoint from epoch {self.start_epoch}")

        return True


def create_enhanced_trainer(
    model: nn.Module,
    config: Dict[str, Any],
    **kwargs,
) -> EnhancedTrainer:
    """
    Factory function to create an EnhancedTrainer from config.

    Args:
        model: Model to train
        config: Configuration dictionary
        **kwargs: Additional arguments to override config

    Returns:
        Configured EnhancedTrainer instance
    """
    training_config = config.get('training', config)

    return EnhancedTrainer(
        model=model,
        task=kwargs.get('task', training_config.get('task', 'regression')),
        learning_rate=kwargs.get('learning_rate', training_config.get('base_learning_rate', 1e-4)),
        weight_decay=kwargs.get('weight_decay', training_config.get('weight_decay', 1e-5)),
        max_epochs=kwargs.get('max_epochs', training_config.get('max_epochs', 300)),
        early_stopping_patience=kwargs.get(
            'early_stopping_patience',
            training_config.get('early_stopping_patience', 30)
        ),
        gradient_clip_max_norm=kwargs.get(
            'gradient_clip_max_norm',
            training_config.get('gradient_clip_max_norm', 1.0)
        ),
        device=kwargs.get('device', training_config.get('device')),
        use_mixed_precision=kwargs.get(
            'use_mixed_precision',
            training_config.get('mixed_precision', True)
        ),
        experiment_dir=kwargs.get('experiment_dir', training_config.get('experiment_dir', './experiments')),
        experiment_name=kwargs.get('experiment_name', training_config.get('experiment_name')),
        use_wandb=kwargs.get('use_wandb', training_config.get('use_wandb', False)),
        wandb_project=kwargs.get('wandb_project', training_config.get('wandb_project')),
        save_every_n_epochs=kwargs.get(
            'save_every_n_epochs',
            training_config.get('save_every_n_epochs', 10)
        ),
        config=config,
        resume_from=kwargs.get('resume_from', training_config.get('resume_from')),
        verbose=kwargs.get('verbose', training_config.get('verbose', True)),
    )


if __name__ == "__main__":
    # Test enhanced trainer
    print("Testing EnhancedTrainer...")

    import tempfile
    import shutil
    from torch.utils.data import TensorDataset

    # Create temp directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Create dummy model
        model = nn.Sequential(
            nn.Linear(50, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Wrapper to handle DKO format
        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, mu, sigma=None, fit_pca=False):
                return self.model(mu)

        wrapped_model = ModelWrapper(model)

        # Create dummy data
        def collate_fn(batch):
            x, y = zip(*batch)
            return {
                'mu': torch.stack(x),
                'sigma': torch.zeros(len(x), 50, 50),
                'label': torch.stack(y),
            }

        X_train = torch.randn(64, 50)
        y_train = torch.randn(64, 1)
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=collate_fn)

        X_val = torch.randn(32, 50)
        y_val = torch.randn(32, 1)
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=16, collate_fn=collate_fn)

        # Create trainer
        trainer = EnhancedTrainer(
            wrapped_model,
            device='cpu',
            max_epochs=3,
            early_stopping_patience=10,
            experiment_dir=temp_dir,
            experiment_name='test_experiment',
            use_mixed_precision=False,
            use_wandb=False,
            verbose=True,
        )

        # Train
        history = trainer.fit(train_loader, val_loader)

        print("\n[OK] EnhancedTrainer test passed!")
        print(f"  Epochs completed: {len(history['train_loss'])}")
        print(f"  Best val loss: {trainer.best_val_loss:.4f} at epoch {trainer.best_epoch}")
        print(f"  Experiment dir: {trainer.logger.experiment_dir}")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
