"""
Training infrastructure for DKO and baseline models.

Implements:
- Training loop with early stopping
- AdamW optimizer with cosine annealing
- Mixed precision training (FP16)
- Gradient clipping
- Checkpointing best models
- W&B + local logging

Research plan specifications:
- Optimizer: AdamW with base_lr=1e-4, weight_decay=1e-5
- Scheduler: Cosine annealing from 1e-4 to 1e-6
- Max epochs: 300
- Early stopping: patience=30 on validation loss
- Batch size: 32
- Gradient clipping: max_norm=1.0
- Mixed precision: FP16 enabled
"""

import os
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# Handle different PyTorch versions for mixed precision
try:
    from torch.amp import GradScaler, autocast
    AUTOCAST_DEVICE_ARG = True  # torch.amp.autocast takes device_type arg
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    AUTOCAST_DEVICE_ARG = False  # torch.cuda.amp.autocast doesn't

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None


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


logger = get_logger("trainer")


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.

    Research plan specification: patience=30 epochs
    """

    def __init__(
        self,
        patience: int = 30,
        min_delta: float = 0.0,
        mode: str = 'min',
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy/AUC
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation metric

        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class Trainer:
    """
    Trainer for DKO and baseline models.

    Specifications from research plan:
    - AdamW optimizer (lr=1e-4, weight_decay=1e-5)
    - Cosine annealing to 1e-6
    - Early stopping (patience=30)
    - Gradient clipping (max_norm=1.0)
    - Mixed precision (FP16)
    - Checkpointing
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
        checkpoint_dir: Optional[Union[str, Path]] = None,
        log_dir: Optional[Union[str, Path]] = None,
        checkpoint_every: int = 25,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_config: Optional[Dict] = None,
        verbose: bool = True,
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model to train
            task: 'regression' or 'classification'
            learning_rate: Base learning rate (default: 1e-4)
            weight_decay: Weight decay for AdamW (default: 1e-5)
            max_epochs: Maximum number of epochs (default: 300)
            early_stopping_patience: Patience for early stopping (default: 30)
            gradient_clip_max_norm: Max norm for gradient clipping (default: 1.0)
            device: Device to train on (auto-detect if None)
            use_mixed_precision: Whether to use FP16 training (default: True)
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for log files and CSVs
            checkpoint_every: Save checkpoint every N epochs (default: 25)
            use_wandb: Whether to use W&B logging
            wandb_project: W&B project name
            wandb_run_name: W&B run name
            wandb_config: Additional W&B config
            verbose: Whether to print progress
        """
        # Auto-detect device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = device
        self.task = task

        # Check if model is DKO variant (PCA fitting doesn't work with DataParallel)
        model_class_name = model.__class__.__name__
        is_dko = model_class_name in [
            'DKO', 'DKOFirstOrder', 'DKOFull', 'DKONoPSD',
            'DKOEigenspectrum', 'DKOScalarInvariants', 'DKOLowRank',
            'DKOGatedFusion', 'DKOResidual', 'DKOCrossAttention', 'DKOSCCRouter',
        ]

        # Multi-GPU support with DataParallel (but not for DKO due to PCA)
        if device == 'cuda' and torch.cuda.device_count() > 1 and not is_dko:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = model.to(device)
            self.model = nn.DataParallel(model)
        else:
            if is_dko and torch.cuda.device_count() > 1:
                print(f"DKO model: using single GPU (PCA doesn't support DataParallel)")
            self.model = model.to(device)
        self.max_epochs = max_epochs
        self.gradient_clip_max_norm = gradient_clip_max_norm
        self.use_mixed_precision = use_mixed_precision and device == 'cuda'
        self.verbose = verbose

        # Optimizer: AdamW as specified in research plan
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Scheduler: Cosine annealing to eta_min=1e-6 as specified
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
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
            mode='min'  # Minimize validation loss
        )

        # Mixed precision scaler
        # Note: PyTorch 2.0.x GradScaler doesn't take device arg (first arg is init_scale)
        # PyTorch 2.1+ torch.amp.GradScaler takes device as first arg
        if self.use_mixed_precision:
            if AUTOCAST_DEVICE_ARG:
                # torch.amp.GradScaler (PyTorch 2.1+)
                self.scaler = GradScaler('cuda')
            else:
                # torch.cuda.amp.GradScaler (PyTorch 2.0.x) - no device arg
                self.scaler = GradScaler()
        else:
            self.scaler = None

        # Checkpointing
        self.checkpoint_every = checkpoint_every
        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir = None

        # File logging + CSV
        self.log_dir = None
        self.file_logger = None
        self.batch_csv_path = None
        self.epoch_csv_path = None
        if log_dir:
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)

            # Set up file logger
            self.file_logger = logging.getLogger(f"trainer.{wandb_run_name or 'default'}")
            self.file_logger.setLevel(logging.INFO)
            self.file_logger.handlers.clear()
            fh = logging.FileHandler(self.log_dir / "training.log")
            fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.file_logger.addHandler(fh)
            # Also log to console
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.file_logger.addHandler(ch)

            # Batch-level CSV
            self.batch_csv_path = self.log_dir / "batch_logs.csv"
            with open(self.batch_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'batch', 'loss', 'grad_norm', 'lr', 'timestamp'])

            # Epoch-level CSV
            self.epoch_csv_path = self.log_dir / "epoch_logs.csv"
            with open(self.epoch_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'epoch', 'train_loss', 'val_loss', 'lr', 'epoch_time_s',
                    'val_rmse', 'val_mae', 'val_r2', 'val_pearson',
                    'val_auc', 'val_accuracy', 'val_f1',
                    'is_best', 'early_stop_counter',
                ])

        # W&B logging
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_initialized = False
        if self.use_wandb:
            try:
                config = {
                    'learning_rate': learning_rate,
                    'weight_decay': weight_decay,
                    'max_epochs': max_epochs,
                    'early_stopping_patience': early_stopping_patience,
                    'task': task,
                    'device': device,
                    'mixed_precision': use_mixed_precision,
                }
                if wandb_config:
                    config.update(wandb_config)

                wandb.init(
                    project=wandb_project or "dko-training",
                    name=wandb_run_name,
                    config=config,
                    reinit=True,
                )
                self.wandb_initialized = True
            except Exception as e:
                logger.warning(f"Failed to initialize W&B: {e}")
                self.use_wandb = False

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
        }

        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.current_epoch = 0

    def _get_batch_data(self, batch: Dict) -> Tuple:
        """
        Extract data from batch, handling both DKO and baseline formats.

        Args:
            batch: Batch dictionary

        Returns:
            Tuple of (data_dict, labels) where data_dict contains model inputs
        """
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

            # Check for Boltzmann weights (DeepSets)
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

    def _compute_mu_sigma(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute first-order (mu) and second-order (sigma) features from conformers.

        Args:
            features: (batch, n_conformers, feature_dim) conformer features
            mask: (batch, n_conformers) valid conformer mask
            weights: (batch, n_conformers) Boltzmann weights

        Returns:
            mu: (batch, feature_dim) mean features
            sigma: (batch, feature_dim, feature_dim) covariance features
        """
        batch_size, n_conf, feat_dim = features.shape

        # Check for NaN/Inf in input features
        if torch.isnan(features).any() or torch.isinf(features).any():
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        # Per-sample normalization across conformers only (dim=1).
        # Normalizing across features too (dim=(1,2)) destroys inter-feature variance
        # that sigma is meant to capture. dim=1 preserves feature-wise structure.
        feat_mean = features.mean(dim=1, keepdim=True)
        feat_std = features.std(dim=1, keepdim=True).clamp(min=1e-6)
        features = (features - feat_mean) / feat_std

        # Create mask if not provided
        if mask is None:
            mask = torch.ones(batch_size, n_conf, dtype=torch.bool, device=features.device)

        # Create uniform weights if not provided
        if weights is None:
            # Count valid conformers per molecule
            valid_counts = mask.sum(dim=1, keepdim=True).float().clamp(min=1)
            weights = mask.float() / valid_counts
        else:
            # Normalize weights
            weights = weights * mask.float()
            weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)

        # Compute weighted mean: mu = sum_i w_i * x_i
        weights_expanded = weights.unsqueeze(-1)  # (batch, n_conf, 1)
        mu = (features * weights_expanded).sum(dim=1)  # (batch, feat_dim)

        # Compute weighted covariance: sigma = sum_i w_i * (x_i - mu)(x_i - mu)^T
        centered = features - mu.unsqueeze(1)  # (batch, n_conf, feat_dim)
        centered = centered * mask.unsqueeze(-1).float()  # Zero out invalid conformers

        # Clamp centered values to prevent extreme covariances
        centered = torch.clamp(centered, min=-10.0, max=10.0)

        # Weighted outer product sum
        # sigma[b] = sum_i w[b,i] * centered[b,i] @ centered[b,i].T
        weighted_centered = centered * weights_expanded.sqrt()  # (batch, n_conf, feat_dim)
        sigma = torch.bmm(
            weighted_centered.transpose(1, 2),  # (batch, feat_dim, n_conf)
            weighted_centered  # (batch, n_conf, feat_dim)
        )  # (batch, feat_dim, feat_dim)

        # Add regularization to diagonal for numerical stability.
        # 1e-2 is appropriate for geometric feature scales (~0.1-10.0) to prevent
        # near-singular covariance matrices.
        eye = torch.eye(feat_dim, device=sigma.device, dtype=sigma.dtype)
        sigma = sigma + 1e-2 * eye.unsqueeze(0)

        return mu, sigma

    def _is_dko_model(self) -> bool:
        """Check if the model is a DKO variant that needs mu/sigma input."""
        # Handle DataParallel wrapped models
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        model_class_name = model.__class__.__name__
        return model_class_name in [
            'DKO', 'DKOFirstOrder', 'DKOFull', 'DKONoPSD',
            'DKOEigenspectrum', 'DKOScalarInvariants', 'DKOLowRank',
            'DKOGatedFusion', 'DKOResidual', 'DKOCrossAttention', 'DKOSCCRouter',
        ]

    def _forward_pass(
        self,
        data: Dict,
        fit_pca: bool = False
    ) -> torch.Tensor:
        """
        Perform forward pass for any model format.

        Args:
            data: Data dictionary from _get_batch_data
            fit_pca: Whether to fit PCA (for DKO)

        Returns:
            Model predictions
        """
        if data['format'] == 'dko':
            # DKO model with pre-computed mu, sigma
            outputs = self.model(
                data['mu'],
                data['sigma'],
                fit_pca=fit_pca
            )
        elif self._is_dko_model():
            # DKO model but data has conformer features - compute mu/sigma
            features = data['features']
            mask = data.get('mask')
            weights = data.get('weights')

            mu, sigma = self._compute_mu_sigma(features, mask, weights)
            outputs = self.model(mu, sigma, fit_pca=fit_pca)
        else:
            # Baseline model with conformer features
            features = data['features']
            mask = data.get('mask')
            weights = data.get('weights')

            # Handle different baseline model signatures
            try:
                if weights is not None:
                    outputs = self.model(features, weights, mask=mask)
                elif mask is not None:
                    outputs = self.model(features, mask=mask)
                else:
                    outputs = self.model(features)
            except TypeError:
                # Model doesn't accept mask/weights, try simpler call
                outputs = self.model(features)

            # Handle tuple output (output, attention_info)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

        return outputs

    def train_epoch(
        self,
        train_loader: DataLoader,
        fit_pca: bool = False,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            fit_pca: Whether to fit PCA (first epoch for DKO)

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        # Progress bar
        if self.verbose and TQDM_AVAILABLE:
            pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        else:
            pbar = train_loader

        for batch_idx, batch in enumerate(pbar):
            # Extract data
            data, labels = self._get_batch_data(batch)

            # Fit PCA on first batch of first epoch (for DKO)
            do_fit_pca = fit_pca and batch_idx == 0

            # Forward pass with optional mixed precision
            self.optimizer.zero_grad()

            if self.use_mixed_precision:
                # Handle different PyTorch versions
                ctx = autocast('cuda') if AUTOCAST_DEVICE_ARG else autocast()
                with ctx:
                    outputs = self._forward_pass(data, fit_pca=do_fit_pca)
                    loss = self.criterion(outputs.squeeze(), labels.squeeze())

                # Backward pass with scaling
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

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_max_norm
                )
                self.optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss
            num_batches += 1

            # Compute gradient norm for logging
            grad_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5

            if self.verbose and TQDM_AVAILABLE:
                pbar.set_postfix({'loss': f'{batch_loss:.4f}'})

            # Batch-level CSV logging
            if self.batch_csv_path is not None:
                current_lr = self.optimizer.param_groups[0]['lr']
                with open(self.batch_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        self.current_epoch + 1, batch_idx + 1,
                        f'{batch_loss:.6f}', f'{grad_norm:.6f}',
                        f'{current_lr:.2e}', time.strftime('%Y-%m-%d %H:%M:%S'),
                    ])

        avg_loss = total_loss / max(num_batches, 1)
        return {'loss': avg_loss}

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Validate on validation set.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of validation metrics including loss and task-specific metrics
        """
        self.model.eval()

        total_loss = 0.0
        num_batches = 0
        all_preds = []
        all_labels = []

        for batch in val_loader:
            # Extract data
            data, labels = self._get_batch_data(batch)

            # Forward pass
            outputs = self._forward_pass(data, fit_pca=False)
            loss = self.criterion(outputs.squeeze(), labels.squeeze())

            total_loss += loss.item()
            num_batches += 1

            all_preds.append(outputs.squeeze().cpu().numpy())
            all_labels.append(labels.squeeze().cpu().numpy())

        avg_loss = total_loss / max(num_batches, 1)
        metrics = {'loss': avg_loss}

        # Compute task-specific metrics
        try:
            preds = np.concatenate(all_preds, axis=0).flatten()
            labels_np = np.concatenate(all_labels, axis=0).flatten()

            # Remove NaN
            valid = ~(np.isnan(preds) | np.isnan(labels_np))
            preds = preds[valid]
            labels_np = labels_np[valid]

            if len(preds) > 0:
                if self.task == 'regression':
                    metrics['rmse'] = float(np.sqrt(np.mean((preds - labels_np) ** 2)))
                    metrics['mae'] = float(np.mean(np.abs(preds - labels_np)))
                    ss_res = np.sum((labels_np - preds) ** 2)
                    ss_tot = np.sum((labels_np - np.mean(labels_np)) ** 2)
                    metrics['r2'] = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
                    if len(preds) > 1:
                        from scipy import stats as sp_stats
                        r, _ = sp_stats.pearsonr(preds, labels_np)
                        metrics['pearson'] = float(r)
                elif self.task == 'classification':
                    # Apply sigmoid if logits
                    if preds.min() < 0 or preds.max() > 1:
                        probs = 1 / (1 + np.exp(-np.clip(preds, -500, 500)))
                    else:
                        probs = preds
                    binary_preds = (probs >= 0.5).astype(int)
                    targets_int = labels_np.astype(int)
                    metrics['accuracy'] = float(np.mean(binary_preds == targets_int))
                    tp = np.sum((binary_preds == 1) & (targets_int == 1))
                    fp = np.sum((binary_preds == 1) & (targets_int == 0))
                    fn = np.sum((binary_preds == 0) & (targets_int == 1))
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                    metrics['f1'] = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
                    if len(np.unique(targets_int)) > 1:
                        try:
                            from sklearn.metrics import roc_auc_score
                            metrics['auc'] = float(roc_auc_score(targets_int, probs))
                        except Exception:
                            pass
        except Exception as e:
            if self.file_logger:
                self.file_logger.warning(f"Failed to compute val metrics: {e}")

        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, Any]:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader

        Returns:
            Training history
        """
        n_params = sum(p.numel() for p in self.model.parameters())
        start_msg = (f"Starting training on {self.device} | "
                     f"params: {n_params:,} | max_epochs: {self.max_epochs}")
        if self.verbose:
            print(start_msg)
        if self.file_logger:
            self.file_logger.info(start_msg)

        # Reset early stopping
        self.early_stopping.reset()

        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Train (fit PCA only on first epoch)
            fit_pca = (epoch == 0)
            train_metrics = self.train_epoch(train_loader, fit_pca=fit_pca)

            # Validate (now includes RMSE/MAE/R²/AUC)
            val_metrics = self.validate(val_loader)

            # Update scheduler
            self.scheduler.step()

            # Log metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['learning_rate'].append(current_lr)

            # Store val metrics in history
            for k, v in val_metrics.items():
                if k != 'loss':
                    key = f'val_{k}'
                    if key not in self.history:
                        self.history[key] = []
                    self.history[key].append(v)

            epoch_time = time.time() - epoch_start

            # Build log message with val metrics
            is_best = val_metrics['loss'] < self.best_val_loss
            log_parts = [
                f"Epoch {epoch+1}/{self.max_epochs} ({epoch_time:.1f}s)",
                f"train_loss: {train_metrics['loss']:.4f}",
                f"val_loss: {val_metrics['loss']:.4f}",
            ]
            if 'rmse' in val_metrics:
                log_parts.append(f"val_rmse: {val_metrics['rmse']:.4f}")
            if 'mae' in val_metrics:
                log_parts.append(f"val_mae: {val_metrics['mae']:.4f}")
            if 'r2' in val_metrics:
                log_parts.append(f"val_r2: {val_metrics['r2']:.4f}")
            if 'pearson' in val_metrics:
                log_parts.append(f"val_pearson: {val_metrics['pearson']:.4f}")
            if 'auc' in val_metrics:
                log_parts.append(f"val_auc: {val_metrics['auc']:.4f}")
            if 'accuracy' in val_metrics:
                log_parts.append(f"val_acc: {val_metrics['accuracy']:.4f}")
            log_parts.append(f"lr: {current_lr:.2e}")
            epoch_msg = " - ".join(log_parts)

            if self.verbose:
                print(epoch_msg)
            if self.file_logger:
                self.file_logger.info(epoch_msg)

            # Epoch-level CSV logging
            if self.epoch_csv_path is not None:
                with open(self.epoch_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch + 1,
                        f'{train_metrics["loss"]:.6f}',
                        f'{val_metrics["loss"]:.6f}',
                        f'{current_lr:.2e}',
                        f'{epoch_time:.1f}',
                        f'{val_metrics.get("rmse", ""):.6f}' if 'rmse' in val_metrics else '',
                        f'{val_metrics.get("mae", ""):.6f}' if 'mae' in val_metrics else '',
                        f'{val_metrics.get("r2", ""):.6f}' if 'r2' in val_metrics else '',
                        f'{val_metrics.get("pearson", ""):.6f}' if 'pearson' in val_metrics else '',
                        f'{val_metrics.get("auc", ""):.6f}' if 'auc' in val_metrics else '',
                        f'{val_metrics.get("accuracy", ""):.6f}' if 'accuracy' in val_metrics else '',
                        f'{val_metrics.get("f1", ""):.6f}' if 'f1' in val_metrics else '',
                        is_best,
                        self.early_stopping.counter,
                    ])

            # W&B logging
            if self.use_wandb and self.wandb_initialized:
                wb_metrics = {
                    'epoch': epoch + 1,
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'learning_rate': current_lr,
                }
                for k, v in val_metrics.items():
                    if k != 'loss':
                        wb_metrics[f'val_{k}'] = v
                wandb.log(wb_metrics)

            # Save best model
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.best_epoch = epoch + 1
                if self.checkpoint_dir:
                    self.save_checkpoint('best_model.pt')
                if self.verbose:
                    print(f"  -> New best model (val_loss: {self.best_val_loss:.4f})")
                if self.file_logger:
                    self.file_logger.info(f"New best model at epoch {epoch+1} (val_loss: {self.best_val_loss:.4f})")

            # Periodic checkpoint every N epochs
            if self.checkpoint_dir and self.checkpoint_every > 0 and (epoch + 1) % self.checkpoint_every == 0:
                self.save_checkpoint(f'checkpoint_epoch{epoch+1}.pt')
                if self.file_logger:
                    self.file_logger.info(f"Saved periodic checkpoint at epoch {epoch+1}")

            # Early stopping
            if self.early_stopping(val_metrics['loss']):
                stop_msg = (f"Early stopping at epoch {epoch+1} | "
                            f"Best epoch: {self.best_epoch}, Best val_loss: {self.best_val_loss:.4f}")
                if self.verbose:
                    print(f"\n{stop_msg}")
                if self.file_logger:
                    self.file_logger.info(stop_msg)
                break

        # Load best model
        if self.checkpoint_dir and (self.checkpoint_dir / 'best_model.pt').exists():
            self.load_checkpoint('best_model.pt')

        # Final summary
        if self.file_logger:
            self.file_logger.info(f"Training complete. Best epoch: {self.best_epoch}, "
                                  f"Best val_loss: {self.best_val_loss:.4f}")

        return self.history

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        if self.checkpoint_dir is None:
            return

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'history': self.history,
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, self.checkpoint_dir / filename)

    def load_checkpoint(self, filename: str) -> bool:
        """
        Load model checkpoint.

        Args:
            filename: Checkpoint filename

        Returns:
            True if loaded successfully
        """
        if self.checkpoint_dir is None:
            return False

        checkpoint_path = self.checkpoint_dir / filename
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint {checkpoint_path} not found")
            return False

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_epoch = checkpoint['best_epoch']
        self.history = checkpoint.get('history', self.history)
        self.current_epoch = checkpoint.get('epoch', 0)

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        if self.verbose:
            print(f"Loaded checkpoint from epoch {self.best_epoch}")

        return True

    def finish(self) -> None:
        """Cleanup after training (close W&B, etc.)."""
        if self.use_wandb and self.wandb_initialized:
            wandb.finish()


def create_trainer(
    model: nn.Module,
    config: Dict[str, Any],
    **kwargs,
) -> Trainer:
    """
    Factory function to create a Trainer from config.

    Args:
        model: Model to train
        config: Configuration dictionary with training parameters
        **kwargs: Additional arguments to override config

    Returns:
        Configured Trainer instance
    """
    # Extract config values with defaults matching research plan
    training_config = config.get('training', config)

    return Trainer(
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
        checkpoint_dir=kwargs.get('checkpoint_dir', training_config.get('checkpoint_dir')),
        log_dir=kwargs.get('log_dir', training_config.get('log_dir')),
        checkpoint_every=kwargs.get('checkpoint_every', training_config.get('checkpoint_every', 25)),
        use_wandb=kwargs.get('use_wandb', training_config.get('use_wandb', False)),
        wandb_project=kwargs.get('wandb_project', training_config.get('wandb_project')),
        wandb_run_name=kwargs.get('wandb_run_name', training_config.get('wandb_run_name')),
        verbose=kwargs.get('verbose', training_config.get('verbose', True)),
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: str = "cuda",
    experiment_name: str = "experiment",
    output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train a model using the Trainer class.

    Convenience wrapper function for training models.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration dictionary
        device: Device to train on
        experiment_name: Name for the experiment
        output_dir: Base output directory for logs/checkpoints

    Returns:
        Tuple of (trained model, training results dict)
    """
    # Set up output directories
    checkpoint_dir = config.get('checkpoint_dir')
    log_dir = config.get('log_dir')

    if output_dir is not None:
        output_dir = Path(output_dir)
        if checkpoint_dir is None:
            checkpoint_dir = output_dir / "checkpoints" / experiment_name
        if log_dir is None:
            log_dir = output_dir / "logs" / experiment_name

    # Create trainer from config
    trainer = Trainer(
        model=model,
        task=config.get('task_type', config.get('task', 'regression')),
        learning_rate=config.get('learning_rate', config.get('base_learning_rate', 1e-4)),
        weight_decay=config.get('weight_decay', 1e-5),
        max_epochs=config.get('max_epochs', 300),
        early_stopping_patience=config.get('early_stopping_patience', 30),
        gradient_clip_max_norm=config.get('gradient_clip_max_norm', 1.0),
        device=device,
        use_mixed_precision=config.get('mixed_precision', True),
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        checkpoint_every=config.get('checkpoint_every', 25),
        use_wandb=config.get('use_wandb', False),
        wandb_project=config.get('wandb_project'),
        wandb_run_name=experiment_name,
        verbose=config.get('verbose', True),
    )

    # Train
    history = trainer.fit(train_loader, val_loader)

    # Prepare results
    results = {
        'history': history,
        'best_epoch': trainer.best_epoch,
        'best_val_loss': trainer.best_val_loss,
        'total_epochs': len(history.get('train_loss', [])),
    }

    return model, results


if __name__ == "__main__":
    # Test trainer
    print("Testing Trainer...")

    from dko.models.dko import DKO
    from torch.utils.data import TensorDataset

    # Create dummy data
    batch_size = 16
    n_samples = 100
    D = 50

    mu_train = torch.randn(n_samples, D)
    sigma_train = torch.randn(n_samples, D, D)
    sigma_train = torch.bmm(sigma_train, sigma_train.transpose(1, 2))
    labels_train = torch.randn(n_samples, 1)

    mu_val = torch.randn(50, D)
    sigma_val = torch.randn(50, D, D)
    sigma_val = torch.bmm(sigma_val, sigma_val.transpose(1, 2))
    labels_val = torch.randn(50, 1)

    # Create datasets
    train_dataset = TensorDataset(mu_train, sigma_train, labels_train)
    val_dataset = TensorDataset(mu_val, sigma_val, labels_val)

    # Custom collate function
    def collate_fn(batch):
        mu, sigma, labels = zip(*batch)
        return {
            'mu': torch.stack(mu),
            'sigma': torch.stack(sigma),
            'label': torch.stack(labels),
        }

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

    # Create model
    model = DKO(feature_dim=D, output_dim=1, verbose=False)

    # Create trainer
    trainer = Trainer(
        model=model,
        task='regression',
        max_epochs=5,
        early_stopping_patience=3,
        use_wandb=False,
        checkpoint_dir=Path('./checkpoints_test'),
        verbose=True,
    )

    # Train
    history = trainer.fit(train_loader, val_loader)

    print("\n[OK] Trainer test passed!")
    print(f"  Training history: {len(history['train_loss'])} epochs")
    print(f"  Best val loss: {trainer.best_val_loss:.4f} at epoch {trainer.best_epoch}")

    # Cleanup test directory
    import shutil
    if Path('./checkpoints_test').exists():
        shutil.rmtree('./checkpoints_test')
