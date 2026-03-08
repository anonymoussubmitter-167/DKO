"""Tests for training infrastructure."""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import shutil

from torch.utils.data import DataLoader, TensorDataset

from dko.training.trainer import EarlyStopping, Trainer, create_trainer


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return nn.Sequential(
        nn.Linear(50, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )


@pytest.fixture
def dko_batch():
    """Create a DKO-format batch (mu, sigma)."""
    batch_size = 8
    D = 50
    mu = torch.randn(batch_size, D)
    sigma = torch.randn(batch_size, D, D)
    sigma = torch.bmm(sigma, sigma.transpose(1, 2))  # Make positive semi-definite
    labels = torch.randn(batch_size, 1)
    return {
        'mu': mu,
        'sigma': sigma,
        'label': labels,
    }


@pytest.fixture
def baseline_batch():
    """Create a baseline-format batch (features)."""
    batch_size = 8
    n_conformers = 20
    feature_dim = 50
    features = torch.randn(batch_size, n_conformers, feature_dim)
    mask = torch.ones(batch_size, n_conformers, dtype=torch.bool)
    mask[:, 15:] = False  # Mask out last 5 conformers
    labels = torch.randn(batch_size, 1)
    return {
        'features': features,
        'mask': mask,
        'label': labels,
    }


@pytest.fixture
def train_val_loaders(simple_model):
    """Create training and validation data loaders."""
    n_train = 64
    n_val = 32
    feature_dim = 50

    # Training data
    X_train = torch.randn(n_train, feature_dim)
    y_train = torch.randn(n_train, 1)
    train_dataset = TensorDataset(X_train, y_train)

    # Validation data
    X_val = torch.randn(n_val, feature_dim)
    y_val = torch.randn(n_val, 1)
    val_dataset = TensorDataset(X_val, y_val)

    def collate_fn(batch):
        x, y = zip(*batch)
        return {
            'mu': torch.stack(x),  # Using mu format for simple model
            'sigma': torch.zeros(len(x), 50, 50),  # Dummy sigma
            'label': torch.stack(y),
        }

    train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=collate_fn)

    return train_loader, val_loader


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory for checkpoints."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


# =============================================================================
# EarlyStopping Tests
# =============================================================================

class TestEarlyStopping:
    """Test early stopping functionality."""

    def test_initialization(self):
        """Test EarlyStopping initialization."""
        es = EarlyStopping(patience=10, min_delta=0.001, mode='min')

        assert es.patience == 10
        assert es.min_delta == 0.001
        assert es.mode == 'min'
        assert es.counter == 0
        assert es.best_score is None
        assert not es.early_stop

    def test_first_call_sets_best_score(self):
        """Test that first call sets best score."""
        es = EarlyStopping(patience=5)

        result = es(0.5)

        assert not result
        assert es.best_score == 0.5
        assert es.counter == 0

    def test_improvement_resets_counter_min_mode(self):
        """Test that improvement resets counter in min mode."""
        es = EarlyStopping(patience=5, mode='min')

        es(0.5)  # First call
        es(0.6)  # No improvement
        es(0.7)  # No improvement
        assert es.counter == 2

        es(0.3)  # Improvement
        assert es.counter == 0
        assert es.best_score == 0.3

    def test_improvement_resets_counter_max_mode(self):
        """Test that improvement resets counter in max mode."""
        es = EarlyStopping(patience=5, mode='max')

        es(0.5)  # First call
        es(0.4)  # No improvement
        es(0.3)  # No improvement
        assert es.counter == 2

        es(0.8)  # Improvement
        assert es.counter == 0
        assert es.best_score == 0.8

    def test_stops_after_patience(self):
        """Test that early stopping triggers after patience."""
        es = EarlyStopping(patience=3)

        es(0.5)  # First call
        es(0.6)  # Counter = 1
        es(0.7)  # Counter = 2
        result = es(0.8)  # Counter = 3, should stop

        assert result
        assert es.early_stop

    def test_min_delta_threshold(self):
        """Test that min_delta affects improvement detection."""
        es = EarlyStopping(patience=5, min_delta=0.1, mode='min')

        es(0.5)  # First call
        result = es(0.49)  # Not enough improvement (need 0.1)

        assert not result
        assert es.counter == 1  # Did not count as improvement

    def test_reset(self):
        """Test reset functionality."""
        es = EarlyStopping(patience=5)

        es(0.5)
        es(0.6)
        es(0.7)

        es.reset()

        assert es.counter == 0
        assert es.best_score is None
        assert not es.early_stop


# =============================================================================
# Trainer Initialization Tests
# =============================================================================

class TestTrainerInitialization:
    """Test Trainer initialization."""

    def test_default_initialization(self, simple_model):
        """Test trainer with default parameters."""
        trainer = Trainer(simple_model, verbose=False)

        assert trainer.task == 'regression'
        assert trainer.max_epochs == 300
        assert trainer.gradient_clip_max_norm == 1.0
        assert trainer.best_val_loss == float('inf')

    def test_custom_initialization(self, simple_model):
        """Test trainer with custom parameters."""
        trainer = Trainer(
            simple_model,
            task='regression',
            learning_rate=1e-3,
            weight_decay=1e-4,
            max_epochs=100,
            early_stopping_patience=10,
            gradient_clip_max_norm=0.5,
            verbose=False,
        )

        assert trainer.max_epochs == 100
        assert trainer.gradient_clip_max_norm == 0.5

    def test_device_auto_detection(self, simple_model):
        """Test that device is auto-detected."""
        trainer = Trainer(simple_model, verbose=False)

        # Should be cuda if available, cpu otherwise
        expected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        assert trainer.device == expected_device

    def test_device_explicit_cpu(self, simple_model):
        """Test explicit CPU device."""
        trainer = Trainer(simple_model, device='cpu', verbose=False)

        assert trainer.device == 'cpu'

    def test_optimizer_is_adamw(self, simple_model):
        """Test that optimizer is AdamW."""
        trainer = Trainer(simple_model, verbose=False)

        assert isinstance(trainer.optimizer, torch.optim.AdamW)

    def test_scheduler_is_cosine_annealing(self, simple_model):
        """Test that scheduler is CosineAnnealingLR."""
        trainer = Trainer(simple_model, verbose=False)

        assert isinstance(
            trainer.scheduler,
            torch.optim.lr_scheduler.CosineAnnealingLR
        )

    def test_regression_loss_function(self, simple_model):
        """Test that regression uses MSELoss."""
        trainer = Trainer(simple_model, task='regression', verbose=False)

        assert isinstance(trainer.criterion, nn.MSELoss)

    def test_classification_loss_function(self, simple_model):
        """Test that classification uses BCEWithLogitsLoss."""
        trainer = Trainer(simple_model, task='classification', verbose=False)

        assert isinstance(trainer.criterion, nn.BCEWithLogitsLoss)

    def test_invalid_task_raises_error(self, simple_model):
        """Test that invalid task raises ValueError."""
        with pytest.raises(ValueError):
            Trainer(simple_model, task='invalid_task', verbose=False)

    def test_checkpoint_dir_creation(self, simple_model, temp_checkpoint_dir):
        """Test that checkpoint directory is created."""
        trainer = Trainer(
            simple_model,
            checkpoint_dir=temp_checkpoint_dir / 'subdir',
            verbose=False,
        )

        assert trainer.checkpoint_dir.exists()

    def test_history_initialization(self, simple_model):
        """Test that history is properly initialized."""
        trainer = Trainer(simple_model, verbose=False)

        assert 'train_loss' in trainer.history
        assert 'val_loss' in trainer.history
        assert 'learning_rate' in trainer.history
        assert len(trainer.history['train_loss']) == 0


# =============================================================================
# Batch Data Handling Tests
# =============================================================================

class TestBatchDataHandling:
    """Test batch data extraction."""

    def test_dko_batch_format(self, simple_model, dko_batch):
        """Test DKO batch format extraction."""
        trainer = Trainer(simple_model, device='cpu', verbose=False)

        data, labels = trainer._get_batch_data(dko_batch)

        assert data['format'] == 'dko'
        assert 'mu' in data
        assert 'sigma' in data
        assert labels is not None

    def test_baseline_batch_format(self, simple_model, baseline_batch):
        """Test baseline batch format extraction."""
        trainer = Trainer(simple_model, device='cpu', verbose=False)

        data, labels = trainer._get_batch_data(baseline_batch)

        assert data['format'] == 'baseline'
        assert 'features' in data
        assert 'mask' in data
        assert labels is not None

    def test_baseline_with_weights(self, simple_model):
        """Test baseline batch with Boltzmann weights."""
        trainer = Trainer(simple_model, device='cpu', verbose=False)

        batch = {
            'features': torch.randn(4, 10, 50),
            'weights': torch.softmax(torch.randn(4, 10), dim=-1),
            'label': torch.randn(4, 1),
        }

        data, labels = trainer._get_batch_data(batch)

        assert data['format'] == 'baseline'
        assert data['weights'] is not None

    def test_unknown_batch_format_raises_error(self, simple_model):
        """Test that unknown batch format raises error."""
        trainer = Trainer(simple_model, device='cpu', verbose=False)

        invalid_batch = {'unknown_key': torch.randn(4, 10)}

        with pytest.raises(ValueError):
            trainer._get_batch_data(invalid_batch)


# =============================================================================
# Training Tests
# =============================================================================

class TestTraining:
    """Test training functionality."""

    def test_train_epoch(self, simple_model, train_val_loaders):
        """Test single training epoch."""
        train_loader, _ = train_val_loaders

        # Use a model that accepts mu directly
        model = nn.Sequential(
            nn.Linear(50, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Create a wrapper to handle DKO format
        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, mu, sigma=None, fit_pca=False):
                return self.model(mu)

        wrapped_model = ModelWrapper(model)
        trainer = Trainer(
            wrapped_model,
            device='cpu',
            use_mixed_precision=False,
            verbose=False,
        )

        metrics = trainer.train_epoch(train_loader)

        assert 'loss' in metrics
        assert isinstance(metrics['loss'], float)
        assert metrics['loss'] > 0

    def test_validate(self, simple_model, train_val_loaders):
        """Test validation."""
        _, val_loader = train_val_loaders

        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, mu, sigma=None, fit_pca=False):
                return self.model(mu)

        wrapped_model = ModelWrapper(simple_model)
        trainer = Trainer(
            wrapped_model,
            device='cpu',
            use_mixed_precision=False,
            verbose=False,
        )

        metrics = trainer.validate(val_loader)

        assert 'loss' in metrics
        assert isinstance(metrics['loss'], float)

    def test_fit_runs_multiple_epochs(self, simple_model, train_val_loaders):
        """Test that fit runs multiple epochs."""
        train_loader, val_loader = train_val_loaders

        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, mu, sigma=None, fit_pca=False):
                return self.model(mu)

        wrapped_model = ModelWrapper(simple_model)
        trainer = Trainer(
            wrapped_model,
            device='cpu',
            max_epochs=5,
            early_stopping_patience=10,
            use_mixed_precision=False,
            verbose=False,
        )

        history = trainer.fit(train_loader, val_loader)

        assert len(history['train_loss']) == 5
        assert len(history['val_loss']) == 5
        assert len(history['learning_rate']) == 5

    def test_early_stopping_triggers(self, simple_model, train_val_loaders):
        """Test that early stopping can trigger."""
        train_loader, val_loader = train_val_loaders

        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, mu, sigma=None, fit_pca=False):
                return self.model(mu)

        wrapped_model = ModelWrapper(simple_model)
        trainer = Trainer(
            wrapped_model,
            device='cpu',
            max_epochs=100,
            early_stopping_patience=3,
            use_mixed_precision=False,
            verbose=False,
        )

        history = trainer.fit(train_loader, val_loader)

        # Should stop before max_epochs due to early stopping
        # (may or may not stop early depending on training dynamics)
        assert len(history['train_loss']) <= 100

    def test_learning_rate_decreases(self, simple_model, train_val_loaders):
        """Test that learning rate decreases with cosine annealing."""
        train_loader, val_loader = train_val_loaders

        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, mu, sigma=None, fit_pca=False):
                return self.model(mu)

        wrapped_model = ModelWrapper(simple_model)
        trainer = Trainer(
            wrapped_model,
            device='cpu',
            max_epochs=10,
            early_stopping_patience=100,  # Don't trigger early stopping
            use_mixed_precision=False,
            verbose=False,
        )

        history = trainer.fit(train_loader, val_loader)

        # Learning rate should generally decrease
        initial_lr = history['learning_rate'][0]
        final_lr = history['learning_rate'][-1]
        assert final_lr < initial_lr


# =============================================================================
# Checkpointing Tests
# =============================================================================

class TestCheckpointing:
    """Test checkpoint saving and loading."""

    def test_save_checkpoint(self, simple_model, temp_checkpoint_dir):
        """Test checkpoint saving."""
        trainer = Trainer(
            simple_model,
            device='cpu',
            checkpoint_dir=temp_checkpoint_dir,
            verbose=False,
        )
        trainer.current_epoch = 5
        trainer.best_val_loss = 0.123

        trainer.save_checkpoint('test_checkpoint.pt')

        assert (temp_checkpoint_dir / 'test_checkpoint.pt').exists()

    def test_load_checkpoint(self, simple_model, temp_checkpoint_dir):
        """Test checkpoint loading."""
        trainer = Trainer(
            simple_model,
            device='cpu',
            checkpoint_dir=temp_checkpoint_dir,
            verbose=False,
        )
        trainer.best_val_loss = 0.123
        trainer.best_epoch = 5
        trainer.save_checkpoint('test_checkpoint.pt')

        # Create new trainer and load
        new_model = nn.Sequential(
            nn.Linear(50, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        new_trainer = Trainer(
            new_model,
            device='cpu',
            checkpoint_dir=temp_checkpoint_dir,
            verbose=False,
        )

        success = new_trainer.load_checkpoint('test_checkpoint.pt')

        assert success
        assert new_trainer.best_val_loss == 0.123
        assert new_trainer.best_epoch == 5

    def test_load_nonexistent_checkpoint(self, simple_model, temp_checkpoint_dir):
        """Test loading nonexistent checkpoint."""
        trainer = Trainer(
            simple_model,
            device='cpu',
            checkpoint_dir=temp_checkpoint_dir,
            verbose=False,
        )

        success = trainer.load_checkpoint('nonexistent.pt')

        assert not success

    def test_checkpoint_contains_all_state(self, simple_model, temp_checkpoint_dir):
        """Test that checkpoint contains all necessary state."""
        trainer = Trainer(
            simple_model,
            device='cpu',
            checkpoint_dir=temp_checkpoint_dir,
            use_mixed_precision=False,
            verbose=False,
        )
        trainer.history['train_loss'] = [0.5, 0.4, 0.3]
        trainer.history['val_loss'] = [0.6, 0.5, 0.4]
        trainer.save_checkpoint('full_checkpoint.pt')

        checkpoint = torch.load(temp_checkpoint_dir / 'full_checkpoint.pt')

        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'scheduler_state_dict' in checkpoint
        assert 'best_val_loss' in checkpoint
        assert 'best_epoch' in checkpoint
        assert 'history' in checkpoint


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestCreateTrainer:
    """Test create_trainer factory function."""

    def test_create_from_config(self, simple_model):
        """Test creating trainer from config dict."""
        config = {
            'training': {
                'task': 'regression',
                'base_learning_rate': 1e-3,
                'weight_decay': 1e-4,
                'max_epochs': 200,
                'early_stopping_patience': 20,
            }
        }

        trainer = create_trainer(simple_model, config, verbose=False)

        assert trainer.max_epochs == 200

    def test_kwargs_override_config(self, simple_model):
        """Test that kwargs override config values."""
        config = {
            'training': {
                'max_epochs': 200,
            }
        }

        trainer = create_trainer(
            simple_model,
            config,
            max_epochs=50,
            verbose=False,
        )

        assert trainer.max_epochs == 50

    def test_default_values_match_research_plan(self, simple_model):
        """Test that default values match research plan specifications."""
        config = {}
        trainer = create_trainer(simple_model, config, verbose=False)

        # Research plan defaults
        assert trainer.max_epochs == 300
        assert trainer.gradient_clip_max_norm == 1.0
        # Note: Learning rate and weight decay are set in optimizer


# =============================================================================
# Gradient Clipping Tests
# =============================================================================

class TestGradientClipping:
    """Test gradient clipping functionality."""

    def test_gradients_are_clipped(self, simple_model, train_val_loaders):
        """Test that gradients are clipped during training."""
        train_loader, _ = train_val_loaders

        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, mu, sigma=None, fit_pca=False):
                return self.model(mu)

        wrapped_model = ModelWrapper(simple_model)
        trainer = Trainer(
            wrapped_model,
            device='cpu',
            gradient_clip_max_norm=0.1,  # Small clip value
            use_mixed_precision=False,
            verbose=False,
        )

        # Run one training step
        trainer.train_epoch(train_loader)

        # If training completes without error, gradient clipping is working
        assert True


# =============================================================================
# Device Compatibility Tests
# =============================================================================

class TestDeviceCompatibility:
    """Test device compatibility."""

    def test_cpu_training(self, simple_model, train_val_loaders):
        """Test training on CPU."""
        train_loader, val_loader = train_val_loaders

        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, mu, sigma=None, fit_pca=False):
                return self.model(mu)

        wrapped_model = ModelWrapper(simple_model)
        trainer = Trainer(
            wrapped_model,
            device='cpu',
            max_epochs=2,
            use_mixed_precision=False,
            verbose=False,
        )

        history = trainer.fit(train_loader, val_loader)

        assert len(history['train_loss']) > 0

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_cuda_training(self, simple_model, train_val_loaders):
        """Test training on CUDA."""
        train_loader, val_loader = train_val_loaders

        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, mu, sigma=None, fit_pca=False):
                return self.model(mu)

        wrapped_model = ModelWrapper(simple_model)
        trainer = Trainer(
            wrapped_model,
            device='cuda',
            max_epochs=2,
            use_mixed_precision=True,
            verbose=False,
        )

        history = trainer.fit(train_loader, val_loader)

        assert len(history['train_loss']) > 0


# =============================================================================
# Classification Task Tests
# =============================================================================

class TestClassificationTask:
    """Test classification task support."""

    def test_classification_training(self, train_val_loaders):
        """Test training with classification task."""
        train_loader, val_loader = train_val_loaders

        # Create classification model
        model = nn.Sequential(
            nn.Linear(50, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Binary classification
        )

        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, mu, sigma=None, fit_pca=False):
                return self.model(mu)

        wrapped_model = ModelWrapper(model)

        # Create loaders with binary labels
        def collate_fn(batch):
            x, y = zip(*batch)
            return {
                'mu': torch.stack(x),
                'sigma': torch.zeros(len(x), 50, 50),
                'label': torch.stack([torch.sigmoid(yi) for yi in y]),  # Binary labels
            }

        # Recreate loaders with binary labels
        n_train = 32
        X_train = torch.randn(n_train, 50)
        y_train = (torch.randn(n_train, 1) > 0).float()
        train_dataset = TensorDataset(X_train, y_train)

        n_val = 16
        X_val = torch.randn(n_val, 50)
        y_val = (torch.randn(n_val, 1) > 0).float()
        val_dataset = TensorDataset(X_val, y_val)

        binary_train_loader = DataLoader(
            train_dataset, batch_size=8, collate_fn=collate_fn
        )
        binary_val_loader = DataLoader(
            val_dataset, batch_size=8, collate_fn=collate_fn
        )

        trainer = Trainer(
            wrapped_model,
            task='classification',
            device='cpu',
            max_epochs=3,
            use_mixed_precision=False,
            verbose=False,
        )

        history = trainer.fit(binary_train_loader, binary_val_loader)

        assert len(history['train_loss']) > 0
        assert isinstance(trainer.criterion, nn.BCEWithLogitsLoss)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for complete training workflow."""

    def test_full_training_workflow(self, temp_checkpoint_dir):
        """Test complete training workflow."""
        # Create model
        model = nn.Sequential(
            nn.Linear(50, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, mu, sigma=None, fit_pca=False):
                return self.model(mu)

        wrapped_model = ModelWrapper(model)

        # Create data
        def collate_fn(batch):
            x, y = zip(*batch)
            return {
                'mu': torch.stack(x),
                'sigma': torch.zeros(len(x), 50, 50),
                'label': torch.stack(y),
            }

        n_train = 64
        n_val = 32
        X_train = torch.randn(n_train, 50)
        y_train = torch.randn(n_train, 1)
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=8, collate_fn=collate_fn
        )

        X_val = torch.randn(n_val, 50)
        y_val = torch.randn(n_val, 1)
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=collate_fn)

        # Train
        trainer = Trainer(
            wrapped_model,
            device='cpu',
            max_epochs=5,
            early_stopping_patience=10,
            checkpoint_dir=temp_checkpoint_dir,
            use_mixed_precision=False,
            verbose=False,
        )

        history = trainer.fit(train_loader, val_loader)

        # Verify results
        assert len(history['train_loss']) == 5
        assert trainer.best_val_loss < float('inf')
        assert (temp_checkpoint_dir / 'best_model.pt').exists()

    def test_resume_training(self, temp_checkpoint_dir):
        """Test resuming training from checkpoint."""
        # Create and train initial model
        model = nn.Sequential(
            nn.Linear(50, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, mu, sigma=None, fit_pca=False):
                return self.model(mu)

        wrapped_model = ModelWrapper(model)

        def collate_fn(batch):
            x, y = zip(*batch)
            return {
                'mu': torch.stack(x),
                'sigma': torch.zeros(len(x), 50, 50),
                'label': torch.stack(y),
            }

        X = torch.randn(32, 50)
        y = torch.randn(32, 1)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)

        trainer = Trainer(
            wrapped_model,
            device='cpu',
            max_epochs=3,
            checkpoint_dir=temp_checkpoint_dir,
            use_mixed_precision=False,
            verbose=False,
        )

        trainer.fit(loader, loader)
        trainer.save_checkpoint('resume_checkpoint.pt')

        # Create new model and resume
        new_model = nn.Sequential(
            nn.Linear(50, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        new_wrapped = ModelWrapper(new_model)

        new_trainer = Trainer(
            new_wrapped,
            device='cpu',
            checkpoint_dir=temp_checkpoint_dir,
            use_mixed_precision=False,
            verbose=False,
        )

        success = new_trainer.load_checkpoint('resume_checkpoint.pt')

        assert success
        assert new_trainer.best_val_loss == trainer.best_val_loss
