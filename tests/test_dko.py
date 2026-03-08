"""
Comprehensive tests for DKO (Distribution Kernel Operators) model.

Tests cover:
- Basic forward pass
- PCA fitting and reduction
- PSD constraint enforcement
- First-order ablation
- Gradient flow
- Classification task
- Model creation utilities
- Edge cases
"""

import pytest
import torch
import numpy as np

from dko.models.dko import (
    DKO,
    DKOFirstOrder,
    DKOFull,
    DKONoPSD,
    MLP,
    create_dko_model,
    create_dko_first_order,
)


class TestMLP:
    """Tests for the MLP helper class."""

    def test_basic_forward(self):
        """Test basic MLP forward pass."""
        mlp = MLP(input_dim=32, hidden_dims=[64, 32], output_dim=16)
        x = torch.randn(8, 32)
        output = mlp(x)

        assert output.shape == (8, 16)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_no_hidden_layers(self):
        """Test MLP with no hidden layers."""
        mlp = MLP(input_dim=32, hidden_dims=[], output_dim=16)
        x = torch.randn(8, 32)
        output = mlp(x)

        assert output.shape == (8, 16)

    def test_different_activations(self):
        """Test different activation functions."""
        for activation in ['relu', 'gelu', 'silu', 'tanh', 'leaky_relu']:
            mlp = MLP(input_dim=32, hidden_dims=[64], output_dim=16, activation=activation)
            x = torch.randn(8, 32)
            output = mlp(x)
            assert output.shape == (8, 16)

    def test_batch_norm(self):
        """Test MLP with and without batch normalization."""
        mlp_bn = MLP(input_dim=32, hidden_dims=[64, 32], output_dim=16, use_batch_norm=True)
        mlp_no_bn = MLP(input_dim=32, hidden_dims=[64, 32], output_dim=16, use_batch_norm=False)

        x = torch.randn(8, 32)
        output_bn = mlp_bn(x)
        output_no_bn = mlp_no_bn(x)

        assert output_bn.shape == output_no_bn.shape


class TestDKOForwardPass:
    """Tests for DKO forward pass."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        batch_size = 16
        D = 50
        mu = torch.randn(batch_size, D)
        sigma_raw = torch.randn(batch_size, D, D)
        sigma = torch.bmm(sigma_raw, sigma_raw.transpose(1, 2))  # Make PSD
        return mu, sigma, D

    def test_basic_forward_pass(self, sample_data):
        """Test basic forward pass."""
        mu, sigma, D = sample_data
        model = DKO(feature_dim=D, output_dim=1, verbose=False)

        model.train()
        output = model(mu, sigma, fit_pca=True)

        assert output.shape == (16, 1)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_without_pca_fit(self, sample_data):
        """Test forward pass after PCA is already fitted."""
        mu, sigma, D = sample_data
        model = DKO(feature_dim=D, output_dim=1, verbose=False)

        # Fit PCA
        model.train()
        model(mu, sigma, fit_pca=True)

        # Forward without fitting PCA
        output = model(mu, sigma, fit_pca=False)

        assert output.shape == (16, 1)

    def test_batch_sizes(self, sample_data):
        """Test different batch sizes."""
        _, _, D = sample_data
        model = DKO(feature_dim=D, output_dim=1, verbose=False)

        # First train with larger batch to fit PCA
        mu_init = torch.randn(32, D)
        sigma_init = torch.randn(32, D, D)
        sigma_init = torch.bmm(sigma_init, sigma_init.transpose(1, 2))
        model.train()
        model(mu_init, sigma_init, fit_pca=True)

        # Now test different batch sizes in eval mode (BatchNorm works with batch_size=1 in eval)
        model.eval()
        for batch_size in [1, 8, 32, 64]:
            mu = torch.randn(batch_size, D)
            sigma_raw = torch.randn(batch_size, D, D)
            sigma = torch.bmm(sigma_raw, sigma_raw.transpose(1, 2))

            output = model(mu, sigma, fit_pca=False)
            assert output.shape == (batch_size, 1)

    def test_multi_output(self, sample_data):
        """Test model with multiple outputs."""
        mu, sigma, D = sample_data
        model = DKO(feature_dim=D, output_dim=5, verbose=False)

        model.train()
        output = model(mu, sigma, fit_pca=True)

        assert output.shape == (16, 5)


class TestPCAFitting:
    """Tests for PCA fitting and reduction."""

    def test_pca_fitting(self):
        """Test that PCA is fitted correctly."""
        D = 50
        model = DKO(feature_dim=D, pca_variance=0.90, verbose=False)

        mu = torch.randn(32, D)
        sigma_raw = torch.randn(32, D, D)
        sigma = torch.bmm(sigma_raw, sigma_raw.transpose(1, 2))

        assert not model.pca_fitted

        model.train()
        model(mu, sigma, fit_pca=True)

        assert model.pca_fitted
        assert model.reduced_dim is not None
        # Upper triangle of DxD matrix has D*(D+1)/2 elements
        assert model.reduced_dim <= D * (D + 1) // 2

    def test_pca_variance_retention(self):
        """Test PCA retains specified variance."""
        D = 50
        model = DKO(feature_dim=D, pca_variance=0.95, verbose=False)

        mu = torch.randn(64, D)
        sigma_raw = torch.randn(64, D, D)
        sigma = torch.bmm(sigma_raw, sigma_raw.transpose(1, 2))

        model.train()
        model(mu, sigma, fit_pca=True)

        if model.pca is not None:
            explained = model.pca.explained_variance_ratio_.sum()
            assert explained >= 0.90  # Allow some tolerance

    def test_pca_max_components(self):
        """Test PCA max components limit."""
        D = 50
        max_comp = 100
        model = DKO(feature_dim=D, pca_max_components=max_comp, verbose=False)

        mu = torch.randn(32, D)
        sigma_raw = torch.randn(32, D, D)
        sigma = torch.bmm(sigma_raw, sigma_raw.transpose(1, 2))

        model.train()
        model(mu, sigma, fit_pca=True)

        assert model.reduced_dim <= max_comp


class TestPSDConstraint:
    """Tests for PSD (positive semi-definite) constraint."""

    def test_psd_constraint_enforced(self):
        """Test that PSD constraint produces positive semi-definite kernel."""
        D = 30
        model = DKO(feature_dim=D, use_psd_constraint=True, verbose=False)

        mu = torch.randn(8, D)
        sigma_raw = torch.randn(8, D, D)
        sigma = torch.bmm(sigma_raw, sigma_raw.transpose(1, 2))

        model.train()
        model(mu, sigma, fit_pca=True)

        # Get kernel matrix
        K = model.get_kernel_matrix(mu, sigma)

        # Check eigenvalues are non-negative (with numerical tolerance)
        for i in range(8):
            eigenvalues = torch.linalg.eigvalsh(K[i])
            # Use 1e-5 tolerance for numerical stability
            assert (eigenvalues >= -1e-5).all(), f"Kernel {i} is not PSD, min eigenvalue: {eigenvalues.min()}"

    def test_without_psd_constraint(self):
        """Test model without PSD constraint."""
        D = 30
        model = DKONoPSD(feature_dim=D, verbose=False)

        mu = torch.randn(8, D)

        output = model(mu, sigma=None)

        assert output.shape == (8, 1)
        # Should not raise error about kernel matrix
        with pytest.raises(ValueError):
            model.get_kernel_matrix(mu)


class TestFirstOrderAblation:
    """Tests for first-order only ablation."""

    def test_first_order_only(self):
        """Test first-order only variant (DKOFirstOrder)."""
        D = 50
        model = DKOFirstOrder(feature_dim=D, output_dim=1, verbose=False)

        mu = torch.randn(16, D)

        output = model(mu, sigma=None)

        assert output.shape == (16, 1)
        assert not model.use_second_order

    def test_first_order_ignores_sigma(self):
        """Test that first-order model ignores sigma input."""
        D = 50
        model = DKOFirstOrder(feature_dim=D, output_dim=1, verbose=False)

        mu = torch.randn(16, D)
        sigma_raw = torch.randn(16, D, D)
        sigma = torch.bmm(sigma_raw, sigma_raw.transpose(1, 2))

        # Use eval mode to disable dropout for deterministic comparison
        model.eval()

        # Should produce same output with or without sigma
        output_without = model(mu, sigma=None)
        output_with = model(mu, sigma=sigma)

        # Outputs should be identical
        torch.testing.assert_close(output_without, output_with)

    def test_dko_full_uses_sigma(self):
        """Test that DKOFull model uses sigma."""
        D = 30
        model = DKOFull(feature_dim=D, output_dim=1, verbose=False)

        mu = torch.randn(16, D)
        sigma_raw = torch.randn(16, D, D)
        sigma = torch.bmm(sigma_raw, sigma_raw.transpose(1, 2))

        assert model.use_second_order
        model.train()
        model(mu, sigma, fit_pca=True)
        assert model.pca_fitted


class TestGradientFlow:
    """Tests for gradient flow through the model."""

    def test_gradients_flow_to_input(self):
        """Test that gradients flow back to input."""
        D = 30
        model = DKO(feature_dim=D, output_dim=1, verbose=False)

        mu = torch.randn(16, D, requires_grad=True)
        sigma_raw = torch.randn(16, D, D)
        sigma = torch.bmm(sigma_raw, sigma_raw.transpose(1, 2))

        model.train()
        output = model(mu, sigma, fit_pca=True)
        loss = output.mean()
        loss.backward()

        assert mu.grad is not None
        assert mu.grad.abs().sum() > 0

    def test_gradients_flow_to_parameters(self):
        """Test that gradients flow to model parameters."""
        D = 30
        model = DKO(feature_dim=D, output_dim=1, verbose=False)

        mu = torch.randn(16, D)
        sigma_raw = torch.randn(16, D, D)
        sigma = torch.bmm(sigma_raw, sigma_raw.transpose(1, 2))

        model.train()
        output = model(mu, sigma, fit_pca=True)
        loss = output.mean()
        loss.backward()

        # Check that at least some parameters have gradients
        params_with_grad = sum(
            1 for p in model.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert params_with_grad > 0

    def test_backward_pass_no_nan(self):
        """Test that backward pass doesn't produce NaN gradients."""
        D = 30
        model = DKO(feature_dim=D, output_dim=1, verbose=False)

        mu = torch.randn(16, D, requires_grad=True)
        sigma_raw = torch.randn(16, D, D)
        sigma = torch.bmm(sigma_raw, sigma_raw.transpose(1, 2))

        model.train()
        output = model(mu, sigma, fit_pca=True)
        loss = output.sum()
        loss.backward()

        assert not torch.isnan(mu.grad).any()

        for p in model.parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any()


class TestClassificationTask:
    """Tests for classification task."""

    def test_classification_output_shape(self):
        """Test classification output shape."""
        D = 30
        model = DKO(feature_dim=D, output_dim=2, task='classification', verbose=False)

        mu = torch.randn(16, D)
        sigma_raw = torch.randn(16, D, D)
        sigma = torch.bmm(sigma_raw, sigma_raw.transpose(1, 2))

        model.train()
        output_train = model(mu, sigma, fit_pca=True)
        assert output_train.shape == (16, 2)

    def test_classification_sigmoid_eval(self):
        """Test that classification applies sigmoid in eval mode."""
        D = 30
        model = DKO(feature_dim=D, output_dim=2, task='classification', verbose=False)

        mu = torch.randn(16, D)
        sigma_raw = torch.randn(16, D, D)
        sigma = torch.bmm(sigma_raw, sigma_raw.transpose(1, 2))

        model.train()
        model(mu, sigma, fit_pca=True)

        model.eval()
        output_eval = model(mu, sigma)

        # Sigmoid output should be in [0, 1]
        assert (output_eval >= 0).all() and (output_eval <= 1).all()

    def test_classification_no_sigmoid_train(self):
        """Test that classification doesn't apply sigmoid in train mode."""
        D = 30
        model = DKO(feature_dim=D, output_dim=2, task='classification', verbose=False)

        mu = torch.randn(16, D)
        sigma_raw = torch.randn(16, D, D)
        sigma = torch.bmm(sigma_raw, sigma_raw.transpose(1, 2))

        model.train()
        output_train = model(mu, sigma, fit_pca=True)

        # Output can be outside [0, 1] in train mode
        # Just check it's valid (not NaN)
        assert not torch.isnan(output_train).any()


class TestModelCreation:
    """Tests for model creation utilities."""

    def test_create_dko_model(self):
        """Test model creation from config."""
        config = {
            'output_dim': 1,
            'task': 'regression',
            'pca_variance': 0.95,
            'kernel_hidden_dims': [256, 128, 64],
            'dropout': 0.2,
            'verbose': False,
        }

        model = create_dko_model(config, feature_dim=50)

        assert model.feature_dim == 50
        assert model.output_dim == 1
        assert model.task == 'regression'
        assert model.kernel_hidden_dims == [256, 128, 64]

    def test_create_dko_first_order(self):
        """Test first-order model creation."""
        config = {
            'output_dim': 1,
            'task': 'regression',
            'verbose': False,
        }

        model = create_dko_first_order(config, feature_dim=50)

        assert model.feature_dim == 50
        assert not model.use_second_order

    def test_create_model_default_config(self):
        """Test model creation with default config."""
        config = {}
        model = create_dko_model(config, feature_dim=50)

        assert model.feature_dim == 50
        assert model.output_dim == 1
        assert model.task == 'regression'


class TestEmbedding:
    """Tests for embedding extraction."""

    def test_get_embedding(self):
        """Test embedding extraction."""
        D = 30
        model = DKO(feature_dim=D, kernel_output_dim=64, verbose=False)

        mu = torch.randn(16, D)
        sigma_raw = torch.randn(16, D, D)
        sigma = torch.bmm(sigma_raw, sigma_raw.transpose(1, 2))

        model.train()
        model(mu, sigma, fit_pca=True)

        embedding = model.get_embedding(mu, sigma)

        assert embedding.shape == (16, 64)
        assert not torch.isnan(embedding).any()

    def test_get_kernel_matrix(self):
        """Test kernel matrix extraction."""
        D = 30
        k_dim = 32
        model = DKO(feature_dim=D, kernel_output_dim=k_dim, verbose=False)

        mu = torch.randn(8, D)
        sigma_raw = torch.randn(8, D, D)
        sigma = torch.bmm(sigma_raw, sigma_raw.transpose(1, 2))

        model.train()
        model(mu, sigma, fit_pca=True)

        K = model.get_kernel_matrix(mu, sigma)

        assert K.shape == (8, k_dim, k_dim)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_sample(self):
        """Test with single sample."""
        D = 30
        # First-order model without batch norm to support batch_size=1
        model_first = DKOFirstOrder(
            feature_dim=D,
            use_batch_norm=False,  # Disable batch norm for single sample
            verbose=False
        )

        mu = torch.randn(1, D)

        output = model_first(mu, sigma=None)

        assert output.shape == (1, 1)

    def test_small_feature_dim(self):
        """Test with small feature dimension."""
        D = 5
        model = DKO(feature_dim=D, verbose=False)

        mu = torch.randn(16, D)
        sigma_raw = torch.randn(16, D, D)
        sigma = torch.bmm(sigma_raw, sigma_raw.transpose(1, 2))

        model.train()
        output = model(mu, sigma, fit_pca=True)

        assert output.shape == (16, 1)

    def test_large_feature_dim(self):
        """Test with larger feature dimension."""
        D = 200
        model = DKO(feature_dim=D, pca_max_components=500, verbose=False)

        mu = torch.randn(32, D)
        sigma_raw = torch.randn(32, D, D)
        sigma = torch.bmm(sigma_raw, sigma_raw.transpose(1, 2))

        model.train()
        output = model(mu, sigma, fit_pca=True)

        assert output.shape == (32, 1)

    def test_deterministic_output(self):
        """Test that model produces deterministic output in eval mode."""
        D = 30
        model = DKO(feature_dim=D, verbose=False)

        mu = torch.randn(16, D)
        sigma_raw = torch.randn(16, D, D)
        sigma = torch.bmm(sigma_raw, sigma_raw.transpose(1, 2))

        model.train()
        model(mu, sigma, fit_pca=True)

        model.eval()
        output1 = model(mu, sigma)
        output2 = model(mu, sigma)

        torch.testing.assert_close(output1, output2)


class TestModelSaveLoad:
    """Tests for model saving and loading."""

    def test_state_dict(self):
        """Test that model state dict can be saved."""
        D = 30
        model = DKO(feature_dim=D, verbose=False)

        mu = torch.randn(16, D)
        sigma_raw = torch.randn(16, D, D)
        sigma = torch.bmm(sigma_raw, sigma_raw.transpose(1, 2))

        model.train()
        model(mu, sigma, fit_pca=True)

        state_dict = model.state_dict()

        # Check that PCA parameters are in state dict
        assert 'pca_mean_' in state_dict or model.pca_mean_ is None

    def test_load_state_dict(self):
        """Test that model can be loaded from state dict."""
        D = 30

        # Create and initialize model
        model1 = DKO(feature_dim=D, verbose=False)
        mu = torch.randn(16, D)
        sigma_raw = torch.randn(16, D, D)
        sigma = torch.bmm(sigma_raw, sigma_raw.transpose(1, 2))

        model1.train()
        model1(mu, sigma, fit_pca=True)

        # Save state dict
        state_dict = model1.state_dict()

        # Create new model and load
        model2 = DKO(feature_dim=D, verbose=False)
        # Need to initialize kernel layers and PCA state first
        model2._build_kernel_network(model1.first_order_dim + model1.reduced_dim)
        model2.pca_fitted = True
        model2.reduced_dim = model1.reduced_dim
        model2.second_order_dim = model1.second_order_dim
        model2.pca = model1.pca

        # Load state dict with strict=False to handle buffer mismatches
        model2.load_state_dict(state_dict, strict=False)

        # Both in eval mode for deterministic comparison
        model1.eval()
        model2.eval()

        output1 = model1(mu, sigma)
        output2 = model2(mu, sigma)

        # Outputs should be very close
        torch.testing.assert_close(output1, output2, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
