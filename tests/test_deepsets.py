"""Tests for DeepSets models with permutation invariance."""

import pytest
import torch
import torch.nn as nn
import numpy as np

from dko.models.deepsets import (
    DeepSetsBaseline,
    DeepSets,
    DeepSetsWithAttention,
)


class TestDeepSetsBaseline:
    """Test capacity-matched DeepSets baseline."""

    @pytest.fixture
    def model(self):
        """Create baseline model."""
        return DeepSetsBaseline(
            feature_dim=100,
            output_dim=1,
            task='regression',
            encoder_hidden_dims=[256, 256, 128],
            decoder_hidden_dim=128,
            pooling_method='boltzmann_sum',
            dropout=0.1,
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample conformer data."""
        batch_size = 4
        n_conformers = 20
        feature_dim = 100
        x = torch.randn(batch_size, n_conformers, feature_dim)
        # Create Boltzmann weights that sum to 1
        energies = torch.randn(batch_size, n_conformers)
        weights = torch.softmax(-energies, dim=-1)
        return x, weights

    def test_forward_shape(self, model, sample_data):
        """Test forward pass output shape."""
        x, weights = sample_data
        model.eval()
        with torch.no_grad():
            output = model(x, weights)

        assert output.shape == (4, 1)

    def test_forward_without_weights(self, model, sample_data):
        """Test forward pass without Boltzmann weights."""
        x, _ = sample_data
        model.eval()
        with torch.no_grad():
            output = model(x)

        assert output.shape == (4, 1)

    def test_forward_with_mask(self, model, sample_data):
        """Test forward pass with masking."""
        x, weights = sample_data
        mask = torch.ones(4, 20, dtype=torch.bool)
        mask[:, 10:] = False

        model.eval()
        with torch.no_grad():
            output = model(x, weights, mask=mask)

        assert output.shape == (4, 1)

    def test_gradient_flow(self, model, sample_data):
        """Test that gradients flow through the model."""
        x, weights = sample_data
        x.requires_grad = True
        model.train()

        output = model(x, weights)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_different_batch_sizes(self, model):
        """Test with different batch sizes."""
        model.eval()

        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, 20, 100)
            weights = torch.softmax(torch.randn(batch_size, 20), dim=-1)
            with torch.no_grad():
                output = model(x, weights)
            assert output.shape == (batch_size, 1)

    def test_different_conformer_counts(self, model):
        """Test with different numbers of conformers."""
        model.eval()

        for n_conf in [5, 10, 20, 50]:
            x = torch.randn(4, n_conf, 100)
            weights = torch.softmax(torch.randn(4, n_conf), dim=-1)
            with torch.no_grad():
                output = model(x, weights)
            assert output.shape == (4, 1)

    def test_classification_mode(self):
        """Test model in classification mode."""
        model = DeepSetsBaseline(
            feature_dim=100,
            output_dim=2,
            task='classification',
        )
        model.eval()

        x = torch.randn(4, 20, 100)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (4, 2)

    def test_count_parameters(self, model):
        """Test parameter counting."""
        n_params = model.count_parameters()
        assert n_params > 0
        assert n_params < 10_000_000

    def test_create_capacity_matched(self):
        """Test capacity-matched model creation."""
        target_params = 100_000

        model = DeepSetsBaseline.create_capacity_matched(
            feature_dim=100,
            target_params=target_params,
            output_dim=1,
        )

        actual_params = model.count_parameters()
        # Should be within reasonable range of target
        assert actual_params > target_params * 0.3
        assert actual_params < target_params * 2.0


class TestPermutationInvariance:
    """Test permutation invariance property of DeepSets."""

    @pytest.fixture
    def model(self):
        """Create model for permutation tests."""
        model = DeepSetsBaseline(
            feature_dim=50,
            output_dim=1,
            encoder_hidden_dims=[128, 64],
            decoder_hidden_dim=64,
            dropout=0.0,  # No dropout for deterministic tests
        )
        model.eval()
        return model

    def test_permutation_invariance_basic(self, model):
        """Test that output is invariant to conformer permutation."""
        torch.manual_seed(42)
        batch_size = 4
        n_conf = 10
        feature_dim = 50

        x = torch.randn(batch_size, n_conf, feature_dim)
        weights = torch.softmax(torch.randn(batch_size, n_conf), dim=-1)

        with torch.no_grad():
            output1 = model(x, weights)

        # Permute conformers
        perm = torch.randperm(n_conf)
        x_perm = x[:, perm, :]
        weights_perm = weights[:, perm]

        with torch.no_grad():
            output2 = model(x_perm, weights_perm)

        torch.testing.assert_close(output1, output2, atol=1e-5, rtol=1e-5)

    def test_permutation_invariance_multiple_perms(self, model):
        """Test permutation invariance with multiple random permutations."""
        torch.manual_seed(123)
        x = torch.randn(4, 15, 50)
        weights = torch.softmax(torch.randn(4, 15), dim=-1)

        with torch.no_grad():
            reference_output = model(x, weights)

        # Test 5 different permutations
        for _ in range(5):
            perm = torch.randperm(15)
            x_perm = x[:, perm, :]
            weights_perm = weights[:, perm]

            with torch.no_grad():
                perm_output = model(x_perm, weights_perm)

            torch.testing.assert_close(
                reference_output, perm_output, atol=1e-5, rtol=1e-5
            )

    def test_permutation_invariance_per_sample(self, model):
        """Test permutation invariance with different perms per sample."""
        torch.manual_seed(456)
        batch_size = 4
        n_conf = 10
        feature_dim = 50

        x = torch.randn(batch_size, n_conf, feature_dim)
        weights = torch.softmax(torch.randn(batch_size, n_conf), dim=-1)

        with torch.no_grad():
            reference_output = model(x, weights)

        # Apply different permutation to each sample
        x_perm = x.clone()
        weights_perm = weights.clone()
        for i in range(batch_size):
            perm = torch.randperm(n_conf)
            x_perm[i] = x[i, perm]
            weights_perm[i] = weights[i, perm]

        with torch.no_grad():
            perm_output = model(x_perm, weights_perm)

        torch.testing.assert_close(
            reference_output, perm_output, atol=1e-5, rtol=1e-5
        )

    def test_standard_deepsets_permutation_invariance(self):
        """Test permutation invariance of standard DeepSets."""
        model = DeepSets(
            feature_dim=50,
            phi_hidden_dims=[64, 32],
            phi_output_dim=32,
            rho_hidden_dims=[32],
            pooling_method='mean',
            use_batch_norm=False,
            dropout=0.0,
        )
        model.eval()

        x = torch.randn(4, 15, 50)

        with torch.no_grad():
            output1 = model(x)

        perm = torch.randperm(15)
        x_perm = x[:, perm, :]

        with torch.no_grad():
            output2 = model(x_perm)

        torch.testing.assert_close(output1, output2, atol=1e-5, rtol=1e-5)


class TestPoolingMethods:
    """Test different pooling methods."""

    @pytest.fixture
    def feature_dim(self):
        return 50

    def test_sum_pooling(self, feature_dim):
        """Test sum pooling."""
        model = DeepSetsBaseline(
            feature_dim=feature_dim,
            output_dim=1,
            pooling_method='sum',
            dropout=0.0,
        )
        model.eval()

        x = torch.randn(4, 10, feature_dim)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (4, 1)

    def test_mean_pooling(self, feature_dim):
        """Test mean pooling."""
        model = DeepSetsBaseline(
            feature_dim=feature_dim,
            output_dim=1,
            pooling_method='mean',
            dropout=0.0,
        )
        model.eval()

        x = torch.randn(4, 10, feature_dim)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (4, 1)

    def test_max_pooling(self, feature_dim):
        """Test max pooling."""
        model = DeepSetsBaseline(
            feature_dim=feature_dim,
            output_dim=1,
            pooling_method='max',
            dropout=0.0,
        )
        model.eval()

        x = torch.randn(4, 10, feature_dim)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (4, 1)

    def test_boltzmann_sum_pooling(self, feature_dim):
        """Test Boltzmann-weighted sum pooling."""
        model = DeepSetsBaseline(
            feature_dim=feature_dim,
            output_dim=1,
            pooling_method='boltzmann_sum',
            dropout=0.0,
        )
        model.eval()

        x = torch.randn(4, 10, feature_dim)
        weights = torch.softmax(torch.randn(4, 10), dim=-1)

        with torch.no_grad():
            output = model(x, weights)

        assert output.shape == (4, 1)

    def test_boltzmann_weights_affect_output(self, feature_dim):
        """Test that different Boltzmann weights produce different outputs."""
        model = DeepSetsBaseline(
            feature_dim=feature_dim,
            output_dim=1,
            pooling_method='boltzmann_sum',
            dropout=0.0,
        )
        model.eval()

        x = torch.randn(4, 10, feature_dim)

        # Uniform weights
        weights1 = torch.ones(4, 10) / 10

        # Concentrated weights (first conformer)
        weights2 = torch.zeros(4, 10)
        weights2[:, 0] = 1.0

        with torch.no_grad():
            output1 = model(x, weights1)
            output2 = model(x, weights2)

        # Outputs should be different
        assert not torch.allclose(output1, output2)


class TestBoltzmannWeighting:
    """Test Boltzmann weighting functionality."""

    @pytest.fixture
    def model(self):
        """Create model with Boltzmann pooling."""
        return DeepSetsBaseline(
            feature_dim=50,
            output_dim=1,
            pooling_method='boltzmann_sum',
            dropout=0.0,
        )

    def test_weights_normalization(self, model):
        """Test that unnormalized weights are handled correctly."""
        model.eval()
        x = torch.randn(4, 10, 50)

        # Unnormalized weights
        weights = torch.rand(4, 10) * 10  # Not summing to 1

        with torch.no_grad():
            output = model(x, weights)

        assert output.shape == (4, 1)
        assert not torch.any(torch.isnan(output))

    def test_extreme_weights(self, model):
        """Test with extreme weight distributions."""
        model.eval()
        x = torch.randn(4, 10, 50)

        # Very concentrated weights
        weights = torch.zeros(4, 10)
        weights[:, 0] = 0.99
        weights[:, 1] = 0.01

        with torch.no_grad():
            output = model(x, weights)

        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))

    def test_weights_with_mask(self, model):
        """Test Boltzmann weights with masking."""
        model.eval()
        x = torch.randn(4, 10, 50)
        weights = torch.softmax(torch.randn(4, 10), dim=-1)
        mask = torch.ones(4, 10, dtype=torch.bool)
        mask[:, 5:] = False

        with torch.no_grad():
            output = model(x, weights, mask=mask)

        assert output.shape == (4, 1)


class TestDeepSets:
    """Test standard DeepSets implementation."""

    @pytest.fixture
    def model(self):
        """Create DeepSets model."""
        return DeepSets(
            feature_dim=50,
            phi_hidden_dims=[128, 64],
            phi_output_dim=32,
            rho_hidden_dims=[64],
            prediction_hidden_dims=[32],
            num_outputs=1,
            pooling_method='mean',
            dropout=0.1,
        )

    def test_forward_shape(self, model):
        """Test forward pass output shape."""
        model.eval()
        x = torch.randn(4, 20, 50)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (4, 1)

    def test_mean_max_pooling(self):
        """Test mean_max pooling method."""
        model = DeepSets(
            feature_dim=50,
            phi_hidden_dims=[64],
            phi_output_dim=32,
            rho_hidden_dims=[64],
            pooling_method='mean_max',
            use_batch_norm=False,
            dropout=0.0,
        )
        model.eval()

        x = torch.randn(4, 10, 50)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (4, 1)


class TestDeepSetsWithAttention:
    """Test DeepSets with attention pooling."""

    @pytest.fixture
    def model(self):
        """Create DeepSetsWithAttention model."""
        return DeepSetsWithAttention(
            feature_dim=50,
            phi_hidden_dims=[64, 32],
            phi_output_dim=32,
            rho_hidden_dims=[32],
            prediction_hidden_dims=[16],
            num_outputs=1,
            attention_hidden_dim=16,
            dropout=0.1,
        )

    def test_forward_shape(self, model):
        """Test forward pass output shape."""
        model.eval()
        x = torch.randn(4, 20, 50)

        with torch.no_grad():
            output, _ = model(x)

        assert output.shape == (4, 1)

    def test_forward_with_attention(self, model):
        """Test forward with attention extraction."""
        model.eval()
        x = torch.randn(4, 20, 50)

        with torch.no_grad():
            output, attn_weights = model(x, return_attention=True)

        assert output.shape == (4, 1)
        assert attn_weights.shape == (4, 20)

    def test_attention_weights_valid(self, model):
        """Test that attention weights are valid probabilities."""
        model.eval()
        x = torch.randn(4, 20, 50)

        with torch.no_grad():
            _, attn_weights = model(x, return_attention=True)

        # Should be non-negative
        assert torch.all(attn_weights >= 0)
        # Should sum to 1
        sums = attn_weights.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)

    def test_get_conformer_importances(self, model):
        """Test conformer importance extraction."""
        model.eval()
        x = torch.randn(4, 15, 50)

        with torch.no_grad():
            importances = model.get_conformer_importances(x)

        assert importances.shape == (4, 15)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_conformer(self):
        """Test with single conformer."""
        model = DeepSetsBaseline(
            feature_dim=50,
            output_dim=1,
            dropout=0.0,
        )
        model.eval()

        x = torch.randn(4, 1, 50)
        weights = torch.ones(4, 1)

        with torch.no_grad():
            output = model(x, weights)

        assert output.shape == (4, 1)

    def test_large_conformer_set(self):
        """Test with large number of conformers."""
        model = DeepSetsBaseline(
            feature_dim=50,
            output_dim=1,
            dropout=0.0,
        )
        model.eval()

        x = torch.randn(2, 100, 50)
        weights = torch.softmax(torch.randn(2, 100), dim=-1)

        with torch.no_grad():
            output = model(x, weights)

        assert output.shape == (2, 1)

    def test_small_feature_dim(self):
        """Test with small feature dimension."""
        model = DeepSetsBaseline(
            feature_dim=10,
            output_dim=1,
            encoder_hidden_dims=[32, 16],
            decoder_hidden_dim=16,
        )
        model.eval()

        x = torch.randn(4, 15, 10)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (4, 1)

    def test_multi_output(self):
        """Test with multiple outputs."""
        model = DeepSetsBaseline(
            feature_dim=50,
            output_dim=5,
        )
        model.eval()

        x = torch.randn(4, 10, 50)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (4, 5)


class TestDeviceCompatibility:
    """Test device compatibility."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self):
        """Test forward pass on CUDA."""
        model = DeepSetsBaseline(feature_dim=50, output_dim=1).cuda()
        x = torch.randn(4, 10, 50).cuda()
        weights = torch.softmax(torch.randn(4, 10), dim=-1).cuda()

        output = model(x, weights)

        assert output.device.type == 'cuda'
        assert output.shape == (4, 1)

    def test_cpu_forward(self):
        """Test forward pass on CPU."""
        model = DeepSetsBaseline(feature_dim=50, output_dim=1).cpu()
        x = torch.randn(4, 10, 50).cpu()
        weights = torch.softmax(torch.randn(4, 10), dim=-1).cpu()

        model.eval()
        with torch.no_grad():
            output = model(x, weights)

        assert output.device.type == 'cpu'
        assert output.shape == (4, 1)


class TestDeterminism:
    """Test deterministic behavior."""

    def test_deterministic_inference(self):
        """Test that inference is deterministic in eval mode."""
        model = DeepSetsBaseline(feature_dim=50, output_dim=1)
        model.eval()

        x = torch.randn(4, 10, 50)
        weights = torch.softmax(torch.randn(4, 10), dim=-1)

        with torch.no_grad():
            output1 = model(x, weights)
            output2 = model(x, weights)

        torch.testing.assert_close(output1, output2)

    def test_training_mode_differs(self):
        """Test that training mode with dropout differs."""
        model = DeepSetsBaseline(feature_dim=50, output_dim=1, dropout=0.5)
        model.train()

        x = torch.randn(4, 10, 50)
        weights = torch.softmax(torch.randn(4, 10), dim=-1)

        torch.manual_seed(42)
        output1 = model(x, weights)
        torch.manual_seed(123)
        output2 = model(x, weights)

        # Should be different due to dropout
        assert not torch.allclose(output1, output2)
