"""Tests for attention-based models."""

import pytest
import torch
import torch.nn as nn
import numpy as np

from dko.models.attention import (
    AttentionPoolingBaseline,
    MultiHeadAttention,
    AttentionPooling,
    AttentionAggregation,
)


class TestMultiHeadAttention:
    """Test multi-head attention module."""

    @pytest.fixture
    def attention(self):
        """Create attention module."""
        return MultiHeadAttention(
            embed_dim=64,
            num_heads=4,
            head_dim=16,
            dropout=0.1,
            use_layer_norm=True,
        )

    def test_self_attention_shape(self, attention):
        """Test self-attention output shape."""
        batch_size = 4
        seq_len = 10
        x = torch.randn(batch_size, seq_len, 64)

        output, attn_weights = attention(x, return_attention=True)

        assert output.shape == (batch_size, seq_len, 64)
        assert attn_weights.shape == (batch_size, seq_len, seq_len)

    def test_cross_attention_shape(self, attention):
        """Test cross-attention output shape."""
        batch_size = 4
        query_len = 1
        kv_len = 10
        query = torch.randn(batch_size, query_len, 64)
        key = torch.randn(batch_size, kv_len, 64)
        value = torch.randn(batch_size, kv_len, 64)

        output, attn_weights = attention(
            query, key, value, return_attention=True
        )

        assert output.shape == (batch_size, query_len, 64)
        assert attn_weights.shape == (batch_size, query_len, kv_len)

    def test_attention_with_mask(self, attention):
        """Test attention with masking."""
        batch_size = 4
        seq_len = 10
        x = torch.randn(batch_size, seq_len, 64)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        mask[:, 5:] = False  # Mask out last 5 positions

        output, attn_weights = attention(x, mask=mask, return_attention=True)

        assert output.shape == (batch_size, seq_len, 64)
        # Attention to masked positions should be near zero
        # (after softmax, masked positions have very low weight)

    def test_attention_weights_sum_to_one(self, attention):
        """Test that attention weights sum to 1."""
        attention.eval()
        x = torch.randn(4, 10, 64)

        with torch.no_grad():
            _, attn_weights = attention(x, return_attention=True)

        sums = attn_weights.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)


class TestAttentionPooling:
    """Test attention pooling module."""

    @pytest.fixture
    def pooling(self):
        """Create attention pooling."""
        return AttentionPooling(
            embed_dim=64,
            num_heads=4,
            temperature=1.0,
            dropout=0.1,
        )

    def test_pooling_shape(self, pooling):
        """Test pooling output shape."""
        batch_size = 4
        n_conf = 20
        x = torch.randn(batch_size, n_conf, 64)

        output, attn_weights = pooling(x, return_attention=True)

        assert output.shape == (batch_size, 64)
        assert attn_weights.shape == (batch_size, n_conf)

    def test_pooling_with_mask(self, pooling):
        """Test pooling with mask."""
        batch_size = 4
        n_conf = 20
        x = torch.randn(batch_size, n_conf, 64)
        mask = torch.ones(batch_size, n_conf, dtype=torch.bool)
        mask[:, 10:] = False

        pooling.eval()
        with torch.no_grad():
            output, attn_weights = pooling(x, mask=mask, return_attention=True)

        assert output.shape == (batch_size, 64)

    def test_pooling_weights_sum_to_one(self, pooling):
        """Test that pooling weights sum to 1."""
        pooling.eval()
        x = torch.randn(4, 20, 64)

        with torch.no_grad():
            _, attn_weights = pooling(x, return_attention=True)

        sums = attn_weights.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)


class TestAttentionPoolingBaseline:
    """Test capacity-matched attention baseline."""

    @pytest.fixture
    def model(self):
        """Create baseline model."""
        return AttentionPoolingBaseline(
            feature_dim=100,
            output_dim=1,
            task='regression',
            embed_dim=128,
            qkv_dim=128,
            num_heads=4,
            num_attention_layers=2,
            prediction_hidden_dim=128,
            dropout=0.1,
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample conformer data."""
        batch_size = 4
        n_conformers = 20
        feature_dim = 100
        x = torch.randn(batch_size, n_conformers, feature_dim)
        return x

    def test_forward_shape(self, model, sample_data):
        """Test forward pass output shape."""
        model.eval()
        with torch.no_grad():
            output, _ = model(sample_data)

        assert output.shape == (4, 1)

    def test_forward_with_attention(self, model, sample_data):
        """Test forward pass with attention extraction."""
        model.eval()
        with torch.no_grad():
            output, attention_info = model(sample_data, return_attention=True)

        assert output.shape == (4, 1)
        assert 'self_attention' in attention_info
        assert 'pooling_weights' in attention_info
        assert len(attention_info['self_attention']) == 2  # 2 layers
        assert attention_info['pooling_weights'].shape == (4, 20)

    def test_forward_with_mask(self, model, sample_data):
        """Test forward pass with masking."""
        mask = torch.ones(4, 20, dtype=torch.bool)
        mask[:, 10:] = False

        model.eval()
        with torch.no_grad():
            output, attention_info = model(sample_data, mask=mask, return_attention=True)

        assert output.shape == (4, 1)

    def test_get_conformer_weights(self, model, sample_data):
        """Test conformer weight extraction."""
        weights = model.get_conformer_weights(sample_data)

        assert weights.shape == (4, 20)
        # Weights should sum to ~1
        sums = weights.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-4, rtol=1e-4)

    def test_gradient_flow(self, model, sample_data):
        """Test that gradients flow through the model."""
        sample_data.requires_grad = True
        model.train()

        output, _ = model(sample_data)
        loss = output.sum()
        loss.backward()

        assert sample_data.grad is not None
        assert not torch.all(sample_data.grad == 0)

    def test_different_batch_sizes(self, model):
        """Test with different batch sizes."""
        model.eval()

        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, 20, 100)
            with torch.no_grad():
                output, _ = model(x)
            assert output.shape == (batch_size, 1)

    def test_different_conformer_counts(self, model):
        """Test with different numbers of conformers."""
        model.eval()

        for n_conf in [5, 10, 20, 50]:
            x = torch.randn(4, n_conf, 100)
            with torch.no_grad():
                output, _ = model(x)
            assert output.shape == (4, 1)

    def test_classification_mode(self):
        """Test model in classification mode."""
        model = AttentionPoolingBaseline(
            feature_dim=100,
            output_dim=2,
            task='classification',
        )
        model.eval()

        x = torch.randn(4, 20, 100)
        with torch.no_grad():
            output, _ = model(x)

        assert output.shape == (4, 2)

    def test_count_parameters(self, model):
        """Test parameter counting."""
        n_params = model.count_parameters()
        assert n_params > 0
        # Should be in reasonable range for capacity matching
        assert n_params < 10_000_000

    def test_create_capacity_matched(self):
        """Test capacity-matched model creation."""
        target_params = 100_000

        model = AttentionPoolingBaseline.create_capacity_matched(
            feature_dim=100,
            target_params=target_params,
            output_dim=1,
        )

        actual_params = model.count_parameters()
        # Should be within 50% of target
        assert actual_params > target_params * 0.5
        assert actual_params < target_params * 1.5

    def test_deterministic_inference(self, model, sample_data):
        """Test that inference is deterministic in eval mode."""
        model.eval()

        with torch.no_grad():
            output1, _ = model(sample_data)
            output2, _ = model(sample_data)

        torch.testing.assert_close(output1, output2)

    def test_training_mode_differs(self, model, sample_data):
        """Test that training mode with dropout differs."""
        model.train()
        torch.manual_seed(42)
        output1, _ = model(sample_data)
        torch.manual_seed(123)
        output2, _ = model(sample_data)

        # Should be different due to dropout
        assert not torch.allclose(output1, output2)


class TestAttentionAggregation:
    """Test full attention aggregation model."""

    @pytest.fixture
    def model(self):
        """Create attention aggregation model."""
        return AttentionAggregation(
            feature_dim=100,
            encoder_hidden_dims=[128, 64],
            num_heads=4,
            head_dim=16,
            num_attention_layers=2,
            pooling_heads=1,
            prediction_hidden_dims=[32],
            num_outputs=1,
            dropout=0.1,
        )

    def test_forward_shape(self, model):
        """Test forward pass output shape."""
        model.eval()
        x = torch.randn(4, 20, 100)

        with torch.no_grad():
            output, _ = model(x)

        assert output.shape == (4, 1)

    def test_forward_with_attention(self, model):
        """Test forward with attention extraction."""
        model.eval()
        x = torch.randn(4, 20, 100)

        with torch.no_grad():
            output, attn_weights = model(x, return_attention=True)

        assert output.shape == (4, 1)
        assert attn_weights.shape == (4, 20)

    def test_get_conformer_importances(self, model):
        """Test conformer importance extraction."""
        model.eval()
        x = torch.randn(4, 20, 100)

        with torch.no_grad():
            importances = model.get_conformer_importances(x)

        assert importances.shape == (4, 20)


class TestAttentionWeightAnalysis:
    """Test attention weight analysis for Experiment 4."""

    @pytest.fixture
    def model(self):
        """Create model for analysis."""
        return AttentionPoolingBaseline(
            feature_dim=50,
            output_dim=1,
            embed_dim=64,
            num_heads=4,
        )

    def test_attention_weights_extractable(self, model):
        """Test that attention weights can be extracted."""
        model.eval()
        x = torch.randn(8, 15, 50)

        with torch.no_grad():
            _, attention_info = model(x, return_attention=True)

        # Should have pooling weights
        assert attention_info['pooling_weights'].shape == (8, 15)

        # Weights should be valid probabilities
        weights = attention_info['pooling_weights']
        assert torch.all(weights >= 0)
        assert torch.all(weights <= 1)
        sums = weights.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-4, rtol=1e-4)

    def test_attention_vs_boltzmann_comparison(self, model):
        """Test setup for comparing attention vs Boltzmann weights."""
        model.eval()
        batch_size = 8
        n_conf = 15
        feature_dim = 50

        # Simulate conformer features
        x = torch.randn(batch_size, n_conf, feature_dim)

        # Simulate Boltzmann weights (from energy-based computation)
        energies = torch.randn(batch_size, n_conf)
        boltzmann_weights = torch.softmax(-energies, dim=-1)

        # Get attention weights
        with torch.no_grad():
            _, attention_info = model(x, return_attention=True)
        attention_weights = attention_info['pooling_weights']

        # Both should be valid probability distributions
        assert boltzmann_weights.shape == attention_weights.shape

        # Can compute correlation between them
        for i in range(batch_size):
            b_weights = boltzmann_weights[i].numpy()
            a_weights = attention_weights[i].numpy()
            correlation = np.corrcoef(b_weights, a_weights)[0, 1]
            # Correlation can be anything (not necessarily high)
            assert -1 <= correlation <= 1


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_conformer(self):
        """Test with single conformer."""
        model = AttentionPoolingBaseline(feature_dim=50, output_dim=1)
        model.eval()

        x = torch.randn(4, 1, 50)
        with torch.no_grad():
            output, _ = model(x)

        assert output.shape == (4, 1)

    def test_large_conformer_set(self):
        """Test with large number of conformers."""
        model = AttentionPoolingBaseline(feature_dim=50, output_dim=1)
        model.eval()

        x = torch.randn(2, 100, 50)
        with torch.no_grad():
            output, _ = model(x)

        assert output.shape == (2, 1)

    def test_all_masked(self):
        """Test behavior when all conformers are masked."""
        model = AttentionPoolingBaseline(feature_dim=50, output_dim=1)
        model.eval()

        x = torch.randn(4, 10, 50)
        mask = torch.zeros(4, 10, dtype=torch.bool)

        # Should not raise, but output may be nan/inf
        with torch.no_grad():
            output, _ = model(x, mask=mask)

    def test_partial_mask(self):
        """Test with partial masking."""
        model = AttentionPoolingBaseline(feature_dim=50, output_dim=1)
        model.eval()

        x = torch.randn(4, 10, 50)
        mask = torch.ones(4, 10, dtype=torch.bool)
        mask[0, 5:] = False  # First sample has fewer conformers
        mask[1, :3] = False  # Second sample has some early ones masked

        with torch.no_grad():
            output, _ = model(x, mask=mask)

        assert output.shape == (4, 1)
        assert not torch.any(torch.isnan(output))


class TestDeviceCompatibility:
    """Test device compatibility."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self):
        """Test forward pass on CUDA."""
        model = AttentionPoolingBaseline(feature_dim=50, output_dim=1).cuda()
        x = torch.randn(4, 10, 50).cuda()

        output, _ = model(x)

        assert output.device.type == 'cuda'
        assert output.shape == (4, 1)

    def test_cpu_forward(self):
        """Test forward pass on CPU."""
        model = AttentionPoolingBaseline(feature_dim=50, output_dim=1).cpu()
        x = torch.randn(4, 10, 50).cpu()

        model.eval()
        with torch.no_grad():
            output, _ = model(x)

        assert output.device.type == 'cpu'
        assert output.shape == (4, 1)
