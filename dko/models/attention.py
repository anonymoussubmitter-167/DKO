"""
Attention-based models for conformer aggregation.

This module implements attention mechanisms for learning to weight
conformers in an ensemble.

Key models:
- AttentionPoolingBaseline: Capacity-matched baseline for DKO comparison
- MultiHeadAttention: Core multi-head attention mechanism
- AttentionPooling: Learnable query-based pooling
- AttentionAggregation: Full attention model with encoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
import math


class AttentionPoolingBaseline(nn.Module):
    """
    Capacity-matched attention baseline for DKO comparison.

    This is the primary baseline model for Experiment 4, designed to:
    1. Match DKO's parameter count for fair comparison
    2. Learn attention weights over conformers
    3. Support attention weight extraction for analysis

    Architecture:
        1. Feature encoder: Linear projection to embed_dim
        2. Multi-head self-attention with qkv_dim=128, num_heads=4
        3. Attention pooling with learnable query
        4. Prediction head with dropout

    Reference:
        Used for Experiment 4: Attention Weight Analysis
    """

    def __init__(
        self,
        feature_dim: int,
        output_dim: int = 1,
        task: str = 'regression',
        embed_dim: int = 128,
        qkv_dim: int = 128,
        num_heads: int = 4,
        num_attention_layers: int = 2,
        prediction_hidden_dim: int = 128,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        """
        Initialize capacity-matched attention baseline.

        Args:
            feature_dim: Dimension of input conformer features
            output_dim: Number of output predictions
            task: 'regression' or 'classification'
            embed_dim: Embedding dimension (default: 128 for capacity matching)
            qkv_dim: QKV projection dimension (default: 128)
            num_heads: Number of attention heads (default: 4)
            num_attention_layers: Number of self-attention layers
            prediction_hidden_dim: Hidden dimension for prediction head
            dropout: Dropout rate
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.task = task
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Feature encoder: project to embedding dimension
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, embed_dim),
            nn.LayerNorm(embed_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Self-attention layers
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                head_dim=qkv_dim // num_heads,
                dropout=dropout,
                use_layer_norm=use_layer_norm,
            )
            for _ in range(num_attention_layers)
        ])

        # Learnable query for pooling (PMA-style)
        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pool_attention = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=qkv_dim // num_heads,
            dropout=dropout,
            use_layer_norm=False,
        )

        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(embed_dim, prediction_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(prediction_hidden_dim, output_dim),
        )

        # Store last attention weights for analysis
        self._last_attention_weights = None
        self._last_pooling_weights = None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass.

        Args:
            x: Conformer features (batch, n_conformers, feature_dim)
            mask: Boolean mask for valid conformers (batch, n_conformers)
            return_attention: Whether to return attention weights

        Returns:
            predictions: Output predictions (batch, output_dim)
            attention_info: Dict with attention weights if return_attention=True
        """
        batch_size, n_conf, _ = x.shape

        # Encode features
        encoded = self.feature_encoder(x)

        # Self-attention layers
        all_attn_weights = []
        for attn_layer in self.attention_layers:
            encoded, attn_weights = attn_layer(
                encoded, mask=mask, return_attention=True
            )
            all_attn_weights.append(attn_weights)

        # Pooling attention with learnable query
        query = self.pool_query.expand(batch_size, -1, -1)
        pooled, pool_weights = self.pool_attention(
            query=query,
            key=encoded,
            value=encoded,
            mask=mask,
            return_attention=True,
        )
        pooled = pooled.squeeze(1)  # (batch, embed_dim)

        # Store for analysis
        self._last_attention_weights = all_attn_weights
        self._last_pooling_weights = pool_weights.squeeze(1)  # (batch, n_conf)

        # Predict
        predictions = self.prediction_head(pooled)

        if return_attention:
            attention_info = {
                'self_attention': all_attn_weights,  # List of (batch, n_conf, n_conf)
                'pooling_weights': self._last_pooling_weights,  # (batch, n_conf)
            }
            return predictions, attention_info

        return predictions, None

    def get_conformer_weights(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get attention weights for each conformer.

        Used for Experiment 4 to compare with Boltzmann weights.

        Args:
            x: Conformer features (batch, n_conformers, feature_dim)
            mask: Boolean mask for valid conformers

        Returns:
            Attention weights (batch, n_conformers)
        """
        self.eval()
        with torch.no_grad():
            _, attention_info = self.forward(x, mask, return_attention=True)
        return attention_info['pooling_weights']

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def create_capacity_matched(
        cls,
        feature_dim: int,
        target_params: int,
        output_dim: int = 1,
        task: str = 'regression',
        **kwargs,
    ) -> 'AttentionPoolingBaseline':
        """
        Create model with approximately target_params parameters.

        Args:
            feature_dim: Input feature dimension
            target_params: Target parameter count
            output_dim: Output dimension
            task: Task type
            **kwargs: Additional arguments

        Returns:
            AttentionPoolingBaseline with matched capacity
        """
        # Start with default config
        model = cls(
            feature_dim=feature_dim,
            output_dim=output_dim,
            task=task,
            **kwargs,
        )

        current_params = model.count_parameters()

        # Adjust embed_dim to match capacity
        if current_params < target_params:
            # Scale up
            scale = math.sqrt(target_params / current_params)
            new_embed_dim = int(128 * scale)
            new_hidden_dim = int(128 * scale)
        else:
            # Scale down
            scale = math.sqrt(target_params / current_params)
            new_embed_dim = max(32, int(128 * scale))
            new_hidden_dim = max(32, int(128 * scale))

        return cls(
            feature_dim=feature_dim,
            output_dim=output_dim,
            task=task,
            embed_dim=new_embed_dim,
            prediction_hidden_dim=new_hidden_dim,
            **kwargs,
        )


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention for conformer sets.

    Learns to attend over conformers to identify the most relevant
    ones for property prediction.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        head_dim: Optional[int] = None,
        dropout: float = 0.1,
        bias: bool = True,
        use_layer_norm: bool = True,
    ):
        """
        Initialize multi-head attention.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            head_dim: Dimension per head (if None, embed_dim // num_heads)
            dropout: Dropout rate
            bias: Whether to use bias in projections
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (embed_dim // num_heads)
        self.inner_dim = self.num_heads * self.head_dim
        self.scale = self.head_dim ** -0.5

        # Projections
        self.q_proj = nn.Linear(embed_dim, self.inner_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, self.inner_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.inner_dim, bias=bias)
        self.out_proj = nn.Linear(self.inner_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim) if use_layer_norm else None

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of multi-head attention.

        Args:
            query: Query tensor (batch, seq_len, embed_dim)
            key: Key tensor (if None, self-attention)
            value: Value tensor (if None, self-attention)
            mask: Attention mask (batch, seq_len)
            return_attention: Whether to return attention weights

        Returns:
            Output tensor and optionally attention weights
        """
        if key is None:
            key = query
        if value is None:
            value = query

        batch_size, seq_len, _ = query.shape

        # Project to Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape for multi-head: (batch, heads, seq, head_dim)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask
        if mask is not None:
            # Expand mask: (batch, seq) -> (batch, 1, 1, seq)
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back: (batch, seq, inner_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.inner_dim)

        # Output projection
        output = self.out_proj(attn_output)

        # Layer norm (residual connection)
        if self.layer_norm is not None:
            output = self.layer_norm(output + query)

        if return_attention:
            # Average attention weights over heads
            attn_weights_avg = attn_weights.mean(dim=1)
            return output, attn_weights_avg
        return output, None


class AttentionPooling(nn.Module):
    """
    Attention-based pooling for aggregating conformer features.

    Uses a learnable query to attend over conformers and produce
    a single aggregated representation.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        temperature: float = 1.0,
        dropout: float = 0.1,
    ):
        """
        Initialize attention pooling.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            temperature: Temperature for softmax
            dropout: Dropout rate
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.temperature = temperature

        # Learnable query for pooling
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Attention
        self.attention = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_layer_norm=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Pool conformer features using attention.

        Args:
            x: Conformer features (batch, n_conformers, embed_dim)
            mask: Mask for valid conformers
            return_attention: Whether to return attention weights

        Returns:
            Pooled features (batch, embed_dim) and optionally attention weights
        """
        batch_size = x.shape[0]

        # Expand query for batch
        query = self.query.expand(batch_size, -1, -1)

        # Cross-attention: query attends to conformers
        output, attn_weights = self.attention(
            query=query,
            key=x,
            value=x,
            mask=mask,
            return_attention=return_attention,
        )

        # Squeeze sequence dimension
        output = output.squeeze(1)

        if return_attention:
            attn_weights = attn_weights.squeeze(1)  # (batch, n_conformers)
            return output, attn_weights
        return output, None


class AttentionAggregation(nn.Module):
    """
    Full attention-based model for conformer aggregation.

    Combines feature encoding, self-attention, and attention pooling
    for property prediction.
    """

    def __init__(
        self,
        feature_dim: int,
        encoder_hidden_dims: List[int] = [256, 128],
        num_heads: int = 4,
        head_dim: int = 32,
        num_attention_layers: int = 2,
        pooling_heads: int = 1,
        prediction_hidden_dims: List[int] = [64, 32],
        num_outputs: int = 1,
        dropout: float = 0.1,
        temperature: float = 1.0,
        activation: str = "relu",
    ):
        """
        Initialize attention aggregation model.

        Args:
            feature_dim: Input feature dimension
            encoder_hidden_dims: Hidden dimensions for feature encoder
            num_heads: Number of attention heads
            head_dim: Dimension per attention head
            num_attention_layers: Number of self-attention layers
            pooling_heads: Number of heads for pooling attention
            prediction_hidden_dims: Hidden dimensions for prediction head
            num_outputs: Number of outputs
            dropout: Dropout rate
            temperature: Temperature for attention
            activation: Activation function
        """
        super().__init__()

        self.temperature = temperature

        # Feature encoder
        encoder_layers = []
        dims = [feature_dim] + encoder_hidden_dims
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }
        act_fn = activations.get(activation, nn.ReLU())

        for i in range(len(dims) - 1):
            encoder_layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                act_fn,
                nn.Dropout(dropout),
            ])
        self.encoder = nn.Sequential(*encoder_layers)

        embed_dim = encoder_hidden_dims[-1] if encoder_hidden_dims else feature_dim

        # Self-attention layers
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                dropout=dropout,
                use_layer_norm=True,
            )
            for _ in range(num_attention_layers)
        ])

        # Attention pooling
        self.pooling = AttentionPooling(
            embed_dim=embed_dim,
            num_heads=pooling_heads,
            temperature=temperature,
            dropout=dropout,
        )

        # Prediction head
        pred_layers = []
        pred_dims = [embed_dim] + prediction_hidden_dims + [num_outputs]
        for i in range(len(pred_dims) - 1):
            pred_layers.append(nn.Linear(pred_dims[i], pred_dims[i + 1]))
            if i < len(pred_dims) - 2:
                pred_layers.extend([act_fn, nn.Dropout(dropout)])
        self.prediction_head = nn.Sequential(*pred_layers)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Conformer features (batch, n_conformers, feature_dim)
            mask: Mask for valid conformers
            return_attention: Whether to return attention weights

        Returns:
            Predictions and optionally attention weights
        """
        batch_size, n_conf, _ = x.shape

        # Encode features
        x_flat = x.view(-1, x.shape[-1])
        encoded = self.encoder(x_flat)
        encoded = encoded.view(batch_size, n_conf, -1)

        # Self-attention layers
        for attn_layer in self.attention_layers:
            encoded, _ = attn_layer(encoded, mask=mask)

        # Pool conformers
        pooled, attn_weights = self.pooling(encoded, mask, return_attention)

        # Predict
        predictions = self.prediction_head(pooled)

        if return_attention:
            return predictions, attn_weights
        return predictions, None

    def get_conformer_importances(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get importance scores for each conformer based on attention.

        Args:
            x: Conformer features
            mask: Mask for valid conformers

        Returns:
            Importance scores (batch, n_conformers)
        """
        _, attn_weights = self.forward(x, mask, return_attention=True)
        return attn_weights


class AttentionAugmented(nn.Module):
    """
    Attention-Augmented model with explicit second-order features.

    This model augments the attention input with explicit outer products of
    features, allowing the model to access second-order statistics similar
    to DKO but through learned attention over augmented features.

    Used for Experiment 3: Representation vs Architecture study.
    Tests whether giving attention explicit access to second-order
    information closes the gap with DKO.

    Architecture:
        1. Compute augmented features: [phi(x), outer_product(phi(x))]
        2. Apply multi-head attention over augmented features
        3. Attention pooling with learnable query
        4. Prediction head

    Reference:
        Research Plan Section 4.3 - Representation vs Architecture
    """

    def __init__(
        self,
        feature_dim: int,
        output_dim: int = 1,
        task: str = 'regression',
        embed_dim: int = 128,
        qkv_dim: int = 128,
        num_heads: int = 4,
        num_attention_layers: int = 2,
        prediction_hidden_dim: int = 128,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        outer_product_dim: Optional[int] = None,
        use_diagonal_only: bool = False,
    ):
        """
        Initialize Attention-Augmented model.

        Args:
            feature_dim: Dimension of input conformer features
            output_dim: Number of output predictions
            task: 'regression' or 'classification'
            embed_dim: Embedding dimension
            qkv_dim: QKV projection dimension
            num_heads: Number of attention heads
            num_attention_layers: Number of self-attention layers
            prediction_hidden_dim: Hidden dimension for prediction head
            dropout: Dropout rate
            use_layer_norm: Whether to use layer normalization
            outer_product_dim: Dimension to project features before outer product
                              (reduces quadratic explosion). If None, uses embed_dim // 4
            use_diagonal_only: If True, only use diagonal of outer product (variance)
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.task = task
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_diagonal_only = use_diagonal_only

        # Project features to lower dimension for outer product computation
        self.outer_dim = outer_product_dim or max(16, embed_dim // 4)

        # Feature encoder: project to embedding dimension
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, embed_dim),
            nn.LayerNorm(embed_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Projection for outer product computation (reduces dimensionality)
        self.outer_proj = nn.Linear(embed_dim, self.outer_dim)

        # Dimension of outer product features
        if use_diagonal_only:
            outer_feat_dim = self.outer_dim  # Just variances
        else:
            outer_feat_dim = self.outer_dim * (self.outer_dim + 1) // 2  # Upper triangular

        # Combined feature dimension
        self.augmented_dim = embed_dim + outer_feat_dim

        # Project augmented features back to embed_dim
        self.augment_proj = nn.Sequential(
            nn.Linear(self.augmented_dim, embed_dim),
            nn.LayerNorm(embed_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Self-attention layers
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                head_dim=qkv_dim // num_heads,
                dropout=dropout,
                use_layer_norm=use_layer_norm,
            )
            for _ in range(num_attention_layers)
        ])

        # Learnable query for pooling (PMA-style)
        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pool_attention = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=qkv_dim // num_heads,
            dropout=dropout,
            use_layer_norm=False,
        )

        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(embed_dim, prediction_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(prediction_hidden_dim, output_dim),
        )

        # Store attention weights for analysis
        self._last_attention_weights = None
        self._last_pooling_weights = None

    def compute_outer_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute outer product features for each conformer.

        Args:
            x: Encoded features (batch, n_conformers, embed_dim)

        Returns:
            Outer product features (batch, n_conformers, outer_feat_dim)
        """
        batch_size, n_conf, _ = x.shape

        # Project to lower dimension
        x_proj = self.outer_proj(x)  # (batch, n_conf, outer_dim)

        if self.use_diagonal_only:
            # Just squared values (variance-like)
            return x_proj ** 2
        else:
            # Compute upper triangular part of outer product
            outer_feats = []
            for i in range(self.outer_dim):
                for j in range(i, self.outer_dim):
                    outer_feats.append(x_proj[..., i] * x_proj[..., j])

            return torch.stack(outer_feats, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass.

        Args:
            x: Conformer features (batch, n_conformers, feature_dim)
            mask: Boolean mask for valid conformers (batch, n_conformers)
            return_attention: Whether to return attention weights

        Returns:
            predictions: Output predictions (batch, output_dim)
            attention_info: Dict with attention weights if return_attention=True
        """
        batch_size, n_conf, _ = x.shape

        # Encode features
        encoded = self.feature_encoder(x)

        # Compute second-order features (outer products)
        outer_feats = self.compute_outer_features(encoded)

        # Concatenate first-order and second-order features
        augmented = torch.cat([encoded, outer_feats], dim=-1)

        # Project back to embed_dim
        augmented = self.augment_proj(augmented)

        # Self-attention layers
        all_attn_weights = []
        for attn_layer in self.attention_layers:
            augmented, attn_weights = attn_layer(
                augmented, mask=mask, return_attention=True
            )
            all_attn_weights.append(attn_weights)

        # Pooling attention with learnable query
        query = self.pool_query.expand(batch_size, -1, -1)
        pooled, pool_weights = self.pool_attention(
            query=query,
            key=augmented,
            value=augmented,
            mask=mask,
            return_attention=True,
        )
        pooled = pooled.squeeze(1)  # (batch, embed_dim)

        # Store for analysis
        self._last_attention_weights = all_attn_weights
        self._last_pooling_weights = pool_weights.squeeze(1)

        # Predict
        predictions = self.prediction_head(pooled)

        if return_attention:
            attention_info = {
                'self_attention': all_attn_weights,
                'pooling_weights': self._last_pooling_weights,
            }
            return predictions, attention_info

        return predictions, None

    def get_conformer_weights(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get attention weights for each conformer."""
        self.eval()
        with torch.no_grad():
            _, attention_info = self.forward(x, mask, return_attention=True)
        return attention_info['pooling_weights']

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
