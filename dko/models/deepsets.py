"""
DeepSets model for permutation-invariant set learning.

This module implements DeepSets, a fundamental baseline for learning
from set-structured data like conformer ensembles.

Key models:
- DeepSetsBaseline: Capacity-matched baseline with Boltzmann weighting for DKO comparison
- DeepSets: Standard DeepSets implementation
- DeepSetsWithAttention: DeepSets with attention-weighted pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Any
import math


class DeepSetsBaseline(nn.Module):
    """
    Capacity-matched DeepSets baseline for DKO comparison.

    This is the primary DeepSets baseline designed to:
    1. Match DKO's parameter count for fair comparison
    2. Support Boltzmann-weighted aggregation
    3. Use sum pooling as specified in research plan

    Architecture:
        1. Encoder: [256, 256, 128] MLP with LayerNorm
        2. Boltzmann-weighted sum pooling
        3. Decoder: [128, output_dim] with dropout

    Key feature:
        Uses Boltzmann weights from conformer energies instead of
        learned attention, following thermodynamic principles.

    Reference:
        Zaheer et al. "Deep Sets" (2017)
    """

    def __init__(
        self,
        feature_dim: int,
        output_dim: int = 1,
        task: str = 'regression',
        encoder_hidden_dims: List[int] = [256, 256, 128],
        decoder_hidden_dim: int = 128,
        pooling_method: str = "boltzmann_sum",
        use_boltzmann_weights: bool = True,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        """
        Initialize capacity-matched DeepSets baseline.

        Args:
            feature_dim: Dimension of input conformer features
            output_dim: Number of output predictions
            task: 'regression' or 'classification'
            encoder_hidden_dims: Hidden dimensions for encoder (phi network)
            decoder_hidden_dim: Hidden dimension for decoder (rho network)
            pooling_method: 'boltzmann_sum', 'sum', 'mean', or 'max'
            use_boltzmann_weights: Whether to use Boltzmann weights in pooling
            dropout: Dropout rate
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.task = task
        self.pooling_method = pooling_method
        self.use_boltzmann_weights = use_boltzmann_weights

        # Encoder (phi network): processes each conformer independently
        encoder_layers = []
        dims = [feature_dim] + encoder_hidden_dims
        for i in range(len(dims) - 1):
            encoder_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if use_layer_norm:
                encoder_layers.append(nn.LayerNorm(dims[i + 1]))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (rho network): processes aggregated representation
        encoder_output_dim = encoder_hidden_dims[-1]
        self.decoder = nn.Sequential(
            nn.Linear(encoder_output_dim, decoder_hidden_dim),
            nn.LayerNorm(decoder_hidden_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_dim, output_dim),
        )

    def pool(
        self,
        x: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Permutation-invariant pooling with optional Boltzmann weighting.

        Args:
            x: Features (batch, n_conformers, feature_dim)
            weights: Boltzmann weights (batch, n_conformers), should sum to 1
            mask: Boolean mask for valid conformers (batch, n_conformers)

        Returns:
            Pooled features (batch, feature_dim)
        """
        # Apply mask if provided
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()

        if self.pooling_method == "boltzmann_sum" and weights is not None:
            # Boltzmann-weighted sum: sum_i w_i * phi(x_i)
            if mask is not None:
                weights = weights * mask.float()
                weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
            weighted = x * weights.unsqueeze(-1)
            return weighted.sum(dim=1)

        elif self.pooling_method == "sum":
            return x.sum(dim=1)

        elif self.pooling_method == "mean":
            if mask is not None:
                count = mask.sum(dim=1, keepdim=True).clamp(min=1)
                return x.sum(dim=1) / count
            return x.mean(dim=1)

        elif self.pooling_method == "max":
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            return x.max(dim=1)[0]

        else:
            # Default: use weights if available, otherwise mean
            if weights is not None:
                weighted = x * weights.unsqueeze(-1)
                return weighted.sum(dim=1)
            return x.mean(dim=1)

    def forward(
        self,
        x: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of DeepSets.

        Args:
            x: Conformer features (batch, n_conformers, feature_dim)
            weights: Boltzmann weights (batch, n_conformers)
            mask: Boolean mask for valid conformers

        Returns:
            Predictions (batch, output_dim)
        """
        batch_size, n_conf, feat_dim = x.shape

        # Encode each conformer
        x_flat = x.view(-1, feat_dim)
        encoded = self.encoder(x_flat)
        encoded = encoded.view(batch_size, n_conf, -1)

        # Pool conformers with optional Boltzmann weighting
        pooled = self.pool(encoded, weights, mask)

        # Decode
        predictions = self.decoder(pooled)

        return predictions

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
    ) -> 'DeepSetsBaseline':
        """
        Create model with approximately target_params parameters.

        Args:
            feature_dim: Input feature dimension
            target_params: Target parameter count
            output_dim: Output dimension
            task: Task type
            **kwargs: Additional arguments

        Returns:
            DeepSetsBaseline with matched capacity
        """
        # Start with default config
        model = cls(
            feature_dim=feature_dim,
            output_dim=output_dim,
            task=task,
            **kwargs,
        )

        current_params = model.count_parameters()

        # Adjust hidden dims to match capacity
        if current_params < target_params:
            scale = (target_params / current_params) ** 0.33  # Cube root for 3 layers
            new_hidden = [int(256 * scale), int(256 * scale), int(128 * scale)]
            new_decoder_hidden = int(128 * scale)
        else:
            scale = (target_params / current_params) ** 0.33
            new_hidden = [max(64, int(256 * scale)), max(64, int(256 * scale)), max(32, int(128 * scale))]
            new_decoder_hidden = max(32, int(128 * scale))

        return cls(
            feature_dim=feature_dim,
            output_dim=output_dim,
            task=task,
            encoder_hidden_dims=new_hidden,
            decoder_hidden_dim=new_decoder_hidden,
            **kwargs,
        )


class DeepSets(nn.Module):
    """
    DeepSets model for conformer ensemble learning.

    DeepSets processes each element (conformer) independently through a
    shared network (phi), aggregates using a permutation-invariant operation,
    then processes the aggregated representation through another network (rho).

    Architecture:
        y = rho(pool(phi(x_1), phi(x_2), ..., phi(x_n)))

    Reference:
        Zaheer et al. "Deep Sets" (2017)
    """

    def __init__(
        self,
        feature_dim: int,
        phi_hidden_dims: List[int] = [256, 128],
        phi_output_dim: int = 64,
        rho_hidden_dims: List[int] = [128, 64],
        prediction_hidden_dims: List[int] = [32],
        num_outputs: int = 1,
        pooling_method: str = "mean",
        activation: str = "relu",
        use_batch_norm: bool = True,
        dropout: float = 0.1,
    ):
        """
        Initialize DeepSets model.

        Args:
            feature_dim: Dimension of input conformer features
            phi_hidden_dims: Hidden dimensions for phi network
            phi_output_dim: Output dimension of phi network
            rho_hidden_dims: Hidden dimensions for rho network
            prediction_hidden_dims: Hidden dimensions for prediction head
            num_outputs: Number of output predictions
            pooling_method: 'mean', 'sum', 'max', or 'mean_max'
            activation: Activation function
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout rate
        """
        super().__init__()

        self.pooling_method = pooling_method

        # Activation function
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
        }
        act_fn = activations.get(activation, nn.ReLU())

        # Phi network: processes each conformer independently
        phi_layers = []
        dims = [feature_dim] + phi_hidden_dims + [phi_output_dim]
        for i in range(len(dims) - 1):
            phi_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No batch norm on last layer
                if use_batch_norm:
                    phi_layers.append(nn.BatchNorm1d(dims[i + 1]))
                phi_layers.append(act_fn)
                if dropout > 0:
                    phi_layers.append(nn.Dropout(dropout))
        self.phi = nn.Sequential(*phi_layers)

        # Rho network: processes aggregated representation
        rho_input_dim = phi_output_dim * 2 if pooling_method == "mean_max" else phi_output_dim
        rho_layers = []
        dims = [rho_input_dim] + rho_hidden_dims
        for i in range(len(dims) - 1):
            rho_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if use_batch_norm:
                rho_layers.append(nn.BatchNorm1d(dims[i + 1]))
            rho_layers.append(act_fn)
            if dropout > 0:
                rho_layers.append(nn.Dropout(dropout))
        self.rho = nn.Sequential(*rho_layers) if rho_layers else nn.Identity()

        # Prediction head
        rho_output_dim = rho_hidden_dims[-1] if rho_hidden_dims else rho_input_dim
        pred_layers = []
        dims = [rho_output_dim] + prediction_hidden_dims + [num_outputs]
        for i in range(len(dims) - 1):
            pred_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                pred_layers.append(act_fn)
                if dropout > 0:
                    pred_layers.append(nn.Dropout(dropout))
        self.prediction_head = nn.Sequential(*pred_layers)

    def pool(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Permutation-invariant pooling operation.

        Args:
            x: Features (batch, n_conformers, feature_dim)
            mask: Mask for valid conformers (batch, n_conformers)

        Returns:
            Pooled features (batch, feature_dim) or (batch, 2*feature_dim) for mean_max
        """
        if mask is not None:
            # Apply mask
            x = x * mask.unsqueeze(-1).float()

        if self.pooling_method == "mean":
            if mask is not None:
                sum_x = x.sum(dim=1)
                count = mask.sum(dim=1, keepdim=True).clamp(min=1)
                return sum_x / count
            return x.mean(dim=1)

        elif self.pooling_method == "sum":
            return x.sum(dim=1)

        elif self.pooling_method == "max":
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            return x.max(dim=1)[0]

        elif self.pooling_method == "mean_max":
            # Concatenate mean and max pooling
            if mask is not None:
                sum_x = x.sum(dim=1)
                count = mask.sum(dim=1, keepdim=True).clamp(min=1)
                mean_pool = sum_x / count
                x_masked = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
                max_pool = x_masked.max(dim=1)[0]
            else:
                mean_pool = x.mean(dim=1)
                max_pool = x.max(dim=1)[0]
            return torch.cat([mean_pool, max_pool], dim=-1)

        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of DeepSets.

        Args:
            x: Conformer features (batch, n_conformers, feature_dim)
            mask: Mask for valid conformers

        Returns:
            Predictions (batch, num_outputs)
        """
        batch_size, n_conf, feat_dim = x.shape

        # Apply phi to each conformer
        x_flat = x.view(-1, feat_dim)
        phi_out = self.phi(x_flat)
        phi_out = phi_out.view(batch_size, n_conf, -1)

        # Pool conformers
        pooled = self.pool(phi_out, mask)

        # Apply rho
        rho_out = self.rho(pooled)

        # Predict
        predictions = self.prediction_head(rho_out)

        return predictions


class DeepSetsWithAttention(nn.Module):
    """
    DeepSets variant with attention-weighted pooling.

    Instead of simple mean/max pooling, uses learned attention weights
    to aggregate conformer representations.
    """

    def __init__(
        self,
        feature_dim: int,
        phi_hidden_dims: List[int] = [256, 128],
        phi_output_dim: int = 64,
        rho_hidden_dims: List[int] = [128, 64],
        prediction_hidden_dims: List[int] = [32],
        num_outputs: int = 1,
        attention_hidden_dim: int = 32,
        temperature: float = 1.0,
        activation: str = "relu",
        use_batch_norm: bool = True,
        dropout: float = 0.1,
    ):
        """
        Initialize DeepSets with attention.

        Args:
            feature_dim: Input feature dimension
            phi_hidden_dims: Hidden dims for phi network
            phi_output_dim: Output dim for phi network
            rho_hidden_dims: Hidden dims for rho network
            prediction_hidden_dims: Hidden dims for prediction head
            num_outputs: Number of outputs
            attention_hidden_dim: Hidden dim for attention network
            temperature: Temperature for attention softmax
            activation: Activation function
            use_batch_norm: Whether to use batch norm
            dropout: Dropout rate
        """
        super().__init__()

        self.temperature = temperature

        # Activation
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }
        act_fn = activations.get(activation, nn.ReLU())

        # Phi network
        phi_layers = []
        dims = [feature_dim] + phi_hidden_dims + [phi_output_dim]
        for i in range(len(dims) - 1):
            phi_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if use_batch_norm:
                    phi_layers.append(nn.BatchNorm1d(dims[i + 1]))
                phi_layers.append(act_fn)
                if dropout > 0:
                    phi_layers.append(nn.Dropout(dropout))
        self.phi = nn.Sequential(*phi_layers)

        # Attention network: computes attention scores
        self.attention_net = nn.Sequential(
            nn.Linear(phi_output_dim, attention_hidden_dim),
            act_fn,
            nn.Linear(attention_hidden_dim, 1),
        )

        # Rho network
        rho_layers = []
        dims = [phi_output_dim] + rho_hidden_dims
        for i in range(len(dims) - 1):
            rho_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if use_batch_norm:
                rho_layers.append(nn.BatchNorm1d(dims[i + 1]))
            rho_layers.append(act_fn)
            if dropout > 0:
                rho_layers.append(nn.Dropout(dropout))
        self.rho = nn.Sequential(*rho_layers) if rho_layers else nn.Identity()

        # Prediction head
        rho_output_dim = rho_hidden_dims[-1] if rho_hidden_dims else phi_output_dim
        pred_layers = []
        dims = [rho_output_dim] + prediction_hidden_dims + [num_outputs]
        for i in range(len(dims) - 1):
            pred_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                pred_layers.append(act_fn)
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
        batch_size, n_conf, feat_dim = x.shape

        # Apply phi
        x_flat = x.view(-1, feat_dim)
        phi_out = self.phi(x_flat)
        phi_out = phi_out.view(batch_size, n_conf, -1)

        # Compute attention scores
        attn_scores = self.attention_net(phi_out).squeeze(-1)  # (batch, n_conf)
        attn_scores = attn_scores / self.temperature

        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, n_conf)

        # Weighted sum
        weighted = torch.bmm(
            attn_weights.unsqueeze(1),  # (batch, 1, n_conf)
            phi_out,  # (batch, n_conf, phi_dim)
        ).squeeze(1)  # (batch, phi_dim)

        # Apply rho
        rho_out = self.rho(weighted)

        # Predict
        predictions = self.prediction_head(rho_out)

        if return_attention:
            return predictions, attn_weights
        return predictions, None

    def get_conformer_importances(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get attention weights as conformer importances."""
        _, attn_weights = self.forward(x, mask, return_attention=True)
        return attn_weights


class DeepSetsAugmented(nn.Module):
    """
    DeepSets-Augmented model with explicit second-order features.

    This model augments DeepSets input with explicit outer products of
    features, providing the model with second-order statistics similar
    to DKO but processed through the DeepSets architecture.

    Used for Experiment 3: Representation vs Architecture study.
    Tests whether giving DeepSets explicit access to second-order
    information closes the gap with DKO.

    Architecture:
        1. Compute augmented features: [phi(x), outer_product(phi(x))]
        2. Apply phi network to augmented features
        3. Sum pooling (permutation invariant)
        4. Apply rho network
        5. Prediction head

    The key difference from standard DeepSets is that each conformer's
    representation includes both the encoded features and their outer
    products, giving explicit access to second-order information.

    Reference:
        Research Plan Section 4.3 - Representation vs Architecture
    """

    def __init__(
        self,
        feature_dim: int,
        output_dim: int = 1,
        task: str = 'regression',
        encoder_hidden_dims: List[int] = [256, 256, 128],
        decoder_hidden_dim: int = 128,
        pooling_method: str = "sum",
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        outer_product_dim: Optional[int] = None,
        use_diagonal_only: bool = False,
    ):
        """
        Initialize DeepSets-Augmented model.

        Args:
            feature_dim: Dimension of input conformer features
            output_dim: Number of output predictions
            task: 'regression' or 'classification'
            encoder_hidden_dims: Hidden dimensions for encoder (phi network)
            decoder_hidden_dim: Hidden dimension for decoder (rho network)
            pooling_method: 'sum', 'mean', or 'max'
            dropout: Dropout rate
            use_layer_norm: Whether to use layer normalization
            outer_product_dim: Dimension to project features before outer product.
                              If None, uses feature_dim // 4
            use_diagonal_only: If True, only use diagonal of outer product (variance)
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.task = task
        self.pooling_method = pooling_method
        self.use_diagonal_only = use_diagonal_only

        # Project features to lower dimension for outer product computation
        self.outer_dim = outer_product_dim or max(16, feature_dim // 4)

        # Initial feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Projection for outer product computation
        self.outer_proj = nn.Linear(feature_dim, self.outer_dim)

        # Dimension of outer product features
        if use_diagonal_only:
            outer_feat_dim = self.outer_dim  # Just variances
        else:
            outer_feat_dim = self.outer_dim * (self.outer_dim + 1) // 2  # Upper triangular

        # Augmented input dimension
        augmented_dim = feature_dim + outer_feat_dim

        # Encoder (phi network): processes each conformer independently
        encoder_layers = []
        dims = [augmented_dim] + encoder_hidden_dims
        for i in range(len(dims) - 1):
            encoder_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if use_layer_norm:
                encoder_layers.append(nn.LayerNorm(dims[i + 1]))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (rho network): processes aggregated representation
        encoder_output_dim = encoder_hidden_dims[-1]
        self.decoder = nn.Sequential(
            nn.Linear(encoder_output_dim, decoder_hidden_dim),
            nn.LayerNorm(decoder_hidden_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_dim, output_dim),
        )

    def compute_outer_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute outer product features for each conformer.

        Args:
            x: Projected features (batch, n_conformers, feature_dim)

        Returns:
            Outer product features (batch, n_conformers, outer_feat_dim)
        """
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

    def pool(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Permutation-invariant pooling.

        Args:
            x: Features (batch, n_conformers, feature_dim)
            mask: Boolean mask for valid conformers (batch, n_conformers)

        Returns:
            Pooled features (batch, feature_dim)
        """
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()

        if self.pooling_method == "sum":
            return x.sum(dim=1)

        elif self.pooling_method == "mean":
            if mask is not None:
                count = mask.sum(dim=1, keepdim=True).clamp(min=1)
                return x.sum(dim=1) / count
            return x.mean(dim=1)

        elif self.pooling_method == "max":
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            return x.max(dim=1)[0]

        else:
            return x.sum(dim=1)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of DeepSets-Augmented.

        Args:
            x: Conformer features (batch, n_conformers, feature_dim)
            mask: Boolean mask for valid conformers

        Returns:
            Predictions (batch, output_dim)
        """
        batch_size, n_conf, feat_dim = x.shape

        # Project features
        x_proj = self.feature_proj(x)

        # Compute second-order features (outer products)
        outer_feats = self.compute_outer_features(x_proj)

        # Concatenate first-order and second-order features
        augmented = torch.cat([x_proj, outer_feats], dim=-1)

        # Encode each conformer
        aug_flat = augmented.view(-1, augmented.shape[-1])
        encoded = self.encoder(aug_flat)
        encoded = encoded.view(batch_size, n_conf, -1)

        # Pool conformers
        pooled = self.pool(encoded, mask)

        # Decode
        predictions = self.decoder(pooled)

        return predictions

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
    ) -> 'DeepSetsAugmented':
        """
        Create model with approximately target_params parameters.

        Args:
            feature_dim: Input feature dimension
            target_params: Target parameter count
            output_dim: Output dimension
            task: Task type
            **kwargs: Additional arguments

        Returns:
            DeepSetsAugmented with matched capacity
        """
        # Start with default config
        model = cls(
            feature_dim=feature_dim,
            output_dim=output_dim,
            task=task,
            **kwargs,
        )

        current_params = model.count_parameters()

        # Adjust hidden dims to match capacity
        if current_params < target_params:
            scale = (target_params / current_params) ** 0.33
            new_hidden = [int(256 * scale), int(256 * scale), int(128 * scale)]
            new_decoder_hidden = int(128 * scale)
        else:
            scale = (target_params / current_params) ** 0.33
            new_hidden = [max(64, int(256 * scale)), max(64, int(256 * scale)), max(32, int(128 * scale))]
            new_decoder_hidden = max(32, int(128 * scale))

        return cls(
            feature_dim=feature_dim,
            output_dim=output_dim,
            task=task,
            encoder_hidden_dims=new_hidden,
            decoder_hidden_dim=new_decoder_hidden,
            **kwargs,
        )
