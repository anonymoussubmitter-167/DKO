"""
Ensemble baseline models for conformer aggregation.

This module implements simple baseline approaches:
- SingleConformer: Uses only the lowest-energy conformer
- MeanFeatureAggregation (MFA): Averages features before network
- MeanEnsemble: Simple averaging of conformer predictions
- BoltzmannEnsemble: Energy-weighted averaging
- MultiInstanceLearning (MIL): Instance-level with max pooling
- LearnedWeightEnsemble: Learned conformer weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union


class MLP(nn.Module):
    """Multi-layer perceptron helper."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "relu",
        use_batch_norm: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }
        act_fn = activations.get(activation, nn.ReLU())

        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(act_fn)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SingleConformer(nn.Module):
    """
    Single conformer baseline.

    Uses only a single conformer for prediction. Supports multiple selection
    strategies to deconfound sampling effects from aggregation effects in the
    80/20 decomposition study:

    - 'lowest_energy': Standard approach (lowest energy conformer)
    - 'random': Random conformer from ensemble (controls for sampling)
    - 'centroid': Geometric centroid in feature space (controls for outliers)
    - 'first': First conformer (simple baseline)

    For decomposition analysis:
        SingleConformer(lowest) → MFA measures: sampling + aggregation
        SingleConformer(random) → MFA measures: aggregation only
        SingleConformer(centroid) → MFA measures: outlier reduction
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dims: List[int] = [256, 128],
        prediction_hidden_dims: List[int] = [64, 32],
        num_outputs: int = 1,
        selection_method: str = "lowest_energy",
        activation: str = "relu",
        use_batch_norm: bool = True,
        dropout: float = 0.1,
    ):
        """
        Initialize single conformer model.

        Args:
            feature_dim: Dimension of conformer features
            hidden_dims: Hidden dimensions for feature encoder
            prediction_hidden_dims: Hidden dimensions for prediction head
            num_outputs: Number of output predictions
            selection_method: One of:
                - 'lowest_energy': Use lowest-energy conformer (default)
                - 'random': Random conformer from ensemble
                - 'centroid': Conformer closest to mean in feature space
                - 'first': First conformer
            activation: Activation function
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout rate
        """
        super().__init__()

        self.selection_method = selection_method

        # Feature encoder
        self.encoder = MLP(
            input_dim=feature_dim,
            hidden_dims=hidden_dims,
            output_dim=hidden_dims[-1] if hidden_dims else feature_dim,
            activation=activation,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
        )

        # Prediction head
        encoder_out_dim = hidden_dims[-1] if hidden_dims else feature_dim
        self.prediction_head = MLP(
            input_dim=encoder_out_dim,
            hidden_dims=prediction_hidden_dims,
            output_dim=num_outputs,
            activation=activation,
            use_batch_norm=False,
            dropout=0.0,
        )

    def select_conformer(
        self,
        x: torch.Tensor,
        energies: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Select a single conformer from each ensemble.

        Args:
            x: Conformer features (batch, n_conformers, feature_dim)
            energies: Conformer energies (batch, n_conformers)

        Returns:
            Selected conformer features (batch, feature_dim)
        """
        batch_size, n_conf, _ = x.shape

        if self.selection_method == "lowest_energy" and energies is not None:
            # Select conformer with lowest energy
            indices = energies.argmin(dim=1)  # (batch,)
        elif self.selection_method == "random":
            # Random selection (different per batch element)
            indices = torch.randint(0, n_conf, (batch_size,), device=x.device)
        elif self.selection_method == "centroid":
            # Select conformer closest to mean (geometric centroid)
            mean_features = x.mean(dim=1, keepdim=True)  # (batch, 1, feature_dim)
            distances = torch.norm(x - mean_features, dim=2)  # (batch, n_conformers)
            indices = distances.argmin(dim=1)  # (batch,)
        else:  # first
            # Use first conformer
            indices = torch.zeros(batch_size, dtype=torch.long, device=x.device)

        # Gather selected conformers
        indices = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[-1])
        selected = x.gather(1, indices).squeeze(1)

        return selected

    def forward(
        self,
        x: torch.Tensor,
        energies: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Conformer features (batch, n_conformers, feature_dim)
            energies: Optional conformer energies
            mask: Optional mask (unused, for API compatibility)

        Returns:
            Predictions (batch, num_outputs)
        """
        # Select single conformer
        selected = self.select_conformer(x, energies)

        # Encode
        encoded = self.encoder(selected)

        # Predict
        predictions = self.prediction_head(encoded)

        return predictions


# Alias for backward compatibility
SingleConformerBaseline = SingleConformer


class MeanFeatureAggregation(nn.Module):
    """
    Mean Feature Aggregation (MFA) baseline.

    Averages conformer features BEFORE passing to the network.
    This is distinct from MeanEnsemble which averages predictions AFTER
    processing each conformer independently.

    MFA represents the "better mean estimation" component in the 80/20
    decomposition study. It captures ensemble benefits from averaging
    conformer features without learning complex aggregation functions.

    Key distinction:
    - MFA: mean(features) -> network -> prediction
    - MeanEnsemble: features -> network -> mean(predictions)
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dims: List[int] = [256, 128],
        prediction_hidden_dims: List[int] = [64, 32],
        num_outputs: int = 1,
        output_dim: int = None,  # Alias for num_outputs
        task: str = "regression",
        aggregation: str = "mean",  # 'mean' or 'boltzmann'
        temperature: float = 300.0,
        activation: str = "relu",
        use_batch_norm: bool = True,
        dropout: float = 0.1,
    ):
        """
        Initialize MFA model.

        Args:
            feature_dim: Dimension of conformer features
            hidden_dims: Hidden dimensions for feature encoder
            prediction_hidden_dims: Hidden dimensions for prediction head
            num_outputs: Number of output predictions
            output_dim: Alias for num_outputs (for compatibility)
            task: 'regression' or 'classification'
            aggregation: 'mean' for uniform weights, 'boltzmann' for energy weights
            temperature: Boltzmann temperature (only used if aggregation='boltzmann')
            activation: Activation function
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout rate
        """
        super().__init__()

        self.task = task
        self.aggregation = aggregation
        self.kB = 0.001987204  # kcal/(mol*K)
        self.temperature = temperature

        # Handle output_dim alias
        if output_dim is not None:
            num_outputs = output_dim

        # Feature encoder (processes aggregated features)
        self.encoder = MLP(
            input_dim=feature_dim,
            hidden_dims=hidden_dims,
            output_dim=hidden_dims[-1] if hidden_dims else feature_dim,
            activation=activation,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
        )

        # Prediction head
        encoder_out_dim = hidden_dims[-1] if hidden_dims else feature_dim
        self.prediction_head = MLP(
            input_dim=encoder_out_dim,
            hidden_dims=prediction_hidden_dims,
            output_dim=num_outputs,
            activation=activation,
            use_batch_norm=False,
            dropout=0.0,
        )

    def compute_weights(
        self,
        energies: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        batch_size: int,
        n_conf: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute aggregation weights.

        Args:
            energies: Conformer energies (batch, n_conformers)
            mask: Mask for valid conformers
            batch_size: Batch size
            n_conf: Number of conformers
            device: Device

        Returns:
            Weights (batch, n_conformers)
        """
        if self.aggregation == "boltzmann" and energies is not None:
            kT = self.kB * self.temperature

            # Mask invalid conformers
            if mask is not None:
                energies = energies.masked_fill(~mask, float("inf"))

            # Shift to prevent overflow
            min_energy = energies.min(dim=1, keepdim=True)[0]
            shifted = energies - min_energy

            # Boltzmann weights
            weights = torch.exp(-shifted / kT)

            if mask is not None:
                weights = weights * mask.float()

            # Normalize
            weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        else:
            # Uniform weights
            if mask is not None:
                weights = mask.float()
                weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                weights = torch.ones(batch_size, n_conf, device=device) / n_conf

        return weights

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        energies: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Conformer features (batch, n_conformers, feature_dim)
            mask: Mask for valid conformers
            energies: Conformer energies (for Boltzmann weighting)

        Returns:
            Predictions (batch, num_outputs)
        """
        batch_size, n_conf, feat_dim = x.shape

        # Compute aggregation weights
        weights = self.compute_weights(energies, mask, batch_size, n_conf, x.device)

        # Aggregate features BEFORE network (this is the key difference from MeanEnsemble)
        weights_expanded = weights.unsqueeze(-1)  # (batch, n_conf, 1)
        aggregated_features = (x * weights_expanded).sum(dim=1)  # (batch, feat_dim)

        # Encode aggregated features
        encoded = self.encoder(aggregated_features)

        # Predict
        predictions = self.prediction_head(encoded)

        return predictions


class MultiInstanceLearning(nn.Module):
    """
    Multi-Instance Learning (MIL) baseline for conformer aggregation.

    Implements the classic MIL paradigm where:
    1. Each conformer (instance) is encoded independently
    2. Instance representations are aggregated via permutation-invariant pooling
    3. Aggregated representation is used for prediction

    Key pooling options:
    - 'max': Classic MIL assumption (most important instance dominates)
    - 'attention': Learned weighted combination (Ilse et al., 2018)
    - 'mean': Simple averaging
    - 'lse': Log-sum-exp (smooth approximation to max)
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dims: List[int] = [256, 128],
        prediction_hidden_dims: List[int] = [64, 32],
        num_outputs: int = 1,
        output_dim: int = None,
        task: str = "regression",
        pooling: str = "attention",  # 'max', 'attention', 'mean', 'lse'
        attention_hidden_dim: int = 64,
        lse_temperature: float = 1.0,
        activation: str = "relu",
        use_batch_norm: bool = True,
        dropout: float = 0.1,
    ):
        """
        Initialize MIL model.

        Args:
            feature_dim: Dimension of conformer features
            hidden_dims: Hidden dimensions for instance encoder
            prediction_hidden_dims: Hidden dimensions for bag-level prediction
            num_outputs: Number of outputs
            output_dim: Alias for num_outputs
            task: 'regression' or 'classification'
            pooling: Pooling method ('max', 'attention', 'mean', 'lse')
            attention_hidden_dim: Hidden dim for attention network
            lse_temperature: Temperature for log-sum-exp pooling
            activation: Activation function
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout rate
        """
        super().__init__()

        self.task = task
        self.pooling = pooling
        self.lse_temperature = lse_temperature

        if output_dim is not None:
            num_outputs = output_dim

        # Instance encoder (shared across all conformers)
        self.instance_encoder = MLP(
            input_dim=feature_dim,
            hidden_dims=hidden_dims,
            output_dim=hidden_dims[-1] if hidden_dims else feature_dim,
            activation=activation,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
        )

        encoder_out_dim = hidden_dims[-1] if hidden_dims else feature_dim

        # Attention mechanism for attention-based MIL
        if pooling == "attention":
            self.attention_V = nn.Sequential(
                nn.Linear(encoder_out_dim, attention_hidden_dim),
                nn.Tanh(),
            )
            self.attention_U = nn.Sequential(
                nn.Linear(encoder_out_dim, attention_hidden_dim),
                nn.Sigmoid(),
            )
            self.attention_W = nn.Linear(attention_hidden_dim, 1)

        # Bag-level prediction head
        self.prediction_head = MLP(
            input_dim=encoder_out_dim,
            hidden_dims=prediction_hidden_dims,
            output_dim=num_outputs,
            activation=activation,
            use_batch_norm=False,
            dropout=0.0,
        )

    def compute_attention(
        self,
        encoded: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute gated attention weights (Ilse et al., 2018).

        Args:
            encoded: Encoded instances (batch, n_instances, hidden_dim)
            mask: Mask for valid instances (boolean or 0/1)

        Returns:
            Attention weights (batch, n_instances)
        """
        batch_size, n_inst, hidden_dim = encoded.shape

        # Flatten for attention computation
        h = encoded.view(-1, hidden_dim)

        # Gated attention: a = softmax(w^T (tanh(Vh) * sigmoid(Uh)))
        A_V = self.attention_V(h)  # (batch*n_inst, attn_hidden)
        A_U = self.attention_U(h)  # (batch*n_inst, attn_hidden)
        A = self.attention_W(A_V * A_U)  # (batch*n_inst, 1)
        A = A.view(batch_size, n_inst)  # (batch, n_inst)

        # Apply mask (convert to boolean if needed)
        if mask is not None:
            mask_bool = mask.bool() if mask.dtype != torch.bool else mask
            A = A.masked_fill(~mask_bool, float("-inf"))

        # Softmax to get weights
        weights = F.softmax(A, dim=1)

        return weights

    def pool(
        self,
        encoded: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Pool instance representations.

        Args:
            encoded: Encoded instances (batch, n_instances, hidden_dim)
            mask: Mask for valid instances (boolean or 0/1)

        Returns:
            Pooled representation (batch, hidden_dim) and optional weights
        """
        if self.pooling == "max":
            if mask is not None:
                # Mask invalid instances with -inf (convert mask to boolean)
                mask_bool = mask.bool() if mask.dtype != torch.bool else mask
                encoded = encoded.masked_fill(~mask_bool.unsqueeze(-1), float("-inf"))
            pooled, _ = encoded.max(dim=1)
            return pooled, None

        elif self.pooling == "mean":
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                pooled = (encoded * mask_expanded).sum(dim=1)
                pooled = pooled / mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                pooled = encoded.mean(dim=1)
            return pooled, None

        elif self.pooling == "lse":
            # Log-sum-exp: smooth approximation to max
            scaled = encoded / self.lse_temperature
            if mask is not None:
                # Convert mask to boolean for bitwise NOT
                mask_bool = mask.bool() if mask.dtype != torch.bool else mask
                scaled = scaled.masked_fill(~mask_bool.unsqueeze(-1), float("-inf"))
            pooled = self.lse_temperature * torch.logsumexp(scaled, dim=1)
            return pooled, None

        elif self.pooling == "attention":
            weights = self.compute_attention(encoded, mask)
            pooled = (encoded * weights.unsqueeze(-1)).sum(dim=1)
            return pooled, weights

        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Instance features (batch, n_instances, feature_dim)
            mask: Mask for valid instances
            return_attention: Whether to return attention weights

        Returns:
            Predictions (batch, num_outputs), optionally with attention weights
        """
        batch_size, n_inst, feat_dim = x.shape

        # Encode all instances
        x_flat = x.view(-1, feat_dim)
        encoded = self.instance_encoder(x_flat)
        encoded = encoded.view(batch_size, n_inst, -1)

        # Pool instances
        pooled, weights = self.pool(encoded, mask)

        # Predict
        predictions = self.prediction_head(pooled)

        if return_attention and weights is not None:
            return predictions, weights
        return predictions


# Alias for backward compatibility
MILBaseline = MultiInstanceLearning


class MeanEnsemble(nn.Module):
    """
    Mean ensemble baseline.

    Processes each conformer independently and averages predictions.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dims: List[int] = [256, 128],
        prediction_hidden_dims: List[int] = [64, 32],
        num_outputs: int = 1,
        share_weights: bool = True,
        activation: str = "relu",
        use_batch_norm: bool = True,
        dropout: float = 0.1,
    ):
        """
        Initialize mean ensemble model.

        Args:
            feature_dim: Dimension of conformer features
            hidden_dims: Hidden dimensions for conformer network
            prediction_hidden_dims: Hidden dimensions for prediction head
            num_outputs: Number of outputs
            share_weights: Whether to share weights across conformers
            activation: Activation function
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout rate
        """
        super().__init__()

        self.share_weights = share_weights

        # Conformer network
        self.conformer_net = MLP(
            input_dim=feature_dim,
            hidden_dims=hidden_dims,
            output_dim=hidden_dims[-1] if hidden_dims else feature_dim,
            activation=activation,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
        )

        # Prediction head
        encoder_out_dim = hidden_dims[-1] if hidden_dims else feature_dim
        self.prediction_head = MLP(
            input_dim=encoder_out_dim,
            hidden_dims=prediction_hidden_dims,
            output_dim=num_outputs,
            activation=activation,
            use_batch_norm=False,
            dropout=0.0,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Conformer features (batch, n_conformers, feature_dim)
            mask: Mask for valid conformers

        Returns:
            Predictions (batch, num_outputs)
        """
        batch_size, n_conf, feat_dim = x.shape

        # Process each conformer
        x_flat = x.view(-1, feat_dim)
        encoded = self.conformer_net(x_flat)
        encoded = encoded.view(batch_size, n_conf, -1)

        # Get predictions for each conformer
        pred_flat = self.prediction_head(encoded.view(-1, encoded.shape[-1]))
        predictions = pred_flat.view(batch_size, n_conf, -1)

        # Average predictions
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            predictions = (predictions * mask_expanded).sum(dim=1)
            predictions = predictions / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            predictions = predictions.mean(dim=1)

        return predictions


class BoltzmannEnsemble(nn.Module):
    """
    Boltzmann ensemble baseline.

    Weights conformer predictions by Boltzmann factors exp(-E/kT).
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dims: List[int] = [256, 128],
        prediction_hidden_dims: List[int] = [64, 32],
        num_outputs: int = 1,
        temperature: float = 300.0,
        trainable_temperature: bool = False,
        activation: str = "relu",
        use_batch_norm: bool = True,
        dropout: float = 0.1,
    ):
        """
        Initialize Boltzmann ensemble model.

        Args:
            feature_dim: Dimension of conformer features
            hidden_dims: Hidden dimensions
            prediction_hidden_dims: Hidden dims for prediction
            num_outputs: Number of outputs
            temperature: Boltzmann temperature in Kelvin
            trainable_temperature: Whether temperature is learnable
            activation: Activation function
            use_batch_norm: Whether to use batch norm
            dropout: Dropout rate
        """
        super().__init__()

        # Boltzmann constant in kcal/(mol*K)
        self.kB = 0.001987204  # kcal/(mol*K)

        if trainable_temperature:
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        else:
            self.register_buffer("log_temperature", torch.log(torch.tensor(temperature)))

        # Conformer network
        self.conformer_net = MLP(
            input_dim=feature_dim,
            hidden_dims=hidden_dims,
            output_dim=hidden_dims[-1] if hidden_dims else feature_dim,
            activation=activation,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
        )

        # Prediction head
        encoder_out_dim = hidden_dims[-1] if hidden_dims else feature_dim
        self.prediction_head = MLP(
            input_dim=encoder_out_dim,
            hidden_dims=prediction_hidden_dims,
            output_dim=num_outputs,
            activation=activation,
            use_batch_norm=False,
            dropout=0.0,
        )

    @property
    def temperature(self) -> float:
        """Get current temperature."""
        return torch.exp(self.log_temperature).item()

    def compute_boltzmann_weights(
        self,
        energies: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Boltzmann weights from energies.

        Args:
            energies: Conformer energies in kcal/mol (batch, n_conformers)
            mask: Mask for valid conformers

        Returns:
            Normalized Boltzmann weights (batch, n_conformers)
        """
        temperature = torch.exp(self.log_temperature)
        kT = self.kB * temperature

        # Shift energies to prevent overflow (subtract minimum)
        if mask is not None:
            energies = energies.masked_fill(~mask, float("inf"))

        min_energy = energies.min(dim=1, keepdim=True)[0]
        shifted = energies - min_energy

        # Compute Boltzmann factors
        boltzmann = torch.exp(-shifted / kT)

        if mask is not None:
            boltzmann = boltzmann * mask.float()

        # Normalize
        weights = boltzmann / boltzmann.sum(dim=1, keepdim=True).clamp(min=1e-8)

        return weights

    def forward(
        self,
        x: torch.Tensor,
        energies: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Conformer features (batch, n_conformers, feature_dim)
            energies: Conformer energies in kcal/mol (batch, n_conformers)
            mask: Mask for valid conformers

        Returns:
            Predictions (batch, num_outputs)
        """
        batch_size, n_conf, feat_dim = x.shape

        # Compute Boltzmann weights
        weights = self.compute_boltzmann_weights(energies, mask)

        # Process each conformer
        x_flat = x.view(-1, feat_dim)
        encoded = self.conformer_net(x_flat)
        encoded = encoded.view(batch_size, n_conf, -1)

        # Get predictions for each conformer
        pred_flat = self.prediction_head(encoded.view(-1, encoded.shape[-1]))
        predictions = pred_flat.view(batch_size, n_conf, -1)

        # Weighted average
        weights = weights.unsqueeze(-1)  # (batch, n_conf, 1)
        weighted_pred = (predictions * weights).sum(dim=1)

        return weighted_pred

    def get_conformer_importances(
        self,
        energies: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get Boltzmann weights as importance scores."""
        return self.compute_boltzmann_weights(energies, mask)


class LearnedWeightEnsemble(nn.Module):
    """
    Ensemble with learned conformer weights.

    Similar to attention, but weights are computed from a separate
    network rather than through dot-product attention.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dims: List[int] = [256, 128],
        weight_hidden_dims: List[int] = [64],
        prediction_hidden_dims: List[int] = [64, 32],
        num_outputs: int = 1,
        temperature: float = 1.0,
        activation: str = "relu",
        use_batch_norm: bool = True,
        dropout: float = 0.1,
    ):
        """
        Initialize learned weight ensemble.

        Args:
            feature_dim: Feature dimension
            hidden_dims: Hidden dims for feature encoder
            weight_hidden_dims: Hidden dims for weight network
            prediction_hidden_dims: Hidden dims for prediction
            num_outputs: Number of outputs
            temperature: Temperature for softmax
            activation: Activation function
            use_batch_norm: Whether to use batch norm
            dropout: Dropout rate
        """
        super().__init__()

        self.temperature = temperature

        # Feature encoder
        self.encoder = MLP(
            input_dim=feature_dim,
            hidden_dims=hidden_dims,
            output_dim=hidden_dims[-1] if hidden_dims else feature_dim,
            activation=activation,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
        )

        # Weight network
        encoder_out_dim = hidden_dims[-1] if hidden_dims else feature_dim
        self.weight_net = MLP(
            input_dim=encoder_out_dim,
            hidden_dims=weight_hidden_dims,
            output_dim=1,
            activation=activation,
            use_batch_norm=False,
            dropout=0.0,
        )

        # Prediction head
        self.prediction_head = MLP(
            input_dim=encoder_out_dim,
            hidden_dims=prediction_hidden_dims,
            output_dim=num_outputs,
            activation=activation,
            use_batch_norm=False,
            dropout=0.0,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Conformer features (batch, n_conformers, feature_dim)
            mask: Mask for valid conformers
            return_weights: Whether to return learned weights

        Returns:
            Predictions and optionally weights
        """
        batch_size, n_conf, feat_dim = x.shape

        # Encode conformers
        x_flat = x.view(-1, feat_dim)
        encoded = self.encoder(x_flat)
        encoded = encoded.view(batch_size, n_conf, -1)

        # Compute weights
        weight_scores = self.weight_net(encoded.view(-1, encoded.shape[-1]))
        weight_scores = weight_scores.view(batch_size, n_conf)
        weight_scores = weight_scores / self.temperature

        if mask is not None:
            weight_scores = weight_scores.masked_fill(~mask, float("-inf"))

        weights = F.softmax(weight_scores, dim=1)

        # Weighted aggregation of encoded features
        weighted_encoded = (encoded * weights.unsqueeze(-1)).sum(dim=1)

        # Predict
        predictions = self.prediction_head(weighted_encoded)

        if return_weights:
            return predictions, weights
        return predictions, None
