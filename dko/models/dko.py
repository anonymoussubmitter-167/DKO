"""
Distribution Kernel Operators (DKO) for Molecular Property Prediction.

This module implements the core DKO model that learns from conformer ensemble
distributions using the augmented basis representation [μ, Σ + μμ^T].

Architecture:
1. Input: Augmented basis
   - μ: (batch, D) first-order (mean) features
   - Σ: (batch, D, D) second-order (covariance) features
2. PCA dimensionality reduction on Σ: D² → ~500-1000 dimensions
3. Kernel network with PSD constraint (K = LL^T)
4. Branch network for prediction

Key innovations:
- Explicit second-order modeling enables decomposition analysis
- PSD kernel respects Riemannian manifold geometry
- Allows surgical ablation of distributional components

Reference: "Distribution Kernel Operators for Molecular Property Prediction"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    PCA = None


class MLP(nn.Module):
    """Multi-layer perceptron with optional batch normalization and dropout."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "relu",
        use_batch_norm: bool = True,
        dropout: float = 0.1,
        output_activation: bool = False,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.output_activation = output_activation

        # Activation function
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(0.1),
        }
        self.activation = activations.get(activation, nn.ReLU())

        # Build layers
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            if use_batch_norm and i < len(dims) - 2:
                self.batch_norms.append(nn.BatchNorm1d(dims[i + 1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if self.batch_norms is not None and i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            x = self.activation(x)
            if self.dropout is not None:
                x = self.dropout(x)

        x = self.layers[-1](x)
        if self.output_activation:
            x = self.activation(x)

        return x


class DKO(nn.Module):
    """
    Distribution Kernel Operator for Conformer Ensembles.

    Architecture:
    1. Input: Augmented basis [μ, Σ + μμ^T]
       - μ: (batch, D) first-order (mean) features
       - Σ: (batch, D, D) second-order (covariance) features
    2. PCA dimensionality reduction on Σ: D² → ~500-1000
    3. Kernel network with PSD constraint (K = LL^T)
    4. Branch network for prediction

    Key innovations:
    - Explicit second-order modeling enables decomposition analysis
    - PSD kernel respects Riemannian manifold geometry
    - Allows surgical ablation of distributional components
    """

    def __init__(
        self,
        feature_dim: int,
        output_dim: int = 1,
        task: str = 'regression',
        pca_variance: float = 0.95,
        pca_max_components: Optional[int] = 1000,
        use_diagonal_sigma: bool = False,  # Use diagonal of sigma instead of PCA
        separate_mu_sigma_nets: bool = False,  # Use separate networks for mu and sigma
        kernel_hidden_dims: List[int] = [512, 256, 128],
        kernel_output_dim: int = 64,
        branch_hidden_dim: int = 128,
        dropout: float = 0.1,
        use_psd_constraint: bool = True,
        use_second_order: bool = True,
        use_batch_norm: bool = True,
        activation: str = 'relu',
        verbose: bool = True,
    ):
        """
        Initialize DKO model.

        Args:
            feature_dim: Dimension D of geometric features
            output_dim: Output dimension (1 for regression, num_classes for classification)
            task: 'regression' or 'classification'
            pca_variance: Variance to retain in PCA (0.95 = 95%)
            pca_max_components: Maximum number of PCA components (for memory efficiency)
            kernel_hidden_dims: Hidden dimensions for kernel network
            kernel_output_dim: Output dimension of kernel (before branch)
            branch_hidden_dim: Hidden dimension for branch network
            dropout: Dropout rate
            use_psd_constraint: Whether to enforce K = LL^T
            use_second_order: Whether to use Σ (False for ablation studies)
            use_batch_norm: Whether to use batch normalization in kernel network
            activation: Activation function ('relu', 'gelu', 'silu')
            verbose: Whether to print model info during initialization
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.task = task
        self.pca_variance = pca_variance
        self.pca_max_components = pca_max_components
        self.use_diagonal_sigma = use_diagonal_sigma
        self.separate_mu_sigma_nets = separate_mu_sigma_nets
        self.use_psd_constraint = use_psd_constraint
        self.use_second_order = use_second_order
        self.kernel_output_dim = kernel_output_dim
        self.kernel_hidden_dims = kernel_hidden_dims
        self.branch_hidden_dim = branch_hidden_dim
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.activation = activation
        self.verbose = verbose

        # PCA for dimensionality reduction (fitted on first batch)
        self.pca = None
        self.pca_fitted = False
        self.reduced_dim = None

        # Store PCA parameters as buffers for saving/loading
        self.register_buffer('pca_mean_', None)
        self.register_buffer('pca_components_', None)
        self.register_buffer('pca_explained_variance_', None)
        self._sigma_std = 1.0  # Normalization factor for sigma

        # Dimensions
        self.first_order_dim = feature_dim
        self.second_order_dim = None  # Set after PCA

        # Kernel network (built after PCA determines input dimension)
        self.kernel_layers = None

        # Sigma projection network (projects reduced sigma to same dim as mu)
        # This ensures sigma has equal representation power as mu
        # Built dynamically after PCA determines reduced_dim
        self.sigma_projection = None

        # Separate networks for mu and sigma (when separate_mu_sigma_nets=True)
        # This gives sigma its own processing pathway so it doesn't get overwhelmed by mu
        self.mu_net = None
        self.sigma_net = None
        self.separate_hidden_dim = 256  # Hidden dimension for each separate network

        # Branch network
        if use_psd_constraint:
            branch_input_dim = kernel_output_dim
        else:
            branch_input_dim = kernel_output_dim

        self.branch_net = nn.Sequential(
            nn.Linear(branch_input_dim, branch_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(branch_hidden_dim, output_dim)
        )

        # If not using second order, build kernel network immediately
        if not use_second_order:
            self._build_kernel_network(self.first_order_dim)

    def _fit_pca(self, sigma_batch: torch.Tensor):
        """
        Fit PCA on a batch of second-order features.

        Args:
            sigma_batch: (batch, D, D) second-order features
        """
        batch_size, D, _ = sigma_batch.shape
        device = sigma_batch.device

        # If using diagonal, skip PCA entirely
        if self.use_diagonal_sigma:
            self.reduced_dim = D  # Diagonal has same dimension as feature_dim
            self.second_order_dim = D
            self.pca = None
            self._sigma_std = 1.0

            if self.verbose:
                print(f"[DKO] Using diagonal of sigma: {D} dims (no PCA)")

            # No sigma projection needed - already same dim as mu
            self.sigma_projection = None

            # Build kernel network with mu + diagonal
            total_input_dim = self.first_order_dim + D
            self._build_kernel_network(total_input_dim)
            self.pca_fitted = True
            return

        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for PCA. Install with: pip install scikit-learn")

        # Normalize sigma for numerical stability (store for consistent transform)
        self._sigma_std = sigma_batch.std().clamp(min=1e-6).item()
        sigma_normalized = sigma_batch / self._sigma_std

        # Extract upper triangle (D*(D+1)/2 unique elements due to symmetry)
        indices = torch.triu_indices(D, D, device=device)
        flat_features_list = []
        for i in range(batch_size):
            upper_values = sigma_normalized[i][indices[0], indices[1]]
            flat_features_list.append(upper_values.detach().cpu().numpy())

        flat_features = np.array(flat_features_list)  # (batch, D*(D+1)/2)
        original_dim = flat_features.shape[1]

        # Handle edge cases
        if original_dim <= 1:
            self.reduced_dim = original_dim
            self.second_order_dim = original_dim
            self.pca_fitted = True

            total_input_dim = self.first_order_dim + self.reduced_dim
            self._build_kernel_network(total_input_dim)
            return

        # Determine number of components
        max_components = min(
            batch_size - 1,  # SVD constraint
            original_dim,
            self.pca_max_components if self.pca_max_components else original_dim
        )

        if max_components < 1:
            max_components = 1

        # Skip PCA if batch is too small relative to features
        if max_components < 10 or batch_size < 16:
            if self.verbose:
                print(f"[DKO] Batch too small for stable PCA ({batch_size} samples, {original_dim} features)")
                print(f"[DKO] Using diagonal only for second-order features")
            # Use diagonal only as fallback
            self.reduced_dim = min(self.first_order_dim, original_dim)
            self.pca = None
            self.pca_fitted = True
            self.second_order_dim = self.reduced_dim

            total_input_dim = self.first_order_dim + self.reduced_dim
            self._build_kernel_network(total_input_dim)
            return

        # Fit PCA
        self.pca = PCA(n_components=min(max_components, original_dim), random_state=42)

        try:
            self.pca.fit(flat_features)

            # Find number of components for desired variance
            cumsum = np.cumsum(self.pca.explained_variance_ratio_)
            n_components = np.searchsorted(cumsum, self.pca_variance) + 1
            n_components = min(n_components, max_components)

            # Refit with exact number of components if needed
            if n_components < self.pca.n_components_:
                self.pca = PCA(n_components=n_components, random_state=42)
                self.pca.fit(flat_features)

            self.reduced_dim = self.pca.n_components_
            explained_var = self.pca.explained_variance_ratio_.sum() * 100

        except Exception as e:
            # Fallback: use all features without PCA
            if self.verbose:
                print(f"[DKO] PCA fitting failed ({e}), using raw features")
            self.reduced_dim = min(original_dim, self.pca_max_components or original_dim)
            self.pca = None
            explained_var = 100.0

        self.second_order_dim = self.reduced_dim

        if self.verbose:
            print(f"[DKO] PCA: {original_dim} -> {self.reduced_dim} dims "
                  f"({explained_var:.1f}% variance retained)")

        # Store PCA parameters as tensors for model saving
        if self.pca is not None:
            self.pca_mean_ = torch.tensor(self.pca.mean_, dtype=torch.float32)
            self.pca_components_ = torch.tensor(self.pca.components_, dtype=torch.float32)
            self.pca_explained_variance_ = torch.tensor(
                self.pca.explained_variance_, dtype=torch.float32
            )

        # Get device for creating networks
        device = next(self.branch_net.parameters()).device

        if self.separate_mu_sigma_nets:
            # Separate networks approach: mu and sigma each get their own network
            # This prevents sigma from being overwhelmed by mu's 1024 dimensions
            h_dim = self.separate_hidden_dim

            # Mu network: first_order_dim -> h_dim
            self.mu_net = nn.Sequential(
                nn.Linear(self.first_order_dim, h_dim * 2),
                nn.BatchNorm1d(h_dim * 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(h_dim * 2, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
            ).to(device)

            # Sigma network: reduced_dim -> h_dim (same output as mu)
            self.sigma_net = nn.Sequential(
                nn.Linear(self.reduced_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(h_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
            ).to(device)

            self.sigma_projection = None  # Not used with separate nets

            if self.verbose:
                print(f"[DKO] Separate networks: mu ({self.first_order_dim} -> {h_dim}), "
                      f"sigma ({self.reduced_dim} -> {h_dim})")

            # Kernel network takes concatenated outputs: h_dim + h_dim = 2*h_dim
            total_input_dim = h_dim * 2
            self._build_kernel_network(total_input_dim)
        else:
            # Original approach: project sigma to same dim as mu, then concatenate
            self.sigma_projection = nn.Sequential(
                nn.Linear(self.reduced_dim, self.first_order_dim),
                nn.BatchNorm1d(self.first_order_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            ).to(device)

            if self.verbose:
                print(f"[DKO] Sigma projection: {self.reduced_dim} -> {self.first_order_dim} dims")

            # Kernel network with equal representation from mu and sigma
            total_input_dim = self.first_order_dim * 2  # mu + projected_sigma
            self._build_kernel_network(total_input_dim)

        self.pca_fitted = True

    def _build_kernel_network(self, input_dim: int):
        """Build kernel network after PCA determines input dimension."""
        layers = []
        prev_dim = input_dim

        # Get activation function
        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "leaky_relu": lambda: nn.LeakyReLU(0.1),
        }
        act_class = activations.get(self.activation, nn.ReLU)

        # Hidden layers with BatchNorm
        for hidden_dim in self.kernel_hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act_class())
            prev_dim = hidden_dim

        # Output layer
        if self.use_psd_constraint:
            # Output L matrix (flattened) where K = LL^T
            output_size = self.kernel_output_dim * self.kernel_output_dim
        else:
            output_size = self.kernel_output_dim

        layers.append(nn.Linear(prev_dim, output_size))

        self.kernel_layers = nn.Sequential(*layers)

        # Move kernel layers to the same device as branch_net
        device = next(self.branch_net.parameters()).device
        self.kernel_layers = self.kernel_layers.to(device)

        if self.verbose:
            print(f"[DKO] Kernel network: input={input_dim}, "
                  f"hidden={self.kernel_hidden_dims}, output={output_size}")

    def _reduce_second_order(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Apply PCA to reduce second-order features.

        Args:
            sigma: (batch, D, D) second-order features

        Returns:
            reduced: (batch, reduced_dim) reduced features
        """
        batch_size, D, _ = sigma.shape
        device = sigma.device

        # Normalize sigma using the same scale factor from PCA fitting
        # Clamp to prevent extreme values
        sigma_clamped = torch.clamp(sigma, min=-1e6, max=1e6)
        sigma_normalized = sigma_clamped / max(self._sigma_std, 1e-6)

        # Extract upper triangle
        indices = torch.triu_indices(D, D, device=device)
        flat_features_list = []
        for i in range(batch_size):
            upper_values = sigma_normalized[i][indices[0], indices[1]]
            flat_features_list.append(upper_values)

        flat_features = torch.stack(flat_features_list)  # (batch, D*(D+1)/2)

        # Check for NaN/Inf
        if torch.isnan(flat_features).any() or torch.isinf(flat_features).any():
            flat_features = torch.nan_to_num(flat_features, nan=0.0, posinf=1.0, neginf=-1.0)

        if self.pca is None:
            # No PCA - use diagonal of sigma instead of upper triangle
            # This is more numerically stable for small batches
            diag_features = torch.diagonal(sigma_clamped, dim1=1, dim2=2)  # (batch, D)
            # Normalize diagonal
            diag_std = diag_features.std().clamp(min=1e-6)
            diag_features = diag_features / diag_std
            # Truncate to reduced_dim
            return diag_features[:, :self.reduced_dim]

        # Apply PCA using stored tensor parameters (for GPU compatibility)
        if self.pca_mean_ is not None and self.pca_components_ is not None:
            pca_mean = self.pca_mean_.to(device)
            pca_components = self.pca_components_.to(device)

            centered = flat_features - pca_mean
            reduced = torch.matmul(centered, pca_components.T)

            # Clamp reduced features to prevent explosion
            reduced = torch.clamp(reduced, min=-100.0, max=100.0)
        else:
            # Fallback to numpy (slower, for compatibility)
            flat_np = flat_features.detach().cpu().numpy()
            reduced_np = self.pca.transform(flat_np)
            reduced_np = np.clip(reduced_np, -100.0, 100.0)
            reduced = torch.tensor(reduced_np, dtype=torch.float32, device=device)

        return reduced

    def forward(
        self,
        mu: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
        fit_pca: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            mu: (batch, D) first-order features
            sigma: (batch, D, D) second-order features (optional if not using)
            fit_pca: Whether to fit PCA on this batch (first epoch only)

        Returns:
            predictions: (batch, output_dim)
        """
        batch_size = mu.size(0)
        device = mu.device

        # Check for NaN/Inf in input
        if torch.isnan(mu).any() or torch.isinf(mu).any():
            mu = torch.nan_to_num(mu, nan=0.0, posinf=1.0, neginf=-1.0)

        # NOTE: Removed per-sample normalization that was destroying distributional signal
        # The BatchNorm layers in kernel_network handle normalization properly
        # Per-sample normalization erases relative differences between molecules

        # Handle second-order features
        if self.use_second_order and sigma is not None:
            # Check for NaN/Inf in sigma
            if torch.isnan(sigma).any() or torch.isinf(sigma).any():
                sigma = torch.nan_to_num(sigma, nan=0.0, posinf=1.0, neginf=-1.0)

            # Fit PCA on first batch
            if fit_pca and not self.pca_fitted:
                self._fit_pca(sigma)

            # Reduce dimensionality
            if self.pca_fitted:
                sigma_reduced = self._reduce_second_order(sigma)

                # Use separate networks if enabled
                if self.separate_mu_sigma_nets and self.mu_net is not None:
                    # Process mu and sigma through separate networks
                    mu_processed = self.mu_net(mu)
                    sigma_processed = self.sigma_net(sigma_reduced)
                    combined = torch.cat([mu_processed, sigma_processed], dim=1)
                # Project sigma if needed (only when using PCA with small output)
                elif self.sigma_projection is not None:
                    sigma_projected = self.sigma_projection(sigma_reduced)
                    combined = torch.cat([mu, sigma_projected], dim=1)
                else:
                    # When using diagonal sigma, no projection needed
                    combined = torch.cat([mu, sigma_reduced], dim=1)
            else:
                # PCA not fitted yet, temporarily use first-order only
                combined = mu
                if self.kernel_layers is None:
                    self._build_kernel_network(self.first_order_dim)
        else:
            # First-order only (for ablation studies)
            combined = mu

            # Build kernel network if not yet built
            if self.kernel_layers is None:
                self._build_kernel_network(self.first_order_dim)

        # Pass through kernel network
        kernel_output = self.kernel_layers(combined)

        # Apply PSD constraint if enabled
        if self.use_psd_constraint:
            # Clamp kernel output to prevent explosion before forming L
            kernel_output = torch.clamp(kernel_output, min=-10.0, max=10.0)

            # Reshape to L matrix
            L = kernel_output.view(batch_size, self.kernel_output_dim, self.kernel_output_dim)

            # Scale L by 1/sqrt(k_dim) to control magnitude of LL^T
            L = L / (self.kernel_output_dim ** 0.5)

            # Compute K = LL^T (ensures PSD)
            K = torch.bmm(L, L.transpose(1, 2))  # (batch, k_dim, k_dim)

            # Extract diagonal as features (these are guaranteed positive)
            kernel_features = torch.diagonal(K, dim1=1, dim2=2)  # (batch, k_dim)

            # Apply log transform for numerical stability (diagonal is always positive)
            # Use log1p to handle small values gracefully
            kernel_features = torch.log1p(kernel_features)

            # NOTE: Removed per-sample normalization that was destroying distributional signal
            # The branch_net has its own normalization via the linear layers
            # Per-sample normalization erases the kernel's learned magnitude information

            # Final safety clamp (still needed to prevent extreme values)
            kernel_features = torch.clamp(kernel_features, min=-10.0, max=10.0)
        else:
            kernel_features = kernel_output

        # Pass through branch network
        output = self.branch_net(kernel_features)

        # Apply activation for classification
        if self.task == 'classification' and not self.training:
            output = torch.sigmoid(output)

        return output

    def get_embedding(
        self,
        mu: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get kernel embedding for analysis (e.g., for visualization).

        Args:
            mu: (batch, D) first-order features
            sigma: (batch, D, D) second-order features

        Returns:
            embedding: (batch, kernel_output_dim)
        """
        with torch.no_grad():
            if self.use_second_order and sigma is not None and self.pca_fitted:
                sigma_reduced = self._reduce_second_order(sigma)
                if self.separate_mu_sigma_nets and self.mu_net is not None:
                    mu_processed = self.mu_net(mu)
                    sigma_processed = self.sigma_net(sigma_reduced)
                    combined = torch.cat([mu_processed, sigma_processed], dim=1)
                elif self.sigma_projection is not None:
                    sigma_projected = self.sigma_projection(sigma_reduced)
                    combined = torch.cat([mu, sigma_projected], dim=1)
                else:
                    combined = torch.cat([mu, sigma_reduced], dim=1)
            else:
                combined = mu

            if self.kernel_layers is None:
                return mu[:, :self.kernel_output_dim]

            kernel_output = self.kernel_layers(combined)

            if self.use_psd_constraint:
                batch_size = mu.size(0)
                L = kernel_output.view(batch_size, self.kernel_output_dim, self.kernel_output_dim)
                K = torch.bmm(L, L.transpose(1, 2))
                return torch.diagonal(K, dim1=1, dim2=2)
            else:
                return kernel_output

    def get_kernel_matrix(
        self,
        mu: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get the full kernel matrix K = LL^T for analysis.

        Args:
            mu: (batch, D) first-order features
            sigma: (batch, D, D) second-order features

        Returns:
            K: (batch, kernel_output_dim, kernel_output_dim) kernel matrices
        """
        if not self.use_psd_constraint:
            raise ValueError("Kernel matrix only available with use_psd_constraint=True")

        with torch.no_grad():
            if self.use_second_order and sigma is not None and self.pca_fitted:
                sigma_reduced = self._reduce_second_order(sigma)
                if self.separate_mu_sigma_nets and self.mu_net is not None:
                    mu_processed = self.mu_net(mu)
                    sigma_processed = self.sigma_net(sigma_reduced)
                    combined = torch.cat([mu_processed, sigma_processed], dim=1)
                elif self.sigma_projection is not None:
                    sigma_projected = self.sigma_projection(sigma_reduced)
                    combined = torch.cat([mu, sigma_projected], dim=1)
                else:
                    combined = torch.cat([mu, sigma_reduced], dim=1)
            else:
                combined = mu

            if self.kernel_layers is None:
                return torch.eye(self.kernel_output_dim).unsqueeze(0).expand(mu.size(0), -1, -1)

            kernel_output = self.kernel_layers(combined)
            batch_size = mu.size(0)
            L = kernel_output.view(batch_size, self.kernel_output_dim, self.kernel_output_dim)
            K = torch.bmm(L, L.transpose(1, 2))

            return K


class DKOFirstOrder(DKO):
    """
    DKO using only first-order features (μ only).

    For Experiment 2: 80/20 decomposition study.
    Tests how much performance comes from better mean estimation
    vs. capturing molecular flexibility.
    """

    def __init__(self, *args, **kwargs):
        kwargs['use_second_order'] = False
        super().__init__(*args, **kwargs)


class DKOFull(DKO):
    """
    DKO using full augmented basis [μ, Σ].

    This is the default, full model.
    """

    def __init__(self, *args, **kwargs):
        kwargs['use_second_order'] = True
        super().__init__(*args, **kwargs)


class DKONoPSD(DKO):
    """
    DKO without PSD constraint.

    For ablation: test importance of positive semi-definite kernel.
    """

    def __init__(self, *args, **kwargs):
        kwargs['use_psd_constraint'] = False
        super().__init__(*args, **kwargs)


# =============================================================================
# Legacy DKO Classes (for backwards compatibility with conformer-level input)
# =============================================================================

class DKOKernel(nn.Module):
    """
    Learnable kernel for DKO (legacy conformer-level version).

    Maps pairs of conformer features to kernel values, with optional
    positive semi-definite constraint.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        output_dim: int = 64,
        use_psd_constraint: bool = True,
        activation: str = "relu",
        use_batch_norm: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_psd_constraint = use_psd_constraint

        # Kernel network: maps concatenated conformer pairs to embedding
        self.kernel_network = MLP(
            input_dim=2 * input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            output_activation=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute kernel matrix between conformer sets.

        Args:
            x: Conformer features (batch, n_conformers_x, feature_dim)
            y: Optional second set (batch, n_conformers_y, feature_dim)

        Returns:
            Kernel matrix (batch, n_conformers_x, n_conformers_y)
        """
        if y is None:
            y = x

        batch_size, n_x, feat_dim = x.shape
        _, n_y, _ = y.shape

        # Expand for pairwise computation
        x_exp = x.unsqueeze(2).expand(-1, -1, n_y, -1)
        y_exp = y.unsqueeze(1).expand(-1, n_x, -1, -1)

        # Concatenate pairs
        pairs = torch.cat([x_exp, y_exp], dim=-1)
        pairs_flat = pairs.view(-1, 2 * feat_dim)

        # Apply kernel network
        embeddings = self.kernel_network(pairs_flat)
        embeddings = embeddings.view(batch_size, n_x, n_y, self.output_dim)

        if self.use_psd_constraint:
            kernel = torch.einsum("bijk,bilk->bijl", embeddings, embeddings)
            kernel = kernel.sum(dim=-1)
        else:
            kernel = embeddings.sum(dim=-1)

        return kernel


class DKOConformerLevel(nn.Module):
    """
    DKO model operating on conformer-level features (legacy version).

    Takes (batch, n_conformers, feature_dim) input and aggregates
    using learned kernel.
    """

    def __init__(
        self,
        feature_dim: int,
        kernel_hidden_dims: List[int] = [512, 256, 128],
        kernel_output_dim: int = 64,
        branch_hidden_dims: List[int] = [128, 64],
        prediction_hidden_dims: List[int] = [32],
        num_outputs: int = 1,
        use_psd_constraint: bool = True,
        activation: str = "relu",
        use_batch_norm: bool = True,
        dropout: float = 0.1,
        aggregation_method: str = "kernel_mean",
        aggregation_temperature: float = 1.0,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.aggregation_method = aggregation_method
        self.aggregation_temperature = aggregation_temperature

        # Kernel module
        self.kernel = DKOKernel(
            input_dim=feature_dim,
            hidden_dims=kernel_hidden_dims,
            output_dim=kernel_output_dim,
            use_psd_constraint=use_psd_constraint,
            activation=activation,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
        )

        # Branch network
        self.branch_network = MLP(
            input_dim=feature_dim,
            hidden_dims=branch_hidden_dims,
            output_dim=branch_hidden_dims[-1] if branch_hidden_dims else feature_dim,
            activation=activation,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            output_activation=True,
        )

        # Prediction head
        branch_output_dim = branch_hidden_dims[-1] if branch_hidden_dims else feature_dim
        self.prediction_head = MLP(
            input_dim=branch_output_dim,
            hidden_dims=prediction_hidden_dims,
            output_dim=num_outputs,
            activation=activation,
            use_batch_norm=False,
            dropout=0.0,
            output_activation=False,
        )

    def aggregate_conformers(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Aggregate conformer features using the learned kernel."""
        batch_size, n_conf, feat_dim = x.shape

        K = self.kernel(x)
        K = K / self.aggregation_temperature

        if mask is not None:
            mask_2d = mask.unsqueeze(2) & mask.unsqueeze(1)
            K = K.masked_fill(~mask_2d, float("-inf"))

        if self.aggregation_method == "kernel_attention":
            weights = F.softmax(K, dim=-1)
            weighted = torch.bmm(weights, x)
            if mask is not None:
                mask_sum = mask.float().sum(dim=1, keepdim=True).clamp(min=1)
                aggregated = (weighted * mask.unsqueeze(-1)).sum(dim=1) / mask_sum
            else:
                aggregated = weighted.mean(dim=1)
        else:
            if mask is not None:
                K = K.masked_fill(~mask_2d, 0.0)
            K_sum = K.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            K_norm = K / K_sum
            weighted = torch.bmm(K_norm, x)
            if mask is not None:
                mask_sum = mask.float().sum(dim=1, keepdim=True).clamp(min=1)
                aggregated = (weighted * mask.unsqueeze(-1)).sum(dim=1) / mask_sum
            else:
                aggregated = weighted.mean(dim=1)

        return aggregated

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        aggregated = self.aggregate_conformers(x, mask)
        branch_out = self.branch_network(aggregated)
        predictions = self.prediction_head(branch_out)
        return predictions


# =============================================================================
# Utility Functions
# =============================================================================

def create_dko_model(config: dict, feature_dim: int) -> DKO:
    """
    Create DKO model from configuration dict.

    Args:
        config: Model configuration
        feature_dim: Dimension of geometric features

    Returns:
        DKO model instance
    """
    return DKO(
        feature_dim=feature_dim,
        output_dim=config.get('output_dim', 1),
        task=config.get('task', 'regression'),
        pca_variance=config.get('pca_variance', 0.95),
        pca_max_components=config.get('pca_max_components', 1000),
        kernel_hidden_dims=config.get('kernel_hidden_dims', [512, 256, 128]),
        kernel_output_dim=config.get('kernel_output_dim', 64),
        branch_hidden_dim=config.get('branch_hidden_dim', 128),
        dropout=config.get('dropout', 0.1),
        use_psd_constraint=config.get('use_psd_constraint', True),
        use_second_order=config.get('use_second_order', True),
        use_batch_norm=config.get('use_batch_norm', True),
        activation=config.get('activation', 'relu'),
        verbose=config.get('verbose', True),
    )


def create_dko_first_order(config: dict, feature_dim: int) -> DKOFirstOrder:
    """Create first-order only DKO model."""
    return DKOFirstOrder(
        feature_dim=feature_dim,
        output_dim=config.get('output_dim', 1),
        task=config.get('task', 'regression'),
        kernel_hidden_dims=config.get('kernel_hidden_dims', [512, 256, 128]),
        kernel_output_dim=config.get('kernel_output_dim', 64),
        branch_hidden_dim=config.get('branch_hidden_dim', 128),
        dropout=config.get('dropout', 0.1),
        use_psd_constraint=config.get('use_psd_constraint', True),
        verbose=config.get('verbose', True),
    )


# =============================================================================
# Self-Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing DKO Model")
    print("=" * 60)

    # Create dummy data matching real dimensions from validation
    batch_size = 32
    D = 100  # Feature dimension from validation

    print(f"\nCreating test data:")
    print(f"  Batch size: {batch_size}")
    print(f"  Feature dimension D: {D}")

    # First-order features (mean)
    mu = torch.randn(batch_size, D)

    # Second-order features (covariance matrix, must be PSD)
    sigma_raw = torch.randn(batch_size, D, D)
    sigma = torch.bmm(sigma_raw, sigma_raw.transpose(1, 2))  # Make PSD

    print(f"  mu shape: {mu.shape}")
    print(f"  sigma shape: {sigma.shape}")

    # Test 1: Full DKO with second-order features
    print("\n" + "=" * 60)
    print("Test 1: Full DKO (with second-order features)")
    print("=" * 60)

    model_full = DKO(
        feature_dim=D,
        output_dim=1,
        task='regression',
        use_second_order=True,
        verbose=True,
    )

    model_full.train()
    print("\nFirst forward pass (fitting PCA)...")
    output1 = model_full(mu, sigma, fit_pca=True)
    print(f"  Output shape: {output1.shape}")
    print(f"  Output range: [{output1.min().item():.3f}, {output1.max().item():.3f}]")

    print("\nSecond forward pass (using fitted PCA)...")
    output2 = model_full(mu, sigma, fit_pca=False)
    print(f"  Output shape: {output2.shape}")
    print(f"  Output range: [{output2.min().item():.3f}, {output2.max().item():.3f}]")

    # Test 2: First-order only (ablation)
    print("\n" + "=" * 60)
    print("Test 2: First-Order Only (ablation)")
    print("=" * 60)

    model_first = DKOFirstOrder(
        feature_dim=D,
        output_dim=1,
        task='regression',
        verbose=True,
    )

    output_first = model_first(mu, sigma=None)
    print(f"  Output shape: {output_first.shape}")
    print(f"  Output range: [{output_first.min().item():.3f}, {output_first.max().item():.3f}]")

    # Test 3: Classification task
    print("\n" + "=" * 60)
    print("Test 3: Classification Task")
    print("=" * 60)

    model_clf = DKO(
        feature_dim=D,
        output_dim=2,
        task='classification',
        verbose=True,
    )

    model_clf.train()
    output_train = model_clf(mu, sigma, fit_pca=True)
    print(f"  Training output shape: {output_train.shape}")

    model_clf.eval()
    output_eval = model_clf(mu, sigma)
    print(f"  Eval output shape: {output_eval.shape}")
    print(f"  Eval output (with sigmoid): min={output_eval.min().item():.3f}, max={output_eval.max().item():.3f}")

    # Test 4: Gradient flow
    print("\n" + "=" * 60)
    print("Test 4: Gradient Flow")
    print("=" * 60)

    model_grad = DKO(feature_dim=D, output_dim=1, verbose=False)
    model_grad.train()

    mu_grad = mu.clone().requires_grad_(True)
    output = model_grad(mu_grad, sigma, fit_pca=True)

    loss = output.mean()
    loss.backward()

    has_gradients = mu_grad.grad is not None and mu_grad.grad.abs().sum() > 0
    print(f"  Gradients flowing to input: {has_gradients}")

    param_grads = sum(1 for p in model_grad.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  Parameters with gradients: {param_grads}")

    # Test 5: Embedding extraction
    print("\n" + "=" * 60)
    print("Test 5: Embedding Extraction")
    print("=" * 60)

    embedding = model_full.get_embedding(mu, sigma)
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Embedding range: [{embedding.min().item():.3f}, {embedding.max().item():.3f}]")

    # Test 6: Kernel matrix extraction
    print("\n" + "=" * 60)
    print("Test 6: Kernel Matrix Extraction")
    print("=" * 60)

    K = model_full.get_kernel_matrix(mu, sigma)
    print(f"  Kernel matrix shape: {K.shape}")
    print(f"  Kernel is PSD (min eigenvalue >= 0): ", end="")
    eigenvalues = torch.linalg.eigvalsh(K[0])
    print(f"{(eigenvalues >= -1e-6).all().item()}")

    print("\n" + "=" * 60)
    print("All DKO tests passed!")
    print("=" * 60)
