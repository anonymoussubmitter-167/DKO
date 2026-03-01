"""
DKO Variant Models — Eigendecomposition-based second-order representations.

All 7 variants are standalone nn.Module classes with signature:
    forward(mu, sigma=None, fit_pca=False)

They compress sigma via spectral features, avoiding the PCA fitting step
entirely. For large feature dimensions (D > 256), the diagonal of sigma
(per-feature variances) is used as a fast proxy for eigenvalues, making
these variants O(D) instead of O(D^3).

Variants:
    A — DKOEigenspectrum: Top-k eigenvalues appended to mu
    B — DKOScalarInvariants: 5 scalar matrix invariants appended to mu
    C — DKOLowRank: Top-k eigenvalues + eigenvector projections
    D — DKOGatedFusion: Gated combination of mu and sigma encodings
    E — DKOResidual: Base prediction from mu + learned correction from sigma
    F — DKOCrossAttention: Cross-attention between mu and sigma representations
    S — DKOSCCRouter: Mixture-of-experts routing by conformational complexity
"""

import torch
import torch.nn as nn
from typing import Optional


def _make_mlp(in_dim: int, hidden_dims: list, out_dim: int, dropout: float = 0.1) -> nn.Sequential:
    """Build an MLP with BatchNorm, ReLU, and Dropout."""
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers.extend([
            nn.Linear(prev, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Dropout(dropout),
        ])
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


# Threshold above which full eigendecomposition is too slow (O(D^3)).
# For D > _DIAG_THRESHOLD, use the diagonal of sigma as a fast O(D) proxy.
_DIAG_THRESHOLD = 256


def _get_spectral_values(sigma: torch.Tensor) -> torch.Tensor:
    """
    Get spectral values from sigma: eigenvalues for small D, diagonal for large D.

    For D <= 256: full eigvalsh (accurate, O(D^3) is manageable)
    For D > 256: sorted diagonal (fast O(D) proxy for eigenvalues)

    Returns: (batch, D) sorted ascending
    """
    sigma = torch.nan_to_num(sigma, nan=0.0, posinf=1.0, neginf=-1.0)
    D = sigma.shape[-1]

    if D <= _DIAG_THRESHOLD:
        try:
            return torch.nan_to_num(
                torch.linalg.eigvalsh(sigma), nan=0.0, posinf=1.0, neginf=-1.0
            )
        except RuntimeError:
            pass  # fall through to diagonal

    # Fast path: use diagonal (per-feature variances), sorted ascending
    diag = torch.diagonal(sigma, dim1=-2, dim2=-1)  # (batch, D)
    diag = torch.nan_to_num(diag, nan=0.0, posinf=1.0, neginf=-1.0)
    return diag.sort(dim=-1).values


def _get_spectral_decomp(sigma: torch.Tensor, k: int):
    """
    Get top-k spectral components from sigma.

    For D <= 256: full eigh, take top-k
    For D > 256: use random projection for approximate top-k directions

    Returns: (eigenvalues (batch, k), eigenvectors (batch, D, k))
    """
    sigma = torch.nan_to_num(sigma, nan=0.0, posinf=1.0, neginf=-1.0)
    D = sigma.shape[-1]
    batch = sigma.shape[0]

    if D <= _DIAG_THRESHOLD:
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(sigma)
            eigenvalues = torch.nan_to_num(eigenvalues, nan=0.0, posinf=1.0, neginf=-1.0)
            eigenvectors = torch.nan_to_num(eigenvectors, nan=0.0, posinf=1.0, neginf=-1.0)
            return eigenvalues[:, -k:], eigenvectors[:, :, -k:]
        except RuntimeError:
            pass  # fall through to approximate

    # Fast approximate: project sigma to k dimensions via random projection
    # sigma_small = P^T @ sigma @ P, then eigh(sigma_small) for top-k
    # Use the top-k diagonal indices as projection directions (data-adaptive)
    diag = torch.diagonal(sigma, dim1=-2, dim2=-1)  # (batch, D)
    diag = torch.nan_to_num(diag, nan=0.0, posinf=1.0, neginf=-1.0)

    # Get indices of top-k diagonal elements (per sample)
    _, top_indices = diag.topk(k, dim=-1)  # (batch, k)

    # Extract top-k eigenvalues (diagonal elements)
    top_vals = diag.gather(1, top_indices)  # (batch, k)

    # Approximate eigenvectors as one-hot in the top-k feature directions
    top_vecs = torch.zeros(batch, D, k, device=sigma.device, dtype=sigma.dtype)
    for i in range(k):
        idx = top_indices[:, i]  # (batch,)
        top_vecs[torch.arange(batch, device=sigma.device), idx, i] = 1.0

    return top_vals, top_vecs


class DKOEigenspectrum(nn.Module):
    """
    Variant A: Top-k eigenvalues of sigma appended to mu.

    Captures the "shape" of the conformer distribution via its eigenvalue
    spectrum without the full D^2 covariance matrix.

    forward(mu, sigma=None, fit_pca=False) -> (batch, output_dim)
    """

    def __init__(
        self,
        feature_dim: int,
        output_dim: int = 1,
        k: int = 10,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.k = k

        self.prediction_mlp = _make_mlp(
            in_dim=feature_dim + k,
            hidden_dims=[256, 128],
            out_dim=output_dim,
            dropout=dropout,
        )

    def forward(
        self,
        mu: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
        fit_pca: bool = False,
    ) -> torch.Tensor:
        mu = torch.nan_to_num(mu, nan=0.0, posinf=1.0, neginf=-1.0)

        if sigma is not None:
            eigenvalues = _get_spectral_values(sigma)
            top_k = eigenvalues[:, -self.k:]  # (batch, k) — largest eigenvalues
            combined = torch.cat([mu, top_k], dim=1)
        else:
            # Fallback: zeros for eigenvalues
            zeros = torch.zeros(mu.shape[0], self.k, device=mu.device, dtype=mu.dtype)
            combined = torch.cat([mu, zeros], dim=1)

        return self.prediction_mlp(combined)


class DKOScalarInvariants(nn.Module):
    """
    Variant B: 5 scalar matrix invariants of sigma appended to mu.

    Invariants: trace(sigma), log_det(sigma), frobenius_norm(sigma),
    lambda_1/sum(lambda), lambda_1/lambda_2.

    Minimal overhead — only 5 extra features.

    forward(mu, sigma=None, fit_pca=False) -> (batch, output_dim)
    """

    def __init__(
        self,
        feature_dim: int,
        output_dim: int = 1,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim

        self.prediction_mlp = _make_mlp(
            in_dim=feature_dim + 5,
            hidden_dims=[256, 128],
            out_dim=output_dim,
            dropout=dropout,
        )

    def forward(
        self,
        mu: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
        fit_pca: bool = False,
    ) -> torch.Tensor:
        mu = torch.nan_to_num(mu, nan=0.0, posinf=1.0, neginf=-1.0)

        if sigma is not None:
            eigenvalues = _get_spectral_values(sigma)  # (batch, D) sorted ascending

            trace_val = eigenvalues.sum(dim=1, keepdim=True)  # (batch, 1)

            # log_det = sum of log(eigenvalues), with clamping for stability
            log_det = torch.log(eigenvalues.clamp(min=1e-8)).sum(dim=1, keepdim=True)

            # Frobenius norm = sqrt(sum of eigenvalues^2) for symmetric matrices
            frob = eigenvalues.pow(2).sum(dim=1, keepdim=True).sqrt()

            # Spectral ratio: lambda_max / sum(lambda)
            lambda_max = eigenvalues[:, -1:]  # (batch, 1)
            spectral_ratio = lambda_max / trace_val.clamp(min=1e-8)

            # Condition ratio: lambda_max / lambda_2nd
            lambda_2nd = eigenvalues[:, -2:-1]  # (batch, 1)
            condition_ratio = lambda_max / lambda_2nd.clamp(min=1e-8)

            invariants = torch.cat([
                trace_val, log_det, frob, spectral_ratio, condition_ratio
            ], dim=1)  # (batch, 5)
            invariants = torch.nan_to_num(invariants, nan=0.0, posinf=10.0, neginf=-10.0)
            invariants = torch.clamp(invariants, min=-100.0, max=100.0)

            combined = torch.cat([mu, invariants], dim=1)
        else:
            zeros = torch.zeros(mu.shape[0], 5, device=mu.device, dtype=mu.dtype)
            combined = torch.cat([mu, zeros], dim=1)

        return self.prediction_mlp(combined)


class DKOLowRank(nn.Module):
    """
    Variant C: Top-k eigenvalues + eigenvector projection.

    Captures both eigenvalue magnitudes and eigenvector directions via
    a learned projection of the flattened top-k eigenvectors.

    forward(mu, sigma=None, fit_pca=False) -> (batch, output_dim)
    """

    def __init__(
        self,
        feature_dim: int,
        output_dim: int = 1,
        k: int = 5,
        vec_proj_dim: int = 64,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.k = k
        self.vec_proj_dim = vec_proj_dim

        # Project flattened eigenvectors: (batch, D*k) -> (batch, vec_proj_dim)
        self.u_projection = nn.Sequential(
            nn.Linear(feature_dim * k, vec_proj_dim),
            nn.BatchNorm1d(vec_proj_dim),
            nn.ReLU(),
        )

        self.prediction_mlp = _make_mlp(
            in_dim=feature_dim + k + vec_proj_dim,
            hidden_dims=[256, 128],
            out_dim=output_dim,
            dropout=dropout,
        )

    def forward(
        self,
        mu: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
        fit_pca: bool = False,
    ) -> torch.Tensor:
        mu = torch.nan_to_num(mu, nan=0.0, posinf=1.0, neginf=-1.0)

        if sigma is not None:
            top_vals, top_vecs = _get_spectral_decomp(sigma, self.k)
            # top_vals: (batch, k), top_vecs: (batch, D, k)

            # Flatten eigenvectors and project
            batch_size = mu.shape[0]
            vecs_flat = top_vecs.reshape(batch_size, -1)     # (batch, D*k)
            u_proj = self.u_projection(vecs_flat)            # (batch, vec_proj_dim)

            combined = torch.cat([mu, top_vals, u_proj], dim=1)
        else:
            zeros_vals = torch.zeros(mu.shape[0], self.k, device=mu.device, dtype=mu.dtype)
            zeros_proj = torch.zeros(mu.shape[0], self.vec_proj_dim, device=mu.device, dtype=mu.dtype)
            combined = torch.cat([mu, zeros_vals, zeros_proj], dim=1)

        return self.prediction_mlp(combined)


class DKOGatedFusion(nn.Module):
    """
    Variant D: Gated fusion of mu and sigma encodings.

    A per-hidden-unit gate learns whether to use mu or sigma information.
    Should suppress sigma for rigid molecules automatically.

    forward(mu, sigma=None, fit_pca=False) -> (batch, output_dim)
    """

    def __init__(
        self,
        feature_dim: int,
        output_dim: int = 1,
        hidden_dim: int = 128,
        k: int = 10,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.k = k
        self.hidden_dim = hidden_dim

        self.mu_encoder = _make_mlp(feature_dim, [256], hidden_dim, dropout=dropout)
        self.sigma_encoder = _make_mlp(k, [256], hidden_dim, dropout=dropout)
        self.gate_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.prediction_head = _make_mlp(hidden_dim, [64], output_dim, dropout=dropout)

    def forward(
        self,
        mu: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
        fit_pca: bool = False,
    ) -> torch.Tensor:
        mu = torch.nan_to_num(mu, nan=0.0, posinf=1.0, neginf=-1.0)

        h_mu = self.mu_encoder(mu)  # (batch, hidden)

        if sigma is not None:
            eigenvalues = _get_spectral_values(sigma)
            top_k = eigenvalues[:, -self.k:]
            h_sigma = self.sigma_encoder(top_k)  # (batch, hidden)
        else:
            h_sigma = torch.zeros_like(h_mu)

        gate = self.gate_net(mu)  # (batch, hidden)
        h = gate * h_mu + (1 - gate) * h_sigma

        return self.prediction_head(h)


class DKOResidual(nn.Module):
    """
    Variant E: Base prediction from mu + learned correction from sigma.

    Starts near first-order (scale_factor init=0.1, learned). Only adds
    sigma if it helps. Should perform >= first-order by construction.

    forward(mu, sigma=None, fit_pca=False) -> (batch, output_dim)
    """

    def __init__(
        self,
        feature_dim: int,
        output_dim: int = 1,
        k: int = 10,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.k = k

        self.base_net = _make_mlp(feature_dim, [256, 128], output_dim, dropout=dropout)
        self.correction_net = _make_mlp(k, [128, 64], output_dim, dropout=dropout)
        # Learned scale factor, initialized small so model starts near first-order
        self.scale_factor = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        mu: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
        fit_pca: bool = False,
    ) -> torch.Tensor:
        mu = torch.nan_to_num(mu, nan=0.0, posinf=1.0, neginf=-1.0)

        h_base = self.base_net(mu)  # (batch, output_dim)

        if sigma is not None:
            eigenvalues = _get_spectral_values(sigma)
            top_k = eigenvalues[:, -self.k:]
            h_correction = self.correction_net(top_k)  # (batch, output_dim)
            return h_base + self.scale_factor * h_correction
        else:
            return h_base


class DKOCrossAttention(nn.Module):
    """
    Variant F: Cross-attention between mu and sigma representations.

    mu queries sigma via cross-attention with a residual connection.

    forward(mu, sigma=None, fit_pca=False) -> (batch, output_dim)
    """

    def __init__(
        self,
        feature_dim: int,
        output_dim: int = 1,
        hidden_dim: int = 128,
        k: int = 10,
        num_heads: int = 4,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.k = k

        self.mu_encoder = _make_mlp(feature_dim, [256], hidden_dim, dropout=dropout)
        self.sigma_encoder = _make_mlp(k, [256], hidden_dim, dropout=dropout)

        # Cross-attention: mu queries sigma
        # MultiheadAttention expects (seq_len, batch, embed_dim), we use seq_len=1
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.prediction_head = _make_mlp(hidden_dim, [64], output_dim, dropout=dropout)

    def forward(
        self,
        mu: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
        fit_pca: bool = False,
    ) -> torch.Tensor:
        mu = torch.nan_to_num(mu, nan=0.0, posinf=1.0, neginf=-1.0)

        mu_encoded = self.mu_encoder(mu)  # (batch, hidden)

        if sigma is not None:
            eigenvalues = _get_spectral_values(sigma)
            top_k = eigenvalues[:, -self.k:]
            sigma_encoded = self.sigma_encoder(top_k)  # (batch, hidden)

            # Reshape for attention: (batch, 1, hidden)
            q = mu_encoded.unsqueeze(1)
            kv = sigma_encoded.unsqueeze(1)

            attn_out, _ = self.cross_attn(q, kv, kv)  # (batch, 1, hidden)
            attn_out = attn_out.squeeze(1)  # (batch, hidden)

            # Residual connection + layer norm
            h = self.layer_norm(attn_out + mu_encoded)
        else:
            h = self.layer_norm(mu_encoded)

        return self.prediction_head(h)


class DKOSCCRouter(nn.Module):
    """
    Variant S: Mixture-of-experts routing by conformational complexity.

    Routes rigid molecules (low SCC) to first-order expert and flexible
    molecules (high SCC) to second-order expert via a learned router.

    forward(mu, sigma=None, fit_pca=False) -> (batch, output_dim)
    """

    def __init__(
        self,
        feature_dim: int,
        output_dim: int = 1,
        k: int = 10,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.k = k

        # Router based on SCC (trace of sigma, a scalar)
        self.router_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # First-order expert (mu only)
        self.first_order_net = _make_mlp(feature_dim, [256, 128], output_dim, dropout=dropout)

        # Second-order expert (mu + eigenvalues)
        self.second_order_net = _make_mlp(
            feature_dim + k, [256, 128], output_dim, dropout=dropout
        )

    def forward(
        self,
        mu: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
        fit_pca: bool = False,
    ) -> torch.Tensor:
        mu = torch.nan_to_num(mu, nan=0.0, posinf=1.0, neginf=-1.0)

        h_first = self.first_order_net(mu)  # (batch, output_dim)

        if sigma is not None:
            eigenvalues = _get_spectral_values(sigma)
            top_k = eigenvalues[:, -self.k:]

            # SCC proxy: trace of sigma
            scc = eigenvalues.sum(dim=1, keepdim=True)  # (batch, 1)
            routing_prob = self.router_net(scc)  # (batch, 1)

            h_second = self.second_order_net(
                torch.cat([mu, top_k], dim=1)
            )  # (batch, output_dim)

            return (1 - routing_prob) * h_first + routing_prob * h_second
        else:
            return h_first
