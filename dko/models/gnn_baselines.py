"""
Graph Neural Network baselines for molecular property prediction.

This module provides wrappers around popular 3D GNN architectures
(SchNet, DimeNet++, SphereNet) with conformer ensemble support.

Implementations use PyTorch Geometric when available, with fallback
to simplified versions for environments without PyG.

Usage:
    # With PyTorch Geometric installed:
    from dko.models.gnn_baselines import SchNetPyG, DimeNetPPPyG

    # Without PyTorch Geometric (simplified versions):
    from dko.models.gnn_baselines import SchNet, DimeNetPP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math
import warnings

# Check for PyTorch Geometric availability
try:
    import torch_geometric
    from torch_geometric.nn import (
        SchNet as PyGSchNet,
        DimeNetPlusPlus as PyGDimeNet,
        radius_graph,
        global_mean_pool,
        global_add_pool,
        global_max_pool,
    )
    from torch_geometric.data import Data, Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    warnings.warn(
        "PyTorch Geometric not installed. Using simplified GNN implementations. "
        "For production use, install with: pip install torch-geometric"
    )


# =============================================================================
# Helper Modules
# =============================================================================

class RadialBasisFunctions(nn.Module):
    """Radial basis functions for distance embedding."""

    def __init__(
        self,
        num_rbf: int = 50,
        cutoff: float = 10.0,
        rbf_type: str = "gaussian",
    ):
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.rbf_type = rbf_type

        # Centers and widths for Gaussian RBF
        self.register_buffer(
            "centers",
            torch.linspace(0, cutoff, num_rbf)
        )
        self.register_buffer(
            "widths",
            torch.FloatTensor([cutoff / num_rbf] * num_rbf)
        )

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Embed distances using radial basis functions."""
        if self.rbf_type == "gaussian":
            return torch.exp(
                -((distances.unsqueeze(-1) - self.centers) ** 2)
                / (2 * self.widths ** 2)
            )
        elif self.rbf_type == "bessel":
            # Bessel basis (used in DimeNet)
            d_scaled = distances.unsqueeze(-1) / self.cutoff
            n = torch.arange(1, self.num_rbf + 1, device=distances.device).float()
            return torch.sqrt(2.0 / self.cutoff) * torch.sin(n * math.pi * d_scaled) / distances.unsqueeze(-1)
        else:
            raise ValueError(f"Unknown RBF type: {self.rbf_type}")


class CutoffFunction(nn.Module):
    """Smooth cutoff function for distance-based interactions."""

    def __init__(self, cutoff: float = 10.0, envelope_type: str = "cosine"):
        super().__init__()
        self.cutoff = cutoff
        self.envelope_type = envelope_type

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Apply smooth cutoff."""
        if self.envelope_type == "cosine":
            # Cosine cutoff
            cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
            cutoffs = cutoffs * (distances < self.cutoff).float()
        elif self.envelope_type == "polynomial":
            # Polynomial envelope (DimeNet style)
            p = 6
            x = distances / self.cutoff
            cutoffs = 1.0 - (p + 1) * (p + 2) / 2 * x.pow(p) + p * (p + 2) * x.pow(p + 1) - p * (p + 1) / 2 * x.pow(p + 2)
            cutoffs = cutoffs * (distances < self.cutoff).float()
        else:
            cutoffs = (distances < self.cutoff).float()
        return cutoffs


class ConformerAggregation(nn.Module):
    """
    Module for aggregating conformer-level predictions/embeddings.

    Supports multiple aggregation strategies:
    - mean: Simple averaging
    - boltzmann: Energy-weighted averaging
    - attention: Learned attention weights
    - max: Max pooling
    """

    def __init__(
        self,
        hidden_dim: int,
        aggregation: str = "mean",
        temperature: float = 300.0,
        attention_heads: int = 4,
    ):
        super().__init__()
        self.aggregation = aggregation
        self.temperature = temperature
        self.kB = 0.001987204  # kcal/(mol*K)

        if aggregation == "attention":
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=attention_heads,
                batch_first=True,
            )
            self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(
        self,
        x: torch.Tensor,
        energies: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Aggregate conformer representations.

        Args:
            x: Conformer embeddings (batch, n_conformers, hidden_dim)
            energies: Conformer energies (batch, n_conformers)
            mask: Valid conformer mask

        Returns:
            Aggregated representation (batch, hidden_dim)
        """
        batch_size = x.shape[0]

        if self.aggregation == "mean":
            if mask is not None:
                x = x * mask.unsqueeze(-1).float()
                return x.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
            return x.mean(dim=1)

        elif self.aggregation == "max":
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            return x.max(dim=1)[0]

        elif self.aggregation == "boltzmann":
            if energies is not None:
                kT = self.kB * self.temperature
                if mask is not None:
                    energies = energies.masked_fill(~mask, float("inf"))
                shifted = energies - energies.min(dim=1, keepdim=True)[0]
                weights = F.softmax(-shifted / kT, dim=1)
                return (x * weights.unsqueeze(-1)).sum(dim=1)
            return x.mean(dim=1)

        elif self.aggregation == "attention":
            query = self.query.expand(batch_size, -1, -1)
            key_padding_mask = ~mask if mask is not None else None
            out, _ = self.attention(query, x, x, key_padding_mask=key_padding_mask)
            return out.squeeze(1)

        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")


# =============================================================================
# PyTorch Geometric Implementations (Full)
# =============================================================================

if HAS_PYG:

    class SchNetPyG(nn.Module):
        """
        Full SchNet implementation using PyTorch Geometric.

        SchNet uses continuous-filter convolutional layers to model
        quantum interactions based on interatomic distances.

        Reference:
            Schütt et al. "SchNet: A continuous-filter convolutional neural
            network for modeling quantum interactions" (NeurIPS 2017)
        """

        def __init__(
            self,
            hidden_channels: int = 128,
            num_filters: int = 128,
            num_interactions: int = 6,
            num_gaussians: int = 50,
            cutoff: float = 10.0,
            max_num_neighbors: int = 32,
            readout: str = "add",
            num_outputs: int = 1,
            output_dim: int = None,
            task: str = "regression",
            conformer_aggregation: str = "mean",
            atomref: Optional[torch.Tensor] = None,
        ):
            """
            Initialize SchNet.

            Args:
                hidden_channels: Hidden embedding size
                num_filters: Number of filters in CFConv
                num_interactions: Number of interaction blocks
                num_gaussians: Number of Gaussian RBFs
                cutoff: Distance cutoff in Angstroms
                max_num_neighbors: Max neighbors for radius graph
                readout: Readout method ('add', 'mean')
                num_outputs: Number of output properties
                output_dim: Alias for num_outputs
                task: 'regression' or 'classification'
                conformer_aggregation: How to aggregate conformers
                atomref: Atomic reference energies
            """
            super().__init__()

            if output_dim is not None:
                num_outputs = output_dim

            self.task = task
            self.cutoff = cutoff
            self.max_num_neighbors = max_num_neighbors

            # PyG SchNet backbone
            self.schnet = PyGSchNet(
                hidden_channels=hidden_channels,
                num_filters=num_filters,
                num_interactions=num_interactions,
                num_gaussians=num_gaussians,
                cutoff=cutoff,
                max_num_neighbors=max_num_neighbors,
                readout=readout,
                atomref=atomref,
            )

            # Override output layer for custom output dimension
            self.output_head = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.SiLU(),
                nn.Linear(hidden_channels // 2, num_outputs),
            )

            # Conformer aggregation
            self.conformer_agg = ConformerAggregation(
                hidden_dim=num_outputs,
                aggregation=conformer_aggregation,
            )

        def forward(
            self,
            z: torch.Tensor,
            pos: torch.Tensor,
            batch: Optional[torch.Tensor] = None,
            conformer_idx: Optional[torch.Tensor] = None,
            energies: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Forward pass.

            Args:
                z: Atomic numbers (num_atoms,)
                pos: Atomic positions (num_atoms, 3)
                batch: Batch indices (num_atoms,)
                conformer_idx: Conformer indices for ensemble
                energies: Conformer energies for Boltzmann weighting

            Returns:
                Property predictions
            """
            if batch is None:
                batch = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)

            # Build radius graph
            edge_index = radius_graph(
                pos,
                r=self.cutoff,
                batch=batch,
                max_num_neighbors=self.max_num_neighbors,
            )

            # SchNet forward
            # Get node embeddings before final output
            h = self.schnet.embedding(z)

            edge_weight = (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)
            edge_attr = self.schnet.distance_expansion(edge_weight)

            for interaction in self.schnet.interactions:
                h = h + interaction(h, edge_index, edge_weight, edge_attr)

            # Readout
            h = self.schnet.lin1(h)
            h = self.schnet.act(h)
            h = self.schnet.lin2(h)

            # Pool to molecule level
            out = global_add_pool(h, batch)

            # Custom output head
            out = self.output_head(out)

            # Conformer aggregation if needed
            if conformer_idx is not None:
                # Reshape for conformer aggregation
                # Assumes conformer_idx indicates which molecule each conformer belongs to
                n_molecules = conformer_idx.max().item() + 1
                n_conformers = (conformer_idx == 0).sum().item()
                out = out.view(n_molecules, n_conformers, -1)
                out = self.conformer_agg(out, energies)

            return out


    class DimeNetPPPyG(nn.Module):
        """
        Full DimeNet++ implementation using PyTorch Geometric.

        DimeNet++ uses directional message passing with spherical harmonics
        to capture angular information in molecular structures.

        Reference:
            Klicpera et al. "Fast and Uncertainty-Aware Directional Message
            Passing for Non-Equilibrium Molecules" (2020)
        """

        def __init__(
            self,
            hidden_channels: int = 128,
            out_channels: int = 1,
            num_blocks: int = 4,
            int_emb_size: int = 64,
            basis_emb_size: int = 8,
            out_emb_channels: int = 256,
            num_spherical: int = 7,
            num_radial: int = 6,
            cutoff: float = 5.0,
            max_num_neighbors: int = 32,
            envelope_exponent: int = 5,
            num_before_skip: int = 1,
            num_after_skip: int = 2,
            num_output_layers: int = 3,
            output_dim: int = None,
            task: str = "regression",
            conformer_aggregation: str = "mean",
        ):
            """
            Initialize DimeNet++.

            Args:
                hidden_channels: Hidden embedding size
                out_channels: Output channels
                num_blocks: Number of building blocks
                int_emb_size: Interaction embedding size
                basis_emb_size: Basis embedding size
                out_emb_channels: Output embedding channels
                num_spherical: Number of spherical harmonics
                num_radial: Number of radial basis functions
                cutoff: Distance cutoff
                max_num_neighbors: Max neighbors
                envelope_exponent: Exponent for envelope function
                num_before_skip: Layers before skip connection
                num_after_skip: Layers after skip connection
                num_output_layers: Number of output layers
                output_dim: Alias for out_channels
                task: 'regression' or 'classification'
                conformer_aggregation: How to aggregate conformers
            """
            super().__init__()

            if output_dim is not None:
                out_channels = output_dim

            self.task = task
            self.cutoff = cutoff
            self.max_num_neighbors = max_num_neighbors

            # PyG DimeNet++ backbone
            self.dimenet = PyGDimeNet(
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,  # Get embeddings
                num_blocks=num_blocks,
                int_emb_size=int_emb_size,
                basis_emb_size=basis_emb_size,
                out_emb_channels=out_emb_channels,
                num_spherical=num_spherical,
                num_radial=num_radial,
                cutoff=cutoff,
                max_num_neighbors=max_num_neighbors,
                envelope_exponent=envelope_exponent,
                num_before_skip=num_before_skip,
                num_after_skip=num_after_skip,
                num_output_layers=num_output_layers,
            )

            # Custom output head
            self.output_head = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.SiLU(),
                nn.Linear(hidden_channels // 2, out_channels),
            )

            # Conformer aggregation
            self.conformer_agg = ConformerAggregation(
                hidden_dim=out_channels,
                aggregation=conformer_aggregation,
            )

        def forward(
            self,
            z: torch.Tensor,
            pos: torch.Tensor,
            batch: Optional[torch.Tensor] = None,
            conformer_idx: Optional[torch.Tensor] = None,
            energies: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """Forward pass."""
            if batch is None:
                batch = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)

            # DimeNet forward
            out = self.dimenet(z, pos, batch)

            # Custom output
            out = self.output_head(out)

            # Conformer aggregation if needed
            if conformer_idx is not None:
                n_molecules = conformer_idx.max().item() + 1
                n_conformers = (conformer_idx == 0).sum().item()
                out = out.view(n_molecules, n_conformers, -1)
                out = self.conformer_agg(out, energies)

            return out


    class GNNEnsembleWrapper(nn.Module):
        """
        Wrapper for processing conformer ensembles with any PyG GNN.

        Processes each conformer through the GNN and aggregates results.
        """

        def __init__(
            self,
            gnn: nn.Module,
            aggregation: str = "mean",
            hidden_dim: int = 128,
        ):
            """
            Initialize wrapper.

            Args:
                gnn: Base GNN model
                aggregation: Aggregation method
                hidden_dim: Hidden dimension for attention
            """
            super().__init__()
            self.gnn = gnn
            self.conformer_agg = ConformerAggregation(
                hidden_dim=hidden_dim,
                aggregation=aggregation,
            )

        def forward(
            self,
            data_list: List[Data],
            energies: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Process conformer ensemble.

            Args:
                data_list: List of PyG Data objects (one per conformer)
                energies: Conformer energies

            Returns:
                Aggregated predictions
            """
            # Process each conformer
            outputs = []
            for data in data_list:
                out = self.gnn(data.z, data.pos, data.batch)
                outputs.append(out)

            # Stack and aggregate
            outputs = torch.stack(outputs, dim=1)  # (batch, n_conf, out_dim)
            return self.conformer_agg(outputs, energies)


# =============================================================================
# Simplified Implementations (No PyG Required)
# =============================================================================

class SchNetInteraction(nn.Module):
    """SchNet continuous-filter convolution block."""

    def __init__(
        self,
        hidden_dim: int,
        num_filters: int,
        num_rbf: int,
        cutoff: float,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_filters = num_filters

        # Filter-generating network
        self.filter_network = nn.Sequential(
            nn.Linear(num_rbf, num_filters),
            nn.SiLU(),
            nn.Linear(num_filters, num_filters),
        )

        # Atom-wise layers
        self.atom_dense1 = nn.Linear(hidden_dim, num_filters)
        self.atom_dense2 = nn.Linear(num_filters, hidden_dim)

        self.cutoff_fn = CutoffFunction(cutoff)

    def forward(
        self,
        x: torch.Tensor,
        rbf: torch.Tensor,
        distances: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of SchNet interaction."""
        # Generate continuous filters
        filters = self.filter_network(rbf)
        filters = filters * self.cutoff_fn(distances).unsqueeze(-1)

        # Message passing
        x_j = x[edge_index[0]]
        x_j = self.atom_dense1(x_j)

        # Apply filters
        messages = x_j * filters

        # Aggregate messages
        out = torch.zeros_like(x[:, :self.num_filters])
        out.scatter_add_(0, edge_index[1].unsqueeze(-1).expand_as(messages), messages)

        # Output transformation
        out = self.atom_dense2(out)

        return x + out


class SchNet(nn.Module):
    """
    SchNet: Continuous-filter convolutional neural network.

    Simplified implementation for molecular property prediction.
    For production, use SchNetPyG with PyTorch Geometric.

    Reference:
        Schütt et al. "SchNet: A continuous-filter convolutional neural
        network for modeling quantum interactions" (2017)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_rbf: int = 50,
        cutoff: float = 10.0,
        max_atomic_num: int = 100,
        num_outputs: int = 1,
        output_dim: int = None,
        task: str = "regression",
        conformer_aggregation: str = "mean",
    ):
        """Initialize SchNet."""
        super().__init__()

        if output_dim is not None:
            num_outputs = output_dim

        self.cutoff = cutoff
        self.task = task

        # Atom embedding
        self.atom_embedding = nn.Embedding(max_atomic_num, hidden_dim)

        # Radial basis functions
        self.rbf = RadialBasisFunctions(num_rbf, cutoff)

        # Interaction blocks
        self.interactions = nn.ModuleList([
            SchNetInteraction(hidden_dim, num_filters, num_rbf, cutoff)
            for _ in range(num_interactions)
        ])

        # Output network
        self.output_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, num_outputs),
        )

        # Conformer aggregation
        self.conformer_agg = ConformerAggregation(
            hidden_dim=num_outputs,
            aggregation=conformer_aggregation,
        )

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        conformer_idx: Optional[torch.Tensor] = None,
        energies: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        # Initial embedding
        x = self.atom_embedding(atomic_numbers)

        # Compute edges within cutoff
        num_atoms = positions.shape[0]
        row = torch.arange(num_atoms, device=positions.device).repeat_interleave(num_atoms)
        col = torch.arange(num_atoms, device=positions.device).repeat(num_atoms)

        # Remove self-loops
        mask = row != col
        row, col = row[mask], col[mask]

        # Compute distances
        diff = positions[row] - positions[col]
        distances = torch.norm(diff, dim=-1)

        # Apply cutoff
        cutoff_mask = distances < self.cutoff
        row, col = row[cutoff_mask], col[cutoff_mask]
        distances = distances[cutoff_mask]

        edge_index = torch.stack([row, col], dim=0)

        # RBF embedding
        rbf = self.rbf(distances)

        # Interaction blocks
        for interaction in self.interactions:
            x = interaction(x, rbf, distances, edge_index)

        # Aggregate per molecule
        num_molecules = batch.max().item() + 1
        molecule_features = torch.zeros(
            num_molecules, x.shape[-1],
            device=x.device, dtype=x.dtype
        )
        molecule_features.scatter_add_(
            0, batch.unsqueeze(-1).expand_as(x), x
        )

        # Mean over atoms
        atom_counts = torch.zeros(num_molecules, device=x.device)
        atom_counts.scatter_add_(0, batch, torch.ones_like(batch, dtype=torch.float))
        molecule_features = molecule_features / atom_counts.unsqueeze(-1).clamp(min=1)

        # Output
        predictions = self.output_network(molecule_features)

        return predictions


class DimeNetPP(nn.Module):
    """
    Simplified DimeNet++ implementation.

    For production use, install PyTorch Geometric and use DimeNetPPPyG.

    Reference:
        Klicpera et al. "Fast and Uncertainty-Aware Directional Message
        Passing for Non-Equilibrium Molecules" (2020)
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        out_channels: int = 1,
        num_blocks: int = 4,
        num_bilinear: int = 8,
        num_spherical: int = 7,
        num_radial: int = 6,
        cutoff: float = 5.0,
        envelope_exponent: int = 5,
        output_dim: int = None,
        task: str = "regression",
        conformer_aggregation: str = "mean",
    ):
        """Initialize DimeNet++ (simplified)."""
        super().__init__()

        if output_dim is not None:
            out_channels = output_dim

        self.hidden_channels = hidden_channels
        self.cutoff = cutoff
        self.task = task

        # Atom embedding
        self.embedding = nn.Embedding(100, hidden_channels)

        # RBF for distances
        self.rbf = RadialBasisFunctions(num_radial * 10, cutoff, rbf_type="gaussian")

        # Interaction blocks with distance-dependent weights
        self.interactions = nn.ModuleList()
        for _ in range(num_blocks):
            self.interactions.append(nn.ModuleDict({
                "msg": nn.Sequential(
                    nn.Linear(hidden_channels + num_radial * 10, hidden_channels),
                    nn.SiLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                ),
                "upd": nn.Sequential(
                    nn.Linear(hidden_channels * 2, hidden_channels),
                    nn.SiLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                ),
            }))

        self.output = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, out_channels),
        )

        # Conformer aggregation
        self.conformer_agg = ConformerAggregation(
            hidden_dim=out_channels,
            aggregation=conformer_aggregation,
        )

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        conformer_idx: Optional[torch.Tensor] = None,
        energies: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        # Embedding
        x = self.embedding(atomic_numbers.clamp(0, 99))

        # Build edges
        num_atoms = positions.shape[0]
        row = torch.arange(num_atoms, device=positions.device).repeat_interleave(num_atoms)
        col = torch.arange(num_atoms, device=positions.device).repeat(num_atoms)
        mask = (row != col)
        row, col = row[mask], col[mask]

        distances = (positions[row] - positions[col]).norm(dim=-1)
        cutoff_mask = distances < self.cutoff
        row, col = row[cutoff_mask], col[cutoff_mask]
        distances = distances[cutoff_mask]

        edge_index = torch.stack([row, col], dim=0)
        rbf = self.rbf(distances)

        # Message passing
        for interaction in self.interactions:
            # Message
            x_j = x[edge_index[0]]
            msg_input = torch.cat([x_j, rbf], dim=-1)
            messages = interaction["msg"](msg_input)

            # Aggregate
            aggr = torch.zeros_like(x)
            aggr.scatter_add_(0, edge_index[1].unsqueeze(-1).expand_as(messages), messages)

            # Update
            x = x + interaction["upd"](torch.cat([x, aggr], dim=-1))

        # Pool
        num_molecules = batch.max().item() + 1
        out = torch.zeros(num_molecules, x.shape[-1], device=x.device, dtype=x.dtype)
        out.scatter_add_(0, batch.unsqueeze(-1).expand_as(x), x)
        counts = torch.zeros(num_molecules, device=x.device)
        counts.scatter_add_(0, batch, torch.ones_like(batch, dtype=torch.float))
        out = out / counts.unsqueeze(-1).clamp(min=1)

        return self.output(out)


class SphereNet(nn.Module):
    """
    Simplified SphereNet implementation.

    For production use, install PyTorch Geometric and use the full implementation.

    Reference:
        Liu et al. "Spherical Message Passing for 3D Molecular Graphs" (2022)
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        out_channels: int = 1,
        num_layers: int = 4,
        cutoff: float = 5.0,
        lmax: int = 2,
        output_dim: int = None,
        task: str = "regression",
        conformer_aggregation: str = "mean",
    ):
        """Initialize SphereNet (simplified)."""
        super().__init__()

        if output_dim is not None:
            out_channels = output_dim

        self.hidden_channels = hidden_channels
        self.cutoff = cutoff
        self.task = task

        # Embedding
        self.embedding = nn.Embedding(100, hidden_channels)

        # Distance RBF
        self.rbf = RadialBasisFunctions(50, cutoff)

        # Simplified spherical layers (without full spherical harmonics)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "msg": nn.Sequential(
                    nn.Linear(hidden_channels + 50 + 3, hidden_channels),  # +3 for direction
                    nn.SiLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                ),
                "upd": nn.Sequential(
                    nn.Linear(hidden_channels * 2, hidden_channels),
                    nn.SiLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                ),
            }))

        self.output = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, out_channels),
        )

        # Conformer aggregation
        self.conformer_agg = ConformerAggregation(
            hidden_dim=out_channels,
            aggregation=conformer_aggregation,
        )

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        conformer_idx: Optional[torch.Tensor] = None,
        energies: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with directional information."""
        x = self.embedding(atomic_numbers.clamp(0, 99))

        # Build edges
        num_atoms = positions.shape[0]
        row = torch.arange(num_atoms, device=positions.device).repeat_interleave(num_atoms)
        col = torch.arange(num_atoms, device=positions.device).repeat(num_atoms)
        mask = row != col
        row, col = row[mask], col[mask]

        diff = positions[row] - positions[col]
        distances = diff.norm(dim=-1)
        cutoff_mask = distances < self.cutoff
        row, col = row[cutoff_mask], col[cutoff_mask]
        distances = distances[cutoff_mask]
        diff = diff[cutoff_mask]

        # Normalized direction vectors (simplified spherical info)
        directions = diff / distances.unsqueeze(-1).clamp(min=1e-8)

        edge_index = torch.stack([row, col], dim=0)
        rbf = self.rbf(distances)

        # Message passing with directional info
        for layer in self.layers:
            x_j = x[edge_index[0]]
            msg_input = torch.cat([x_j, rbf, directions], dim=-1)
            messages = layer["msg"](msg_input)

            aggr = torch.zeros_like(x)
            aggr.scatter_add_(0, edge_index[1].unsqueeze(-1).expand_as(messages), messages)

            x = x + layer["upd"](torch.cat([x, aggr], dim=-1))

        # Pool
        num_molecules = batch.max().item() + 1
        out = torch.zeros(num_molecules, x.shape[-1], device=x.device, dtype=x.dtype)
        out.scatter_add_(0, batch.unsqueeze(-1).expand_as(x), x)
        counts = torch.zeros(num_molecules, device=x.device)
        counts.scatter_add_(0, batch, torch.ones_like(batch, dtype=torch.float))
        out = out / counts.unsqueeze(-1).clamp(min=1)

        return self.output(out)


# =============================================================================
# 3D-Infomax Implementation
# =============================================================================

class InfomaxEncoder(nn.Module):
    """
    Encoder network for 3D-Infomax.

    Uses message passing with RBF-expanded distances to learn
    node representations.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_rbf: int = 50,
        cutoff: float = 5.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cutoff = cutoff

        self.embedding = nn.Embedding(100, hidden_dim)
        self.rbf = RadialBasisFunctions(num_rbf, cutoff)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "msg": nn.Sequential(
                    nn.Linear(hidden_dim + num_rbf, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ),
                "upd": nn.GRU(hidden_dim, hidden_dim, batch_first=True),
            }))

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both node and graph representations.

        Returns:
            node_rep: Node representations (num_atoms, hidden_dim)
            graph_rep: Graph representations (num_molecules, hidden_dim)
        """
        x = self.embedding(atomic_numbers.clamp(0, 99))

        # Build edges
        num_atoms = positions.shape[0]
        row = torch.arange(num_atoms, device=positions.device).repeat_interleave(num_atoms)
        col = torch.arange(num_atoms, device=positions.device).repeat(num_atoms)
        mask = row != col
        row, col = row[mask], col[mask]

        distances = (positions[row] - positions[col]).norm(dim=-1)
        cutoff_mask = distances < self.cutoff
        row, col = row[cutoff_mask], col[cutoff_mask]
        distances = distances[cutoff_mask]

        edge_index = torch.stack([row, col], dim=0)
        rbf = self.rbf(distances)

        # Message passing with GRU updates
        for layer in self.layers:
            x_j = x[edge_index[0]]
            msg_input = torch.cat([x_j, rbf], dim=-1)
            messages = layer["msg"](msg_input)

            aggr = torch.zeros_like(x)
            aggr.scatter_add_(0, edge_index[1].unsqueeze(-1).expand_as(messages), messages)

            # GRU update
            x_gru, _ = layer["upd"](aggr.unsqueeze(1))
            x = x + x_gru.squeeze(1)

        x = self.layer_norm(x)

        # Global pooling for graph representation
        num_molecules = batch.max().item() + 1
        graph_rep = torch.zeros(num_molecules, x.shape[-1], device=x.device, dtype=x.dtype)
        graph_rep.scatter_add_(0, batch.unsqueeze(-1).expand_as(x), x)
        counts = torch.zeros(num_molecules, device=x.device)
        counts.scatter_add_(0, batch, torch.ones_like(batch, dtype=torch.float))
        graph_rep = graph_rep / counts.unsqueeze(-1).clamp(min=1)

        return x, graph_rep


class ThreeDInfomax(nn.Module):
    """
    3D-Infomax: Contrastive learning for 3D molecular representations.

    Learns molecular representations by maximizing mutual information
    between local (node) and global (graph) representations.

    Architecture:
        1. GNN encoder producing node and graph representations
        2. Local-global contrastive objective during pre-training
        3. Fine-tuning head for property prediction

    Reference:
        Stärk et al. "3D Infomax improves GNNs for Molecular Property Prediction" (2022)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_rbf: int = 50,
        cutoff: float = 5.0,
        output_dim: int = 1,
        task: str = "regression",
        conformer_aggregation: str = "mean",
        use_contrastive_head: bool = False,
    ):
        """
        Initialize 3D-Infomax.

        Args:
            hidden_dim: Hidden dimension
            num_layers: Number of message passing layers
            num_rbf: Number of radial basis functions
            cutoff: Distance cutoff
            output_dim: Output dimension for property prediction
            task: 'regression' or 'classification'
            conformer_aggregation: How to aggregate conformers
            use_contrastive_head: Whether to include contrastive learning head
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.task = task
        self.use_contrastive_head = use_contrastive_head

        # Encoder
        self.encoder = InfomaxEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_rbf=num_rbf,
            cutoff=cutoff,
        )

        # Projection head for contrastive learning
        if use_contrastive_head:
            self.local_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.global_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        # Property prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        # Conformer aggregation
        self.conformer_agg = ConformerAggregation(
            hidden_dim=output_dim,
            aggregation=conformer_aggregation,
        )

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        conformer_idx: Optional[torch.Tensor] = None,
        energies: Optional[torch.Tensor] = None,
        return_contrastive: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass.

        Args:
            atomic_numbers: Atomic numbers
            positions: 3D positions
            batch: Batch indices
            conformer_idx: Conformer indices for ensemble
            energies: Conformer energies
            return_contrastive: Whether to return contrastive representations

        Returns:
            Predictions and optionally contrastive representations
        """
        # Encode
        node_rep, graph_rep = self.encoder(atomic_numbers, positions, batch)

        # Property prediction
        predictions = self.prediction_head(graph_rep)

        if return_contrastive and self.use_contrastive_head:
            local_z = self.local_proj(node_rep)
            global_z = self.global_proj(graph_rep)
            return predictions, {
                'local': local_z,
                'global': global_z,
                'batch': batch,
            }

        return predictions

    def contrastive_loss(
        self,
        local_z: torch.Tensor,
        global_z: torch.Tensor,
        batch: torch.Tensor,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """
        Compute contrastive loss for pre-training.

        Maximizes agreement between node representations and their
        corresponding graph representation.
        """
        # Gather global representations for each node
        global_expanded = global_z[batch]  # (num_atoms, hidden_dim)

        # Normalize
        local_z = F.normalize(local_z, dim=-1)
        global_z_norm = F.normalize(global_expanded, dim=-1)
        all_global = F.normalize(global_z, dim=-1)

        # Positive pairs: local-global from same molecule
        pos_sim = (local_z * global_z_norm).sum(dim=-1) / temperature

        # Negative pairs: local vs all other globals
        neg_sim = torch.mm(local_z, all_global.t()) / temperature

        # NCE loss
        labels = batch
        loss = F.cross_entropy(neg_sim, labels)

        return loss


# =============================================================================
# GEM (Geometry-Enhanced Molecular) Implementation
# =============================================================================

class GEM(nn.Module):
    """
    GEM: Geometry-Enhanced Molecular representation learning.

    Incorporates both 2D topology and 3D geometry through a
    unified message passing framework with geometry-aware attention.

    Architecture:
        1. Atom and bond embeddings
        2. Geometry-enhanced message passing with distance/angle features
        3. Virtual node for global information aggregation
        4. Multi-task output heads

    Reference:
        Fang et al. "Geometry-enhanced molecular representation learning
        for property prediction" (Nature Machine Intelligence, 2022)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_rbf: int = 50,
        cutoff: float = 5.0,
        output_dim: int = 1,
        task: str = "regression",
        conformer_aggregation: str = "mean",
        use_virtual_node: bool = True,
        num_heads: int = 4,
    ):
        """
        Initialize GEM.

        Args:
            hidden_dim: Hidden dimension
            num_layers: Number of message passing layers
            num_rbf: Number of radial basis functions
            cutoff: Distance cutoff
            output_dim: Output dimension
            task: 'regression' or 'classification'
            conformer_aggregation: How to aggregate conformers
            use_virtual_node: Whether to use virtual node
            num_heads: Number of attention heads
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.task = task
        self.cutoff = cutoff
        self.use_virtual_node = use_virtual_node

        # Atom embedding
        self.atom_embedding = nn.Embedding(100, hidden_dim)

        # RBF for distance embedding
        self.rbf = RadialBasisFunctions(num_rbf, cutoff)

        # Bond/edge embedding
        self.edge_embedding = nn.Linear(num_rbf, hidden_dim)

        # Virtual node embedding
        if use_virtual_node:
            self.virtual_node = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
            self.vn_update = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # Geometry-enhanced message passing layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "attn": nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=0.1,
                    batch_first=True,
                ),
                "msg": nn.Sequential(
                    nn.Linear(hidden_dim * 2 + num_rbf, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ),
                "upd": nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                ),
            }))

        # Output head
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2 if use_virtual_node else hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

        # Conformer aggregation
        self.conformer_agg = ConformerAggregation(
            hidden_dim=output_dim,
            aggregation=conformer_aggregation,
        )

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        batch: torch.Tensor,
        conformer_idx: Optional[torch.Tensor] = None,
        energies: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with geometry-enhanced message passing.

        Args:
            atomic_numbers: Atomic numbers
            positions: 3D atomic positions
            batch: Batch indices
            conformer_idx: Conformer indices
            energies: Conformer energies

        Returns:
            Property predictions
        """
        x = self.atom_embedding(atomic_numbers.clamp(0, 99))
        num_atoms = positions.shape[0]

        # Build edges
        row = torch.arange(num_atoms, device=positions.device).repeat_interleave(num_atoms)
        col = torch.arange(num_atoms, device=positions.device).repeat(num_atoms)
        mask = row != col
        row, col = row[mask], col[mask]

        distances = (positions[row] - positions[col]).norm(dim=-1)
        cutoff_mask = distances < self.cutoff
        row, col = row[cutoff_mask], col[cutoff_mask]
        distances = distances[cutoff_mask]

        edge_index = torch.stack([row, col], dim=0)
        rbf = self.rbf(distances)
        edge_features = self.edge_embedding(rbf)

        # Initialize virtual node
        num_molecules = batch.max().item() + 1
        if self.use_virtual_node:
            vn = self.virtual_node.expand(num_molecules, -1)

        # Message passing
        for layer in self.layers:
            # Get neighbor features
            x_j = x[edge_index[0]]
            x_i = x[edge_index[1]]

            # Geometry-enhanced message
            msg_input = torch.cat([x_i, x_j, rbf], dim=-1)
            messages = layer["msg"](msg_input)

            # Aggregate
            aggr = torch.zeros_like(x)
            aggr.scatter_add_(0, edge_index[1].unsqueeze(-1).expand_as(messages), messages)

            # Update with virtual node information
            if self.use_virtual_node:
                vn_expanded = vn[batch]
                x = layer["upd"](torch.cat([aggr + vn_expanded, x], dim=-1))

                # Update virtual node
                graph_sum = torch.zeros(num_molecules, x.shape[-1], device=x.device, dtype=x.dtype)
                graph_sum.scatter_add_(0, batch.unsqueeze(-1).expand_as(x), x)
                counts = torch.zeros(num_molecules, device=x.device)
                counts.scatter_add_(0, batch, torch.ones_like(batch, dtype=torch.float))
                graph_mean = graph_sum / counts.unsqueeze(-1).clamp(min=1)

                vn_input = graph_mean.unsqueeze(1)
                vn_out, _ = self.vn_update(vn_input)
                vn = vn + vn_out.squeeze(1)
            else:
                x = layer["upd"](torch.cat([aggr, x], dim=-1))

        # Global pooling
        graph_rep = torch.zeros(num_molecules, x.shape[-1], device=x.device, dtype=x.dtype)
        graph_rep.scatter_add_(0, batch.unsqueeze(-1).expand_as(x), x)
        counts = torch.zeros(num_molecules, device=x.device)
        counts.scatter_add_(0, batch, torch.ones_like(batch, dtype=torch.float))
        graph_rep = graph_rep / counts.unsqueeze(-1).clamp(min=1)

        # Combine with virtual node
        if self.use_virtual_node:
            graph_rep = torch.cat([graph_rep, vn], dim=-1)

        # Output
        predictions = self.output(graph_rep)

        return predictions


# =============================================================================
# Factory Functions
# =============================================================================

def get_gnn(
    name: str,
    use_pyg: bool = True,
    **kwargs,
) -> nn.Module:
    """
    Factory function to get GNN model.

    Args:
        name: Model name ('schnet', 'dimenet', 'spherenet', '3d-infomax', 'gem')
        use_pyg: Whether to use PyTorch Geometric (if available)
        **kwargs: Model-specific arguments

    Returns:
        GNN model instance
    """
    name = name.lower().replace("-", "").replace("_", "")

    if use_pyg and HAS_PYG:
        if name == "schnet":
            return SchNetPyG(**kwargs)
        elif name in ["dimenet", "dimenetpp", "dimenet++"]:
            return DimeNetPPPyG(**kwargs)
        elif name not in ["3dinfomax", "gem", "spherenet"]:
            warnings.warn(f"PyG implementation not available for {name}, using simplified")

    # Simplified implementations and custom models
    if name == "schnet":
        return SchNet(**kwargs)
    elif name in ["dimenet", "dimenetpp", "dimenet++"]:
        return DimeNetPP(**kwargs)
    elif name == "spherenet":
        return SphereNet(**kwargs)
    elif name in ["3dinfomax", "infomax", "threediinfomax"]:
        return ThreeDInfomax(**kwargs)
    elif name == "gem":
        return GEM(**kwargs)
    else:
        raise ValueError(f"Unknown GNN: {name}. Available: schnet, dimenet, spherenet, 3d-infomax, gem")


class GNNWithConformerAggregation(nn.Module):
    """
    Wrapper to add conformer aggregation to any GNN.

    Processes each conformer through the base GNN, then aggregates
    the per-conformer predictions.
    """

    def __init__(
        self,
        base_gnn: nn.Module,
        aggregation: str = "mean",
        use_attention: bool = False,
        attention_hidden_dim: int = 64,
    ):
        """
        Initialize wrapper.

        Args:
            base_gnn: Base GNN model
            aggregation: 'mean', 'max', 'attention', or 'boltzmann'
            use_attention: Whether to use attention aggregation
            attention_hidden_dim: Hidden dim for attention
        """
        super().__init__()

        self.base_gnn = base_gnn
        self.aggregation = aggregation
        self.use_attention = use_attention

        if use_attention:
            out_dim = getattr(base_gnn, 'out_channels', 128)
            self.attention = nn.Sequential(
                nn.Linear(out_dim, attention_hidden_dim),
                nn.ReLU(),
                nn.Linear(attention_hidden_dim, 1),
            )

    def forward(
        self,
        conformer_data: List[Dict],
        energies: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with conformer aggregation."""
        conformer_outputs = []
        for conf in conformer_data:
            out = self.base_gnn(
                conf['atomic_numbers'],
                conf['positions'],
                conf['batch'],
            )
            conformer_outputs.append(out)

        outputs = torch.stack(conformer_outputs, dim=1)

        if self.aggregation == "mean":
            return outputs.mean(dim=1)
        elif self.aggregation == "max":
            return outputs.max(dim=1)[0]
        elif self.use_attention:
            attn = self.attention(outputs).squeeze(-1)
            attn = F.softmax(attn, dim=1)
            return (outputs * attn.unsqueeze(-1)).sum(dim=1)
        elif self.aggregation == "boltzmann" and energies is not None:
            weights = F.softmax(-energies, dim=1)
            return (outputs * weights.unsqueeze(-1)).sum(dim=1)
        else:
            return outputs.mean(dim=1)
