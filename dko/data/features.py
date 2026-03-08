"""
Feature extraction for molecular conformers.

This module provides utilities for extracting geometric features from 3D
molecular conformers, including pairwise distances, bond angles, torsion
angles, and augmented basis construction for the DKO algorithm.

The augmented basis construction follows the approach from:
"Distribution Kernel Operators for Molecular Property Prediction"
where features are represented as [μ, Σ + μμ^T] for proper kernel embedding.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None
    AllChem = None
    Descriptors = None


def check_rdkit():
    """Check if RDKit is available."""
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for feature extraction.")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class GeometricFeatures:
    """
    Container for geometric features extracted from a conformer.

    Attributes:
        distances: Pairwise distances between atoms within cutoff (n_pairs,)
        angles: Bond angles for connected atom triplets (n_angles,)
        torsions: Torsion angles for connected atom quadruplets (n_torsions,)
        distance_pairs: Atom pair indices for distances (n_pairs, 2)
        angle_triplets: Atom triplet indices for angles (n_angles, 3)
        torsion_quadruplets: Atom quadruplet indices for torsions (n_torsions, 4)
        coordinates: Raw 3D coordinates (n_atoms, 3)
        atom_features: Atom-level features (n_atoms, n_atom_features)
        n_atoms: Number of atoms
        metadata: Additional metadata
    """
    distances: np.ndarray
    angles: np.ndarray
    torsions: np.ndarray
    distance_pairs: np.ndarray
    angle_triplets: np.ndarray
    torsion_quadruplets: np.ndarray
    coordinates: np.ndarray
    atom_features: Optional[np.ndarray] = None
    n_atoms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_flat_vector(self, include_torsions: bool = True) -> np.ndarray:
        """
        Flatten all geometric features to a single vector.

        Args:
            include_torsions: Whether to include torsion angles

        Returns:
            Flattened feature vector
        """
        components = [self.distances, self.angles]
        if include_torsions:
            components.append(self.torsions)
        if self.atom_features is not None:
            components.append(self.atom_features.flatten())
        return np.concatenate(components)

    def get_feature_dim(self, include_torsions: bool = True) -> int:
        """Get total feature dimension."""
        dim = len(self.distances) + len(self.angles)
        if include_torsions:
            dim += len(self.torsions)
        if self.atom_features is not None:
            dim += self.atom_features.size
        return dim


@dataclass
class AugmentedBasis:
    """
    Augmented basis representation for DKO.

    The augmented basis is [μ, Σ + μμ^T] which provides a proper embedding
    for the distribution kernel. This representation captures both the mean
    and covariance structure of the conformer ensemble.

    Attributes:
        mean: Mean feature vector μ (feature_dim,)
        second_order: Second-order tensor Σ + μμ^T (feature_dim, feature_dim)
        flat_representation: Flattened [μ, vec(Σ + μμ^T)] vector
        n_conformers: Number of conformers used to compute statistics
        weights: Weights used for computing statistics (e.g., Boltzmann)
        feature_dim: Dimension of the feature vector
        metadata: Additional metadata
    """
    mean: np.ndarray
    second_order: np.ndarray
    flat_representation: np.ndarray
    n_conformers: int
    weights: Optional[np.ndarray] = None
    feature_dim: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_tensor(self) -> torch.Tensor:
        """Convert flat representation to PyTorch tensor."""
        return torch.tensor(self.flat_representation, dtype=torch.float32)

    def get_mean_tensor(self) -> torch.Tensor:
        """Get mean as PyTorch tensor."""
        return torch.tensor(self.mean, dtype=torch.float32)

    def get_second_order_tensor(self) -> torch.Tensor:
        """Get second-order tensor as PyTorch tensor."""
        return torch.tensor(self.second_order, dtype=torch.float32)


# =============================================================================
# Geometric Feature Extractor
# =============================================================================

class GeometricFeatureExtractor:
    """
    Extract geometric features from molecular conformers.

    Computes pairwise distances, bond angles, and torsion angles from
    3D molecular structures. Features are designed for use with the
    DKO kernel method.

    Attributes:
        distance_cutoff: Maximum distance to consider (Angstroms)
        include_atom_features: Whether to include atom-level features
        normalize: Whether to normalize features
    """

    def __init__(
        self,
        distance_cutoff: float = 4.0,
        include_atom_features: bool = True,
        normalize: bool = False,
        use_cos_sin_torsions: bool = True,
    ):
        """
        Initialize the geometric feature extractor.

        Args:
            distance_cutoff: Maximum distance for pairwise features (Angstroms)
            include_atom_features: Whether to extract atom-level features
            normalize: Whether to normalize extracted features
            use_cos_sin_torsions: Whether to use cos/sin representation for torsions
        """
        self.distance_cutoff = distance_cutoff
        self.include_atom_features = include_atom_features
        self.normalize = normalize
        self.use_cos_sin_torsions = use_cos_sin_torsions

    def extract(self, mol, conformer_id: int = -1) -> GeometricFeatures:
        """
        Extract geometric features from a molecule conformer.

        Args:
            mol: RDKit Mol object with conformer(s)
            conformer_id: ID of conformer to extract (-1 for first/only)

        Returns:
            GeometricFeatures object containing all extracted features
        """
        check_rdkit()

        # Get conformer
        if conformer_id < 0:
            conf = mol.GetConformer(0)
        else:
            conf = mol.GetConformer(conformer_id)

        n_atoms = mol.GetNumAtoms()

        # Extract 3D coordinates
        coordinates = np.zeros((n_atoms, 3))
        for i in range(n_atoms):
            pos = conf.GetAtomPosition(i)
            coordinates[i] = [pos.x, pos.y, pos.z]

        # Compute features
        distances, distance_pairs = self._compute_distances(coordinates)
        angles, angle_triplets = self._compute_angles(mol, coordinates)
        torsions, torsion_quadruplets = self._compute_torsions(mol, coordinates)

        # Atom features
        atom_features = None
        if self.include_atom_features:
            atom_features = self._compute_atom_features(mol)

        # Normalize if requested
        if self.normalize:
            distances = self._normalize(distances)
            angles = self._normalize(angles)
            if len(torsions) > 0:
                torsions = self._normalize(torsions)
            if atom_features is not None:
                atom_features = self._normalize(atom_features)

        return GeometricFeatures(
            distances=distances,
            angles=angles,
            torsions=torsions,
            distance_pairs=distance_pairs,
            angle_triplets=angle_triplets,
            torsion_quadruplets=torsion_quadruplets,
            coordinates=coordinates,
            atom_features=atom_features,
            n_atoms=n_atoms,
            metadata={"conformer_id": conformer_id},
        )

    def extract_batch(
        self,
        mol,
        conformer_ids: Optional[List[int]] = None,
    ) -> List[GeometricFeatures]:
        """
        Extract features from multiple conformers of the same molecule.

        Args:
            mol: RDKit Mol object with multiple conformers
            conformer_ids: List of conformer IDs to extract (None for all)

        Returns:
            List of GeometricFeatures objects
        """
        if conformer_ids is None:
            conformer_ids = list(range(mol.GetNumConformers()))

        return [self.extract(mol, conf_id) for conf_id in conformer_ids]

    def _compute_distances(
        self,
        coordinates: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute pairwise distances between atoms within cutoff.

        Args:
            coordinates: Atomic coordinates (n_atoms, 3)

        Returns:
            Tuple of (distances, pair_indices)
        """
        n_atoms = coordinates.shape[0]

        # Compute full distance matrix
        diff = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

        # Get upper triangle indices (excluding diagonal)
        i_indices, j_indices = np.triu_indices(n_atoms, k=1)

        # Filter by cutoff
        all_distances = dist_matrix[i_indices, j_indices]
        within_cutoff = all_distances <= self.distance_cutoff

        distances = all_distances[within_cutoff]
        pair_indices = np.column_stack([
            i_indices[within_cutoff],
            j_indices[within_cutoff]
        ])

        return distances, pair_indices

    def _compute_angles(
        self,
        mol,
        coordinates: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute bond angles for all connected atom triplets.

        Args:
            mol: RDKit Mol object
            coordinates: Atomic coordinates (n_atoms, 3)

        Returns:
            Tuple of (angles in radians, triplet_indices)
        """
        angles = []
        triplets = []

        for atom in mol.GetAtoms():
            center_idx = atom.GetIdx()
            neighbors = [n.GetIdx() for n in atom.GetNeighbors()]

            if len(neighbors) < 2:
                continue

            # Compute angles for all neighbor pairs
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    n1_idx, n2_idx = neighbors[i], neighbors[j]

                    # Vectors from central atom to neighbors
                    v1 = coordinates[n1_idx] - coordinates[center_idx]
                    v2 = coordinates[n2_idx] - coordinates[center_idx]

                    # Compute angle using dot product
                    norm1 = np.linalg.norm(v1)
                    norm2 = np.linalg.norm(v2)

                    if norm1 > 1e-8 and norm2 > 1e-8:
                        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)
                        angle = np.arccos(cos_angle)
                        angles.append(angle)
                        triplets.append([n1_idx, center_idx, n2_idx])

        if len(angles) == 0:
            return np.array([]), np.array([]).reshape(0, 3)

        return np.array(angles), np.array(triplets)

    def _compute_torsions(
        self,
        mol,
        coordinates: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute torsion angles for all connected atom quadruplets.

        Args:
            mol: RDKit Mol object
            coordinates: Atomic coordinates (n_atoms, 3)

        Returns:
            Tuple of (torsion angles, quadruplet_indices)
        """
        torsions = []
        quadruplets = []

        # Find all paths of length 4 (torsion backbone)
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            # Get neighbors excluding the other atom in the central bond
            atom_i_neighbors = [
                n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()
                if n.GetIdx() != j
            ]
            atom_j_neighbors = [
                n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors()
                if n.GetIdx() != i
            ]

            # Form quadruplets
            for a in atom_i_neighbors:
                for d in atom_j_neighbors:
                    torsion = compute_dihedral(
                        coordinates[a],
                        coordinates[i],
                        coordinates[j],
                        coordinates[d]
                    )

                    if self.use_cos_sin_torsions:
                        # Store as [cos(θ), sin(θ)] for better numerical properties
                        torsions.extend([np.cos(torsion), np.sin(torsion)])
                    else:
                        torsions.append(torsion)

                    quadruplets.append([a, i, j, d])

        if len(torsions) == 0:
            return np.array([]), np.array([]).reshape(0, 4)

        return np.array(torsions), np.array(quadruplets)

    def _compute_atom_features(self, mol) -> np.ndarray:
        """
        Extract atom-level features.

        Args:
            mol: RDKit Mol object

        Returns:
            Atom features (n_atoms, n_features)
        """
        features = []

        for atom in mol.GetAtoms():
            atom_feat = []

            # Atomic number (one-hot for common elements)
            atomic_num = atom.GetAtomicNum()
            common_elements = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 15: 5, 16: 6, 17: 7, 35: 8, 53: 9}
            atom_onehot = [0] * 11  # 10 common + 1 other
            if atomic_num in common_elements:
                atom_onehot[common_elements[atomic_num]] = 1
            else:
                atom_onehot[10] = 1  # Other
            atom_feat.extend(atom_onehot)

            # Degree (normalized)
            atom_feat.append(atom.GetDegree() / 4.0)

            # Hybridization (one-hot)
            hyb = atom.GetHybridization()
            hyb_map = {
                Chem.HybridizationType.S: 0,
                Chem.HybridizationType.SP: 1,
                Chem.HybridizationType.SP2: 2,
                Chem.HybridizationType.SP3: 3,
                Chem.HybridizationType.SP3D: 4,
                Chem.HybridizationType.SP3D2: 5,
            }
            hyb_onehot = [0] * 6
            if hyb in hyb_map:
                hyb_onehot[hyb_map[hyb]] = 1
            atom_feat.extend(hyb_onehot)

            # Is aromatic
            atom_feat.append(float(atom.GetIsAromatic()))

            # Formal charge (normalized)
            atom_feat.append(atom.GetFormalCharge() / 2.0)

            # Number of hydrogens (normalized)
            atom_feat.append(atom.GetTotalNumHs() / 4.0)

            # Is in ring
            atom_feat.append(float(atom.IsInRing()))

            # Ring size (if in ring)
            ring_size = 0
            if atom.IsInRing():
                for size in range(3, 8):
                    if atom.IsInRingSize(size):
                        ring_size = size
                        break
            atom_feat.append(ring_size / 7.0)

            features.append(atom_feat)

        return np.array(features)

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to zero mean and unit variance."""
        if features.size == 0:
            return features
        mean = features.mean()
        std = features.std()
        if std > 1e-8:
            return (features - mean) / std
        return features - mean


# =============================================================================
# Augmented Basis Constructor
# =============================================================================

class AugmentedBasisConstructor:
    """
    Construct augmented basis representations for DKO.

    Computes the augmented basis [μ, Σ + μμ^T] from a set of conformer
    features. This representation is used as input to the DKO kernel.

    The augmented basis captures both the mean and covariance structure
    of the conformer ensemble distribution, enabling the kernel to compute
    proper distribution distances.

    Attributes:
        use_diagonal_only: Only use diagonal of covariance (for scalability)
        regularization: Regularization for covariance computation
        use_sqrt_weights: Whether to use sqrt of weights (for proper weighting)
    """

    def __init__(
        self,
        use_diagonal_only: bool = False,
        regularization: float = 1e-6,
        use_sqrt_weights: bool = True,
        max_feature_dim: Optional[int] = None,
    ):
        """
        Initialize the augmented basis constructor.

        Args:
            use_diagonal_only: Only use diagonal of second-order tensor
            regularization: Regularization term for numerical stability
            use_sqrt_weights: Use sqrt of weights for proper weighted averaging
            max_feature_dim: Maximum feature dimension (for truncation)
        """
        self.use_diagonal_only = use_diagonal_only
        self.regularization = regularization
        self.use_sqrt_weights = use_sqrt_weights
        self.max_feature_dim = max_feature_dim

    def construct(
        self,
        features_list: List[np.ndarray],
        weights: Optional[np.ndarray] = None,
    ) -> AugmentedBasis:
        """
        Construct augmented basis from a list of feature vectors.

        Args:
            features_list: List of feature vectors, one per conformer
            weights: Optional weights (e.g., Boltzmann weights)

        Returns:
            AugmentedBasis object
        """
        n_conformers = len(features_list)

        if n_conformers == 0:
            raise ValueError("Cannot construct augmented basis from empty list")

        # Pad features to same length (different conformers may have different feature dims)
        max_len = max(len(f) for f in features_list)
        padded_features = []
        for f in features_list:
            if len(f) < max_len:
                padded = np.pad(f, (0, max_len - len(f)), mode='constant', constant_values=0)
                padded_features.append(padded)
            else:
                padded_features.append(f)

        # Stack features
        features = np.stack(padded_features, axis=0)  # (n_conformers, feature_dim)
        feature_dim = features.shape[1]

        # Truncate if needed
        if self.max_feature_dim is not None and feature_dim > self.max_feature_dim:
            features = features[:, :self.max_feature_dim]
            feature_dim = self.max_feature_dim

        # Handle weights
        if weights is None:
            weights = np.ones(n_conformers) / n_conformers
        else:
            weights = np.asarray(weights)
            weights = weights / weights.sum()  # Normalize

        if self.use_sqrt_weights:
            sqrt_weights = np.sqrt(weights)
        else:
            sqrt_weights = weights

        # Compute weighted mean: μ = Σ_i w_i * x_i
        mean = np.sum(weights[:, np.newaxis] * features, axis=0)

        # Compute weighted covariance: Σ = Σ_i w_i * (x_i - μ)(x_i - μ)^T
        centered = features - mean[np.newaxis, :]

        if self.use_diagonal_only:
            # Only diagonal elements
            variance = np.sum(
                weights[:, np.newaxis] * centered ** 2,
                axis=0
            )
            # Second order: σ^2 + μ^2 (diagonal of Σ + μμ^T)
            second_order_diag = variance + mean ** 2
            second_order = np.diag(second_order_diag)
            flat_second_order = second_order_diag
        else:
            # Full covariance matrix
            covariance = np.zeros((feature_dim, feature_dim))
            for i in range(n_conformers):
                covariance += weights[i] * np.outer(centered[i], centered[i])

            # Add regularization
            covariance += self.regularization * np.eye(feature_dim)

            # Second order: Σ + μμ^T
            second_order = covariance + np.outer(mean, mean)
            flat_second_order = self.flatten_second_order(second_order)

        # Flatten representation: [μ, vec(Σ + μμ^T)]
        flat_representation = np.concatenate([mean, flat_second_order])

        return AugmentedBasis(
            mean=mean,
            second_order=second_order,
            flat_representation=flat_representation,
            n_conformers=n_conformers,
            weights=weights,
            feature_dim=feature_dim,
            metadata={
                "use_diagonal_only": self.use_diagonal_only,
                "regularization": self.regularization,
            },
        )

    def construct_from_geometric_features(
        self,
        geo_features_list: List[GeometricFeatures],
        weights: Optional[np.ndarray] = None,
        include_torsions: bool = True,
    ) -> AugmentedBasis:
        """
        Construct augmented basis from GeometricFeatures objects.

        Args:
            geo_features_list: List of GeometricFeatures objects
            weights: Optional weights
            include_torsions: Whether to include torsion features

        Returns:
            AugmentedBasis object
        """
        features_list = [
            gf.to_flat_vector(include_torsions=include_torsions)
            for gf in geo_features_list
        ]
        return self.construct(features_list, weights)

    def flatten_second_order(self, matrix: np.ndarray) -> np.ndarray:
        """
        Flatten second-order tensor to vector.

        For symmetric matrices, only stores upper triangle.

        Args:
            matrix: Square symmetric matrix (n, n)

        Returns:
            Flattened vector of upper triangle elements
        """
        n = matrix.shape[0]
        # Extract upper triangle including diagonal
        indices = np.triu_indices(n)
        return matrix[indices]

    def unflatten_second_order(
        self,
        flat: np.ndarray,
        feature_dim: int,
    ) -> np.ndarray:
        """
        Reconstruct second-order matrix from flattened vector.

        Args:
            flat: Flattened upper triangle vector
            feature_dim: Original matrix dimension

        Returns:
            Reconstructed symmetric matrix
        """
        matrix = np.zeros((feature_dim, feature_dim))
        indices = np.triu_indices(feature_dim)
        matrix[indices] = flat
        # Mirror to lower triangle
        matrix = matrix + matrix.T - np.diag(np.diag(matrix))
        return matrix


# =============================================================================
# SCC Computation
# =============================================================================

def compute_scc_simple(
    features_list: List[np.ndarray],
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Structural Conformational Complexity (SCC).

    SCC measures the structural diversity of a conformer ensemble based
    on the determinant of the covariance matrix of geometric features.
    Higher SCC indicates greater conformational diversity.

    SCC = log(det(Σ + εI)) / (2 * d)

    where Σ is the weighted covariance, ε is regularization, and d is dimension.

    Args:
        features_list: List of feature vectors, one per conformer
        weights: Optional Boltzmann weights for each conformer

    Returns:
        SCC score (scalar)
    """
    if len(features_list) < 2:
        return 0.0

    # Handle variable-length feature vectors by padding to max length
    max_len = max(len(f) for f in features_list)
    padded_features = []
    for f in features_list:
        if len(f) < max_len:
            padded = np.pad(f, (0, max_len - len(f)), mode='constant', constant_values=0)
            padded_features.append(padded)
        else:
            padded_features.append(f)

    # Stack features
    features = np.stack(padded_features, axis=0)
    n_conformers, feature_dim = features.shape

    # Handle weights
    if weights is None:
        weights = np.ones(n_conformers) / n_conformers
    else:
        weights = np.asarray(weights)
        weights = weights / weights.sum()

    # Weighted mean
    mean = np.sum(weights[:, np.newaxis] * features, axis=0)

    # Weighted covariance
    centered = features - mean[np.newaxis, :]
    covariance = np.zeros((feature_dim, feature_dim))
    for i in range(n_conformers):
        covariance += weights[i] * np.outer(centered[i], centered[i])

    # Add regularization
    epsilon = 1e-6
    covariance += epsilon * np.eye(feature_dim)

    # Compute log determinant
    sign, logdet = np.linalg.slogdet(covariance)

    if sign <= 0:
        # Matrix is not positive definite
        return 0.0

    # Normalize by dimension
    scc = logdet / (2 * feature_dim)

    return float(scc)


def compute_scc_from_ensemble(
    mol,
    conformer_ids: Optional[List[int]] = None,
    weights: Optional[np.ndarray] = None,
    extractor: Optional[GeometricFeatureExtractor] = None,
) -> float:
    """
    Compute SCC directly from an RDKit molecule with conformers.

    Args:
        mol: RDKit Mol object with conformers
        conformer_ids: List of conformer IDs to use (None for all)
        weights: Optional Boltzmann weights
        extractor: Optional GeometricFeatureExtractor instance

    Returns:
        SCC score
    """
    check_rdkit()

    if extractor is None:
        extractor = GeometricFeatureExtractor()

    if conformer_ids is None:
        conformer_ids = list(range(mol.GetNumConformers()))

    if len(conformer_ids) < 2:
        return 0.0

    # Extract features for all conformers
    features_list = []
    for conf_id in conformer_ids:
        geo_features = extractor.extract(mol, conf_id)
        features_list.append(geo_features.to_flat_vector())

    return compute_scc_simple(features_list, weights)


# =============================================================================
# Legacy Functions (for backwards compatibility)
# =============================================================================

def compute_pairwise_distances(
    coords: np.ndarray,
    cutoff: float = 4.0,
) -> np.ndarray:
    """
    Compute pairwise distances between atoms.

    Args:
        coords: Atomic coordinates (n_atoms, 3)
        cutoff: Distance cutoff in Angstroms

    Returns:
        Flattened upper triangle of distance matrix (with cutoff applied)
    """
    n_atoms = coords.shape[0]

    # Compute distance matrix
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

    # Apply cutoff (set distances beyond cutoff to cutoff value)
    dist_matrix = np.minimum(dist_matrix, cutoff)

    # Extract upper triangle (excluding diagonal)
    indices = np.triu_indices(n_atoms, k=1)
    distances = dist_matrix[indices]

    return distances


def compute_bond_angles(mol, coords: np.ndarray) -> np.ndarray:
    """
    Compute bond angles for all bonded atom triplets.

    Args:
        mol: RDKit Mol object
        coords: Atomic coordinates (n_atoms, 3)

    Returns:
        Array of bond angles in radians
    """
    angles = []

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]

        if len(neighbors) < 2:
            continue

        # Compute angles for all neighbor pairs
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                n1, n2 = neighbors[i], neighbors[j]

                # Vectors from central atom to neighbors
                v1 = coords[n1] - coords[idx]
                v2 = coords[n2] - coords[idx]

                # Compute angle
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                angles.append(angle)

    return np.array(angles) if angles else np.array([0.0])


def compute_torsion_angles(mol, coords: np.ndarray) -> np.ndarray:
    """
    Compute torsion angles for all rotatable bonds.

    Args:
        mol: RDKit Mol object
        coords: Atomic coordinates (n_atoms, 3)

    Returns:
        Array of torsion angles in radians
    """
    torsions = []

    # Get rotatable bonds
    rotatable_pattern = Chem.MolFromSmarts("[!$(*#*)&!D1]-!@[!$(*#*)&!D1]")
    matches = mol.GetSubstructMatches(rotatable_pattern)

    for match in matches:
        i, j = match

        # Get neighbors to form quadruplet
        atom_i_neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() != j]
        atom_j_neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != i]

        if not atom_i_neighbors or not atom_j_neighbors:
            continue

        # Use first neighbor for simplicity
        a = atom_i_neighbors[0]
        d = atom_j_neighbors[0]

        # Compute torsion angle
        torsion = compute_dihedral(coords[a], coords[i], coords[j], coords[d])
        torsions.append(torsion)

    return np.array(torsions) if torsions else np.array([0.0])


def compute_dihedral(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
) -> float:
    """
    Compute dihedral angle between four points.

    Args:
        p1, p2, p3, p4: 3D coordinates of four points

    Returns:
        Dihedral angle in radians
    """
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    # Normal vectors
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    # Handle degenerate cases
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)

    if n1_norm < 1e-8 or n2_norm < 1e-8:
        return 0.0

    # Normalize
    n1 = n1 / n1_norm
    n2 = n2 / n2_norm

    # Compute angle
    cos_angle = np.dot(n1, n2)
    cos_angle = np.clip(cos_angle, -1, 1)

    # Determine sign
    b2_norm = np.linalg.norm(b2)
    if b2_norm > 1e-8:
        m1 = np.cross(n1, b2 / b2_norm)
        dot_product = np.dot(m1, n2)
        # Handle the edge case where dot product is exactly 0
        sign = np.sign(dot_product) if abs(dot_product) > 1e-8 else 1.0
    else:
        sign = 1.0

    angle = sign * np.arccos(cos_angle)

    return angle


# =============================================================================
# Feature Extractor (Legacy)
# =============================================================================

class FeatureExtractor:
    """
    Extract features from molecular conformers.

    Supports various feature types including pairwise distances,
    bond angles, torsion angles, and atom-level features.

    Note: Consider using GeometricFeatureExtractor for new code.
    """

    def __init__(
        self,
        include_pairwise_distances: bool = True,
        include_bond_angles: bool = True,
        include_torsion_angles: bool = True,
        include_atom_features: bool = True,
        distance_cutoff: float = 4.0,
        normalize: bool = True,
        flatten: bool = True,
    ):
        """
        Initialize feature extractor.

        Args:
            include_pairwise_distances: Whether to include pairwise distances
            include_bond_angles: Whether to include bond angles
            include_torsion_angles: Whether to include torsion angles
            include_atom_features: Whether to include atom-level features
            distance_cutoff: Cutoff for pairwise distances in Angstroms
            normalize: Whether to normalize features
            flatten: Whether to flatten features to 1D
        """
        self.include_pairwise_distances = include_pairwise_distances
        self.include_bond_angles = include_bond_angles
        self.include_torsion_angles = include_torsion_angles
        self.include_atom_features = include_atom_features
        self.distance_cutoff = distance_cutoff
        self.normalize = normalize
        self.flatten = flatten

        # Feature dimension will be computed on first extraction
        self._feature_dim = None

    @property
    def feature_dim(self) -> int:
        """Get feature dimension."""
        if self._feature_dim is None:
            # Estimate based on typical small molecule
            return 1000  # Placeholder
        return self._feature_dim

    def extract(
        self,
        conformers: List,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Extract features from a list of conformers.

        Args:
            conformers: List of RDKit Mol objects with conformers
            return_dict: Whether to return a dictionary of feature types

        Returns:
            Feature tensor of shape (n_conformers, feature_dim) or dict
        """
        check_rdkit()

        all_features = []

        for mol in conformers:
            features = self._extract_single(mol)
            all_features.append(features)

        # Stack features
        if return_dict:
            result = {}
            for key in all_features[0].keys():
                result[key] = torch.stack([f[key] for f in all_features])
            return result
        else:
            # Concatenate all feature types
            stacked = []
            for features in all_features:
                concat = torch.cat([v.flatten() for v in features.values()])
                stacked.append(concat)

            result = torch.stack(stacked)
            self._feature_dim = result.shape[-1]

            return result

    def _extract_single(self, mol) -> Dict[str, torch.Tensor]:
        """Extract features from a single conformer."""
        features = {}
        conf = mol.GetConformer()
        n_atoms = mol.GetNumAtoms()

        # Get coordinates
        coords = np.zeros((n_atoms, 3))
        for i in range(n_atoms):
            pos = conf.GetAtomPosition(i)
            coords[i] = [pos.x, pos.y, pos.z]

        # Pairwise distances
        if self.include_pairwise_distances:
            distances = compute_pairwise_distances(coords, self.distance_cutoff)
            features["distances"] = torch.tensor(distances, dtype=torch.float32)

        # Bond angles
        if self.include_bond_angles:
            angles = compute_bond_angles(mol, coords)
            features["angles"] = torch.tensor(angles, dtype=torch.float32)

        # Torsion angles
        if self.include_torsion_angles:
            torsions = compute_torsion_angles(mol, coords)
            features["torsions"] = torch.tensor(torsions, dtype=torch.float32)

        # Atom features
        if self.include_atom_features:
            atom_feats = self._get_atom_features(mol)
            features["atoms"] = torch.tensor(atom_feats, dtype=torch.float32)

        # Normalize if requested
        if self.normalize:
            for key in features:
                f = features[key]
                if f.numel() > 0 and f.std() > 0:
                    features[key] = (f - f.mean()) / (f.std() + 1e-8)

        return features

    def _get_atom_features(self, mol) -> np.ndarray:
        """Extract atom-level features."""
        features = []

        for atom in mol.GetAtoms():
            atom_feat = []

            # Atomic number (one-hot for common elements)
            atomic_num = atom.GetAtomicNum()
            atom_onehot = [0] * 10
            common_elements = {6: 0, 7: 1, 8: 2, 9: 3, 15: 4, 16: 5, 17: 6, 35: 7, 53: 8}
            if atomic_num in common_elements:
                atom_onehot[common_elements[atomic_num]] = 1
            else:
                atom_onehot[9] = 1  # Other
            atom_feat.extend(atom_onehot)

            # Degree
            atom_feat.append(atom.GetDegree() / 4.0)

            # Hybridization (one-hot)
            hyb = atom.GetHybridization()
            hyb_onehot = [0] * 5
            hyb_map = {
                Chem.HybridizationType.SP: 0,
                Chem.HybridizationType.SP2: 1,
                Chem.HybridizationType.SP3: 2,
                Chem.HybridizationType.SP3D: 3,
                Chem.HybridizationType.SP3D2: 4,
            }
            if hyb in hyb_map:
                hyb_onehot[hyb_map[hyb]] = 1
            atom_feat.extend(hyb_onehot)

            # Is aromatic
            atom_feat.append(float(atom.GetIsAromatic()))

            # Formal charge
            atom_feat.append(atom.GetFormalCharge() / 2.0)

            # Num H
            atom_feat.append(atom.GetTotalNumHs() / 4.0)

            # Is in ring
            atom_feat.append(float(atom.IsInRing()))

            features.append(atom_feat)

        return np.array(features).flatten()


class MolecularFingerprints:
    """
    Generate molecular fingerprints for each conformer.

    Combines 2D fingerprints with 3D shape-based fingerprints.
    """

    def __init__(
        self,
        fp_type: str = "morgan",
        radius: int = 2,
        n_bits: int = 2048,
        include_3d: bool = True,
    ):
        """
        Initialize fingerprint generator.

        Args:
            fp_type: Fingerprint type ('morgan', 'rdkit', 'maccs')
            radius: Morgan fingerprint radius
            n_bits: Number of bits for fingerprint
            include_3d: Whether to include 3D-based fingerprints
        """
        self.fp_type = fp_type
        self.radius = radius
        self.n_bits = n_bits
        self.include_3d = include_3d

    def generate(self, mol) -> np.ndarray:
        """Generate fingerprint for a molecule."""
        check_rdkit()

        if self.fp_type == "morgan":
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, self.radius, nBits=self.n_bits
            )
        elif self.fp_type == "rdkit":
            fp = Chem.RDKFingerprint(mol, fpSize=self.n_bits)
        elif self.fp_type == "maccs":
            fp = AllChem.GetMACCSKeysFingerprint(mol)
        else:
            raise ValueError(f"Unknown fingerprint type: {self.fp_type}")

        fp_array = np.zeros(fp.GetNumBits())
        for i in range(fp.GetNumBits()):
            fp_array[i] = fp.GetBit(i)

        return fp_array


# =============================================================================
# Entropy Recovery (Theorem 4)
# =============================================================================

class ConformationalEntropyCalculator:
    """
    Calculate conformational entropy from covariance matrix.

    Theorem 4 (Entropy Recovery):
    Under the harmonic approximation (near-Gaussian conformational landscapes),
    conformational entropy equals:

        S = S_0 + (1/2) * ln(det(Σ))

    where:
        S_0 is a baseline constant depending on dimension
        Σ is the covariance matrix of geometric features

    This is significant because conformational entropy contributes to binding
    free energy via entropy-enthalpy compensation. DKO's explicit covariance
    representation enables direct recovery of this quantity.

    Limitations:
        - Assumes harmonic/near-Gaussian energy landscapes
        - May not hold for systems with multiple deep energy minima
        - Regularization affects small eigenvalues
    """

    # Physical constants
    KB = 1.987204e-3  # Boltzmann constant in kcal/(mol*K)
    HBAR = 1.054571817e-34  # Reduced Planck constant in J*s

    def __init__(
        self,
        temperature: float = 300.0,
        regularization: float = 1e-6,
        use_log_det: bool = True,
    ):
        """
        Initialize entropy calculator.

        Args:
            temperature: Temperature in Kelvin
            regularization: Regularization for numerical stability
            use_log_det: Use numerically stable log-det computation
        """
        self.temperature = temperature
        self.regularization = regularization
        self.use_log_det = use_log_det

    def compute_from_covariance(
        self,
        covariance: np.ndarray,
        normalize: bool = True,
    ) -> Dict[str, float]:
        """
        Compute conformational entropy from covariance matrix.

        S = S_0 + (1/2) * ln(det(Σ))

        For a D-dimensional Gaussian, S_0 = D/2 * (1 + ln(2π))

        Args:
            covariance: Covariance matrix (D, D)
            normalize: Whether to normalize by dimension

        Returns:
            Dictionary with entropy and related quantities
        """
        D = covariance.shape[0]

        # Add regularization
        cov_reg = covariance + self.regularization * np.eye(D)

        # Compute log determinant (numerically stable)
        if self.use_log_det:
            sign, log_det = np.linalg.slogdet(cov_reg)
            if sign <= 0:
                # Matrix not positive definite - use eigenvalue approach
                eigenvalues = np.linalg.eigvalsh(cov_reg)
                eigenvalues = np.maximum(eigenvalues, self.regularization)
                log_det = np.sum(np.log(eigenvalues))
        else:
            det = np.linalg.det(cov_reg)
            log_det = np.log(max(det, 1e-300))

        # Baseline entropy for D-dimensional Gaussian
        # S_0 = D/2 * (1 + ln(2π))
        S_0 = D / 2 * (1 + np.log(2 * np.pi))

        # Total entropy
        # S = S_0 + (1/2) * ln(det(Σ))
        entropy = S_0 + 0.5 * log_det

        # Normalized entropy (per degree of freedom)
        entropy_per_dof = entropy / D if D > 0 else 0.0

        # Convert to kcal/mol at temperature
        # Using S in natural units, -TS gives free energy contribution
        free_energy_contribution = -self.KB * self.temperature * entropy

        return {
            "entropy": float(entropy),
            "entropy_per_dof": float(entropy_per_dof),
            "log_det": float(log_det),
            "baseline_entropy": float(S_0),
            "dimension": D,
            "free_energy_contribution": float(free_energy_contribution),
            "temperature": self.temperature,
        }

    def compute_from_augmented_basis(
        self,
        augmented_basis: 'AugmentedBasis',
    ) -> Dict[str, float]:
        """
        Compute entropy from DKO's AugmentedBasis representation.

        Args:
            augmented_basis: AugmentedBasis object

        Returns:
            Entropy dictionary
        """
        # Extract covariance from second-order representation
        # second_order = Σ + μμ^T, so Σ = second_order - μμ^T
        mean = augmented_basis.mean
        second_order = augmented_basis.second_order
        covariance = second_order - np.outer(mean, mean)

        return self.compute_from_covariance(covariance)

    def compute_from_conformer_features(
        self,
        features_list: List[np.ndarray],
        weights: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute entropy directly from conformer features.

        Args:
            features_list: List of feature vectors, one per conformer
            weights: Optional Boltzmann weights

        Returns:
            Entropy dictionary
        """
        if len(features_list) < 2:
            return {
                "entropy": 0.0,
                "entropy_per_dof": 0.0,
                "log_det": float("-inf"),
                "error": "Need at least 2 conformers",
            }

        # Pad features to same length
        max_len = max(len(f) for f in features_list)
        padded = []
        for f in features_list:
            if len(f) < max_len:
                padded.append(np.pad(f, (0, max_len - len(f))))
            else:
                padded.append(f)

        features = np.stack(padded, axis=0)
        n_conf, D = features.shape

        # Handle weights
        if weights is None:
            weights = np.ones(n_conf) / n_conf
        else:
            weights = np.asarray(weights)
            weights = weights / weights.sum()

        # Weighted mean
        mean = np.sum(weights[:, np.newaxis] * features, axis=0)

        # Weighted covariance
        centered = features - mean[np.newaxis, :]
        covariance = np.zeros((D, D))
        for i in range(n_conf):
            covariance += weights[i] * np.outer(centered[i], centered[i])

        return self.compute_from_covariance(covariance)

    def compute_from_ensemble(
        self,
        mol,
        conformer_ids: Optional[List[int]] = None,
        weights: Optional[np.ndarray] = None,
        extractor: Optional['GeometricFeatureExtractor'] = None,
    ) -> Dict[str, float]:
        """
        Compute entropy from RDKit molecule with conformers.

        Args:
            mol: RDKit Mol object with conformers
            conformer_ids: Conformer IDs to use (None for all)
            weights: Boltzmann weights
            extractor: Feature extractor

        Returns:
            Entropy dictionary
        """
        if extractor is None:
            extractor = GeometricFeatureExtractor()

        if conformer_ids is None:
            conformer_ids = list(range(mol.GetNumConformers()))

        if len(conformer_ids) < 2:
            return {"entropy": 0.0, "error": "Need at least 2 conformers"}

        # Extract features
        features_list = []
        for conf_id in conformer_ids:
            geo_features = extractor.extract(mol, conf_id)
            features_list.append(geo_features.to_flat_vector())

        return self.compute_from_conformer_features(features_list, weights)


def compute_conformational_entropy(
    features_list: List[np.ndarray],
    weights: Optional[np.ndarray] = None,
    temperature: float = 300.0,
) -> float:
    """
    Convenience function to compute conformational entropy.

    Args:
        features_list: List of feature vectors
        weights: Optional Boltzmann weights
        temperature: Temperature in Kelvin

    Returns:
        Conformational entropy
    """
    calculator = ConformationalEntropyCalculator(temperature=temperature)
    result = calculator.compute_from_conformer_features(features_list, weights)
    return result["entropy"]


def compute_entropy_contribution_to_binding(
    ligand_features: List[np.ndarray],
    complex_features: List[np.ndarray],
    ligand_weights: Optional[np.ndarray] = None,
    complex_weights: Optional[np.ndarray] = None,
    temperature: float = 300.0,
) -> Dict[str, float]:
    """
    Compute entropy contribution to binding free energy.

    ΔS_binding = S_complex - S_free_ligand

    This is typically negative (entropy loss upon binding) and contributes
    to -TΔS term in the binding free energy.

    Args:
        ligand_features: Features for free ligand conformers
        complex_features: Features for bound ligand conformers
        ligand_weights: Boltzmann weights for free ligand
        complex_weights: Boltzmann weights for complex
        temperature: Temperature in Kelvin

    Returns:
        Dictionary with entropy change and free energy contribution
    """
    calculator = ConformationalEntropyCalculator(temperature=temperature)

    # Compute entropies
    S_ligand = calculator.compute_from_conformer_features(ligand_features, ligand_weights)
    S_complex = calculator.compute_from_conformer_features(complex_features, complex_weights)

    # Entropy change upon binding
    delta_S = S_complex["entropy"] - S_ligand["entropy"]

    # Free energy contribution: -TΔS
    delta_G_entropy = -calculator.KB * temperature * delta_S

    return {
        "S_ligand": S_ligand["entropy"],
        "S_complex": S_complex["entropy"],
        "delta_S": delta_S,
        "delta_G_entropy": delta_G_entropy,
        "temperature": temperature,
        "entropy_loss": delta_S < 0,  # Typical for binding
    }
