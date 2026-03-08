"""
Enhanced 3D conformer features for molecular property prediction.

This module implements additional 3D descriptors that capture molecular shape,
surface properties, and conformer-specific information beyond basic geometric
features (distances, angles, torsions).

All features in this module are CONFORMER-VARYING, meaning they change
depending on the 3D geometry of the molecule.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    from rdkit.Chem import rdFreeSASA
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None


# Van der Waals radii for common elements (Angstroms)
VDW_RADII = {
    1: 1.20,   # H
    6: 1.70,   # C
    7: 1.55,   # N
    8: 1.52,   # O
    9: 1.47,   # F
    15: 1.80,  # P
    16: 1.80,  # S
    17: 1.75,  # Cl
    35: 1.85,  # Br
    53: 1.98,  # I
}


@dataclass
class Enhanced3DFeatures:
    """
    Container for enhanced 3D conformer features.

    All features vary across conformers of the same molecule.

    Attributes:
        pmi_ratios: Principal moments of inertia ratios [I1/I3, I2/I3] (2,)
        pmi_raw: Raw principal moments [I1, I2, I3] (3,)
        radius_of_gyration: Radius of gyration (1,)
        asphericity: Shape asphericity (1,)
        eccentricity: Shape eccentricity (1,)
        inertial_shape_factor: ISF descriptor (1,)
        molecular_volume: Approximate molecular volume (1,)
        sasa: Solvent accessible surface area (1,)
        sasa_per_atom: Per-atom SASA contributions (n_atoms,)
        usr_descriptors: USR shape fingerprint (12,)
        span: Maximum interatomic distance (1,)
        compactness: Volume / span^3 ratio (1,)
        center_of_mass: Center of mass coordinates (3,)
        extent: Bounding box dimensions (3,)
    """
    pmi_ratios: np.ndarray  # (2,)
    pmi_raw: np.ndarray  # (3,)
    radius_of_gyration: float
    asphericity: float
    eccentricity: float
    inertial_shape_factor: float
    molecular_volume: float
    sasa: float
    sasa_per_atom: np.ndarray  # (n_atoms,)
    usr_descriptors: np.ndarray  # (12,)
    span: float
    compactness: float
    center_of_mass: np.ndarray  # (3,)
    extent: np.ndarray  # (3,)
    n_atoms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_flat_vector(self) -> np.ndarray:
        """Flatten to feature vector (excludes per-atom SASA for fixed dimension)."""
        return np.concatenate([
            self.pmi_ratios,  # 2
            self.pmi_raw,  # 3
            [self.radius_of_gyration],  # 1
            [self.asphericity],  # 1
            [self.eccentricity],  # 1
            [self.inertial_shape_factor],  # 1
            [self.molecular_volume],  # 1
            [self.sasa],  # 1
            self.usr_descriptors,  # 12
            [self.span],  # 1
            [self.compactness],  # 1
            # center_of_mass excluded (frame-dependent)
            self.extent,  # 3
        ])  # Total: 28 features

    @staticmethod
    def feature_dim() -> int:
        """Return fixed feature dimension."""
        return 28


class Enhanced3DFeatureExtractor:
    """
    Extract enhanced 3D features from molecular conformers.

    These features capture shape, surface, and global geometry properties
    that vary across conformers.
    """

    def __init__(
        self,
        probe_radius: float = 1.4,  # Water probe radius for SASA
        use_rdkit_sasa: bool = True,  # Use RDKit SASA if available
    ):
        self.probe_radius = probe_radius
        self.use_rdkit_sasa = use_rdkit_sasa

    def extract(
        self,
        mol: 'Chem.Mol',
        conformer_id: int = 0,
    ) -> Enhanced3DFeatures:
        """
        Extract enhanced 3D features from a conformer.

        Args:
            mol: RDKit molecule with conformer(s)
            conformer_id: Which conformer to use

        Returns:
            Enhanced3DFeatures object
        """
        conf = mol.GetConformer(conformer_id)
        coords = conf.GetPositions()
        n_atoms = mol.GetNumAtoms()

        # Get atomic masses for center of mass / inertia
        masses = np.array([mol.GetAtomWithIdx(i).GetMass() for i in range(n_atoms)])

        # Get atomic numbers for VDW radii
        atomic_nums = np.array([mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(n_atoms)])
        radii = np.array([VDW_RADII.get(z, 1.70) for z in atomic_nums])

        # Compute all features
        com = self._center_of_mass(coords, masses)
        pmi_raw, pmi_ratios = self._principal_moments(coords, masses, com)
        rg = self._radius_of_gyration(coords, masses, com)
        asphericity, eccentricity, isf = self._shape_descriptors(pmi_raw)
        volume = self._molecular_volume(coords, radii)
        sasa, sasa_per_atom = self._compute_sasa(mol, conformer_id, coords, radii)
        usr = self._usr_descriptors(coords, com)
        span = self._molecular_span(coords)
        compactness = volume / (span ** 3 + 1e-10)
        extent = self._bounding_box(coords)

        return Enhanced3DFeatures(
            pmi_ratios=pmi_ratios,
            pmi_raw=pmi_raw,
            radius_of_gyration=rg,
            asphericity=asphericity,
            eccentricity=eccentricity,
            inertial_shape_factor=isf,
            molecular_volume=volume,
            sasa=sasa,
            sasa_per_atom=sasa_per_atom,
            usr_descriptors=usr,
            span=span,
            compactness=compactness,
            center_of_mass=com,
            extent=extent,
            n_atoms=n_atoms,
        )

    def extract_batch(
        self,
        mol: 'Chem.Mol',
        conformer_ids: List[int],
    ) -> List[Enhanced3DFeatures]:
        """Extract features from multiple conformers."""
        return [self.extract(mol, cid) for cid in conformer_ids]

    def _center_of_mass(
        self,
        coords: np.ndarray,
        masses: np.ndarray,
    ) -> np.ndarray:
        """Compute center of mass."""
        total_mass = masses.sum()
        return (coords * masses[:, np.newaxis]).sum(axis=0) / total_mass

    def _principal_moments(
        self,
        coords: np.ndarray,
        masses: np.ndarray,
        com: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute principal moments of inertia.

        Returns:
            pmi_raw: [I1, I2, I3] sorted ascending
            pmi_ratios: [I1/I3, I2/I3] normalized ratios
        """
        # Center coordinates
        centered = coords - com

        # Build inertia tensor
        inertia_tensor = np.zeros((3, 3))
        for i, (r, m) in enumerate(zip(centered, masses)):
            inertia_tensor += m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))

        # Eigenvalues are principal moments
        eigenvalues = np.linalg.eigvalsh(inertia_tensor)
        pmi_raw = np.sort(eigenvalues)  # I1 <= I2 <= I3

        # Ratios (NPR1, NPR2 normalized principal ratios)
        I3 = pmi_raw[2] + 1e-10  # Avoid division by zero
        pmi_ratios = np.array([pmi_raw[0] / I3, pmi_raw[1] / I3])

        return pmi_raw, pmi_ratios

    def _radius_of_gyration(
        self,
        coords: np.ndarray,
        masses: np.ndarray,
        com: np.ndarray,
    ) -> float:
        """Compute radius of gyration."""
        centered = coords - com
        r_squared = np.sum(centered ** 2, axis=1)
        total_mass = masses.sum()
        return np.sqrt(np.sum(masses * r_squared) / total_mass)

    def _shape_descriptors(
        self,
        pmi_raw: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Compute shape descriptors from principal moments.

        Returns:
            asphericity: (I3 - 0.5*(I1+I2)) / I3
            eccentricity: sqrt(1 - I1/I3)
            inertial_shape_factor: I2 / (I1 * I3)
        """
        I1, I2, I3 = pmi_raw

        # Asphericity: deviation from sphere
        asphericity = (I3 - 0.5 * (I1 + I2)) / (I3 + 1e-10)

        # Eccentricity: elongation
        eccentricity = np.sqrt(1 - I1 / (I3 + 1e-10))

        # Inertial shape factor
        isf = I2 / (I1 * I3 + 1e-10)

        return asphericity, eccentricity, isf

    def _molecular_volume(
        self,
        coords: np.ndarray,
        radii: np.ndarray,
    ) -> float:
        """
        Approximate molecular volume using sum of atomic spheres
        with overlap correction.
        """
        # Simple approximation: sum of sphere volumes with 50% overlap correction
        volumes = (4/3) * np.pi * radii ** 3

        # Overlap correction based on average interatomic distances
        if len(coords) > 1:
            # Compute pairwise distances
            diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))

            # Estimate overlap
            r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
            overlap_fraction = np.clip(1 - dists / r_sum, 0, 1)
            np.fill_diagonal(overlap_fraction, 0)

            # Reduce volume by estimated overlap
            overlap_correction = 0.5 * np.sum(overlap_fraction) / len(coords)
            total_volume = volumes.sum() * (1 - 0.3 * overlap_correction)
        else:
            total_volume = volumes.sum()

        return total_volume

    def _compute_sasa(
        self,
        mol: 'Chem.Mol',
        conformer_id: int,
        coords: np.ndarray,
        radii: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute solvent accessible surface area.

        Uses RDKit FreeSASA if available, otherwise falls back to
        simple sphere-based approximation.
        """
        if self.use_rdkit_sasa and RDKIT_AVAILABLE:
            try:
                # Use RDKit's FreeSASA implementation
                radii_list = rdFreeSASA.classifyAtoms(mol)
                sasa_opts = rdFreeSASA.SASAOpts()
                sasa_opts.probeRadius = self.probe_radius
                sasa_per_atom = rdFreeSASA.CalcSASA(mol, radii_list, confIdx=conformer_id, opts=sasa_opts)
                total_sasa = sum(sasa_per_atom)
                return total_sasa, np.array(sasa_per_atom)
            except Exception:
                pass  # Fall back to approximation

        # Simple approximation: exposed surface based on coordination
        n_atoms = len(coords)
        sasa_per_atom = np.zeros(n_atoms)

        for i in range(n_atoms):
            # Count nearby atoms
            dists = np.sqrt(np.sum((coords - coords[i]) ** 2, axis=1))
            # Atoms within 2*VDW radius are blocking
            blocking = np.sum((dists < 2 * radii[i]) & (dists > 0))
            # Exposed fraction decreases with more blocking atoms
            exposed = max(0, 1 - blocking * 0.1)
            # Surface area of sphere * exposed fraction
            sasa_per_atom[i] = 4 * np.pi * (radii[i] + self.probe_radius) ** 2 * exposed

        return sasa_per_atom.sum(), sasa_per_atom

    def _usr_descriptors(
        self,
        coords: np.ndarray,
        com: np.ndarray,
    ) -> np.ndarray:
        """
        Compute USR (Ultrafast Shape Recognition) descriptors.

        USR uses distances from 4 reference points:
        - ctd: molecular centroid (center of mass)
        - cst: closest atom to ctd
        - fct: farthest atom from ctd
        - ftf: farthest atom from fct

        For each reference, compute mean, std, skewness of distances.
        Total: 4 references * 3 moments = 12 features
        """
        n_atoms = len(coords)

        # Reference points
        ctd = com  # Centroid

        # Distances from centroid
        d_ctd = np.sqrt(np.sum((coords - ctd) ** 2, axis=1))

        # Closest to centroid
        cst_idx = np.argmin(d_ctd)
        cst = coords[cst_idx]
        d_cst = np.sqrt(np.sum((coords - cst) ** 2, axis=1))

        # Farthest from centroid
        fct_idx = np.argmax(d_ctd)
        fct = coords[fct_idx]
        d_fct = np.sqrt(np.sum((coords - fct) ** 2, axis=1))

        # Farthest from fct
        ftf_idx = np.argmax(d_fct)
        ftf = coords[ftf_idx]
        d_ftf = np.sqrt(np.sum((coords - ftf) ** 2, axis=1))

        # Compute moments for each distance distribution
        usr = []
        for d in [d_ctd, d_cst, d_fct, d_ftf]:
            mean = np.mean(d)
            std = np.std(d) + 1e-10
            skew = np.mean(((d - mean) / std) ** 3) if std > 1e-8 else 0
            usr.extend([mean, std, skew])

        return np.array(usr)

    def _molecular_span(self, coords: np.ndarray) -> float:
        """Compute maximum interatomic distance (molecular span)."""
        if len(coords) < 2:
            return 0.0

        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        return np.max(dists)

    def _bounding_box(self, coords: np.ndarray) -> np.ndarray:
        """Compute bounding box dimensions."""
        if len(coords) == 0:
            return np.zeros(3)
        return coords.max(axis=0) - coords.min(axis=0)


def compute_feature_variance(
    mol: 'Chem.Mol',
    conformer_ids: List[int],
) -> Dict[str, float]:
    """
    Compute variance of each feature type across conformers.

    Useful for validating that features actually vary.
    """
    extractor = Enhanced3DFeatureExtractor()
    features_list = extractor.extract_batch(mol, conformer_ids)

    # Stack features
    stacked = np.stack([f.to_flat_vector() for f in features_list])

    # Compute variance per feature
    variances = np.var(stacked, axis=0)

    # Aggregate by feature type
    return {
        'pmi_ratios': np.mean(variances[0:2]),
        'pmi_raw': np.mean(variances[2:5]),
        'radius_of_gyration': variances[5],
        'asphericity': variances[6],
        'eccentricity': variances[7],
        'inertial_shape_factor': variances[8],
        'molecular_volume': variances[9],
        'sasa': variances[10],
        'usr_descriptors': np.mean(variances[11:23]),
        'span': variances[23],
        'compactness': variances[24],
        'extent': np.mean(variances[25:28]),
        'total_variance': np.sum(variances),
        'mean_variance': np.mean(variances),
    }


def validate_feature_variation(
    mol: 'Chem.Mol',
    conformer_ids: List[int],
    threshold: float = 0.01,
) -> Dict[str, bool]:
    """
    Check which features have meaningful variation across conformers.

    Returns dict mapping feature names to whether they vary.
    """
    variances = compute_feature_variance(mol, conformer_ids)
    return {
        name: var > threshold
        for name, var in variances.items()
        if name not in ['total_variance', 'mean_variance']
    }
