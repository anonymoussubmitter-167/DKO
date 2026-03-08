"""
Conformer generation utilities.

This module provides functionality for generating 3D conformer ensembles
from SMILES strings using RDKit's ETKDG algorithm (Riniker & Landrum, 2015).

Features:
- ETKDG algorithm for distance geometry conformer generation
- MMFF94 force field for energy minimization
- RMSD-based pruning with configurable threshold
- Energy window filtering
- Boltzmann weighting at specified temperature
- Disk caching for efficiency
- Parallel processing support
"""

from typing import List, Optional, Tuple, Union, Dict, Any
import numpy as np
import torch
from dataclasses import dataclass, field
from pathlib import Path
import pickle
import hashlib
import logging

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolAlign
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None
    AllChem = None
    rdMolAlign = None
    print("Warning: RDKit not available. Conformer generation will be limited.")


def check_rdkit():
    """Check if RDKit is available."""
    if not RDKIT_AVAILABLE:
        raise ImportError(
            "RDKit is required for conformer generation. "
            "Install with: conda install -c conda-forge rdkit"
        )


@dataclass
class ConformerEnsemble:
    """
    Container for conformer ensemble with metadata.

    This is the primary data structure for representing molecular conformers
    in the DKO pipeline.

    Attributes:
        mol: RDKit Mol object with all conformers embedded
        conformer_ids: List of conformer IDs in the molecule
        energies: Array of energies in kcal/mol, shape (n_conformers,)
        boltzmann_weights: Normalized Boltzmann weights, shape (n_conformers,)
        n_conformers: Number of conformers
        generation_successful: Whether generation succeeded normally
        smiles: Original SMILES string (for caching/logging)
        metadata: Additional metadata dictionary
    """
    mol: Any  # Chem.Mol - use Any for type hint to avoid import issues
    conformer_ids: List[int]
    energies: np.ndarray
    boltzmann_weights: np.ndarray
    n_conformers: int
    generation_successful: bool
    smiles: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_conformer(self, idx: int):
        """Get conformer by index."""
        if idx >= self.n_conformers:
            raise IndexError(f"Conformer index {idx} out of range (n={self.n_conformers})")
        return self.mol.GetConformer(self.conformer_ids[idx])

    def get_coordinates(self, idx: int) -> np.ndarray:
        """Get coordinates for conformer at index, shape (n_atoms, 3)."""
        conf = self.get_conformer(idx)
        return conf.GetPositions()

    def get_all_coordinates(self) -> np.ndarray:
        """Get all conformer coordinates, shape (n_conformers, n_atoms, 3)."""
        coords = []
        for i in range(self.n_conformers):
            coords.append(self.get_coordinates(i))
        return np.stack(coords, axis=0)

    def get_lowest_energy_conformer(self):
        """Get the lowest energy conformer."""
        min_idx = np.argmin(self.energies)
        return self.get_conformer(min_idx)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'smiles': self.smiles,
            'n_conformers': self.n_conformers,
            'energies': self.energies.tolist(),
            'boltzmann_weights': self.boltzmann_weights.tolist(),
            'generation_successful': self.generation_successful,
            'metadata': self.metadata,
        }


class ConformerGenerator:
    """
    Generate and manage molecular conformer ensembles.

    Uses the ETKDG algorithm (Riniker & Landrum, 2015) for conformer generation
    with MMFF94 force field optimization and Boltzmann weighting.

    Specifications:
    - ETKDG algorithm for generation
    - MMFF94 for energy minimization
    - Boltzmann weighting at 300K
    - RMSD-based pruning for diversity

    Example:
        >>> generator = ConformerGenerator(max_conformers=50)
        >>> ensemble = generator.generate_from_smiles("CCO")
        >>> print(f"Generated {ensemble.n_conformers} conformers")
    """

    # Physical constants
    KB = 1.987204e-3  # Boltzmann constant in kcal/(mol*K)

    def __init__(
        self,
        max_conformers: int = 50,
        rmsd_threshold: float = 0.5,
        energy_window: float = 15.0,
        force_field: str = "MMFF94",
        temperature: float = 300.0,
        random_seed: int = 42,
        cache_dir: Optional[Union[str, Path]] = None,
        n_jobs: int = 1,
        use_torsion_preferences: bool = True,
        method: str = "ETKDGv3",
        enforce_chirality: bool = True,
    ):
        """
        Initialize conformer generator.

        Args:
            max_conformers: Maximum number of conformers to generate
            rmsd_threshold: RMSD threshold for pruning (Angstroms)
            energy_window: Energy window for pruning (kcal/mol)
            force_field: Force field for optimization ('MMFF94' or 'UFF')
            temperature: Temperature for Boltzmann weighting (Kelvin)
            random_seed: Random seed for reproducibility
            cache_dir: Directory for caching conformers
            n_jobs: Number of parallel jobs
            use_torsion_preferences: Use torsion angle preferences in ETKDG
            method: ETKDG variant ('ETKDG', 'ETKDGv2', 'ETKDGv3')
            enforce_chirality: Whether to enforce stereochemistry
        """
        check_rdkit()

        self.max_conformers = max_conformers
        self.rmsd_threshold = rmsd_threshold
        self.energy_window = energy_window
        self.force_field = force_field
        self.temperature = temperature
        self.random_seed = random_seed
        self.n_jobs = n_jobs
        self.use_torsion_preferences = use_torsion_preferences
        self.method = method
        self.enforce_chirality = enforce_chirality

        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)

    def _get_cache_key(self, smiles: str) -> str:
        """Generate cache key from SMILES."""
        # Include parameters in hash for cache invalidation
        params_str = f"{self.max_conformers}_{self.rmsd_threshold}_{self.energy_window}"
        key_str = f"{smiles}_{params_str}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_path(self, smiles: str) -> Optional[Path]:
        """Get cache file path for a SMILES."""
        if self.cache_dir is None:
            return None
        key = self._get_cache_key(smiles)
        return self.cache_dir / f"{key}.pkl"

    def _load_from_cache(self, smiles: str) -> Optional[ConformerEnsemble]:
        """Load ensemble from cache if available."""
        cache_path = self._get_cache_path(smiles)
        if cache_path and cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache for {smiles}: {e}")
        return None

    def _save_to_cache(self, ensemble: ConformerEnsemble):
        """Save ensemble to cache."""
        if ensemble.smiles and self.cache_dir:
            cache_path = self._get_cache_path(ensemble.smiles)
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(ensemble, f)
            except Exception as e:
                self.logger.warning(f"Failed to save cache: {e}")

    def _get_embed_params(self):
        """Get ETKDG embedding parameters."""
        if self.method == "ETKDGv3":
            params = AllChem.ETKDGv3()
        elif self.method == "ETKDGv2":
            params = AllChem.ETKDGv2()
        else:
            params = AllChem.ETKDG()

        params.randomSeed = self.random_seed
        params.numThreads = self.n_jobs
        params.pruneRmsThresh = -1  # Don't prune during generation
        params.enforceChirality = self.enforce_chirality
        params.useExpTorsionAnglePrefs = self.use_torsion_preferences
        params.useBasicKnowledge = True

        return params

    def generate_from_smiles(self, smiles: str) -> ConformerEnsemble:
        """
        Generate conformer ensemble from SMILES string.

        Args:
            smiles: SMILES string

        Returns:
            ConformerEnsemble object
        """
        # Handle empty or invalid SMILES
        if not smiles or not smiles.strip():
            return ConformerEnsemble(
                mol=None,
                conformer_ids=[],
                energies=np.array([]),
                boltzmann_weights=np.array([]),
                n_conformers=0,
                generation_successful=False,
                smiles=smiles,
                metadata={'error': 'Empty SMILES'}
            )

        # Check cache first
        cached = self._load_from_cache(smiles)
        if cached is not None:
            return cached

        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ConformerEnsemble(
                mol=None,
                conformer_ids=[],
                energies=np.array([]),
                boltzmann_weights=np.array([]),
                n_conformers=0,
                generation_successful=False,
                smiles=smiles,
                metadata={'error': f'Invalid SMILES: {smiles}'}
            )

        ensemble = self.generate(mol, smiles)

        # Cache result
        self._save_to_cache(ensemble)

        return ensemble

    def generate(self, mol: Chem.Mol, smiles: Optional[str] = None) -> ConformerEnsemble:
        """
        Generate conformer ensemble for a molecule.

        Args:
            mol: RDKit molecule object
            smiles: SMILES string (for caching/metadata)

        Returns:
            ConformerEnsemble object
        """
        try:
            # Add hydrogens
            mol = Chem.AddHs(mol)

            # Generate conformers using ETKDG
            conformer_ids = self._generate_conformers_etkdg(mol)

            if len(conformer_ids) == 0:
                # Fallback to single conformer
                self.logger.warning(f"ETKDG failed for {smiles}, using fallback")
                return self._generate_fallback_conformer(mol, smiles)

            # Minimize energies with force field
            energies = self._minimize_energies(mol, conformer_ids)

            # Prune conformers by RMSD and energy
            kept_ids, kept_energies = self._prune_conformers(
                mol, conformer_ids, energies
            )

            # Compute Boltzmann weights
            boltzmann_weights = self._compute_boltzmann_weights(kept_energies)

            ensemble = ConformerEnsemble(
                mol=mol,
                conformer_ids=kept_ids,
                energies=kept_energies,
                boltzmann_weights=boltzmann_weights,
                n_conformers=len(kept_ids),
                generation_successful=True,
                smiles=smiles,
                metadata={
                    'initial_conformers': len(conformer_ids),
                    'pruned_conformers': len(kept_ids),
                    'force_field': self.force_field,
                    'temperature': self.temperature,
                }
            )

            return ensemble

        except Exception as e:
            self.logger.error(f"Conformer generation failed for {smiles}: {e}")
            return self._generate_fallback_conformer(mol, smiles)

    def _generate_conformers_etkdg(self, mol: Chem.Mol) -> List[int]:
        """Generate conformers using ETKDG algorithm."""
        params = self._get_embed_params()

        # Generate more conformers than needed to allow for pruning
        n_attempts = min(self.max_conformers * 3, 500)

        conformer_ids = AllChem.EmbedMultipleConfs(
            mol,
            numConfs=n_attempts,
            params=params
        )

        if len(conformer_ids) == 0:
            # Fallback: try with random coordinates
            params.useRandomCoords = True
            conformer_ids = AllChem.EmbedMultipleConfs(
                mol,
                numConfs=self.max_conformers,
                params=params
            )

        return list(conformer_ids)

    def _minimize_energies(
        self, mol: Chem.Mol, conformer_ids: List[int]
    ) -> np.ndarray:
        """Minimize conformer energies using force field."""
        energies = []

        if self.force_field == "MMFF94":
            # Get MMFF properties once
            props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")

            if props is None:
                # Fall back to UFF if MMFF fails
                self.logger.warning("MMFF94 parameterization failed, using UFF")
                return self._minimize_with_uff(mol, conformer_ids)

            for cid in conformer_ids:
                ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
                if ff is None:
                    energies.append(np.inf)
                    continue

                # Minimize
                converged = ff.Minimize(maxIts=500)
                energy = ff.CalcEnergy()
                energies.append(energy)
        else:
            # Use UFF
            energies = self._minimize_with_uff(mol, conformer_ids)

        return np.array(energies)

    def _minimize_with_uff(
        self, mol: Chem.Mol, conformer_ids: List[int]
    ) -> np.ndarray:
        """Minimize with UFF force field."""
        energies = []
        for cid in conformer_ids:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
            if ff is None:
                energies.append(np.inf)
                continue

            ff.Minimize(maxIts=500)
            energy = ff.CalcEnergy()
            energies.append(energy)

        return np.array(energies)

    def _prune_conformers(
        self,
        mol: Chem.Mol,
        conformer_ids: List[int],
        energies: np.ndarray
    ) -> Tuple[List[int], np.ndarray]:
        """
        Prune conformers based on RMSD and energy window.

        Strategy:
        1. Sort by energy
        2. Keep lowest energy conformer
        3. For each remaining, keep if RMSD > threshold from all kept
        4. Stop at energy window or max_conformers
        """
        # Remove failed conformers
        valid_mask = energies < np.inf
        valid_ids = [cid for cid, valid in zip(conformer_ids, valid_mask) if valid]
        valid_energies = energies[valid_mask]

        if len(valid_ids) == 0:
            return [], np.array([])

        # Sort by energy
        sorted_indices = np.argsort(valid_energies)
        sorted_ids = [valid_ids[i] for i in sorted_indices]
        sorted_energies = valid_energies[sorted_indices]

        # Energy window relative to minimum
        min_energy = sorted_energies[0]
        in_window = sorted_energies <= (min_energy + self.energy_window)

        kept_ids = [sorted_ids[0]]
        kept_energies = [sorted_energies[0]]

        # RMSD-based pruning
        for i in range(1, len(sorted_ids)):
            if not in_window[i]:
                break

            if len(kept_ids) >= self.max_conformers:
                break

            cid = sorted_ids[i]

            # Check RMSD against all kept conformers
            is_diverse = True
            for kept_cid in kept_ids:
                try:
                    rmsd = rdMolAlign.GetBestRMS(mol, mol, kept_cid, cid)
                    if rmsd < self.rmsd_threshold:
                        is_diverse = False
                        break
                except Exception:
                    # If RMSD calculation fails, keep the conformer
                    pass

            if is_diverse:
                kept_ids.append(cid)
                kept_energies.append(sorted_energies[i])

        return kept_ids, np.array(kept_energies)

    def _compute_boltzmann_weights(self, energies: np.ndarray) -> np.ndarray:
        """
        Compute Boltzmann weights from energies.

        w_i = exp(-E_i / k_B*T) / Z
        where Z = sum_j exp(-E_j / k_B*T)
        """
        if len(energies) == 0:
            return np.array([])

        if len(energies) == 1:
            return np.array([1.0])

        # Shift energies to avoid numerical issues
        energies_shifted = energies - energies.min()

        # Compute unnormalized weights
        beta = 1.0 / (self.KB * self.temperature)
        unnormalized = np.exp(-beta * energies_shifted)

        # Normalize
        weights = unnormalized / unnormalized.sum()

        return weights

    def _generate_fallback_conformer(
        self, mol: Chem.Mol, smiles: Optional[str] = None
    ) -> ConformerEnsemble:
        """Generate single fallback conformer when ETKDG fails."""
        try:
            mol = Chem.AddHs(mol)

            # Try simple embedding
            success = AllChem.EmbedMolecule(mol, randomSeed=self.random_seed)

            if success == -1:
                # Last resort: use random coordinates
                AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=self.random_seed)

            # Try to minimize
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except Exception:
                try:
                    AllChem.UFFOptimizeMolecule(mol)
                except Exception:
                    pass

            return ConformerEnsemble(
                mol=mol,
                conformer_ids=[0],
                energies=np.array([0.0]),
                boltzmann_weights=np.array([1.0]),
                n_conformers=1,
                generation_successful=False,
                smiles=smiles,
                metadata={'fallback': True}
            )
        except Exception as e:
            raise ValueError(f"Complete conformer generation failure for {smiles}: {e}")

    def generate_batch(
        self,
        smiles_list: List[str],
        show_progress: bool = True,
    ) -> List[ConformerEnsemble]:
        """
        Generate conformers for batch of molecules with progress bar.

        Args:
            smiles_list: List of SMILES strings
            show_progress: Whether to show progress bar

        Returns:
            List of ConformerEnsemble objects
        """
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(smiles_list, desc="Generating conformers")
        else:
            iterator = smiles_list

        ensembles = []
        for smiles in iterator:
            try:
                ensemble = self.generate_from_smiles(smiles)
                ensembles.append(ensemble)
            except Exception as e:
                self.logger.error(f"Failed to generate conformers for {smiles}: {e}")
                # Create minimal fallback
                ensembles.append(None)

        return ensembles

    def generate_ensemble(
        self,
        smiles: str,
        n_conformers: Optional[int] = None,
    ) -> ConformerEnsemble:
        """
        Generate conformer ensemble for a molecule from SMILES.

        This is an alias for generate_from_smiles that accepts an
        optional n_conformers parameter.

        Args:
            smiles: SMILES string
            n_conformers: Number of conformers to generate (optional,
                         uses max_conformers from initialization if not specified)

        Returns:
            ConformerEnsemble object
        """
        # Temporarily adjust max_conformers if specified
        original_max = self.max_conformers
        if n_conformers is not None:
            self.max_conformers = n_conformers

        try:
            ensemble = self.generate_from_smiles(smiles)
        finally:
            # Restore original setting
            self.max_conformers = original_max

        return ensemble


def compute_boltzmann_weights(
    energies: Union[List[float], np.ndarray, torch.Tensor],
    temperature: float = 300.0,
) -> torch.Tensor:
    """
    Compute Boltzmann weights from conformer energies.

    w_i = exp(-E_i / k_B*T) / Z

    Args:
        energies: Conformer energies in kcal/mol
        temperature: Temperature in Kelvin

    Returns:
        Normalized Boltzmann weights as torch.Tensor
    """
    # Boltzmann constant in kcal/(mol*K)
    kB = 1.987204e-3

    if isinstance(energies, list):
        energies = torch.tensor(energies, dtype=torch.float32)
    elif isinstance(energies, np.ndarray):
        energies = torch.from_numpy(energies).float()

    if len(energies) == 0:
        return torch.tensor([])

    if len(energies) == 1:
        return torch.tensor([1.0])

    # Shift to prevent overflow
    energies = energies - energies.min()

    # Compute Boltzmann factors
    beta = 1.0 / (kB * temperature)
    boltzmann = torch.exp(-beta * energies)

    # Normalize
    weights = boltzmann / boltzmann.sum()

    return weights


def get_conformer_coordinates(mol, conf_id: int = 0) -> np.ndarray:
    """
    Extract 3D coordinates from an RDKit conformer.

    Args:
        mol: RDKit Mol object with conformer
        conf_id: Conformer ID

    Returns:
        Coordinates array of shape (n_atoms, 3)
    """
    check_rdkit()

    conf = mol.GetConformer(conf_id)
    return conf.GetPositions()


def compute_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """
    Compute RMSD between two coordinate sets.

    Args:
        coords1: First coordinates (n_atoms, 3)
        coords2: Second coordinates (n_atoms, 3)

    Returns:
        RMSD value in Angstroms
    """
    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))


def compute_aligned_rmsd(mol, conf_id1: int, conf_id2: int) -> float:
    """
    Compute RMSD between two conformers after optimal alignment.

    Args:
        mol: RDKit Mol object with conformers
        conf_id1: First conformer ID
        conf_id2: Second conformer ID

    Returns:
        RMSD after alignment
    """
    check_rdkit()
    return rdMolAlign.GetBestRMS(mol, mol, conf_id1, conf_id2)


def filter_conformers_by_rmsd(
    mol,
    conformer_ids: List[int],
    threshold: float = 0.5,
) -> List[int]:
    """
    Filter conformers to ensure diversity based on RMSD.

    Args:
        mol: RDKit Mol object with conformers
        conformer_ids: List of conformer IDs to filter
        threshold: RMSD threshold for filtering (Angstroms)

    Returns:
        Filtered list of diverse conformer IDs
    """
    check_rdkit()

    if len(conformer_ids) <= 1:
        return conformer_ids

    kept_ids = [conformer_ids[0]]

    for cid in conformer_ids[1:]:
        is_diverse = True
        for kept_cid in kept_ids:
            try:
                rmsd = rdMolAlign.GetBestRMS(mol, mol, kept_cid, cid)
                if rmsd < threshold:
                    is_diverse = False
                    break
            except Exception:
                pass

        if is_diverse:
            kept_ids.append(cid)

    return kept_ids


def align_conformers(mol, reference_id: int = 0):
    """
    Align all conformers to a reference conformer.

    Args:
        mol: RDKit Mol object with conformers
        reference_id: Conformer ID to use as reference (default 0)

    Returns:
        Molecule with aligned conformers
    """
    check_rdkit()

    # Get list of conformer IDs
    conf_ids = [conf.GetId() for conf in mol.GetConformers()]

    if len(conf_ids) <= 1:
        return mol

    # Align all conformers to reference
    for cid in conf_ids:
        if cid != reference_id:
            try:
                rdMolAlign.AlignMol(mol, mol, prbCid=cid, refCid=reference_id)
            except Exception:
                # If alignment fails, skip this conformer
                pass

    return mol


def generate_conformers(
    smiles: str,
    n_conformers: int = 50,
    **kwargs,
) -> ConformerEnsemble:
    """
    Convenience function for conformer generation.

    Args:
        smiles: SMILES string
        n_conformers: Maximum number of conformers
        **kwargs: Additional arguments for ConformerGenerator

    Returns:
        ConformerEnsemble object
    """
    generator = ConformerGenerator(max_conformers=n_conformers, **kwargs)
    return generator.generate_from_smiles(smiles)
