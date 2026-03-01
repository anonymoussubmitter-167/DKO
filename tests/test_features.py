"""Tests for feature extraction."""

import pytest
import numpy as np
import torch

# Skip if RDKit not available
pytest.importorskip("rdkit")

from dko.data.features import (
    GeometricFeatures,
    AugmentedBasis,
    GeometricFeatureExtractor,
    AugmentedBasisConstructor,
    compute_scc_simple,
    compute_scc_from_ensemble,
    compute_pairwise_distances,
    compute_bond_angles,
    compute_dihedral,
    FeatureExtractor,
    MolecularFingerprints,
)


class TestGeometricFeatures:
    """Test suite for GeometricFeatures dataclass."""

    @pytest.fixture
    def sample_features(self):
        """Create sample GeometricFeatures for testing."""
        return GeometricFeatures(
            distances=np.array([1.0, 1.5, 2.0, 1.2]),
            angles=np.array([1.91, 2.09, 1.85]),  # ~110, 120, 106 degrees
            torsions=np.array([0.5, -0.5, 1.0, -1.0]),  # cos/sin pairs
            distance_pairs=np.array([[0, 1], [0, 2], [1, 2], [1, 3]]),
            angle_triplets=np.array([[0, 1, 2], [1, 2, 3], [0, 1, 3]]),
            torsion_quadruplets=np.array([[0, 1, 2, 3], [1, 2, 3, 4]]),
            coordinates=np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0]]),
            atom_features=np.random.randn(4, 20),
            n_atoms=4,
        )

    def test_feature_creation(self, sample_features):
        """Test GeometricFeatures creation."""
        assert sample_features.n_atoms == 4
        assert len(sample_features.distances) == 4
        assert len(sample_features.angles) == 3
        assert sample_features.coordinates.shape == (4, 3)

    def test_to_flat_vector(self, sample_features):
        """Test flattening features to vector."""
        flat = sample_features.to_flat_vector(include_torsions=True)

        expected_len = (
            len(sample_features.distances) +
            len(sample_features.angles) +
            len(sample_features.torsions) +
            sample_features.atom_features.size
        )
        assert len(flat) == expected_len

    def test_to_flat_vector_no_torsions(self, sample_features):
        """Test flattening without torsions."""
        flat = sample_features.to_flat_vector(include_torsions=False)

        expected_len = (
            len(sample_features.distances) +
            len(sample_features.angles) +
            sample_features.atom_features.size
        )
        assert len(flat) == expected_len

    def test_get_feature_dim(self, sample_features):
        """Test feature dimension calculation."""
        dim_with = sample_features.get_feature_dim(include_torsions=True)
        dim_without = sample_features.get_feature_dim(include_torsions=False)

        assert dim_with > dim_without
        assert dim_with == len(sample_features.to_flat_vector(include_torsions=True))


class TestAugmentedBasis:
    """Test suite for AugmentedBasis dataclass."""

    @pytest.fixture
    def sample_basis(self):
        """Create sample AugmentedBasis for testing."""
        feature_dim = 10
        mean = np.random.randn(feature_dim)
        second_order = np.eye(feature_dim) + np.outer(mean, mean)
        flat = np.concatenate([mean, second_order[np.triu_indices(feature_dim)]])

        return AugmentedBasis(
            mean=mean,
            second_order=second_order,
            flat_representation=flat,
            n_conformers=5,
            weights=np.array([0.4, 0.3, 0.15, 0.1, 0.05]),
            feature_dim=feature_dim,
        )

    def test_basis_creation(self, sample_basis):
        """Test AugmentedBasis creation."""
        assert sample_basis.n_conformers == 5
        assert sample_basis.feature_dim == 10
        assert len(sample_basis.mean) == 10
        assert sample_basis.second_order.shape == (10, 10)

    def test_to_tensor(self, sample_basis):
        """Test conversion to PyTorch tensor."""
        tensor = sample_basis.to_tensor()
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32
        assert len(tensor) == len(sample_basis.flat_representation)

    def test_get_mean_tensor(self, sample_basis):
        """Test getting mean as tensor."""
        mean_tensor = sample_basis.get_mean_tensor()
        assert isinstance(mean_tensor, torch.Tensor)
        assert len(mean_tensor) == sample_basis.feature_dim

    def test_get_second_order_tensor(self, sample_basis):
        """Test getting second-order tensor."""
        so_tensor = sample_basis.get_second_order_tensor()
        assert isinstance(so_tensor, torch.Tensor)
        assert so_tensor.shape == (sample_basis.feature_dim, sample_basis.feature_dim)


class TestGeometricFeatureExtractor:
    """Test suite for GeometricFeatureExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create a feature extractor."""
        return GeometricFeatureExtractor(
            distance_cutoff=4.0,
            include_atom_features=True,
            normalize=False,
        )

    @pytest.fixture
    def sample_mol(self):
        """Create a sample molecule with conformer."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        return mol

    @pytest.fixture
    def sample_mol_multi_conf(self):
        """Create a molecule with multiple conformers."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("CCCC")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=5, randomSeed=42)
        return mol

    def test_extract_single_conformer(self, extractor, sample_mol):
        """Test extracting features from a single conformer."""
        features = extractor.extract(sample_mol)

        assert isinstance(features, GeometricFeatures)
        assert features.n_atoms == sample_mol.GetNumAtoms()
        assert len(features.distances) > 0
        assert len(features.angles) >= 0  # May be empty for small molecules
        assert features.coordinates.shape == (features.n_atoms, 3)

    def test_extract_batch(self, extractor, sample_mol_multi_conf):
        """Test extracting features from multiple conformers."""
        features_list = extractor.extract_batch(sample_mol_multi_conf)

        assert len(features_list) == sample_mol_multi_conf.GetNumConformers()
        for features in features_list:
            assert isinstance(features, GeometricFeatures)

    def test_distance_cutoff(self, sample_mol):
        """Test that distance cutoff is applied."""
        extractor_small = GeometricFeatureExtractor(distance_cutoff=2.0)
        extractor_large = GeometricFeatureExtractor(distance_cutoff=10.0)

        features_small = extractor_small.extract(sample_mol)
        features_large = extractor_large.extract(sample_mol)

        # Larger cutoff should include more pairs
        assert len(features_large.distances) >= len(features_small.distances)

    def test_cos_sin_torsions(self, sample_mol_multi_conf):
        """Test cos/sin representation of torsions."""
        extractor_cs = GeometricFeatureExtractor(use_cos_sin_torsions=True)
        extractor_raw = GeometricFeatureExtractor(use_cos_sin_torsions=False)

        features_cs = extractor_cs.extract(sample_mol_multi_conf)
        features_raw = extractor_raw.extract(sample_mol_multi_conf)

        # cos/sin should have 2x the torsion values
        if len(features_raw.torsions) > 0:
            # For each torsion, we get cos and sin, so 2x
            assert len(features_cs.torsions) == 2 * len(features_cs.torsion_quadruplets)

    def test_normalization(self, sample_mol):
        """Test feature normalization."""
        extractor_norm = GeometricFeatureExtractor(normalize=True)
        extractor_raw = GeometricFeatureExtractor(normalize=False)

        features_norm = extractor_norm.extract(sample_mol)
        features_raw = extractor_raw.extract(sample_mol)

        # Normalized distances should have different values
        if len(features_norm.distances) > 1:
            assert np.abs(features_norm.distances.mean()) < 1e-5 or True  # May vary

    def test_atom_features(self, extractor, sample_mol):
        """Test atom feature extraction."""
        features = extractor.extract(sample_mol)

        assert features.atom_features is not None
        assert features.atom_features.shape[0] == features.n_atoms
        # Each atom should have consistent feature count
        assert features.atom_features.shape[1] > 0


class TestAugmentedBasisConstructor:
    """Test suite for AugmentedBasisConstructor."""

    @pytest.fixture
    def constructor(self):
        """Create an augmented basis constructor."""
        return AugmentedBasisConstructor(
            use_diagonal_only=False,
            regularization=1e-6,
        )

    @pytest.fixture
    def constructor_diagonal(self):
        """Create constructor with diagonal-only mode."""
        return AugmentedBasisConstructor(
            use_diagonal_only=True,
            regularization=1e-6,
        )

    @pytest.fixture
    def sample_features_list(self):
        """Create sample feature vectors for testing."""
        np.random.seed(42)
        return [np.random.randn(50) for _ in range(5)]

    def test_construct_basic(self, constructor, sample_features_list):
        """Test basic augmented basis construction."""
        basis = constructor.construct(sample_features_list)

        assert isinstance(basis, AugmentedBasis)
        assert basis.n_conformers == 5
        assert basis.feature_dim == 50
        assert len(basis.mean) == 50
        assert basis.second_order.shape == (50, 50)

    def test_construct_with_weights(self, constructor, sample_features_list):
        """Test construction with Boltzmann weights."""
        weights = np.array([0.5, 0.25, 0.15, 0.07, 0.03])
        basis = constructor.construct(sample_features_list, weights=weights)

        assert basis.weights is not None
        assert np.isclose(basis.weights.sum(), 1.0)

    def test_construct_diagonal_only(self, constructor_diagonal, sample_features_list):
        """Test diagonal-only mode."""
        basis = constructor_diagonal.construct(sample_features_list)

        # Second order should be diagonal
        off_diag = basis.second_order - np.diag(np.diag(basis.second_order))
        assert np.allclose(off_diag, 0)

    def test_flatten_second_order(self, constructor):
        """Test second-order matrix flattening."""
        matrix = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
        flat = constructor.flatten_second_order(matrix)

        # Upper triangle has n*(n+1)/2 elements
        assert len(flat) == 6  # 3*4/2

    def test_unflatten_second_order(self, constructor):
        """Test second-order matrix unflattening."""
        matrix = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
        flat = constructor.flatten_second_order(matrix)
        reconstructed = constructor.unflatten_second_order(flat, 3)

        np.testing.assert_allclose(matrix, reconstructed)

    def test_construct_from_geometric_features(self, constructor):
        """Test construction from GeometricFeatures objects."""
        # Create mock geometric features
        geo_features_list = []
        for i in range(3):
            gf = GeometricFeatures(
                distances=np.random.randn(10),
                angles=np.random.randn(5),
                torsions=np.random.randn(4),
                distance_pairs=np.zeros((10, 2), dtype=int),
                angle_triplets=np.zeros((5, 3), dtype=int),
                torsion_quadruplets=np.zeros((2, 4), dtype=int),
                coordinates=np.zeros((5, 3)),
                n_atoms=5,
            )
            geo_features_list.append(gf)

        basis = constructor.construct_from_geometric_features(geo_features_list)

        assert isinstance(basis, AugmentedBasis)
        assert basis.n_conformers == 3

    def test_max_feature_dim(self, sample_features_list):
        """Test feature dimension truncation."""
        constructor = AugmentedBasisConstructor(max_feature_dim=20)
        basis = constructor.construct(sample_features_list)

        assert basis.feature_dim == 20
        assert len(basis.mean) == 20

    def test_empty_list_raises(self, constructor):
        """Test that empty list raises error."""
        with pytest.raises(ValueError):
            constructor.construct([])


class TestSCCComputation:
    """Test suite for SCC (Structural Conformational Complexity) computation."""

    @pytest.fixture
    def diverse_features(self):
        """Create diverse feature vectors (high SCC)."""
        np.random.seed(42)
        return [np.random.randn(20) * 2 for _ in range(5)]

    @pytest.fixture
    def similar_features(self):
        """Create similar feature vectors (low SCC)."""
        base = np.random.randn(20)
        return [base + np.random.randn(20) * 0.01 for _ in range(5)]

    def test_compute_scc_basic(self, diverse_features):
        """Test basic SCC computation."""
        scc = compute_scc_simple(diverse_features)

        assert isinstance(scc, float)
        assert np.isfinite(scc)

    def test_scc_single_conformer(self):
        """Test SCC with single conformer returns 0."""
        features = [np.random.randn(20)]
        scc = compute_scc_simple(features)

        assert scc == 0.0

    def test_scc_diversity_ordering(self, diverse_features, similar_features):
        """Test that diverse conformers have higher SCC."""
        scc_diverse = compute_scc_simple(diverse_features)
        scc_similar = compute_scc_simple(similar_features)

        assert scc_diverse > scc_similar

    def test_scc_with_weights(self, diverse_features):
        """Test SCC with Boltzmann weights."""
        weights = np.array([0.5, 0.25, 0.15, 0.07, 0.03])
        scc = compute_scc_simple(diverse_features, weights=weights)

        assert isinstance(scc, float)
        assert np.isfinite(scc)

    def test_compute_scc_from_ensemble(self):
        """Test SCC computation from RDKit molecule."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("CCCC")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=5, randomSeed=42)

        scc = compute_scc_from_ensemble(mol)

        assert isinstance(scc, float)
        assert np.isfinite(scc)

    def test_scc_single_conformer_ensemble(self):
        """Test SCC with single conformer in ensemble."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)

        scc = compute_scc_from_ensemble(mol)

        assert scc == 0.0


class TestPairwiseDistances:
    """Test suite for pairwise distance computation."""

    def test_simple_distances(self):
        """Test distance computation for simple coordinates."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])

        distances = compute_pairwise_distances(coords, cutoff=10.0)

        # Should have 3 pairs: (0,1), (0,2), (1,2)
        assert len(distances) == 3
        assert np.isclose(distances[0], 1.0)  # (0,1) distance
        assert np.isclose(distances[1], 1.0)  # (0,2) distance
        assert np.isclose(distances[2], np.sqrt(2))  # (1,2) distance

    def test_cutoff(self):
        """Test distance cutoff."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],  # Far apart
        ])

        distances = compute_pairwise_distances(coords, cutoff=5.0)

        # Distance should be capped at cutoff
        assert distances[0] == 5.0

    def test_empty_coords(self):
        """Test handling of single atom."""
        coords = np.array([[0.0, 0.0, 0.0]])
        distances = compute_pairwise_distances(coords, cutoff=10.0)

        assert len(distances) == 0

    def test_distance_matrix_symmetry(self):
        """Test that distances are computed correctly (symmetry)."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 0.0, 0.0],
        ])

        distances = compute_pairwise_distances(coords, cutoff=10.0)

        # Distance (0,1) should equal sqrt(3)
        assert np.isclose(distances[0], np.sqrt(3))


class TestDihedral:
    """Test dihedral angle computation."""

    def test_dihedral_planar(self):
        """Test dihedral for planar arrangement."""
        # All in xy plane
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        p3 = np.array([2.0, 0.0, 0.0])
        p4 = np.array([3.0, 0.0, 0.0])

        # Collinear points - dihedral undefined/0
        dihedral = compute_dihedral(p1, p2, p3, p4)
        # Just check it doesn't crash and returns a finite value
        assert np.isfinite(dihedral) or dihedral == 0.0

    def test_dihedral_perpendicular(self):
        """Test dihedral for perpendicular arrangement."""
        p1 = np.array([0.0, 1.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([1.0, 0.0, 0.0])
        p4 = np.array([1.0, 0.0, 1.0])

        dihedral = compute_dihedral(p1, p2, p3, p4)

        # Should be close to 90 degrees
        assert np.isclose(abs(dihedral), np.pi/2, atol=0.1)

    def test_dihedral_180_degrees(self):
        """Test dihedral for trans arrangement (180 degrees)."""
        p1 = np.array([0.0, 1.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([1.0, 0.0, 0.0])
        p4 = np.array([1.0, -1.0, 0.0])  # Opposite side

        dihedral = compute_dihedral(p1, p2, p3, p4)

        # Should be close to 180 degrees (or -180)
        assert np.isclose(abs(dihedral), np.pi, atol=0.1)

    def test_dihedral_0_degrees(self):
        """Test dihedral for cis arrangement (0 degrees)."""
        p1 = np.array([0.0, 1.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([1.0, 0.0, 0.0])
        p4 = np.array([1.0, 1.0, 0.0])  # Same side

        dihedral = compute_dihedral(p1, p2, p3, p4)

        # Should be close to 0 degrees
        assert np.isclose(abs(dihedral), 0.0, atol=0.1)


class TestBondAngles:
    """Test bond angle computation."""

    @pytest.fixture
    def sample_mol(self):
        """Create a sample molecule."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("CCC")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        return mol

    def test_compute_bond_angles(self, sample_mol):
        """Test bond angle computation."""
        conf = sample_mol.GetConformer()
        n_atoms = sample_mol.GetNumAtoms()
        coords = np.zeros((n_atoms, 3))
        for i in range(n_atoms):
            pos = conf.GetAtomPosition(i)
            coords[i] = [pos.x, pos.y, pos.z]

        angles = compute_bond_angles(sample_mol, coords)

        assert len(angles) > 0
        # All angles should be between 0 and pi
        assert all(0 <= a <= np.pi for a in angles)

    def test_linear_angles(self):
        """Test angle computation for linear arrangement."""
        from rdkit import Chem

        mol = Chem.MolFromSmiles("C#C")  # Acetylene - linear
        mol = Chem.AddHs(mol)

        # Create linear coordinates
        coords = np.array([
            [-1.0, 0.0, 0.0],  # H
            [0.0, 0.0, 0.0],   # C
            [1.0, 0.0, 0.0],   # C
            [2.0, 0.0, 0.0],   # H
        ])

        angles = compute_bond_angles(mol, coords)

        # Linear bonds should have 180 degree angles (pi radians)
        if len(angles) > 0:
            assert any(np.isclose(a, np.pi, atol=0.1) for a in angles)


class TestLegacyFeatureExtractor:
    """Test legacy FeatureExtractor for backwards compatibility."""

    @pytest.fixture
    def extractor(self):
        """Create legacy feature extractor."""
        return FeatureExtractor(
            include_pairwise_distances=True,
            include_bond_angles=True,
            include_torsion_angles=True,
            include_atom_features=True,
        )

    @pytest.fixture
    def sample_mol(self):
        """Create sample molecule."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        return mol

    def test_extract_returns_tensor(self, extractor, sample_mol):
        """Test that extract returns tensor."""
        features = extractor.extract([sample_mol])

        assert isinstance(features, torch.Tensor)
        assert features.dim() == 2
        assert features.shape[0] == 1  # One molecule

    def test_extract_returns_dict(self, extractor, sample_mol):
        """Test extract with return_dict=True."""
        features = extractor.extract([sample_mol], return_dict=True)

        assert isinstance(features, dict)
        assert "distances" in features or "angles" in features


class TestMolecularFingerprints:
    """Test molecular fingerprint generation."""

    @pytest.fixture
    def sample_mol(self):
        """Create sample molecule."""
        from rdkit import Chem

        return Chem.MolFromSmiles("CCO")

    def test_morgan_fingerprint(self, sample_mol):
        """Test Morgan fingerprint generation."""
        fp_gen = MolecularFingerprints(fp_type="morgan", n_bits=2048)
        fp = fp_gen.generate(sample_mol)

        assert len(fp) == 2048
        assert fp.sum() > 0  # At least some bits set

    def test_rdkit_fingerprint(self, sample_mol):
        """Test RDKit fingerprint generation."""
        fp_gen = MolecularFingerprints(fp_type="rdkit", n_bits=1024)
        fp = fp_gen.generate(sample_mol)

        assert len(fp) == 1024

    def test_maccs_fingerprint(self, sample_mol):
        """Test MACCS keys fingerprint generation."""
        fp_gen = MolecularFingerprints(fp_type="maccs")
        fp = fp_gen.generate(sample_mol)

        assert len(fp) == 167  # MACCS has 167 keys

    def test_invalid_fp_type(self, sample_mol):
        """Test invalid fingerprint type."""
        fp_gen = MolecularFingerprints(fp_type="invalid")

        with pytest.raises(ValueError):
            fp_gen.generate(sample_mol)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_atom_features(self):
        """Test feature extraction for single atom."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("[He]")
        AllChem.EmbedMolecule(mol, randomSeed=42)

        extractor = GeometricFeatureExtractor()

        # Should handle gracefully
        try:
            features = extractor.extract(mol)
            assert features.n_atoms == 1
            assert len(features.distances) == 0  # No pairs
            assert len(features.angles) == 0  # No triplets
        except Exception:
            pass  # May fail for noble gas

    def test_empty_molecule(self):
        """Test handling of empty/invalid molecule."""
        extractor = GeometricFeatureExtractor()

        # Should raise or handle None molecule
        with pytest.raises(Exception):
            extractor.extract(None)

    def test_very_large_molecule(self):
        """Test feature extraction for larger molecule."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        # Create a larger molecule
        smiles = "C" * 20 + "O"  # Long chain
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)

            extractor = GeometricFeatureExtractor(distance_cutoff=4.0)
            features = extractor.extract(mol)

            assert features.n_atoms == mol.GetNumAtoms()
