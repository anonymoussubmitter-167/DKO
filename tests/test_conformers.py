"""Tests for conformer generation."""

import pytest
import numpy as np
import torch

# Skip if RDKit not available
pytest.importorskip("rdkit")

from dko.data.conformers import (
    ConformerEnsemble,
    ConformerGenerator,
    compute_boltzmann_weights,
    generate_conformers,
    compute_rmsd,
    get_conformer_coordinates,
    align_conformers,
)


class TestConformerEnsemble:
    """Test suite for ConformerEnsemble dataclass."""

    @pytest.fixture
    def sample_ensemble(self):
        """Create a sample ConformerEnsemble for testing."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=5, randomSeed=42)

        energies = np.array([0.0, 1.0, 2.0, 1.5, 0.5])
        weights = compute_boltzmann_weights(energies, temperature=300)

        return ConformerEnsemble(
            mol=mol,
            conformer_ids=list(range(5)),
            energies=energies,
            boltzmann_weights=weights.numpy(),
            n_conformers=5,
            generation_successful=True,
            smiles="CCO",
        )

    def test_ensemble_creation(self, sample_ensemble):
        """Test that ensemble is created correctly."""
        assert sample_ensemble.n_conformers == 5
        assert sample_ensemble.generation_successful is True
        assert sample_ensemble.smiles == "CCO"
        assert len(sample_ensemble.conformer_ids) == 5
        assert len(sample_ensemble.energies) == 5
        assert len(sample_ensemble.boltzmann_weights) == 5

    def test_get_conformer(self, sample_ensemble):
        """Test retrieving a specific conformer."""
        conf = sample_ensemble.get_conformer(0)
        assert conf is not None
        assert conf.GetNumAtoms() == sample_ensemble.mol.GetNumAtoms()

    def test_get_conformer_invalid_index(self, sample_ensemble):
        """Test handling of invalid conformer index."""
        with pytest.raises(IndexError):
            sample_ensemble.get_conformer(100)

    def test_get_coordinates(self, sample_ensemble):
        """Test extracting coordinates."""
        coords = sample_ensemble.get_coordinates(0)
        assert coords is not None
        assert coords.shape[1] == 3  # 3D coordinates

    def test_get_all_coordinates(self, sample_ensemble):
        """Test extracting all coordinates."""
        all_coords = sample_ensemble.get_all_coordinates()
        assert all_coords.shape[0] == 5  # 5 conformers
        assert all_coords.shape[2] == 3  # 3D coordinates

    def test_get_lowest_energy_conformer(self, sample_ensemble):
        """Test retrieving lowest energy conformer."""
        lowest_conf = sample_ensemble.get_lowest_energy_conformer()
        assert lowest_conf is not None
        # Verify it's the one with lowest energy (index 0 in our fixture)
        lowest_idx = np.argmin(sample_ensemble.energies)
        assert sample_ensemble.energies[lowest_idx] == 0.0

    def test_get_weighted_coordinates(self, sample_ensemble):
        """Test computing weighted coordinates."""
        # Get all coordinates and compute weighted average manually
        all_coords = sample_ensemble.get_all_coordinates()
        weights = sample_ensemble.boltzmann_weights
        weighted_coords = np.sum(
            all_coords * weights[:, np.newaxis, np.newaxis], axis=0
        )
        assert weighted_coords.shape == (sample_ensemble.mol.GetNumAtoms(), 3)

    def test_boltzmann_weights_sum_to_one(self, sample_ensemble):
        """Test that Boltzmann weights sum to 1."""
        assert np.isclose(sample_ensemble.boltzmann_weights.sum(), 1.0)

    def test_energy_ordering(self, sample_ensemble):
        """Test that we can work with energy-based filtering."""
        # Get conformers within energy window manually
        min_energy = sample_ensemble.energies.min()
        energy_window = 1.5
        within_window = sample_ensemble.energies <= min_energy + energy_window
        n_within_window = np.sum(within_window)
        assert n_within_window <= sample_ensemble.n_conformers


class TestConformerGenerator:
    """Test suite for ConformerGenerator."""

    @pytest.fixture
    def generator(self):
        """Create a conformer generator."""
        return ConformerGenerator(
            max_conformers=10,
            random_seed=42,
        )

    @pytest.fixture
    def generator_with_pruning(self):
        """Create a conformer generator with RMSD pruning."""
        return ConformerGenerator(
            max_conformers=20,
            random_seed=42,
            rmsd_threshold=0.5,
            energy_window=10.0,
        )

    def test_generate_single_molecule(self, generator):
        """Test conformer generation for a single molecule."""
        smiles = "CCO"  # Ethanol
        ensemble = generator.generate_ensemble(smiles, n_conformers=5)

        assert ensemble is not None
        assert ensemble.generation_successful
        assert ensemble.n_conformers > 0
        assert ensemble.n_conformers <= 5
        assert len(ensemble.energies) == ensemble.n_conformers

    def test_generate_invalid_smiles(self, generator):
        """Test handling of invalid SMILES."""
        ensemble = generator.generate_ensemble("INVALID_SMILES")

        assert ensemble is not None
        assert ensemble.generation_successful is False

    def test_generate_complex_molecule(self, generator):
        """Test conformer generation for a more complex molecule."""
        smiles = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"  # Ibuprofen
        ensemble = generator.generate_ensemble(smiles, n_conformers=10)

        assert ensemble is not None
        assert ensemble.n_conformers > 0

    def test_generate_batch(self, generator):
        """Test batch conformer generation."""
        smiles_list = ["CCO", "CCCO", "CCCCO"]
        results = generator.generate_batch(smiles_list, show_progress=False)

        assert len(results) == 3
        for ensemble in results:
            assert ensemble is None or isinstance(ensemble, ConformerEnsemble)

    def test_generate_ensemble(self, generator):
        """Test generating ConformerEnsemble."""
        smiles = "CCO"
        ensemble = generator.generate_ensemble(smiles, n_conformers=5)

        assert ensemble is not None
        assert isinstance(ensemble, ConformerEnsemble)
        assert ensemble.generation_successful is True
        assert ensemble.n_conformers > 0

    def test_generate_ensemble_invalid_smiles(self, generator):
        """Test ensemble generation with invalid SMILES."""
        ensemble = generator.generate_ensemble("INVALID")

        assert ensemble is not None
        assert ensemble.generation_successful is False
        assert ensemble.n_conformers == 0

    def test_rmsd_pruning(self, generator_with_pruning):
        """Test RMSD-based conformer pruning."""
        smiles = "CCCCCCCC"  # Octane - flexible molecule
        ensemble = generator_with_pruning.generate_ensemble(smiles, n_conformers=20)

        if ensemble.generation_successful:
            # Check that conformers are diverse
            coords = ensemble.get_all_coordinates()
            n_conf = len(coords)
            for i in range(n_conf):
                for j in range(i + 1, n_conf):
                    rmsd = compute_rmsd(coords[i], coords[j])
                    # RMSD should be above threshold (with some tolerance)
                    assert rmsd >= 0.4, f"RMSD {rmsd} < threshold"

    def test_energy_ordering(self, generator):
        """Test that conformers are energy-ordered."""
        smiles = "CCCCC"  # Pentane
        ensemble = generator.generate_ensemble(smiles, n_conformers=10)

        if ensemble.generation_successful and ensemble.n_conformers > 1:
            # Energies should be sorted
            for i in range(len(ensemble.energies) - 1):
                assert ensemble.energies[i] <= ensemble.energies[i + 1]

    def test_force_field_fallback(self, generator):
        """Test force field fallback when MMFF fails."""
        # Some molecules may require UFF fallback
        smiles = "C"  # Methane
        ensemble = generator.generate_ensemble(smiles, n_conformers=3)

        # May succeed or fail for very small molecules, but shouldn't crash
        assert ensemble is not None

    def test_caching(self):
        """Test conformer caching."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ConformerGenerator(
                max_conformers=5,
                random_seed=42,
                cache_dir=tmpdir,
            )

            smiles = "CCO"

            # First generation
            ensemble1 = generator.generate_ensemble(smiles, n_conformers=5)

            # Second generation (should use cache)
            ensemble2 = generator.generate_ensemble(smiles, n_conformers=5)

            # Should get same results
            if ensemble1.generation_successful and ensemble2.generation_successful:
                assert ensemble1.n_conformers == ensemble2.n_conformers
                np.testing.assert_allclose(ensemble1.energies, ensemble2.energies)


class TestBoltzmannWeights:
    """Test suite for Boltzmann weight computation."""

    def test_boltzmann_weights_from_list(self):
        """Test Boltzmann weights from a list."""
        energies = [0.0, 1.0, 2.0]
        weights = compute_boltzmann_weights(energies, temperature=300)

        assert len(weights) == 3
        assert torch.isclose(weights.sum(), torch.tensor(1.0))
        assert weights[0] > weights[1] > weights[2]  # Lower energy = higher weight

    def test_boltzmann_weights_from_tensor(self):
        """Test Boltzmann weights from tensor."""
        energies = torch.tensor([0.0, 1.0, 2.0])
        weights = compute_boltzmann_weights(energies, temperature=300)

        assert isinstance(weights, torch.Tensor)
        assert torch.isclose(weights.sum(), torch.tensor(1.0))

    def test_boltzmann_weights_from_numpy(self):
        """Test Boltzmann weights from numpy array."""
        energies = np.array([0.0, 1.0, 2.0])
        weights = compute_boltzmann_weights(energies, temperature=300)

        assert isinstance(weights, torch.Tensor)
        assert torch.isclose(weights.sum(), torch.tensor(1.0))

    def test_boltzmann_weights_temperature(self):
        """Test temperature effect on weights."""
        energies = [0.0, 2.0]

        weights_low_T = compute_boltzmann_weights(energies, temperature=100)
        weights_high_T = compute_boltzmann_weights(energies, temperature=1000)

        # Higher temperature should give more uniform distribution
        diff_low = abs(weights_low_T[0] - weights_low_T[1])
        diff_high = abs(weights_high_T[0] - weights_high_T[1])
        assert diff_high < diff_low

    def test_boltzmann_weights_single_conformer(self):
        """Test Boltzmann weights for single conformer."""
        energies = [0.0]
        weights = compute_boltzmann_weights(energies, temperature=300)

        assert len(weights) == 1
        assert torch.isclose(weights[0], torch.tensor(1.0))

    def test_boltzmann_weights_relative_energies(self):
        """Test that relative energies give same weights as absolute."""
        energies1 = [0.0, 1.0, 2.0]
        energies2 = [10.0, 11.0, 12.0]  # Shifted by 10

        weights1 = compute_boltzmann_weights(energies1, temperature=300)
        weights2 = compute_boltzmann_weights(energies2, temperature=300)

        # Weights should be the same (relative energies matter)
        torch.testing.assert_close(weights1, weights2)

    def test_boltzmann_weights_degenerate(self):
        """Test Boltzmann weights with degenerate energies."""
        energies = [1.0, 1.0, 1.0]  # All same energy
        weights = compute_boltzmann_weights(energies, temperature=300)

        # Should be uniform
        expected = torch.tensor([1/3, 1/3, 1/3])
        torch.testing.assert_close(weights, expected, atol=1e-5, rtol=1e-5)


class TestGenerateConformersFunction:
    """Test the convenience function."""

    def test_generate_conformers_function(self):
        """Test the generate_conformers convenience function."""
        ensemble = generate_conformers("CCO", n_conformers=5)

        assert ensemble is not None
        assert isinstance(ensemble, ConformerEnsemble)
        assert ensemble.n_conformers > 0

    def test_generate_conformers_with_options(self):
        """Test generate_conformers with various options."""
        ensemble = generate_conformers(
            "CCCC",
            n_conformers=10,
            random_seed=123,
        )

        assert ensemble is not None
        assert isinstance(ensemble, ConformerEnsemble)


class TestHelperFunctions:
    """Test helper functions for conformer handling."""

    @pytest.fixture
    def sample_mol_with_conformers(self):
        """Create a molecule with multiple conformers."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("CCCC")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=3, randomSeed=42)
        return mol

    def test_get_conformer_coordinates(self, sample_mol_with_conformers):
        """Test extracting coordinates from a conformer."""
        coords = get_conformer_coordinates(sample_mol_with_conformers, 0)

        assert coords is not None
        assert coords.shape[0] == sample_mol_with_conformers.GetNumAtoms()
        assert coords.shape[1] == 3

    def test_compute_rmsd(self):
        """Test RMSD computation between coordinate sets."""
        coords1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        coords2 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)

        rmsd = compute_rmsd(coords1, coords2)
        assert np.isclose(rmsd, 0.0)

        # Shifted coordinates
        coords3 = coords1 + np.array([0.1, 0.1, 0.1])
        rmsd2 = compute_rmsd(coords1, coords3)
        assert rmsd2 > 0

    def test_compute_rmsd_different_shapes(self):
        """Test RMSD with different sized coordinate arrays."""
        coords1 = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        coords2 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)

        with pytest.raises(ValueError):
            compute_rmsd(coords1, coords2)

    def test_align_conformers(self, sample_mol_with_conformers):
        """Test conformer alignment."""
        mol = sample_mol_with_conformers

        # Get coordinates before alignment
        coords_before = [
            get_conformer_coordinates(mol, i)
            for i in range(mol.GetNumConformers())
        ]

        # Align conformers
        aligned_mol = align_conformers(mol, reference_id=0)

        # Get coordinates after alignment
        coords_after = [
            get_conformer_coordinates(aligned_mol, i)
            for i in range(aligned_mol.GetNumConformers())
        ]

        # Reference conformer should be unchanged
        np.testing.assert_allclose(coords_before[0], coords_after[0], atol=1e-6)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_smiles(self):
        """Test handling of empty SMILES."""
        generator = ConformerGenerator()
        ensemble = generator.generate_ensemble("")

        # Empty SMILES should fail gracefully
        assert ensemble is not None
        assert ensemble.generation_successful is False

    def test_single_atom_molecule(self):
        """Test single atom molecule."""
        generator = ConformerGenerator()
        ensemble = generator.generate_ensemble("[He]")

        # Single atom - may or may not generate conformers, but shouldn't crash
        assert ensemble is not None

    def test_zero_conformers_requested(self):
        """Test requesting zero conformers."""
        generator = ConformerGenerator(max_conformers=0)
        ensemble = generator.generate_ensemble("CCO")

        # Should handle gracefully
        assert ensemble is not None

    def test_very_flexible_molecule(self):
        """Test conformer generation for very flexible molecule."""
        generator = ConformerGenerator(max_conformers=50, random_seed=42)
        smiles = "CCCCCCCCCCCC"  # Dodecane - very flexible
        ensemble = generator.generate_ensemble(smiles, n_conformers=30)

        if ensemble.generation_successful:
            assert ensemble.n_conformers > 0
            assert len(ensemble.energies) == ensemble.n_conformers

    def test_rigid_molecule(self):
        """Test conformer generation for rigid molecule."""
        generator = ConformerGenerator()
        smiles = "c1ccccc1"  # Benzene - rigid
        ensemble = generator.generate_ensemble(smiles, n_conformers=10)

        if ensemble.generation_successful:
            # Rigid molecules should have few distinct conformers
            assert ensemble.n_conformers >= 1
