#!/usr/bin/env python3
"""
Prepare BDE dataset (MARCEL benchmark) for DKO experiments.

5,915 reactions with bond dissociation energy targets.
Uses ligand conformers for feature extraction.
"""

import pickle
import sys
import time
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import Chem
from dko.data.features import GeometricFeatureExtractor


SPLIT_SEED = 123  # MARCEL default
TRAIN_FRAC = 0.7
VAL_FRAC = 0.1
MAX_CONFORMERS = 20  # MARCEL default


def compute_mmff_energy(mol):
    """Compute MMFF94 energy for a molecule with a conformer."""
    try:
        from rdkit.Chem import AllChem
        mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94')
        if mp is None:
            return None
        ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=0)
        if ff is None:
            return None
        return ff.CalcEnergy()
    except Exception:
        return None


def extract_features_single(args):
    """Extract features for one reaction's ligand conformers."""
    reaction_name, mol_blocks, extractor_params = args

    extractor = GeometricFeatureExtractor(**extractor_params)
    features = []
    energies = []

    for mol_block in mol_blocks[:MAX_CONFORMERS]:
        try:
            mol = Chem.MolFromMolBlock(mol_block, removeHs=False)
            if mol is None or mol.GetNumConformers() == 0:
                continue
            geo = extractor.extract(mol, conformer_id=0)
            features.append(geo.to_flat_vector())
            # Extract actual MMFF94 energy for Boltzmann weighting
            energy = compute_mmff_energy(mol)
            energies.append(energy if energy is not None else 0.0)
        except Exception:
            continue

    if len(features) == 0:
        return None
    return (reaction_name, features, energies)


def main():
    data_dir = Path("data/marcel/BDE")
    if not data_dir.exists():
        print(f"ERROR: {data_dir} not found")
        sys.exit(1)

    # Load targets
    print("Loading BDE targets...")
    targets = {}
    with open(data_dir / "BDE.txt") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                name = parts[0]
                energy = float(parts[1])
                targets[name] = energy
    print(f"  {len(targets)} reactions with targets")

    # Read ligand SDF and group by reaction name
    print("Reading ligands SDF...")
    suppl = Chem.SDMolSupplier(str(data_dir / "ligands.sdf"), removeHs=False, sanitize=False)

    mol_blocks = defaultdict(list)
    smiles_map = {}
    total = 0
    for mol in tqdm(suppl, desc="Reading ligands"):
        if mol is None:
            continue
        try:
            name = mol.GetProp("Name")
        except Exception:
            continue
        mol_blocks[name].append(Chem.MolToMolBlock(mol))
        if name not in smiles_map:
            try:
                smiles_map[name] = Chem.MolToSmiles(Chem.RemoveHs(mol))
            except Exception:
                smiles_map[name] = ""
        total += 1

    print(f"  {total} ligand conformers, {len(mol_blocks)} unique reactions")

    # Only keep reactions with both conformers and targets
    valid_names = [n for n in mol_blocks if n in targets]
    print(f"  {len(valid_names)} reactions with both conformers and targets")

    # Extract features
    extractor_params = {
        "distance_cutoff": 4.0,
        "include_atom_features": True,
        "normalize": False,
        "use_cos_sin_torsions": True,
    }

    tasks = [(name, mol_blocks[name], extractor_params) for name in valid_names]

    n_workers = min(cpu_count(), 16)
    print(f"Extracting features with {n_workers} workers...")
    start = time.time()

    results = {}
    energies_map = {}
    with Pool(n_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(extract_features_single, tasks, chunksize=20),
            total=len(tasks),
            desc="Featurizing",
        ):
            if result is not None:
                name, features, energies = result
                results[name] = features
                energies_map[name] = energies

    elapsed = time.time() - start
    print(f"Extracted features for {len(results)}/{len(valid_names)} reactions in {elapsed:.1f}s")

    # Order consistently
    reaction_names = sorted(results.keys())
    n_reactions = len(reaction_names)

    # Create splits
    rng = np.random.RandomState(SPLIT_SEED)
    indices = rng.permutation(n_reactions)
    n_train = int(n_reactions * TRAIN_FRAC)
    n_val = int(n_reactions * VAL_FRAC)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Feature stats
    n_confs = [len(results[n]) for n in reaction_names]
    print(f"Conformers: min={min(n_confs)}, max={max(n_confs)}, mean={np.mean(n_confs):.1f}")

    # Save dataset
    dataset_name = "bde"
    out_dir = Path(f"data/conformers/{dataset_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = np.array([targets[reaction_names[i]] for i in range(n_reactions)])

    for split_name, split_idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        # Use actual MMFF94 energies for Boltzmann weighting
        split_energies = []
        for i in split_idx:
            e = np.array(energies_map[reaction_names[i]])
            # Convert energies to Boltzmann weights (kT ~ 0.6 kcal/mol at 300K)
            # Lower energy conformers get higher weight
            if np.any(e != 0):
                e_shifted = e - e.min()  # Shift to avoid overflow
                split_energies.append(e_shifted)
            else:
                # Fallback to uniform if all energies are zero (extraction failed)
                split_energies.append(np.ones(len(e)))

        split_data = {
            "smiles": [smiles_map.get(reaction_names[i], "") for i in split_idx],
            "labels": labels[split_idx],
            "features": [results[reaction_names[i]] for i in split_idx],
            "energies": split_energies,
            "indices": split_idx,
            "dataset_config": {
                "task": "regression",
                "source": "MARCEL/BDE",
                "target": "BindingEnergy",
                "n_molecules": len(split_idx),
            },
        }

        out_path = out_dir / f"{split_name}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(split_data, f)

        print(f"  Saved {out_path} ({len(split_idx)} reactions)")

    print(f"\nDone! Created: bde")


if __name__ == "__main__":
    main()
