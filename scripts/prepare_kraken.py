#!/usr/bin/env python3
"""
Prepare Kraken dataset (from MARCEL benchmark) for DKO experiments.

Kraken: 1,552 organophosphorus ligands with Sterimol steric descriptors.
These descriptors explicitly depend on 3D conformer geometry, making this
the ideal test case for DKO's second-order statistics.

Targets: sterimol_B5, sterimol_L, sterimol_burB5, sterimol_burL
"""

import pickle
import sys
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import Chem
from dko.data.features import GeometricFeatureExtractor


TARGETS = ["sterimol_B5", "sterimol_L", "sterimol_burB5", "sterimol_burL"]
SPLIT_SEED = 42
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
MAX_CONFORMERS = 50


def extract_features_for_molecule(args):
    """Extract geometric features for all conformers of one molecule.

    Returns (mol_id, smiles, targets_dict, features_list, energies_list) or None on failure.
    """
    mol_id, mol_data, extractor_params = args
    smiles_or_molblock, targets, conformers = mol_data

    extractor = GeometricFeatureExtractor(**extractor_params)

    # Get SMILES
    if '\n' in str(smiles_or_molblock):
        # It's a mol block, convert to SMILES
        mol_tmp = Chem.MolFromMolBlock(str(smiles_or_molblock), removeHs=False)
        if mol_tmp is None:
            return None
        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol_tmp))
    else:
        smiles = str(smiles_or_molblock)

    # Sort conformers by Boltzmann weight (descending) and take top MAX_CONFORMERS
    conf_items = list(conformers.items())
    conf_items.sort(key=lambda x: x[1][1], reverse=True)  # sort by weight
    conf_items = conf_items[:MAX_CONFORMERS]

    features_list = []
    energies_list = []

    for conf_id, (mol_block, weight, conf_targets) in conf_items:
        try:
            mol = Chem.MolFromMolBlock(mol_block, removeHs=False)
            if mol is None:
                continue
            if mol.GetNumConformers() == 0:
                continue

            geo_features = extractor.extract(mol, conformer_id=0)
            feat_vec = geo_features.to_flat_vector()
            features_list.append(feat_vec)
            energies_list.append(weight)  # Use Boltzmann weight as "energy"
        except Exception as e:
            continue

    if len(features_list) == 0:
        return None

    return (mol_id, smiles, targets, features_list, energies_list)


def main():
    kraken_path = Path("data/marcel/Kraken.pickle")
    if not kraken_path.exists():
        print(f"ERROR: {kraken_path} not found")
        sys.exit(1)

    print("Loading Kraken dataset...")
    with open(kraken_path, "rb") as f:
        raw_data = pickle.load(f)

    print(f"Loaded {len(raw_data)} molecules")

    # Prepare extraction tasks
    extractor_params = {
        "distance_cutoff": 4.0,
        "include_atom_features": True,
        "normalize": False,
        "use_cos_sin_torsions": True,
    }

    tasks = []
    for mol_id, mol_data in raw_data.items():
        tasks.append((mol_id, mol_data, extractor_params))

    # Extract features with multiprocessing
    n_workers = min(cpu_count(), 16)
    print(f"Extracting features with {n_workers} workers...")
    start = time.time()

    results = []
    with Pool(n_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(extract_features_for_molecule, tasks),
            total=len(tasks),
            desc="Processing molecules",
        ):
            if result is not None:
                results.append(result)

    elapsed = time.time() - start
    print(f"Extracted features for {len(results)}/{len(tasks)} molecules in {elapsed:.1f}s")

    # Collect all data
    all_smiles = []
    all_targets = {t: [] for t in TARGETS}
    all_features = []
    all_energies = []

    for mol_id, smiles, targets, features, energies in results:
        all_smiles.append(smiles)
        for t in TARGETS:
            all_targets[t].append(targets[t])
        all_features.append(features)
        all_energies.append(np.array(energies))

    n_mols = len(all_smiles)
    print(f"\nTotal molecules: {n_mols}")

    # Feature dimension statistics
    feat_dims = [f[0].shape[0] for f in all_features]
    n_confs = [len(f) for f in all_features]
    print(f"Feature dims: min={min(feat_dims)}, max={max(feat_dims)}, median={np.median(feat_dims):.0f}")
    print(f"Conformers: min={min(n_confs)}, max={max(n_confs)}, mean={np.mean(n_confs):.1f}")

    # Create splits (random, reproducible)
    rng = np.random.RandomState(SPLIT_SEED)
    indices = rng.permutation(n_mols)
    n_train = int(n_mols * TRAIN_FRAC)
    n_val = int(n_mols * VAL_FRAC)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Save per-target datasets
    for target_name in TARGETS:
        dataset_name = f"kraken_{target_name.replace('sterimol_', '')}"
        out_dir = Path(f"data/conformers/{dataset_name}")
        out_dir.mkdir(parents=True, exist_ok=True)

        labels = np.array(all_targets[target_name])

        for split_name, split_idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            split_data = {
                "smiles": [all_smiles[i] for i in split_idx],
                "labels": labels[split_idx],
                "features": [all_features[i] for i in split_idx],
                "energies": [all_energies[i] for i in split_idx],
                "boltzmann_weights": [all_energies[i] for i in split_idx],
                "indices": split_idx,
                "dataset_config": {
                    "task": "regression",
                    "source": "MARCEL/Kraken",
                    "target": target_name,
                    "n_molecules": len(split_idx),
                },
            }

            out_path = out_dir / f"{split_name}.pkl"
            with open(out_path, "wb") as f:
                pickle.dump(split_data, f)

            print(f"  Saved {out_path} ({len(split_idx)} molecules)")

    print("\nDone! Created datasets:")
    for target_name in TARGETS:
        dataset_name = f"kraken_{target_name.replace('sterimol_', '')}"
        print(f"  {dataset_name}")


if __name__ == "__main__":
    main()
