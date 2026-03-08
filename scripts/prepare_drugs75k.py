#!/usr/bin/env python3
"""
Prepare Drugs-75K dataset (MARCEL benchmark) for DKO experiments.

75,099 molecules with 3 electronic property targets: ip, ea, chi.
Uses the same geometric feature extraction as other DKO datasets.
"""

import pickle
import sys
import time
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import Chem
from dko.data.features import GeometricFeatureExtractor


TARGETS = ["ip", "ea", "chi"]
SPLIT_SEED = 123  # MARCEL default
TRAIN_FRAC = 0.7
VAL_FRAC = 0.1
MAX_CONFORMERS = 20  # MARCEL default


def extract_features_single(args):
    """Extract features for one molecule's conformers."""
    mol_name, mol_blocks, extractor_params = args

    extractor = GeometricFeatureExtractor(**extractor_params)
    features = []

    for mol_block in mol_blocks[:MAX_CONFORMERS]:
        try:
            mol = Chem.MolFromMolBlock(mol_block, removeHs=False)
            if mol is None or mol.GetNumConformers() == 0:
                continue
            geo = extractor.extract(mol, conformer_id=0)
            features.append(geo.to_flat_vector())
        except Exception:
            continue

    if len(features) == 0:
        return None
    return (mol_name, features)


def main():
    data_dir = Path("data/marcel/Drugs")
    if not data_dir.exists():
        print(f"ERROR: {data_dir} not found")
        sys.exit(1)

    # Load targets from CSV
    print("Loading Drugs CSV...")
    df = pd.read_csv(data_dir / "Drugs.csv")
    targets_by_name = {}
    for _, row in df.iterrows():
        targets_by_name[row["name"]] = {t: row[t] for t in TARGETS}
    print(f"  {len(targets_by_name)} molecules in CSV")

    # Read SDF and group conformers by molecule name
    print("Reading Drugs SDF (this takes a minute)...")
    sdf_path = str(data_dir / "Drugs.sdf")
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)

    mol_blocks = defaultdict(list)
    total = 0
    for mol in tqdm(suppl, desc="Reading SDF"):
        if mol is None:
            continue
        try:
            name = mol.GetProp("name")
        except Exception:
            continue
        mol_blocks[name].append(Chem.MolToMolBlock(mol))
        total += 1

    print(f"  {total} conformer entries, {len(mol_blocks)} unique molecules")

    # Only keep molecules that are in both SDF and CSV
    valid_names = [n for n in mol_blocks if n in targets_by_name]
    print(f"  {len(valid_names)} molecules with both conformers and targets")

    # Extract features with multiprocessing
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
    with Pool(n_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(extract_features_single, tasks, chunksize=50),
            total=len(tasks),
            desc="Featurizing",
        ):
            if result is not None:
                mol_name, features = result
                results[mol_name] = features

    elapsed = time.time() - start
    print(f"Extracted features for {len(results)}/{len(valid_names)} molecules in {elapsed:.1f}s")

    # Order consistently
    mol_names = sorted(results.keys())
    n_mols = len(mol_names)

    # Get SMILES from SDF (first conformer per molecule)
    smiles_map = {}
    suppl2 = Chem.SDMolSupplier(sdf_path, removeHs=False)
    for mol in suppl2:
        if mol is None:
            continue
        try:
            name = mol.GetProp("name")
        except Exception:
            continue
        if name not in smiles_map and name in results:
            try:
                smi = mol.GetProp("smiles")
            except Exception:
                smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
            smiles_map[name] = smi

    # Create splits (MARCEL: 70/10/20 random, seed=123)
    rng = np.random.RandomState(SPLIT_SEED)
    indices = rng.permutation(n_mols)
    n_train = int(n_mols * TRAIN_FRAC)
    n_val = int(n_mols * VAL_FRAC)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Feature stats
    feat_dims = [results[mol_names[0]][0].shape[0]]
    n_confs = [len(results[n]) for n in mol_names]
    print(f"Feature dim (first mol): {feat_dims[0]}")
    print(f"Conformers: min={min(n_confs)}, max={max(n_confs)}, mean={np.mean(n_confs):.1f}")

    # Save per-target datasets
    for target_name in TARGETS:
        dataset_name = f"drugs_{target_name}"
        out_dir = Path(f"data/conformers/{dataset_name}")
        out_dir.mkdir(parents=True, exist_ok=True)

        labels = np.array([targets_by_name[mol_names[i]][target_name] for i in range(n_mols)])

        for split_name, split_idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            split_data = {
                "smiles": [smiles_map.get(mol_names[i], "") for i in split_idx],
                "labels": labels[split_idx],
                "features": [results[mol_names[i]] for i in split_idx],
                "energies": [np.ones(len(results[mol_names[i]])) for i in split_idx],
                "indices": split_idx,
                "dataset_config": {
                    "task": "regression",
                    "source": "MARCEL/Drugs-75K",
                    "target": target_name,
                    "n_molecules": len(split_idx),
                },
            }

            out_path = out_dir / f"{split_name}.pkl"
            with open(out_path, "wb") as f:
                pickle.dump(split_data, f)

            print(f"  Saved {out_path} ({len(split_idx)} molecules)")

    print(f"\nDone! Created: drugs_ip, drugs_ea, drugs_chi")


if __name__ == "__main__":
    main()
