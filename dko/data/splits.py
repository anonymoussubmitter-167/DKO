"""
Data splitting utilities for molecular datasets.

This module provides functions for splitting molecular datasets
using scaffold-based, random, and stratified methods.

Key features:
- Scaffold-based splitting (Bemis-Murcko scaffolds)
- Stratified splitting for classification tasks
- Stratified scaffold splitting (combines both)
- Cross-validation support
- Reproducible splits with caching
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from collections import defaultdict
import hashlib
import json

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def get_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """
    Get Bemis-Murcko scaffold for a molecule.

    Args:
        smiles: SMILES string
        include_chirality: Whether to include chirality in scaffold

    Returns:
        Scaffold SMILES string
    """
    if not RDKIT_AVAILABLE:
        return smiles  # Fallback: use SMILES as its own scaffold

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "INVALID"

    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol,
            includeChirality=include_chirality
        )
        return scaffold
    except Exception:
        return "ERROR"


def get_scaffold_mapping(smiles_list: List[str]) -> Dict[str, List[int]]:
    """
    Get mapping from scaffolds to molecule indices.

    Args:
        smiles_list: List of SMILES strings

    Returns:
        Dictionary mapping scaffold SMILES to list of molecule indices
    """
    scaffolds = defaultdict(list)

    for idx, smiles in enumerate(smiles_list):
        scaffold = get_scaffold(smiles)
        scaffolds[scaffold].append(idx)

    return dict(scaffolds)


def scaffold_split(
    smiles_list: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    stratify: bool = False,
    labels: Optional[np.ndarray] = None,
    balanced: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Split dataset based on Bemis-Murcko scaffolds.

    Ensures that molecules with the same scaffold are in the same split,
    which provides a more realistic evaluation of generalization.

    Args:
        smiles_list: List of SMILES strings
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed
        stratify: Whether to stratify by class (for classification)
        labels: Target labels (required if stratify=True)
        balanced: Whether to balance scaffold sizes across splits

    Returns:
        Dict with 'train', 'val', 'test' keys mapping to index arrays
    """
    np.random.seed(seed)

    # Normalize ratios
    total = train_ratio + val_ratio + test_ratio
    train_ratio /= total
    val_ratio /= total
    test_ratio /= total

    # Get scaffolds for each molecule
    scaffolds = get_scaffold_mapping(smiles_list)

    # Sort scaffolds by size
    if balanced:
        # Largest first for better balance
        scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)
    else:
        # Random order
        scaffold_sets = list(scaffolds.values())
        np.random.shuffle(scaffold_sets)

    # Assign scaffolds to splits
    n_total = len(smiles_list)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    # Handle edge case: ensure train gets at least 1 sample for very small datasets
    if n_total > 0 and n_train == 0 and train_ratio > 0:
        n_train = 1

    if stratify and labels is not None:
        # Stratified scaffold split
        return _stratified_scaffold_split(
            smiles_list, labels, scaffold_sets,
            train_ratio, val_ratio, test_ratio, seed
        )

    train_indices = []
    val_indices = []
    test_indices = []

    for scaffold_set in scaffold_sets:
        if len(train_indices) < n_train:
            train_indices.extend(scaffold_set)
        elif len(val_indices) < n_val:
            val_indices.extend(scaffold_set)
        else:
            test_indices.extend(scaffold_set)

    # Shuffle within each split
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)

    return {
        'train': np.array(train_indices),
        'val': np.array(val_indices),
        'test': np.array(test_indices)
    }


def _stratified_scaffold_split(
    smiles_list: List[str],
    labels: np.ndarray,
    scaffold_sets: List[List[int]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    """
    Stratified scaffold split for classification tasks.

    Attempts to maintain class balance while respecting scaffold boundaries.
    """
    np.random.seed(seed)
    labels = np.array(labels)

    # Handle multi-label case (use first column or most common label)
    if labels.ndim > 1:
        # Use first valid column
        labels = labels[:, 0]

    # Handle NaN values
    valid_mask = ~np.isnan(labels)

    # Get class distribution
    unique_classes = np.unique(labels[valid_mask])
    n_classes = len(unique_classes)

    # Compute scaffold class distributions
    scaffold_class_counts = []
    for scaffold_set in scaffold_sets:
        class_counts = np.zeros(n_classes)
        for idx in scaffold_set:
            if valid_mask[idx]:
                label = labels[idx]
                class_idx = np.where(unique_classes == label)[0]
                if len(class_idx) > 0:
                    class_counts[class_idx[0]] += 1
        scaffold_class_counts.append(class_counts)

    # Compute target counts per split per class
    total_per_class = np.zeros(n_classes)
    for counts in scaffold_class_counts:
        total_per_class += counts

    train_target = train_ratio * total_per_class
    val_target = val_ratio * total_per_class
    test_target = test_ratio * total_per_class

    # Greedy assignment to maintain balance
    train_indices = []
    val_indices = []
    test_indices = []

    train_counts = np.zeros(n_classes)
    val_counts = np.zeros(n_classes)
    test_counts = np.zeros(n_classes)

    # Sort scaffolds by size (largest first)
    sorted_idx = sorted(range(len(scaffold_sets)), key=lambda i: len(scaffold_sets[i]), reverse=True)

    for i in sorted_idx:
        scaffold_set = scaffold_sets[i]
        class_counts = scaffold_class_counts[i]

        # Compute deficit for each split
        train_deficit = np.sum(np.maximum(0, train_target - train_counts))
        val_deficit = np.sum(np.maximum(0, val_target - val_counts))
        test_deficit = np.sum(np.maximum(0, test_target - test_counts))

        # Assign to split with highest deficit
        if train_deficit >= val_deficit and train_deficit >= test_deficit:
            train_indices.extend(scaffold_set)
            train_counts += class_counts
        elif val_deficit >= test_deficit:
            val_indices.extend(scaffold_set)
            val_counts += class_counts
        else:
            test_indices.extend(scaffold_set)
            test_counts += class_counts

    # Shuffle within splits
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)

    return {
        'train': np.array(train_indices),
        'val': np.array(val_indices),
        'test': np.array(test_indices)
    }


def random_split(
    smiles_list: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Random split of dataset.

    Args:
        smiles_list: List of SMILES strings
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed

    Returns:
        Dict with 'train', 'val', 'test' keys mapping to index arrays
    """
    np.random.seed(seed)

    # Normalize ratios
    total = train_ratio + val_ratio + test_ratio
    train_ratio /= total
    val_ratio /= total

    n_total = len(smiles_list)
    indices = np.random.permutation(n_total)

    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    return {
        'train': indices[:n_train],
        'val': indices[n_train:n_train + n_val],
        'test': indices[n_train + n_val:]
    }


def stratified_split(
    smiles_list: List[str],
    labels: Union[List, np.ndarray],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    n_bins: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Stratified split for classification or binned regression.

    For regression tasks, bins the labels and stratifies on bins.

    Args:
        smiles_list: List of SMILES strings
        labels: Target labels
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed
        n_bins: Number of bins for regression stratification

    Returns:
        Dict with 'train', 'val', 'test' keys mapping to index arrays
    """
    np.random.seed(seed)

    # Normalize ratios
    total = train_ratio + val_ratio + test_ratio
    train_ratio /= total
    val_ratio /= total
    test_ratio /= total

    labels = np.array(labels)

    # Handle multi-dimensional labels
    if labels.ndim > 1:
        labels = labels[:, 0]

    # Handle NaN
    valid_mask = ~np.isnan(labels)
    valid_labels = labels[valid_mask]

    # Determine if classification or regression
    unique_labels = np.unique(valid_labels)

    if len(unique_labels) <= 10:
        # Classification: use labels directly
        bins = labels.copy()
    else:
        # Regression: bin the labels
        percentiles = np.percentile(valid_labels, np.linspace(0, 100, n_bins + 1))
        bins = np.digitize(labels, percentiles[1:-1])

    # Group indices by bin
    bin_indices = defaultdict(list)
    for idx, b in enumerate(bins):
        if valid_mask[idx]:
            bin_indices[int(b)].append(idx)

    train_indices = []
    val_indices = []
    test_indices = []

    # Split each bin
    for bin_idx_list in bin_indices.values():
        np.random.shuffle(bin_idx_list)
        n = len(bin_idx_list)
        n_train = max(1, int(train_ratio * n))
        n_val = max(0, int(val_ratio * n))

        train_indices.extend(bin_idx_list[:n_train])
        val_indices.extend(bin_idx_list[n_train:n_train + n_val])
        test_indices.extend(bin_idx_list[n_train + n_val:])

    # Final shuffle
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)

    return {
        'train': np.array(train_indices),
        'val': np.array(val_indices),
        'test': np.array(test_indices)
    }


def temporal_split(
    smiles_list: List[str],
    timestamps: List,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Dict[str, np.ndarray]:
    """
    Temporal split based on timestamps.

    Useful for datasets where data was collected over time.

    Args:
        smiles_list: List of SMILES strings
        timestamps: Timestamps for each molecule
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing

    Returns:
        Dict with 'train', 'val', 'test' keys mapping to index arrays
    """
    # Normalize ratios
    total = train_ratio + val_ratio + test_ratio
    train_ratio /= total
    val_ratio /= total

    # Sort by timestamp
    sorted_indices = np.argsort(timestamps)

    n_total = len(smiles_list)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    return {
        'train': sorted_indices[:n_train],
        'val': sorted_indices[n_train:n_train + n_val],
        'test': sorted_indices[n_train + n_val:]
    }


def get_split(
    smiles_list: List[str],
    labels: Optional[Union[List, np.ndarray]] = None,
    method: str = "scaffold",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """
    Get train/val/test split using specified method.

    Args:
        smiles_list: List of SMILES strings
        labels: Target labels (required for stratified split)
        method: Split method ('scaffold', 'random', 'stratified', 'stratified_scaffold')
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed
        **kwargs: Additional arguments for specific split methods

    Returns:
        Dict with 'train', 'val', 'test' keys mapping to index arrays
    """
    if method == "scaffold":
        return scaffold_split(
            smiles_list, train_ratio, val_ratio, test_ratio, seed,
            stratify=False, labels=labels, **kwargs
        )
    elif method == "stratified_scaffold":
        if labels is None:
            raise ValueError("Labels required for stratified_scaffold split")
        return scaffold_split(
            smiles_list, train_ratio, val_ratio, test_ratio, seed,
            stratify=True, labels=labels, **kwargs
        )
    elif method == "random":
        return random_split(smiles_list, train_ratio, val_ratio, test_ratio, seed)
    elif method == "stratified":
        if labels is None:
            raise ValueError("Labels required for stratified split")
        return stratified_split(
            smiles_list, labels, train_ratio, val_ratio, test_ratio, seed, **kwargs
        )
    else:
        raise ValueError(f"Unknown split method: {method}")


def cross_validation_splits(
    smiles_list: List[str],
    labels: Optional[Union[List, np.ndarray]] = None,
    n_folds: int = 5,
    method: str = "scaffold",
    seed: int = 42,
) -> List[Dict[str, np.ndarray]]:
    """
    Generate cross-validation splits.

    Args:
        smiles_list: List of SMILES strings
        labels: Target labels
        n_folds: Number of cross-validation folds
        method: Split method
        seed: Random seed

    Returns:
        List of dicts with 'train' and 'test' keys for each fold
    """
    np.random.seed(seed)

    n_total = len(smiles_list)

    if method == "scaffold" and RDKIT_AVAILABLE:
        # Group by scaffolds first
        scaffolds = get_scaffold_mapping(smiles_list)
        scaffold_sets = list(scaffolds.values())
        np.random.shuffle(scaffold_sets)

        # Assign scaffold sets to folds
        fold_indices = [[] for _ in range(n_folds)]
        for i, scaffold_set in enumerate(scaffold_sets):
            fold_indices[i % n_folds].extend(scaffold_set)
    else:
        # Random CV
        indices = np.random.permutation(n_total)
        fold_size = n_total // n_folds
        fold_indices = []
        for i in range(n_folds):
            start = i * fold_size
            end = start + fold_size if i < n_folds - 1 else n_total
            fold_indices.append(indices[start:end].tolist())

    # Generate train/test for each fold
    splits = []
    for i in range(n_folds):
        test_indices = fold_indices[i]
        train_indices = []
        for j in range(n_folds):
            if j != i:
                train_indices.extend(fold_indices[j])

        splits.append({
            'train': np.array(train_indices),
            'test': np.array(test_indices)
        })

    return splits


def get_split_hash(
    smiles_list: List[str],
    method: str,
    seed: int,
    **kwargs
) -> str:
    """
    Get a hash for a split configuration.

    Useful for caching splits.

    Args:
        smiles_list: List of SMILES strings
        method: Split method
        seed: Random seed
        **kwargs: Additional arguments

    Returns:
        Hash string
    """
    # Create a fingerprint of the split configuration
    config = {
        'n_molecules': len(smiles_list),
        'first_smiles': smiles_list[0] if smiles_list else '',
        'last_smiles': smiles_list[-1] if smiles_list else '',
        'method': method,
        'seed': seed,
    }
    config.update(kwargs)

    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:12]


def verify_no_scaffold_overlap(splits: Dict[str, np.ndarray], smiles_list: List[str]) -> bool:
    """
    Verify that there is no scaffold overlap between splits.

    Args:
        splits: Dictionary with train/val/test indices
        smiles_list: List of SMILES strings

    Returns:
        True if no overlap, False otherwise
    """
    train_scaffolds = set()
    val_scaffolds = set()
    test_scaffolds = set()

    for idx in splits['train']:
        train_scaffolds.add(get_scaffold(smiles_list[idx]))

    for idx in splits['val']:
        val_scaffolds.add(get_scaffold(smiles_list[idx]))

    for idx in splits['test']:
        test_scaffolds.add(get_scaffold(smiles_list[idx]))

    # Check overlaps
    train_val_overlap = train_scaffolds & val_scaffolds
    train_test_overlap = train_scaffolds & test_scaffolds
    val_test_overlap = val_scaffolds & test_scaffolds

    # Remove invalid scaffolds from overlap check
    invalid_scaffolds = {'INVALID', 'ERROR'}
    train_val_overlap -= invalid_scaffolds
    train_test_overlap -= invalid_scaffolds
    val_test_overlap -= invalid_scaffolds

    return len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0


def compute_split_statistics(
    splits: Dict[str, np.ndarray],
    smiles_list: List[str],
    labels: Optional[np.ndarray] = None,
) -> Dict[str, Dict]:
    """
    Compute statistics for each split.

    Args:
        splits: Dictionary with train/val/test indices
        smiles_list: List of SMILES strings
        labels: Optional target labels

    Returns:
        Dictionary with statistics for each split
    """
    stats = {}

    for split_name, indices in splits.items():
        split_stats = {
            'n_samples': len(indices),
            'n_unique_scaffolds': len(set(get_scaffold(smiles_list[i]) for i in indices))
        }

        if labels is not None:
            indices_arr = np.array(indices, dtype=np.int64)
            split_labels = labels[indices_arr]
            valid_mask = ~np.isnan(split_labels) if split_labels.ndim == 1 else ~np.isnan(split_labels).any(axis=1)
            valid_labels = split_labels[valid_mask] if split_labels.ndim == 1 else split_labels[valid_mask]

            if len(valid_labels) > 0:
                if split_labels.ndim == 1:
                    unique_vals = np.unique(valid_labels)
                    if len(unique_vals) <= 10:
                        # Classification
                        split_stats['class_balance'] = {
                            str(v): int(np.sum(valid_labels == v))
                            for v in unique_vals
                        }
                    else:
                        # Regression
                        split_stats['mean'] = float(np.mean(valid_labels))
                        split_stats['std'] = float(np.std(valid_labels))
                        split_stats['min'] = float(np.min(valid_labels))
                        split_stats['max'] = float(np.max(valid_labels))

        stats[split_name] = split_stats

    return stats
