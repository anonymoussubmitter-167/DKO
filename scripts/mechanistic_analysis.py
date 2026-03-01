#!/usr/bin/env python3
"""
Mechanistic Analysis: Why Do Fingerprints Beat Conformer Features?

This script investigates the information content of different feature types
to understand why Morgan fingerprints outperform conformer-based methods.

Analysis includes:
1. Mutual information between features and targets
2. Feature-target correlations by feature type
3. SHAP/importance analysis for XGBoost models
4. Information overlap analysis
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def load_dataset(dataset_name: str, max_dim: int = 1024) -> Tuple[List, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load dataset with both conformer features and fingerprints.

    Returns:
        smiles_list, conformer_features (mu), fingerprints, targets
    """
    data_path = Path(f"data/conformers/{dataset_name}/train.pkl")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    smiles_list = data["smiles"]
    raw_features = data["features"]  # List of [conformer_features] per molecule
    weights = data.get("boltzmann_weights", None)
    targets = np.array(data["labels"])

    # Compute mu (weighted mean) for each molecule
    features = []
    for i, mol_feats in enumerate(raw_features):
        # mol_feats is list of conformer feature arrays
        mol_feats_padded = []
        for conf_feat in mol_feats:
            conf_feat = np.array(conf_feat).flatten()
            if len(conf_feat) > max_dim:
                conf_feat = conf_feat[:max_dim]
            elif len(conf_feat) < max_dim:
                conf_feat = np.pad(conf_feat, (0, max_dim - len(conf_feat)))
            mol_feats_padded.append(conf_feat)

        mol_feats_arr = np.array(mol_feats_padded)  # (n_conf, max_dim)

        # Compute weighted mean (mu)
        if weights is not None and len(weights[i]) == len(mol_feats):
            w = np.array(weights[i])
            w = w / (w.sum() + 1e-10)
            mu = np.average(mol_feats_arr, axis=0, weights=w)
        else:
            mu = mol_feats_arr.mean(axis=0)

        features.append(mu)

    features = np.array(features)

    # Compute Morgan fingerprints
    if RDKIT_AVAILABLE:
        fps = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                fps.append(np.array(fp))
            else:
                fps.append(np.zeros(2048))
        fingerprints = np.array(fps)
    else:
        fingerprints = None

    return smiles_list, features, fingerprints, targets


def compute_correlations(
    features: np.ndarray,
    targets: np.ndarray,
    feature_name: str = "feature",
) -> Dict:
    """Compute correlations between features and targets."""
    n_features = features.shape[1]

    correlations = []
    p_values = []

    for i in range(n_features):
        feat = features[:, i]
        # Skip constant features
        if np.std(feat) < 1e-10:
            correlations.append(0.0)
            p_values.append(1.0)
            continue

        r, p = stats.pearsonr(feat, targets)
        correlations.append(r if not np.isnan(r) else 0.0)
        p_values.append(p if not np.isnan(p) else 1.0)

    correlations = np.array(correlations)
    p_values = np.array(p_values)

    return {
        "name": feature_name,
        "n_features": n_features,
        "correlations": correlations,
        "p_values": p_values,
        "mean_abs_correlation": np.mean(np.abs(correlations)),
        "max_abs_correlation": np.max(np.abs(correlations)),
        "n_significant": np.sum(p_values < 0.05),
        "n_strong": np.sum(np.abs(correlations) > 0.3),
        "top_10_mean": np.mean(np.sort(np.abs(correlations))[-10:]),
    }


def compute_mutual_information(
    features: np.ndarray,
    targets: np.ndarray,
    feature_name: str = "feature",
    n_neighbors: int = 5,
) -> Dict:
    """Compute mutual information between features and targets."""
    # Subsample for efficiency
    n_samples = min(len(targets), 2000)
    idx = np.random.choice(len(targets), n_samples, replace=False)

    mi = mutual_info_regression(
        features[idx],
        targets[idx],
        n_neighbors=n_neighbors,
        random_state=42,
    )

    return {
        "name": feature_name,
        "n_features": features.shape[1],
        "mutual_information": mi,
        "mean_mi": np.mean(mi),
        "max_mi": np.max(mi),
        "total_mi": np.sum(mi),
        "n_informative": np.sum(mi > 0.01),
        "top_10_mean_mi": np.mean(np.sort(mi)[-10:]),
    }


def analyze_feature_importance(
    features: np.ndarray,
    targets: np.ndarray,
    feature_name: str = "feature",
) -> Dict:
    """Train XGBoost and extract feature importance."""
    if not XGBOOST_AVAILABLE:
        return {"name": feature_name, "error": "XGBoost not available"}

    # Split
    n = len(targets)
    train_idx = np.random.choice(n, int(0.8 * n), replace=False)
    test_idx = np.array([i for i in range(n) if i not in train_idx])

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        tree_method="hist",
    )
    model.fit(features[train_idx], targets[train_idx])

    preds = model.predict(features[test_idx])
    r2 = r2_score(targets[test_idx], preds)
    rmse = np.sqrt(np.mean((preds - targets[test_idx]) ** 2))

    importance = model.feature_importances_

    return {
        "name": feature_name,
        "r2": r2,
        "rmse": rmse,
        "importance": importance,
        "mean_importance": np.mean(importance),
        "max_importance": np.max(importance),
        "n_nonzero": np.sum(importance > 0),
        "top_10_importance": np.mean(np.sort(importance)[-10:]),
    }


def compare_information_content(
    conformer_features: np.ndarray,
    fingerprints: np.ndarray,
    targets: np.ndarray,
) -> Dict:
    """Compare information content between feature types."""
    results = {}

    # Correlations
    print("Computing correlations...")
    results["conf_corr"] = compute_correlations(conformer_features, targets, "conformer")
    if fingerprints is not None:
        results["fp_corr"] = compute_correlations(fingerprints, targets, "fingerprint")

    # Mutual information
    print("Computing mutual information...")
    results["conf_mi"] = compute_mutual_information(conformer_features, targets, "conformer")
    if fingerprints is not None:
        results["fp_mi"] = compute_mutual_information(fingerprints, targets, "fingerprint")

    # Feature importance
    print("Training XGBoost for importance...")
    results["conf_imp"] = analyze_feature_importance(conformer_features, targets, "conformer")
    if fingerprints is not None:
        results["fp_imp"] = analyze_feature_importance(fingerprints, targets, "fingerprint")

    # Hybrid
    if fingerprints is not None:
        hybrid = np.hstack([fingerprints, conformer_features])
        results["hybrid_imp"] = analyze_feature_importance(hybrid, targets, "hybrid")

    return results


def analyze_fingerprint_bits(
    fingerprints: np.ndarray,
    targets: np.ndarray,
    smiles_list: List[str],
    top_k: int = 20,
) -> Dict:
    """Analyze which fingerprint bits are most predictive."""
    # Get correlations
    corrs = []
    for i in range(fingerprints.shape[1]):
        bit = fingerprints[:, i]
        if np.std(bit) > 0:
            r, _ = stats.pearsonr(bit, targets)
            corrs.append((i, r if not np.isnan(r) else 0))
        else:
            corrs.append((i, 0))

    # Sort by absolute correlation
    corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    top_bits = corrs[:top_k]

    # Try to interpret top bits (what substructures do they encode?)
    interpretations = []
    if RDKIT_AVAILABLE:
        for bit_idx, corr in top_bits:
            # Find molecules that have this bit set
            has_bit = fingerprints[:, bit_idx] == 1
            n_with = np.sum(has_bit)
            mean_target_with = np.mean(targets[has_bit]) if n_with > 0 else 0
            mean_target_without = np.mean(targets[~has_bit]) if np.sum(~has_bit) > 0 else 0

            interpretations.append({
                "bit_idx": int(bit_idx),
                "correlation": float(corr),
                "n_molecules_with_bit": int(n_with),
                "frac_with_bit": float(n_with / len(targets)),
                "mean_target_with": float(mean_target_with),
                "mean_target_without": float(mean_target_without),
                "target_delta": float(mean_target_with - mean_target_without),
            })

    return {
        "top_bits": interpretations,
        "n_bits_analyzed": len(corrs),
    }


def print_summary(results: Dict, dataset_name: str):
    """Print summary of analysis."""
    print("\n" + "=" * 70)
    print(f"MECHANISTIC ANALYSIS: {dataset_name}")
    print("=" * 70)

    # Correlation comparison
    print("\n## Feature-Target Correlations")
    print("-" * 50)
    print(f"{'Metric':<30} {'Conformer':<15} {'Fingerprint':<15}")
    print("-" * 50)

    conf = results.get("conf_corr", {})
    fp = results.get("fp_corr", {})

    print(f"{'Mean |r|':<30} {conf.get('mean_abs_correlation', 0):.4f}          {fp.get('mean_abs_correlation', 0):.4f}")
    print(f"{'Max |r|':<30} {conf.get('max_abs_correlation', 0):.4f}          {fp.get('max_abs_correlation', 0):.4f}")
    print(f"{'# Strong (|r|>0.3)':<30} {conf.get('n_strong', 0):<15} {fp.get('n_strong', 0):<15}")
    print(f"{'Top-10 Mean |r|':<30} {conf.get('top_10_mean', 0):.4f}          {fp.get('top_10_mean', 0):.4f}")

    # Mutual information
    print("\n## Mutual Information")
    print("-" * 50)

    conf_mi = results.get("conf_mi", {})
    fp_mi = results.get("fp_mi", {})

    print(f"{'Mean MI':<30} {conf_mi.get('mean_mi', 0):.4f}          {fp_mi.get('mean_mi', 0):.4f}")
    print(f"{'Max MI':<30} {conf_mi.get('max_mi', 0):.4f}          {fp_mi.get('max_mi', 0):.4f}")
    print(f"{'Total MI':<30} {conf_mi.get('total_mi', 0):.4f}          {fp_mi.get('total_mi', 0):.4f}")

    # XGBoost performance
    print("\n## XGBoost Performance")
    print("-" * 50)

    conf_imp = results.get("conf_imp", {})
    fp_imp = results.get("fp_imp", {})
    hybrid_imp = results.get("hybrid_imp", {})

    print(f"{'Feature Type':<20} {'R²':<10} {'RMSE':<10}")
    print(f"{'Conformer':<20} {conf_imp.get('r2', 0):.4f}     {conf_imp.get('rmse', 0):.4f}")
    print(f"{'Fingerprint':<20} {fp_imp.get('r2', 0):.4f}     {fp_imp.get('rmse', 0):.4f}")
    print(f"{'Hybrid':<20} {hybrid_imp.get('r2', 0):.4f}     {hybrid_imp.get('rmse', 0):.4f}")

    # Key insight
    print("\n## Key Insight")
    print("-" * 50)

    if fp_imp.get('r2', 0) > conf_imp.get('r2', 0):
        delta = fp_imp.get('r2', 0) - conf_imp.get('r2', 0)
        print(f"Fingerprints outperform conformer features by {delta:.2%} R²")

        if hybrid_imp.get('r2', 0) > fp_imp.get('r2', 0):
            hybrid_delta = hybrid_imp.get('r2', 0) - fp_imp.get('r2', 0)
            print(f"Hybrid features add {hybrid_delta:.2%} R² over fingerprints alone")
            print("→ Conformer features provide COMPLEMENTARY information")
        else:
            print("→ Conformer features provide NO additional information")
    else:
        print("Conformer features outperform fingerprints on this dataset!")


def main():
    parser = argparse.ArgumentParser(description="Mechanistic analysis of feature types")
    parser.add_argument("--datasets", nargs="+", default=["esol", "lipophilicity", "qm9_gap"])
    parser.add_argument("--output-dir", type=str, default="results/mechanistic_analysis")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for dataset in args.datasets:
        print(f"\n{'='*70}")
        print(f"Analyzing {dataset}...")
        print("="*70)

        try:
            smiles, conf_features, fps, targets = load_dataset(dataset)

            # Squeeze targets if needed
            if targets.ndim > 1:
                targets = targets.squeeze()

            results = compare_information_content(conf_features, fps, targets)

            # Analyze fingerprint bits
            if fps is not None:
                results["fp_bits"] = analyze_fingerprint_bits(fps, targets, smiles)

            print_summary(results, dataset)

            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    json_results[key] = {
                        k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in value.items()
                    }
                else:
                    json_results[key] = value

            all_results[dataset] = json_results

        except Exception as e:
            print(f"Error analyzing {dataset}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    with open(output_dir / "mechanistic_analysis.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}/mechanistic_analysis.json")


if __name__ == "__main__":
    main()
