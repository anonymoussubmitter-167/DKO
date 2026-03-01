#!/usr/bin/env python3
"""Error analysis by molecular properties.

Stratifies prediction errors by molecule size (heavy atoms), flexibility
(rotatable bonds), molecular weight, and polarity (TPSA).
Shows WHERE conformer features help most.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime

import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator
from sklearn.metrics import mean_squared_error, r2_score

# ── Config ──
DATASETS = ["esol", "freesolv", "lipophilicity"]
SEED = 42
OUTPUT_DIR = Path("results/error_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(2048)
    return _FP_GEN.GetFingerprintAsNumPy(mol).astype(np.float64)


def compute_conformer_stats(features, max_dim=256):
    padded = []
    for conf_feat in features:
        cf = np.array(conf_feat).flatten()
        if len(cf) < max_dim:
            cf = np.pad(cf, (0, max_dim - len(cf)))
        else:
            cf = cf[:max_dim]
        padded.append(cf)
    conformers = np.array(padded)
    if len(conformers) == 0:
        return np.zeros(max_dim), np.zeros(5)
    mu = np.mean(conformers, axis=0)
    if len(conformers) < 2:
        return mu, np.zeros(5)
    centered = conformers - mu
    variances = np.mean(centered ** 2, axis=0)
    total_var = np.sum(variances)
    max_var = np.max(variances)
    mean_var = np.mean(variances)
    sorted_var = np.sort(variances)[::-1]
    top5_var = np.sum(sorted_var[:5])
    effective_rank = float(np.sum(variances > 0.01 * total_var)) if total_var > 0 else 0
    return mu, np.array([total_var, max_var, mean_var, top5_var, effective_rank])


def compute_mol_properties(smiles):
    """Compute molecular properties for stratification."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "heavy_atoms": 0, "rotatable_bonds": 0,
            "mw": 0.0, "tpsa": 0.0, "logp": 0.0,
            "rings": 0, "hbd": 0, "hba": 0,
        }
    return {
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "mw": Descriptors.MolWt(mol),
        "tpsa": Descriptors.TPSA(mol),
        "logp": Descriptors.MolLogP(mol),
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
    }


def load_split(dataset_name, split):
    path = Path(f"data/conformers/{dataset_name}/{split}.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["smiles"], np.array([float(y) for y in data["labels"]]), data["features"]


def precompute_all(smiles_list, features_list):
    n = len(smiles_list)
    fp = np.zeros((n, 2048))
    mu = np.zeros((n, 256))
    sigma = np.zeros((n, 5))
    for i, (smi, feats) in enumerate(zip(smiles_list, features_list)):
        fp[i] = smiles_to_fingerprint(smi)
        mu[i], sigma[i] = compute_conformer_stats(feats, max_dim=256)
    return fp, mu, sigma


def train_and_predict(train_X, train_y, test_X, seed):
    model = xgb.XGBRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=seed,
        n_jobs=4, verbosity=0, tree_method='hist',
    )
    model.fit(train_X, train_y)
    return model.predict(test_X)


def quartile_analysis(errors_fp, errors_hybrid, property_values, property_name):
    """Stratify errors by quartiles of a molecular property."""
    q25, q50, q75 = np.percentile(property_values, [25, 50, 75])
    quartiles = {
        f"Q1 (≤{q25:.1f})": property_values <= q25,
        f"Q2 ({q25:.1f}-{q50:.1f})": (property_values > q25) & (property_values <= q50),
        f"Q3 ({q50:.1f}-{q75:.1f})": (property_values > q50) & (property_values <= q75),
        f"Q4 (>{q75:.1f})": property_values > q75,
    }

    results = []
    for q_name, mask in quartiles.items():
        if mask.sum() == 0:
            continue
        fp_rmse = float(np.sqrt(np.mean(errors_fp[mask] ** 2)))
        hybrid_rmse = float(np.sqrt(np.mean(errors_hybrid[mask] ** 2)))
        improvement = (fp_rmse - hybrid_rmse) / fp_rmse * 100 if fp_rmse > 0 else 0
        results.append({
            "quartile": q_name,
            "n_molecules": int(mask.sum()),
            "fp_rmse": fp_rmse,
            "hybrid_rmse": hybrid_rmse,
            "improvement_pct": improvement,
        })
    return results


def main():
    print("Error Analysis by Molecular Properties")
    print("=" * 60)

    all_results = {}

    for dataset in DATASETS:
        print(f"\n=== {dataset.upper()} ===")

        # Load data
        train_smi, train_y, train_feats = load_split(dataset, "train")
        val_smi, val_y, val_feats = load_split(dataset, "val")
        test_smi, test_y, test_feats = load_split(dataset, "test")

        # Combine train+val
        trainval_smi = list(train_smi) + list(val_smi)
        trainval_y = np.concatenate([train_y, val_y])
        trainval_feats = list(train_feats) + list(val_feats)

        # Precompute features
        tv_fp, tv_mu, tv_sigma = precompute_all(trainval_smi, trainval_feats)
        te_fp, te_mu, te_sigma = precompute_all(test_smi, test_feats)

        # Train both models
        fp_pred = train_and_predict(tv_fp, trainval_y, te_fp, SEED)
        hybrid_X_train = np.hstack([tv_fp, tv_mu, tv_sigma])
        hybrid_X_test = np.hstack([te_fp, te_mu, te_sigma])
        hybrid_pred = train_and_predict(hybrid_X_train, trainval_y, hybrid_X_test, SEED)

        # Compute errors
        errors_fp = test_y - fp_pred
        errors_hybrid = test_y - hybrid_pred

        # Overall
        fp_rmse = float(np.sqrt(np.mean(errors_fp ** 2)))
        hybrid_rmse = float(np.sqrt(np.mean(errors_hybrid ** 2)))
        overall_imp = (fp_rmse - hybrid_rmse) / fp_rmse * 100
        print(f"  Overall: FP RMSE={fp_rmse:.4f}, Hybrid RMSE={hybrid_rmse:.4f}, "
              f"Improvement={overall_imp:+.1f}%")

        # Compute molecular properties for test set
        test_props = [compute_mol_properties(smi) for smi in test_smi]
        prop_arrays = {
            key: np.array([p[key] for p in test_props])
            for key in ["heavy_atoms", "rotatable_bonds", "mw", "tpsa", "logp", "rings"]
        }

        # Quartile analysis for each property
        dataset_results = {"overall": {
            "fp_rmse": fp_rmse,
            "hybrid_rmse": hybrid_rmse,
            "improvement_pct": overall_imp,
            "n_test": len(test_y),
        }}

        stratify_props = ["heavy_atoms", "rotatable_bonds", "mw", "tpsa", "logp"]
        for prop_name in stratify_props:
            prop_vals = prop_arrays[prop_name]
            quartile_results = quartile_analysis(
                errors_fp, errors_hybrid, prop_vals, prop_name
            )
            dataset_results[prop_name] = quartile_results

            print(f"\n  Stratified by {prop_name}:")
            for qr in quartile_results:
                print(f"    {qr['quartile']:25s} n={qr['n_molecules']:3d}  "
                      f"FP={qr['fp_rmse']:.4f}  Hybrid={qr['hybrid_rmse']:.4f}  "
                      f"Δ={qr['improvement_pct']:+.1f}%")

        # Correlation between conformer variance and hybrid improvement
        # (per-molecule analysis)
        abs_err_fp = np.abs(errors_fp)
        abs_err_hybrid = np.abs(errors_hybrid)
        improvement_per_mol = abs_err_fp - abs_err_hybrid  # Positive = hybrid better

        # Does flexibility predict where hybrid helps?
        from scipy.stats import pearsonr, spearmanr
        for prop_name in ["rotatable_bonds", "heavy_atoms", "mw"]:
            prop_vals = prop_arrays[prop_name]
            r_pearson, p_pearson = pearsonr(prop_vals, improvement_per_mol)
            r_spearman, p_spearman = spearmanr(prop_vals, improvement_per_mol)
            dataset_results[f"{prop_name}_correlation"] = {
                "pearson_r": float(r_pearson),
                "pearson_p": float(p_pearson),
                "spearman_r": float(r_spearman),
                "spearman_p": float(p_spearman),
            }
            sig = "*" if p_spearman < 0.05 else ""
            print(f"\n  Correlation({prop_name}, hybrid_improvement):")
            print(f"    Spearman r={r_spearman:.3f}, p={p_spearman:.4f} {sig}")

        all_results[dataset] = dataset_results

    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {"datasets": DATASETS, "seed": SEED},
        "results": all_results,
    }
    with open(OUTPUT_DIR / "error_analysis_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
