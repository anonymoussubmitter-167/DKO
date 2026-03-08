#!/usr/bin/env python3
"""
Feature Quality Analysis (Experiment A).

Trains Ridge Regression + Random Forest on raw mean conformer features (mu)
for each dataset. If sklearn models get R² > 0.1 but neural nets get ~0,
the problem is training, not features. If sklearn also gets ~0, the geometric
features themselves aren't predictive.

This runs on CPU and is fast (~5 min total).
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Map dataset config names to precomputed directory names
DATASET_DIR_MAP = {
    "lipo": "lipophilicity",
}


def load_features_and_labels(dataset_name, root="data", max_feature_dim=1024, max_conformers=50):
    """Load precomputed conformers and compute mean features (mu)."""
    # Resolve directory name
    dir_name = DATASET_DIR_MAP.get(dataset_name, dataset_name)
    conformers_dir = Path(root) / "conformers" / dir_name

    splits = {}
    for split in ["train", "val", "test"]:
        split_path = conformers_dir / f"{split}.pkl"
        if not split_path.exists():
            raise FileNotFoundError(f"Not found: {split_path}")

        with open(split_path, "rb") as f:
            data = pickle.load(f)

        labels = np.array(data["labels"], dtype=np.float32)
        features_list = data["features"]

        # Compute mean features (mu) per molecule
        mu_list = []
        for mol_features in features_list:
            # mol_features is a list of arrays (one per conformer)
            conf_feats = []
            for conf_feat in mol_features[:max_conformers]:
                feat = np.array(conf_feat, dtype=np.float32)
                if len(feat) < max_feature_dim:
                    feat = np.pad(feat, (0, max_feature_dim - len(feat)))
                else:
                    feat = feat[:max_feature_dim]
                conf_feats.append(feat)
            conf_feats = np.stack(conf_feats)
            mu = conf_feats.mean(axis=0)
            mu_list.append(mu)

        mu_array = np.stack(mu_list)
        splits[split] = {"mu": mu_array, "labels": labels}

    return splits


def evaluate_sklearn_models(splits, task_type="regression"):
    """Train and evaluate Ridge Regression + Random Forest on mean features."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        r2_score,
        roc_auc_score,
        accuracy_score,
    )
    from sklearn.preprocessing import StandardScaler
    from scipy import stats

    X_train = splits["train"]["mu"]
    y_train = splits["train"]["labels"].flatten()
    X_val = splits["val"]["mu"]
    y_val = splits["val"]["labels"].flatten()
    X_test = splits["test"]["mu"]
    y_test = splits["test"]["labels"].flatten()

    # Combine train+val for final model
    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)

    # Standardize features
    scaler = StandardScaler()
    X_trainval_scaled = scaler.fit_transform(X_trainval)
    X_test_scaled = scaler.transform(X_test)

    # Also scale train-only for validation eval
    scaler_train = StandardScaler()
    X_train_scaled = scaler_train.fit_transform(X_train)
    X_val_scaled = scaler_train.transform(X_val)

    results = {}

    if task_type == "regression":
        models = {
            "ridge": Ridge(alpha=1.0),
            "ridge_alpha10": Ridge(alpha=10.0),
            "ridge_alpha100": Ridge(alpha=100.0),
            "random_forest": RandomForestRegressor(
                n_estimators=100, max_depth=10, n_jobs=-1, random_state=42
            ),
        }

        for name, model in models.items():
            # Train on train+val, test on test
            model.fit(X_trainval_scaled, y_trainval)
            y_pred = model.predict(X_test_scaled)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            pearson_r, _ = stats.pearsonr(y_pred, y_test)

            results[name] = {
                "rmse": float(rmse),
                "mae": float(mae),
                "r2": float(r2),
                "pearson": float(pearson_r),
            }

            # Also eval on val (using train-only scaler)
            model_val = type(model)(**model.get_params())
            model_val.fit(X_train_scaled, y_train)
            y_val_pred = model_val.predict(X_val_scaled)
            val_r2 = r2_score(y_val, y_val_pred)
            results[name]["val_r2"] = float(val_r2)

    else:  # classification
        models = {
            "logistic": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
            "logistic_C01": LogisticRegression(max_iter=1000, C=0.1, random_state=42),
            "random_forest": RandomForestClassifier(
                n_estimators=100, max_depth=10, n_jobs=-1, random_state=42
            ),
        }

        for name, model in models.items():
            model.fit(X_trainval_scaled, y_trainval.astype(int))

            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_proba = model.decision_function(X_test_scaled)

            y_pred = model.predict(X_test_scaled)

            acc = accuracy_score(y_test.astype(int), y_pred)
            try:
                auc = roc_auc_score(y_test.astype(int), y_proba)
            except Exception:
                auc = float("nan")

            results[name] = {
                "accuracy": float(acc),
                "auc": float(auc),
            }

    return results


def main():
    parser = argparse.ArgumentParser(description="Feature Quality Analysis")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to analyze (default: all precomputed)",
    )
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="results/feature_quality")
    args = parser.parse_args()

    # Discover available precomputed datasets
    conformers_dir = Path(args.data_root) / "conformers"
    available = sorted([d.name for d in conformers_dir.iterdir() if d.is_dir()])

    # Task type mapping
    classification_datasets = {"bace", "bbbp", "herg", "cyp3a4", "tox21"}

    if args.datasets:
        datasets = args.datasets
    else:
        datasets = available

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")

        # Determine task type
        task_type = "classification" if dataset in classification_datasets else "regression"

        try:
            # Use the directory name directly (matches precomputed dirs)
            splits = load_features_and_labels(dataset, root=args.data_root)
            n_train = len(splits["train"]["labels"])
            n_test = len(splits["test"]["labels"])
            feat_dim = splits["train"]["mu"].shape[1]
            print(f"  Train: {n_train}, Test: {n_test}, Feature dim: {feat_dim}")
            print(f"  Task: {task_type}")

            # Feature stats
            mu_train = splits["train"]["mu"]
            print(f"  Feature stats: mean={mu_train.mean():.4f}, std={mu_train.std():.4f}")
            print(f"  Feature range: [{mu_train.min():.4f}, {mu_train.max():.4f}]")
            nonzero_frac = (np.abs(mu_train) > 1e-6).mean()
            print(f"  Non-zero fraction: {nonzero_frac:.4f}")

            results = evaluate_sklearn_models(splits, task_type=task_type)
            all_results[dataset] = {
                "task_type": task_type,
                "n_train": n_train,
                "n_test": n_test,
                "feature_dim": feat_dim,
                "models": results,
            }

            # Print results
            for model_name, metrics in results.items():
                print(f"\n  {model_name}:")
                for k, v in metrics.items():
                    print(f"    {k}: {v:.4f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[dataset] = {"error": str(e)}

    # Save results
    results_path = output_dir / "feature_quality_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved to {results_path}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Are features predictive?")
    print(f"{'='*60}")
    for dataset, res in all_results.items():
        if "error" in res:
            print(f"  {dataset}: ERROR - {res['error']}")
            continue

        task_type = res["task_type"]
        models = res["models"]

        if task_type == "regression":
            best_r2 = max(m.get("r2", -999) for m in models.values())
            best_model = max(models.keys(), key=lambda k: models[k].get("r2", -999))
            verdict = "YES" if best_r2 > 0.1 else "WEAK" if best_r2 > 0.0 else "NO"
            print(f"  {dataset}: best R²={best_r2:.4f} ({best_model}) -> {verdict}")
        else:
            best_auc = max(m.get("auc", 0) for m in models.values())
            best_model = max(models.keys(), key=lambda k: models[k].get("auc", 0))
            verdict = "YES" if best_auc > 0.6 else "WEAK" if best_auc > 0.55 else "NO"
            print(f"  {dataset}: best AUC={best_auc:.4f} ({best_model}) -> {verdict}")


if __name__ == "__main__":
    main()
