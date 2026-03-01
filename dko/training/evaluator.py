"""
Comprehensive evaluation metrics for molecular property prediction.

Metrics from research plan:
- Regression: RMSE, MAE, Pearson correlation, R², Spearman
- Classification: AUC-ROC, AUC-PR, Accuracy, Precision, Recall, F1
- Multi-task: Mean metrics, per-task metrics
- Statistical tests: Paired t-test, Wilcoxon, bootstrap
- Stratified evaluation by SCC quartile
- Bootstrap confidence intervals
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
from scipy import stats

# Optional imports
try:
    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        r2_score,
        roc_auc_score,
        average_precision_score,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        precision_recall_fscore_support,
        confusion_matrix,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    task_type: str = "regression",
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        predictions: Model predictions
        targets: Ground truth labels
        task_type: 'regression' or 'classification'
        threshold: Classification threshold for binary classification

    Returns:
        Dictionary of metric names to values
    """
    if task_type == "regression":
        return compute_regression_metrics(predictions, targets)
    else:
        return compute_classification_metrics(predictions, targets, threshold)


def compute_regression_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    """
    Compute regression metrics.

    Metrics from research plan:
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
    - R² (Coefficient of determination)
    - Pearson correlation
    - Spearman correlation
    """
    # Flatten arrays
    predictions = np.asarray(predictions).flatten()
    targets = np.asarray(targets).flatten()

    # Remove NaN values
    valid_mask = ~(np.isnan(predictions) | np.isnan(targets))
    predictions = predictions[valid_mask]
    targets = targets[valid_mask]

    if len(predictions) == 0:
        return {
            "rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
            "pearson": np.nan,
            "pearson_p": np.nan,
            "spearman": np.nan,
            "spearman_p": np.nan,
        }

    # Core metrics
    if SKLEARN_AVAILABLE:
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
    else:
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        mae = np.mean(np.abs(predictions - targets))
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Correlation metrics
    if len(predictions) > 1:
        pearson_r, pearson_p = stats.pearsonr(predictions, targets)
        spearman_r, spearman_p = stats.spearmanr(predictions, targets)
    else:
        pearson_r, pearson_p = np.nan, np.nan
        spearman_r, spearman_p = np.nan, np.nan

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "pearson": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman": float(spearman_r),
        "spearman_p": float(spearman_p),
    }


def compute_classification_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Metrics from research plan:
    - AUC-ROC (Area Under ROC Curve)
    - AUC-PR (Area Under Precision-Recall Curve)
    - Accuracy
    - Precision, Recall, F1
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    # Handle multi-task classification
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        return compute_multitask_classification_metrics(predictions, targets, threshold)

    # Binary classification
    predictions = predictions.flatten()
    targets = targets.flatten()

    # Apply sigmoid if predictions are logits
    if predictions.min() < 0 or predictions.max() > 1:
        probs = 1 / (1 + np.exp(-np.clip(predictions, -500, 500)))
    else:
        probs = predictions

    # Remove NaN values
    valid_mask = ~(np.isnan(probs) | np.isnan(targets))
    probs = probs[valid_mask]
    targets = targets[valid_mask]

    if len(probs) == 0:
        return {
            "auc": np.nan,
            "auc_pr": np.nan,
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "specificity": np.nan,
        }

    # Binary predictions
    binary_preds = (probs >= threshold).astype(int)
    targets_int = targets.astype(int)

    # AUC-ROC
    try:
        if len(np.unique(targets_int)) > 1 and SKLEARN_AVAILABLE:
            auc = roc_auc_score(targets_int, probs)
        else:
            auc = np.nan
    except Exception:
        auc = np.nan

    # AUC-PR
    try:
        if len(np.unique(targets_int)) > 1 and SKLEARN_AVAILABLE:
            auc_pr = average_precision_score(targets_int, probs)
        else:
            auc_pr = np.nan
    except Exception:
        auc_pr = np.nan

    # Other metrics
    if SKLEARN_AVAILABLE:
        accuracy = accuracy_score(targets_int, binary_preds)
        precision = precision_score(targets_int, binary_preds, zero_division=0)
        recall = recall_score(targets_int, binary_preds, zero_division=0)
        f1 = f1_score(targets_int, binary_preds, zero_division=0)
    else:
        tp = np.sum((binary_preds == 1) & (targets_int == 1))
        tn = np.sum((binary_preds == 0) & (targets_int == 0))
        fp = np.sum((binary_preds == 1) & (targets_int == 0))
        fn = np.sum((binary_preds == 0) & (targets_int == 1))

        accuracy = (tp + tn) / len(targets_int) if len(targets_int) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Specificity
    tn = np.sum((binary_preds == 0) & (targets_int == 0))
    fp = np.sum((binary_preds == 1) & (targets_int == 0))
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "auc": float(auc),
        "auc_pr": float(auc_pr),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
    }


def compute_multitask_classification_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute metrics for multi-task classification."""
    n_tasks = predictions.shape[1]

    task_aucs = []
    task_aps = []
    task_accs = []

    for i in range(n_tasks):
        pred_i = predictions[:, i]
        target_i = targets[:, i]

        # Skip tasks with missing labels
        mask = ~np.isnan(target_i)
        if mask.sum() == 0:
            continue

        pred_i = pred_i[mask]
        target_i = target_i[mask]

        # Apply sigmoid if needed
        if pred_i.min() < 0 or pred_i.max() > 1:
            pred_i = 1 / (1 + np.exp(-np.clip(pred_i, -500, 500)))

        if len(np.unique(target_i)) < 2:
            continue

        # AUC-ROC
        try:
            auc = roc_auc_score(target_i, pred_i) if SKLEARN_AVAILABLE else np.nan
            task_aucs.append(auc)
        except Exception:
            pass

        # AUC-PR
        try:
            ap = average_precision_score(target_i, pred_i) if SKLEARN_AVAILABLE else np.nan
            task_aps.append(ap)
        except Exception:
            pass

        # Accuracy
        pred_binary = (pred_i > threshold).astype(int)
        acc = np.mean(pred_binary == target_i.astype(int))
        task_accs.append(acc)

    return {
        "mean_auc": float(np.mean(task_aucs)) if task_aucs else np.nan,
        "mean_auc_pr": float(np.mean(task_aps)) if task_aps else np.nan,
        "mean_accuracy": float(np.mean(task_accs)) if task_accs else np.nan,
        "n_valid_tasks": len(task_aucs),
        "n_total_tasks": n_tasks,
    }


def compute_confidence_intervals(
    values: List[float],
    confidence_level: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Compute confidence interval for a list of values.

    Args:
        values: List of metric values (e.g., from multiple seeds)
        confidence_level: Confidence level (default 0.95)

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    values = np.array(values)
    n = len(values)

    if n == 0:
        return np.nan, np.nan, np.nan

    mean = np.mean(values)

    if n == 1:
        return mean, mean, mean

    std = np.std(values, ddof=1)

    # t-distribution critical value
    alpha = 1 - confidence_level
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)

    margin = t_crit * std / np.sqrt(n)
    lower = mean - margin
    upper = mean + margin

    return float(mean), float(lower), float(upper)


class Evaluator:
    """
    Comprehensive evaluator for molecular property prediction.

    Features from research plan:
    - All regression and classification metrics
    - Statistical significance testing
    - Bootstrap confidence intervals
    - Per-molecule analysis
    - Stratified evaluation by SCC quartile
    """

    def __init__(
        self,
        task_type: str = "regression",
        primary_metric: Optional[str] = None,
        device: Optional[str] = None,
        bootstrap_n_samples: int = 1000,
        confidence_level: float = 0.95,
    ):
        """
        Initialize evaluator.

        Args:
            task_type: 'regression' or 'classification'
            primary_metric: Primary metric for comparison
            device: Device for evaluation (auto-detect if None)
            bootstrap_n_samples: Number of bootstrap samples for CI
            confidence_level: Confidence level for intervals
        """
        self.task_type = task_type
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.bootstrap_n_samples = bootstrap_n_samples
        self.confidence_level = confidence_level

        if primary_metric is None:
            self.primary_metric = "rmse" if task_type == "regression" else "auc"
        else:
            self.primary_metric = primary_metric

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        return_predictions: bool = False,
        compute_ci: bool = False,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate model on a dataset.

        Args:
            model: Model to evaluate
            data_loader: Data loader
            return_predictions: Whether to return predictions
            compute_ci: Whether to compute bootstrap confidence intervals
            verbose: Whether to print progress

        Returns:
            Dictionary of metrics and optionally predictions
        """
        model.eval()
        model = model.to(self.device)

        all_preds = []
        all_labels = []
        all_smiles = []
        all_scc = []

        iterator = data_loader
        if verbose and TQDM_AVAILABLE:
            iterator = tqdm(data_loader, desc="Evaluating")

        for batch in iterator:
            # Forward pass
            predictions = self._forward_batch(model, batch)

            # Get labels
            labels = batch.get("label", batch.get("labels"))
            if labels is not None:
                all_labels.append(labels.cpu().numpy())

            all_preds.append(predictions)

            # Store metadata if available
            if 'smiles' in batch:
                smiles = batch['smiles']
                if isinstance(smiles, (list, tuple)):
                    all_smiles.extend(smiles)
                else:
                    all_smiles.extend(smiles.tolist() if hasattr(smiles, 'tolist') else list(smiles))

            if 'scc' in batch:
                scc = batch['scc']
                if isinstance(scc, torch.Tensor):
                    scc = scc.cpu().numpy()
                all_scc.append(scc)

        # Concatenate predictions
        predictions = np.concatenate(all_preds, axis=0)

        if all_labels:
            labels = np.concatenate(all_labels, axis=0)
            metrics = compute_metrics(predictions, labels, self.task_type)
            metrics['n_samples'] = len(labels)

            # Compute bootstrap confidence intervals
            if compute_ci:
                ci_metrics = self._compute_bootstrap_ci(predictions, labels)
                metrics.update(ci_metrics)
        else:
            labels = None
            metrics = {'n_samples': len(predictions)}

        # Build result
        if return_predictions:
            result = {
                'metrics': metrics,
                'predictions': predictions,
                'labels': labels,
            }

            if all_smiles:
                result['smiles'] = all_smiles

            if all_scc:
                result['scc'] = np.concatenate(all_scc, axis=0)

            return result

        return metrics

    def _is_dko_model(self, model: nn.Module) -> bool:
        """Check if model is a DKO variant that needs mu/sigma input."""
        # Handle DataParallel wrapped models
        actual_model = model.module if isinstance(model, nn.DataParallel) else model
        model_class_name = actual_model.__class__.__name__
        return model_class_name in [
            'DKO', 'DKOFirstOrder', 'DKOFull', 'DKONoPSD',
            'DKOEigenspectrum', 'DKOScalarInvariants', 'DKOLowRank',
            'DKOGatedFusion', 'DKOResidual', 'DKOCrossAttention', 'DKOSCCRouter',
        ]

    def _compute_mu_sigma(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Compute mu and sigma from conformer features for DKO models."""
        batch_size, n_conf, feat_dim = features.shape

        # Check for NaN/Inf in input features
        if torch.isnan(features).any() or torch.isinf(features).any():
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        # Per-sample normalization across conformers only (dim=1).
        # Normalizing across features too (dim=(1,2)) destroys inter-feature variance
        # that sigma is meant to capture. dim=1 preserves feature-wise structure.
        feat_mean = features.mean(dim=1, keepdim=True)
        feat_std = features.std(dim=1, keepdim=True).clamp(min=1e-6)
        features = (features - feat_mean) / feat_std

        # Create mask if not provided
        if mask is None:
            mask = torch.ones(batch_size, n_conf, dtype=torch.bool, device=features.device)

        # Compute weights
        valid_counts = mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        weights = mask.float() / valid_counts
        weights_expanded = weights.unsqueeze(-1)

        # Compute mu
        mu = (features * weights_expanded).sum(dim=1)

        # Compute sigma
        centered = features - mu.unsqueeze(1)
        centered = centered * mask.unsqueeze(-1).float()

        # Clamp centered values to prevent extreme covariances
        centered = torch.clamp(centered, min=-10.0, max=10.0)

        weighted_centered = centered * weights_expanded.sqrt()
        sigma = torch.bmm(weighted_centered.transpose(1, 2), weighted_centered)

        # Add regularization to diagonal for numerical stability.
        # 1e-2 is appropriate for geometric feature scales (~0.1-10.0) to prevent
        # near-singular covariance matrices.
        eye = torch.eye(feat_dim, device=sigma.device, dtype=sigma.dtype)
        sigma = sigma + 1e-2 * eye.unsqueeze(0)

        return mu, sigma

    def _forward_batch(self, model: nn.Module, batch: Dict) -> np.ndarray:
        """Forward pass for a batch."""
        # Check for DKO format (mu, sigma)
        if 'mu' in batch and 'sigma' in batch:
            mu = batch['mu'].to(self.device)
            sigma = batch['sigma'].to(self.device)
            outputs = model(mu, sigma, fit_pca=False)

        # Check for baseline format (features)
        elif 'features' in batch:
            features = batch['features'].to(self.device)
            mask = batch.get('mask')
            weights = batch.get('weights', batch.get('boltzmann_weights'))

            if mask is not None:
                mask = mask.to(self.device)
            if weights is not None:
                weights = weights.to(self.device)

            # Check if model is DKO variant - needs mu/sigma computation
            if self._is_dko_model(model):
                mu, sigma = self._compute_mu_sigma(features, mask)
                outputs = model(mu, sigma, fit_pca=False)
            else:
                # Handle different baseline model signatures
                try:
                    if weights is not None:
                        outputs = model(features, weights, mask=mask)
                    elif mask is not None:
                        outputs = model(features, mask=mask)
                    else:
                        outputs = model(features)
                except TypeError:
                    # Model doesn't accept mask/weights
                    outputs = model(features)

            # Handle tuple output
            if isinstance(outputs, tuple):
                outputs = outputs[0]

        else:
            raise ValueError("Unknown batch format. Expected 'mu'/'sigma' or 'features'")

        # Apply sigmoid for classification
        if self.task_type == 'classification':
            outputs = torch.sigmoid(outputs)

        return outputs.cpu().numpy()

    def _compute_bootstrap_ci(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """Compute bootstrap confidence intervals for key metrics."""
        ci_metrics = {}

        predictions = predictions.flatten()
        labels = labels.flatten()

        if self.task_type == 'regression':
            # RMSE CI
            rmse_lower, rmse_upper = self._bootstrap_metric_ci(
                predictions, labels, metric='rmse'
            )
            ci_metrics['rmse_ci_lower'] = rmse_lower
            ci_metrics['rmse_ci_upper'] = rmse_upper

            # MAE CI
            mae_lower, mae_upper = self._bootstrap_metric_ci(
                predictions, labels, metric='mae'
            )
            ci_metrics['mae_ci_lower'] = mae_lower
            ci_metrics['mae_ci_upper'] = mae_upper

        else:
            # AUC CI
            auc_lower, auc_upper = self._bootstrap_metric_ci(
                predictions, labels, metric='auc'
            )
            ci_metrics['auc_ci_lower'] = auc_lower
            ci_metrics['auc_ci_upper'] = auc_upper

        return ci_metrics

    def _bootstrap_metric_ci(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        metric: str,
    ) -> Tuple[float, float]:
        """Compute bootstrap CI for a single metric."""
        n_samples = len(predictions)
        metric_values = []

        for _ in range(self.bootstrap_n_samples):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            pred_sample = predictions[indices]
            label_sample = labels[indices]

            try:
                if metric == 'rmse':
                    value = np.sqrt(np.mean((pred_sample - label_sample) ** 2))
                elif metric == 'mae':
                    value = np.mean(np.abs(pred_sample - label_sample))
                elif metric == 'auc':
                    if len(np.unique(label_sample)) < 2:
                        continue
                    value = roc_auc_score(label_sample, pred_sample) if SKLEARN_AVAILABLE else np.nan
                else:
                    continue

                if not np.isnan(value):
                    metric_values.append(value)
            except Exception:
                continue

        if not metric_values:
            return np.nan, np.nan

        alpha = 1 - self.confidence_level
        lower = np.percentile(metric_values, 100 * alpha / 2)
        upper = np.percentile(metric_values, 100 * (1 - alpha / 2))

        return float(lower), float(upper)

    def stratified_evaluation(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        stratify_by: str = 'scc',
        n_bins: int = 4,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate model stratified by a variable (e.g., SCC quartiles).

        This is important for Experiment 3 in the research plan.

        Args:
            model: Model to evaluate
            data_loader: Data loader
            stratify_by: Variable to stratify by ('scc')
            n_bins: Number of bins (4 for quartiles)
            verbose: Whether to print progress

        Returns:
            Metrics per bin
        """
        # Get predictions and stratification variable
        results = self.evaluate(
            model, data_loader,
            return_predictions=True,
            verbose=verbose
        )

        if stratify_by not in results:
            raise ValueError(f"Stratification variable '{stratify_by}' not in data")

        stratify_values = results[stratify_by]
        predictions = results['predictions']
        labels = results['labels']

        # Create bins based on percentiles
        bin_edges = np.percentile(stratify_values, np.linspace(0, 100, n_bins + 1))
        bin_indices = np.digitize(stratify_values, bin_edges[1:-1])

        # Evaluate per bin
        stratified_metrics = {}

        for bin_idx in range(n_bins):
            mask = (bin_indices == bin_idx)

            if mask.sum() == 0:
                continue

            bin_predictions = predictions[mask]
            bin_labels = labels[mask]

            bin_metrics = compute_metrics(bin_predictions, bin_labels, self.task_type)
            bin_metrics['n_samples'] = int(mask.sum())
            bin_metrics['bin_range'] = (
                float(stratify_values[mask].min()),
                float(stratify_values[mask].max())
            )

            stratified_metrics[f'quartile_{bin_idx + 1}'] = bin_metrics

        return {
            'overall': results['metrics'],
            'stratified': stratified_metrics,
            'stratify_by': stratify_by,
            'n_bins': n_bins,
        }

    def evaluate_multi_seed(
        self,
        model_class: type,
        model_kwargs: Dict,
        train_fn: Callable,
        data_loaders: Dict[str, DataLoader],
        seeds: List[int] = [42, 123, 456],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate model across multiple random seeds.

        Args:
            model_class: Model class to instantiate
            model_kwargs: Arguments for model initialization
            train_fn: Training function
            data_loaders: Dictionary with 'train', 'val', 'test' loaders
            seeds: List of random seeds

        Returns:
            Dictionary with aggregated statistics
        """
        all_results = []

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = model_class(**model_kwargs)
            train_fn(model, data_loaders["train"], data_loaders["val"])
            test_metrics = self.evaluate(model, data_loaders["test"], verbose=False)
            all_results.append(test_metrics)

        # Aggregate results
        aggregated = {}
        metric_names = [k for k in all_results[0].keys() if isinstance(all_results[0][k], (int, float))]

        for metric in metric_names:
            values = [r[metric] for r in all_results if not np.isnan(r.get(metric, np.nan))]
            if values:
                mean, ci_lower, ci_upper = compute_confidence_intervals(values)
                aggregated[metric] = {
                    "mean": mean,
                    "std": float(np.std(values)),
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "values": values,
                }

        return aggregated

    def significance_test(
        self,
        model1_values: List[float],
        model2_values: List[float],
        test: str = "paired_t",
    ) -> Tuple[float, float]:
        """
        Perform statistical significance test between two models.

        Args:
            model1_values: Metric values for model 1 across seeds
            model2_values: Metric values for model 2 across seeds
            test: Type of test ('paired_t', 'wilcoxon')

        Returns:
            Tuple of (statistic, p_value)
        """
        model1_values = np.array(model1_values)
        model2_values = np.array(model2_values)

        if test == "paired_t":
            stat, p_value = stats.ttest_rel(model1_values, model2_values)
        elif test == "wilcoxon":
            stat, p_value = stats.wilcoxon(model1_values, model2_values)
        else:
            raise ValueError(f"Unknown test: {test}")

        return float(stat), float(p_value)

    def compare_models(
        self,
        results_dict: Dict[str, Dict[str, Dict]],
        baseline_name: str = "single_conformer",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple models against a baseline.

        Args:
            results_dict: Dictionary mapping model names to their results
            baseline_name: Name of baseline model

        Returns:
            Dictionary with comparison statistics
        """
        comparisons = {}
        baseline_values = results_dict[baseline_name][self.primary_metric]["values"]

        for model_name, results in results_dict.items():
            if model_name == baseline_name:
                continue

            model_values = results[self.primary_metric]["values"]

            # Compute improvement
            baseline_mean = np.mean(baseline_values)
            model_mean = np.mean(model_values)

            if self.task_type == "regression":
                # Lower is better for RMSE
                improvement = (baseline_mean - model_mean) / baseline_mean * 100
            else:
                # Higher is better for AUC
                improvement = (model_mean - baseline_mean) / baseline_mean * 100

            # Significance test
            stat, p_value = self.significance_test(baseline_values, model_values)

            comparisons[model_name] = {
                "improvement_percent": float(improvement),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "model_mean": float(model_mean),
                "baseline_mean": float(baseline_mean),
            }

        return comparisons

    def save_predictions(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        output_path: Union[str, Path],
        smiles: Optional[List[str]] = None,
        scc: Optional[np.ndarray] = None,
    ):
        """Save predictions to CSV for analysis."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required for save_predictions")

        predictions = np.asarray(predictions).flatten()
        labels = np.asarray(labels).flatten()

        data = {
            'prediction': predictions,
            'label': labels,
            'error': predictions - labels,
            'abs_error': np.abs(predictions - labels),
        }

        if smiles is not None:
            data['smiles'] = smiles

        if scc is not None:
            data['scc'] = np.asarray(scc).flatten()

        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)


# Convenience functions
def paired_t_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
) -> Tuple[float, float]:
    """Perform paired t-test between two models."""
    t_stat, p_value = stats.ttest_rel(errors_a, errors_b)
    return float(t_stat), float(p_value)


def wilcoxon_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
) -> Tuple[float, float]:
    """Perform Wilcoxon signed-rank test between two models."""
    statistic, p_value = stats.wilcoxon(errors_a, errors_b)
    return float(statistic), float(p_value)


if __name__ == "__main__":
    # Test evaluator
    print("Testing Evaluator...")

    # Create dummy data
    n_samples = 100
    D = 50

    mu = torch.randn(n_samples, D)
    sigma = torch.randn(n_samples, D, D)
    sigma = torch.bmm(sigma, sigma.transpose(1, 2))
    labels = torch.randn(n_samples, 1)

    def collate_fn(batch):
        mu, sigma, labels = zip(*batch)
        return {
            'mu': torch.stack(mu),
            'sigma': torch.stack(sigma),
            'label': torch.stack(labels),
        }

    from torch.utils.data import TensorDataset
    dataset = TensorDataset(mu, sigma, labels)
    data_loader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn)

    # Create simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.fc = nn.Linear(input_dim, output_dim)

        def forward(self, mu, sigma=None, fit_pca=False):
            return self.fc(mu)

    model = SimpleModel(D, 1)
    model.eval()

    # Test evaluator
    evaluator = Evaluator(task_type='regression', device='cpu')
    metrics = evaluator.evaluate(model, data_loader, verbose=False)

    print("\n[OK] Evaluator test passed!")
    print(f"  Metrics: {list(metrics.keys())}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  Pearson: {metrics['pearson']:.4f}")
    print(f"  R²: {metrics['r2']:.4f}")

    # Test with predictions returned
    results = evaluator.evaluate(model, data_loader, return_predictions=True, verbose=False)
    print(f"  Predictions shape: {results['predictions'].shape}")

    # Test classification metrics
    print("\nTesting classification metrics...")
    pred_binary = np.random.rand(100)
    labels_binary = np.random.randint(0, 2, 100)
    cls_metrics = compute_classification_metrics(pred_binary, labels_binary)
    print(f"  AUC-ROC: {cls_metrics['auc']:.4f}")
    print(f"  AUC-PR: {cls_metrics['auc_pr']:.4f}")
    print(f"  Accuracy: {cls_metrics['accuracy']:.4f}")

    print("\n[OK] All Evaluator tests passed!")
