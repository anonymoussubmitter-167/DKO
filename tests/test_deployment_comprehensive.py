#!/usr/bin/env python
"""
Comprehensive deployment test suite.

Tests every component of the DKO codebase to ensure it's ready for cluster deployment.
Run this before submitting jobs to the cluster.

Usage:
    python tests/test_deployment_comprehensive.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from typing import Dict, List


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"{title}")
    print('=' * 70)


def test_model_instantiation() -> Dict[str, bool]:
    """Test 1: All models can be instantiated and run forward pass."""
    print_section("TEST 1: MODEL INSTANTIATION & FORWARD PASS")

    from dko.experiments.main_benchmark import MODEL_REGISTRY

    feature_dim = 64
    batch_size = 2
    n_conformers = 5

    # Test data
    x = torch.randn(batch_size, n_conformers, feature_dim)
    energies = torch.randn(batch_size, n_conformers)

    # Test mu/sigma for DKO variants
    mu = torch.randn(batch_size, feature_dim)
    sigma_raw = torch.randn(batch_size, feature_dim, feature_dim)
    sigma = torch.bmm(sigma_raw, sigma_raw.transpose(1, 2))

    # GNN test data
    atomic_numbers = torch.randint(1, 10, (10,))
    positions = torch.randn(10, 3)
    batch = torch.repeat_interleave(torch.arange(batch_size), 5)

    results = {}

    for name, model_factory in MODEL_REGISTRY.items():
        try:
            # Create model
            if name in ['dko', 'dko_first_order']:
                model = model_factory(feature_dim=feature_dim, verbose=False)
            elif name in ['schnet', 'dimenet++', 'dimenetpp', 'spherenet', '3d_infomax', '3dinfomax', 'gem']:
                model = model_factory()
            else:
                model = model_factory(feature_dim=feature_dim)

            # Forward pass
            if name in ['dko']:
                out = model(mu, sigma, fit_pca=True)
            elif name == 'dko_first_order':
                out = model(mu)
            elif name in ['schnet', 'dimenet++', 'dimenetpp', 'spherenet', '3d_infomax', '3dinfomax', 'gem']:
                out = model(atomic_numbers, positions, batch)
            elif name in ['attention', 'attention_augmented']:
                out, _ = model(x)
            elif name == 'boltzmann_ensemble':
                out = model(x, energies)  # BoltzmannEnsemble requires energies
            else:
                out = model(x)

            # Check output shape
            assert out.shape[0] == batch_size, f"Batch size mismatch for {name}"
            assert out.shape[1] == 1, f"Output dim mismatch for {name}"

            param_count = sum(p.numel() for p in model.parameters())
            print(f"  {name:30s} PASS ({param_count:>8,} params)")
            results[name] = True

        except Exception as e:
            print(f"  {name:30s} FAIL: {e}")
            results[name] = False

    passed = sum(results.values())
    total = len(results)
    print(f"\nRESULT: {passed}/{total} models passed")

    return results


def test_model_registry() -> bool:
    """Test 2: Model registry has expected models."""
    print_section("TEST 2: MODEL REGISTRY COMPLETENESS")

    from dko.experiments.main_benchmark import MODEL_REGISTRY

    expected_models = {
        'dko', 'dko_first_order',
        'attention', 'attention_augmented',
        'deepsets', 'deepsets_augmented',
        'single_conformer', 'single_conformer_random', 'single_conformer_centroid',
        'mean_ensemble', 'boltzmann_ensemble',
        'mfa', 'mil',
        'schnet', 'dimenetpp', 'spherenet', '3dinfomax', 'gem'
    }

    registered = set(MODEL_REGISTRY.keys())

    print(f"  Expected models: {len(expected_models)}")
    print(f"  Registered models: {len(registered)}")

    missing = expected_models - registered
    extra = registered - expected_models

    if missing:
        print(f"\n  MISSING models: {missing}")
    if extra:
        print(f"\n  EXTRA models (OK): {extra}")

    passed = len(missing) == 0
    print(f"\nRESULT: {'PASS' if passed else 'FAIL'}")

    return passed


def test_experiments_registry() -> bool:
    """Test 3: Experiments registry has expected experiments."""
    print_section("TEST 3: EXPERIMENTS REGISTRY COMPLETENESS")

    sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
    from run_experiment import EXPERIMENTS

    expected_experiments = {
        'benchmark', 'decomposition', 'sample_efficiency', 'attention',
        'sketching', 'scc_validation', 'rep_vs_arch', 'negative_controls', 'decision_rule'
    }

    registered = set(EXPERIMENTS.keys())

    print(f"  Expected experiments: {len(expected_experiments)}")
    print(f"  Registered experiments: {len(registered)}")

    missing = expected_experiments - registered
    extra = registered - expected_experiments

    if missing:
        print(f"\n  MISSING experiments: {missing}")
    if extra:
        print(f"\n  EXTRA experiments (OK): {extra}")

    for name in sorted(registered):
        print(f"  - {name}")

    passed = len(missing) == 0
    print(f"\nRESULT: {'PASS' if passed else 'FAIL'}")

    return passed


def test_analysis_functions() -> bool:
    """Test 4: Statistical analysis functions work."""
    print_section("TEST 4: STATISTICAL ANALYSIS FUNCTIONS")

    from dko.analysis import (
        bonferroni_correction,
        holm_bonferroni_correction,
        benjamini_hochberg_correction,
        multiple_comparison_summary,
        StatisticalConsistencyChecker,
    )

    # Test p-value corrections
    p_values = [0.001, 0.01, 0.03, 0.05, 0.10, 0.50]

    try:
        bonf = bonferroni_correction(p_values)
        print(f"  Bonferroni correction: PASS ({bonf['n_significant']} significant)")

        holm = holm_bonferroni_correction(p_values)
        print(f"  Holm-Bonferroni correction: PASS ({holm['n_significant']} significant)")

        bh = benjamini_hochberg_correction(p_values)
        print(f"  Benjamini-Hochberg correction: PASS ({bh['n_significant']} significant)")

        summary = multiple_comparison_summary(p_values)
        print(f"  Multiple comparison summary: PASS (recommended: {summary['recommendation']})")

        # Test SCC
        scc = StatisticalConsistencyChecker()
        values1 = [0.5, 0.6, 0.55, 0.58]
        values2 = [0.4, 0.45, 0.42, 0.43]
        result = scc.check_significance(values1, values2)
        print(f"  SCC significance test: PASS (significant={result['significant']})")

        print(f"\nRESULT: PASS")
        return True

    except Exception as e:
        print(f"\nRESULT: FAIL - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_selection_variants() -> bool:
    """Test 5: SingleConformer selection variants work correctly."""
    print_section("TEST 5: CONFORMER SELECTION VARIANTS")

    from dko.models import SingleConformer

    feature_dim = 32
    batch_size = 3
    n_conformers = 5

    x = torch.randn(batch_size, n_conformers, feature_dim)
    energies = torch.randn(batch_size, n_conformers)

    selection_methods = ['lowest_energy', 'random', 'centroid', 'first']

    try:
        for method in selection_methods:
            model = SingleConformer(feature_dim=feature_dim, selection_method=method)
            out = model(x, energies)
            assert out.shape == (batch_size, 1)
            print(f"  {method:20s}: PASS")

        print(f"\nRESULT: PASS")
        return True

    except Exception as e:
        print(f"\nRESULT: FAIL - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scripts_imports() -> bool:
    """Test 6: All scripts can import dko without pip install."""
    print_section("TEST 6: SCRIPT IMPORTS (WITHOUT PIP INSTALL)")

    scripts_to_test = [
        'scripts/run_experiment.py',
        'scripts/analyze_results.py',
        'scripts/prepare_datasets.py',
    ]

    passed = True

    for script_path in scripts_to_test:
        script_file = PROJECT_ROOT / script_path
        try:
            # Read script and check for PYTHONPATH setup
            with open(script_file, 'r') as f:
                content = f.read()

            has_pythonpath = 'PROJECT_ROOT' in content and 'sys.path.insert' in content

            if has_pythonpath:
                print(f"  {script_path:40s}: PASS (has PYTHONPATH setup)")
            else:
                print(f"  {script_path:40s}: FAIL (missing PYTHONPATH setup)")
                passed = False

        except Exception as e:
            print(f"  {script_path:40s}: FAIL ({e})")
            passed = False

    print(f"\nRESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_gradient_flow() -> bool:
    """Test 7: Gradients flow correctly through models."""
    print_section("TEST 7: GRADIENT FLOW")

    from dko.models import DKO, AttentionAggregation, MultiInstanceLearning

    feature_dim = 64
    batch_size = 2
    n_conformers = 5

    x = torch.randn(batch_size, n_conformers, feature_dim)
    mu = torch.randn(batch_size, feature_dim)
    sigma_raw = torch.randn(batch_size, feature_dim, feature_dim)
    sigma = torch.bmm(sigma_raw, sigma_raw.transpose(1, 2))

    models_to_test = [
        ('DKO', DKO(feature_dim=feature_dim, verbose=False), (mu, sigma)),
        ('AttentionAggregation', AttentionAggregation(feature_dim=feature_dim), (x,)),
        ('MultiInstanceLearning', MultiInstanceLearning(feature_dim=feature_dim), (x,)),
    ]

    try:
        for name, model, inputs in models_to_test:
            model.train()

            # Forward pass
            if name == 'DKO':
                out = model(*inputs, fit_pca=True)
            elif name == 'AttentionAggregation':
                out, _ = model(*inputs)
            else:
                out = model(*inputs)

            # Backward pass
            loss = out.mean()
            loss.backward()

            # Check gradients exist
            has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)

            if has_grads:
                print(f"  {name:30s}: PASS (gradients flow)")
            else:
                print(f"  {name:30s}: FAIL (no gradients)")
                return False

        print(f"\nRESULT: PASS")
        return True

    except Exception as e:
        print(f"\nRESULT: FAIL - {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("DKO COMPREHENSIVE DEPLOYMENT TEST SUITE")
    print("=" * 70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Run all tests
    tests = [
        ("Model Instantiation", test_model_instantiation),
        ("Model Registry", test_model_registry),
        ("Experiments Registry", test_experiments_registry),
        ("Analysis Functions", test_analysis_functions),
        ("Selection Variants", test_selection_variants),
        ("Script Imports", test_scripts_imports),
        ("Gradient Flow", test_gradient_flow),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"\n{test_name} CRASHED: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False

    # Summary
    print_section("FINAL SUMMARY")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        symbol = "[PASS]" if result else "[FAIL]"
        print(f"  {symbol} {test_name:30s}: {status}")

    print(f"\n{'=' * 70}")
    print(f"OVERALL: {passed}/{total} tests passed")

    if passed == total:
        print("STATUS: [OK] READY FOR CLUSTER DEPLOYMENT")
    else:
        print("STATUS: [ERROR] ISSUES FOUND - FIX BEFORE DEPLOYMENT")

    print('=' * 70)

    # Exit with appropriate code
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
