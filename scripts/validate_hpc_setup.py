#!/usr/bin/env python
"""
HPC Setup Validation Script.

Validates that all components are properly configured for HPC deployment.
Run this before submitting jobs to catch configuration issues early.

Usage:
    python scripts/validate_hpc_setup.py
    python scripts/validate_hpc_setup.py --config configs/experiments/dko_esol.yaml
    python scripts/validate_hpc_setup.py --full  # Run more comprehensive tests
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any
import importlib
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class ValidationResult:
    def __init__(self, name: str, passed: bool, message: str = "", details: str = ""):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details

    def __str__(self):
        icon = "[OK]" if self.passed else "[FAIL]"
        result = f"{icon} {self.name}"
        if self.message:
            result += f": {self.message}"
        if self.details:
            result += f"\n    {self.details}"
        return result


class HPCValidator:
    """Validate HPC setup for DKO training."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[ValidationResult] = []

    def add_result(self, name: str, passed: bool, message: str = "", details: str = ""):
        self.results.append(ValidationResult(name, passed, message, details))
        if self.verbose:
            print(self.results[-1])

    def check_python_version(self) -> bool:
        """Check Python version is 3.8+."""
        version = sys.version_info
        passed = version.major == 3 and version.minor >= 8
        self.add_result(
            "Python version",
            passed,
            f"{version.major}.{version.minor}.{version.micro}",
            "" if passed else "Python 3.8+ required"
        )
        return passed

    def check_pytorch(self) -> bool:
        """Check PyTorch installation."""
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            cuda_info = ""
            if cuda_available:
                cuda_info = f"CUDA {torch.version.cuda}, {torch.cuda.device_count()} GPU(s)"
            else:
                cuda_info = "CPU only (no CUDA)"

            self.add_result(
                "PyTorch",
                True,
                f"v{torch.__version__}",
                cuda_info
            )

            if cuda_available:
                self.add_result(
                    "GPU",
                    True,
                    torch.cuda.get_device_name(0)
                )
            else:
                self.add_result(
                    "GPU",
                    False,
                    "CUDA not available",
                    "Training will be slow on CPU"
                )

            return True
        except ImportError as e:
            self.add_result("PyTorch", False, str(e))
            return False

    def check_rdkit(self) -> bool:
        """Check RDKit installation."""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            mol = Chem.MolFromSmiles("CCO")
            passed = mol is not None
            self.add_result(
                "RDKit",
                passed,
                "OK" if passed else "Failed to parse test molecule"
            )
            return passed
        except ImportError as e:
            self.add_result("RDKit", False, str(e))
            return False

    def check_dko_package(self) -> bool:
        """Check DKO package installation."""
        try:
            import dko
            self.add_result("DKO package", True, "Importable")

            # Check key modules
            modules_to_check = [
                'dko.models.dko',
                'dko.training.hpc_trainer',
                'dko.data.datasets',
                'dko.scripts.train',
            ]

            all_ok = True
            for module in modules_to_check:
                try:
                    importlib.import_module(module)
                except ImportError as e:
                    self.add_result(f"Module {module}", False, str(e))
                    all_ok = False

            if all_ok:
                self.add_result("DKO modules", True, "All key modules importable")

            return all_ok
        except ImportError as e:
            self.add_result("DKO package", False, f"Not installed: {e}")
            self.add_result("", False, "", "Run: pip install -e .")
            return False

    def check_optional_dependencies(self) -> bool:
        """Check optional dependencies."""
        optional = {
            'yaml': 'PyYAML',
            'wandb': 'Weights & Biases',
            'tqdm': 'Progress bars',
            'pandas': 'Data analysis',
            'scipy': 'Scientific computing',
            'sklearn': 'Scikit-learn',
        }

        all_ok = True
        for module, name in optional.items():
            try:
                importlib.import_module(module)
                self.add_result(f"{name}", True, "Available")
            except ImportError:
                self.add_result(f"{name}", False, "Not installed (optional)")
                if module == 'yaml':
                    all_ok = False  # YAML is needed for config files

        return all_ok

    def check_directories(self) -> bool:
        """Check required directories exist."""
        required_dirs = [
            ('configs', "Configuration files"),
            ('dko', "Main package"),
            ('scripts', "Utility scripts"),
        ]

        optional_dirs = [
            ('data', "Dataset storage"),
            ('experiments', "Experiment outputs"),
            ('checkpoints', "Model checkpoints"),
            ('logs', "SLURM logs"),
        ]

        all_ok = True
        for dir_name, description in required_dirs:
            path = project_root / dir_name
            if path.exists():
                self.add_result(f"Directory: {dir_name}", True, description)
            else:
                self.add_result(f"Directory: {dir_name}", False, f"Missing - {description}")
                all_ok = False

        for dir_name, description in optional_dirs:
            path = project_root / dir_name
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                self.add_result(f"Directory: {dir_name}", True, f"Created - {description}")
            else:
                self.add_result(f"Directory: {dir_name}", True, description)

        return all_ok

    def check_config_files(self) -> bool:
        """Check configuration files."""
        config_dir = project_root / 'configs'

        if not config_dir.exists():
            self.add_result("Config directory", False, "configs/ not found")
            return False

        base_config = config_dir / 'base_config.yaml'
        if base_config.exists():
            self.add_result("Base config", True, str(base_config))
        else:
            self.add_result("Base config", False, "configs/base_config.yaml not found")

        # Check for experiment configs
        exp_configs = list((config_dir / 'experiments').glob('*.yaml')) if (config_dir / 'experiments').exists() else []
        if exp_configs:
            self.add_result("Experiment configs", True, f"{len(exp_configs)} found")
        else:
            self.add_result("Experiment configs", False, "No experiment configs found",
                          "Create configs in configs/experiments/")

        return base_config.exists()

    def check_slurm_scripts(self) -> bool:
        """Check SLURM submission scripts."""
        scripts = [
            ('scripts/submit_hpc.sh', "Main submission script"),
            ('scripts/submit_all_experiments.py', "Batch submission"),
            ('scripts/monitor_jobs.py', "Job monitoring"),
        ]

        all_ok = True
        for script, description in scripts:
            path = project_root / script
            if path.exists():
                self.add_result(f"Script: {script}", True, description)
            else:
                self.add_result(f"Script: {script}", False, f"Missing - {description}")
                all_ok = False

        return all_ok

    def check_slurm_environment(self) -> bool:
        """Check if running on a SLURM cluster."""
        slurm_vars = ['SLURM_JOB_ID', 'SLURM_NODELIST', 'SLURM_CLUSTER_NAME']

        is_slurm = any(var in os.environ for var in slurm_vars)

        if is_slurm:
            self.add_result("SLURM environment", True, "Running on SLURM cluster")
            for var in slurm_vars:
                if var in os.environ:
                    self.add_result(f"  {var}", True, os.environ[var])
        else:
            # Check if SLURM commands are available
            try:
                result = subprocess.run(['squeue', '--version'], capture_output=True, text=True)
                if result.returncode == 0:
                    self.add_result("SLURM commands", True, "Available (not in a job)")
                else:
                    self.add_result("SLURM commands", False, "Not available")
            except FileNotFoundError:
                self.add_result("SLURM environment", False, "Not on SLURM cluster",
                              "This is fine for local testing")

        return True  # Not critical for validation

    def validate_config(self, config_path: str) -> bool:
        """Validate a specific config file."""
        path = Path(config_path)

        if not path.exists():
            self.add_result(f"Config: {config_path}", False, "File not found")
            return False

        try:
            import yaml
            with open(path, 'r') as f:
                config = yaml.safe_load(f)

            # Check required sections
            required_sections = ['dataset', 'model', 'training']
            missing = [s for s in required_sections if s not in config]

            if missing:
                self.add_result(
                    f"Config: {path.name}",
                    False,
                    f"Missing sections: {missing}"
                )
                return False

            # Validate with dko.scripts.train
            from dko.scripts.train import validate_config
            validated = validate_config(config)
            self.add_result(f"Config: {path.name}", True, "Valid configuration")

            # Show key settings
            print(f"    Dataset: {validated['dataset']['name']}")
            print(f"    Model: {validated['model']['type']}")
            print(f"    Epochs: {validated['training']['max_epochs']}")
            print(f"    Batch size: {validated['training']['batch_size']}")

            return True

        except Exception as e:
            self.add_result(f"Config: {path.name}", False, str(e))
            return False

    def run_quick_test(self) -> bool:
        """Run a quick sanity test of the training pipeline."""
        print("\n" + "=" * 60)
        print("Running Quick Pipeline Test")
        print("=" * 60)

        try:
            import torch
            from dko.models.dko import DKO, DKOFirstOrder
            from dko.training.hpc_trainer import EnhancedTrainer

            # Create small model
            feature_dim = 32
            model = DKOFirstOrder(
                feature_dim=feature_dim,
                output_dim=1,
                task='regression',
                kernel_hidden_dims=[64, 32],
                kernel_output_dim=16,
                verbose=False,
            )

            # Create synthetic batch
            batch_size = 4
            mu = torch.randn(batch_size, feature_dim)
            sigma = torch.randn(batch_size, feature_dim, feature_dim)
            sigma = torch.bmm(sigma, sigma.transpose(1, 2)) + 0.1 * torch.eye(feature_dim)
            labels = torch.randn(batch_size, 1)

            # Forward pass
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = model.to(device)
            mu = mu.to(device)
            sigma = sigma.to(device)

            output = model(mu, sigma)

            self.add_result(
                "Pipeline test",
                True,
                f"Forward pass OK (device: {device})",
                f"Input: ({batch_size}, {feature_dim}), Output: {tuple(output.shape)}"
            )

            # Test gradient computation
            loss = output.mean()
            loss.backward()

            self.add_result("Gradient computation", True, "Backward pass OK")

            return True

        except Exception as e:
            self.add_result("Pipeline test", False, str(e))
            import traceback
            print(traceback.format_exc())
            return False

    def run_all(self, config_path: str = None, full: bool = False) -> Tuple[int, int]:
        """Run all validation checks."""
        print("=" * 60)
        print("DKO HPC Setup Validation")
        print("=" * 60)
        print()

        print("Core Dependencies:")
        print("-" * 40)
        self.check_python_version()
        self.check_pytorch()
        self.check_rdkit()
        print()

        print("DKO Package:")
        print("-" * 40)
        self.check_dko_package()
        print()

        print("Optional Dependencies:")
        print("-" * 40)
        self.check_optional_dependencies()
        print()

        print("Project Structure:")
        print("-" * 40)
        self.check_directories()
        self.check_config_files()
        self.check_slurm_scripts()
        print()

        print("Environment:")
        print("-" * 40)
        self.check_slurm_environment()
        print()

        if config_path:
            print(f"Config Validation:")
            print("-" * 40)
            self.validate_config(config_path)
            print()

        if full:
            self.run_quick_test()
            print()

        # Summary
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)

        print("=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print()

        if failed == 0:
            print("[OK] All checks passed! Ready for HPC deployment.")
        else:
            print("[FAIL] Some checks failed. Please address the issues above.")
            print()
            print("Failed checks:")
            for r in self.results:
                if not r.passed and r.name:
                    print(f"  - {r.name}: {r.message}")

        return passed, failed


def main():
    parser = argparse.ArgumentParser(
        description='Validate HPC setup for DKO training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic validation
    python scripts/validate_hpc_setup.py

    # Validate specific config
    python scripts/validate_hpc_setup.py --config configs/experiments/dko_esol.yaml

    # Full validation with pipeline test
    python scripts/validate_hpc_setup.py --full
        """
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to config file to validate'
    )
    parser.add_argument(
        '--full', '-f',
        action='store_true',
        help='Run comprehensive tests including pipeline test'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Only show summary'
    )

    args = parser.parse_args()

    validator = HPCValidator(verbose=not args.quiet)
    passed, failed = validator.run_all(config_path=args.config, full=args.full)

    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()
