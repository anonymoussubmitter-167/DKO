#!/usr/bin/env python
"""
Recovery script for failed DKO experiments.

Scans experiment directories for failed or incomplete jobs and provides
options for recovery, cleanup, or resubmission.

Usage:
    python scripts/recover_failed_jobs.py --scan
    python scripts/recover_failed_jobs.py --recover
    python scripts/recover_failed_jobs.py --cleanup --dry-run
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class ExperimentStatus:
    """Status categories for experiments."""
    COMPLETED = "completed"
    IN_PROGRESS = "in_progress"
    FAILED = "failed"
    INCOMPLETE = "incomplete"
    CORRUPT = "corrupt"
    UNKNOWN = "unknown"


class FailedJobRecovery:
    """Recover failed DKO experiments."""

    def __init__(self, experiments_dir: str = "experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.slurm_available = self._check_slurm()

    def _check_slurm(self) -> bool:
        """Check if SLURM is available."""
        try:
            subprocess.run(['squeue', '--version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def scan_experiments(self) -> Dict[str, List[Dict[str, Any]]]:
        """Scan all experiments and categorize by status."""
        results = {
            ExperimentStatus.COMPLETED: [],
            ExperimentStatus.IN_PROGRESS: [],
            ExperimentStatus.FAILED: [],
            ExperimentStatus.INCOMPLETE: [],
            ExperimentStatus.CORRUPT: [],
        }

        if not self.experiments_dir.exists():
            print(f"Experiments directory not found: {self.experiments_dir}")
            return results

        for exp_dir in sorted(self.experiments_dir.iterdir()):
            if not exp_dir.is_dir():
                continue

            status, info = self._check_experiment_status(exp_dir)
            results[status].append(info)

        return results

    def _check_experiment_status(self, exp_dir: Path) -> Tuple[str, Dict[str, Any]]:
        """Check the status of a single experiment."""
        info = {
            "name": exp_dir.name,
            "path": str(exp_dir),
            "created": None,
            "modified": None,
        }

        # Get timestamps
        try:
            stat = exp_dir.stat()
            info["created"] = datetime.fromtimestamp(stat.st_ctime).isoformat()
            info["modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
        except:
            pass

        # Check for key files
        results_file = exp_dir / "results.json"
        config_file = exp_dir / "config.json"
        checkpoint_dir = exp_dir / "checkpoints"
        log_file = exp_dir / "training.log"

        info["has_config"] = config_file.exists()
        info["has_results"] = results_file.exists()
        info["has_log"] = log_file.exists()

        # Check checkpoints
        checkpoints = []
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("*.pt"))
        info["num_checkpoints"] = len(checkpoints)
        info["has_checkpoint"] = len(checkpoints) > 0

        # Load config if available
        if config_file.exists():
            try:
                with open(config_file) as f:
                    info["config"] = json.load(f)
            except:
                info["config"] = None

        # Check results
        if results_file.exists():
            try:
                with open(results_file) as f:
                    results = json.load(f)
                info["results"] = results

                if "test_metrics" in results:
                    return ExperimentStatus.COMPLETED, info
                else:
                    return ExperimentStatus.INCOMPLETE, info
            except json.JSONDecodeError:
                info["error"] = "Corrupt results.json"
                return ExperimentStatus.CORRUPT, info

        # No results file - check for progress
        if info["has_checkpoint"]:
            # Has checkpoint but no results - might be in progress or failed
            if self._is_job_running(exp_dir.name):
                return ExperimentStatus.IN_PROGRESS, info
            else:
                # Check log for errors
                if log_file.exists():
                    error = self._extract_error_from_log(log_file)
                    if error:
                        info["error"] = error
                        return ExperimentStatus.FAILED, info
                return ExperimentStatus.INCOMPLETE, info

        # No checkpoint and no results
        if log_file.exists():
            error = self._extract_error_from_log(log_file)
            if error:
                info["error"] = error
                return ExperimentStatus.FAILED, info

        return ExperimentStatus.INCOMPLETE, info

    def _is_job_running(self, exp_name: str) -> bool:
        """Check if a SLURM job is running for this experiment."""
        if not self.slurm_available:
            return False

        try:
            user = os.environ.get("USER")
            result = subprocess.run(
                ["squeue", "-u", user, "--format=%j", "--noheader"],
                capture_output=True, text=True, check=True
            )
            running_jobs = result.stdout.strip().split("\n")
            return any(exp_name in job for job in running_jobs)
        except:
            return False

    def _extract_error_from_log(self, log_file: Path, max_lines: int = 100) -> Optional[str]:
        """Extract error message from log file."""
        try:
            with open(log_file, 'r', errors='ignore') as f:
                lines = f.readlines()

            # Look for common error patterns
            error_patterns = [
                "RuntimeError:", "ValueError:", "TypeError:",
                "CUDA error:", "OutOfMemoryError:", "Exception:",
                "Error:", "FAILED", "Traceback"
            ]

            for line in reversed(lines[-max_lines:]):
                for pattern in error_patterns:
                    if pattern in line:
                        # Get context around the error
                        idx = lines.index(line) if line in lines else -1
                        if idx >= 0:
                            context = lines[max(0, idx-2):min(len(lines), idx+5)]
                            return "\n".join(l.strip() for l in context)
                        return line.strip()

            return None
        except:
            return None

    def print_scan_results(self, results: Dict[str, List[Dict]]):
        """Print formatted scan results."""
        print("\n" + "=" * 80)
        print("DKO Experiment Recovery Scan")
        print("=" * 80)

        total = sum(len(v) for v in results.values())
        print(f"\nTotal experiments: {total}")
        print(f"  Completed: {len(results[ExperimentStatus.COMPLETED])}")
        print(f"  In Progress: {len(results[ExperimentStatus.IN_PROGRESS])}")
        print(f"  Failed: {len(results[ExperimentStatus.FAILED])}")
        print(f"  Incomplete: {len(results[ExperimentStatus.INCOMPLETE])}")
        print(f"  Corrupt: {len(results[ExperimentStatus.CORRUPT])}")

        if results[ExperimentStatus.FAILED]:
            print("\n" + "-" * 80)
            print("FAILED EXPERIMENTS:")
            print("-" * 80)
            for exp in results[ExperimentStatus.FAILED]:
                print(f"\n  {exp['name']}:")
                if exp.get("error"):
                    error_preview = exp["error"][:200] + "..." if len(exp.get("error", "")) > 200 else exp.get("error", "")
                    print(f"    Error: {error_preview}")
                print(f"    Checkpoints: {exp['num_checkpoints']}")

        if results[ExperimentStatus.INCOMPLETE]:
            print("\n" + "-" * 80)
            print("INCOMPLETE EXPERIMENTS:")
            print("-" * 80)
            for exp in results[ExperimentStatus.INCOMPLETE]:
                print(f"  {exp['name']}: checkpoints={exp['num_checkpoints']}")

        if results[ExperimentStatus.CORRUPT]:
            print("\n" + "-" * 80)
            print("CORRUPT EXPERIMENTS:")
            print("-" * 80)
            for exp in results[ExperimentStatus.CORRUPT]:
                print(f"  {exp['name']}: {exp.get('error', 'Unknown error')}")

    def recover_experiment(
        self,
        exp_info: Dict[str, Any],
        dry_run: bool = False
    ) -> bool:
        """Attempt to recover a single experiment."""
        exp_path = Path(exp_info["path"])
        exp_name = exp_info["name"]

        print(f"\nRecovering: {exp_name}")

        # Check for checkpoint
        checkpoint_dir = exp_path / "checkpoints"
        latest_checkpoint = None

        if checkpoint_dir.exists():
            checkpoints = sorted(checkpoint_dir.glob("*.pt"))
            if checkpoints:
                latest_checkpoint = checkpoints[-1]
                print(f"  Found checkpoint: {latest_checkpoint.name}")

        # Find config
        config_path = exp_path / "config.json"
        if not config_path.exists():
            print(f"  ERROR: No config.json found")
            return False

        if dry_run:
            print(f"  [DRY RUN] Would resubmit with checkpoint: {latest_checkpoint}")
            return True

        # Build resubmit command
        if self.slurm_available:
            cmd = [
                "sbatch",
                "scripts/submit_hpc.sh",
                str(config_path),
                exp_name,
            ]
            if latest_checkpoint:
                cmd.extend(["--resume-from", str(latest_checkpoint)])

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                job_id = result.stdout.strip().split()[-1]
                print(f"  Submitted as job {job_id}")
                return True
            except subprocess.CalledProcessError as e:
                print(f"  ERROR: Failed to submit: {e.stderr}")
                return False
        else:
            print("  SLURM not available, cannot resubmit")
            return False

    def recover_all_failed(self, results: Dict[str, List[Dict]], dry_run: bool = False):
        """Recover all failed and incomplete experiments."""
        to_recover = results[ExperimentStatus.FAILED] + results[ExperimentStatus.INCOMPLETE]

        if not to_recover:
            print("No experiments to recover")
            return

        print(f"\nRecovering {len(to_recover)} experiments...")

        succeeded = 0
        failed = 0

        for exp in to_recover:
            if self.recover_experiment(exp, dry_run=dry_run):
                succeeded += 1
            else:
                failed += 1

        print(f"\nRecovery complete: {succeeded} succeeded, {failed} failed")

    def cleanup_failed(
        self,
        results: Dict[str, List[Dict]],
        dry_run: bool = False,
        keep_checkpoints: bool = True
    ):
        """Clean up failed experiments."""
        to_cleanup = results[ExperimentStatus.FAILED] + results[ExperimentStatus.CORRUPT]

        if not to_cleanup:
            print("No experiments to clean up")
            return

        print(f"\nCleaning up {len(to_cleanup)} experiments...")

        for exp in to_cleanup:
            exp_path = Path(exp["path"])
            print(f"\n  {exp['name']}:")

            if keep_checkpoints:
                # Only remove logs and corrupt files
                files_to_remove = [
                    exp_path / "training.log",
                    exp_path / "results.json",
                ]
                for f in files_to_remove:
                    if f.exists():
                        if dry_run:
                            print(f"    [DRY RUN] Would remove: {f.name}")
                        else:
                            f.unlink()
                            print(f"    Removed: {f.name}")
            else:
                # Remove entire directory
                if dry_run:
                    print(f"    [DRY RUN] Would remove entire directory")
                else:
                    shutil.rmtree(exp_path)
                    print(f"    Removed directory")

    def generate_recovery_script(self, results: Dict[str, List[Dict]], output: str = "recover.sh"):
        """Generate a shell script for manual recovery."""
        to_recover = results[ExperimentStatus.FAILED] + results[ExperimentStatus.INCOMPLETE]

        if not to_recover:
            print("No experiments to recover")
            return

        lines = [
            "#!/bin/bash",
            "# Auto-generated recovery script",
            f"# Generated: {datetime.now().isoformat()}",
            f"# Experiments to recover: {len(to_recover)}",
            "",
        ]

        for exp in to_recover:
            exp_path = Path(exp["path"])
            config_path = exp_path / "config.json"
            checkpoint_dir = exp_path / "checkpoints"

            # Find latest checkpoint
            checkpoint_arg = ""
            if checkpoint_dir.exists():
                checkpoints = sorted(checkpoint_dir.glob("*.pt"))
                if checkpoints:
                    checkpoint_arg = f"--resume-from {checkpoints[-1]}"

            lines.append(f"# {exp['name']}")
            lines.append(f"sbatch scripts/submit_hpc.sh {config_path} {exp['name']} {checkpoint_arg}")
            lines.append("")

        with open(output, "w") as f:
            f.write("\n".join(lines))

        print(f"Recovery script written to: {output}")
        print(f"Review and run: bash {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Recover failed DKO experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Scan experiments
    python scripts/recover_failed_jobs.py --scan

    # Recover all failed experiments
    python scripts/recover_failed_jobs.py --recover

    # Dry run - see what would be done
    python scripts/recover_failed_jobs.py --recover --dry-run

    # Clean up failed experiments (keep checkpoints)
    python scripts/recover_failed_jobs.py --cleanup

    # Generate recovery script
    python scripts/recover_failed_jobs.py --generate-script
        """
    )

    parser.add_argument(
        "--experiments-dir",
        type=str,
        default="experiments",
        help="Directory containing experiments"
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan and report experiment status"
    )
    parser.add_argument(
        "--recover",
        action="store_true",
        help="Recover failed experiments"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up failed experiments"
    )
    parser.add_argument(
        "--generate-script",
        action="store_true",
        help="Generate a recovery shell script"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without doing it"
    )
    parser.add_argument(
        "--remove-all",
        action="store_true",
        help="When cleaning, remove entire directories (not just logs)"
    )

    args = parser.parse_args()

    recovery = FailedJobRecovery(args.experiments_dir)
    results = recovery.scan_experiments()

    if args.scan or not any([args.recover, args.cleanup, args.generate_script]):
        recovery.print_scan_results(results)

    if args.recover:
        recovery.recover_all_failed(results, dry_run=args.dry_run)

    if args.cleanup:
        recovery.cleanup_failed(
            results,
            dry_run=args.dry_run,
            keep_checkpoints=not args.remove_all
        )

    if args.generate_script:
        recovery.generate_recovery_script(results)


if __name__ == "__main__":
    main()
