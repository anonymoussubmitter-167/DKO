#!/usr/bin/env python
"""
Job Monitoring and Auto-Resubmission Script for HPC Experiments.

Monitors SLURM jobs and automatically resubmits failed experiments.

Usage:
    python scripts/monitor_jobs.py --watch                    # Monitor running jobs
    python scripts/monitor_jobs.py --check-failed             # Find and report failed jobs
    python scripts/monitor_jobs.py --resubmit-failed          # Resubmit failed jobs
    python scripts/monitor_jobs.py --status experiments/      # Check experiment status
"""

import argparse
import subprocess
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import re

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class JobMonitor:
    """Monitor and manage SLURM jobs."""

    def __init__(self, experiments_dir: str = 'experiments'):
        self.experiments_dir = Path(experiments_dir)
        self.slurm_available = self._check_slurm()

    def _check_slurm(self) -> bool:
        """Check if SLURM commands are available."""
        try:
            subprocess.run(['squeue', '--version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def get_running_jobs(self, user: Optional[str] = None) -> List[Dict[str, str]]:
        """Get list of running SLURM jobs."""
        if not self.slurm_available:
            print("SLURM not available")
            return []

        user = user or os.environ.get('USER')
        cmd = ['squeue', '-u', user, '--format=%i|%j|%T|%M|%N|%P', '--noheader']

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            jobs = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split('|')
                    if len(parts) >= 6:
                        jobs.append({
                            'job_id': parts[0],
                            'name': parts[1],
                            'state': parts[2],
                            'time': parts[3],
                            'node': parts[4],
                            'partition': parts[5],
                        })
            return jobs
        except subprocess.CalledProcessError as e:
            print(f"Error getting jobs: {e}")
            return []

    def get_job_info(self, job_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific job."""
        if not self.slurm_available:
            return {}

        cmd = ['scontrol', 'show', 'job', job_id]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info = {}
            for item in result.stdout.split():
                if '=' in item:
                    key, value = item.split('=', 1)
                    info[key] = value
            return info
        except subprocess.CalledProcessError:
            return {}

    def get_completed_jobs(self, hours: int = 24) -> List[Dict[str, str]]:
        """Get recently completed jobs."""
        if not self.slurm_available:
            return []

        user = os.environ.get('USER')
        cmd = [
            'sacct', '-u', user,
            '--starttime', f'now-{hours}hours',
            '--format=JobID,JobName%50,State,ExitCode,Elapsed,Start,End',
            '--noheader', '--parsable2'
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            jobs = []
            for line in result.stdout.strip().split('\n'):
                if line and '.batch' not in line and '.extern' not in line:
                    parts = line.split('|')
                    if len(parts) >= 7:
                        jobs.append({
                            'job_id': parts[0],
                            'name': parts[1],
                            'state': parts[2],
                            'exit_code': parts[3],
                            'elapsed': parts[4],
                            'start': parts[5],
                            'end': parts[6],
                        })
            return jobs
        except subprocess.CalledProcessError as e:
            print(f"Error getting completed jobs: {e}")
            return []

    def find_failed_experiments(self) -> List[Dict[str, Any]]:
        """Find experiments that failed or didn't complete."""
        failed = []

        if not self.experiments_dir.exists():
            return failed

        for exp_dir in self.experiments_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            # Check for results file
            results_file = exp_dir / 'results.json'
            config_file = exp_dir / 'config.json'
            log_file = exp_dir / 'train.log'

            status = {
                'name': exp_dir.name,
                'path': str(exp_dir),
                'has_results': results_file.exists(),
                'has_config': config_file.exists(),
                'has_log': log_file.exists(),
            }

            # Check if experiment is incomplete
            if not results_file.exists():
                # Check if there's a checkpoint (partial progress)
                checkpoint_dir = exp_dir / 'checkpoints'
                has_checkpoint = checkpoint_dir.exists() and any(checkpoint_dir.glob('*.pt'))
                status['has_checkpoint'] = has_checkpoint

                # Check log for errors
                if log_file.exists():
                    with open(log_file, 'r') as f:
                        log_content = f.read()
                        status['has_error'] = 'ERROR' in log_content or 'Exception' in log_content
                        status['last_lines'] = log_content.split('\n')[-10:]
                else:
                    status['has_error'] = False
                    status['last_lines'] = []

                status['status'] = 'incomplete'
                failed.append(status)
            else:
                # Check if results indicate failure
                try:
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                        if 'test_metrics' not in results:
                            status['status'] = 'no_metrics'
                            failed.append(status)
                except json.JSONDecodeError:
                    status['status'] = 'corrupt_results'
                    failed.append(status)

        return failed

    def resubmit_experiment(
        self,
        exp_name: str,
        config_path: Optional[str] = None,
        dry_run: bool = False
    ) -> Optional[str]:
        """Resubmit a failed experiment."""
        if not self.slurm_available:
            print("SLURM not available - cannot resubmit")
            return None

        exp_dir = self.experiments_dir / exp_name

        # Find config
        if config_path is None:
            config_path = exp_dir / 'config.json'
            if not config_path.exists():
                print(f"Config not found for {exp_name}")
                return None

        # Build resubmit command
        cmd = [
            'sbatch',
            'scripts/submit_hpc.sh',
            str(config_path),
            exp_name,
        ]

        if dry_run:
            print(f"Would run: {' '.join(cmd)}")
            return None

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            job_id = result.stdout.strip().split()[-1]
            print(f"Resubmitted {exp_name} as job {job_id}")
            return job_id
        except subprocess.CalledProcessError as e:
            print(f"Failed to resubmit: {e.stderr}")
            return None

    def watch_jobs(self, interval: int = 60, max_duration: int = 0):
        """Watch running jobs with periodic updates."""
        print(f"Watching jobs (Ctrl+C to stop)...")
        print(f"Update interval: {interval}s")
        print()

        start_time = time.time()

        try:
            while True:
                os.system('clear' if os.name == 'posix' else 'cls')

                print("=" * 80)
                print(f"DKO Job Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("=" * 80)

                jobs = self.get_running_jobs()

                if not jobs:
                    print("No running jobs")
                else:
                    print(f"\nRunning Jobs ({len(jobs)}):")
                    print("-" * 80)
                    print(f"{'Job ID':<12} {'Name':<35} {'State':<12} {'Time':<12} {'Node':<10}")
                    print("-" * 80)
                    for job in jobs:
                        print(f"{job['job_id']:<12} {job['name'][:35]:<35} {job['state']:<12} {job['time']:<12} {job['node'][:10]:<10}")

                # Show recent completions
                completed = self.get_completed_jobs(hours=2)
                if completed:
                    print(f"\nRecent Completions (last 2 hours):")
                    print("-" * 80)
                    for job in completed[-5:]:
                        status_icon = "✓" if job['state'] == 'COMPLETED' else "✗"
                        print(f"  {status_icon} {job['name'][:40]} - {job['state']} ({job['exit_code']})")

                print("\n" + "=" * 80)
                print(f"Press Ctrl+C to stop | Next update in {interval}s")

                if max_duration > 0 and (time.time() - start_time) > max_duration:
                    print("Max duration reached, stopping...")
                    break

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\nStopped watching.")

    def generate_status_report(self) -> str:
        """Generate a status report for all experiments."""
        lines = []
        lines.append("=" * 80)
        lines.append("DKO Experiment Status Report")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)

        if not self.experiments_dir.exists():
            lines.append(f"Experiments directory not found: {self.experiments_dir}")
            return '\n'.join(lines)

        # Count experiments by status
        completed = []
        in_progress = []
        failed = []

        for exp_dir in sorted(self.experiments_dir.iterdir()):
            if not exp_dir.is_dir():
                continue

            results_file = exp_dir / 'results.json'
            checkpoint_dir = exp_dir / 'checkpoints'

            if results_file.exists():
                try:
                    with open(results_file) as f:
                        results = json.load(f)
                    completed.append({
                        'name': exp_dir.name,
                        'metrics': results.get('test_metrics', {}),
                    })
                except:
                    failed.append({'name': exp_dir.name, 'reason': 'corrupt results'})
            elif checkpoint_dir.exists() and any(checkpoint_dir.glob('*.pt')):
                in_progress.append({'name': exp_dir.name})
            else:
                failed.append({'name': exp_dir.name, 'reason': 'no checkpoint'})

        lines.append(f"\nSummary:")
        lines.append(f"  Completed: {len(completed)}")
        lines.append(f"  In Progress: {len(in_progress)}")
        lines.append(f"  Failed/Incomplete: {len(failed)}")

        if completed:
            lines.append(f"\nCompleted Experiments:")
            lines.append("-" * 80)
            for exp in completed:
                metrics = exp['metrics']
                if 'rmse' in metrics:
                    lines.append(f"  {exp['name']}: RMSE={metrics['rmse']:.4f}")
                elif 'auroc' in metrics:
                    lines.append(f"  {exp['name']}: AUROC={metrics['auroc']:.4f}")
                else:
                    lines.append(f"  {exp['name']}: {list(metrics.keys())}")

        if in_progress:
            lines.append(f"\nIn Progress:")
            lines.append("-" * 80)
            for exp in in_progress:
                lines.append(f"  {exp['name']}")

        if failed:
            lines.append(f"\nFailed/Incomplete:")
            lines.append("-" * 80)
            for exp in failed:
                lines.append(f"  {exp['name']}: {exp.get('reason', 'unknown')}")

        lines.append("\n" + "=" * 80)
        return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Monitor and manage DKO experiments on SLURM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Watch running jobs
    python scripts/monitor_jobs.py --watch

    # Check for failed experiments
    python scripts/monitor_jobs.py --check-failed

    # Resubmit all failed experiments
    python scripts/monitor_jobs.py --resubmit-failed

    # Generate status report
    python scripts/monitor_jobs.py --status

    # Watch jobs for 1 hour then stop
    python scripts/monitor_jobs.py --watch --max-duration 3600
        """
    )

    parser.add_argument(
        '--experiments-dir',
        type=str,
        default='experiments',
        help='Directory containing experiments (default: experiments)'
    )
    parser.add_argument(
        '--watch',
        action='store_true',
        help='Watch running jobs with live updates'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Update interval in seconds for watch mode (default: 60)'
    )
    parser.add_argument(
        '--max-duration',
        type=int,
        default=0,
        help='Maximum watch duration in seconds (0 = unlimited)'
    )
    parser.add_argument(
        '--check-failed',
        action='store_true',
        help='Find and report failed experiments'
    )
    parser.add_argument(
        '--resubmit-failed',
        action='store_true',
        help='Resubmit all failed experiments'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Generate status report for all experiments'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without doing it'
    )
    parser.add_argument(
        '--list-jobs',
        action='store_true',
        help='List current running jobs'
    )

    args = parser.parse_args()

    monitor = JobMonitor(args.experiments_dir)

    if args.watch:
        monitor.watch_jobs(interval=args.interval, max_duration=args.max_duration)

    elif args.list_jobs:
        jobs = monitor.get_running_jobs()
        if not jobs:
            print("No running jobs")
        else:
            print(f"\nRunning Jobs ({len(jobs)}):")
            print("-" * 80)
            for job in jobs:
                print(f"  {job['job_id']}: {job['name']} ({job['state']}) on {job['node']}")

    elif args.check_failed:
        failed = monitor.find_failed_experiments()
        if not failed:
            print("No failed experiments found!")
        else:
            print(f"\nFailed/Incomplete Experiments ({len(failed)}):")
            print("-" * 80)
            for exp in failed:
                print(f"\n{exp['name']}:")
                print(f"  Status: {exp['status']}")
                print(f"  Has checkpoint: {exp.get('has_checkpoint', False)}")
                print(f"  Has error in log: {exp.get('has_error', False)}")
                if exp.get('last_lines'):
                    print(f"  Last log lines:")
                    for line in exp['last_lines'][-3:]:
                        if line.strip():
                            print(f"    {line}")

    elif args.resubmit_failed:
        failed = monitor.find_failed_experiments()
        if not failed:
            print("No failed experiments to resubmit")
        else:
            print(f"Found {len(failed)} failed experiments")
            for exp in failed:
                print(f"\nResubmitting: {exp['name']}")
                monitor.resubmit_experiment(exp['name'], dry_run=args.dry_run)

    elif args.status:
        report = monitor.generate_status_report()
        print(report)

    else:
        # Default: show brief status
        jobs = monitor.get_running_jobs()
        failed = monitor.find_failed_experiments()

        print("\nDKO Experiment Status")
        print("=" * 40)
        print(f"Running jobs: {len(jobs)}")
        print(f"Failed/incomplete experiments: {len(failed)}")
        print()
        print("Use --help for more options")


if __name__ == '__main__':
    main()
