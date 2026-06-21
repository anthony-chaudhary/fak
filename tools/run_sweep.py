#!/usr/bin/env python3
"""
run_sweep.py - Python sweep orchestrator for multi-model benchmark sweeps.

Replaces/adapts the monolithic PowerShell script (run_transcript_adapter_sweep.ps1)
with a YAML-configurable driver that:
- Reads sweep configs from YAML/JSON
- Manages local shim processes
- Generates unified summary artifacts (JSON + Markdown)
- Integrates with existing benchmark tooling

Usage:
    python tools/run_sweep.py --profile quick-smoke
    python tools/run_sweep.py --profile custom --list-models
    python tools/run_sweep.py --profile custom --models glm-4.7-flash,gpt-4.1-nano
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import sweep config module
try:
    from sweep_config import (
        load_profile, save_profile, list_profiles, get_profile_path,
        SweepProfile, ModelConfig, WorkloadConfig, SweepResult
    )
except ImportError:
    # Add tools directory to path if running directly
    sys.path.insert(0, str(Path(__file__).parent))
    from sweep_config import (
        load_profile, save_profile, list_profiles, get_profile_path,
        SweepProfile, ModelConfig, WorkloadConfig, SweepResult
    )


# Constants
ROOT = Path(__file__).parent.parent
FAK_DIR = ROOT / "fak"
DEFAULT_BIN_DIR = ROOT / "tools" / ".bin"
DEFAULT_OUTPUT_DIR = ROOT / "fak" / "experiments" / "agent-live" / "sweep"


def run_command(cmd: List[str], cwd: Path = None, env: Dict[str, str] = None,
                timeout: int = None, capture: bool = True) -> subprocess.CompletedProcess:
    """Run a command with optional timeout and custom environment."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    kwargs = {
        'cwd': cwd or ROOT,
        'env': merged_env,
        'text': True,
        'capture_output': capture
    }

    if timeout:
        kwargs['timeout'] = timeout

    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, **kwargs)


def build_fak(bin_dir: Path = DEFAULT_BIN_DIR) -> bool:
    """Build fak executable."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    fak_bin = bin_dir / "fak.exe"

    result = run_command(
        ['go', 'build', '-o', str(fak_bin), './cmd/fak'],
        cwd=FAK_DIR
    )

    if result.returncode != 0:
        print(f"Build failed: {result.stderr}")
        return False

    print(f"Built fak: {fak_bin}")
    return True


def run_api_model(model: ModelConfig, workload: WorkloadConfig,
                  output_dir: Path, trial: int) -> SweepResult:
    """Run sweep against an API model."""
    # TODO: Implement API model sweep
    # This would call the API endpoint with the workload and measure results
    start = time.time()

    # Placeholder: simulate API call
    print(f"Would run API model {model.name} (trial {trial})")

    return SweepResult(
        profile_name="api",
        model_name=model.name,
        trial=trial,
        success=False,
        duration_ms=0,
        tokens_total=0,
        tokens_per_sec=0.0,
        error="Not yet implemented"
    )


def run_local_shim(model: ModelConfig, workload: WorkloadConfig,
                   output_dir: Path, trial: int, fak_bin: Path) -> SweepResult:
    """Run sweep against a local shim (fak in-process model)."""
    start = time.time()

    # Use local_shim.py if specified, otherwise use fak directly
    shim_path = Path(model.local_shim) if model.local_shim else (FAK_DIR / "experiments" / "agent-live" / "local_shim.py")

    if not shim_path.exists():
        return SweepResult(
            profile_name="local",
            model_name=model.name,
            trial=trial,
            success=False,
            duration_ms=0,
            tokens_total=0,
            tokens_per_sec=0.0,
            error=f"Shim not found: {shim_path}"
        )

    # Prepare environment
    env = {
        'PYTHONPATH': str(FAK_DIR),
        'FAK_MAX_TURNS': str(workload.max_turns),
        'FAK_TRIAL': str(trial),
        'FAK_OUTPUT_DIR': str(output_dir)
    }

    # Run the shim
    try:
        result = run_command(
            [sys.executable, str(shim_path)],
            timeout=workload.timeout_s,
            env=env,
            capture=True
        )

        duration_ms = (time.time() - start) * 1000

        if result.returncode != 0:
            return SweepResult(
                profile_name="local",
                model_name=model.name,
                trial=trial,
                success=False,
                duration_ms=duration_ms,
                tokens_total=0,
                tokens_per_sec=0.0,
                error=result.stderr
            )

        # Parse output for metrics
        # TODO: Parse shim output for tokens/sec
        return SweepResult(
            profile_name="local",
            model_name=model.name,
            trial=trial,
            success=True,
            duration_ms=duration_ms,
            tokens_total=0,  # Parse from output
            tokens_per_sec=0.0,  # Parse from output
            metadata={'stdout': result.stdout}
        )

    except subprocess.TimeoutExpired:
        return SweepResult(
            profile_name="local",
            model_name=model.name,
            trial=trial,
            success=False,
            duration_ms=workload.timeout_s * 1000,
            tokens_total=0,
            tokens_per_sec=0.0,
            error=f"Timeout after {workload.timeout_s}s"
        )


def run_sweep(profile: SweepProfile, output_dir: Path = None,
              fak_bin: Path = None) -> List[SweepResult]:
    """Run a full sweep across all enabled models in the profile."""
    results = []
    output_dir = output_dir or Path(profile.output_dir)
    output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Sweep: {profile.name} ===")
    print(f"Description: {profile.description}")
    print(f"Output: {output_dir}")
    print(f"Models: {len([m for m in profile.models if m.enabled])} enabled")

    # Build fak if needed
    if not fak_bin:
        fak_bin = DEFAULT_BIN_DIR / "fak.exe"
    if not fak_bin.exists():
        print("Building fak...")
        if not build_fak(fak_bin.parent):
            print("Failed to build fak")
            return results

    # Run each model
    for model in profile.models:
        if not model.enabled:
            continue

        print(f"\n--- Model: {model.name} ---")

        for trial in range(1, profile.workload.trials + 1):
            trial_output = output_dir / f"{model.name}-trial{trial}.json"

            # Determine run type
            if model.local_shim or model.provider == "fak":
                result = run_local_shim(model, profile.workload, output_dir, trial, fak_bin)
            elif model.base_url and not profile.skip_api:
                result = run_api_model(model, profile.workload, output_dir, trial)
            else:
                print(f"Skipping {model.name} (no run method configured)")
                continue

            results.append(result)

            # Save trial result
            with open(trial_output, 'w') as f:
                json.dump({
                    'model': model.name,
                    'trial': trial,
                    'success': result.success,
                    'duration_ms': result.duration_ms,
                    'tokens_per_sec': result.tokens_per_sec,
                    'error': result.error,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)

            if not result.success and profile.fail_fast:
                print(f"Trial {trial} failed and fail_fast is set - stopping sweep")
                return results

    return results


def generate_summary(results: List[SweepResult], output_dir: Path, profile: SweepProfile):
    """Generate summary artifacts (JSON + Markdown)."""
    summary = {
        'profile': profile.name,
        'timestamp': datetime.now().isoformat(),
        'total_runs': len(results),
        'successful': sum(1 for r in results if r.success),
        'results': [
            {
                'model': r.model_name,
                'trial': r.trial,
                'success': r.success,
                'duration_ms': r.duration_ms,
                'tokens_per_sec': r.tokens_per_sec,
                'error': r.error
            }
            for r in results
        ]
    }

    # JSON summary
    json_path = output_dir / "summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Markdown summary
    md_path = output_dir / "summary.md"
    with open(md_path, 'w') as f:
        f.write(f"# Sweep Summary: {profile.name}\n\n")
        f.write(f"**Timestamp:** {summary['timestamp']}\n\n")
        f.write(f"**Results:** {summary['successful']}/{summary['total_runs']} successful\n\n")
        f.write("## Model Results\n\n")
        f.write("| Model | Trial | Success | Duration (ms) | Tok/s |\n")
        f.write("|-------|-------|--------|--------------|-------|\n")

        for r in results:
            status = "✓" if r.success else "✗"
            f.write(f"| {r.model_name} | {r.trial} | {status} | {r.duration_ms:.1f} | {r.tokens_per_sec:.1f} |\n")

        if r.error:
            f.write(f"\n**Error:** `{r.error}`\n")

    print(f"\nSummary written to:")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")


def list_available_profiles(profile_dir: Path = None):
    """List all available sweep profiles."""
    profile_dir = profile_dir or (Path(__file__).parent / 'sweep_profiles')
    profiles = list_profiles(profile_dir)

    print("\nAvailable sweep profiles:")
    print("=" * 60)

    for p in profiles:
        public_tag = "PUBLIC" if p.public else "INTERNAL"
        enabled_count = sum(1 for m in p.models if m.enabled)
        print(f"\n{p.name} [{public_tag}]")
        print(f"  {p.description}")
        print(f"  Models: {enabled_count} enabled")
        for m in p.models:
            if m.enabled:
                status = "[+]"
            else:
                status = "[ ]"
            print(f"    {status} {m.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark sweeps across multiple models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_sweep.py --list
  python run_sweep.py --profile quick-smoke
  python run_sweep.py --profile quick-smoke --output ./results
  python run_sweep.py --profile custom --models glm-4.7-flash --trials 5
        """
    )

    parser.add_argument('--list', action='store_true',
                       help='List available sweep profiles')
    parser.add_argument('--profile', type=str,
                       help='Sweep profile to run (name or path)')
    parser.add_argument('--models', type=str,
                       help='Comma-separated list of models to run (overrides profile)')
    parser.add_argument('--trials', type=int,
                       help='Number of trials (overrides profile)')
    parser.add_argument('--max-turns', type=int,
                       help='Max turns per trial (overrides profile)')
    parser.add_argument('--output', type=str,
                       help='Output directory (overrides profile)')
    parser.add_argument('--fail-fast', action='store_true',
                       help='Stop on first failure')
    parser.add_argument('--fak-bin', type=str,
                       help='Path to fak executable')
    parser.add_argument('--profile-dir', type=str,
                       help='Directory containing sweep profiles')

    args = parser.parse_args()

    if args.list:
        profile_dir = Path(args.profile_dir) if args.profile_dir else None
        list_available_profiles(profile_dir)
        return 0

    if not args.profile:
        parser.error("--profile or --list is required")

    # Load profile
    profile_path = get_profile_path(args.profile, Path(args.profile_dir) if args.profile_dir else None)
    try:
        profile = load_profile(profile_path)
    except Exception as e:
        print(f"Failed to load profile {args.profile}: {e}")
        return 1

    # Override with CLI args
    if args.models:
        model_list = [m.strip() for m in args.models.split(',')]
        # Enable only specified models
        for m in profile.models:
            m.enabled = m.name in model_list

    if args.trials:
        profile.workload.trials = args.trials

    if args.max_turns:
        profile.workload.max_turns = args.max_turns

    if args.output:
        profile.output_dir = args.output

    if args.fail_fast:
        profile.fail_fast = True

    # Run sweep
    fak_bin = Path(args.fak_bin) if args.fak_bin else None
    results = run_sweep(profile, output_dir=Path(profile.output_dir), fak_bin=fak_bin)

    # Generate summary
    if results:
        output_dir = ROOT / profile.output_dir
        generate_summary(results, output_dir, profile)

    return 0 if all(r.success for r in results) else 1


if __name__ == '__main__':
    sys.exit(main())
