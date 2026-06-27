#!/usr/bin/env python3
"""
industry_freshness_cadence.py — runs the recurring industry scorecard freshness check.

Captures output for the freshness cadence (--stale + --verify-sources) so the
stale count never silently regrows. Run on a schedule via `fak cron emit`.

Usage:
    python tools/industry_freshness_cadence.py [--output-dir DIR]

The output dir defaults to docs/industry-scorecard/cadence-output; each run
writes a timestamped JSONL entry with the full scorecard output.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run recurring industry scorecard freshness check"
    )
    parser.add_argument(
        "--output-dir",
        default="docs/industry-scorecard/cadence-output",
        help="Directory to write timestamped output (default: docs/industry-scorecard/cadence-output)",
    )
    args = parser.parse_args()

    # Ensure we're at the repo root
    repo_root = Path(__file__).parent.parent
    os.chdir(repo_root)

    # Import the scorecard tool
    sys.path.insert(0, str(repo_root / "tools"))
    try:
        from industry_scorecard import main as scorecard_main
    except ImportError:
        print("Error: cannot import industry_scorecard.py from tools/", file=sys.stderr)
        return 1

    # Prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run the freshness check (--stale + --verify-sources)
    print(f"[{datetime.now(timezone.utc).isoformat()}] Running industry freshness cadence...")
    
    # Capture scorecard output
    timestamp = datetime.now(timezone.utc).isoformat()
    result = {
        "timestamp": timestamp,
        "run_type": "freshness-cadence",
        "checks": [],
    }

    # Run --stale check
    sys.argv = ["industry_scorecard.py", "--stale"]
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    from io import StringIO
    stdout_buf = StringIO()
    stderr_buf = StringIO()
    sys.stdout = stdout_buf
    sys.stderr = stderr_buf
    
    try:
        scorecard_main()
    except SystemExit:
        pass  # Scorecard exits after printing
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    stale_output = stdout_buf.getvalue() + stderr_buf.getvalue()
    result["checks"].append({
        "name": "stale",
        "output": stale_output,
    })
    print(stale_output)

    # Run --verify-sources check
    stdout_buf = StringIO()
    stderr_buf = StringIO()
    sys.stdout = stdout_buf
    sys.stderr = stderr_buf
    sys.argv = ["industry_scorecard.py", "--verify-sources"]
    
    try:
        scorecard_main()
    except SystemExit:
        pass
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    verify_output = stdout_buf.getvalue() + stderr_buf.getvalue()
    result["checks"].append({
        "name": "verify-sources",
        "output": verify_output,
    })
    print(verify_output)

    # Run full scorecard for grade/coverage
    stdout_buf = StringIO()
    stderr_buf = StringIO()
    sys.stdout = stdout_buf
    sys.stderr = stderr_buf
    sys.argv = ["industry_scorecard.py"]
    
    try:
        scorecard_main()
    except SystemExit:
        pass
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    scorecard_output = stdout_buf.getvalue() + stderr_buf.getvalue()
    result["checks"].append({
        "name": "scorecard",
        "output": scorecard_output,
    })

    # Write timestamped output
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    output_file = output_dir / f"freshness-cadence-{date_str}.jsonl"
    with open(output_file, "a") as f:
        f.write(json.dumps(result) + "\n")
    
    print(f"\n[{datetime.now(timezone.utc).isoformat()}] Freshness cadence complete. Output written to {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())