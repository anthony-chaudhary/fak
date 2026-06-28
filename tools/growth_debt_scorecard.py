#!/usr/bin/env python3
"""Growth-debt scorecard: count silently undefined model × backend cells.

This scorecard wraps `fak coverage-matrix` and extracts the growth_debt
metric — the count of cells that are silently undefined (neither a panic
fence nor a proven conformance path).

Usage:
    python tools/growth_debt_scorecard.py        # human report
    python tools/growth_debt_scorecard.py --json  # JSON payload
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCHEMA = "fak-growth-debt-scorecard/1"

def run_fak_coverage_matrix(repo_root: str) -> dict:
    """Run `fak coverage-matrix --json` and parse the output."""
    cmd = ["fak", "coverage-matrix", "--json"]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(result.stdout)
        # Extract growth_debt from corpus
        debt = data.get("corpus", {}).get("growth_debt", 0)
        return {
            "growth_debt": debt,
            "families": data.get("corpus", {}).get("families", 0),
            "backends": data.get("corpus", {}).get("backends", 0),
            "supported": data.get("corpus", {}).get("supported", 0),
            "proof_path_only": data.get("corpus", {}).get("proof_path_only", 0),
            "fenced": data.get("corpus", {}).get("fenced", 0),
            "undefined": data.get("corpus", {}).get("undefined", 0),
        }
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"growth-debt scorecard: fak coverage-matrix failed: {e}\n")
        if e.stderr:
            sys.stderr.write(f"stderr: {e.stderr}\n")
        sys.exit(1)
    except json.JSONDecodeError as e:
        sys.stderr.write(f"growth-debt scorecard: failed to parse JSON: {e}\n")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Growth-debt scorecard")
    parser.add_argument("--json", action="store_true", help="emit JSON payload")
    parser.add_argument("--repo-root", default=".", help="repository root directory (unused, kept for compatibility)")
    args = parser.parse_args()

    data = run_fak_coverage_matrix(args.repo_root)

    debt = data.get("growth_debt", 0)

    if args.json:
        payload = {
            "schema": SCHEMA,
            "growth_debt": debt,
            "families": data.get("families", 0),
            "backends": data.get("backends", 0),
            "total_cells": data.get("families", 0) * data.get("backends", 0),
            "supported": data.get("supported", 0),
            "proof_path_only": data.get("proof_path_only", 0),
            "fenced": data.get("fenced", 0),
            "undefined": data.get("undefined", 0),
        }
        print(json.dumps(payload, indent=2))
    else:
        families = data.get("families", 0)
        backends = data.get("backends", 0)
        total_cells = families * backends
        supported = data.get("supported", 0)
        fenced = data.get("fenced", 0)
        proofPathOnly = data.get("proof_path_only", 0)
        undefined = data.get("undefined", 0)

        print(f"== growth-debt scorecard ({SCHEMA}) ==")
        print(f"Families: {families}, Backends: {backends}, Total cells: {total_cells}")
        print(f"  Supported:       {supported}")
        print(f"  Proof-path-only: {proofPathOnly}")
        print(f"  Fenced:          {fenced}")
        print(f"  Undefined:       {undefined}  <- growth_debt")
        print(f"\nGrowth debt (silently undefined cells): {debt}")

if __name__ == "__main__":
    main()