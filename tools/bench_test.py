#!/usr/bin/env python3
"""bench_test.py -- validate benchmark infrastructure schemas and tools.

Usage:
  python tools/bench_test.py --schemas      # Validate JSON schemas
  python tools/bench_test.py --tools        # Test tool imports
  python tools/bench_test.py --all          # Run all tests
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS_DIR = ROOT / "tools" / "schemas"
TOOLS_DIR = ROOT / "tools"


def validate_schemas() -> bool:
    """Validate that all schema files are valid JSON."""
    print("[test] Validating JSON schemas...", file=sys.stderr)

    schemas = [
        "machine-specs.v1.json",
        "run-manifest.v1.json",
        "kernel-results.v1.json",
        "batch-results.v1.json",
        "catalog.v1.json"
    ]

    all_valid = True
    for schema_name in schemas:
        schema_path = SCHEMAS_DIR / schema_name
        if not schema_path.exists():
            print(f"  [FAIL] Missing schema: {schema_name}", file=sys.stderr)
            all_valid = False
            continue

        try:
            with open(schema_path, encoding="utf-8") as f:
                data = json.load(f)
            # Basic structure checks
            if "$schema" not in data:
                print(f"  [WARN] Schema missing $schema: {schema_name}", file=sys.stderr)
            if "$id" not in data and "title" not in data:
                print(f"  [WARN] Schema missing $id and title: {schema_name}", file=sys.stderr)
            print(f"  [OK] {schema_name}", file=sys.stderr)
        except json.JSONDecodeError as e:
            print(f"  [FAIL] Invalid JSON in {schema_name}: {e}", file=sys.stderr)
            all_valid = False
        except OSError as e:
            print(f"  [FAIL] Cannot read {schema_name}: {e}", file=sys.stderr)
            all_valid = False

    return all_valid


def test_tool_imports() -> bool:
    """Test that all benchmark tools can be imported."""
    print("[test] Testing tool imports...", file=sys.stderr)

    tools = [
        "bench_catalog",
        "bench_cli",
        "bench_chart",
        "bench_onboard",
        "bench_migrate"
    ]

    all_ok = True
    for tool_name in tools:
        tool_path = TOOLS_DIR / f"{tool_name}.py"
        if not tool_path.exists():
            print(f"  [FAIL] Missing tool: {tool_name}.py", file=sys.stderr)
            all_ok = False
            continue

        # Try to compile it
        try:
            with open(tool_path, encoding="utf-8") as f:
                code = f.read()
            compile(code, str(tool_path), "exec")
            print(f"  [OK] {tool_name}.py", file=sys.stderr)
        except SyntaxError as e:
            print(f"  [FAIL] Syntax error in {tool_name}.py: {e}", file=sys.stderr)
            all_ok = False

    return all_ok


def test_schema_examples() -> bool:
    """Test that examples in schemas are valid."""
    print("[test] Testing schema examples...", file=sys.stderr)

    # This would test the example JSON from the design doc
    # For now, just verify schemas parse
    return validate_schemas()


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--schemas", action="store_true", help="Validate JSON schemas")
    ap.add_argument("--tools", action="store_true", help="Test tool imports")
    ap.add_argument("--all", action="store_true", help="Run all tests")
    args = ap.parse_args(argv)

    if not (args.schemas or args.tools or args.all):
        ap.print_help()
        return 1

    all_ok = True

    if args.schemas or args.all:
        if not validate_schemas():
            all_ok = False

    if args.tools or args.all:
        if not test_tool_imports():
            all_ok = False

    if all_ok:
        print("[test] All tests passed", file=sys.stderr)
        return 0
    else:
        print("[test] Some tests failed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    from typing import List
    sys.exit(main(sys.argv[1:]))
