#!/usr/bin/env python3
"""Tests for claim-reproducibility scorecard.

Pure-stdlib tests for parsing CLAIMS.md and BENCHMARK-AUTHORITY.md, extracting witness
handles, and validating resolvability against the tracked tree.
"""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

# Import the scorecard module
import sys

sys.path.insert(0, str(Path(__file__).parent))
import claim_repro_scorecard as crs

# Test fixtures
_SAMPLE_CLAIMS_GOOD = """
# Test claims

## The product

- [SHIPPED] One statically-linked Go binary runs an agentic tool loop. Witness: `go build ./...` exit 0.
- [SHIPPED] TestNoOsExecOnHotPath proves no spawned hook. Witness: `go test ./internal/adjudicator -run TestNoOsExecOnHotPath`.
- [SIMULATED] Modeled with stand-in data.
"""

_SAMPLE_CLAIMS_BAD = """
# Test claims

## Bad witnesses

- [SHIPPED] Missing package test. Witness: `go test ./nonexistent/pkg -run TestSomething`.
- [SHIPPED] Missing artifact. Witness: `experiments/missing-file.json` shows the result.
- [STUB] Plumbing present.
"""

_SAMPLE_BENCHMARKS_GOOD = """
# Benchmark authority

| Claim | Number | Artifact |
|---|---|---|
| Test result | 1.23x | `experiments/test-result.json` |

## Quick Reference

| Claim | Number | Commit | Artifact | Reproduce |
|---|---|---|---|---|
| Speedup | 4.58x | abc123 | `experiments/speedup.json` | `go run ./cmd/bench` |
"""

_SAMPLE_BENCHMARKS_BAD = """
# Benchmark authority

| Claim | Number | Artifact |
|---|---|---|
| Missing artifact | 1.23x | `experiments/missing.json` |

## Quick Reference

| Claim | Number | Commit | Artifact | Reproduce |
|---|---|---|---|---|
| Missing cmd | 4.58x | abc123 | `experiments/speedup.json` | `go run ./cmd/missing` |
"""


def test_sample_claims_good() -> None:
    """Good claims parse correctly with zero defects."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        # Create minimal structure
        (root / "internal").mkdir()
        (root / "internal" / "adjudicator").mkdir()

        # Create a test file with the test function
        test_file = root / "internal" / "adjudicator" / "adjudicator_test.go"
        test_file.write_text("""
package adjudicator

func TestNoOsExecOnHotPath(t *testing.T) {
}
""")

        result = crs._check_claims(_SAMPLE_CLAIMS_GOOD, root)
        assert result["kpi"] == "claims"
        assert result["score"] == 100
        assert len(result["defects"]) == 0
        assert "all falsifiable" in result["detail"]


def test_sample_claims_bad() -> None:
    """Bad claims with missing package and artifact."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        result = crs._check_claims(_SAMPLE_CLAIMS_BAD, root)
        assert result["kpi"] == "claims"
        assert result["score"] < 100
        assert len(result["defects"]) > 0
        # Should detect missing package and missing artifact
        defect_str = " ".join(result["defects"])
        assert "missing" in defect_str.lower()


def test_sample_benchmarks_good() -> None:
    """Good benchmarks parse correctly with zero defects."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        # Create experiments directory and artifact
        experiments = root / "experiments"
        experiments.mkdir()
        (experiments / "test-result.json").write_text("{}")
        (experiments / "speedup.json").write_text("{}")
        # Create cmd/bench
        cmd = root / "cmd"
        cmd.mkdir()
        (cmd / "bench").mkdir()

        result = crs._check_benchmarks(_SAMPLE_BENCHMARKS_GOOD, root)
        assert result["kpi"] == "benchmarks"
        assert result["score"] == 100
        assert len(result["defects"]) == 0


def test_sample_benchmarks_bad() -> None:
    """Bad benchmarks with missing artifact and cmd."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        result = crs._check_benchmarks(_SAMPLE_BENCHMARKS_BAD, root)
        assert result["kpi"] == "benchmarks"
        assert result["score"] < 100
        assert len(result["defects"]) > 0
        defect_str = " ".join(result["defects"])
        assert "missing" in defect_str.lower()


def test_claim_repro_scorecard_payload_structure() -> None:
    """Payload has correct schema structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        # Create minimal structure
        (root / "internal").mkdir()

        result = crs.collect(root)
        assert result["schema"] == crs.SCHEMA
        assert "ok" in result
        assert "verdict" in result
        assert "finding" in result
        assert "corpus" in result
        assert "kpis" in result
        assert len(result["kpis"]) == 2  # claims and benchmarks


def test_claim_repro_debt_counts_defects() -> None:
    """claim_repro_debt equals total defect count across KPIs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        result = crs.collect(root)
        corpus = result["corpus"]
        total_defects = sum(len(k["defects"]) for k in result["kpis"])
        assert corpus["claim_repro_debt"] == total_defects


def test_grade_letter() -> None:
    """Grade letter function works."""
    assert crs.grade_letter(95) == "A"
    assert crs.grade_letter(85) == "B"
    assert crs.grade_letter(75) == "C"
    assert crs.grade_letter(65) == "D"
    assert crs.grade_letter(55) == "F"


def test_repo_root() -> None:
    """repo_root returns correct path."""
    root = crs.repo_root()
    assert (root / "tools").exists()
    assert (root / "CLAIMS.md").exists()


def test_render() -> None:
    """render produces human-readable output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        result = crs.collect(root)
        output = crs.render(result)
        assert "claim-repro-scorecard:" in output
        assert "score" in output
        assert "CLAIM-REPRO-DEBT" in output


def test_render_markdown() -> None:
    """render_markdown produces markdown output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        result = crs.collect(root)
        output = crs.render_markdown(result, stamp="2026-06-28")
        assert "# Claim-reproducibility scorecard" in output
        assert "| **Un-falsifiable claims" in output
        assert "claim-repro-scorecard: 2026-06-28" in output


def test_json_output() -> None:
    """JSON output is valid."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        result = crs.collect(root)
        json_str = json.dumps(result, indent=2)
        parsed = json.loads(json_str)
        assert parsed["schema"] == crs.SCHEMA


def test_file_exists() -> None:
    """File existence check works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        test_file = root / "test.txt"
        test_file.write_text("test")
        assert crs._file_exists("test.txt", root)
        assert not crs._file_exists("nonexistent.txt", root)


def test_cmd_dir_exists() -> None:
    """cmd directory existence check works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        cmd = root / "cmd"
        cmd.mkdir()
        (cmd / "testcmd").mkdir()
        assert crs._cmd_dir_exists("testcmd", root)
        assert not crs._cmd_dir_exists("missing", root)


def test_package_exists() -> None:
    """Package existence check works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        pkg = root / "internal" / "testpkg"
        pkg.mkdir()
        assert crs._package_exists("internal/testpkg", root)
        assert not crs._package_exists("nonexistent/pkg", root)


def test_witness_extraction_patterns() -> None:
    """Witness extraction patterns match expected formats."""
    # Test pattern matching
    test_line = "Witness: `go test ./internal/pkg -run TestFoo`"
    assert crs._WITNESS_TEST_RE.search(test_line)

    cmd_line = "Witness: `go run ./cmd/testcmd`"
    assert crs._CMD_DIR_RE.search(cmd_line)

    artifact_line = "See `experiments/result.json`"
    assert crs._ARTIFACT_PATH_RE.search(artifact_line)


def test_empty_files() -> None:
    """Empty or missing files are handled gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Empty files
        result = crs._check_claims("", root)
        assert result["score"] == 100
        assert "skipped" in result["detail"]

        result = crs._check_benchmarks("", root)
        assert result["score"] == 100
        assert "skipped" in result["detail"]


def test_score_clamping() -> None:
    """Score clamping works."""
    assert crs._clamp(-10) == 0
    assert crs._clamp(0) == 0
    assert crs._clamp(50) == 50
    assert crs._clamp(100) == 100
    assert crs._clamp(150) == 100


def run_all() -> int:
    """Run all tests."""
    tests = [
        test_sample_claims_good,
        test_sample_claims_bad,
        test_sample_benchmarks_good,
        test_sample_benchmarks_bad,
        test_claim_repro_scorecard_payload_structure,
        test_claim_repro_debt_counts_defects,
        test_grade_letter,
        test_repo_root,
        test_render,
        test_render_markdown,
        test_json_output,
        test_file_exists,
        test_cmd_dir_exists,
        test_package_exists,
        test_witness_extraction_patterns,
        test_empty_files,
        test_score_clamping,
    ]

    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except AssertionError as e:
            print(f"FAIL {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"FAIL {test.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(run_all())