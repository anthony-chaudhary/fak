#!/usr/bin/env python3
"""Tests for the SOTA-coverage scorecard  -  the prior-art-matrix stick.

Drives the PURE parsing + fold with in-memory fixtures (a clean synthetic matrix,
and the defect cases each HARD KPI catches: a row whose FakPath points at deleted
code, a row with no link, a row with no oracle, a kernel file no row covers), then
closes with the load-bearing LIVE smoke against the real tracked tree: the matrix
source must parse to >0 rows, every row must point at code that exists with a link
and an oracle, and the headline number must be a non-negative int whose grade is
consistent with its debt  -  the proof the matrix is real, and a regression sentinel
for the day a row starts pointing at deleted code.

Run: `python tools/sota_coverage_scorecard_test.py`  (exit 0 = all pass),
or `python -m pytest tools/sota_coverage_scorecard_test.py -q`.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sota_coverage_scorecard as sc  # noqa: E402

ROOT = sc.repo_root()


# --- a synthetic matrix source the parser can chew on -----------------------

# The synthetic kernel files must use names the tightened KERNEL_PATHSPECS actually
# treat as kernels (cpuref.go is a named compute kernel; moe*.go is a model kernel
# lane) - otherwise the fixture's "kernels" are invisible to tree_coverage and the
# clean matrix would (wrongly) read "no kernel files found".
CLEAN_MATRIX = '''package sotamatrix
// Provenance: docs/notes/RESEARCH-backend-sota-matrix-2026-06-26.md
var matrix = []Op{
\t{
\t\tSlug:        "alpha",
\t\tFileGlobs:   []string{"internal/compute/cpuref.go"},
\t\tFakPath:     "internal/compute/cpuref.go (Reference) + internal/model/x.go",
\t\tPrimaryLink: "https://example.com/alpha",
\t\tOracle:      "cpuref bit-identity",
\t},
\t{
\t\tSlug:        "beta",
\t\tFileGlobs:   []string{"internal/model/moe_beta*.go"},
\t\tFakPath:     "internal/model/moe_beta.go:42 (loader)",
\t\tPrimaryLink: "https://example.com/beta",
\t\tOracle:      "HF reference",
\t},
}
'''


def _mk_repo(matrix_src: str) -> Path:
    """A throwaway git repo with a go.mod, the matrix source, and the two kernel
    files the clean matrix points at + covers. The file names are chosen to match
    KERNEL_PATHSPECS (cpuref.go, moe_beta.go) so tree_coverage sees them as the two
    kernels the matrix is responsible for."""
    d = Path(tempfile.mkdtemp())
    (d / "go.mod").write_text("module x\n", encoding="utf-8")
    (d / "cmd").mkdir()  # repo_root() wants a cmd/ dir
    md = d / "internal" / "sotamatrix"
    md.mkdir(parents=True)
    (md / "sotamatrix.go").write_text(matrix_src, encoding="utf-8")
    comp = d / "internal" / "compute"
    comp.mkdir(parents=True)
    (comp / "cpuref.go").write_text("package compute\n", encoding="utf-8")
    model = d / "internal" / "model"
    model.mkdir(parents=True)
    (model / "x.go").write_text("package model\n", encoding="utf-8")
    (model / "moe_beta.go").write_text("package model\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=d, creationflags=sc._no_window())
    subprocess.run(["git", "add", "-A"], cwd=d, creationflags=sc._no_window())
    return d


# --- pure helpers -----------------------------------------------------------

def test_grade_bands() -> None:
    assert sc.grade_letter(0) == "A"
    assert sc.grade_letter(1) == "B"
    assert sc.grade_letter(2) == "B"
    assert sc.grade_letter(4) == "C"
    assert sc.grade_letter(9) == "D"
    assert sc.grade_letter(40) == "F"


def test_parse_matrix_pulls_fields() -> None:
    rows = sc.parse_matrix(CLEAN_MATRIX)
    assert [r["slug"] for r in rows] == ["alpha", "beta"], rows
    alpha = rows[0]
    assert alpha["fak_path_file"] == "internal/compute/cpuref.go", alpha
    assert alpha["primary_link"] == "https://example.com/alpha"
    assert alpha["oracle"] == "cpuref bit-identity"
    assert "internal/compute/cpuref.go" in alpha["file_globs"]
    # a `:LINE` suffix is stripped to the bare file
    assert rows[1]["fak_path_file"] == "internal/model/moe_beta.go", rows[1]


def test_first_fakpath_handles_directory_pointer() -> None:
    # A FakPath that leads with a DIRECTORY (the metalgemm row) is a valid pointer
    # to real code, not an unparseable path.
    assert sc._first_fakpath_file("internal/metalgemm/; internal/model/q.go") == "internal/metalgemm"
    assert sc._first_fakpath_file("internal/compute/cuda.go:1101 (AWQMatMul)") == "internal/compute/cuda.go"


def test_glob_match_normalizes_separators() -> None:
    assert sc.covered_by_matrix("internal\\model\\beta.go", ["internal/model/beta*.go"])
    assert not sc.covered_by_matrix("internal/model/gamma.go", ["internal/model/beta*.go"])


def test_clean_matrix_has_no_hard_debt() -> None:
    d = _mk_repo(CLEAN_MATRIX)
    payload = sc.collect(d, today="2026-06-30")
    assert not payload.get("error"), payload.get("error")
    c = payload["corpus"]
    assert c["hard_debt"] == 0, payload["kpis"]
    assert c["grade"] == "A", c
    assert payload["ok"]


def test_missing_fak_path_raises_debt() -> None:
    # The headline regression: a row whose FakPath points at code that was deleted.
    broken = CLEAN_MATRIX.replace("internal/compute/cpuref.go (Reference)",
                                  "internal/compute/DELETED.go (Reference)")
    d = _mk_repo(broken)
    payload = sc.collect(d, today="2026-06-30")
    by = {k["name"]: k for k in payload["kpis"]}
    fpe = by["fak_path_exists"]
    assert not fpe["passed"], fpe
    assert fpe["debt"] == 1, fpe
    assert payload["corpus"]["hard_debt"] >= 1
    assert not payload["ok"]


def test_missing_link_and_oracle_raise_debt() -> None:
    broken = CLEAN_MATRIX.replace('PrimaryLink: "https://example.com/beta"',
                                  'PrimaryLink: ""')
    broken = broken.replace('Oracle:      "HF reference"', 'Oracle:      ""')
    d = _mk_repo(broken)
    payload = sc.collect(d, today="2026-06-30")
    by = {k["name"]: k for k in payload["kpis"]}
    assert not by["has_primary_link"]["passed"] and by["has_primary_link"]["debt"] == 1
    assert not by["has_oracle"]["passed"] and by["has_oracle"]["debt"] == 1


def test_uncovered_kernel_file_raises_tree_coverage_debt() -> None:
    # Add a kernel file the matrix's FileGlobs do NOT cover  -  the blind-spot defect.
    d = _mk_repo(CLEAN_MATRIX)
    (d / "internal" / "model" / "moe.go").write_text("package model\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=d, creationflags=sc._no_window())
    payload = sc.collect(d, today="2026-06-30")
    by = {k["name"]: k for k in payload["kpis"]}
    tc = by["tree_coverage"]
    assert not tc["passed"], tc
    assert tc["debt"] >= 1, tc
    assert any("moe.go" in it for it in tc["items"]), tc["items"]


def test_freshness_is_soft_and_needs_today() -> None:
    # No --today: freshness passes (cannot evaluate, stays deterministic).
    d = _mk_repo(CLEAN_MATRIX)
    payload = sc.collect(d, today="")
    by = {k["name"]: k for k in payload["kpis"]}
    assert by["freshness"]["passed"], by["freshness"]
    assert not by["freshness"]["hard"]
    # A far-future --today pushes the 2026-06-26 note past the 90d window  -  soft debt.
    stale = sc.collect(d, today="2027-01-01")
    sby = {k["name"]: k for k in stale["kpis"]}
    assert not sby["freshness"]["passed"], sby["freshness"]
    assert sby["freshness"]["debt"] == 1
    # but a SOFT defect does not flip the HARD-only ok gate
    assert stale["corpus"]["hard_debt"] == 0
    assert stale["ok"]


# --- the load-bearing live smoke against the real tracked tree --------------

def test_live_matrix_parses_and_rows_present() -> None:
    payload = sc.collect(ROOT, today="2026-06-30")
    assert not payload.get("error"), payload.get("error")
    assert payload["corpus"]["matrix_rows"] >= 5, payload["corpus"]


def test_live_complete_group_is_clean() -> None:
    # Every real matrix row must point at code that exists, with a link and oracle.
    payload = sc.collect(ROOT, today="2026-06-30")
    by = {k["name"]: k for k in payload["kpis"]}
    assert by["fak_path_exists"]["passed"], by["fak_path_exists"]["items"]
    assert by["has_primary_link"]["passed"], by["has_primary_link"]["items"]
    assert by["has_oracle"]["passed"], by["has_oracle"]["items"]
    assert payload["corpus"]["debt_by_group"]["complete"] == 0, payload["corpus"]


def test_live_debt_and_grade_consistent() -> None:
    payload = sc.collect(ROOT, today="2026-06-30")
    c = payload["corpus"]
    assert isinstance(c["sota_debt"], int) and c["sota_debt"] >= 0
    assert c["grade"] == sc.grade_letter(c["sota_debt"]), c
    # ok is the HARD-only gate; it must agree with hard_debt.
    assert payload["ok"] == (c["hard_debt"] == 0)


def test_json_roundtrips() -> None:
    payload = sc.collect(ROOT, today="2026-06-30")
    assert json.loads(json.dumps(payload))["schema"] == sc.SCHEMA


def _run() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERR  {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
