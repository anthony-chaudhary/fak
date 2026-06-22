#!/usr/bin/env python3
"""Unit tests for the code-quality scorecard. Pure KPIs take fixtures — no disk,
no toolchain — so the testable seam is exercised without Go installed.

Run:  python -m pytest tools/code_quality_scorecard_test.py -q
"""
from __future__ import annotations

import code_quality_scorecard as cq


# --- build / vet / format -------------------------------------------------

def test_build_ok_clean():
    k = cq.kpi_build(True, "")
    assert k["score"] == 100 and not k["defects"]


def test_build_fail_is_max_debt():
    k = cq.kpi_build(False, "internal/x/y.go:3:1: undefined: Foo")
    assert k["score"] == 0 and len(k["defects"]) == 1


def test_build_skipped_no_debt():
    k = cq.kpi_build(None, "")
    assert k["score"] == 100 and not k["defects"] and k["soft"]


def test_vet_counts_each_diag():
    k = cq.kpi_vet(False, ["a.go:1:1: x", "b.go:2:2: y"])
    assert len(k["defects"]) == 2 and k["score"] < 100


def test_format_each_unformatted_is_debt():
    k = cq.kpi_format(["a.go", "b.go"])
    assert len(k["defects"]) == 2
    assert cq.kpi_format([])["defects"] == []


# --- deps -----------------------------------------------------------------

def test_deps_clean_stdlib_only():
    k = cq.kpi_deps("module x\n\ngo 1.26\n", gosum_exists=False)
    assert k["score"] == 100 and not k["defects"]


def test_deps_external_require_is_debt():
    mod = "module x\n\ngo 1.26\n\nrequire github.com/foo/bar v1.2.3\n"
    k = cq.kpi_deps(mod, gosum_exists=True)
    # one external require + a present go.sum = 2 defects
    assert len(k["defects"]) == 2


def test_deps_require_block():
    mod = ("module x\n\ngo 1.26\n\nrequire (\n"
           "\tgithub.com/a/b v1.0.0\n\tgolang.org/x/c v0.1.0\n)\n")
    k = cq.kpi_deps(mod, gosum_exists=False)
    assert len(k["defects"]) == 2


# --- honesty --------------------------------------------------------------

def test_honesty_all_tagged_clean():
    txt = "- [SHIPPED] a\n- [STUB] b\n- [SIMULATED] c\n"
    k = cq.kpi_honesty(txt)
    assert k["score"] == 100 and not k["defects"]


def test_honesty_untagged_is_debt():
    txt = "- [SHIPPED] a\n- [ ] b has no tag\n- [SHIPPED][STUB] c double\n"
    k = cq.kpi_honesty(txt)
    assert len(k["defects"]) == 2  # the untagged and the double-tagged


# --- architecture ---------------------------------------------------------

def test_architecture_god_file_and_func():
    files = [
        {"path": "a.go", "n_lines": cq.FILE_HARD_MAX + 1, "long_funcs": []},
        {"path": "b.go", "n_lines": 50, "long_funcs": [("Big", cq.FUNC_HARD_MAX + 5)]},
        {"path": "c.go", "n_lines": cq.FILE_SOFT_MAX + 1, "long_funcs": []},  # soft only
    ]
    k = cq.kpi_architecture(files)
    assert len(k["defects"]) == 2          # one god-file + one god-func
    assert any("large file" in s for s in k["soft"])  # c.go is a soft nudge


def test_architecture_clean():
    k = cq.kpi_architecture([{"path": "a.go", "n_lines": 100, "long_funcs": []}])
    assert not k["defects"] and k["score"] == 100


# --- tests ----------------------------------------------------------------

def test_tests_untested_packages_are_debt():
    k = cq.kpi_tests(["internal/x", "internal/y"], n_packages=10)
    assert len(k["defects"]) == 2 and k["score"] == 80


# --- hygiene --------------------------------------------------------------

def test_hygiene_is_soft_never_hard():
    k = cq.kpi_hygiene([("a.go", 2), ("b.go", cq.HYGIENE_CAP_PER_FILE)])
    assert k["defects"] == []          # advisory only, never hard debt
    assert len(k["soft"]) == 2 and k["score"] < 100
    assert cq.kpi_hygiene([])["score"] == 100


# --- godoc (soft only) ----------------------------------------------------

def test_godoc_is_soft_never_hard():
    k = cq.kpi_godoc(10, 4, ["a.go:Foo (func)"])
    assert k["defects"] == []            # never hard debt
    assert k["score"] == 40 and k["soft"]


# --- ship integrity (DOS) -------------------------------------------------

def test_ship_integrity_residual_is_debt():
    dos = {"rev_range": "HEAD~5..HEAD", "checkable": 5, "cleared_rate": 0.8,
           "residual": [{"sha": "abc", "subject": "feat: x"}]}
    k = cq.kpi_ship_integrity(dos)
    assert len(k["defects"]) == 1 and k["score"] == 75


def test_ship_integrity_clean():
    dos = {"rev_range": "HEAD~5..HEAD", "checkable": 5, "cleared_rate": 1.0,
           "residual": []}
    k = cq.kpi_ship_integrity(dos)
    assert not k["defects"] and k["score"] == 100


def test_ship_integrity_unavailable_is_soft():
    k = cq.kpi_ship_integrity({"error": "dos not found"})
    assert not k["defects"] and k["soft"]


# --- fold -----------------------------------------------------------------

def test_payload_sums_debt_and_gates_ok():
    kpis = [
        cq.kpi_build(True, ""),
        cq.kpi_format(["a.go"]),
        cq.kpi_deps("module x\ngo 1.26\n", False),
    ]
    p = cq.build_payload(workspace="/x", kpis=kpis)
    assert p["corpus"]["code_debt"] == 1
    assert p["ok"] is False and p["verdict"] == "ACTION"


def test_payload_clean_is_ok():
    kpis = [cq.kpi_build(True, ""), cq.kpi_format([])]
    p = cq.build_payload(workspace="/x", kpis=kpis)
    assert p["corpus"]["code_debt"] == 0 and p["ok"] is True


# --- the brace-depth scanner ----------------------------------------------

def test_scan_counts_function_length_and_exports():
    src = (
        "package x\n"
        "\n"
        "// Foo is documented.\n"
        "func Foo() {\n"
        + "\ta := 1\n" * 5 +
        "}\n"
        "\n"
        "func bar() {\n"      # unexported, short
        "\treturn\n"
        "}\n"
        "\n"
        "type Public struct{}\n"   # undocumented exported type
    )
    info = cq.scan_go_file(src)
    names = {n for _, n, _ in info["exported"]}
    assert "Foo" in names and "Public" in names and "bar" not in names
    documented = {n: d for _, n, d in info["exported"]}
    assert documented["Foo"] is True and documented["Public"] is False


def test_scan_flags_long_function():
    body = "\ta := 1\n" * (cq.FUNC_SOFT_MAX + 10)
    src = "package x\nfunc Big() {\n" + body + "}\n"
    info = cq.scan_go_file(src)
    assert info["long_funcs"] and info["long_funcs"][0][0] == "Big"
    assert info["long_funcs"][0][1] > cq.FUNC_SOFT_MAX


if __name__ == "__main__":
    import subprocess
    import sys
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-q"]))
