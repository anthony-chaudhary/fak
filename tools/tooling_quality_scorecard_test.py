#!/usr/bin/env python3
"""Unit tests for the tooling-quality scorecard. Pure KPIs take fixtures — no
disk, no ruff — so the testable seam is exercised without the toolchain.

Run:  python -m pytest tools/tooling_quality_scorecard_test.py -q
"""
from __future__ import annotations

import tooling_quality_scorecard as tq


# --- lint -----------------------------------------------------------------

def test_lint_clean():
    k = tq.kpi_lint([])
    assert k["score"] == 100 and not k["defects"]


def test_lint_each_diag_is_debt():
    diags = [
        {"code": "F401", "filename": "/x/tools/a.py", "location": {"row": 3},
         "message": "`os` imported but unused"},
        {"code": "F821", "filename": "/x/tools/b.py", "location": {"row": 7},
         "message": "undefined name `foo`"},
    ]
    k = tq.kpi_lint(diags)
    assert len(k["defects"]) == 2 and k["score"] < 100
    # path is normalised to repo-relative tools/...
    assert any(d.startswith("ruff F401 tools/a.py:3:") for d in k["defects"])


def test_lint_skipped_when_ruff_absent():
    k = tq.kpi_lint(None)
    assert k["score"] == 100 and not k["defects"] and k["soft"]


# --- architecture ---------------------------------------------------------

def test_architecture_god_file_and_func():
    files = [
        {"path": "tools/a.py", "n_lines": tq.FILE_HARD_MAX + 1, "long_funcs": []},
        {"path": "tools/b.py", "n_lines": 50, "long_funcs": [("big", tq.FUNC_HARD_MAX + 5)]},
        {"path": "tools/c.py", "n_lines": tq.FILE_SOFT_MAX + 1, "long_funcs": []},  # soft only
    ]
    k = tq.kpi_architecture(files)
    assert len(k["defects"]) == 2          # one god-file + one god-func
    assert any("large file" in s for s in k["soft"])  # c.py is a soft nudge


def test_architecture_clean():
    k = tq.kpi_architecture([{"path": "tools/a.py", "n_lines": 100, "long_funcs": []}])
    assert not k["defects"] and k["score"] == 100


# --- tests ----------------------------------------------------------------

def test_tests_untested_modules_are_debt():
    k = tq.kpi_tests(["tools/x.py", "tools/y.py"], n_modules=10)
    assert len(k["defects"]) == 2 and k["score"] == 80


def test_tests_all_covered_is_clean():
    k = tq.kpi_tests([], n_modules=10)
    assert not k["defects"] and k["score"] == 100


# --- format (soft only) ---------------------------------------------------

def test_format_is_soft_never_hard():
    k = tq.kpi_format(300, n_total=301)
    assert k["defects"] == []          # advisory only, never hard debt
    assert k["soft"] and k["score"] < 100


def test_format_clean_and_skipped():
    assert tq.kpi_format(0, n_total=10)["score"] == 100
    skipped = tq.kpi_format(None, n_total=10)
    assert skipped["score"] == 100 and skipped["soft"]


# --- docstrings (soft only) -----------------------------------------------

def test_docstrings_is_soft_never_hard():
    k = tq.kpi_docstrings(10, 4, ["tools/a.py"])
    assert k["defects"] == []            # never hard debt
    assert k["score"] == 40 and k["soft"]


def test_docstrings_extra_uncounted_noted():
    k = tq.kpi_docstrings(10, 4, ["tools/a.py"])  # 6 undocumented, 1 sampled
    assert any("5 more" in s for s in k["soft"])


# --- fold -----------------------------------------------------------------

def test_payload_sums_debt_and_gates_ok():
    kpis = [
        tq.kpi_lint([{"code": "F401", "filename": "tools/a.py",
                      "location": {"row": 1}, "message": "unused"}]),
        tq.kpi_architecture([{"path": "tools/a.py", "n_lines": 100, "long_funcs": []}]),
        tq.kpi_tests([], n_modules=5),
        tq.kpi_format(0, n_total=5),
        tq.kpi_docstrings(5, 5, []),
    ]
    p = tq.build_payload(workspace="/x", kpis=kpis)
    assert p["corpus"]["py_debt"] == 1
    assert p["ok"] is False and p["verdict"] == "ACTION"


def test_payload_clean_is_ok():
    kpis = [tq.kpi_lint([]), tq.kpi_architecture([]), tq.kpi_tests([], 0),
            tq.kpi_format(0, 0), tq.kpi_docstrings(0, 0, [])]
    p = tq.build_payload(workspace="/x", kpis=kpis)
    assert p["corpus"]["py_debt"] == 0 and p["ok"] is True
    assert p["finding"] == "tooling_clean"


def test_soft_kpis_emit_no_hard_debt_in_fold():
    # a tree with only soft signals (format divergence + undocumented) is OK
    kpis = [tq.kpi_lint([]), tq.kpi_architecture([]), tq.kpi_tests([], 0),
            tq.kpi_format(300, 301), tq.kpi_docstrings(10, 0, ["tools/a.py"])]
    p = tq.build_payload(workspace="/x", kpis=kpis)
    assert p["corpus"]["py_debt"] == 0 and p["ok"] is True
    assert p["corpus"]["soft_signals"] >= 2


# --- the ast scanner ------------------------------------------------------

def test_scan_counts_defs_and_module_doc():
    src = (
        '"""Module doc."""\n'
        "def foo():\n"
        "    return 1\n"
        "\n"
        "async def bar():\n"
        "    return 2\n"
    )
    info = tq.scan_py_file(src)
    assert info["n_defs"] == 2 and info["module_doc"] is True
    assert info["long_funcs"] == []


def test_scan_flags_long_function_exactly():
    body = "    x = 1\n" * (tq.FUNC_SOFT_MAX + 10)
    src = "def big():\n" + body
    info = tq.scan_py_file(src)
    assert info["long_funcs"] and info["long_funcs"][0][0] == "big"
    assert info["long_funcs"][0][1] > tq.FUNC_SOFT_MAX


def test_scan_counts_nested_and_method_defs():
    src = (
        "class C:\n"
        "    def m(self):\n"
        "        def inner():\n"
        "            return 1\n"
        "        return inner\n"
    )
    info = tq.scan_py_file(src)
    assert info["n_defs"] == 2          # method + nested function both counted


def test_scan_syntax_error_counts_lines_not_defs():
    src = "def broken(:\n    pass\n"   # invalid syntax
    info = tq.scan_py_file(src)
    assert info["n_defs"] == 0 and info["module_doc"] is False
    assert info["n_lines"] == 2        # raw lines still counted, file not dropped


def test_scan_no_module_doc():
    info = tq.scan_py_file("import os\nx = os.getcwd()\n")
    assert info["module_doc"] is False


# --- file classification --------------------------------------------------

def test_is_test_file_both_conventions():
    assert tq.is_test_file("foo_test.py")
    assert tq.is_test_file("test_foo.py")
    assert not tq.is_test_file("foo.py")
    assert not tq.is_test_file("sub/dir/bar.py")


def test_excluded_py():
    assert tq._excluded_py("__init__.py")
    assert tq._excluded_py("conftest.py")
    assert tq._excluded_py("__pycache__/x.py")
    assert not tq._excluded_py("real_tool.py")
    # agent-machinery repo COPIES (`.dos/_dos_park/_iso_build/`, `.fak/tmp/`,
    # `.claude/worktrees/`) must be pruned or every copied .py double-counts as debt.
    assert tq._excluded_py(".dos/_dos_park/_iso_build/tools/foo.py")
    assert tq._excluded_py(".fak/tmp/issue1-clean-abc/tools/foo.py")
    assert tq._excluded_py(".claude/worktrees/wt/tools/foo.py")
    # a cp1252<->utf-8 mojibake phantom path (U+F05C backslash-glyph) is never source.
    assert tq._excluded_py(chr(0xF05C) + "/tools/foo.py")
    assert not tq._excluded_py("tools/real_tool.py")


# --- compare --------------------------------------------------------------

def test_render_compare_reports_ratio_and_verdict():
    base = {"corpus": {"py_debt": 20, "score": 60.0,
                       "debt_by_kpi": {"lint": 10, "tests": 10}}}
    cur = {"corpus": {"py_debt": 8, "score": 78.0,
                      "debt_by_kpi": {"lint": 3, "tests": 5}}}
    out = tq.render_compare(base, cur)
    assert "20 -> 8" in out
    assert "2x py-debt reduction achieved" in out  # 8 <= 20//2


def test_render_compare_not_yet_2x():
    base = {"corpus": {"py_debt": 20, "score": 60.0, "debt_by_kpi": {}}}
    cur = {"corpus": {"py_debt": 15, "score": 65.0, "debt_by_kpi": {}}}
    out = tq.render_compare(base, cur)
    assert "not yet 2x" in out


def main() -> int:
    """Pure-stdlib runner: collects and runs every module-level test_* function so the
    suite runs in the pytest-free CI exactly like its scorecard-family siblings
    (observability/doc_appeal/...). `python -m pytest <file>` still discovers the same
    tests; this only changes what a direct `python <file>` invocation does."""
    tests = sorted((n, f) for n, f in globals().items()
                   if n.startswith("test_") and callable(f))
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL {name}: {exc}")
    print(f"{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
