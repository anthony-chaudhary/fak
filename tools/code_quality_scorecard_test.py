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


def test_agent_checkout_go_files_excluded_from_corpus():
    # `.claude/worktrees/<wt>/` (worktree-isolation) and `.fak/tmp/issue<N>-clean-<sha>/`
    # (dispatch clean-checkout) are full repo CHECKOUTS the agent machinery leaves behind.
    # Walking them double-grades every copied .go as phantom debt (a `.fak/tmp` checkout
    # once inflated the architecture work-list 13 -> 27). Both must be pruned by name.
    assert cq._excluded_go(".claude/worktrees/wt/internal/gateway/metrics.go")
    assert cq._excluded_go(".fak/tmp/issue1155-clean-deadbeef/cmd/fak/guard.go")
    # first-party kernel paths are still graded
    assert not cq._excluded_go("internal/gateway/metrics.go")
    assert not cq._excluded_go("cmd/fak/guard.go")


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
    assert info["n_funcs"] == 1


# --- scanner hardening: literals & comments (review findings 1, 7, 8) ------

def test_code_only_blanks_string_and_rune_braces():
    code, raw, blk = cq._code_only('\ts := "}" + \'{\'  // }', False, False)
    assert "}" not in code and "{" not in code and raw is False and blk is False


def test_code_only_raw_backtick_spans_lines():
    code, raw, _ = cq._code_only("\tq := `SELECT {", False, False)
    assert raw is True and "{" not in code
    code2, raw2, _ = cq._code_only("} done`  + 1", True, False)
    assert raw2 is False and "}" not in code2


def test_code_only_block_comment_spans_lines():
    code, _, blk = cq._code_only("a /* { ", False, False)
    assert blk is True and "{" not in code
    code2, _, blk2 = cq._code_only(" } */ b", False, True)
    assert blk2 is False and "}" not in code2 and "b" in code2


def test_scanner_not_gamed_by_brace_in_literal():
    # the core gaming exploit: a `s := "}"` line near the top of a 250-line
    # god-function used to collapse it to length ~3 and erase architecture debt.
    body = '\ts := "}"\n' + "\tx := 1\n" * 250
    src = "package x\nfunc Big() {\n" + body + "}\n"
    info = cq.scan_go_file(src)
    assert info["long_funcs"] and info["long_funcs"][0][0] == "Big"
    assert info["long_funcs"][0][1] > cq.FUNC_HARD_MAX  # NOT collapsed


def test_scanner_balanced_iface_in_signature_no_early_break():
    src = ("package x\nfunc F(\n\tx interface{},\n\ty int,\n) {\n"
           + "\tz := 1\n" * (cq.FUNC_SOFT_MAX + 5) + "}\n")
    info = cq.scan_go_file(src)
    assert info["long_funcs"] and info["long_funcs"][0][0] == "F"


def test_scanner_one_line_body_not_run_into_following_decls():
    # Regression: a one-line function body `func f() { return x }` is net-zero on
    # its header line. The old per-line-net scan never saw depth go positive, so it
    # ran PAST the real `}` into the following package-level declarations and forged
    # a 200+-line "god-function" out of a one-liner (e.g. catalog.go:Offline). The
    # one-liner must measure as length 1, and the long block below it must NOT be
    # attributed to it.
    src = ("package x\n"
           "func One() bool { return true }\n"
           "var registry = []int{\n"
           + "\t1,\n" * (cq.FUNC_HARD_MAX + 50)
           + "}\n")
    info = cq.scan_go_file(src)
    longs = [(n, ln) for n, ln in info["long_funcs"] if ln > cq.FUNC_HARD_MAX]
    assert longs == [], f"one-liner forged a god-function: {longs}"


def test_scanner_real_one_line_body_then_real_god_function():
    # The fix must still SEE a genuine god-function that sits right after a
    # one-line wrapper (e.g. accounts.go: a one-line `cmdAccounts` then the real
    # 380-line `runAccounts`). The wrapper is length 1; the long one is flagged.
    src = ("package x\n"
           "func Wrap() { run() }\n"
           "func Big() {\n"
           + "\tz := 1\n" * (cq.FUNC_HARD_MAX + 5)
           + "}\n")
    info = cq.scan_go_file(src)
    names = {n for n, ln in info["long_funcs"] if ln > cq.FUNC_HARD_MAX}
    assert names == {"Big"}, names


def test_scanner_func_inside_raw_string_not_counted():
    src = "package x\nvar q = `\nfunc Fake() {\n`\n"
    info = cq.scan_go_file(src)
    assert info["n_funcs"] == 0
    assert all(n != "Fake" for _, n, _ in info["exported"])


# --- honesty: fenced code blocks (review finding 12) ----------------------

def test_honesty_skips_fenced_examples():
    txt = ("- [SHIPPED] real\n"
           "```\n"
           "- [ ] example inside a fence, not a ledger claim\n"
           "```\n"
           "- [STUB] also real\n")
    k = cq.kpi_honesty(txt)
    assert k["defects"] == [] and "2 claims" in k["detail"]


# --- deps: replace directives (review finding 3) --------------------------

def test_deps_replace_to_external_is_debt():
    mod = "module x\n\ngo 1.26\n\nreplace github.com/a/b => github.com/a/b-fork v1.0.0\n"
    k = cq.kpi_deps(mod, gosum_exists=False)
    assert len(k["defects"]) == 1 and "replace" in k["defects"][0]


def test_deps_local_replace_not_debt():
    mod = "module x\n\ngo 1.26\n\nreplace github.com/a/b => ./vendorlocal\n"
    k = cq.kpi_deps(mod, gosum_exists=False)
    assert k["defects"] == []


def test_external_replaces_block_and_local():
    mod = ("module x\nreplace (\n"
           "\tgithub.com/a/b => github.com/a/b-fork v1.2.3\n"
           "\tgithub.com/c/d => ../localcd\n)\n")
    assert cq._external_replaces(mod) == ["github.com/a/b-fork"]


# --- tests KPI: real test funcs (review finding 2) ------------------------

def test_testfunc_regex_distinguishes_real_from_empty():
    assert cq._TESTFUNC_RE.search("package foo\n") is None
    assert cq._TESTFUNC_RE.search("package foo\nfunc TestX(t *testing.T){}\n")
    assert cq._TESTFUNC_RE.search("package foo\nfunc BenchmarkY(b *testing.B){}\n")
    assert cq._TESTFUNC_RE.search("package foo\nfunc FuzzZ(f *testing.F){}\n")


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
