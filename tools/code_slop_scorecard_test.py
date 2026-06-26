#!/usr/bin/env python3
"""Unit tests for the code-slop scorecard. Pure KPIs take in-memory fixtures — no
disk, no toolchain, no git — so the testable seam runs anywhere. Each test locks in
one calibration decision (especially the false-positive guards: a benchmark is not a
vacuous test, an interface guard is an assertion, a shell example is not commented-out
code, struct fields are not clones).

Dual-runnable (the repo runs the suite pytest-free in CI):
    python tools/code_slop_scorecard_test.py
    python -m pytest tools/code_slop_scorecard_test.py -q
"""
from __future__ import annotations

import builtins

import code_slop_scorecard as cs


# --- duplication ----------------------------------------------------------

def _dup_block(name: str) -> str:
    # an 8-line function body that is identical across files -> a real clone
    return (f"func {name}(xs []int) int {{\n"
            "\ttotal := 0\n"
            "\tfor _, v := range xs {\n"
            "\t\tif v > 0 {\n"
            "\t\t\ttotal += v * 2\n"
            "\t\t\ttotal -= v % 3\n"
            "\t\t}\n"
            "\t}\n"
            "\treturn total\n"
            "}\n")


def test_duplication_real_clone_is_debt():
    files = {"a.go": "package a\n" + _dup_block("sum"),
             "b.go": "package b\n" + _dup_block("sum")}
    k = cs.kpi_duplication(files)
    assert len(k["defects"]) >= 1
    assert k["score"] < 100


def test_duplication_unique_code_is_clean():
    files = {"a.go": "package a\n" + _dup_block("sum"),
             "b.go": ("package b\n"
                      "func other(s string) string {\n"
                      "\tr := s + s\n"
                      "\treturn r\n"
                      "}\n")}
    k = cs.kpi_duplication(files)
    assert k["defects"] == []
    assert k["score"] == 100


def test_duplication_token_window_keys_are_collision_exact():
    # The clone engine must not reduce token windows to a lossy integer hash. Force every
    # explicit hash() call to collide; distinct token windows should still stay distinct
    # because kpi_duplication keys by the token tuple itself.
    original_hash = builtins.hash
    builtins.hash = lambda _: 1
    try:
        files = {"a.go": "package a\n" + _dup_block("sum"),
                 "b.go": ("package b\n"
                          "func other(s string, xs []string) string {\n"
                          "\tbuf := s\n"
                          "\tfor _, part := range xs {\n"
                          "\t\tif len(part) > 3 {\n"
                          "\t\t\tbuf = buf + \":\" + part\n"
                          "\t\t} else {\n"
                          "\t\t\tbuf = part + \":\" + buf\n"
                          "\t\t}\n"
                          "\t}\n"
                          "\treturn buf\n"
                          "}\n")}
        k = cs.kpi_duplication(files)
    finally:
        builtins.hash = original_hash
    assert k["defects"] == []
    assert k["score"] == 100


def test_duplication_ignores_package_and_import_boilerplate():
    # two files with identical package+import preamble but distinct bodies must NOT
    # register as a clone — that's language boilerplate, not copy-paste.
    pre = "package x\n\nimport (\n\t\"fmt\"\n\t\"os\"\n\t\"strings\"\n\t\"bytes\"\n)\n"
    files = {"a.go": pre + "func aaa() { fmt.Println(1) }\n",
             "b.go": pre + "func bbb() { os.Exit(2) }\n"}
    k = cs.kpi_duplication(files)
    assert k["defects"] == []


def test_duplication_ignores_struct_field_lists():
    # two structs sharing field names are a data-shape overlap, not a logic clone.
    s1 = ("package a\ntype Report struct {\n\tName string\n\tID int\n"
          "\tHash string\n\tSeed int64\n\tCount int\n\tCost float64\n}\n")
    s2 = ("package b\ntype Curve struct {\n\tName string\n\tID int\n"
          "\tHash string\n\tSeed int64\n\tCount int\n\tCost float64\n}\n")
    k = cs.kpi_duplication({"a.go": s1, "b.go": s2})
    assert k["defects"] == []


def test_duplication_ignores_composite_literal_fields():
    lit = ("package {p}\nfunc make{n}() T {{\n\treturn T{{\n"
           "\t\tName: \"x\",\n\t\tID: 1,\n\t\tHash: \"h\",\n"
           "\t\tSeed: 2,\n\t\tCount: 3,\n\t\tCost: 4.0,\n\t}}\n}}\n")
    files = {"a.go": lit.format(p="a", n="A"),
             "b.go": lit.format(p="b", n="B")}
    k = cs.kpi_duplication(files)
    assert k["defects"] == []


def test_duplication_catches_reformatted_clone():
    # The token-stream engine (#780) is line-break invariant: the same code wrapped
    # across a DIFFERENT set of lines (binary operators at line end suppress Go's
    # semicolon insertion, so no token changes) is one normalized token stream. The old
    # line-shingle engine keyed on per-line text and MISSED such a reformatted clone;
    # the token engine catches it. Both copies use identical identifiers — only the line
    # layout differs.
    compact = ("package a\n"
               "func tally(a, b, c, d, e, f int) int {\n"
               "\tr := a + b + c + d + e + f\n"
               "\ts := a*b + c*d + e*f\n"
               "\tu := r - s + a - b + c - d\n"
               "\treturn u + r - s\n"
               "}\n")
    wrapped = ("package b\n"
               "func tally(a, b, c, d, e, f int) int {\n"
               "\tr := a + b +\n\t\tc + d +\n\t\te + f\n"
               "\ts := a*b +\n\t\tc*d +\n\t\te*f\n"
               "\tu := r - s + a -\n\t\tb + c - d\n"
               "\treturn u + r - s\n"
               "}\n")
    k = cs.kpi_duplication({"a.go": compact, "b.go": wrapped})
    assert len(k["defects"]) >= 1
    assert k["score"] < 100


# --- go_tokens (the #780 token-stream foundation) -------------------------

def test_go_tokens_literal_is_one_token():
    # A string/raw/number literal is ONE `L` token — code-like content inside a string
    # (`if`, `:=`, braces) is NEVER parsed as code. This is the precision win over the
    # line engine, whose blanked string husks could stitch unrelated code into a clone.
    src = ('package p\n'
           'func f() {\n'
           '\tx := "if y := 0 { return }"\n'
           '\ty := 1 + 2\n'
           '}\n')
    syms = [t[0] for t in cs.go_tokens(src, normalize_idents=False)]
    assert "if" not in syms and "return" not in syms  # those live inside the string
    assert syms.count("L") == 3  # the string, the 1, the 2
    assert ":=" in syms and "func" in syms  # real structure is kept


def test_go_tokens_identifier_normalization_and_logic_flag():
    toks = cs.go_tokens("a := bcd + e", normalize_idents=True)
    assert [t[0] for t in toks] == ["I", ":=", "I", "+", "I"]
    # the `:=` and `+` are logic tokens; the identifiers/keywords are not
    assert [t[2] for t in toks] == [False, True, False, True, False]


# --- vacuous_tests --------------------------------------------------------

def test_vacuous_test_no_assertion_is_debt():
    tf = {"x_test.go": ("package x\n"
                        "func TestThing(t *testing.T) {\n"
                        "\t_ = Compute()\n"
                        "}\n")}
    k = cs.kpi_vacuous_tests(tf)
    assert len(k["defects"]) == 1
    assert "TestThing" in k["defects"][0]


def test_test_with_assertion_is_clean():
    tf = {"x_test.go": ("package x\n"
                        "func TestThing(t *testing.T) {\n"
                        "\tif Compute() != 4 {\n\t\tt.Fatalf(\"bad\")\n\t}\n"
                        "}\n")}
    k = cs.kpi_vacuous_tests(tf)
    assert k["defects"] == []
    assert k["score"] == 100


def test_benchmark_is_not_vacuous():
    # a benchmark's job is to time, not assert — it must NOT be flagged.
    tf = {"x_test.go": ("package x\n"
                        "func BenchmarkThing(b *testing.B) {\n"
                        "\tb.ResetTimer()\n"
                        "\tfor i := 0; i < b.N; i++ {\n\t\t_ = Compute()\n\t}\n"
                        "}\n")}
    k = cs.kpi_vacuous_tests(tf)
    assert k["defects"] == []


def test_interface_guard_is_an_assertion():
    # a compile-time interface guard fails the BUILD if unsatisfied — a real assertion.
    tf = {"x_test.go": ("package x\n"
                        "func TestResolverInterface(t *testing.T) {\n"
                        "\tvar _ abi.Resolver = (*Store)(nil)\n"
                        "}\n")}
    k = cs.kpi_vacuous_tests(tf)
    assert k["defects"] == []


def test_table_test_with_subrun_is_clean():
    tf = {"x_test.go": ("package x\n"
                        "func TestCases(t *testing.T) {\n"
                        "\tfor _, c := range cases {\n"
                        "\t\tt.Run(c.name, func(t *testing.T) { check(t, c) })\n"
                        "\t}\n"
                        "}\n")}
    k = cs.kpi_vacuous_tests(tf)
    assert k["defects"] == []


# --- dead_code ------------------------------------------------------------

def test_dead_unexported_symbol_is_debt():
    files = {"a.go": ("package a\n"
                      "func used() int { return 1 }\n"
                      "func deadHelper() int { return 2 }\n"
                      "func Entry() int { return used() }\n")}
    k = cs.kpi_dead_code(files, {})
    names = " ".join(k["defects"])
    assert "deadHelper" in names
    assert "used" not in names  # referenced by Entry
    assert "Entry" not in names  # exported, not graded


def test_referenced_unexported_is_clean():
    files = {"a.go": ("package a\n"
                      "func helper() int { return 7 }\n"
                      "func Run() int { return helper() + helper() }\n")}
    k = cs.kpi_dead_code(files, {})
    assert k["defects"] == []


def test_dead_code_sees_test_references():
    # a helper used only by a _test.go is NOT dead.
    files = {"a.go": "package a\nfunc helper() int { return 1 }\n"}
    tests = {"a_test.go": ("package a\n"
                           "func TestH(t *testing.T) { if helper() != 1 { t.Fatal() } }\n")}
    k = cs.kpi_dead_code(files, tests)
    assert k["defects"] == []


def test_dead_code_capped_per_file():
    body = "package a\n" + "".join(f"func dead{i}() {{}}\n" for i in range(20))
    k = cs.kpi_dead_code({"a.go": body}, {})
    assert len(k["defects"]) <= cs.DEAD_CAP_PER_FILE
    assert k["soft"]  # the cap is surfaced as an advisory


# --- comment_slop ---------------------------------------------------------

def test_commented_out_code_block_is_debt():
    files = {"a.go": ("package a\n"
                      "func f() {\n"
                      "\t// x := compute()\n"
                      "\t// if x > 0 {\n"
                      "\t// \treturn x\n"
                      "\t// }\n"
                      "\treturn\n"
                      "}\n")}
    k = cs.kpi_comment_slop(files)
    assert any("commented-out code" in d for d in k["defects"])


def test_shell_example_comment_is_not_commented_code():
    # a doc-comment usage example must NOT be flagged as commented-out code.
    files = {"a.go": ("// Package x does things.\n"
                      "//\n"
                      "//\tgo run ./cmd/x --flag\n"
                      "//\tgo run ./cmd/x -json\n"
                      "//\t$ fak serve\n"
                      "package x\n")}
    k = cs.kpi_comment_slop(files)
    assert not any("commented-out code" in d for d in k["defects"])


def test_tautological_doc_comment_is_debt():
    files = {"a.go": ("package a\n"
                      "// Foo does foo.\n"
                      "func Foo() {}\n")}
    k = cs.kpi_comment_slop(files)
    assert any("tautological" in d for d in k["defects"])


def test_substantive_doc_comment_is_clean():
    files = {"a.go": ("package a\n"
                      "// Foo splices the prefix through the last cache breakpoint so the\n"
                      "// upstream cache hit survives a history rewrite.\n"
                      "func Foo() {}\n")}
    k = cs.kpi_comment_slop(files)
    assert not any("tautological" in d for d in k["defects"])


# --- stub_masquerade (SOFT) -----------------------------------------------

def test_stub_masquerade_never_emits_hard_debt():
    files = {"a.go": ("package a\n"
                      "func Future() int { panic(\"not implemented\") }\n")}
    k = cs.kpi_stub_masquerade(files, claims_text="")
    assert k["defects"] == []  # SOFT — never gates


def test_return_nil_accessor_is_not_a_stub():
    # a bare `return nil` interface method is a legitimate empty-result impl, NOT a
    # stub — the narrow detector must not flag it (the ~40-false-positive guard).
    files = {"a.go": ("package a\n"
                      "func (m *MMU) Caps() []abi.Capability { return nil }\n")}
    k = cs.kpi_stub_masquerade(files, claims_text="")
    assert k["soft"] == []


def test_panic_unimplemented_is_soft_signal():
    files = {"a.go": ("package a\n"
                      "func Future() int { panic(\"not implemented\") }\n")}
    k = cs.kpi_stub_masquerade(files, claims_text="")
    assert k["defects"] == []
    assert any("Future" in s for s in k["soft"])


def test_stub_declared_in_claims_is_not_flagged():
    # the TIGHT link (#781): a [STUB] line suppresses a func when the func's name
    # appears as a BACKTICK code symbol on that line.
    files = {"a.go": "package a\nfunc Future() int { panic(\"unimplemented\") }\n"}
    claims = "- [STUB] `Future` is the planned X path, not yet wired.\n"
    k = cs.kpi_stub_masquerade(files, claims_text=claims)
    assert k["soft"] == []


def test_stub_prose_mention_does_not_suppress():
    # the TIGHT link (#781): a [STUB] line that names the symbol only in PROSE (not in a
    # backtick code span) must NOT suppress — keying on prose was the v1 fuzz that
    # granted a false free pass. The func is still surfaced as a SOFT signal.
    files = {"a.go": "package a\nfunc Future() int { panic(\"unimplemented\") }\n"}
    claims = "- [STUB] the Future path is deferred; not yet wired.\n"
    k = cs.kpi_stub_masquerade(files, claims_text=claims)
    assert any("Future" in s for s in k["soft"])
    assert k["defects"] == []  # SOFT — still never gates


def test_stub_qualified_symbol_suppresses_by_last_component():
    # `recall.Page` in the ledger suppresses an exported func named Page (the last
    # dotted component) — the package-qualified symbol<->ledger match.
    files = {"a.go": "package recall\nfunc Page() int { panic(\"not implemented\") }\n"}
    claims = "- [STUB] `recall.Page` validity interval, tracked #81.\n"
    k = cs.kpi_stub_masquerade(files, claims_text=claims)
    assert k["soft"] == []


def test_ledger_stub_symbols_keys_on_backticks_only():
    # the tight-link helper: backtick code spans on a [STUB] line, last dotted
    # component, case-sensitive; prose and non-[STUB] lines are ignored.
    claims = ("- [STUB] `recall.Page` and `LookupOp` are deferred behaviors.\n"
              "- not a stub line: `Ignored` here.\n")
    syms = cs._ledger_stub_symbols(claims)
    assert "Page" in syms        # last component of `recall.Page`
    assert "LookupOp" in syms
    assert "Ignored" not in syms  # not on a [STUB] line
    assert "deferred" not in syms  # prose, not backticked
    assert "behaviors" not in syms


def test_ledger_stub_symbols_ignores_tag_legend():
    # The CLAIMS.md glossary line is "- `[STUB]` ...", not a real stub claim.
    # Mentioning the tag in a backtick span must not suppress an exported stub.
    claims = ("- `[STUB]` - plumbing present, behavior deferred.\n"
              "- [SHIPPED] The docs mention `[STUB]` in prose.\n"
              "- [STUB] `Future` is deferred.\n")
    syms = cs._ledger_stub_symbols(claims)
    assert "Future" in syms
    assert "STUB" not in syms


def test_real_implementation_is_clean():
    files = {"a.go": ("package a\n"
                      "func Add(x, y int) int {\n"
                      "\tz := x + y\n"
                      "\tz *= 2\n"
                      "\treturn z\n"
                      "}\n")}
    k = cs.kpi_stub_masquerade(files, claims_text="")
    assert k["soft"] == []


# --- stub_masquerade SOFT->HARD promotion gate (#781) ---------------------

def test_promotion_awaiting_soak_at_ship_release():
    # At the detector's own ship release the soak window has NOT elapsed: link tight,
    # but releases_since_ship == 0 -> not promotable, stays SOFT. The honest state today.
    s = cs.stub_promotion_status(cs.STUB_DETECTOR_SHIP_RELEASE, n_findings=0)
    assert s["link_tight"] is True
    assert s["releases_since_ship"] == 0
    assert s["soak_met"] is False
    assert s["promotable"] is False
    assert "AWAITING SOAK" in s["status"]


def test_promotion_ready_after_soak_window_elapsed_and_clean():
    # SOAK_RELEASES minor bumps later with a clean tree -> the mechanical gate is met
    # (promotable=True), which tells a maintainer to review the window and flip. It does
    # not itself flip the axis.
    smaj, smin = cs._parse_release(cs.STUB_DETECTOR_SHIP_RELEASE)
    later = f"{smaj}.{smin + cs.STUB_SOAK_RELEASES}.0"
    s = cs.stub_promotion_status(later, n_findings=0)
    assert s["releases_since_ship"] == cs.STUB_SOAK_RELEASES
    assert s["soak_met"] is True
    assert s["promotable"] is True
    assert "READY" in s["status"]


def test_promotion_findings_block_even_after_window():
    # A live finding means the tree is NOT clean: the soak is not met no matter how many
    # releases elapsed (zero false positives is a precondition, and any live finding is
    # an unconfirmed signal).
    smaj, smin = cs._parse_release(cs.STUB_DETECTOR_SHIP_RELEASE)
    later = f"{smaj}.{smin + cs.STUB_SOAK_RELEASES + 1}.0"
    s = cs.stub_promotion_status(later, n_findings=1)
    assert s["soak_met"] is False
    assert s["promotable"] is False


def test_promotion_unparseable_or_cross_major_fails_safe():
    # An unknown version, or a major-version jump, must never falsely claim the soak ran.
    assert cs.stub_promotion_status("garbage", n_findings=0)["releases_since_ship"] == 0
    smaj, smin = cs._parse_release(cs.STUB_DETECTOR_SHIP_RELEASE)
    cross_major = f"{smaj + 1}.{smin + cs.STUB_SOAK_RELEASES}.0"
    s = cs.stub_promotion_status(cross_major, n_findings=0)
    assert s["releases_since_ship"] == 0
    assert s["promotable"] is False


def test_kpi_attaches_promotion_block_without_changing_scoring():
    # The promotion block is advisory metadata: it must be present on the KPI result but
    # change no score and emit no HARD defect.
    files = {"a.go": "package a\nfunc Future() int { panic(\"not implemented\") }\n"}
    k = cs.kpi_stub_masquerade(files, claims_text="", version="0.34.0")
    assert k["defects"] == []          # still SOFT — never gates
    assert k["score"] == 96            # one SOFT signal, score unaffected by promotion
    assert "promotion" in k
    assert k["promotion"]["link_tight"] is True
    assert k["promotion"]["current_release"] == "0.34.0"
    # the fold ignores the extra key: a clean stub KPI still folds to zero debt
    p = cs.build_payload(workspace="/x", kpis=[
        _kpi("duplication", []), _kpi("dead_code", []), _kpi("comment_slop", []),
        _kpi("vacuous_tests", []), k, _kpi("churn_bloat", [])])
    assert p["corpus"]["slop_debt"] == 0


# --- churn_bloat (SOFT) ---------------------------------------------------

def test_churn_bloat_accretion_is_soft():
    k = cs.kpi_churn_bloat(added=12, removed=0, n_commits=20)
    assert k["defects"] == []
    assert k["soft"]
    assert k["score"] < 100


def test_churn_bloat_balanced_is_clean():
    k = cs.kpi_churn_bloat(added=5, removed=4, n_commits=20)
    assert k["defects"] == []
    assert k["soft"] == []
    assert k["score"] == 100


# --- build_payload fold ---------------------------------------------------

def _kpi(name: str, defects: list[str], soft: list[str] | None = None) -> dict:
    return {"kpi": name, "score": 100 if not defects else 0,
            "detail": "", "defects": defects, "soft": soft or []}


def test_payload_clean_is_ok_zero_debt():
    kpis = [_kpi(n, []) for n in cs.KPI_WEIGHTS]
    p = cs.build_payload(workspace="/x", kpis=kpis)
    assert p["ok"] is True
    assert p["corpus"]["slop_debt"] == 0
    assert p["finding"] == "code_slop_clean"


def test_payload_sums_slop_debt_and_gates_ok():
    kpis = [_kpi("duplication", ["c1", "c2"]), _kpi("dead_code", ["d1"]),
            _kpi("comment_slop", []), _kpi("vacuous_tests", []),
            _kpi("stub_masquerade", [], soft=["s1"]), _kpi("churn_bloat", [])]
    p = cs.build_payload(workspace="/x", kpis=kpis)
    assert p["corpus"]["slop_debt"] == 3
    assert p["ok"] is False
    assert p["finding"] == "code_slop"
    # corpus key the control-pane find_int reads
    assert isinstance(p["corpus"]["slop_debt"], int)
    assert p["corpus"]["debt_by_kpi"]["duplication"] == 2


def test_payload_soft_never_counts_as_debt():
    kpis = [_kpi(n, []) for n in cs.KPI_WEIGHTS]
    # add a SOFT-only signal
    kpis[0]["soft"] = ["advisory only"]
    p = cs.build_payload(workspace="/x", kpis=kpis)
    assert p["ok"] is True
    assert p["corpus"]["slop_debt"] == 0
    assert p["corpus"]["soft_signals"] == 1


def test_payload_breakdown_worst_first():
    kpis = [_kpi("duplication", ["a", "b", "c"]), _kpi("dead_code", ["d"]),
            _kpi("comment_slop", []), _kpi("vacuous_tests", []),
            _kpi("stub_masquerade", []), _kpi("churn_bloat", [])]
    p = cs.build_payload(workspace="/x", kpis=kpis)
    assert p["corpus"]["breakdown"][0]["kpi"] == "duplication"
    assert p["corpus"]["breakdown"][0]["debt"] == 3


def test_grade_letter_boundaries():
    assert cs.grade_letter(90) == "A"
    assert cs.grade_letter(80) == "B"
    assert cs.grade_letter(59) == "F"


# --- markdown / doc-freshness ---------------------------------------------

def test_markdown_stamp_roundtrips():
    kpis = [_kpi(n, []) for n in cs.KPI_WEIGHTS]
    p = cs.build_payload(workspace="/x", kpis=kpis)
    body = cs.render_markdown(p, stamp="2026-06-25")
    assert cs.markdown_stamp(body) == "2026-06-25"
    assert "code-slop-scorecard:" in body


def main() -> int:
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
