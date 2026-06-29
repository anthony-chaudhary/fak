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


def _same_skeleton(fn: str, a: str, b: str, c: str, r: str) -> str:
    return (f"func {fn}({a}, {b}, {c} int) int {{\n"
            f"\t{r} := {a} + {b}*{c}\n"
            f"\t{r} = {r} - {a}%{b} + {c}\n"
            f"\tif {r} > {a} {{\n"
            f"\t\t{r} = {r}*{b} - {c}\n"
            f"\t}} else {{\n"
            f"\t\t{r} = {r} + {a} - {b}\n"
            f"\t}}\n"
            f"\treturn {r} + {a}*{b} - {c}\n"
            f"}}\n")


def _retuned_scale(threshold: int, mul: int, modulus: int) -> str:
    return ("func scale(xs []int) int {\n"
            "\ttotal := 0\n"
            "\tfor _, v := range xs {\n"
            f"\t\tif v > {threshold} {{\n"
            f"\t\t\ttotal += v * {mul}\n"
            f"\t\t\ttotal -= v % {modulus}\n"
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


def test_duplication_entry_point_scaffold_is_advisory_not_debt():
    # A clone whose EVERY site is a cmd/*/main.go command entry point is parallel-by-
    # design CLI scaffolding (the flag->build->write->Fprintf skeleton each sub-binary
    # repeats). De-duplicating across independent mains worsens readability, so it is
    # demoted to advisory (soft), not counted as HARD slop-debt.
    mains = {"cmd/onebench/main.go": "package main\n" + _dup_block("run"),
             "cmd/twobench/main.go": "package main\n" + _dup_block("run")}
    k = cs.kpi_duplication(mains)
    assert k["defects"] == [], k["defects"]
    assert any("entry-point scaffold" in s for s in k["soft"]), k["soft"]


def test_duplication_entry_point_clone_leaking_to_internal_stays_debt():
    # The conjunct is strict: ONE internal/ (real shipped kernel) site makes the group
    # HARD again. A clone that leaked from a benchmark main into a hot path is still
    # caught -- only a PURE-entry-point group is excused.
    mixed = {"cmd/onebench/main.go": "package main\n" + _dup_block("run"),
             "internal/hot/path.go": "package hot\n" + _dup_block("run")}
    k = cs.kpi_duplication(mixed)
    assert len(k["defects"]) >= 1, k
    assert k["score"] < 100


def test_duplication_short_fragment_is_not_a_clone():
    # FP: a sub-6-line fragment repeated across files (an idiomatic err-check / sort
    # closure / one-line setup) is NOT extractable copy-paste slop — a shared helper for
    # it would cost as much as the inline code. Must not count (CLONE_MIN_GROUP_SPAN=6).
    frag = ("func {n}(cp []int, budget int) error {{\n"
            "\tsort.Slice(cp, func(i, j int) bool {{ return cp[i] < cp[j] }})\n"
            "\tif err := setBudget(budget); err != nil {{\n"
            "\t\treturn err\n\t}}\n"
            "\treturn nil\n}}\n")  # ~6 src lines but the SHARED window spans <6 after the differing header
    a = "package a\n" + frag.format(n="runA") + "\nfunc tailA() {}\n"
    b = "package b\n" + frag.format(n="runB") + "\nfunc tailB() {}\n"
    # The only shared multi-token window is the 3-line sort+err-check middle; its merged
    # span is < 6 lines, so it must be filtered as idiomatic, not flagged.
    k = cs.kpi_duplication({"a.go": a, "b.go": b})
    short = [d for d in k["defects"] if "sort.Slice" in d]
    assert short == [], short


def test_duplication_six_line_body_clone_still_caught():
    # RECALL: a genuine >=6-line copy-pasted body MUST still be flagged — the min-span
    # gate filters fragments, never real blocks. (_dup_block is an 8-line identical body.)
    files = {"a.go": "package a\n" + _dup_block("sum"),
             "b.go": "package b\n" + _dup_block("sum")}
    assert len(cs.kpi_duplication(files)["defects"]) >= 1


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


def test_duplication_keeps_identifiers_for_clone_precision_780():
    # A shared operator/control skeleton with disjoint identifiers is not a copy-paste
    # clone. The control proves the fixture is long enough to match when identifiers
    # are actually identical.
    distinct = {"a.go": "package a\n" + _same_skeleton("f", "p", "q", "s", "r"),
                "b.go": "package b\n" + _same_skeleton("g", "w", "x", "z", "y")}
    assert cs.kpi_duplication(distinct)["defects"] == []

    identical = {"a.go": "package a\n" + _same_skeleton("f", "p", "q", "s", "r"),
                 "b.go": "package b\n" + _same_skeleton("f", "p", "q", "s", "r")}
    k = cs.kpi_duplication(identical)
    assert len(k["defects"]) >= 1
    assert k["score"] < 100


def test_duplication_normalizes_literals_for_clone_recall_780():
    # Retuned constants are still a copy-pasted structure because literals collapse to
    # the `L` token. A single copy remains clean.
    retuned = {"a.go": "package a\n" + _retuned_scale(0, 2, 3),
               "b.go": "package b\n" + _retuned_scale(7, 5, 9)}
    k = cs.kpi_duplication(retuned)
    assert len(k["defects"]) >= 1
    assert k["score"] < 100
    assert cs.kpi_duplication({"a.go": "package a\n" + _retuned_scale(0, 2, 3)})["defects"] == []


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


def test_payload_lifts_stub_promotion_to_corpus():
    # the #781 promotion readiness must be reachable from corpus (what the control-pane
    # reads), surfaced from the stub_masquerade KPI's `promotion` block.
    stub = cs.kpi_stub_masquerade({}, claims_text="", version="0.34.0")
    kpis = [_kpi("duplication", []), _kpi("dead_code", []), _kpi("comment_slop", []),
            _kpi("vacuous_tests", []), stub, _kpi("churn_bloat", [])]
    p = cs.build_payload(workspace="/x", kpis=kpis)
    promo = p["corpus"]["stub_masquerade_promotion"]
    assert promo["promotable"] is False
    assert promo["current_release"] == "0.34.0"


def test_payload_without_promotion_block_omits_corpus_key():
    # fixtures that don't carry a promotion block must not synthesize the corpus key.
    kpis = [_kpi(n, []) for n in cs.KPI_WEIGHTS]
    p = cs.build_payload(workspace="/x", kpis=kpis)
    assert "stub_masquerade_promotion" not in p["corpus"]


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


def test_markdown_surfaces_stub_promotion_781():
    # The committed doc is the canonical operator-facing artifact, so it must surface the
    # #781 SOFT->HARD promotion readiness (not just --json/text). At 0.34.0 the honest
    # state is AWAITING SOAK / not promotable — the doc must say so, never fake READY.
    stub = cs.kpi_stub_masquerade({}, claims_text="", version="0.34.0")
    kpis = [_kpi("duplication", []), _kpi("dead_code", []), _kpi("comment_slop", []),
            _kpi("vacuous_tests", []), stub, _kpi("churn_bloat", [])]
    p = cs.build_payload(workspace="/x", kpis=kpis)
    body = cs.render_markdown(p, stamp="2026-06-25")
    assert "## stub_masquerade SOFT->HARD promotion (#781)" in body
    assert "AWAITING SOAK" in body
    assert "| promotable now | no |" in body
    # the flip procedure is documented so the eventual promotion is not forgotten
    assert "soft` to `defects`" in body


def test_markdown_omits_promotion_when_block_absent():
    # a payload without the promotion block renders no promotion section (no empty header).
    kpis = [_kpi(n, []) for n in cs.KPI_WEIGHTS]
    p = cs.build_payload(workspace="/x", kpis=kpis)
    body = cs.render_markdown(p, stamp="2026-06-25")
    assert "promotion (#781)" not in body


# --- #789 detector PRECISION fixes: each paired (FP must drop, recall must hold) ------

def test_fix1_signature_only_func_header_is_not_a_clone():
    # FP: two DIFFERENTLY-NAMED funcs whose single signature LINE collides on
    # param-list/type punctuation is not a body clone (every site e==s on a `func ` line).
    sig = ("func {n}(stdout, stderr io.Writer, positional []string, reg *Registry) error {{\n"
           "\t{body}\n}}\n")
    files = {"a.go": "package a\n" + sig.format(n="runHybrid", body="return hybrid()"),
             "b.go": "package b\n" + sig.format(n="runStandard", body="return standard()")}
    # bodies differ; only the long signature line is identical -> must NOT be a clone.
    assert cs.kpi_duplication(files)["defects"] == []


def test_fix1_real_body_clone_with_func_header_still_caught():
    # RECALL: same header AND same multi-line body is a real clone and MUST stay caught
    # (the e==s suppression only fires when EVERY site is a one-line signature collision).
    body = ("func {n}(xs []int) int {{\n"
            "\ttotal := 0\n\tfor _, v := range xs {{\n\t\tif v > 0 {{\n"
            "\t\t\ttotal += v * 2\n\t\t\ttotal -= v % 3\n\t\t}}\n\t}}\n\treturn total\n}}\n")
    files = {"a.go": "package a\n" + body.format(n="sumA"),
             "b.go": "package b\n" + body.format(n="sumB")}
    assert len(cs.kpi_duplication(files)["defects"]) >= 1


def _reexec_helper(name: str) -> str:
    # a re-exec'd *HelperProcess child: os.Exit + out-of-band write + env-guard return.
    return (f"func {name}(t *testing.T) {{\n"
            "\tif os.Getenv(\"FAK_PROBE\") != \"1\" {\n\t\treturn\n\t}\n"
            "\tos.Stdout.WriteString(\"ok\\n\")\n\tos.Exit(0)\n}\n")


def test_fix2_reexec_helper_test_not_vacuous():
    # FP: a re-exec helper asserts through the process boundary (os.Exit/stdout), not t.* .
    tf = {"x_test.go": "package x\n" + _reexec_helper("TestProbeHelper")}
    assert cs.kpi_vacuous_tests(tf)["defects"] == []


def test_fix2_does_not_panic_test_not_vacuous():
    # FP: running a production call without panicking IS the assertion (name marker +
    # single bare call body, composite-literal arg allowed).
    tf = {"x_test.go": ("package x\n"
                        "func TestApplyDoesNotPanicOnEmptySpec(t *testing.T) {\n"
                        "\tapplyHookFloor(RulesetSpec{})\n}\n")}
    assert cs.kpi_vacuous_tests(tf)["defects"] == []


def test_fix2_genuinely_vacuous_test_still_caught():
    # RECALL: a test that neither asserts, nor exits, nor is marked does-not-panic is
    # truly vacuous and MUST stay caught (no os.Exit, ordinary name, multi-statement body).
    tf = {"x_test.go": ("package x\n"
                        "func TestThing(t *testing.T) {\n"
                        "\tx := compute()\n\t_ = x\n}\n")}
    assert len(cs.kpi_vacuous_tests(tf)["defects"]) >= 1


def test_fix2_reexec_tell_required_no_free_pass():
    # RECALL guard on the exemption: os.Exit alone (no env-guard return) does NOT exempt —
    # a body that exits but never asserts and isn't a guarded helper stays graded.
    tf = {"x_test.go": ("package x\n"
                        "func TestNoGuard(t *testing.T) {\n"
                        "\tdoWork()\n\tos.Exit(0)\n}\n")}
    # missing the env-guard-return tell -> not exempted -> still vacuous.
    assert len(cs.kpi_vacuous_tests(tf)["defects"]) >= 1


def _flag_block(name: str) -> str:
    return (f"func {name}(args []string) error {{\n"
            "\tfs := flag.NewFlagSet(\"x\", flag.ContinueOnError)\n"
            "\tengine := fs.String(\"engine\", \"auto\", \"engine id\")\n"
            "\taddr := fs.String(\"addr\", \":0\", \"bind addr\")\n"
            "\tif err := fs.Parse(args); err != nil {\n\t\treturn err\n\t}\n"
            "\t_ = engine\n\t_ = addr\n\treturn nil\n}\n")


def test_fix3_pure_flag_plumbing_block_dropped():
    # FP: two subcommands sharing only the uniform flag-decl/parse plumbing.
    files = {"a.go": "package a\n" + _flag_block("subA"),
             "b.go": "package b\n" + _flag_block("subB")}
    assert cs.kpi_duplication(files)["defects"] == []


def test_fix3_flag_block_with_real_body_still_caught():
    # RECALL: a flag preamble FOLLOWED BY a real copy-pasted computation body is a clone —
    # the group is not ALL-flag-plumbing, so the gate must not drop it. (Uses a NAME
    # sentinel replace, not .format(), to avoid colliding with the Go literal braces.)
    blk = ("func NAME(args []string) error {\n"
           "\tfs := flag.NewFlagSet(\"x\", flag.ContinueOnError)\n"
           "\tn := fs.Int(\"n\", 0, \"count\")\n"
           "\tif err := fs.Parse(args); err != nil {\n\t\treturn err\n\t}\n"
           "\ttotal := 0\n\tfor i := 0; i < *n; i++ {\n\t\ttotal += i * i\n\t\ttotal -= i % 3\n\t}\n"
           "\tfmt.Println(total)\n\treturn nil\n}\n")
    files = {"a.go": "package a\n" + blk.replace("NAME", "runA"),
             "b.go": "package b\n" + blk.replace("NAME", "runB")}
    assert len(cs.kpi_duplication(files)["defects"]) >= 1


def test_fix4_dispatch_arm_is_advisory_not_debt():
    # FP: same-file sibling `case X:` arms sharing an identical loop-free preamble ->
    # advisory soft, not slop-debt. Each arm carries enough identical logic (a guard with
    # comparisons + a delegating call) to form a clone window, but no loop.
    # the shared arm must form a clone window (>=34 tokens, >=2 logic) yet merge to a span
    # <= DISPATCH_ARM_MAX_SPAN(8) lines to qualify as advisory. This mirrors the real
    # gguf-dequant validate-arm shape (a token-dense fmt.Errorf gives the window length;
    # ~5 lines keeps the span in-band; no loop).
    # mirrors the real gguf-dequant validate arm verbatim in shape (two guards + a
    # delegating call): the shared window spans the elems-guard + the len(raw)-guard,
    # which clears 34 tokens; the per-arm `dequantX` call and `Tensor` literal differ.
    def _arm(q: str) -> str:
        tmpl = ("\tcase TensorQQ:\n"
                "\t\tif elems%qk != 0 {\n"
                "\t\t\treturn nil, fmt.Errorf(\"gguf: tensor %s QQ count %d not a multiple of %d\", t.Name, elems, qk)\n"
                "\t\t}\n"
                "\t\twant := int(elems / qk * blockBytes)\n"
                "\t\tif len(raw) != want {\n"
                "\t\t\treturn nil, fmt.Errorf(\"gguf: tensor %s QQ payload has %d bytes, want %d\", t.Name, len(raw), want)\n"
                "\t\t}\n"
                "\t\tdequantQQ(out, raw)\n")
        return tmpl.replace("QQ", q)
    sw = ("package a\nfunc decode(k Kind, out []float32, raw []byte, t T, elems, qk, blockBytes int) error {\n"
          "\tswitch k {\n" + _arm("Q4_0") + _arm("Q4_1") + _arm("Q5_0")
          + "\t}\n\treturn errKind\n}\n")
    k = cs.kpi_duplication({"a.go": sw})
    assert k["defects"] == [], k["defects"]
    assert any("dispatch-arm" in s for s in k["soft"]), k["soft"]


def test_fix4_dispatch_arm_with_loop_stays_debt():
    # RECALL: a duplicated computation LOOP inside arms is real slop regardless of `case`.
    sw = ("package a\nfunc decode(k Kind, raw []byte) []float32 {\n"
          "\tswitch k {\n"
          "\tcase TensorF32:\n\t\tout := make([]float32, len(raw))\n\t\tfor i := range raw {\n\t\t\tout[i] = float32(raw[i]) * 2\n\t\t}\n\t\treturn out\n"
          "\tcase TensorF16:\n\t\tout := make([]float32, len(raw))\n\t\tfor i := range raw {\n\t\t\tout[i] = float32(raw[i]) * 2\n\t\t}\n\t\treturn out\n"
          "\t}\n\treturn nil\n}\n")
    assert len(cs.kpi_duplication({"a.go": sw})["defects"]) >= 1


def test_fix5_sort_scaffold_only_window_skipped():
    # FP: a shared sort.Slice scaffold with NO comparison operator in the window.
    blk = ("package {p}\nfunc sort{n}(rows []Row) {{\n"
           "\tsort.Slice(rows, func(i, j int) bool {{\n"
           "\t\ta, b := rows[i], rows[j]\n\t\t_ = a\n\t\t_ = b\n\t\treturn false\n\t}})\n}}\n")
    files = {"a.go": blk.format(p="a", n="A"), "b.go": blk.format(p="b", n="B")}
    assert cs.kpi_duplication(files)["defects"] == []


def test_fix5_real_shared_comparator_still_caught():
    # RECALL: a sort closure carrying a real comparison (a shared comparator re-implemented
    # in two places) MUST stay caught.
    blk = ("package {p}\nfunc sort{n}(eps []Ep) {{\n"
           "\tsort.SliceStable(eps, func(i, j int) bool {{\n"
           "\t\tif eps[i].CalibErr != eps[j].CalibErr {{\n\t\t\treturn eps[i].CalibErr < eps[j].CalibErr\n\t\t}}\n"
           "\t\treturn eps[i].Name < eps[j].Name\n\t}})\n}}\n")
    files = {"a.go": blk.format(p="a", n="A"), "b.go": blk.format(p="b", n="B")}
    assert len(cs.kpi_duplication(files)["defects"]) >= 1


def test_fix6_declaration_block_demoted_by_context_gate():
    # FP: a pure declaration/composite-literal block clears no logic without a
    # non-assignment logic token, so its `=`/`:=` are demoted and it is not a clone.
    blk = ("package {p}\nvar cfg{n} = Config{{\n"
           "\tName: \"x\",\n\tID: 1,\n\tHash: \"h\",\n\tSeed: 2,\n\tCount: 3,\n\tCost: 4.0,\n\tMode: \"fast\",\n}}\n")
    files = {"a.go": blk.format(p="a", n="A"), "b.go": blk.format(p="b", n="B")}
    assert cs.kpi_duplication(files)["defects"] == []


def test_fix6_assignment_heavy_real_body_still_caught():
    # RECALL: a real copied statement body keeps its `:=` because it co-occurs with
    # control/compute (!=, ||) — the 74:1 trap the context-gate avoids vs blunt deletion.
    body = ("func {n}() error {{\n"
            "\tvar ol Overlapped\n\tr, _, err := proc.Call(h, 0, 0, 0, ref(&ol))\n"
            "\tif r != 0 {{\n\t\treturn nil\n\t}}\n"
            "\tif errors.Is(err, errA) || errors.Is(err, errB) {{\n\t\treturn errBusy\n\t}}\n"
            "\treturn err\n}}\n")
    files = {"a.go": "package a\n" + body.format(n="tryLock"),
             "b.go": "package b\n" + body.format(n="tryFlock")}
    assert len(cs.kpi_duplication(files)["defects"]) >= 1


def test_fix7_slop_keep_directive_exempts_dead_symbol():
    # FP: an intentionally-unreferenced symbol marked `//slop:keep` is not dead-code debt.
    files = {"a.go": ("package a\n"
                      "//slop:keep symbol-table provenance marker\n"
                      "const buildTag = \"x+native\"\n")}
    assert cs.kpi_dead_code(files, {})["defects"] == []


def test_fix7_unmarked_dead_symbol_still_caught():
    # RECALL: an unexported symbol with NO directive and no reference is still dead-code.
    files = {"a.go": "package a\nconst leftover = \"unused\"\n"}
    assert len(cs.kpi_dead_code(files, {})["defects"]) >= 1


def test_asm_referenced_symbol_is_not_dead_1392():
    # FP (#1392): an unexported Go symbol whose ONLY reference lives in a hand-written
    # sibling `.s` file — the func body IS the asm (`TEXT <dot>sym(SB)`) or asm reads a Go
    # gate flag (`CMPB <dot>sym(SB)`) — is LIVE, not dead. The `.go` scan alone never sees
    # it, so without the asm union it reads as "defined, never referenced" and an agent
    # retiring worst-first would delete a SIMD kernel. `<dot>` is the asm package-path
    # separator U+00B7.
    dot = "·"
    files = {"internal/model/quant_amd64.go": (
        "package model\n"
        "func qdot8gemv512(a, b []int8) int32\n"   # asm-bodied: declared, no Go body
        "var q5kUseVNNI bool\n")}                   # asm-read gate flag
    asm = {"internal/model/quant_amd64.s": (
        f"TEXT {dot}qdot8gemv512(SB), NOSPLIT, $0-56\n"
        "\tVMOVDQU8 (AX), Z0\n"
        f"\tCMPB {dot}q5kUseVNNI(SB), $1\n"
        "\tRET\n")}
    k = cs.kpi_dead_code(files, {}, asm)
    names = " ".join(k["defects"])
    assert "qdot8gemv512" not in names, names
    assert "q5kUseVNNI" not in names, names


def test_dead_symbol_with_no_go_or_asm_ref_still_flagged_1392():
    # RECALL (#1392): the asm union must not blanket-exempt. A genuinely unreferenced
    # symbol — no `.go` caller and its name absent from the `.s` corpus — is still dead,
    # so the detector stays honest even with an asm file present.
    dot = "·"
    files = {"internal/model/quant_amd64.go":
             "package model\nfunc trulyDead() int { return 1 }\n"}
    asm = {"internal/model/quant_amd64.s": f"TEXT {dot}other(SB), NOSPLIT, $0\n\tRET\n"}
    k = cs.kpi_dead_code(files, {}, asm)
    assert any("trulyDead" in d for d in k["defects"]), k["defects"]


def test_claude_worktree_go_files_excluded_from_gather():
    # A peer's `.claude/worktrees/<wt>/` is a full repo CHECKOUT; walking it counts every
    # copied .go as a phantom clone of the real tree (a peer worktree once inflated
    # slop-debt 473 -> 2613, which would make the snapshot + #779 gate flap on a transient
    # checkout). The gather drops the whole `.claude` subtree, like `.git`/`vendor`.
    assert cs._excluded_go(".claude/worktrees/upstream-err/internal/model/arch.go")
    assert cs._excluded_go(".claude/worktrees/wt/cmd/fak/main.go")
    # RECALL GUARD: a first-party kernel file is still gathered.
    assert not cs._excluded_go("internal/model/arch.go")
    assert not cs._excluded_go("cmd/fak/main.go")


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
