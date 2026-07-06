#!/usr/bin/env python3
"""Unit tests for the behavior-contract (contract-vs-change-detector) scorecard.

Pure classifier over in-memory Go-test fixtures — no disk, no `go`, no git — so the
testable seam runs anywhere. Each test locks one calibration decision, especially
the false-positive guards that keep an ordinary example-based unit test OUT of the
change-detector debt (a bare scalar compare is unclassified; a test that also holds
an invariant is exonerated; a `{`/`!=` inside a string never breaks block matching).

Dual-runnable (the repo runs the suite pytest-free in CI):
    python tools/behavior_contract_scorecard_test.py
    python -m pytest tools/behavior_contract_scorecard_test.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import behavior_contract_scorecard as bc


def _fn(body: str, tparam: str = "t") -> dict:
    return bc.classify_func(body, tparam)


def _cls(body: str, tparam: str = "t") -> str:
    return _fn(body)["class"]


# --- change_detector: the three Hermes-named freezes -----------------------

def test_count_freeze_is_change_detector():
    body = '\n\tmodels := List()\n\tif len(models) != 7 {\n\t\tt.Fatalf("got %d", len(models))\n\t}\n'
    res = _fn(body)
    assert res["class"] == "change_detector", res
    assert any(k == "count" for k, _ in res["pins"]), res["pins"]


def test_count_freeze_via_literal_bound_var():
    body = '\n\tn := 3\n\tif len(got) != n {\n\t\tt.Errorf("bad count")\n\t}\n'
    assert _cls(body) == "change_detector"


def test_version_literal_freeze_is_change_detector():
    body = '\n\tv := Config().Version\n\tif v != "v1.4" {\n\t\tt.Fatalf("bad version: %s", v)\n\t}\n'
    res = _fn(body)
    assert res["class"] == "change_detector", res
    assert any(k == "version" for k, _ in res["pins"]), res["pins"]


def test_version_freeze_via_literal_bound_var():
    body = '\n\twant := "2.0"\n\tif Version() != want {\n\t\tt.Fatal("version drift")\n\t}\n'
    assert _cls(body) == "change_detector"


def test_enum_list_deepequal_literal_slice_is_change_detector():
    body = ('\n\tgot := Names()\n'
            '\tif !reflect.DeepEqual(got, []string{"a", "b", "c"}) {\n'
            '\t\tt.Fatalf("names drift: %v", got)\n\t}\n')
    res = _fn(body)
    assert res["class"] == "change_detector", res
    assert any(k == "enum" for k, _ in res["pins"]), res["pins"]


# --- contract: a real invariant exonerates ---------------------------------

def test_two_computed_sides_is_contract():
    body = '\n\tgot := Sum(xs)\n\twant := brute(xs)\n\tif got != want {\n\t\tt.Fatalf("mismatch")\n\t}\n'
    assert _cls(body) == "contract"


def test_len_equals_len_is_contract():
    body = '\n\tif len(keys) != len(vals) {\n\t\tt.Fatalf("keys/vals length invariant broken")\n\t}\n'
    assert _cls(body) == "contract"


def test_field_relation_is_contract():
    body = '\n\tif acc.Total != acc.Used+acc.Free {\n\t\tt.Errorf("total != used+free")\n\t}\n'
    assert _cls(body) == "contract"


def test_deepequal_two_computed_values_is_contract():
    body = '\n\tif !reflect.DeepEqual(got, want) {\n\t\tt.Fatalf("round-trip broke")\n\t}\n'
    assert _cls(body) == "contract"


def test_sort_invariant_is_contract():
    body = '\n\tif !sort.IsSorted(byName(got)) {\n\t\tt.Fatalf("not sorted")\n\t}\n'
    assert _cls(body) == "contract"


def test_table_driven_struct_field_want_is_contract():
    # tt.want is a struct field, NOT a literal-bound var -> a relation, not a freeze.
    body = ('\n\tfor _, tt := range cases {\n'
            '\t\tif got := f(tt.in); got != tt.want {\n'
            '\t\t\tt.Errorf("f(%v)=%v want %v", tt.in, got, tt.want)\n\t\t}\n\t}\n')
    assert _cls(body) == "contract"


def test_count_freeze_plus_relation_is_exonerated():
    # pins a count AND asserts an invariant -> the invariant wins: not debt.
    body = ('\n\tif len(got) != 3 {\n\t\tt.Fatalf("len")\n\t}\n'
            '\tif got[0] != want0 {\n\t\tt.Fatalf("elem")\n\t}\n')
    assert _cls(body) == "contract"


# --- unclassified: neither a named freeze nor a relation (NOT debt) --------

def test_bare_scalar_compare_is_unclassified_not_debt():
    # got != 3 is a scalar example compare, NOT one of the three named freezes.
    # We cannot tell a frozen enumeration from a meaningful value, so it is neutral.
    body = '\n\tif Parse("3") != 3 {\n\t\tt.Fatalf("parse")\n\t}\n'
    assert _cls(body) == "unclassified"


def test_bare_string_compare_is_unclassified():
    body = '\n\tif Greet() != "hello" {\n\t\tt.Fatalf("greet")\n\t}\n'
    assert _cls(body) == "unclassified"


def test_err_nil_guard_is_unclassified():
    body = '\n\tif err != nil {\n\t\tt.Fatalf("unexpected err: %v", err)\n\t}\n'
    assert _cls(body) == "unclassified"


def test_empty_len_zero_is_not_a_freeze():
    # len(x) != 0 is an emptiness bound (an invariant-ish structural check), not a
    # frozen enumeration count -> below COUNT_MIN, so it is not a pin.
    body = '\n\tif len(errs) != 0 {\n\t\tt.Fatalf("expected no errors")\n\t}\n'
    assert _cls(body) == "unclassified"


def test_singleton_len_one_is_not_a_freeze():
    body = '\n\tif len(got) != 1 {\n\t\tt.Fatalf("want one")\n\t}\n'
    assert _cls(body) == "unclassified"


def test_helper_delegated_assertion_is_unclassified():
    # no if-guarded t.* comparison -> we don't guess; not a false positive.
    body = '\n\tcheckEqual(t, got, 7)\n'
    assert _cls(body) == "unclassified"


# --- precision / robustness guards -----------------------------------------

def test_string_with_braces_and_ops_does_not_break_matching():
    # a `{` and `!=` inside a string literal must not fool block/condition scanning;
    # the RHS is a non-version string -> unclassified, not a crash or misparse.
    body = '\n\tif render() != "a{b}!=c" {\n\t\tt.Fatalf("bad")\n\t}\n'
    assert _cls(body) == "unclassified"


def test_no_fail_call_means_no_assertion():
    # an if with no t.Fatal/Error in its block is not an assertion (e.g. a guard).
    body = '\n\tif len(got) != 5 {\n\t\treturn\n\t}\n\t_ = got\n'
    assert _cls(body) == "unclassified"


def test_init_statement_condition_is_classified():
    # `if got := f(); len(got) != 4 { t.Fatal }` -> the count freeze after the ';'.
    body = '\n\tif got := f(); len(got) != 4 {\n\t\tt.Fatalf("len")\n\t}\n'
    assert _cls(body) == "change_detector"


# --- suite-level: iteration, counts, debt, score ---------------------------

def test_iter_test_funcs_skips_benchmarks_and_fuzz():
    src = ('package p\n\n'
           'func TestReal(t *testing.T) {\n\tif len(x) != 9 {\n\t\tt.Fatal("c")\n\t}\n}\n\n'
           'func BenchmarkX(b *testing.B) {\n\tfor i := 0; i < b.N; i++ {\n\t\t_ = len(x)\n\t}\n}\n\n'
           'func FuzzY(f *testing.F) {\n\tf.Fuzz(func(t *testing.T, s string) {})\n}\n')
    names = [n for n, _, _, _ in bc._iter_test_funcs(src)]
    assert names == ["TestReal"], names


def test_kpi_counts_and_debt():
    files = {
        "a_test.go": (
            'package a\n'
            'func TestCount(t *testing.T) {\n\tif len(m) != 7 {\n\t\tt.Fatal("c")\n\t}\n}\n'
            'func TestContract(t *testing.T) {\n\tif got != want {\n\t\tt.Fatal("r")\n\t}\n}\n'
            'func TestNeutral(t *testing.T) {\n\tif err != nil {\n\t\tt.Fatal("e")\n\t}\n}\n'
        ),
    }
    kpi = bc.kpi_contract_discipline(files)
    assert kpi["counts"] == {"tests": 3, "contract": 1, "change_detector": 1, "unclassified": 1}, kpi["counts"]
    assert kpi["debt"] == 1
    assert kpi["score"] == 96  # 100 - 4*1
    assert len(kpi["defects"]) == 1
    assert "TestCount" in kpi["defects"][0]


def test_zero_debt_is_grade_a_and_ok():
    files = {"a_test.go": 'package a\nfunc TestOk(t *testing.T) {\n\tif got != want {\n\t\tt.Fatal("r")\n\t}\n}\n'}
    payload = bc.build_payload("/w", bc.kpi_contract_discipline(files))
    assert payload["ok"] is True
    assert payload["change_detector_debt"] == 0
    assert payload["grade"] == "A"
    assert payload["contract_ratio"] == 1.0


def test_debt_reds_the_bare_run():
    files = {"a_test.go": 'package a\nfunc TestBad(t *testing.T) {\n\tif len(m) != 4 {\n\t\tt.Fatal("c")\n\t}\n}\n'}
    payload = bc.build_payload("/w", bc.kpi_contract_discipline(files))
    assert payload["ok"] is False
    assert payload["change_detector_debt"] == 1
    assert "change-detector" in payload["reason"]


def test_empty_suite_is_an_honest_error_not_a_crash():
    payload = bc.build_payload("/w", None, error="no first-party _test.go files found (run from repo ROOT)")
    assert payload["ok"] is False
    assert "no first-party" in payload["error"]


def test_render_is_stable_text():
    files = {"a_test.go": 'package a\nfunc TestBad(t *testing.T) {\n\tif len(m) != 5 {\n\t\tt.Fatal("c")\n\t}\n}\n'}
    out = bc.render(bc.build_payload("/w", bc.kpi_contract_discipline(files)))
    assert "behavior-contract-scorecard" in out
    assert "change-detector-debt 1" in out


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
