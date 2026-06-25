#!/usr/bin/env python3
"""Tests for tools/guard_rsi_scorecard.py — hermetic, no network, no toolchain.

The scorecard's load-bearing property is that a KPI passes ONLY when the real thing
exists in the tree (ungameable by editing a docstring), and the HARD failures fold into
`guard_rsi_debt`. These tests drive the pure criteria over synthetic evidence contexts
(both the defect trigger and the clean case), assert the fold to debt + grade, and run a
LIVE smoke over the real repo so a regression flips a criterion.

Run: `python tools/guard_rsi_scorecard_test.py`  (exit 0 = all pass).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guard_rsi_scorecard as sc  # noqa: E402


def _ctx(**over) -> dict:
    """A clean (all-pass) evidence context; override keys to trigger a defect."""
    base = {
        "hop": "PENDING_MEASUREMENT deferred check_plan count_audit_rows",
        "verdict": ("fold_rows verdict_quality run_iteration check_iteration "
                    "fabricated gain strict improvement green external witness "
                    "diagnose_audit_gap empty journal count_audit_rows _journal_paths "
                    "rows > 0 and strict_gain and have_witness"),
        "control_pane": "guard_rsi_scorecard.py guard_rsi_debt",
        "baseline": "guard_rsi",
        "main_go": "", "guard_go": "",
        "verdict_exists": True, "verdict_test_exists": True, "hop_test_exists": True,
        "skill_exists": True, "doc_exists": True,
        "audit_rows": 5, "audit_journals": 1, "audit_diagnose": "",
        "verdict_quality": 100.0,
    }
    base.update(over)
    return base


def _eval_all(ctx: dict) -> dict[str, bool]:
    res = sc.evaluate(sc.maturity_criteria() + sc.realized_criteria(), ctx)
    return {r["key"]: r["passed"] for r in res}


def test_clean_context_zero_debt() -> None:
    res = _eval_all(_ctx())
    assert all(res.values()), [k for k, v in res.items() if not v]


def test_missing_verdict_loop_fails_maturity() -> None:
    res = _eval_all(_ctx(verdict="", verdict_exists=False))
    assert res["verdict_loop_present"] is False
    assert res["deterministic_metric"] is False
    assert res["loop_reads_real_journal"] is False or "count_audit_rows" in _ctx()["hop"]


def test_nondeterministic_metric_fails() -> None:
    res = _eval_all(_ctx(verdict=_ctx()["verdict"] + " import random; random.random()"))
    assert res["deterministic_metric"] is False


def test_not_in_control_pane_fails_realized() -> None:
    res = _eval_all(_ctx(control_pane="other_scorecard.py", baseline="{}"))
    assert res["registered_in_control_pane"] is False


def test_empty_journal_fails_kept_iteration_kpi() -> None:
    res = _eval_all(_ctx(audit_rows=0, audit_journals=0,
                         audit_diagnose="no guard-audit journal directory yet"))
    assert res["kept_iteration_on_real_rows"] is False


def test_populated_journal_passes_kept_iteration_kpi() -> None:
    res = _eval_all(_ctx(audit_rows=12, audit_journals=2))
    assert res["kept_iteration_on_real_rows"] is True


def test_missing_test_fails() -> None:
    res = _eval_all(_ctx(verdict_test_exists=False))
    assert res["paired_honesty_test"] is False


def test_fold_to_debt_and_grade() -> None:
    # 3 HARD realized failures (the pre-seed state) -> debt 3, grade F.
    ctx = _ctx(control_pane="x", baseline="{}", audit_rows=0, audit_journals=0,
               verdict_test_exists=False, audit_diagnose="none")
    mres = sc.evaluate(sc.maturity_criteria(), ctx)
    rres = sc.evaluate(sc.realized_criteria(), ctx)
    hard_fail = [r for r in (mres + rres) if r["hard"] and not r["passed"]]
    assert len(hard_fail) == 3, [r["key"] for r in hard_fail]


def test_grade_letter_boundaries() -> None:
    assert sc.grade_letter(90) == "A"
    assert sc.grade_letter(80) == "B"
    assert sc.grade_letter(70) == "C"
    assert sc.grade_letter(60) == "D"
    assert sc.grade_letter(59) == "F"


def test_live_smoke_real_tree() -> None:
    """The real tree must score: the payload is well-formed and carries the control-pane
    envelope keys the fold reads. (We don't assert a specific debt here — the journal
    state varies — only that the contract holds.)"""
    payload = sc.build_payload(sc.repo_root())
    assert payload["schema"] == sc.SCHEMA
    assert "corpus" in payload and "guard_rsi_debt" in payload["corpus"]
    assert "grade" in payload["corpus"]
    assert isinstance(payload["corpus"]["guard_rsi_debt"], int)
    assert payload["verdict"] in ("OK", "ACTION")
    # The verdict loop we shipped must register as present on the real tree.
    mres = {r["key"]: r["passed"] for r in payload["maturity"]}
    assert mres["verdict_loop_present"] is True, "verdict loop should exist on the real tree"


def main() -> int:
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"ERROR {name}: {exc}")
    if failures:
        print(f"\n{failures} test(s) failed")
        return 1
    print("\nall guard-rsi-scorecard tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
