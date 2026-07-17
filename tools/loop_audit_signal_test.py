#!/usr/bin/env python3
"""Unit tests for loop-audit-signal's pure core — the DARK candidate filter (incl. the
#4989 os-fired exclusion and the registered gate), the SPINNING gate, the open-issue
dedup, the worst-first cap, and the issue render. In-memory fixtures only (no gh, no
`fak loop health` fold), so the testable seam runs on the hermetic CI box.

Dual-runnable (the repo runs the suite pytest-free in CI):
    python tools/loop_audit_signal_test.py
    python -m pytest tools/loop_audit_signal_test.py -q
"""
from __future__ import annotations

from typing import Any

import loop_audit_signal as la


# --- fixtures ---------------------------------------------------------------
def _row(loop_id: str, *, state: str = "dark", dark: bool = True,
         registered: bool = True, os_fired: bool = False,
         last_tick: int = 0, age: int = 0, cadence: int = 86400,
         cadence_source: str = "registry", runs: int = 0, witnessed: int = 0,
         keep_rate: float | None = None) -> dict[str, Any]:
    """One HealthRow, shaped like internal/loopmgr.HealthRow's JSON."""
    if keep_rate is None:
        keep_rate = -1.0 if runs == 0 else witnessed / runs
    return {
        "loop_id": loop_id, "state": state, "dark": dark,
        "os_fired_no_ledger_row": os_fired, "registered": registered,
        "ledgered": last_tick != 0, "cadence_seconds": cadence,
        "cadence_source": cadence_source, "last_tick_unix_nano": last_tick,
        "age_seconds": age, "runs": runs, "witnessed": witnessed,
        "keep_rate": keep_rate,
    }


def _report(rows: list[dict[str, Any]], **rollup: int) -> dict[str, Any]:
    base = {"loops": len(rows), "dark": sum(1 for r in rows if r.get("dark"))}
    base.update(rollup)
    return {"schema": "fak-loop-health/1", "ts_unix_nano": 1_700_000_000_000_000_000,
            "rows": rows, "rollup": base}


# --- DARK candidate filter --------------------------------------------------
def test_dark_candidate_basic() -> None:
    rep = _report([_row("nightrun-collect")])
    cands = la.dark_candidates(rep)
    assert [r["loop_id"] for r in cands] == ["nightrun-collect"]


def test_dark_excludes_unregistered() -> None:
    # An ad-hoc ledger entry (registered=false) is not a scheduled loop -> not a DARK
    # loop to file, even when the state reads dark.
    rep = _report([_row("adhoc-thing", registered=False)])
    assert la.dark_candidates(rep) == []


def test_dark_excludes_os_fired() -> None:
    # #4989: a ledger-DARK loop whose mapped OS task fired 0x0 within cadence is not a
    # clean DARK to re-file — the OS witness already explains the missing ledger row.
    rep = _report([_row("cadence-mon", os_fired=True)])
    assert la.dark_candidates(rep) == []


def test_dark_excludes_live() -> None:
    rep = _report([_row("healthy", state="live", dark=False, last_tick=1, runs=3,
                        witnessed=3)])
    assert la.dark_candidates(rep) == []


def test_dark_worst_first_never_ticked_before_aged() -> None:
    aged = _row("aged", last_tick=1_600_000_000_000_000_000, age=90_000)
    never = _row("never", last_tick=0, age=0)
    rep = _report([aged, never])
    assert [r["loop_id"] for r in la.dark_candidates(rep)] == ["never", "aged"]


def test_dark_worst_first_by_age() -> None:
    young = _row("young", last_tick=2, age=100)
    old = _row("old", last_tick=2, age=999_999)
    rep = _report([young, old])
    assert [r["loop_id"] for r in la.dark_candidates(rep)] == ["old", "young"]


# --- SPINNING gate ----------------------------------------------------------
def test_spin_candidate_when_included() -> None:
    row = _row("spinner", state="live", dark=False, last_tick=5, runs=8, witnessed=0)
    cands = la.spin_candidates(_report([row]), min_runs=5, max_keep=0.0)
    assert [r["loop_id"] for r in cands] == ["spinner"]


def test_spin_excludes_low_runs() -> None:
    row = _row("young-spin", state="live", dark=False, last_tick=5, runs=2, witnessed=0)
    assert la.spin_candidates(_report([row]), min_runs=5, max_keep=0.0) == []


def test_spin_excludes_keep_above_floor() -> None:
    row = _row("keeps", state="live", dark=False, last_tick=5, runs=10, witnessed=4)
    assert la.spin_candidates(_report([row]), min_runs=5, max_keep=0.0) == []


def test_spin_excludes_no_denominator() -> None:
    # keep_rate == -1 (runs == 0) is "no rate", never a 0% keep — excluded by the runs
    # gate, asserted here so the -1 sentinel can never be read as a floor breach.
    row = _row("fresh", state="live", dark=False, last_tick=5, runs=0)
    assert row["keep_rate"] == -1.0
    assert la.spin_candidates(_report([row]), min_runs=5, max_keep=0.0) == []


def test_spin_excludes_dark() -> None:
    row = _row("dark-loop", state="dark", dark=True, runs=9, witnessed=0)
    assert la.spin_candidates(_report([row]), min_runs=5, max_keep=0.0) == []


def test_spin_worst_first_by_runs() -> None:
    a = _row("few", state="live", dark=False, last_tick=5, runs=6, witnessed=0)
    b = _row("many", state="live", dark=False, last_tick=5, runs=40, witnessed=0)
    cands = la.spin_candidates(_report([a, b]), min_runs=5, max_keep=0.0)
    assert [r["loop_id"] for r in cands] == ["many", "few"]


# --- plan(): dedup + cap ----------------------------------------------------
def test_plan_off_by_default_files_dark_only() -> None:
    dark = _row("dark-loop")
    spin = _row("spin-loop", state="live", dark=False, last_tick=5, runs=9, witnessed=0)
    to_file, stats = la.plan(_report([dark, spin]), open_ids=set(), max_issues=5,
                             today="2026-07-20")  # include_spin defaults False
    assert [i["loop_id"] for i in to_file] == ["dark-loop"]
    assert stats["dark"] == 1 and stats["spinning"] == 0


def test_plan_include_spin_files_both() -> None:
    dark = _row("dark-loop")
    spin = _row("spin-loop", state="live", dark=False, last_tick=5, runs=9, witnessed=0)
    to_file, stats = la.plan(_report([dark, spin]), open_ids=set(), max_issues=5,
                             today="2026-07-20", include_spin=True)
    # dark is ranked first, spinning second.
    assert [i["reason"] for i in to_file] == ["dark", "spinning"]


def test_plan_dedup_open() -> None:
    rep = _report([_row("already-open"), _row("fresh-dark")])
    to_file, stats = la.plan(rep, open_ids={"already-open"}, max_issues=5,
                             today="2026-07-20")
    assert [i["loop_id"] for i in to_file] == ["fresh-dark"]
    assert stats["already-open"] == 1


def test_plan_within_run_dup_guarded() -> None:
    # Two rows sharing a loop id (pathological fold) must file once, not twice.
    rep = _report([_row("dup"), _row("dup")])
    to_file, stats = la.plan(rep, open_ids=set(), max_issues=5, today="2026-07-20")
    assert [i["loop_id"] for i in to_file] == ["dup"]
    assert stats["within-run-dup"] == 1


def test_plan_cap_worst_first() -> None:
    rows = [_row(f"d{i}", last_tick=2, age=1000 * i) for i in range(1, 6)]  # 5 dark
    to_file, stats = la.plan(_report(rows), open_ids=set(), max_issues=2,
                             today="2026-07-20")
    assert len(to_file) == 2
    assert stats["over-cap"] == 3
    # Worst-first: the two oldest (largest age) survive the cap.
    assert [i["loop_id"] for i in to_file] == ["d5", "d4"]


def test_plan_cap_drops_spin_before_dark() -> None:
    dark = [_row(f"d{i}", last_tick=2, age=1000) for i in range(2)]
    spin = [_row(f"s{i}", state="live", dark=False, last_tick=5, runs=9, witnessed=0)
            for i in range(3)]
    to_file, _ = la.plan(_report(dark + spin), open_ids=set(), max_issues=2,
                         today="2026-07-20", include_spin=True)
    # dark ranks above spin, so a cap of 2 keeps both dark and drops all spin.
    assert all(i["reason"] == "dark" for i in to_file)


def test_plan_empty_report() -> None:
    to_file, stats = la.plan(_report([]), open_ids=set(), max_issues=5,
                             today="2026-07-20")
    assert to_file == []
    assert stats["dark"] == 0


# --- render + the marker round-trip (the load-bearing dedup invariant) ------
def test_render_dark_issue_fields() -> None:
    rep = _report([_row("nightrun-collect")])
    issue = la.render_issue(rep["rows"][0], "dark", rep, "2026-07-20")
    assert "DARK" in issue["title"] and "nightrun-collect" in issue["title"]
    assert issue["labels"] == [la.SIGNAL_LABEL, la.HEALTH_LABEL]
    assert issue["loop_id"] == "nightrun-collect"
    assert la.marker("nightrun-collect") in issue["body"]
    assert "### Lane\ncmd" in issue["body"]
    assert "never ticked" in issue["body"]  # last_tick 0 -> never-ticked phrase


def test_render_spinning_issue_fields() -> None:
    row = _row("spinner", state="live", dark=False, last_tick=5, runs=8, witnessed=0)
    issue = la.render_issue(row, "spinning", _report([row]), "2026-07-20")
    assert "SPINNING" in issue["title"]
    assert "0%" in issue["title"]  # 0/8 kept
    assert la.marker("spinner") in issue["body"]


def test_marker_roundtrip_dedups_own_output() -> None:
    # THE invariant: the marker the tool WRITES into a body is exactly the anchor its
    # own open-issue parser READS back. If these ever drift, the feeder re-files a
    # duplicate for the same loop every single run. Prove write==read end-to-end.
    rep = _report([_row("some/loop.id-42"), _row("another-loop")])
    issues = la.plan(rep, open_ids=set(), max_issues=5, today="2026-07-20")[0]
    # Feed the rendered bodies back through the dedup parser as if they were open.
    open_again = la.open_issue_ids([{"body": i["body"]} for i in issues])
    assert open_again == {"some/loop.id-42", "another-loop"}
    # And a second plan against that index files nothing (idempotent).
    again, stats = la.plan(rep, open_ids=open_again, max_issues=5, today="2026-07-20")
    assert again == []
    assert stats["already-open"] == 2


def test_open_issue_ids_parses_markers() -> None:
    issues = [
        {"body": f"blah\n{la.marker('loop-a')}\n"},
        {"body": f"{la.marker('loop-b')}"},
        {"body": "no marker here"},
    ]
    assert la.open_issue_ids(issues) == {"loop-a", "loop-b"}


def _run() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
