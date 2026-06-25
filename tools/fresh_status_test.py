#!/usr/bin/env python3
"""Tests for the cross-domain fresh-status rollup.

Drives the PURE folds — provenance classification (incl. the unknown ->
fail-closed case), catalog freshness/staleness math across both timestamp forms,
the four per-pane folds (git / benchmarks / work / industry), and the rollup
verdict ladder (all_green vs needs_attention, SOFT-SKIP never trips, a HARD pane
ERROR/ACTION does) — over synthetic payloads, then a tolerant live smoke that the
real tree folds to a well-formed envelope. No network, no live sub-tool calls in
the pure tests (hermetic, so the no-blackhole gate runs it).

Run: `python tools/fresh_status_test.py`  (exit 0 = all pass),
or `python -m pytest tools/fresh_status_test.py -q`.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fresh_status as fs  # noqa: E402

NOW = datetime(2026, 6, 25, 12, 0, 0, tzinfo=timezone.utc)


# --- provenance classification (the fail-closed honesty floor) --------------

def test_provenance_measured() -> None:
    assert fs.classify_provenance("decode measured wall-clock 38 t/s") == "measured"
    assert fs.classify_provenance("Wall-Clock parity run") == "measured"


def test_provenance_modeled() -> None:
    assert fs.classify_provenance("webvoyager geometry closed-form 9.7x") == "modeled"
    assert fs.classify_provenance("synthetic kernel projection") == "modeled"


def test_provenance_modeled_wins_over_measured() -> None:
    # "modeled from the measured baseline" is, as a whole, a MODELED number — the
    # conservative call the provenance gate makes.
    assert fs.classify_provenance("modeled from the measured baseline") == "modeled"


def test_provenance_unknown_is_fail_closed() -> None:
    # The load-bearing case: text that names NEITHER word is unknown, never
    # silently measured. A real catalog run tag like "radix-benchmark" lands here.
    assert fs.classify_provenance("radix-benchmark engine-fak-cpu qwen2.5") == "unknown"
    assert fs.classify_provenance("") == "unknown"


# --- catalog freshness / staleness math -------------------------------------

def test_parse_run_ts_compact_and_iso() -> None:
    assert fs._parse_run_ts("20260625T050017Z") == datetime(2026, 6, 25, 5, 0, 17, tzinfo=timezone.utc)
    assert fs._parse_run_ts("2026-06-25T05:00:17Z") == datetime(2026, 6, 25, 5, 0, 17, tzinfo=timezone.utc)
    assert fs._parse_run_ts("garbage") is None
    assert fs._parse_run_ts("") is None


def _catalog(*runs: dict, machines: int = 2) -> dict:
    return {"runs": list(runs), "machines": {f"m{i}": {} for i in range(machines)}}


def test_fold_benchmarks_counts_and_provenance() -> None:
    cat = _catalog(
        {"timestamp": "20260625T050017Z", "model": "qwen", "tags": ["measured", "decode"]},
        {"timestamp": "20260624T050017Z", "model": "webvoyager", "tags": ["geometry"]},
        {"timestamp": "20260623T050017Z", "model": "agent-workload", "tags": ["experiment"]},
    )
    pane = fs.fold_benchmarks(cat, now=NOW)
    assert pane["runs"] == 3 and pane["machines"] == 2
    assert pane["provenance"] == {"measured": 1, "modeled": 1, "unknown": 1}
    assert pane["verdict"] == "OK" and pane["ok"] is True
    # newest run is the 06-25 one, ~0.3d old
    assert pane["newest"] == "20260625T050017Z" and pane["age_days"] < 1


def test_fold_benchmarks_flags_stale() -> None:
    cat = _catalog({"timestamp": "20260501T000000Z", "model": "x", "tags": []})
    pane = fs.fold_benchmarks(cat, now=NOW, stale_days=30)
    assert pane["stale"] is True and pane["verdict"] == "ACTION" and pane["ok"] is False
    assert "STALE" in pane["reason"]


def test_fold_benchmarks_missing_catalog_is_hard_error() -> None:
    pane = fs.fold_benchmarks(None, now=NOW)
    assert pane["verdict"] == "ERROR" and pane["ok"] is False


def test_fold_benchmarks_empty_runs_is_action() -> None:
    pane = fs.fold_benchmarks(_catalog(machines=1), now=NOW)
    assert pane["runs"] == 0 and pane["verdict"] == "ACTION"


# --- work pane: SOFT (zero / absent -> SKIP, never a failure) ---------------

def test_fold_work_zero_plans_is_skip_not_error() -> None:
    pane = fs.fold_work({"counts": {"total_plans": 0}})
    assert pane["verdict"] == "SKIP" and pane["ok"] is True
    assert pane["total_plans"] == 0


def test_fold_work_absent_is_skip() -> None:
    pane = fs.fold_work(None, error="plan_audit unavailable")
    assert pane["verdict"] == "SKIP" and pane["ok"] is True


def test_fold_work_with_plans_reports_ok() -> None:
    pane = fs.fold_work({"counts": {"total_plans": 4, "shipped": 3, "remaining": 1}})
    assert pane["verdict"] == "OK" and pane["total_plans"] == 4
    assert "3 shipped" in pane["reason"] and "1 remaining" in pane["reason"]


# --- industry pane: SOFT ----------------------------------------------------

def test_fold_industry_reports_parity_debt() -> None:
    pane = fs.fold_industry({"corpus": {"parity_debt": 7, "grade": "B"}})
    assert pane["verdict"] == "OK" and pane["parity_debt"] == 7 and pane["grade"] == "B"


def test_fold_industry_absent_is_skip() -> None:
    pane = fs.fold_industry(None, error="timed out")
    assert pane["verdict"] == "SKIP" and pane["ok"] is True


def test_fold_industry_no_debt_key_is_skip() -> None:
    pane = fs.fold_industry({"corpus": {"grade": "A"}})
    assert pane["verdict"] == "SKIP" and pane["grade"] == "A"


# --- git pane ---------------------------------------------------------------

def test_git_pane_smoke_on_real_repo() -> None:
    pane = fs.git_pane(fs.repo_root())
    # In this repo git is available, so the pane must report a sha + branch.
    assert pane["key"] == "git"
    assert pane["verdict"] in ("OK", "ERROR")
    if pane["verdict"] == "OK":
        assert pane["sha"] and isinstance(pane["dirty"], int)


# --- the rollup fold: verdict ladder ----------------------------------------

def _panes(git_v="OK", bench_v="OK", work_v="SKIP", ind_v="OK", *, bench_stale=False) -> list:
    return [
        {"key": "git", "label": "git", "verdict": git_v, "ok": git_v == "OK",
         "reason": "abc (main), clean", "sha": "abc", "branch": "main", "dirty": 0,
         "ahead": 0, "behind": 0},
        {"key": "benchmarks", "label": "benchmarks", "verdict": bench_v,
         "ok": bench_v == "OK", "reason": "55 runs", "stale": bench_stale,
         "provenance": {"measured": 1, "modeled": 1, "unknown": 1}, "runs": 55,
         "machines": 5, "newest": "x", "age_days": 1},
        {"key": "work", "label": "work", "verdict": work_v, "ok": work_v != "ACTION",
         "reason": "0 plans"},
        {"key": "industry", "label": "industry", "verdict": ind_v, "ok": ind_v == "OK",
         "reason": "parity 0"},
    ]


def _fold(panes):
    return fs.fold(panes, workspace=".", commit="c0", generated_at="2026-06-25T12:00:00Z")


def test_fold_all_green_when_no_action() -> None:
    out = _fold(_panes())
    assert out["ok"] is True and out["verdict"] == "OK" and out["finding"] == "all_green"
    assert out["schema"] == fs.SCHEMA and set(out["panes"]) == {"git", "benchmarks", "work", "industry"}


def test_fold_soft_skip_never_trips() -> None:
    # work SKIP + industry SKIP must still read green.
    out = _fold(_panes(work_v="SKIP", ind_v="SKIP"))
    assert out["verdict"] == "OK" and "skipped" in out["reason"]


def test_fold_hard_git_error_trips_to_action() -> None:
    out = _fold(_panes(git_v="ERROR"))
    assert out["ok"] is False and out["verdict"] == "ACTION" and out["finding"] == "needs_attention"
    assert "git" in out["next_action"]


def test_fold_stale_benchmarks_trips_and_points_to_catalog() -> None:
    out = _fold(_panes(bench_v="ACTION", bench_stale=True))
    assert out["verdict"] == "ACTION"
    assert "bench_catalog.py build" in out["next_action"]


def test_render_lists_every_pane() -> None:
    text = fs.render(_fold(_panes()))
    for label in ("git", "benchmarks", "work", "industry"):
        assert label in text
    assert "fresh status" in text


def test_render_doc_has_provenance_and_panes() -> None:
    doc = fs.render_doc(_fold(_panes()), date="2026-06-25")
    assert "Fresh status snapshot (2026-06-25)" in doc
    assert "measured" in doc and "modeled" in doc and "unknown" in doc
    assert "| Pane | Verdict | Detail |" in doc


# --- main wiring ------------------------------------------------------------

def test_main_check_returns_zero_when_green() -> None:
    """`main(['--check'])` returns 0 when the rollup is OK — driven with a stubbed
    collect so it's fast and deterministic, not the live sub-tools."""
    orig = fs.collect
    try:
        fs.collect = lambda root, timeout=60, now=None: _panes()
        assert fs.main(["--check"]) == 0
        # a hard git error -> non-zero advisory exit
        fs.collect = lambda root, timeout=60, now=None: _panes(git_v="ERROR")
        assert fs.main(["--check"]) == 1
    finally:
        fs.collect = orig


def test_main_json_emits_envelope() -> None:
    import contextlib
    import io
    import json as _json
    orig = fs.collect
    try:
        fs.collect = lambda root, timeout=60, now=None: _panes()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = fs.main(["--json"])
        out = _json.loads(buf.getvalue())
        assert code == 0 and out["schema"] == fs.SCHEMA
        assert set(out["panes"]) == {"git", "benchmarks", "work", "industry"}
    finally:
        fs.collect = orig


# --- tolerant live smoke ----------------------------------------------------

def test_live_collect_and_fold() -> None:
    root = fs.repo_root()
    panes = fs.collect(root, timeout=90)
    assert len(panes) == 4
    out = fs.fold(panes, workspace=str(root), commit=fs.head_commit(root),
                  generated_at="now")
    for field in ("schema", "ok", "verdict", "finding", "reason", "next_action", "panes"):
        assert field in out, f"missing {field}"
    # git + benchmarks must be present as HARD panes (no silent drop).
    assert "git" in out["panes"] and "benchmarks" in out["panes"]


def _run_all() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
