#!/usr/bin/env python3
"""Unit tests for score-signal's pure core — the relevance filter, the open-issue
dedup, the worst-first cap, and the issue render. In-memory fixtures only (no gh,
no network, no control-pane fold), so the testable seam runs on the hermetic CI box.

Dual-runnable (the repo runs the suite pytest-free in CI):
    python tools/score_signal_test.py
    python -m pytest tools/score_signal_test.py -q
"""
from __future__ import annotations

import score_signal as ss


# --- fixtures ---------------------------------------------------------------
def _pane(*, direction: str, early_warning: list[dict], metrics: list[dict] | None = None,
          commit: str = "abc1234", baseline_commit: str = "dead00f") -> dict:
    """A minimal folded control-pane payload, shaped like scorecard_control_pane.fold."""
    return {
        "schema": "fak-scorecard-control-pane/1",
        "commit": commit,
        "metrics": metrics or [],
        "trend": {
            "direction": direction,
            "baseline_commit": baseline_commit,
            "early_warning": early_warning,
        },
    }


def _ew(key: str, label: str, delta: int, frm: int, to: int) -> dict:
    return {"key": key, "label": label, "delta": delta, "from": frm, "to": to}


# --- relevance filter: regressions() ----------------------------------------
def test_regressions_extracts_risen_worst_first():
    pane = _pane(
        direction="regressed",
        early_warning=[_ew("slop", "code-slop", 2, 4, 6),
                       _ew("code", "code", 9, 30, 39)],
        metrics=[{"key": "code", "grade": "C"}, {"key": "slop", "grade": "B"}],
    )
    rs = ss.regressions(pane, min_delta=1)
    assert [r["key"] for r in rs] == ["code", "slop"], "worst regression first"
    assert rs[0]["delta"] == 9 and rs[0]["from"] == 30 and rs[0]["to"] == 39
    assert rs[0]["grade"] == "C", "grade enriched from metrics"
    assert rs[0]["portfolio_regressed"] is True
    assert rs[0]["baseline_commit"] == "dead00f"


def test_regressions_min_delta_filters_small_rises():
    pane = _pane(direction="flat",
                 early_warning=[_ew("seo", "seo", 1, 6, 7),
                                _ew("code", "code", 5, 30, 35)])
    rs = ss.regressions(pane, min_delta=3)
    assert [r["key"] for r in rs] == ["code"], "the +1 rise is below min-delta"


def test_regressions_unpinned_yields_nothing():
    # No baseline pinned -> no delta to act on, even if early_warning somehow present.
    pane = _pane(direction="unpinned", early_warning=[_ew("code", "code", 5, 0, 5)])
    assert ss.regressions(pane, min_delta=1) == []


def test_regressions_empty_early_warning():
    assert ss.regressions(_pane(direction="improved", early_warning=[]), 1) == []


def test_regressions_skips_malformed_entries():
    pane = _pane(direction="flat",
                 early_warning=["nope", {"key": "", "delta": 4},
                                {"key": "ok", "label": "ok", "delta": True, "from": 0, "to": 1},
                                _ew("good", "good", 2, 1, 3)])
    rs = ss.regressions(pane, min_delta=1)
    assert [r["key"] for r in rs] == ["good"], "bad rows dropped, the real one kept"


# --- dedup: open_issue_keys() + marker round-trip ---------------------------
def test_marker_and_open_issue_keys_roundtrip():
    body = "some text\n" + ss.marker("slop") + "\nmore"
    keys = ss.open_issue_keys([{"number": 1, "body": body},
                               {"number": 2, "body": "no marker here"},
                               {"number": 3, "body": ss.marker("code")}])
    assert keys == {"slop", "code"}


def test_plan_skips_already_open_key():
    pane = _pane(direction="regressed",
                 early_warning=[_ew("slop", "code-slop", 2, 4, 6),
                                _ew("code", "code", 9, 30, 39)])
    to_file, stats = ss.plan_issues(pane, open_keys={"code"},
                                    min_delta=1, max_issues=10, today="2026-06-26")
    keys = [i["key"] for i in to_file]
    assert keys == ["slop"], "the open 'code' regression is deduped out"
    assert stats["already-open"] == 1


# --- cap --------------------------------------------------------------------
def test_plan_caps_worst_first():
    ew = [_ew(f"k{i}", f"label{i}", i + 1, 0, i + 1) for i in range(6)]
    pane = _pane(direction="regressed", early_warning=ew)
    to_file, stats = ss.plan_issues(pane, open_keys=set(),
                                    min_delta=1, max_issues=2, today="2026-06-26")
    # Worst-first: k5 (+6), k4 (+5) survive the cap of 2.
    assert [i["key"] for i in to_file] == ["k5", "k4"]
    assert stats["over-cap"] == 4


# --- render -----------------------------------------------------------------
def test_render_issue_has_marker_evidence_and_contract():
    cand = {"key": "slop", "label": "code-slop", "delta": 6, "from": 4, "to": 10,
            "grade": "B", "portfolio_regressed": False, "baseline_commit": "base123"}
    issue = ss.render_issue(cand, commit="head456", today="2026-06-26",
                            tools={"slop": "tools/code_slop_scorecard.py"})
    assert issue["key"] == "slop"
    assert issue["labels"] == [ss.SIGNAL_LABEL, ss.DEBT_LABEL]
    assert ss.marker("slop") in issue["body"], "dedup anchor present"
    assert "+6" in issue["body"] and "(4 -> 10)" in issue["body"], "evidence: delta + from->to"
    assert "/slop-score" in issue["body"], "owning skill named"
    assert "tools/code_slop_scorecard.py" in issue["body"], "re-measure command named"
    assert "#N" in issue["body"], "the worker's #N-stamp contract is spelled out"
    assert "Advisory" in issue["body"], "green-portfolio rise framed as advisory"
    assert "+6" in issue["title"] and "code-slop" in issue["title"]


def test_render_issue_blocking_when_portfolio_regressed():
    cand = {"key": "code", "label": "code", "delta": 9, "from": 30, "to": 39,
            "grade": "C", "portfolio_regressed": True, "baseline_commit": "base123"}
    issue = ss.render_issue(cand, commit="head456", today="2026-06-26", tools={})
    assert "BLOCKING" in issue["body"], "portfolio regression escalates severity"
    # No mapped tool -> the generic control-pane re-check is offered instead.
    assert "scorecard_control_pane.py" in issue["body"]


def test_render_issue_unmapped_key_degrades_gracefully():
    cand = {"key": "mystery", "label": "mystery-metric", "delta": 3, "from": 1, "to": 4,
            "grade": None, "portfolio_regressed": False, "baseline_commit": ""}
    issue = ss.render_issue(cand, commit="c", today="2026-06-26", tools={})
    # No false skill lead; the generic conductor is offered instead.
    assert "/score-2x mystery-metric" in issue["body"]


def test_plan_unpinned_files_nothing():
    pane = _pane(direction="unpinned", early_warning=[_ew("code", "code", 9, 0, 9)])
    to_file, _ = ss.plan_issues(pane, open_keys=set(), min_delta=1,
                                max_issues=5, today="2026-06-26")
    assert to_file == []


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
