#!/usr/bin/env python3
"""Unit tests for score-signal's pure core — the relevance filter, the open-issue
dedup, the worst-first cap, the on-disk skill gating, and the issue render. In-memory
fixtures only (no gh, no network, no control-pane fold), so the testable seam runs on
the hermetic CI box.

Dual-runnable (the repo runs the suite pytest-free in CI):
    python tools/score_signal_test.py
    python -m pytest tools/score_signal_test.py -q
"""
from __future__ import annotations

import score_signal as ss

# The on-disk skill set the real render would compute; tests pass it explicitly so
# they never touch the filesystem.
SKILLS = {"slop-score", "quality-score", "refresh-readme", "industry-score"}


# --- fixtures ---------------------------------------------------------------
def _pane(*, direction: str, early_warning: list[dict], metrics: list[dict] | None = None,
          commit: str = "abc1234", baseline_commit: str = "dead00f",
          total_delta: int = 0) -> dict:
    """A minimal folded control-pane payload, shaped like scorecard_control_pane.fold."""
    return {
        "schema": "fak-scorecard-control-pane/1",
        "commit": commit,
        "metrics": metrics or [],
        "trend": {
            "direction": direction,
            "baseline_commit": baseline_commit,
            "total_delta": total_delta,
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
    to_file, refreshes, stats = ss.plan_issues(pane, open_keys={"code"},
                                               min_delta=1, max_issues=10,
                                               today="2026-06-26", available=SKILLS)
    keys = [i["key"] for i in to_file]
    assert keys == ["slop"], "the open 'code' regression is deduped out"
    assert stats["already-open"] == 1
    # With no open_index passed, the original skip-only behavior holds: no refresh.
    assert refreshes == [], "no index -> no refresh (back-compat)"


# --- cap --------------------------------------------------------------------
def test_plan_caps_worst_first():
    ew = [_ew(f"k{i}", f"label{i}", i + 1, 0, i + 1) for i in range(6)]
    pane = _pane(direction="regressed", early_warning=ew)
    to_file, _refreshes, stats = ss.plan_issues(pane, open_keys=set(),
                                                min_delta=1, max_issues=2,
                                                today="2026-06-26", available=SKILLS)
    assert [i["key"] for i in to_file] == ["k5", "k4"]
    assert stats["over-cap"] == 4


def test_plan_below_min_delta_is_accounted():
    pane = _pane(direction="regressed",
                 early_warning=[_ew("a", "a", 1, 0, 1), _ew("b", "b", 5, 0, 5)])
    to_file, _refreshes, stats = ss.plan_issues(pane, open_keys=set(), min_delta=3,
                                                max_issues=5, today="2026-06-26",
                                                available=SKILLS)
    assert [i["key"] for i in to_file] == ["b"]
    assert stats["below-min-delta"] == 1, "the +1 drop is reported, not silent"


# --- render -----------------------------------------------------------------
def test_render_issue_has_marker_evidence_and_contract():
    cand = {"key": "slop", "label": "code-slop", "delta": 6, "from": 4, "to": 10,
            "grade": "B", "portfolio_regressed": False, "baseline_commit": "base123"}
    issue = ss.render_issue(cand, commit="head456", today="2026-06-26",
                            tools={"slop": "tools/code_slop_scorecard.py"},
                            available=SKILLS)
    assert issue["key"] == "slop"
    assert issue["labels"] == [ss.SIGNAL_LABEL, ss.DEBT_LABEL]
    assert ss.marker("slop") in issue["body"], "dedup anchor present"
    assert "+6" in issue["body"] and "(4 -> 10)" in issue["body"], "evidence: delta + from->to"
    assert "/slop-score" in issue["body"], "owning skill named (it resolves on disk)"
    assert "tools/code_slop_scorecard.py" in issue["body"], "re-measure command named"
    assert "#N" in issue["body"], "the worker's #N-stamp contract is spelled out"
    assert "Advisory" in issue["body"], "green-portfolio rise framed as advisory"
    assert "+6" in issue["title"] and "code-slop" in issue["title"]


def test_render_issue_blocking_when_portfolio_regressed():
    cand = {"key": "code", "label": "code", "delta": 9, "from": 30, "to": 39,
            "grade": "C", "portfolio_regressed": True, "baseline_commit": "base123"}
    issue = ss.render_issue(cand, commit="head456", today="2026-06-26", tools={},
                            available=SKILLS)
    assert "BLOCKING" in issue["body"], "portfolio regression escalates severity"
    assert "scorecard_control_pane.py" in issue["body"]


def test_render_issue_unmapped_key_degrades_gracefully():
    cand = {"key": "mystery", "label": "mystery-metric", "delta": 3, "from": 1, "to": 4,
            "grade": None, "portfolio_regressed": False, "baseline_commit": ""}
    issue = ss.render_issue(cand, commit="c", today="2026-06-26", tools={},
                            available=SKILLS)
    assert "/score-2x mystery-metric" in issue["body"]


def test_render_issue_skill_gated_when_absent_on_disk():
    # 'slop' maps to /slop-score, but if that skill is NOT on disk it must NOT be
    # asserted as the owning pass — degrade to the generic conductor (the honesty fix).
    cand = {"key": "slop", "label": "code-slop", "delta": 6, "from": 4, "to": 10,
            "grade": "B", "portfolio_regressed": False, "baseline_commit": ""}
    issue = ss.render_issue(cand, commit="c", today="2026-06-26", tools={},
                            available=set())  # nothing resolves on disk
    assert "Run the owning RSI pass: `/slop-score`" not in issue["body"]
    assert "/score-2x code-slop" in issue["body"], "degraded to the fallback"
    assert "no dedicated skill resolves on disk" in issue["body"]


def test_render_issue_native_cmd_names_go_source_for_cmd_routing():
    # A native-cmd scorecard (dogfood) must name its Go source so the dispatch router
    # path-confirms it to the `cmd` lane, not `tools`.
    cand = {"key": "dogfood", "label": "dogfood-loop", "delta": 4, "from": 1, "to": 5,
            "grade": None, "portfolio_regressed": False, "baseline_commit": ""}
    issue = ss.render_issue(cand, commit="c", today="2026-06-26",
                            tools={"dogfood": "go run ./cmd/fak dogfood-score --json"},
                            available=set())
    assert "Fix location:" in issue["body"]
    assert "fak/cmd/fak/dogfoodscore.go" in issue["body"], "owning Go source named"
    assert "cmd` lane" in issue["body"]
    # No false skill lead: dogfood has no slash skill, so the fallback is used.
    assert "/score-2x dogfood-loop" in issue["body"]


def test_render_issue_footer_is_honest():
    cand = {"key": "code", "label": "code", "delta": 2, "from": 1, "to": 3,
            "grade": None, "portfolio_regressed": False, "baseline_commit": ""}
    body = ss.render_issue(cand, commit="c", today="2026-06-26", tools={},
                           available=SKILLS)["body"]
    assert "files a FRESH issue" in body, "honest: a fresh issue, not a reopen"
    assert "re-opens automatically" not in body, "old overstated wording gone"
    assert "+ its skills are listed" not in body, "footer no longer overclaims"


def test_no_stale_skill_mappings():
    # SKILL_BY_KEY must not carry the 3 skills the review found absent on disk.
    for dead in ("hygiene", "product", "dogfood"):
        assert dead not in ss.SKILL_BY_KEY, f"{dead} maps to a non-existent skill"


def test_plan_unpinned_files_nothing():
    pane = _pane(direction="unpinned", early_warning=[_ew("code", "code", 9, 0, 9)])
    to_file, refreshes, _ = ss.plan_issues(pane, open_keys=set(), min_delta=1,
                                           max_issues=5, today="2026-06-26",
                                           available=SKILLS)
    assert to_file == [] and refreshes == []


# --- #981: refresh an already-open ticket when its regression worsens -------
def test_delta_marker_roundtrips_through_open_index():
    # An issue body carrying both markers yields {number, noted_delta}.
    body = ("text\n" + ss.marker("slop") + "\n" + ss.delta_marker("slop", 6))
    idx = ss.open_issue_index([{"number": 42, "title": "x", "body": body}])
    assert idx == {"slop": {"number": 42, "noted_delta": 6}}


def test_open_index_falls_back_to_title_delta_when_no_marker():
    # An OLD issue filed before the delta-marker existed: noted delta read from title.
    body = "legacy body\n" + ss.marker("code")
    idx = ss.open_issue_index(
        [{"number": 7, "title": "score-signal: code debt rose +9 (30->39)", "body": body}])
    assert idx["code"]["noted_delta"] == 9, "title +N is the fallback"
    assert idx["code"]["number"] == 7


def test_open_index_floor_zero_when_no_marker_no_title_delta():
    body = "no delta anywhere\n" + ss.marker("code")
    idx = ss.open_issue_index([{"number": 5, "title": "no plus here", "body": body}])
    assert idx["code"]["noted_delta"] == 0, "conservative floor so a real rise trips"


def test_plan_refreshes_open_ticket_that_worsened():
    # 'code' is already open at noted +4; it now reads +9 (grew by 5 >= worsen 2).
    pane = _pane(direction="regressed",
                 early_warning=[_ew("code", "code", 9, 30, 39)])
    open_idx = {"code": {"number": 7, "noted_delta": 4}}
    to_file, refreshes, stats = ss.plan_issues(
        pane, open_keys={"code"}, min_delta=1, max_issues=5, today="2026-06-26",
        available=SKILLS, open_index=open_idx, worsen_delta=2)
    assert to_file == [], "no NEW issue — it is already open"
    assert len(refreshes) == 1, "worsened open ticket is REFRESHED, not dropped"
    r = refreshes[0]
    assert r["action"] == "refresh" and r["number"] == 7 and r["key"] == "code"
    assert r["delta"] == 9 and r["noted_delta"] == 4
    assert "+9" in r["title"] and "(30->39)" in r["title"], "title bumped to current"
    assert "+4" in r["comment"] and "+9" in r["comment"], "comment shows old->new"
    assert ss.delta_marker("code", 9) in r["comment"], "new noted-delta recorded"
    assert stats["refresh"] == 1 and stats["already-open"] == 1


def test_plan_does_not_refresh_a_flat_open_ticket():
    # 'code' open at noted +9, still reads +9 (grew by 0 < worsen 2) -> left alone.
    pane = _pane(direction="regressed",
                 early_warning=[_ew("code", "code", 9, 30, 39)])
    open_idx = {"code": {"number": 7, "noted_delta": 9}}
    to_file, refreshes, stats = ss.plan_issues(
        pane, open_keys={"code"}, min_delta=1, max_issues=5, today="2026-06-26",
        available=SKILLS, open_index=open_idx, worsen_delta=2)
    assert to_file == [] and refreshes == [], "a flat metric is never re-commented"
    assert stats["open-but-flat"] == 1 and stats["refresh"] == 0


def test_plan_refresh_below_threshold_is_not_fired():
    # grew by exactly 1, below worsen-delta 2 -> no refresh (bounded).
    pane = _pane(direction="regressed",
                 early_warning=[_ew("code", "code", 10, 30, 40)])
    open_idx = {"code": {"number": 7, "noted_delta": 9}}
    _to, refreshes, stats = ss.plan_issues(
        pane, open_keys={"code"}, min_delta=1, max_issues=5, today="2026-06-26",
        available=SKILLS, open_index=open_idx, worsen_delta=2)
    assert refreshes == [] and stats["open-but-flat"] == 1


def test_plan_refresh_is_bounded_once_per_metric_per_run():
    # Even with the same key surfacing once (dedup), at most ONE refresh is planned.
    pane = _pane(direction="regressed",
                 early_warning=[_ew("code", "code", 20, 0, 20)])
    open_idx = {"code": {"number": 7, "noted_delta": 1}}
    _to, refreshes, _stats = ss.plan_issues(
        pane, open_keys={"code"}, min_delta=1, max_issues=5, today="2026-06-26",
        available=SKILLS, open_index=open_idx, worsen_delta=2)
    assert len(refreshes) == 1, "one open issue per key -> one refresh max"


def test_plan_no_index_keeps_skip_only_behavior():
    # An already-open key with NO index entry (number unknown) is still skipped, not
    # refreshed — there is nothing to comment on.
    pane = _pane(direction="regressed",
                 early_warning=[_ew("code", "code", 9, 30, 39)])
    _to, refreshes, stats = ss.plan_issues(
        pane, open_keys={"code"}, min_delta=1, max_issues=5, today="2026-06-26",
        available=SKILLS, open_index={}, worsen_delta=2)
    assert refreshes == [] and stats["already-open"] == 1


def test_rendered_new_issue_carries_delta_marker():
    # A freshly-filed issue must seed the noted-delta marker so a later worsening
    # has a baseline to compare against.
    cand = {"key": "slop", "label": "code-slop", "delta": 6, "from": 4, "to": 10,
            "grade": "B", "portfolio_regressed": False, "baseline_commit": "b"}
    issue = ss.render_issue(cand, commit="c", today="2026-06-26",
                            tools={}, available=SKILLS)
    assert ss.delta_marker("slop", 6) in issue["body"], "noted-delta seeded on file"
    # And it round-trips: reading it back yields the same noted delta.
    idx = ss.open_issue_index([{"number": 1, "title": issue["title"],
                                "body": issue["body"]}])
    assert idx["slop"]["noted_delta"] == 6


# --- #980: the storm bound holds under AUTONOMOUS live filing -----------------
def test_steady_autonomous_run_plans_no_mutation_when_backlog_covers_every_rise():
    # #980 flipped the daily SCHEDULE to file live with no human in the loop, so the
    # "no storm" guarantee now rests entirely on the planner's own discipline. This
    # locks the load-bearing steady-state: on a day when every CURRENT regression is
    # already tracked by an OPEN issue and none worsened past --worsen-delta, the
    # autonomous run plans ZERO mutations — no new file, no refresh, no duplicate
    # comment. So a daily live run over an unchanged-but-open backlog is idempotent.
    pane = _pane(
        direction="regressed",
        early_warning=[_ew("code", "code", 9, 30, 39),     # open at +9, still +9 (flat)
                       _ew("slop", "code-slop", 6, 4, 10),  # open at +5, now +6 (+1 < 2)
                       _ew("appeal", "doc-appeal", 3, 1, 4)],  # open, number unknown
    )
    open_idx = {
        "code": {"number": 7, "noted_delta": 9},
        "slop": {"number": 8, "noted_delta": 5},
        # 'appeal' deliberately has no index entry — an older issue whose number the
        # dedup fetch knew only as an open KEY; it must still skip, never refresh.
    }
    to_file, refreshes, stats = ss.plan_issues(
        pane, open_keys={"code", "slop", "appeal"}, min_delta=1, max_issues=5,
        today="2026-06-27", available=SKILLS, open_index=open_idx, worsen_delta=2)
    assert to_file == [], "every current regression is already open -> nothing NEW filed"
    assert refreshes == [], "none worsened past --worsen-delta -> no refresh, no storm"
    assert stats["already-open"] == 3, "all three deduped against the open backlog"
    assert stats["open-but-flat"] == 2, "the two indexed-but-not-worsened are left alone"
    assert stats["refresh"] == 0, "a steady autonomous day mutates nothing"


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
