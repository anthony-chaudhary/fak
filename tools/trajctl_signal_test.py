#!/usr/bin/env python3
"""Unit tests for trajctl-signal's pure core — the two-fold normalize + relevance
filter (calibration collapse, stall-rate spike, nudge-outcome regression), the
open-issue dedup, the worst-drop-first cap, the contract-ready render, the
dedupe-to-UPDATE rerun (#2574's done condition), and the ROUTABILITY proof. The
committed regressed fixture (tools/trajctl_signal.data/regressed-health.json) is
the only on-disk input; no gh, no network, so the testable seam runs hermetically.

Dual-runnable (the repo runs the suite pytest-free in CI):
    python tools/trajctl_signal_test.py
    python -m pytest tools/trajctl_signal_test.py -q
"""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
from pathlib import Path

import trajctl_signal as ts

FIXTURE = Path(__file__).resolve().parent / "trajctl_signal.data" / "regressed-health.json"

# A minimal lane taxonomy mirroring dos.toml's trajctl/tools/ci trees, passed
# explicitly so the routability test never shells out to `dos`.
TREES = {"trajctl": ["internal/trajctl/**"], "tools": ["tools/**"], "ci": [".github/**"]}
CONCURRENT = ["trajctl", "tools", "ci"]


# --- fixtures ---------------------------------------------------------------
def _fixture_payload() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _payload(*, methods: list[dict] | None = None, stall: dict | None = None,
             nudges: list[dict] | None = None, commit: str = "abc1234",
             window: str = "7d") -> dict:
    """A minimal in-memory health envelope, shaped like the committed fixture."""
    return {
        "schema": ts.HEALTH_SCHEMA,
        "commit": commit,
        "window": window,
        "calibration": {"methods": methods or []},
        "metrics": {
            **({"stall_rate": stall} if stall is not None else {}),
            "nudge_outcomes": nudges or [],
        },
    }


def _cal(method: str, agreement: float, baseline: float, samples: int = 40,
         version: str = "1") -> dict:
    return {"method": method, "version": version, "agreement": agreement,
            "baseline_agreement": baseline, "samples": samples}


def _nudge(name: str, rate: float, baseline: float, samples: int = 12) -> dict:
    return {"nudge": name, "success_rate": rate,
            "baseline_success_rate": baseline, "samples": samples}


# --- normalize: the committed regressed fixture ------------------------------
def test_fixture_yields_exactly_one_calibration_collapse():
    regs = ts.regressions(_fixture_payload(), min_drop=0.10, min_samples=5)
    assert len(regs) == 1, "the fixture regresses exactly ONE fold row"
    r = regs[0]
    assert r["kind"] == "calibration" and r["key"] == "calibration:judge-verdict@2"
    assert round(r["drop"], 2) == 0.51, "0.82 -> 0.31 agreement collapse"
    assert r["direction"] == "collapsed" and r["samples"] == 40
    assert r["commit"] == "f00dfeed" and r["window"] == "7d"


def test_fixture_healthy_rows_are_skipped():
    # The fixture's flat stall-rate (+0.01), flat nudge (-0.01), and IMPROVED second
    # calibration method must all stay below the relevance filter.
    regs = ts.regressions(_fixture_payload(), min_drop=0.10, min_samples=5)
    keys = [r["key"] for r in regs]
    assert "stall-rate:fleet" not in keys, "a 1pt stall wobble is not a spike"
    assert "nudge-outcome:re-anchor" not in keys, "a 1pt nudge wobble is not a regression"
    assert all("witnessed-commit-progress" not in k for k in keys), \
        "an improved calibration method is never a collapse"


# --- normalize: all three kinds, worst-first ----------------------------------
def test_all_three_kinds_extract_worst_drop_first():
    pay = _payload(
        methods=[_cal("judge-verdict", 0.31, 0.82, version="2")],   # drop 0.51
        stall={"scope": "fleet", "current": 0.42, "baseline": 0.08,
               "sessions": 25},                                     # rise 0.34
        nudges=[_nudge("re-anchor", 0.35, 0.75)],                   # drop 0.40
    )
    regs = ts.regressions(pay, min_drop=0.10, min_samples=5)
    assert [r["key"] for r in regs] == [
        "calibration:judge-verdict@2",   # 0.51
        "nudge-outcome:re-anchor",       # 0.40
        "stall-rate:fleet",              # 0.34
    ], "worst drop first, deterministic"
    assert [r["direction"] for r in regs] == ["collapsed", "regressed", "spiked"]


def test_min_drop_filters_noise():
    pay = _payload(methods=[_cal("m", 0.75, 0.80)])  # drop 0.05 < 0.10
    assert ts.regressions(pay, min_drop=0.10, min_samples=5) == []


def test_sample_floor_skips_thin_evidence():
    pay = _payload(methods=[_cal("m", 0.20, 0.80, samples=2)])  # huge drop, 2 samples
    assert ts.regressions(pay, min_drop=0.10, min_samples=5) == [], \
        "a two-sample collapse is a measurement gap, not a regression"


def test_missing_rate_is_skipped_not_zeroed():
    pay = _payload(methods=[{"method": "m", "version": "1",
                             "baseline_agreement": 0.8, "samples": 40}],
                   stall={"current": None, "baseline": 0.1, "sessions": 30})
    assert ts.regressions(pay, min_drop=0.10, min_samples=5) == [], \
        "a missing current/baseline value never fabricates a collapse"


def test_improvement_is_not_a_regression():
    pay = _payload(
        methods=[_cal("m", 0.9, 0.6)],                                  # improved
        stall={"current": 0.02, "baseline": 0.30, "sessions": 25},      # fell
        nudges=[_nudge("n", 0.9, 0.5)],                                 # improved
    )
    assert ts.regressions(pay, min_drop=0.10, min_samples=5) == []


# --- dedup: marker round-trip -------------------------------------------------
def test_marker_and_open_issue_keys_roundtrip():
    body = "text\n" + ts.marker("calibration:judge-verdict@2") + "\nmore"
    keys = ts.open_issue_keys([{"number": 1, "body": body},
                               {"number": 2, "body": "no marker here"}])
    assert keys == {"calibration:judge-verdict@2"}


def test_drop_marker_roundtrips_through_open_index():
    body = ("x\n" + ts.marker("stall-rate:fleet") + "\n"
            + ts.drop_marker("stall-rate:fleet", 0.34))
    idx = ts.open_issue_index([{"number": 42, "title": "t", "body": body}])
    assert idx == {"stall-rate:fleet": {"number": 42, "noted_drop_bp": 3400}}


def test_open_index_floors_zero_without_drop_marker():
    body = "legacy body\n" + ts.marker("stall-rate:fleet")
    idx = ts.open_issue_index([{"number": 5, "body": body}])
    assert idx["stall-rate:fleet"]["noted_drop_bp"] == 0, \
        "conservative floor so a real worsening still trips"


def test_plan_skips_already_open_key():
    regs = ts.regressions(_fixture_payload(), min_drop=0.10, min_samples=5)
    to_file, updates, stats = ts.plan_issues(
        regs, open_keys={"calibration:judge-verdict@2"}, max_issues=5,
        today="2026-07-07")
    assert to_file == [] and updates == [], "no index -> skip-only dedup"
    assert stats["already-open"] == 1


# --- storm bound: --max-issues, worst-first -----------------------------------
def test_plan_caps_worst_first():
    nudges = [_nudge(f"n{i}", 0.75 - (i + 2) * 0.1, 0.75) for i in range(5)]
    regs = ts.regressions(_payload(nudges=nudges), min_drop=0.10, min_samples=5)
    assert len(regs) == 5, "fixture: five simultaneous regressions"
    to_file, _updates, stats = ts.plan_issues(regs, open_keys=set(), max_issues=2,
                                              today="2026-07-07")
    assert len(to_file) == 2 and stats["over-cap"] == 3
    assert to_file[0]["drop"] > to_file[1]["drop"], "worst-first survives the cap"


# --- render: contract-ready body ----------------------------------------------
def test_render_is_contract_ready():
    reg = ts.regressions(_fixture_payload(), min_drop=0.10, min_samples=5)[0]
    issue = ts.render_issue(reg, today="2026-07-07")
    assert issue["labels"] == [ts.SIGNAL_LABEL, ts.OBS_LABEL]
    body = issue["body"]
    for section in ("### Parent context", "### Current state", "### Why this is next",
                    "### Working spine", "### Priority context", "### In scope",
                    "### Out of scope", "### Done condition", "### Witness",
                    "### Acceptance gate", "### Lane", "### Path hints",
                    "### Boundary notes", "### Closure binding"):
        assert section in body, f"contract section {section} missing"
    assert ts.marker("calibration:judge-verdict@2") in body, "dedup anchor present"
    assert ts.drop_marker("calibration:judge-verdict@2", reg["drop"]) in body, \
        "noted-drop marker seeded on file"
    assert "0.82 -> 0.31" in body, "evidence: before -> after numbers"
    assert "#N" in body, "the worker's #N-stamp contract is spelled out"
    assert "fak/internal/trajctl/audit.go" in body, "owning substrate named (routing)"
    assert "(fak trajctl)" in body, "closure trailer named"
    assert "0.82 -> 0.31" in issue["title"] and "collapsed" in issue["title"]


def test_render_fix_hint_per_kind():
    pay = _payload(stall={"current": 0.42, "baseline": 0.08, "sessions": 25},
                   nudges=[_nudge("re-anchor", 0.35, 0.75)])
    regs = {r["kind"]: r for r in ts.regressions(pay, min_drop=0.10, min_samples=5)}
    stall_body = ts.render_issue(regs["stall-rate"], today="2026-07-07")["body"]
    nudge_body = ts.render_issue(regs["nudge-outcome"], today="2026-07-07")["body"]
    assert "fak/internal/trajctl/curve.go" in stall_body
    assert "fak/internal/trajctl/steer.go" in nudge_body


# --- ROUTABILITY proof --------------------------------------------------------
def test_rendered_body_path_confirms_the_trajctl_lane():
    import issue_lane_router as router
    reg = ts.regressions(_fixture_payload(), min_drop=0.10, min_samples=5)[0]
    issue = ts.render_issue(reg, today="2026-07-07")
    routed = router.route_issue(
        {"number": 1, "title": issue["title"], "body": issue["body"],
         "labels": [{"name": lab} for lab in issue["labels"]]},
        CONCURRENT, TREES)
    assert routed["lane"] == "trajctl", \
        f"owning internal/trajctl path should route to trajctl, got {routed!r}"
    assert routed["confidence"] == "path-confirmed", \
        "the path-grep rung (non-forgeable) confirmed the lane"


# --- THE WITNESS (#2574 done condition, part 1): dry-run -> exactly ONE issue --
def _main_json(argv: list[str], open_issues: list[dict]) -> dict:
    """Drive the real main() dry-run with fetch_open_issues patched (no gh)."""
    orig = ts.fetch_open_issues
    ts.fetch_open_issues = lambda limit: open_issues
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rc = ts.main(argv)
    finally:
        ts.fetch_open_issues = orig
    assert rc == 0, f"main exited {rc}: {buf.getvalue()[:400]}"
    return json.loads(buf.getvalue())


def test_dry_run_on_regressed_fixture_renders_exactly_one_issue():
    out = _main_json(["--from", str(FIXTURE), "--json"], open_issues=[])
    assert out["mode"] == "dry-run"
    assert len(out["planned"]) == 1, "the regressed fixture renders exactly ONE issue"
    assert out["planned"][0]["key"] == "calibration:judge-verdict@2"
    assert out["filed"] == [] and out["updated"] == [], "dry-run mutates nothing"
    assert out["updates"] == []


# --- THE WITNESS (part 2): a rerun DEDUPES — no duplicate, update on worsening --
def _first_run_open_issue() -> dict:
    """The issue the first dry-run rendered, replayed as the OPEN backlog entry."""
    reg = ts.regressions(_fixture_payload(), min_drop=0.10, min_samples=5)[0]
    issue = ts.render_issue(reg, today="2026-07-07")
    return {"number": 7, "title": issue["title"], "body": issue["body"]}


def test_rerun_same_fixture_dedupes_no_duplicate_no_spam():
    out = _main_json(["--from", str(FIXTURE), "--json"],
                     open_issues=[_first_run_open_issue()])
    assert out["planned"] == [], "the rerun files NO duplicate"
    assert out["updates"] == [], "an unchanged regression never re-comments"
    assert out["skipped"]["already-open"] == 1
    assert out["skipped"]["open-but-flat"] == 1, "flat rerun is idempotent"


def test_rerun_on_worsened_fixture_dedupes_to_an_update():
    worse = _fixture_payload()
    worse["calibration"]["methods"][0]["agreement"] = 0.11  # 0.51 -> 0.71 drop
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "worse.json"
        p.write_text(json.dumps(worse), encoding="utf-8")
        out = _main_json(["--from", str(p), "--json"],
                         open_issues=[_first_run_open_issue()])
    assert out["planned"] == [], "still NO duplicate issue"
    assert len(out["updates"]) == 1, "the rerun dedupes to an UPDATE of the open issue"
    u = out["updates"][0]
    assert u["number"] == 7 and u["key"] == "calibration:judge-verdict@2"
    assert u["noted_drop_bp"] == 5100 and round(u["drop"], 2) == 0.71
    assert "0.82 -> 0.11" in u["title"], "title bumped to current severity"
    assert out["updated"] == [], "dry-run plans the update, mutates nothing"


def test_update_render_reseeds_drop_marker_and_title_readback():
    # The planned update's comment carries the NEW noted-drop marker for humans, and
    # the BUMPED TITLE is the machine read-back path: the dedup fetch returns only
    # the (original) issue body, so the index must read the bumped severity from the
    # title — otherwise a live-updated issue would be re-updated every run.
    reg = ts.regressions(_fixture_payload(), min_drop=0.10, min_samples=5)[0]
    worse = dict(reg, current=0.11, drop=reg["baseline"] - 0.11)
    upd = ts.render_update(worse, number=7, noted_bp=5100, today="2026-07-07")
    assert ts.drop_marker(worse["key"], worse["drop"]) in upd["comment"]
    fetched = {"number": 7, "title": upd["title"],
               "body": _first_run_open_issue()["body"]}
    idx = ts.open_issue_index([fetched])
    assert idx[worse["key"]]["noted_drop_bp"] == ts.drop_bp(worse["drop"])
    # And the next flat rerun over the updated issue stays silent.
    worse_pay = _fixture_payload()
    worse_pay["calibration"]["methods"][0]["agreement"] = 0.11
    regs = ts.regressions(worse_pay, min_drop=0.10, min_samples=5)
    to_file, updates, stats = ts.plan_issues(
        regs, open_keys={worse["key"]}, max_issues=5, today="2026-07-07",
        open_index=idx, worsen_drop=0.05)
    assert to_file == [] and updates == [] and stats["open-but-flat"] == 1


def test_worsening_below_threshold_stays_silent():
    slight = _fixture_payload()
    slight["calibration"]["methods"][0]["agreement"] = 0.29  # +0.02 < worsen 0.05
    regs = ts.regressions(slight, min_drop=0.10, min_samples=5)
    open_idx = {"calibration:judge-verdict@2": {"number": 7, "noted_drop_bp": 5100}}
    to_file, updates, stats = ts.plan_issues(
        regs, open_keys={"calibration:judge-verdict@2"}, max_issues=5,
        today="2026-07-07", open_index=open_idx, worsen_drop=0.05)
    assert to_file == [] and updates == [], "a sub-threshold worsening never spams"
    assert stats["open-but-flat"] == 1


def _run() -> int:
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
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
