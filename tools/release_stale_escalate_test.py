#!/usr/bin/env python3
"""Unit tests for release-stale-escalate's pure core — the escalation `decide` rule, the
gate-signal envelope shaping, and the end-to-end proof that the shaped envelope is actually
consumable by the REAL gate_signal feeder and routes to the `ci` lane through the REAL
dispatch router. In-memory fixtures only (no gh, no git, no network), so the whole seam
runs on the hermetic CI box.

Dual-runnable (the repo runs the suite pytest-free in CI):
    python tools/release_stale_escalate_test.py
    python -m pytest tools/release_stale_escalate_test.py -q
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import release_stale_escalate as esc

# The lane taxonomy mirroring `dos doctor`'s ci/tools/cmd trees, passed explicitly so the
# routability proof never shells out to `dos` (same shape as gate_signal_test).
TREES = {"tools": ["tools/**"], "ci": [".github/**"], "cmd": ["cmd/**"]}
CONCURRENT = ["tools", "ci", "cmd"]


# --- fixtures ---------------------------------------------------------------
def _staleness(verdict: str, **over) -> dict:
    """A release-staleness --json envelope (fak-release-staleness/1) for `verdict`."""
    finding = {
        "very_stale": "publish_very_stale",
        "stale": "publish_stale",
        "fresh": "publish_fresh",
        "unknown": "publish_unknown",
    }.get(verdict, "publish_unknown")
    env = {
        "schema": "fak-release-staleness/1",
        "ok": verdict not in ("stale", "very_stale"),
        "verdict": verdict,
        "finding": finding,
        "reason": f"@latest is {verdict}",
        "next_action": "cut a release" if verdict in ("stale", "very_stale") else "",
        "latest_tag": "v0.1.0",
        "commits_behind": 1594,
        "days_behind": 60.0,
        "age_days": 60.0,
    }
    env.update(over)
    return env


def _route(issue: dict) -> dict:
    """Route a rendered gate-signal issue through the real dispatch router."""
    import issue_lane_router as router
    return router.route_issue(
        {"number": 1, "title": issue["title"], "body": issue["body"],
         "labels": [{"name": lab} for lab in issue["labels"]]},
        CONCURRENT, TREES)


# --- decide: only very_stale under a killed cadence escalates ----------------
def test_non_very_stale_verdicts_stay_informational():
    for v in ("fresh", "stale", "unknown", ""):
        # Even with auto-cut disabled AND the fail opt-in, a non-very_stale verdict is
        # informational — the escalation is scoped to very_stale ALONE.
        action, _ = esc.decide(v, auto_cut_disabled=True, fail_opt_in=True)
        assert action == esc.INFORMATIONAL, \
            f"verdict {v!r} must stay informational, got {action}"


def test_very_stale_with_live_auto_cut_stays_informational():
    # The cadence itself will close the lag on the next green window — no escalation.
    action, reason = esc.decide("very_stale", auto_cut_disabled=False, fail_opt_in=False)
    assert action == esc.INFORMATIONAL
    assert "armed" in reason


def test_very_stale_killed_cadence_default_files_issue():
    action, reason = esc.decide("very_stale", auto_cut_disabled=True, fail_opt_in=False)
    assert action == esc.FILE_ISSUE, "very_stale + killed auto-cut defaults to a tracking issue"
    assert "FAK_AUTO_RELEASE=0" in reason


def test_very_stale_killed_cadence_opt_in_fails_tick():
    action, reason = esc.decide("very_stale", auto_cut_disabled=True, fail_opt_in=True)
    assert action == esc.FAIL_TICK, "the repo opt-in escalates to a hard fail"
    assert "FAK_STALE_ESCALATION_FAIL=1" in reason


# --- envelope shaping -------------------------------------------------------
def test_escalation_envelope_shape_and_kill_switch_context():
    env = esc.escalation_envelope(_staleness("very_stale"))
    assert env["ok"] is False
    assert env["verdict"] == "BLOCKING", "BLOCKING drives gate_signal severity 4"
    assert env["finding"] == esc.ESCALATION_FINDING
    assert env["source"] == esc.ESCALATION_SOURCE
    assert env["where"] == ".github/workflows/release-cadence.yml"
    # The reason must NAME the kill switch as the crux and carry the measured lag.
    assert "FAK_AUTO_RELEASE=0" in env["reason"]
    assert "1594 commit" in env["reason"]
    assert "60d behind" in env["reason"]
    # next_action tells the operator how to make it self-heal.
    assert "FAK_AUTO_RELEASE" in env["next_action"]


def test_envelope_prefixes_upstream_next_action():
    env = esc.escalation_envelope(_staleness("very_stale", next_action="run /release now."))
    assert env["next_action"].startswith("run /release now."), \
        "the staleness envelope's own next_action is preserved ahead of ours"


# --- end-to-end: the envelope is gate_signal-consumable and routes to ci -----
def test_envelope_is_gate_signal_consumable_one_stable_finding():
    import gate_signal as gs
    findings = gs.normalize_findings(esc.escalation_envelope(_staleness("very_stale")))
    assert len(findings) == 1, "exactly one actionable finding (the escalation)"
    f = findings[0]
    # A stable per-signature key -> at most ONE open dedup'd issue, re-fileable after close.
    assert f["key"] == "release-cadence:publish_very_stale_under_killed_cadence", \
        f"unexpected dedup key {f['key']!r}"
    assert f["owning_path"] == ".github/workflows/release-cadence.yml"
    assert f["severity"] == 4, "BLOCKING -> severity 4 (worst-first)"


def test_rendered_issue_routes_to_ci_lane():
    import gate_signal as gs
    f = gs.normalize_findings(esc.escalation_envelope(_staleness("very_stale")))[0]
    issue = gs.render_issue(f, today="2026-07-01")
    routed = _route(issue)
    assert routed["lane"] == "ci", f"escalation issue must route to ci, got {routed!r}"
    assert routed["confidence"] == "path-confirmed"
    # The marker anchors gate_signal's open-issue dedup; the same key round-trips.
    assert gs.marker(f["key"]) in issue["body"]
    assert gs.open_issue_keys([{"body": issue["body"]}]) == {f["key"]}


# --- acceptance: the ticket's dry-run-the-tick scenario ---------------------
def test_acceptance_very_stale_fixture_escalates_and_emits_envelope(tmp_path=None):
    # #4025 acceptance: a fixture where @latest lags HEAD past the very-stale thresholds,
    # with auto-cut disabled, dry-runs to a file-issue decision AND writes a ci-routed,
    # gate_signal-consumable envelope. Exercises the driver end-to-end (no gh).
    import gate_signal as gs
    with tempfile.TemporaryDirectory() as d:
        stale_path = Path(d) / "staleness.json"
        env_path = Path(d) / "escalate-envelope.json"
        stale_path.write_text(json.dumps(_staleness("very_stale")), encoding="utf-8")
        rc = esc.main([
            "--from", str(stale_path),
            "--auto-cut-disabled", "true",
            "--emit-envelope", str(env_path),
            "--json",
        ])
        assert rc == 0, "a decided outcome exits 0 (the caller enforces file/fail)"
        assert env_path.exists(), "file-issue must emit the gate-signal envelope"
        emitted = json.loads(env_path.read_text(encoding="utf-8"))
        f = gs.normalize_findings(emitted)[0]
        assert _route(gs.render_issue(f, today="2026-07-01"))["lane"] == "ci"


def test_acceptance_killed_cadence_but_only_stale_emits_nothing():
    # The negative half: a merely `stale` (not very_stale) tick under the same killed
    # cadence stays informational and writes NO envelope — escalation is very_stale-only.
    with tempfile.TemporaryDirectory() as d:
        stale_path = Path(d) / "staleness.json"
        env_path = Path(d) / "escalate-envelope.json"
        stale_path.write_text(json.dumps(_staleness("stale")), encoding="utf-8")
        rc = esc.main([
            "--from", str(stale_path),
            "--auto-cut-disabled", "true",
            "--emit-envelope", str(env_path),
            "--json",
        ])
        assert rc == 0
        assert not env_path.exists(), "a stale (non-very_stale) tick emits no escalation"


def test_missing_envelope_is_infra_error_exit_2():
    rc = esc.main(["--from", "does-not-exist-xyz.json", "--json"])
    assert rc == 2, "an unreadable envelope is an infra error (exit 2), not a decision"


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
