#!/usr/bin/env python3
"""Unit tests for gate-signal's pure core — the two-envelope normalize + relevance
filter, the open-issue dedup, the worst-first cap, the issue render, and the
ROUTABILITY proof (a rendered body path-confirms a lane via issue_lane_router).
In-memory fixtures only (no gh, no network), so the testable seam runs on the
hermetic CI box.

Dual-runnable (the repo runs the suite pytest-free in CI):
    python tools/gate_signal_test.py
    python -m pytest tools/gate_signal_test.py -q
"""
from __future__ import annotations

import gate_signal as gs

# A minimal lane taxonomy mirroring `dos doctor`'s tools/ci trees, passed explicitly
# so the routability test never shells out to `dos`.
TREES = {"tools": ["tools/**"], "ci": [".github/**"], "cmd": ["cmd/**"]}
CONCURRENT = ["tools", "ci", "cmd"]


# --- fixtures ---------------------------------------------------------------
def _posture(findings: list[dict]) -> dict:
    """The security_audit.py --json envelope shape."""
    fails = sum(1 for f in findings if f.get("level") == "FAIL")
    warns = sum(1 for f in findings if f.get("level") == "WARN")
    return {
        "root": "/repo", "schema": "security-audit",
        "checks": ["secret-leak-gate", "dependency-surface"],
        "fail": fails, "warn": warns, "findings": findings,
    }


def _generic(members: list[dict], **top) -> dict:
    base = {"schema": "garden-bundle/1", "ok": all(m.get("ok", True) for m in members)}
    base.update(top)
    base["members"] = members
    return base


# --- normalize: posture envelope --------------------------------------------
def test_posture_fail_is_actionable_warn_skipped():
    env = _posture([
        {"check": "secret-leak-gate", "level": "FAIL", "msg": "scanner missing",
         "where": "tools/scrub_public_copy.py"},
        {"check": "dependency-surface", "level": "WARN", "msg": "go.sum drift"},
    ])
    fs = gs.normalize_findings(env)
    assert [f["key"] for f in fs] == ["security-audit:secret-leak-gate"], \
        "only the FAIL is a finding; the WARN is advisory and skipped"
    assert fs[0]["detail"] == "scanner missing"


def test_posture_all_warn_yields_nothing():
    env = _posture([{"check": "x", "level": "WARN", "msg": "m"}])
    assert gs.normalize_findings(env) == [], "a WARN-only envelope is not actionable"


def test_posture_clean_envelope_yields_nothing():
    assert gs.normalize_findings(_posture([])) == [], "no findings -> nothing"


# --- normalize: generic verdict envelope ------------------------------------
def test_generic_red_member_is_actionable_green_skipped():
    env = _generic([
        {"source": "garden", "verdict": "REGRESSED", "ok": False,
         "finding": "scorecard ratchet red", "reason": "slop debt rose",
         "next_action": "run /slop-score"},
        {"source": "fresh-status", "verdict": "OK", "ok": True,
         "finding": "fresh", "reason": "all current"},
    ])
    fs = gs.normalize_findings(env)
    keys = [f["key"] for f in fs]
    assert any(k.startswith("garden:") for k in keys), "the red member is a finding"
    assert all("fresh-status" not in k for k in keys), "the ok=True member is skipped"
    assert fs[0]["next_action"] == "run /slop-score", "next_action carried through"


def test_generic_blocking_sorts_first():
    env = _generic([
        {"source": "a", "verdict": "FAIL", "ok": False, "finding": "f1"},
        {"source": "b", "verdict": "BLOCKING", "ok": False, "finding": "f2"},
    ])
    fs = gs.normalize_findings(env)
    assert fs[0]["source"] == "b", "BLOCKING (severity 4) sorts before FAIL (3)"


def test_generic_single_member_payload():
    # The whole payload is the member when there is no members[] list. `detail` is
    # the human-readable `reason`; the `finding` slug anchors the key.
    env = {"schema": "loop-audit", "ok": False, "verdict": "FAIL",
           "finding": "lane stalled", "reason": "no heartbeat", "next_action": "kick"}
    fs = gs.normalize_findings(env)
    assert len(fs) == 1
    assert fs[0]["key"] == "loop-audit:lane-stalled", "finding slug anchors the key"
    assert fs[0]["detail"] == "no heartbeat", "detail is the reason"
    assert fs[0]["next_action"] == "kick"


def test_generic_ok_true_surfaced_condition_is_not_a_finding():
    # A pass surfacing a benign condition keeps ok=True -> not red -> skipped.
    env = _generic([{"source": "loop-audit", "ok": True, "verdict": "OK",
                     "finding": "1 idle lane (advisory)"}])
    assert gs.normalize_findings(env) == []


# --- dedup ------------------------------------------------------------------
def test_marker_and_open_issue_keys_roundtrip():
    body = "x\n" + gs.marker("security-audit:secret-leak-gate") + "\ny"
    keys = gs.open_issue_keys([{"number": 1, "body": body},
                               {"number": 2, "body": "no marker"}])
    assert keys == {"security-audit:secret-leak-gate"}


def test_plan_skips_already_open_key():
    fs = gs.normalize_findings(_posture([
        {"check": "a", "level": "FAIL", "msg": "m1"},
        {"check": "b", "level": "FAIL", "msg": "m2"},
    ]))
    to_file, stats = gs.plan_issues(
        fs, open_keys={"security-audit:a"}, max_issues=10, today="2026-06-27")
    keys = [i["key"] for i in to_file]
    assert keys == ["security-audit:b"], "the open 'a' finding is deduped out"
    assert stats["already-open"] == 1


# --- cap --------------------------------------------------------------------
def test_plan_caps_worst_first():
    members = [{"source": f"s{i}", "verdict": "FAIL", "ok": False,
                "finding": f"f{i}"} for i in range(5)]
    fs = gs.normalize_findings(_generic(members))
    to_file, stats = gs.plan_issues(fs, open_keys=set(), max_issues=2,
                                    today="2026-06-27")
    assert len(to_file) == 2 and stats["over-cap"] == 3


# --- render -----------------------------------------------------------------
def test_render_has_marker_owning_path_and_contract():
    env = _posture([{"check": "secret-leak-gate", "level": "FAIL",
                     "msg": "scanner missing"}])
    issue = gs.render_issue(gs.normalize_findings(env)[0], today="2026-06-27")
    assert issue["labels"] == [gs.SIGNAL_LABEL, gs.CI_LABEL]
    assert gs.marker("security-audit:secret-leak-gate") in issue["body"], "dedup anchor"
    assert "tools/security_audit.py" in issue["body"], "owning path named (routing)"
    assert "#N" in issue["body"], "the worker's #N-stamp contract is spelled out"
    assert "FAIL" in issue["title"]


def test_render_where_path_overrides_source_map():
    # A finding that names its own real tree path uses THAT as the owning path.
    env = _generic([{"source": "garden", "verdict": "FAIL", "ok": False,
                     "finding": "broken link", "where": "docs/INDEX.md"}])
    issue = gs.render_issue(gs.normalize_findings(env)[0], today="2026-06-27")
    assert "docs/INDEX.md" in issue["body"], "finding's own path wins"


# --- ROUTABILITY proof (the #978 acceptance) --------------------------------
def test_rendered_body_path_confirms_a_lane():
    import issue_lane_router as router
    env = _posture([{"check": "secret-leak-gate", "level": "FAIL",
                     "msg": "scanner missing"}])
    issue = gs.render_issue(gs.normalize_findings(env)[0], today="2026-06-27")
    # The router reads title+labels+body; build the issue shape it expects.
    routed = router.route_issue(
        {"number": 1, "title": issue["title"], "body": issue["body"],
         "labels": [{"name": lab} for lab in issue["labels"]]},
        CONCURRENT, TREES)
    assert routed["lane"] == "tools", \
        f"owning tools/ path should route to the tools lane, got {routed!r}"
    assert routed["confidence"] == "path-confirmed", \
        "the path-grep rung (non-forgeable) confirmed the lane"


def test_garden_finding_routes_to_owning_tool_lane():
    import issue_lane_router as router
    # A garden-red finding's FIX lives in its owning tool (tools/garden_bundle.py via
    # PATH_BY_SOURCE), so the rendered ticket path-confirms the `tools` lane — the
    # lane a worker actually needs. (`.github/` workflow paths do not path-confirm
    # through the router's _PATH_RE today; the owning *tool* is the routable signal.)
    env = _generic([{"source": "garden", "verdict": "REGRESSED", "ok": False,
                     "finding": "garden red"}])
    f = gs.normalize_findings(env)[0]
    assert f["owning_path"] == "tools/garden_bundle.py", "PATH_BY_SOURCE resolves it"
    issue = gs.render_issue(f, today="2026-06-27")
    routed = router.route_issue(
        {"number": 2, "title": issue["title"], "body": issue["body"],
         "labels": [{"name": lab} for lab in issue["labels"]]},
        CONCURRENT, TREES)
    assert routed["lane"] == "tools", f"owning tool routes to tools, got {routed!r}"
    assert routed["confidence"] == "path-confirmed"


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
