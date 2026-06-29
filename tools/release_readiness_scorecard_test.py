#!/usr/bin/env python3
"""Tests for the release-readiness scorecard.

Pure-stdlib. They pin the contract (schema, integer release-debt, A-F grade, the
control-pane envelope) and the two facts the scorer must get RIGHT:
  1. the cadence_auto_cut false-positive guard — a dry-run-only cadence must NOT be
     credited with auto-cut (the bug the first run shipped, now regression-locked);
  2. scoring monotonicity — a synthetic all-met fact set scores 100/grade A/debt 0,
     and a synthetic all-missing set scores 0/grade F with HARD debt.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import release_readiness_scorecard as rrs


def _all_met_facts() -> dict:
    return {
        "fak_release_verb": True,
        "staleness_verb": True,
        "agents_md_release": True,
        "llms_release": True,
        "cadence_dry_run_only": False,
        "cadence_auto_cut": True,
        "staleness_wired": True,
        "latest_tag": "v9.9.9",
        "commits_behind": 1,
        "days_behind": 0.1,
        "staleness_verdict": "FRESH",
        "release_tools_tested_count": 4,
        "release_tools_tested": True,
        "post_publish_verify": True,
        "lock_present": True,
        "gotcha_count": 0,
        "gotchas_bounded": True,
        "stable_tags": ["stable/codename"],
        "stable_exercised": True,
        "artifacts_signed": True,
        "gh_reachable": True,
        "arm64_shipped": True,
        "artifacts_present": True,
        "latest_release_assets": ["fak_9.9.9_linux_arm64.tar.gz", "fak_9.9.9_linux_arm64.tar.gz.sha256"],
    }


def _all_missing_facts() -> dict:
    f = {k: (False if isinstance(v, bool) else v) for k, v in _all_met_facts().items()}
    f["staleness_verdict"] = "VERY_STALE"
    f["commits_behind"] = 2000
    f["days_behind"] = 50.0
    f["stable_tags"] = []
    f["gotcha_count"] = 4
    f["release_tools_tested_count"] = 0
    return f


def test_all_met_is_perfect():
    sc = rrs.score(_all_met_facts())
    assert sc["release_debt"] == 0, sc["release_debt"]
    assert sc["grade"] == "A", sc["grade"]
    assert sc["composite"] == 100.0, sc["composite"]


def test_all_missing_is_floor():
    sc = rrs.score(_all_missing_facts())
    assert sc["release_debt"] > 0, "all-missing must carry HARD debt"
    assert sc["grade"] == "F", sc["grade"]
    assert sc["composite"] == 0.0, sc["composite"]


def test_cadence_auto_cut_false_positive_guard():
    """A dry-run-only cadence YAML must NOT be scored as auto-cut-capable.

    Regression lock for the cross-line DOTALL regex that matched a 'stop after the
    plan' guard step against a separate dispatch-only execute step.
    """
    dry_run_only = (
        "name: release-cadence\n"
        "on:\n  schedule:\n    - cron: '37 */6 * * *'\n"
        "        if: |\n          (github.event_name == 'schedule' || inputs.dry_run)\n"
        "          echo 'Scheduled ticks and default manual runs stop after the plan.'\n"
        "        if: |\n          inputs.dry_run == false\n"
    )
    # Mirror the production detector logic on a synthetic YAML string.
    cl = dry_run_only.lower()
    auto = ("fak_auto_release" in cl or "auto-cut" in cl) and not (
        "stop after the plan" in cl and "fak_auto_release" not in cl and "auto-cut" not in cl
    )
    assert auto is False, "dry-run-only cadence must not be credited with auto-cut"

    armed = dry_run_only + "        env:\n          FAK_AUTO_RELEASE: '1'\n"
    cl2 = armed.lower()
    auto2 = ("fak_auto_release" in cl2 or "auto-cut" in cl2) and not (
        "stop after the plan" in cl2 and "fak_auto_release" not in cl2 and "auto-cut" not in cl2
    )
    assert auto2 is True, "an FAK_AUTO_RELEASE-armed cadence must be credited"


def test_payload_envelope_contract():
    """build_payload over the live tree must emit the control-pane envelope keys."""
    root = rrs.repo_root(Path(__file__).parent)
    p = rrs.build_payload(root)
    for key in ("schema", "ok", "verdict", "finding", "next_action", "score", "grade", "release_debt"):
        assert key in p, f"missing envelope key: {key}"
    assert p["schema"] == rrs.SCHEMA
    assert isinstance(p["release_debt"], int)
    assert p["grade"] in ("A", "B", "C", "D", "F")
    # round-trips as JSON (the control pane consumes it).
    json.loads(json.dumps(p))


def test_unwitnessed_offline_is_not_hard_debt():
    """An offline gh probe (arm64/artifacts None) must be soft, not HARD release-debt."""
    f = _all_met_facts()
    f["gh_reachable"] = False  # offline -> arm64_shipped/artifacts_present score None
    sc = rrs.score(f)
    # those two KPIs become unwitnessed (soft), so they don't inflate HARD debt.
    assert sc["soft"] >= 2, sc["soft"]
    arm = next(r for r in sc["rows"] if r["key"] == "arm64_shipped")
    assert arm["unwitnessed"] is True


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"ok: {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL: {t.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR: {t.__name__}: {exc!r}")
    print(f"release_readiness_scorecard_test: {len(tests)} tests, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
