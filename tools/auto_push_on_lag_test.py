#!/usr/bin/env python3
"""Hermetic tests for tools/auto_push_on_lag.py.

The push arm shells `fak sync push`; here that seam (`push_main`) and the git
read (`git_pane`) are replaced with synthetic results on the module, so NOTHING
live (fak/git push) is ever invoked. One guarded integration test builds a real
throwaway git repo to witness the trigger end-to-end, and skips on a git-less box.

Run: python tools/auto_push_on_lag_test.py   (also `python -m pytest -q`)
"""
from __future__ import annotations

import sys
import tempfile
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import auto_push_on_lag as apl  # noqa: E402

NOW = datetime(2026, 6, 25, 12, 0, 0, tzinfo=timezone.utc)


def _pane(**over):
    base = {
        "branch": "main", "sha": "abc1234", "ahead": 3, "behind": 0,
        "push_lag_seconds": 3600, "push_lag_stale": True, "dirty_lag_stale": False,
    }
    base.update(over)
    return base


# --- decide(): the pure admission seam --------------------------------------

def test_decide_on_main_and_stale_pushes() -> None:
    should, reason = apl.decide(_pane())
    assert should is True
    assert reason == "push-lag-60m"


def test_decide_dirty_only_does_not_push() -> None:
    # The common ACTION cause: a merely-dirty tree. push_lag_stale is False, so a
    # dirty tree must NEVER trigger a push regardless of dirty_lag_stale.
    should, reason = apl.decide(_pane(push_lag_stale=False, dirty_lag_stale=True))
    assert should is False
    assert reason == "no-push-lag"


def test_decide_off_main_skips() -> None:
    should, reason = apl.decide(_pane(branch="topic/x"))
    assert should is False
    assert reason == "not-on-main"


def test_decide_nothing_ahead_skips() -> None:
    should, reason = apl.decide(_pane(ahead=0, push_lag_seconds=None))
    assert should is False
    assert reason == "nothing-ahead"


# --- push_child_env(): the warn-only build-gate skip (#3300) ----------------

def test_push_child_env_forces_off_when_unset() -> None:
    # Default (unset) is the shell's `warn`: a pure-cost materialize that never
    # blocks. The backstop forces it off so it can't time out delivery.
    child = apl.push_child_env({})
    assert child["FLEET_BUILD_GUARD"] == "off"


def test_push_child_env_forces_off_when_warn() -> None:
    child = apl.push_child_env({"FLEET_BUILD_GUARD": "warn"})
    assert child["FLEET_BUILD_GUARD"] == "off"


def test_push_child_env_case_and_whitespace_insensitive() -> None:
    # The shell lowercases nothing, but an operator may; treat "warn"/"WARN"/" warn "
    # the same so a stray case never leaves the slow materialize armed.
    for raw in ("WARN", " warn ", "Warn"):
        assert apl.push_child_env({"FLEET_BUILD_GUARD": raw})["FLEET_BUILD_GUARD"] == "off"


def test_push_child_env_preserves_block() -> None:
    # block = an operator has made the gate ENFORCING; a real TRUNK_WOULD_NOT_COMPILE
    # must still refuse the push, so we must NOT override it.
    child = apl.push_child_env({"FLEET_BUILD_GUARD": "block"})
    assert child["FLEET_BUILD_GUARD"] == "block"


def test_push_child_env_keeps_other_vars() -> None:
    child = apl.push_child_env({"FLEET_BUILD_GUARD": "warn", "PATH": "/x", "FAK_BIN": "/b"})
    assert child["PATH"] == "/x" and child["FAK_BIN"] == "/b"


def test_push_main_passes_child_env_to_subprocess() -> None:
    # Witness the wiring: push_main must hand the build-gate-off env to the child so
    # the override actually reaches git's pre-push hook (not just compute it and drop it).
    seen: dict = {}

    class _CP:
        returncode = 0
        stdout = "{}"
        stderr = ""

    def fake_run(cmd, **kw):
        seen["env"] = kw.get("env")
        seen["timeout"] = kw.get("timeout")
        return _CP()

    fake_sp = types.SimpleNamespace(run=fake_run,
                                    TimeoutExpired=apl.subprocess.TimeoutExpired)
    with _Patch(subprocess=fake_sp):
        apl.push_main(Path(tempfile.mkdtemp()), ["fak"])
    assert seen["env"] is not None
    assert seen["env"]["FLEET_BUILD_GUARD"] == "off"
    assert seen["timeout"] == apl.PUSH_TIMEOUT_SECONDS


# --- run(): report vs. push, with the seams stubbed -------------------------

class _Patch:
    """Swap module attributes for the duration of a test, then restore."""

    def __init__(self, **attrs):
        self.attrs = attrs
        self.saved: dict = {}

    def __enter__(self):
        for k, v in self.attrs.items():
            self.saved[k] = getattr(apl, k)
            setattr(apl, k, v)
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(apl, k, v)


def _boom(*a, **k):  # a seam that must NOT be reached
    raise AssertionError("push_main must not be called")


def test_run_dry_run_never_pushes() -> None:
    root = Path(tempfile.mkdtemp())
    with _Patch(git_pane=lambda r, **k: _pane(), push_main=_boom):
        res = apl.run(root, live=False, push_lag_mins=45, now=NOW)
    assert res["verdict"] == "WOULD_PUSH"
    assert res["push_result"] is None
    assert res["ok"] is True


def test_run_live_pushes_once() -> None:
    calls = []

    def fake_push(r, fak):
        calls.append((r, fak))
        return {"ok": True, "pushed": True, "returncode": 0, "reason": "pushed"}

    root = Path(tempfile.mkdtemp())
    with _Patch(git_pane=lambda r, **k: _pane(), push_main=fake_push,
                resolve_fak=lambda r: ["fak"]):
        res = apl.run(root, live=True, push_lag_mins=45, now=NOW)
    assert res["verdict"] == "PUSHED"
    assert res["ok"] is True
    assert len(calls) == 1


def test_run_live_push_failure_is_not_ok() -> None:
    def fake_push(r, fak):
        return {"ok": False, "pushed": False, "returncode": 1, "reason": "push-rejected"}

    root = Path(tempfile.mkdtemp())
    with _Patch(git_pane=lambda r, **k: _pane(), push_main=fake_push,
                resolve_fak=lambda r: ["fak"]):
        res = apl.run(root, live=True, push_lag_mins=45, now=NOW)
    assert res["verdict"] == "PUSH_FAILED"
    assert res["ok"] is False  # exit code will be non-zero -> LastTaskResult surfaces it


def test_run_live_push_failure_surfaces_real_reason() -> None:
    # Regression: a failed push must record its REAL cause (push-rejected / push-timeout /
    # push-oserror), NOT the stale decide reason ("push-lag-NNm"). Else a stalled push reads
    # as a generic lag in auto-push.jsonl and the step that actually died — a rejected
    # pre-push gate vs a timed-out tip-materialize vs a credential hang — stays hidden.
    import json

    def fake_push(r, fak):
        return {"ok": False, "pushed": False, "returncode": 124, "reason": "push-timeout"}

    root = Path(tempfile.mkdtemp())
    with _Patch(git_pane=lambda r, **k: _pane(), push_main=fake_push,
                resolve_fak=lambda r: ["fak"]):
        res = apl.run(root, live=True, push_lag_mins=45, now=NOW)
    assert res["verdict"] == "PUSH_FAILED"
    assert res["reason"] == "push-timeout"  # not "push-lag-60m"
    rec = json.loads((root / apl.LOG_REL).read_text(encoding="utf-8").splitlines()[-1])
    assert rec["reason"] == "push-timeout"  # the breadcrumb carries the real cause


def test_run_live_without_fak_skips_never_bypasses() -> None:
    root = Path(tempfile.mkdtemp())
    with _Patch(git_pane=lambda r, **k: _pane(), push_main=_boom,
                resolve_fak=lambda r: None):
        res = apl.run(root, live=True, push_lag_mins=45, now=NOW)
    assert res["verdict"] == "SKIPPED"
    assert res["reason"] == "fak-unavailable"
    assert res["ok"] is False


def test_run_writes_jsonl_breadcrumb() -> None:
    import json
    root = Path(tempfile.mkdtemp())
    with _Patch(git_pane=lambda r, **k: _pane(push_lag_stale=False), push_main=_boom):
        apl.run(root, live=False, push_lag_mins=45, now=NOW)
    log = root / apl.LOG_REL
    assert log.exists()
    rec = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    assert rec["verdict"] == "NO_LAG" and rec["pushed"] is False


# --- integration: real repo, real git_pane, guarded for a git-less box ------

def _git(repo, *args):
    import subprocess
    base = ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t",
            "-c", "core.hooksPath=", "-c", "commit.gpgsign=false"]
    subprocess.run(base + list(args), check=True, capture_output=True, text=True)


def test_integration_stale_ahead_trips_decide() -> None:
    import subprocess
    try:
        repo = Path(tempfile.mkdtemp())
        _git(repo, "init", "-q", "-b", "main")
        (repo / "a.txt").write_text("a", encoding="utf-8")
        _git(repo, "add", "a.txt")
        _git(repo, "commit", "-q", "-m", "A", "--no-gpg-sign")
        # self-referential origin so @{upstream} resolves (per fresh_status_test)
        _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
        _git(repo, "config", "branch.main.remote", "origin")
        _git(repo, "config", "branch.main.merge", "refs/heads/main")
        _git(repo, "config", "remote.origin.url", ".")
        _git(repo, "config", "remote.origin.fetch", "+refs/heads/*:refs/remotes/origin/*")
        (repo / "b.txt").write_text("b", encoding="utf-8")
        _git(repo, "add", "b.txt")
        _git(repo, "commit", "-q", "-m", "B", "--no-gpg-sign")
    except (OSError, subprocess.SubprocessError):
        return  # git unavailable / sandboxed — skip, don't fail
    future = datetime.now(timezone.utc) + timedelta(hours=2)
    # Real git_pane + real decide: the unpushed B, 2h old, must arm the backstop.
    res = apl.run(repo, live=False, push_lag_mins=45, now=future)
    assert res["push_lag_stale"] is True
    assert res["ahead"] == 1 and res["branch"] == "main"
    assert res["verdict"] == "WOULD_PUSH"
    # A generous threshold keeps the same state a healthy no-op (no false alarm).
    ok = apl.run(repo, live=False, push_lag_mins=10 ** 9, now=future)
    assert ok["verdict"] == "NO_LAG"


# --- #6511: event-driven admission, backoff, typed health, counters ---------

NOW_TS = int(NOW.timestamp())


def test_backoff_seconds_is_exponential_and_capped() -> None:
    assert apl.backoff_seconds(0) == 0
    assert apl.backoff_seconds(1) == apl.BACKOFF_BASE_SECONDS
    assert apl.backoff_seconds(2) == 2 * apl.BACKOFF_BASE_SECONDS
    assert apl.backoff_seconds(3) == 4 * apl.BACKOFF_BASE_SECONDS
    # Saturates, and a very long stall must not build a giant int to get there.
    assert apl.backoff_seconds(9) == apl.BACKOFF_MAX_SECONDS
    assert apl.backoff_seconds(10_000) == apl.BACKOFF_MAX_SECONDS


def test_admit_arms_on_a_first_seen_stale_tip() -> None:
    adm = apl.admit(_pane(), {}, NOW_TS)
    assert adm.should is True and adm.verdict == "WOULD_PUSH"


def test_admit_suppresses_the_same_tip_with_no_new_commit() -> None:
    # THE #6511 fix: re-observing the tip we already attempted is not an event.
    # Level-triggered admission pushed here every single cadence tick.
    adm = apl.admit(_pane(), {"tip": "abc1234", "fail_streak": 0}, NOW_TS)
    assert adm.should is False
    assert adm.verdict == "NO_EVENT"
    assert adm.reason == "no-new-commit-since-abc1234"


def test_admit_arms_again_on_an_ahead_of_origin_transition() -> None:
    adm = apl.admit(_pane(sha="def5678"), {"tip": "abc1234", "fail_streak": 0}, NOW_TS)
    assert adm.should is True and adm.verdict == "WOULD_PUSH"


def test_admit_parks_inside_the_backoff_window_with_the_guard_reason() -> None:
    state = {"tip": "abc1234", "fail_streak": 2,
             "backoff_until": NOW_TS + 3600, "guard_reason": "BEHIND:diverged"}
    adm = apl.admit(_pane(), state, NOW_TS)
    assert adm.should is False
    assert adm.verdict == "BACKOFF"
    assert "after-2-rejects" in adm.reason and "BEHIND:diverged" in adm.reason


def test_admit_retries_the_same_tip_once_backoff_elapses() -> None:
    # Backoff must be a RETRY, not a permanent park: the same unpushed tip is
    # admitted again once the window passes, even with no new commit to trigger it.
    state = {"tip": "abc1234", "fail_streak": 2,
             "backoff_until": NOW_TS - 1, "guard_reason": "BEHIND"}
    adm = apl.admit(_pane(), state, NOW_TS)
    assert adm.should is True and adm.verdict == "WOULD_PUSH"


def test_admit_survives_a_corrupt_state_file() -> None:
    adm = apl.admit(_pane(), {"fail_streak": "nonsense", "backoff_until": None}, NOW_TS)
    assert adm.should is True


def test_push_guard_reason_prefers_the_safe_push_vocabulary() -> None:
    pr = {"pushed": False, "reason": "push-rejected",
          "result": {"pushed": False, "reason": "BEHIND", "divergence": "diverged"}}
    assert apl.push_guard_reason(pr) == "BEHIND:diverged"
    assert apl.push_guard_reason({"pushed": False, "reason": "push-rejected",
                                  "result": {"reason": "PUSH_ERROR"}}) == "PUSH_ERROR"
    # No parsable child JSON: fall back to our own transport-level cause.
    assert apl.push_guard_reason({"pushed": False, "reason": "push-timeout",
                                  "result": None}) == "push-timeout"
    assert apl.push_guard_reason({"pushed": True}) == ""


def test_health_and_exit_codes_are_typed() -> None:
    assert apl.health_of("PUSHED") == apl.HEALTH_OK
    assert apl.health_of("NO_EVENT") == apl.HEALTH_OK
    assert apl.health_of("BACKOFF") == apl.HEALTH_DEGRADED
    assert apl.health_of("SKIPPED") == apl.HEALTH_DEGRADED
    assert apl.health_of("PUSH_FAILED") == apl.HEALTH_FAILED
    # An unclassified verdict is degraded, never a green 0.
    assert apl.health_of("SOMETHING_NEW") == apl.HEALTH_DEGRADED
    assert apl.exit_code(apl.HEALTH_OK) == 0
    assert apl.exit_code(apl.HEALTH_FAILED) == 1
    assert apl.exit_code(apl.HEALTH_DEGRADED) == 2


def _fail_push(reason: str = "push-rejected", parsed: dict | None = None):
    def fake_push(r, fak):
        return {"ok": False, "pushed": False, "returncode": 1,
                "reason": reason, "result": parsed}
    return fake_push


def test_run_live_failure_arms_backoff_and_records_the_guard_reason() -> None:
    root = Path(tempfile.mkdtemp())
    with _Patch(git_pane=lambda r, **k: _pane(),
                push_main=_fail_push(parsed={"reason": "BEHIND", "divergence": "diverged"}),
                resolve_fak=lambda r: ["fak"]):
        res = apl.run(root, live=True, push_lag_mins=45, now=NOW)
    assert res["verdict"] == "PUSH_FAILED"
    assert res["health"] == apl.HEALTH_FAILED
    assert res["guard_reason"] == "BEHIND:diverged"
    assert res["fail_streak"] == 1
    assert res["backoff_seconds"] == apl.BACKOFF_BASE_SECONDS
    state = apl.load_state(root)
    assert state["fail_streak"] == 1
    assert state["backoff_until"] == int(NOW.timestamp()) + apl.BACKOFF_BASE_SECONDS
    assert state["tip"] == "abc1234"


def test_run_backoff_window_suppresses_the_next_tick() -> None:
    # The observed pathology: a rejection re-fired every 15 minutes for hours. The
    # second tick must NOT reach push_main at all, and must not read as healthy.
    root = Path(tempfile.mkdtemp())
    with _Patch(git_pane=lambda r, **k: _pane(),
                push_main=_fail_push(parsed={"reason": "PUSH_ERROR"}),
                resolve_fak=lambda r: ["fak"]):
        apl.run(root, live=True, push_lag_mins=45, now=NOW)
        res = apl.run(root, live=True, push_lag_mins=45,
                      now=NOW + timedelta(minutes=15))
    assert res["verdict"] == "BACKOFF"
    assert res["health"] == apl.HEALTH_DEGRADED
    assert res["ok"] is False
    assert res["guard_reason"] == "PUSH_ERROR"
    assert res["backoff_seconds"] > 0


def test_run_retries_and_escalates_the_backoff_after_the_window() -> None:
    root = Path(tempfile.mkdtemp())
    with _Patch(git_pane=lambda r, **k: _pane(),
                push_main=_fail_push(parsed={"reason": "BEHIND"}),
                resolve_fak=lambda r: ["fak"]):
        apl.run(root, live=True, push_lag_mins=45, now=NOW)
        later = NOW + timedelta(seconds=apl.BACKOFF_BASE_SECONDS + 60)
        res = apl.run(root, live=True, push_lag_mins=45, now=later)
    assert res["verdict"] == "PUSH_FAILED"
    assert res["fail_streak"] == 2
    assert res["backoff_seconds"] == 2 * apl.BACKOFF_BASE_SECONDS


def test_run_live_success_clears_the_streak_and_books_lag_reduced() -> None:
    def ok_push(r, fak):
        return {"ok": True, "pushed": True, "returncode": 0, "reason": "pushed"}

    root = Path(tempfile.mkdtemp())
    with _Patch(git_pane=lambda r, **k: _pane(),
                push_main=_fail_push(parsed={"reason": "BEHIND"}),
                resolve_fak=lambda r: ["fak"]):
        apl.run(root, live=True, push_lag_mins=45, now=NOW)
    later = NOW + timedelta(seconds=apl.BACKOFF_BASE_SECONDS + 60)
    with _Patch(git_pane=lambda r, **k: _pane(), push_main=ok_push,
                resolve_fak=lambda r: ["fak"]):
        res = apl.run(root, live=True, push_lag_mins=45, now=later)
    assert res["verdict"] == "PUSHED"
    assert res["fail_streak"] == 0
    assert res["lag_reduced_seconds"] == 3600  # the lag this push actually drained
    state = apl.load_state(root)
    assert state["fail_streak"] == 0 and state["backoff_until"] == 0
    assert state["guard_reason"] == ""


def test_run_second_live_tick_on_the_same_tip_never_pushes_again() -> None:
    calls = []

    def ok_push(r, fak):
        calls.append(r)
        return {"ok": True, "pushed": True, "returncode": 0, "reason": "pushed"}

    root = Path(tempfile.mkdtemp())
    # Still ahead on the SAME tip after the push (e.g. the ref moved under us):
    # there was no new ahead-of-origin event, so the backstop must stand down.
    with _Patch(git_pane=lambda r, **k: _pane(), push_main=ok_push,
                resolve_fak=lambda r: ["fak"]):
        apl.run(root, live=True, push_lag_mins=45, now=NOW)
        res = apl.run(root, live=True, push_lag_mins=45,
                      now=NOW + timedelta(minutes=15))
    assert len(calls) == 1
    assert res["verdict"] == "NO_EVENT"
    assert res["ok"] is True  # a suppressed no-op is healthy, not degraded


def test_run_dry_tick_never_records_the_tip() -> None:
    # A dry-run report must not consume the event; otherwise the next LIVE tick
    # would read as NO_EVENT and a real stall would go undelivered.
    root = Path(tempfile.mkdtemp())
    with _Patch(git_pane=lambda r, **k: _pane(), push_main=_boom):
        apl.run(root, live=False, push_lag_mins=45, now=NOW)
    assert apl.load_state(root).get("tip") in (None, "")

    calls = []

    def ok_push(r, fak):
        calls.append(r)
        return {"ok": True, "pushed": True, "returncode": 0, "reason": "pushed"}

    with _Patch(git_pane=lambda r, **k: _pane(), push_main=ok_push,
                resolve_fak=lambda r: ["fak"]):
        res = apl.run(root, live=True, push_lag_mins=45, now=NOW)
    assert res["verdict"] == "PUSHED" and len(calls) == 1


def test_run_clears_a_stale_backoff_once_nothing_is_ahead() -> None:
    root = Path(tempfile.mkdtemp())
    with _Patch(git_pane=lambda r, **k: _pane(),
                push_main=_fail_push(parsed={"reason": "BEHIND"}),
                resolve_fak=lambda r: ["fak"]):
        apl.run(root, live=True, push_lag_mins=45, now=NOW)
    # A peer pushed for us: the lag is gone, so the parked window is moot and must
    # not delay the NEXT real event.
    with _Patch(git_pane=lambda r, **k: _pane(ahead=0, push_lag_seconds=None,
                                              push_lag_stale=False),
                push_main=_boom):
        res = apl.run(root, live=True, push_lag_mins=45,
                      now=NOW + timedelta(minutes=1))
    assert res["verdict"] == "NO_LAG" and res["ok"] is True
    state = apl.load_state(root)
    assert state["fail_streak"] == 0 and state["backoff_until"] == 0


def test_run_accumulates_yield_and_cost_counters() -> None:
    def ok_push(r, fak):
        return {"ok": True, "pushed": True, "returncode": 0, "reason": "pushed"}

    root = Path(tempfile.mkdtemp())
    with _Patch(git_pane=lambda r, **k: _pane(), push_main=ok_push,
                resolve_fak=lambda r: ["fak"]):
        apl.run(root, live=True, push_lag_mins=45, now=NOW)
        res = apl.run(root, live=True, push_lag_mins=45,
                      now=NOW + timedelta(minutes=15))
    counters = res["counters"]
    assert counters["ticks"] == 2 and counters["pushes"] == 1
    assert counters["suppressed"] == 1              # the NO_EVENT tick
    assert counters["pushes_per_tick"] == 0.5
    assert counters["lag_reduced_seconds"] == 3600
    assert counters["cost_ms"] >= 0
    assert apl.load_state(root)["counters"] == counters


def test_run_ledger_record_carries_health_and_measurements() -> None:
    import json
    root = Path(tempfile.mkdtemp())
    with _Patch(git_pane=lambda r, **k: _pane(),
                push_main=_fail_push(parsed={"reason": "BEHIND"}),
                resolve_fak=lambda r: ["fak"]):
        apl.run(root, live=True, push_lag_mins=45, now=NOW)
    rec = json.loads((root / apl.LOG_REL).read_text(encoding="utf-8").splitlines()[-1])
    assert rec["health"] == apl.HEALTH_FAILED
    assert rec["guard_reason"] == "BEHIND"
    assert rec["fail_streak"] == 1
    assert rec["backoff_seconds"] == apl.BACKOFF_BASE_SECONDS
    assert rec["lag_reduced_seconds"] == 0 and rec["duration_ms"] >= 0


def test_summarize_folds_yield_failure_streaks_and_cost() -> None:
    records = [
        {"verdict": "NO_LAG", "duration_ms": 10},
        {"verdict": "PUSHED", "lag_reduced_seconds": 600, "duration_ms": 100},
        {"verdict": "PUSH_FAILED", "duration_ms": 50},
        {"verdict": "PUSH_FAILED", "duration_ms": 50},
        {"verdict": "BACKOFF", "duration_ms": 5},
        {"verdict": "PUSHED", "lag_reduced_seconds": 300, "duration_ms": 100},
        {"verdict": "PUSH_FAILED", "duration_ms": 50},
    ]
    s = apl.summarize(records)
    assert s["ticks"] == 7 and s["pushes"] == 2 and s["failures"] == 3
    assert s["suppressed"] == 1
    assert s["no_op_ticks"] == 2                     # NO_LAG + BACKOFF
    assert s["max_fail_streak"] == 2                 # the middle pair, not all three
    assert s["current_fail_streak"] == 1             # trailing streak, still open
    assert s["lag_reduced_seconds"] == 900
    assert s["cost_ms"] == 365
    assert s["cost_ms_per_push"] == 182.5
    assert s["failure_rate"] == 0.6                  # over ACTING ticks, not all ticks


def test_summarize_of_an_empty_ledger_is_not_a_divide_by_zero() -> None:
    s = apl.summarize([])
    assert s["ticks"] == 0 and s["pushes_per_tick"] == 0.0
    assert s["no_op_rate"] == 0.0 and s["cost_ms_per_push"] is None


def test_read_ledger_skips_a_truncated_tail() -> None:
    root = Path(tempfile.mkdtemp())
    log = root / apl.LOG_REL
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text('{"verdict":"PUSHED"}\n\n{"verdict":"PUSH_F\n', encoding="utf-8")
    assert [r["verdict"] for r in apl.read_ledger(root)] == ["PUSHED"]
    assert apl.read_ledger(Path(tempfile.mkdtemp())) == []


def test_main_exit_code_follows_typed_health() -> None:
    root = str(Path(tempfile.mkdtemp()))
    for health, want in ((apl.HEALTH_OK, 0), (apl.HEALTH_DEGRADED, 2),
                         (apl.HEALTH_FAILED, 1)):
        def fake_run(r, *, live, push_lag_mins, now=None, _h=health):
            return {"health": _h, "verdict": "X", "reason": "r", "live": live,
                    "branch": "main", "ahead": 1, "push_lag_seconds": 60, "ok": _h == "ok"}

        with _Patch(run=fake_run):
            assert apl.main(["--workspace", root, "--json"]) == want


def test_main_metrics_mode_reads_only() -> None:
    import json
    root = Path(tempfile.mkdtemp())
    log = root / apl.LOG_REL
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps({"verdict": "PUSHED", "lag_reduced_seconds": 60,
                               "duration_ms": 7}) + "\n", encoding="utf-8")

    def _no_tick(*a, **k):
        raise AssertionError("--metrics must never tick")

    with _Patch(run=_no_tick):
        assert apl.main(["--workspace", str(root), "--metrics", "--json"]) == 0


def _run_all() -> int:
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
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
