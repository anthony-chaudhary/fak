#!/usr/bin/env python3
"""Regression tests for fleet_resume_watchdog.py's cross-account env wiring.

These pin the parity the .ps1 already had and the .py port had drifted from:
  * REG_DIR must follow FLEET_REG_DIR so the watchdog READS the registry/plan/ledger
    from the same dir the fleet_sessions.py refresh child WRITES (the silent-no-op /
    split-resume-once-ledger blocker when an ambient FLEET_REG_DIR is set by the
    control pane or an operator) -- and with the variable UNSET both sides must still
    land on the one host registry the shared resolver picks, never a second one (#5390).
  * CLAUDE_EXE must prefer the fleet-wide FLEET_CLAUDE_EXE convention (account_probe.py
    et al.), with FAK_CLAUDE_EXE only a back-compat fallback.
  * Active probing must stay gated to the live tick so a default dry-run spends nothing.

The module reads the env-derived constants at import time, so each test reloads it
under a patched environment. Pure stdlib; no process spawn, no network.

Run:  python -m pytest tools/fleet_resume_watchdog_test.py
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _reload(env):
    """Reload fleet_resume_watchdog with `env` overlaid, then restore the environment."""
    saved = {k: os.environ.get(k) for k in env}
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        import fleet_resume_watchdog as wd
        return importlib.reload(wd)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_reg_dir_follows_fleet_reg_dir(tmp_path):
    target = str(tmp_path / "host_registry")
    wd = _reload({"FLEET_REG_DIR": target})
    assert wd.REG_DIR == target, "watchdog must read where fleet_sessions.py writes"


def test_reg_dir_unset_converges_on_the_shared_resolver():
    """With FLEET_REG_DIR unset the watchdog must land wherever the SHARED resolver lands.

    It used to hard-code the clone-root ``tools/_registry`` here. That is a real dir on a
    maintainer's box and NOT the one the prober writes its ledger to, so an env-unset
    watchdog maintained a second, ledger-less registry beside the live one -- the #5390
    fork. Pinning the resolver rather than a literal path is what keeps the watchdog, the
    fleet_sessions.py writer and the Go reader on one dir no matter which rung wins here.
    """
    import fleet_regdir

    wd = _reload({"FLEET_REG_DIR": None})
    assert wd.REG_DIR == fleet_regdir.reg_dir(), \
        "an unset FLEET_REG_DIR must resolve, not fork into a second registry"


def test_claude_exe_prefers_fleet_convention(tmp_path):
    fleet_bin = str(tmp_path / "claude-fleet")
    wd = _reload({"FLEET_CLAUDE_EXE": fleet_bin, "FAK_CLAUDE_EXE": str(tmp_path / "fak")})
    assert wd.CLAUDE_EXE == fleet_bin


def test_claude_exe_falls_back_to_fak(tmp_path):
    fak_bin = str(tmp_path / "claude-fak")
    wd = _reload({"FLEET_CLAUDE_EXE": None, "FAK_CLAUDE_EXE": fak_bin})
    assert wd.CLAUDE_EXE == fak_bin


def test_probe_mode_dry_run_is_side_effect_free():
    wd = _reload({})
    # auto resolves to a real probe only on a live tick; dry-run probes nothing.
    assert wd.resolve_probe_mode("auto", live=False) == "none"
    # stale, not blocked: an idle available seat needs fresh evidence before it can
    # take a rehomed session (a passive verdict can hide a limit hit after last activity).
    assert wd.resolve_probe_mode("auto", live=True) == "stale"


def test_probe_mode_explicit_setting_is_honored():
    wd = _reload({})
    assert wd.resolve_probe_mode("all", live=False) == "all"
    assert wd.resolve_probe_mode("none", live=True) == "none"


def test_self_sid_reads_harness_session_id():
    # The self-resume guard refuses to resume the session the watchdog runs inside.
    # SELF_SID must mirror CLAUDE_CODE_SESSION_ID (the harness-set running-session id).
    wd = _reload({"CLAUDE_CODE_SESSION_ID": "28f44d89-ecda-4213-a9b0-2e9612a5cd39"})
    assert wd.SELF_SID == "28f44d89-ecda-4213-a9b0-2e9612a5cd39"


def test_self_sid_empty_outside_a_claude_session():
    # Run from cron (no harness session) -> empty, so the guard is inert and the
    # watchdog resumes normally.
    wd = _reload({"CLAUDE_CODE_SESSION_ID": None})
    assert wd.SELF_SID == ""


# ---- outcome-gated ledger (the resume-once-on-launch burn fix) -----------------

def _write_jsonl(tmp_path, name, *records):
    p = tmp_path / name
    with open(p, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(__import__("json").dumps(r) + "\n")
    return str(p)


def _asst_err(text):
    return {"type": "assistant", "isApiErrorMessage": True,
            "message": {"role": "assistant", "content": [{"type": "text", "text": text}]}}


def test_no_history_is_not_blocked():
    wd = _reload({})
    blocked, _ = wd.resume_blocked("sid-1", [])
    assert not blocked, "first resume must be allowed"


def test_recoverable_limit_outcome_stays_eligible(tmp_path, monkeypatch):
    # a resume that died on a usage-limit wall must remain auto-retry-eligible
    # (under the attempt cap) instead of being burned on launch.
    wd = _reload({"FAK_MAX_ATTEMPTS": "3"})
    p = _write_jsonl(tmp_path, "lim.jsonl",
                     {"type": "assistant", "message": {"role": "assistant",
                      "content": [{"type": "text", "text": "did work"}]}},
                     _asst_err("You've hit your session limit · resets 6am (America/Los_Angeles)"))
    monkeypatch.setattr(wd, "_newest_transcript", lambda sid: p)
    assert wd.last_resume_outcome("sid-2") == "recoverable"
    blocked, why = wd.resume_blocked("sid-2", [{"phase": "launched", "attempt": 1}])
    assert not blocked, why


def test_attempt_cap_eventually_blocks(tmp_path, monkeypatch):
    wd = _reload({"FAK_MAX_ATTEMPTS": "2"})
    p = _write_jsonl(tmp_path, "lim2.jsonl",
                     _asst_err("You've hit your session limit · resets 6am"))
    monkeypatch.setattr(wd, "_newest_transcript", lambda sid: p)
    hist = [{"phase": "launched", "attempt": 1}, {"phase": "launched", "attempt": 2}]
    blocked, why = wd.resume_blocked("sid-3", hist)
    assert blocked and "cap" in why.lower()


def test_rearm_marker_clears_attempt_cap():
    """A re-arm reclaim row (#2178) zeroes the attempt budget accrued BEFORE it, so a sid that
    burned its whole cap on a known-transient infra fault (the managed-cache-1h-TTL 400 wave)
    becomes resumable again -- while launches appended AFTER the marker re-cap normally."""
    wd = _reload({"FAK_MAX_ATTEMPTS": "8"})
    capped = [{"phase": "launched", "attempt": i + 1} for i in range(8)]
    blocked, why = wd.resume_blocked("sid-rearm", capped)
    assert blocked and "cap" in why.lower(), "8 launches must hit the cap first"
    # a rearm marker resets the budget -> resume allowed again (and the marker isn't an attempt)
    reclaimed = capped + [{"phase": "rearm", "reason": "managed-cache-1h-ttl-400 #2178"}]
    assert wd.is_resume_attempt_record(reclaimed[-1]) is False
    blocked, why = wd.resume_blocked("sid-rearm", reclaimed)
    assert not blocked, f"a rearm marker must clear the cap, got blocked: {why}"
    # fresh launches AFTER the rearm count again from 0 -> re-caps after another full budget
    recapped = reclaimed + [{"phase": "launched", "attempt": i + 1} for i in range(8)]
    blocked, why = wd.resume_blocked("sid-rearm", recapped)
    assert blocked and "cap" in why.lower(), "8 launches after a rearm must re-cap"
    # a later operator/auth settle after the rearm still wins (last write wins)
    settled = reclaimed + [{"manual_override": True, "action": "consolidate-x"}]
    blocked, why = wd.resume_blocked("sid-rearm", settled)
    assert blocked and "operator" in why.lower(), "a manual override AFTER a rearm re-blocks"


def test_auth_outcome_is_unrecoverable_and_blocks(tmp_path, monkeypatch):
    wd = _reload({})
    p = _write_jsonl(tmp_path, "auth.jsonl", _asst_err("Not logged in. Please run /login"))
    monkeypatch.setattr(wd, "_newest_transcript", lambda sid: p)
    assert wd.last_resume_outcome("sid-4") == "unrecoverable"
    blocked, _ = wd.resume_blocked("sid-4", [{"phase": "launched", "attempt": 1}])
    assert blocked, "an auth wall can't be fixed by a re-resume -> blocked"


def test_clean_progress_burns_once(tmp_path, monkeypatch):
    wd = _reload({})
    p = _write_jsonl(tmp_path, "ok.jsonl",
                     {"type": "assistant", "message": {"role": "assistant",
                      "content": [{"type": "text", "text": "Finished and shipped, nothing left."}]}})
    monkeypatch.setattr(wd, "_newest_transcript", lambda sid: p)
    assert wd.last_resume_outcome("sid-5") == "progressed"
    blocked, _ = wd.resume_blocked("sid-5", [{"phase": "launched", "attempt": 1}])
    assert blocked, "a resume that took must not be resumed again"


def test_stale_limit_banner_superseded_by_clean_finish_is_progressed(tmp_path, monkeypatch):
    # the review's finding: a limit banner EARLIER in the tail, then a clean finish,
    # must read as 'progressed' (the recovery superseded the wall) -- NOT 'recoverable'.
    # Classification reads only the TERMINAL turn, so the stale banner doesn't win.
    wd = _reload({})
    p = _write_jsonl(
        tmp_path, "superseded.jsonl",
        _asst_err("You've hit your session limit · resets 6am (America/Los_Angeles)"),
        {"type": "assistant", "message": {"role": "assistant",
         "content": [{"type": "text", "text": "Recovered after the reset and shipped cleanly."}]}},
        {"type": "mode", "mode": "default"},
        {"type": "permission-mode"})
    monkeypatch.setattr(wd, "_newest_transcript", lambda sid: p)
    assert wd.last_resume_outcome("sid-7") == "progressed", \
        "a clean turn after a limit banner supersedes it -> burn once, do not re-resume"


def test_manual_override_is_authoritative():
    wd = _reload({})
    hist = [{"action": "consolidate-resume-throttle-strand-2026-06-23", "target": ".claude-gem7"}]
    blocked, why = wd.resume_blocked("sid-6", hist)
    assert blocked and "operator" in why.lower()


def test_source_gate_defer_does_not_burn_attempt():
    wd = _reload({"FAK_MAX_ATTEMPTS": "1"})
    hist = [{"phase": "deferred", "cause": "source_concurrency_gate",
             "reason": "SOURCE_SATURATED"}]
    assert wd.is_resume_attempt_record(hist[0]) is False
    blocked, why = wd.resume_blocked("sid-defer", hist)
    assert not blocked, why


def test_powershell_watchdog_invokes_source_governor_and_spacing():
    ps1 = Path(__file__).with_name("fleet_resume_watchdog.ps1").read_text(encoding="utf-8")
    assert "resume admit --json" in ps1
    assert "SourceAdmitGate $ledgerPath $sourcePolicyPath" in ps1
    assert "phase = 'deferred'" in ps1
    assert "source_concurrency_gate" in ps1
    assert "Start-Sleep -Seconds $LaunchSpacingSec" in ps1
    # the governor-unavailable path must be loud, never silent (#2173)
    assert "gate_fail_open" in ps1
    assert "FailOpen" in ps1


def test_gate_fail_open_row_is_not_an_attempt(tmp_path):
    # The durable governor-unavailable warning row (#2173) must be invisible to retry
    # accounting: not an attempt record, never blocking, never launch pressure.
    wd = _reload({"FAK_MAX_ATTEMPTS": "1"})
    rec = wd.record_gate_fail_open(str(tmp_path / "ledger.jsonl"), "no-fak-binary")
    assert rec["phase"] in wd.NON_LAUNCH_PHASES
    assert wd.is_resume_attempt_record(rec) is False
    # even if a reader attributed the row to a session, it must not block
    blocked, why = wd.resume_blocked("sid-failopen", [rec])
    assert not blocked, why
    # and it landed durably
    line = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").strip()
    assert '"gate_fail_open"' in line and "no-fak-binary" in line


# ---- PowerShell watchdog behavioral witnesses (#2172) ---------------------------
#
# These drive the REAL fleet_resume_watchdog.ps1 in -Live mode inside a hermetic temp
# fleet: a fake `fak` (scripted exit code) and a fake `claude` (drops a marker file per
# invocation) prove what the launcher actually DID, not what its source says. Windows-
# only: the launch path uses Start-Process -WindowStyle, and the crash class under test
# (#2170/#2172) is a Windows one.

import shutil as _shutil  # noqa: E402  (section-local: behavioral PS1 harness below)
import subprocess as _subprocess  # noqa: E402

_POWERSHELL = _shutil.which("powershell") or _shutil.which("pwsh")
_ps1_behavioral = __import__("pytest").mark.skipif(
    os.name != "nt" or not _POWERSHELL,
    reason="needs Windows + PowerShell (drives the real .ps1 launch path)")


def _seed_ps1_fleet(tmp_path, *, fak_exit, fak_reason, sessions, ledger_rows=()):
    """Build the hermetic fleet layout the .ps1 runs against and return its paths."""
    import json as _json
    fleet = tmp_path / "fleet"
    reg = tmp_path / "reg"
    log = tmp_path / "log"
    cfg = tmp_path / "cfg"
    for d in (fleet, reg, log, cfg):
        d.mkdir(parents=True, exist_ok=True)

    marker = tmp_path / "claude_launches.txt"
    claude = tmp_path / "claude.cmd"
    claude.write_text('@echo off\r\necho launched>>"%s"\r\n' % marker, encoding="ascii")

    # The fake `fak` serves three roles: the `resume cap` tick sizer (#3581), the `resume admit`
    # gate (echo a decision + exit code), and, once a managed-cache posture is configured, the
    # `fak guard <posture> -- claude ...` front. The guard branch records the argv it was fronted
    # with (the behavioral witness that the posture reached the launch) and is inert for the
    # admit-only tests (%1 != "guard").
    #
    # The `resume cap` branch is REQUIRED, not optional: since #3581 the .ps1 defaults -MaxPerTick
    # to 0 and derives the tick size from `fak resume cap`. A fake that answers only `resume admit`
    # leaves .cap null -> [int]$null -> cap=0 -> "per-tick cap reached (0)" -> the watchdog launches
    # NOTHING, and every behavioral test below silently asserts against a tick that never ran. The
    # payload mirrors internal/resume.WatchdogCap and is self-consistent (2 healthy seats x seat_cap
    # 6 = 12 capacity, 4 active -> headroom 8 -> cap 8, inside floor 4 / ceiling 64) so the derived
    # cap comfortably exceeds the handful of sessions these fixtures seed.
    guard_marker = tmp_path / "guard_fronts.txt"
    fak = tmp_path / "fak.cmd"
    fak.write_text(
        '@echo off\r\n'
        'if "%1"=="guard" (\r\n'
        'echo %* >> "' + str(guard_marker) + '"\r\n'
        'exit /b 0\r\n'
        ')\r\n'
        'if "%2"=="cap" (\r\n'
        'echo {"cap":8,"floor":4,"ceiling":64,"seat_cap":6,"healthy_seats":2,"headroom":8}\r\n'
        'exit /b 0\r\n'
        ')\r\n'
        'echo {"decision":{"reason":"' + fak_reason + '"}}\r\n'
        'exit /b ' + str(fak_exit) + '\r\n',
        encoding="ascii")

    plan = {"plan": [
        {"session": sid, "account": ".claude-t", "resume_account": ".claude-t",
         "project": "C--work-fak", "cwd": None, "disp": "STOPPED_APIERR",
         "rehomed": False, "config_dir": str(cfg), "resume_config_dir": str(cfg)}
        for sid in sessions]}
    (reg / "resume_plan.json").write_text(_json.dumps(plan), encoding="utf-8")

    ledger = reg / "resume_ledger.jsonl"
    if ledger_rows:
        ledger.write_text(
            "".join(_json.dumps(r) + "\n" for r in ledger_rows), encoding="utf-8")

    return {"fleet": fleet, "reg": reg, "log": log, "marker": marker,
            "claude": claude, "fak": fak, "ledger": ledger, "guard_marker": guard_marker}


def _run_ps1_live(tmp_path, paths, *, spacing=0, max_attempts=8, env_extra=None,
                  headless=False):
    ps1 = Path(__file__).with_name("fleet_resume_watchdog.ps1")
    env = dict(os.environ)
    env["FLEET_STATE_DIR"] = str(tmp_path / "state")
    env["FLEET_CLAUDE_DISABLE_MARKER"] = str(tmp_path / "claude.disabled")
    env["FAK_RESUME_SOURCE_POLICY"] = str(tmp_path / "policy.json")  # hermetic (missing = permissive)
    env.pop("FAK_EXE", None)
    # hermetic posture: pin EXPLICIT auto (guard's own billing-gated passive) unless a test opts
    # in. Since the 2026-07-10 on-by-default flip an UNSET knob normalizes to on, which would
    # front EVERY posture-agnostic launch through `fak guard` (and the fake fak.cmd guard branch
    # never invokes claude, so the bare-launch marker would stop appearing). Pinning auto keeps
    # these tests on the direct-launch path; a posture test opts into on via FAK_MANAGED_CACHE=""
    # (unset/on-by-default) or ="on" through env_extra.
    env["FAK_MANAGED_CACHE"] = "auto"
    env.pop("FAK_GUARD_API_KEY_ENV", None)
    for k, v in (env_extra or {}).items():
        env[k] = v
    args = [_POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1),
            "-Live", "-Probe", "none",
            "-FleetDir", str(paths["fleet"]), "-RegistryDir", str(paths["reg"]),
            "-LogDir", str(paths["log"]), "-ClaudeExe", str(paths["claude"]),
            "-FakExe", str(paths["fak"]),
            "-LaunchSpacingSec", str(spacing), "-MaxAttempts", str(max_attempts)]
    if headless:
        args.append("-Headless")
    return _subprocess.run(
        args, capture_output=True, text=True, timeout=180, env=env)


def _ledger_rows(paths):
    import json as _json
    if not paths["ledger"].exists():
        return []
    return [_json.loads(ln) for ln in
            paths["ledger"].read_text(encoding="utf-8").splitlines() if ln.strip()]


@_ps1_behavioral
def test_ps1_refused_admit_never_reaches_start_process(tmp_path):
    """#2172 acceptance: force `fak resume admit` to exit 3 and prove the PowerShell
    path does NOT Start-Process — the launch is deferred with a structured ledger row."""
    sid = "11111111-2222-3333-4444-555555555555"
    paths = _seed_ps1_fleet(tmp_path, fak_exit=3, fak_reason="SOURCE_SATURATED",
                            sessions=[sid])
    r = _run_ps1_live(tmp_path, paths)
    assert r.returncode == 0, r.stdout + r.stderr
    assert not paths["marker"].exists(), \
        "claude was launched despite the source governor refusing (exit 3)"
    rows = _ledger_rows(paths)
    deferred = [x for x in rows if x.get("phase") == "deferred"]
    assert deferred and deferred[0]["cause"] == "source_concurrency_gate"
    assert deferred[0]["session"] == sid
    assert "SOURCE_SATURATED" in deferred[0]["reason"]
    assert "DEFER" in r.stdout


@_ps1_behavioral
def test_ps1_deferred_rows_do_not_trip_max_attempts(tmp_path):
    """#2172 acceptance: phase="deferred" (and gate_fail_open) ledger rows must not
    count as attempts — a session deferred by the gate stays launchable."""
    sid = "22222222-3333-4444-5555-666666666666"
    prior = [{"ts": "2026-07-01T00:00:00Z", "session": sid, "phase": "deferred",
              "cause": "source_concurrency_gate", "reason": "SOURCE_SATURATED"},
             {"ts": "2026-07-01T00:00:01Z", "phase": "gate_fail_open",
              "cause": "source_governor_unavailable", "reason": "no-fak-binary"}]
    paths = _seed_ps1_fleet(tmp_path, fak_exit=0, fak_reason="SOURCE_ADMITTED",
                            sessions=[sid], ledger_rows=prior)
    r = _run_ps1_live(tmp_path, paths, max_attempts=1)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "attempt cap" not in r.stdout, \
        "a deferred row burned the retry cap: " + r.stdout
    assert paths["marker"].exists(), "the launch should have fired (cap not consumed)"
    launched = [x for x in _ledger_rows(paths) if x.get("phase") == "launched"]
    assert launched and launched[0]["attempt"] == 1


def test_ps1_rearm_marker_reclaims_capped_session(tmp_path):
    """#2178 fix: a sid that burned its whole attempt budget on the managed-cache-1h-TTL 400 wave
    is reclaimed by a `phase=rearm` ledger row -- the .ps1 launch gate zeroes the attempts accrued
    before it, so the session resumes again (fresh attempt 1) instead of staying settled at the
    cap. Parity with the .py resume_blocked / fleet_sessions.py planner rearm handling."""
    sid = "77777777-8888-9999-aaaa-bbbbbbbbbbbb"
    prior = [{"ts": f"2026-07-09T00:00:0{i}Z", "session": sid, "phase": "launched",
              "attempt": i + 1} for i in range(8)]
    prior.append({"ts": "2026-07-10T00:00:00Z", "session": sid, "phase": "rearm",
                  "reason": "managed-cache-1h-ttl-400 #2178"})
    paths = _seed_ps1_fleet(tmp_path, fak_exit=0, fak_reason="SOURCE_ADMITTED",
                            sessions=[sid], ledger_rows=prior)
    r = _run_ps1_live(tmp_path, paths, max_attempts=8)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "attempt cap" not in r.stdout, \
        "the rearm marker must clear the cap, but the sid was still settled: " + r.stdout
    assert paths["marker"].exists(), "the reclaimed session should have resumed"
    launched = [x for x in _ledger_rows(paths)
                if x.get("phase") == "launched" and x.get("ts", "") >= "2026-07-10T00:00:00Z"]
    assert launched and launched[-1]["attempt"] == 1, \
        "the post-rearm launch must count from attempt 1, not 9"


@_ps1_behavioral
def test_ps1_live_launch_is_visible_by_default_and_records_mode(tmp_path):
    sid = "22222222-3333-4444-5555-666666666666"
    paths = _seed_ps1_fleet(tmp_path, fak_exit=0, fak_reason="SOURCE_ADMITTED",
                            sessions=[sid])
    r = _run_ps1_live(tmp_path, paths)
    assert r.returncode == 0, r.stdout + r.stderr
    launched = [x for x in _ledger_rows(paths) if x.get("phase") == "launched"]
    assert len(launched) == 1, r.stdout
    assert launched[0]["launch_mode"] == "visible"


@_ps1_behavioral
def test_ps1_headless_is_explicit_and_records_mode(tmp_path):
    sid = "22222222-3333-4444-5555-777777777777"
    paths = _seed_ps1_fleet(tmp_path, fak_exit=0, fak_reason="SOURCE_ADMITTED",
                            sessions=[sid])
    r = _run_ps1_live(tmp_path, paths, headless=True)
    assert r.returncode == 0, r.stdout + r.stderr
    launched = [x for x in _ledger_rows(paths) if x.get("phase") == "launched"]
    assert len(launched) == 1, r.stdout
    assert launched[0]["launch_mode"] == "headless"


@_ps1_behavioral
def test_ps1_failed_launch_is_durable_with_mode(tmp_path):
    sid = "22222222-3333-4444-5555-888888888888"
    paths = _seed_ps1_fleet(tmp_path, fak_exit=0, fak_reason="SOURCE_ADMITTED",
                            sessions=[sid])
    paths["claude"] = tmp_path / "missing-claude.exe"
    r = _run_ps1_live(tmp_path, paths)
    assert r.returncode == 0, r.stdout + r.stderr
    failed = [x for x in _ledger_rows(paths) if x.get("phase") == "launch_failed"]
    assert len(failed) == 1, r.stdout
    assert failed[0]["launch_mode"] == "visible"
    assert failed[0]["attempt"] == 1
    assert "Start-Process" in failed[0]["detail"]


@_ps1_behavioral
def test_ps1_launch_spacing_prevents_same_second_starts(tmp_path):
    """#2172 acceptance: with spacing configured, two live resumes cannot start in the
    same second — witnessed from the per-launch ledger timestamps."""
    from datetime import datetime as _dt
    sids = ["33333333-4444-5555-6666-777777777777",
            "44444444-5555-6666-7777-888888888888"]
    paths = _seed_ps1_fleet(tmp_path, fak_exit=0, fak_reason="SOURCE_ADMITTED",
                            sessions=sids)
    r = _run_ps1_live(tmp_path, paths, spacing=2)
    assert r.returncode == 0, r.stdout + r.stderr
    launched = [x for x in _ledger_rows(paths) if x.get("phase") == "launched"]
    assert len(launched) == 2, r.stdout
    t0, t1 = (_dt.strptime(x["ts"], "%Y-%m-%dT%H:%M:%SZ") for x in launched)
    assert t0 != t1, "two spaced launches recorded the same second (stale tick ts?)"
    assert abs((t1 - t0).total_seconds()) >= 2


@_ps1_behavioral
def test_ps1_gate_error_fails_open_loudly(tmp_path):
    """#2173: an unexpected governor exit fails OPEN (the launch still fires) but
    leaves a durable gate_fail_open warning row and a WARN log line — never silent."""
    sid = "55555555-6666-7777-8888-999999999999"
    paths = _seed_ps1_fleet(tmp_path, fak_exit=7, fak_reason="BOOM",
                            sessions=[sid])
    r = _run_ps1_live(tmp_path, paths)
    assert r.returncode == 0, r.stdout + r.stderr
    assert paths["marker"].exists(), "fail-open must not strand recovery"
    rows = _ledger_rows(paths)
    warn = [x for x in rows if x.get("phase") == "gate_fail_open"]
    assert warn and warn[0]["cause"] == "source_governor_unavailable"
    assert "session" not in warn[0], "warning row must be session-less (invisible to retry accounting)"
    assert [x for x in rows if x.get("phase") == "launched"]
    assert "WARN source governor UNAVAILABLE" in r.stdout


@_ps1_behavioral
def test_ps1_fronts_resume_with_guard_when_posture_configured(tmp_path):
    """#2178 acceptance: with FAK_MANAGED_CACHE=on + FAK_GUARD_API_KEY_ENV set, a live resume
    is fronted through `fak guard <posture> -- claude --resume ...` (witnessed by the guard
    front marker), so the resumed child carries the posture from its OWN guard invocation."""
    sid = "66666666-7777-8888-9999-aaaaaaaaaaaa"
    paths = _seed_ps1_fleet(tmp_path, fak_exit=0, fak_reason="SOURCE_ADMITTED",
                            sessions=[sid])
    r = _run_ps1_live(tmp_path, paths, env_extra={
        "FAK_MANAGED_CACHE": "on", "FAK_GUARD_API_KEY_ENV": "ANTHROPIC_API_KEY"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert paths["guard_marker"].exists(), \
        "posture configured but the resume was NOT fronted with `fak guard`: " + r.stdout
    fronted = paths["guard_marker"].read_text(encoding="ascii", errors="replace")
    # the posture flags reached the guard front, in the Go launchers' stable order
    assert "--api-key-env ANTHROPIC_API_KEY" in fronted
    assert "--managed-cache on" in fronted
    # and the child agent (after `--`) is still the claude --resume for this sid
    assert "-- " in fronted and "--resume " in fronted and sid in fronted
    # the launch was recorded once (guard-fronting doesn't disturb the ledger bookkeeping)
    launched = [x for x in _ledger_rows(paths) if x.get("phase") == "launched"]
    assert launched and launched[0]["session"] == sid
    # the operator sees the posture decision in the tick log
    assert "managed-cache posture" in r.stdout


@_ps1_behavioral
def test_ps1_explicit_auto_never_fronts_with_guard(tmp_path):
    """FAK_MANAGED_CACHE=auto stays byte-identical: claude is launched DIRECTLY, guard is never
    fronted — the express opt-out of on-by-default keeps guard's own billing-gated auto (#2178)."""
    sid = "77777777-8888-9999-aaaa-bbbbbbbbbbbb"
    paths = _seed_ps1_fleet(tmp_path, fak_exit=0, fak_reason="SOURCE_ADMITTED",
                            sessions=[sid])
    r = _run_ps1_live(tmp_path, paths, env_extra={"FAK_MANAGED_CACHE": "auto"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert not paths["guard_marker"].exists(), "explicit auto must not front with guard"
    assert paths["marker"].exists(), "claude should have launched directly"
    assert "managed-cache posture" not in r.stdout


@_ps1_behavioral
def test_ps1_unset_posture_subscription_stays_passive(tmp_path):
    """Subscription-safe UNSET default (2026-07-10, supersedes the blind unset=>on). With no
    FAK_GUARD_API_KEY_ENV configured (a subscription-OAuth seat), an UNSET FAK_MANAGED_CACHE now
    defaults to `auto` -- a bare `claude --resume` with NO forced `--managed-cache on`. Forcing on
    would activate the stable-prefix 1h-TTL upgrade, which the subscription wire rejects as
    `400 upstream rejected the request as malformed` (proven by clean-env ablation; see
    docs/notes/MANAGED-CACHE-1H-TTL-400-FIX-2026-07-09.md), 400-crashing every subscription
    resume. The 1h upgrade only pays off on an API-key seat, so an unconfigured (subscription)
    fleet must resume passive."""
    sid = "88888888-9999-aaaa-bbbb-cccccccccccc"
    paths = _seed_ps1_fleet(tmp_path, fak_exit=0, fak_reason="SOURCE_ADMITTED",
                            sessions=[sid])
    # empty string == unset after the .ps1's "$env:FAK_MANAGED_CACHE" coercion; no api-key env
    r = _run_ps1_live(tmp_path, paths, env_extra={"FAK_MANAGED_CACHE": ""})
    assert r.returncode == 0, r.stdout + r.stderr
    # unset + no api-key env -> auto -> bare launch (NOT fronted with --managed-cache on)
    assert paths["marker"].exists(), \
        "subscription default must resume bare/passive, not fronted with managed-cache on: " + r.stdout
    assert not paths["guard_marker"].exists(), \
        "unset+no-api-key must NOT force a managed-cache posture on a subscription seat: " + r.stdout
    assert "managed-cache posture ->" not in r.stdout


def test_ps1_unset_posture_apikey_fronts_managed_cache_on(tmp_path):
    """The other half of the billing-aware unset default: when an API-key seat IS configured
    (FAK_GUARD_API_KEY_ENV set), an UNSET FAK_MANAGED_CACHE still fronts the resume through
    `fak guard --api-key-env X --managed-cache on -- claude --resume ...`. Managed cache stays
    best-effort-everywhere where it actually helps and is well-formed (the 1h-TTL upgrade is only
    accepted on API-key billing), so the fix narrows #2178 to correct seats, it does not revert it."""
    sid = "88888888-9999-aaaa-bbbb-cccccccccccc"
    paths = _seed_ps1_fleet(tmp_path, fak_exit=0, fak_reason="SOURCE_ADMITTED",
                            sessions=[sid])
    r = _run_ps1_live(tmp_path, paths, env_extra={
        "FAK_MANAGED_CACHE": "", "FAK_GUARD_API_KEY_ENV": "ANTHROPIC_API_KEY"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert paths["guard_marker"].exists(), \
        "api-key seat: an unset knob must still front the resume with guard: " + r.stdout
    fronted = paths["guard_marker"].read_text(encoding="ascii", errors="replace")
    assert "--managed-cache on" in fronted
    assert "--api-key-env ANTHROPIC_API_KEY" in fronted
    # the child agent (after `--`) is still the claude --resume for this sid
    assert "-- " in fronted and "--resume " in fronted and sid in fronted
    # the fronted path runs claude THROUGH guard, so the bare-launch marker is not written
    assert not paths["marker"].exists()
    assert "managed-cache posture" in r.stdout


def test_ps1_parses_clean():
    """#2172 acceptance: the .ps1 stays syntactically valid — parsed by the real
    PowerShell language parser, not a regex."""
    import pytest
    if not _POWERSHELL:
        pytest.skip("needs pwsh or powershell for the language parser")
    ps1 = Path(__file__).with_name("fleet_resume_watchdog.ps1")
    script = (
        "$t=$null;$e=$null;"
        "[System.Management.Automation.Language.Parser]::ParseFile('%s',[ref]$t,[ref]$e)|Out-Null;"
        "if($e.Count){$e|ForEach-Object{$_.Message};exit 1};exit 0" % str(ps1).replace("'", "''"))
    r = _subprocess.run([_POWERSHELL, "-NoProfile", "-Command", script],
                        capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, "parse errors:\n" + r.stdout + r.stderr


def _setenv(**kv):
    """Set env vars for the duration of a test, returning a restore callable. Used for the
    memory-cotravel gate, which memory_cotravel reads at CALL time (so it must be live when
    rehome_transcript runs -- _reload restores env before returning, which would unset it)."""
    saved = {k: os.environ.get(k) for k in kv}
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

    def restore():
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return restore


def test_rehome_co_travels_memory_when_live(tmp_path):
    """End-to-end: rehome_transcript carries the slug-scoped memory store, not just the
    transcript. Forced to the `live` gate so the copy is observable; the additive
    strategy keeps a conflicting dest memory. Proves the switcher<->memory seam is wired
    through the ONE canonical copy primitive both resume paths call."""
    wd = _reload({})
    restore = _setenv(FAK_MEMORY_COTRAVEL="live", FAK_MEMORY_MERGE="additive",
                      FAK_MEMORY_COTRAVEL_LEDGER=str(tmp_path / "led.jsonl"))
    try:
        sid = "11112222-3333-4444-5555-666677778888"
        slug = "C--work-fak"
        src_cfg = str(tmp_path / ".claude-A")
        dst_cfg = str(tmp_path / ".claude-B")
        # source: a transcript + a memory note under the owner slug
        src_proj = os.path.join(src_cfg, "projects", slug)
        src_mem = os.path.join(src_proj, "memory")
        os.makedirs(src_mem, exist_ok=True)
        with open(os.path.join(src_proj, sid + ".jsonl"), "w", encoding="utf-8") as f:
            f.write('{"type":"user"}\n')
        with open(os.path.join(src_mem, "learned.md"), "w", encoding="utf-8") as f:
            f.write("hard-won fleet fact")

        assert wd.rehome_transcript(src_cfg, dst_cfg, slug, sid) is True

        dst_mem = os.path.join(dst_cfg, "projects", slug, "memory", "learned.md")
        assert os.path.isfile(dst_mem), "memory must co-travel with the transcript on re-home"
        assert open(dst_mem, encoding="utf-8").read() == "hard-won fleet fact"
    finally:
        restore()


def test_rehome_default_shadow_does_not_copy_memory(tmp_path):
    """With the default (shadow) gate, the transcript still re-homes but memory is only
    OBSERVED -- copied nowhere. Confirms the feature ships safe-by-default."""
    wd = _reload({})
    restore = _setenv(FAK_MEMORY_COTRAVEL=None,  # unset -> default shadow
                      FAK_MEMORY_COTRAVEL_LEDGER=str(tmp_path / "led.jsonl"))
    try:
        sid = "aaaa1111-2222-3333-4444-555566667777"
        slug = "C--work-fak"
        src_cfg = str(tmp_path / ".claude-A")
        dst_cfg = str(tmp_path / ".claude-B")
        src_proj = os.path.join(src_cfg, "projects", slug)
        src_mem = os.path.join(src_proj, "memory")
        os.makedirs(src_mem, exist_ok=True)
        with open(os.path.join(src_proj, sid + ".jsonl"), "w", encoding="utf-8") as f:
            f.write('{"type":"user"}\n')
        with open(os.path.join(src_mem, "learned.md"), "w", encoding="utf-8") as f:
            f.write("fact")

        assert wd.rehome_transcript(src_cfg, dst_cfg, slug, sid) is True
        # transcript landed, memory did NOT (shadow observes only)
        assert os.path.isfile(os.path.join(dst_cfg, "projects", slug, sid + ".jsonl"))
        assert not os.path.isfile(
            os.path.join(dst_cfg, "projects", slug, "memory", "learned.md"))
    finally:
        restore()


# ---- Slack event posting (--slack / FAK_DISPATCH_SLACK) ------------------------

def test_post_slack_event_disabled_is_noop():
    wd = _reload({})
    calls = []
    out = wd.post_slack_event("Resumed", "sid", "info", enabled=False,
                              transport=lambda *a: calls.append(1))
    assert out is None
    assert calls == []


def test_post_slack_event_posts_when_enabled(monkeypatch):
    import json as _json
    wd = _reload({})
    # env beats the repo .env.slack.local, so the resolution is hermetic.
    for k in ("FAK_DISPATCH_TOKEN", "FAK_DISPATCH_CHANNEL", "FAK_SCOREBOARD_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("FAK_SCOREBOARD_TOKEN", "xoxb-test")
    monkeypatch.setenv("FAK_DISPATCH_CHANNEL", "C0RES")
    calls = []

    def transport(url, body, headers, timeout):
        calls.append(_json.loads(body.decode("utf-8")))
        return 200, _json.dumps({"ok": True, "ts": "1.1", "channel": "C0RES"})

    out = wd.post_slack_event("Resumed dead session", "abcd1234 (acct/proj)", "info",
                              enabled=True, transport=transport)
    assert out["posted"] is True
    assert "Resumed dead session" in calls[0]["text"]
    # an auth wall uses the warn glyph, a resume the ♻️ glyph
    assert calls[0]["text"].startswith("♻️")


def test_toast_routes_to_slack_when_module_flag_set(monkeypatch, tmp_path):
    wd = _reload({})
    monkeypatch.setattr(wd, "SLACK", True)
    monkeypatch.setattr(wd, "SLACK_DRY", False)
    monkeypatch.setattr(wd, "LOG_DIR", str(tmp_path))
    for k in ("FAK_DISPATCH_TOKEN", "FAK_DISPATCH_CHANNEL", "FAK_SCOREBOARD_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("FAK_SCOREBOARD_TOKEN", "xoxb-test")
    monkeypatch.setenv("FAK_DISPATCH_CHANNEL", "C0RES")
    posted = {}
    import slack_post

    def fake_event(title, detail="", *, level="info", **kw):
        posted["title"] = title
        posted["level"] = level
        return {"posted": True}

    monkeypatch.setattr(slack_post, "event", fake_event)
    wd.toast("Account needs re-login", "alpha : run /login", "warn")
    assert posted["title"] == "Account needs re-login"
    assert posted["level"] == "warn"


# ---- managed-cache posture (#2178 parity for the resume wave) -------------------
#
# These pin the resume watchdog's managed_cache_posture_args to the SAME shaping the Go
# launchers' guardCachePostureArgs enforces (cmd/fak/guard_cache_posture_test.go), and the
# resume_child_argv fronting decision: opted-in + fak present => `fak guard <posture> --`,
# otherwise the bare `claude --resume` that shipped before this knob. The helpers read the
# env live at call time, so monkeypatch.setenv suffices (no import-time reload needed).


def _posture(monkeypatch, *, mode=None, api_key_env=None):
    wd = _reload({})
    for k in (wd.FLEET_MANAGED_CACHE_ENV, wd.FLEET_GUARD_API_KEY_ENV_ENV):
        monkeypatch.delenv(k, raising=False)
    if mode is not None:
        monkeypatch.setenv(wd.FLEET_MANAGED_CACHE_ENV, mode)
    if api_key_env is not None:
        monkeypatch.setenv(wd.FLEET_GUARD_API_KEY_ENV_ENV, api_key_env)
    return wd


def test_posture_explicit_auto_no_key_emits_nothing(monkeypatch):
    # EXPLICIT auto + no api-key-env emits NOTHING, so the resume argv stays byte-identical to
    # the bare `claude --resume` and guard keeps its own billing-gated auto.
    wd = _posture(monkeypatch, mode="auto")
    assert wd.managed_cache_posture_args() == ([], None)
    wd = _posture(monkeypatch, mode="  AUTO  ")  # case/whitespace-insensitive
    assert wd.managed_cache_posture_args() == ([], None)


def test_posture_unset_defaults_to_on(monkeypatch):
    # On-by-default (2026-07-10): an UNSET knob normalizes to on (mirrors the Go
    # normalizeManagedCacheMode), so the unconfigured fleet fronts the resume with
    # --managed-cache on rather than staying byte-identical to the bare launch.
    wd = _posture(monkeypatch)
    assert wd.managed_cache_posture_args() == (["--managed-cache", "on"], None)


def test_posture_api_key_alone_with_explicit_auto_activates(monkeypatch):
    # api-key-env + EXPLICIT auto makes guard's AUTO resolve ACTIVE on the Anthropic wire without
    # forcing the mode -- so it emits --api-key-env but no --managed-cache. (An UNSET knob would
    # additionally carry --managed-cache on; this pins the auto-activation path specifically.)
    wd = _posture(monkeypatch, mode="auto", api_key_env="ANTHROPIC_API_KEY")
    assert wd.managed_cache_posture_args() == (["--api-key-env", "ANTHROPIC_API_KEY"], None)


def test_posture_on_and_off_are_emitted_explicitly(monkeypatch):
    wd = _posture(monkeypatch, mode="on")
    assert wd.managed_cache_posture_args() == (["--managed-cache", "on"], None)
    wd = _posture(monkeypatch, mode="OFF")  # normalized lower
    assert wd.managed_cache_posture_args() == (["--managed-cache", "off"], None)


def test_posture_key_and_mode_stable_order(monkeypatch):
    # Both knobs together, stable order: --api-key-env then --managed-cache (matches Go).
    wd = _posture(monkeypatch, mode="on", api_key_env="ANTHROPIC_API_KEY")
    args, warn = wd.managed_cache_posture_args()
    assert args == ["--api-key-env", "ANTHROPIC_API_KEY", "--managed-cache", "on"]
    assert warn is None


def test_posture_malformed_warns_not_raises(monkeypatch):
    # A headless worker must warn-and-continue passive on a bad mode, never raise (which would
    # strand the WHOLE resume wave). No flags, and a warning naming the offending token.
    wd = _posture(monkeypatch, mode="active")
    args, warn = wd.managed_cache_posture_args()
    assert args == []
    assert warn and "active" in warn and "auto|on|off" in warn


def test_resume_child_argv_empty_posture_is_bare_claude(monkeypatch):
    # Empty posture list (explicit auto, or the fak-missing fallback) -> the exact pre-#2188
    # argv, never fronted with guard. (Since on-by-default an UNSET knob yields a non-empty
    # posture; this pins the empty-list branch that still reaches the bare launch.)
    wd = _reload({})
    monkeypatch.setattr(wd, "CLAUDE_EXE", "/bin/claude")
    monkeypatch.setattr(wd, "FAK_EXE", "/bin/fak")  # present, but posture empty => no guard
    argv = wd.resume_child_argv("SID", [])
    assert argv == ["/bin/claude", "--resume", "SID", "-p", wd.RESUME_PROMPT,
                    "--dangerously-skip-permissions"]
    assert "guard" not in argv


def test_resume_child_argv_fronts_with_guard_when_opted_in(monkeypatch):
    # Opted-in + fak resolvable -> `fak guard <posture> -- claude --resume ...`, posture BEFORE
    # `--` so guard parses it and the agent (after `--`) never sees it.
    wd = _reload({})
    monkeypatch.setattr(wd, "CLAUDE_EXE", "/bin/claude")
    monkeypatch.setattr(wd, "FAK_EXE", "/bin/fak")
    posture = ["--api-key-env", "ANTHROPIC_API_KEY", "--managed-cache", "on"]
    argv = wd.resume_child_argv("SID", posture)
    assert argv == ["/bin/fak", "guard", "--api-key-env", "ANTHROPIC_API_KEY",
                    "--managed-cache", "on", "--", "/bin/claude", "--resume", "SID",
                    "-p", wd.RESUME_PROMPT, "--dangerously-skip-permissions"]
    # posture is strictly before the `--` separator; the claude flags are strictly after.
    sep = argv.index("--")
    assert "--managed-cache" in argv[:sep] and "--resume" in argv[sep:]


def test_resume_child_argv_falls_back_when_fak_missing(monkeypatch):
    # Opted-in but fak not on PATH -> cannot front; fall back to the bare direct launch
    # (the caller warns; the argv must not reference a None fak binary).
    wd = _reload({})
    monkeypatch.setattr(wd, "CLAUDE_EXE", "/bin/claude")
    monkeypatch.setattr(wd, "FAK_EXE", None)
    argv = wd.resume_child_argv("SID", ["--managed-cache", "on"])
    assert argv == ["/bin/claude", "--resume", "SID", "-p", wd.RESUME_PROMPT,
                    "--dangerously-skip-permissions"]
    assert "guard" not in argv and None not in argv


def test_powershell_watchdog_fronts_posture_with_guard():
    # #2178 parity: the .ps1 (the Windows scheduled-task launcher) must carry the SAME
    # managed-cache override surface as the .py -- reading the same two env knobs, fronting
    # opted-in resumes with `fak guard <posture> --`, and warning-not-throwing on a bad mode.
    ps1 = Path(__file__).with_name("fleet_resume_watchdog.ps1").read_text(encoding="utf-8")
    assert "Get-ManagedCachePosture" in ps1
    assert "FAK_MANAGED_CACHE" in ps1 and "FAK_GUARD_API_KEY_ENV" in ps1
    assert "--managed-cache" in ps1 and "--api-key-env" in ps1
    # opted-in launch fronts with guard, posture BEFORE the `--` separator
    assert "@('guard') + $resumePostureArgs + @('--', $ClaudeExe)" in ps1
    # gated on fak being resolvable, else a direct launch (never a None-binary front)
    assert "$resumePostureArgs.Count -gt 0 -and $FakExe" in ps1
    # a malformed mode warns and resumes passive rather than stranding the wave
    assert "unknown managed-cache mode (auto|on|off)" in ps1


def test_py_and_ps1_name_the_same_posture_env_knobs():
    # The .py and .ps1 launchers MUST name the identical env knobs so an operator configures the
    # posture once and it applies whichever watchdog the box runs (#2178's single-policy intent).
    wd = _reload({})
    ps1 = Path(__file__).with_name("fleet_resume_watchdog.ps1").read_text(encoding="utf-8")
    assert wd.FLEET_MANAGED_CACHE_ENV == "FAK_MANAGED_CACHE"
    assert wd.FLEET_GUARD_API_KEY_ENV_ENV == "FAK_GUARD_API_KEY_ENV"
    assert wd.FLEET_MANAGED_CACHE_ENV in ps1 and wd.FLEET_GUARD_API_KEY_ENV_ENV in ps1


# ---- bounded artifacts (#3497): retention at the write sites --------------------
#
# The watchdog runs unattended under cron (~288 ticks/day), so every artifact it
# appends must carry an inline bound: per-resume log pairs expire at the launch
# site, the tick/notification logs rotate at a size cap, and the resume-once
# ledger compacts past its window (which also bounds the per-tick re-parse).

import json as _jsonmod  # noqa: E402  (section-local: bounded-artifact tests below)
import time as _time  # noqa: E402
from datetime import datetime as _dtc, timezone as _tzc  # noqa: E402


def _aged(path, days):
    t = _time.time() - days * 86400
    os.utime(path, (t, t))


def _ts_days_ago(days):
    return _dtc.fromtimestamp(_time.time() - days * 86400, tz=_tzc.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def test_prune_resume_logs_removes_only_expired_pairs(tmp_path):
    wd = _reload({})
    old_log = tmp_path / "resume-aaaa1111-1700000000.log"
    old_err = tmp_path / "resume-aaaa1111-1700000000.log.err"
    fresh = tmp_path / "resume-bbbb2222-1800000000.log"
    tick_log = tmp_path / "resume_watchdog.log"  # wrong shape: rotates, never pruned
    for p in (old_log, old_err, fresh, tick_log):
        p.write_text("x", encoding="utf-8")
    _aged(old_log, 20)
    _aged(old_err, 20)
    _aged(tick_log, 20)
    removed = wd.prune_resume_logs(str(tmp_path), retain_days=14)
    assert removed == 2
    assert not old_log.exists() and not old_err.exists(), "expired pair must be pruned"
    assert fresh.exists(), "a pair inside the window must survive"
    assert tick_log.exists(), "the tick log is not a per-resume pair -- never pruned"


def test_prune_resume_logs_zero_retention_disables(tmp_path):
    wd = _reload({})
    p = tmp_path / "resume-cccc3333-1700000000.log"
    p.write_text("x", encoding="utf-8")
    _aged(p, 365)
    assert wd.prune_resume_logs(str(tmp_path), retain_days=0) == 0
    assert p.exists()


def test_rotate_log_caps_at_write_site(tmp_path):
    wd = _reload({})
    p = tmp_path / "resume_watchdog.log"
    p.write_bytes(b"x" * 2048)
    assert wd._rotate_log(str(p), max_bytes=1024) is True
    assert not p.exists() and (tmp_path / "resume_watchdog.log.1").exists()
    p.write_bytes(b"y" * 10)
    assert wd._rotate_log(str(p), max_bytes=1024) is False, "under the cap: untouched"
    assert p.exists()
    # a second rotation replaces the kept generation -> on-disk total stays ~2x cap
    p.write_bytes(b"z" * 2048)
    assert wd._rotate_log(str(p), max_bytes=1024) is True
    assert (tmp_path / "resume_watchdog.log.1").read_bytes()[:1] == b"z"


def test_note_rotates_tick_log_when_capped(tmp_path, monkeypatch):
    wd = _reload({"FAK_WATCHDOG_LOG_MAX_BYTES": "128"})
    monkeypatch.setattr(wd, "LOG_DIR", str(tmp_path))
    for i in range(12):
        wd.note(f"tick filler line {i} xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    log = tmp_path / "resume_watchdog.log"
    assert (tmp_path / "resume_watchdog.log.1").exists(), "cap passed -> rotated"
    assert log.stat().st_size <= 128 + 100, "current log restarts near-empty after rotation"


def test_compact_ledger_drops_expired_keeps_recent_overrides_and_unparsable(tmp_path):
    wd = _reload({})
    ledger = tmp_path / "resume_ledger.jsonl"
    rows = [
        {"ts": _ts_days_ago(40), "session": "old-dead", "phase": "launched"},
        {"ts": _ts_days_ago(40), "session": "settled",
         "action": "consolidate-resume-throttle-strand-2026-06-01"},
        {"ts": _ts_days_ago(1), "session": "fresh", "phase": "launched"},
        {"session": "no-ts", "phase": "launched"},
    ]
    body = "".join(_jsonmod.dumps(r) + "\n" for r in rows) + "not-json garbage line\n"
    ledger.write_text(body, encoding="utf-8")
    dropped = wd.compact_ledger(str(ledger), retain_days=30, compact_bytes=1)
    assert dropped == 1, "only the expired non-override row is dropped"
    kept = ledger.read_text(encoding="utf-8")
    assert "old-dead" not in kept
    assert "settled" in kept, "operator-settled rows are authoritative forever"
    assert "fresh" in kept
    assert "no-ts" in kept and "not-json garbage line" in kept, \
        "rows the compactor cannot date are kept, never guessed away"


def test_compact_ledger_below_threshold_is_untouched(tmp_path):
    wd = _reload({})
    ledger = tmp_path / "resume_ledger.jsonl"
    body = _jsonmod.dumps({"ts": _ts_days_ago(400), "session": "ancient",
                           "phase": "launched"}) + "\n"
    ledger.write_text(body, encoding="utf-8")
    assert wd.compact_ledger(str(ledger), retain_days=30,
                             compact_bytes=1024 * 1024) == 0
    assert ledger.read_text(encoding="utf-8") == body, \
        "under the size threshold the common tick pays one getsize(), no rewrite"


def test_compact_ledger_missing_file_is_noop(tmp_path):
    wd = _reload({})
    assert wd.compact_ledger(str(tmp_path / "absent.jsonl"), compact_bytes=1) == 0


def test_compact_ledger_bounds_do_not_break_once_gate(tmp_path):
    # End-to-end over the boundary: after compaction the surviving history still
    # blocks a resumed-once session and still honors an operator settle.
    wd = _reload({})
    ledger = tmp_path / "resume_ledger.jsonl"
    rows = [
        {"ts": _ts_days_ago(40), "session": "sid-old", "phase": "launched", "attempt": 1},
        {"ts": _ts_days_ago(40), "session": "sid-settled",
         "action": "consolidate-resume-throttle-strand-2026-06-01"},
    ]
    ledger.write_text("".join(_jsonmod.dumps(r) + "\n" for r in rows), encoding="utf-8")
    wd.compact_ledger(str(ledger), retain_days=30, compact_bytes=1)
    history = {}
    for ln in ledger.read_text(encoding="utf-8").splitlines():
        rec = _jsonmod.loads(ln)
        history.setdefault(rec["session"], []).append(rec)
    assert "sid-old" not in history, "expired attempt rows compact away"
    blocked, why = wd.resume_blocked("sid-settled", history["sid-settled"])
    assert blocked and "operator" in why.lower(), \
        "an operator settle survives compaction and still gates"


# ---- continuous-drain of the resume backlog (#3587) ----------------------------
#
# Decouple resume LATENCY from the ~5-min cron. On a LIVE tick with FAK_DRAIN_CONTINUOUS=1 a
# single tick drains the AUTO_RESUME plan PAST FAK_MAX_PER_TICK -- the source governor
# (`fak resume admit`) + LAUNCH_SPACING_SEC become the only rate limiter, so recovery latency
# collapses toward the governor's spacing floor instead of the tick period. Two safety rails keep
# it from becoming a storm: a governor DEFER ENDS the drain for the tick (no launch onto capped
# seats), and a fail-open governor reverts to the tick-quantized cap. Default OFF is byte-identical
# to the pre-#3587 tick-quantized behavior (gen/next: gated until dogfooded).


def test_tick_launch_cap_default_is_tick_quantized():
    wd = _reload({"FAK_MAX_PER_TICK": "4", "FAK_DRAIN_CONTINUOUS": None})
    assert wd.tick_launch_cap(live=True) == 4
    assert wd.tick_launch_cap(live=False) == 4


def test_tick_launch_cap_continuous_live_lifts_to_backstop():
    wd = _reload({"FAK_MAX_PER_TICK": "4", "FAK_DRAIN_CONTINUOUS": "1", "FAK_DRAIN_MAX": "500"})
    # a LIVE tick lifts the per-tick COUNT to the backstop so the governor + spacing bound the rate
    assert wd.tick_launch_cap(live=True) == 500
    # dry-run stays tick-quantized (side-effect-free; the drain isn't exercised without a live launch)
    assert wd.tick_launch_cap(live=False) == 4


def test_tick_launch_cap_backstop_floors_at_max_per_tick():
    # a misconfigured tiny backstop can never make continuous-drain resume FEWER than the baseline
    wd = _reload({"FAK_MAX_PER_TICK": "6", "FAK_DRAIN_CONTINUOUS": "1", "FAK_DRAIN_MAX": "2"})
    assert wd.tick_launch_cap(live=True) == 6


def _drive_tick(tmp_path, monkeypatch, *, backlog, headroom, env):
    """Drive the REAL main() launch loop against a fixture plan of `backlog` dead sessions and a
    source governor with `headroom` free seats. Returns (launched_sids, sleeps, ledger_rows).

    Everything with a side effect is faked so this stays pure stdlib (no spawn, no network): the
    registry refresh + accounts json (subprocess.run), the resume spawn (subprocess.Popen), the
    source governor (source_admit_gate), and the inter-launch spacing (time.sleep). The governor
    admits while fewer than `headroom` resumes are live this tick, then DEFERs -- the per-seat rate
    limiter the issue keeps as the real bound."""
    import json as _json
    reg = tmp_path / "reg"
    log = tmp_path / "log"
    cfg = tmp_path / "cfg"
    for d in (reg, log, cfg):
        d.mkdir(parents=True, exist_ok=True)
    plan = {"plan": [
        {"session": f"{i:08d}-2222-3333-4444-555555555555", "account": ".claude-t",
         "resume_account": ".claude-t", "project": "C--work-fak", "cwd": None,
         "disp": "STOPPED_APIERR", "rehomed": False,
         "config_dir": str(cfg), "resume_config_dir": str(cfg)}
        for i in range(backlog)]}
    (reg / "resume_plan.json").write_text(_json.dumps(plan), encoding="utf-8")

    base_env = {"FAK_LIVE": "1", "FLEET_REG_DIR": str(reg), "FAK_WATCHDOG_LOG_DIR": str(log),
                "FAK_MAX_PER_TICK": "4", "FAK_LAUNCH_SPACING_SEC": "5", "FAK_PROBE": "none",
                "CLAUDE_CODE_SESSION_ID": None}
    base_env.update(env)
    wd = _reload(base_env)

    launches: list[list] = []
    sleeps: list[float] = []

    class _Proc:
        def __init__(self, pid):
            self.pid = pid

    def fake_popen(argv, **kw):
        launches.append(list(argv))
        return _Proc(9000 + len(launches))

    class _R:
        stdout = ""
        stderr = ""
        returncode = 0

    def fake_run(*a, **k):
        return _R()

    def fake_gate():
        # admit while there is still free headroom this tick; DEFER once saturated (host-wide)
        if len(launches) < headroom:
            return True, "admitted"
        return False, "SOURCE_SATURATED"

    monkeypatch.setattr(wd.subprocess, "run", fake_run)
    monkeypatch.setattr(wd.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(wd.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(wd, "source_admit_gate", fake_gate)

    assert wd.main() == 0
    rows = []
    ledger = reg / "resume_ledger.jsonl"
    if ledger.exists():
        rows = [_json.loads(x) for x in ledger.read_text(encoding="utf-8").splitlines() if x.strip()]
    launched_sids = [a[a.index("--resume") + 1] for a in launches if "--resume" in a]
    return launched_sids, sleeps, rows


def test_continuous_drains_full_backlog_in_one_tick(tmp_path, monkeypatch):
    # Acceptance #1: backlog B, headroom for B -> a single LIVE tick drains ALL B (past
    # FAK_MAX_PER_TICK=4), with the source governor gating each launch and spacing enforced
    # BETWEEN launches (but not after the last -- no trailing dead time).
    launched, sleeps, rows = _drive_tick(
        tmp_path, monkeypatch, backlog=10, headroom=10,
        env={"FAK_DRAIN_CONTINUOUS": "1", "FAK_DRAIN_MAX": "500"})
    assert len(launched) == 10, launched
    assert len(set(launched)) == 10, "each dead session resumed exactly once"
    assert sleeps == [5.0] * 9, "spacing honored between launches, skipped after the final one"
    assert len([r for r in rows if r.get("phase") == "launched"]) == 10


def test_baseline_stops_at_max_per_tick(tmp_path, monkeypatch):
    # The tick-quantized baseline continuous-drain beats: flag OFF, SAME backlog + headroom, one
    # tick stops at FAK_MAX_PER_TICK=4 -- the tail is stranded until the next cron tick (the p50
    # death->launch LATENCY the issue removes). ceil(B/cap) ticks to drain vs 1 for continuous.
    import math
    launched, _sleeps, _rows = _drive_tick(
        tmp_path, monkeypatch, backlog=10, headroom=10, env={"FAK_DRAIN_CONTINUOUS": None})
    assert len(launched) == 4, launched
    assert math.ceil(10 / 4) > 1, "baseline quantizes the backlog tail across multiple cron ticks"


def test_continuous_zero_headroom_does_nothing_no_storm(tmp_path, monkeypatch):
    # Acceptance #2: with 0 free headroom the drain launches NOTHING (no storm onto capped seats);
    # exactly one governor DEFER is recorded and the tick ends -- not a deferred-row storm.
    launched, sleeps, rows = _drive_tick(
        tmp_path, monkeypatch, backlog=8, headroom=0,
        env={"FAK_DRAIN_CONTINUOUS": "1", "FAK_DRAIN_MAX": "500"})
    assert launched == []
    assert sleeps == []
    deferred = [r for r in rows if r.get("phase") == "deferred"]
    assert len(deferred) == 1, "one DEFER then the drain ends -- not one row per remaining entry"
    assert deferred[0]["cause"] == "source_concurrency_gate"


def test_continuous_stops_when_governor_defers_midway(tmp_path, monkeypatch):
    # The source governor stays the real rate limiter: headroom for only K < B -> the tick launches
    # exactly K, then the governor DEFERs and the drain ends (never launches past the spacing floor
    # onto saturated seats, and never emits a deferred row per remaining plan entry).
    launched, _sleeps, rows = _drive_tick(
        tmp_path, monkeypatch, backlog=12, headroom=5,
        env={"FAK_DRAIN_CONTINUOUS": "1", "FAK_DRAIN_MAX": "500"})
    assert len(launched) == 5, launched
    assert len([r for r in rows if r.get("phase") == "deferred"]) == 1, \
        "the drain ends on the first DEFER, not once per remaining entry"


def test_continuous_reverts_to_tick_cap_when_governor_fails_open(tmp_path, monkeypatch):
    # Safety rail: a FAIL-OPEN governor (admit granted WITHOUT a verdict) cannot bound a storm, so
    # continuous-drain must revert to the tick-quantized FAK_MAX_PER_TICK -- never drain a deep
    # backlog with no enforcing rate limiter. Modeled by a gate that admits with a non-'admitted'
    # reason (the fail-open signature).
    import json as _json
    reg = tmp_path / "reg"
    log = tmp_path / "log"
    cfg = tmp_path / "cfg"
    for d in (reg, log, cfg):
        d.mkdir(parents=True, exist_ok=True)
    plan = {"plan": [
        {"session": f"{i:08d}-2222-3333-4444-555555555555", "account": ".claude-t",
         "resume_account": ".claude-t", "project": "C--work-fak", "cwd": None,
         "disp": "STOPPED_APIERR", "rehomed": False,
         "config_dir": str(cfg), "resume_config_dir": str(cfg)}
        for i in range(10)]}
    (reg / "resume_plan.json").write_text(_json.dumps(plan), encoding="utf-8")
    wd = _reload({"FAK_LIVE": "1", "FLEET_REG_DIR": str(reg), "FAK_WATCHDOG_LOG_DIR": str(log),
                  "FAK_MAX_PER_TICK": "4", "FAK_LAUNCH_SPACING_SEC": "0", "FAK_PROBE": "none",
                  "CLAUDE_CODE_SESSION_ID": None,
                  "FAK_DRAIN_CONTINUOUS": "1", "FAK_DRAIN_MAX": "500"})
    launches: list[list] = []

    def fake_popen(argv, **kw):
        launches.append(list(argv))

        class _P:
            pid = 9000 + len(launches)
        return _P()

    class _R:
        stdout = ""
        stderr = ""
        returncode = 0

    monkeypatch.setattr(wd.subprocess, "run", lambda *a, **k: _R())
    monkeypatch.setattr(wd.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(wd.time, "sleep", lambda s: None)
    # fail-open: admit granted but reason != 'admitted' (missing binary / gate error signature)
    monkeypatch.setattr(wd, "source_admit_gate", lambda: (True, "no-fak-binary"))

    assert wd.main() == 0
    launched = [a for a in launches if "--resume" in a]
    assert len(launched) == 4, "a fail-open governor caps the drain at the tick-quantized floor"
    rows = [__import__("json").loads(x) for x in
            (reg / "resume_ledger.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    assert any(r.get("phase") == "gate_fail_open" for r in rows), \
        "the fail-open is surfaced durably (#2173), never silent"


def _drive_cron(root, monkeypatch, *, backlog, headroom, ticks, tick_sec, spacing_sec, cap, env):
    """Drive `ticks` successive CRON ticks of the REAL main() launch loop over a fixture backlog
    on a virtual clock. Returns (latencies, ticks_used).

    Fixture: `backlog` sessions that are ALL already dead at t=0 (the moment the backlog exists),
    with the first cron tick firing at t=0 and each later tick at k*`tick_sec`. Every session's
    death is t=0, so its launch offset on the virtual clock IS its death->launch latency.

    The clock advances ONLY through the watchdog's own inter-launch spacing (time.sleep), so the
    latencies are measured from the real loop's pacing decisions rather than asserted from a
    formula. Between ticks the plan is re-derived WITHOUT the sessions already resumed: a live
    `claude --resume` forks the transcript into a NEW sid, so fleet_sessions.py never re-plans a
    resumed sid (the .ps1 models the same fact with PruneClosedPlanRows).

    Conservative by construction: real deaths land at a uniformly random offset INSIDE a cron
    period, so the tick-quantized baseline here (deaths exactly on a tick boundary) understates
    its own real latency. A drop measured against it is a lower bound on the real drop."""
    import json as _json
    reg = root / "reg"
    log = root / "log"
    cfg = root / "cfg"
    for d in (reg, log, cfg):
        d.mkdir(parents=True, exist_ok=True)
    plan_path = reg / "resume_plan.json"
    sids = [f"{i:08d}-2222-3333-4444-555555555555" for i in range(backlog)]

    def _row(sid):
        return {"session": sid, "account": ".claude-t", "resume_account": ".claude-t",
                "project": "C--work-fak", "cwd": None, "disp": "STOPPED_APIERR",
                "rehomed": False, "config_dir": str(cfg), "resume_config_dir": str(cfg)}

    base_env = {"FAK_LIVE": "1", "FLEET_REG_DIR": str(reg), "FAK_WATCHDOG_LOG_DIR": str(log),
                "FAK_MAX_PER_TICK": str(cap), "FAK_LAUNCH_SPACING_SEC": str(spacing_sec),
                "FAK_PROBE": "none", "CLAUDE_CODE_SESSION_ID": None}
    base_env.update(env)
    wd = _reload(base_env)

    clock = [0.0]
    launched_at: dict[str, float] = {}

    class _Proc:
        def __init__(self, pid):
            self.pid = pid

    def fake_popen(argv, **kw):
        argv = list(argv)
        if "--resume" in argv:
            launched_at.setdefault(argv[argv.index("--resume") + 1], clock[0])
        return _Proc(9000 + len(launched_at))

    class _R:
        stdout = ""
        stderr = ""
        returncode = 0

    def fake_gate():
        # the source governor stays the real rate limiter: admit while the box has free
        # headroom across every account, then DEFER.
        if len(launched_at) < headroom:
            return True, "admitted"
        return False, "SOURCE_SATURATED"

    monkeypatch.setattr(wd.subprocess, "run", lambda *a, **k: _R())
    monkeypatch.setattr(wd.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(wd.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    monkeypatch.setattr(wd, "source_admit_gate", fake_gate)

    ticks_used = 0
    for k in range(ticks):
        assert clock[0] <= k * tick_sec, "a tick's launches must fit inside the cron period"
        clock[0] = float(k * tick_sec)
        remaining = [s for s in sids if s not in launched_at]
        if not remaining:
            break
        plan_path.write_text(_json.dumps({"plan": [_row(s) for s in remaining]}), encoding="utf-8")
        assert wd.main() == 0
        ticks_used = k + 1
    return [launched_at[s] for s in sids if s in launched_at], ticks_used


def test_continuous_drain_lowers_p50_death_to_launch_latency(tmp_path, monkeypatch):
    # Acceptance #3: MEASURED p50 death->launch latency on a fixture backlog drops vs the
    # tick-quantized baseline. Same backlog, same headroom, same spacing, same cron period --
    # only FAK_DRAIN_CONTINUOUS differs, so the delta is attributable to the drain alone.
    import math
    import statistics
    backlog, cap, tick_sec, spacing = 24, 4, 300, 5
    common = dict(backlog=backlog, headroom=backlog, ticks=backlog + 1, tick_sec=tick_sec,
                  spacing_sec=spacing, cap=cap)
    baseline, baseline_ticks = _drive_cron(
        tmp_path / "baseline", monkeypatch, env={"FAK_DRAIN_CONTINUOUS": None}, **common)
    continuous, continuous_ticks = _drive_cron(
        tmp_path / "continuous", monkeypatch,
        env={"FAK_DRAIN_CONTINUOUS": "1", "FAK_DRAIN_MAX": "500"}, **common)

    # both modes eventually recover the WHOLE backlog -- the comparison is over the same set
    assert len(baseline) == backlog and len(continuous) == backlog
    # ... but the baseline needs ceil(B/cap) cron ticks to get there; the drain needs one
    assert baseline_ticks == math.ceil(backlog / cap) == 6
    assert continuous_ticks == 1

    p50_baseline = statistics.median(baseline)
    p50_continuous = statistics.median(continuous)
    assert p50_continuous < p50_baseline, (p50_continuous, p50_baseline)
    # the baseline p50 is bounded BELOW by the cron period (tick quantization); the drain's p50
    # collapses to the source-governor spacing floor and the whole backlog lands inside one period
    assert p50_baseline > tick_sec, p50_baseline
    assert p50_continuous <= backlog * spacing, p50_continuous
    assert max(continuous) < tick_sec, max(continuous)
    # Measured on this fixture (B=24, cap=4, tick=300s, spacing=5s): p50 death->launch
    # 757.5s -> 57.5s (13.2x), worst case 1515s -> 115s, 6 cron ticks -> 1. Pinned as
    # inequalities, not magic numbers, so a spacing/cap retune cannot make this assertion lie.
    assert p50_continuous * 10 < p50_baseline, (p50_continuous, p50_baseline)


def test_powershell_watchdog_has_continuous_drain_parity():
    # #3587 parity: the .ps1 (the Windows scheduled-task launcher) carries the SAME drain surface as
    # the .py so an operator configures the drain once and it applies whichever watchdog the box runs.
    ps1 = Path(__file__).with_name("fleet_resume_watchdog.ps1").read_text(encoding="utf-8")
    assert "FAK_DRAIN_CONTINUOUS" in ps1 and "FAK_DRAIN_MAX" in ps1
    assert "$drainCap" in ps1
    # the loop caps on the drain backstop, not the raw tick count
    assert "$launched -ge $drainCap" in ps1
    # a governor DEFER ends the drain (box saturated) -> break, not a per-entry deferred storm
    assert "box saturated, ending drain this tick" in ps1
    # a fail-open governor reverts to the tick-quantized cap (no continuous drain without the limiter)
    assert "reverting to per-tick cap" in ps1
    # spacing still honored between launches, bounded by the drain cap
    assert "$launched -lt $drainCap" in ps1


def test_py_and_ps1_name_the_same_drain_knobs():
    # Single-policy intent: the .py and .ps1 launchers MUST read the identical env knobs.
    wd = _reload({})
    ps1 = Path(__file__).with_name("fleet_resume_watchdog.ps1").read_text(encoding="utf-8")
    assert hasattr(wd, "tick_launch_cap")
    assert hasattr(wd, "DRAIN_CONTINUOUS") and hasattr(wd, "DRAIN_MAX")
    assert "FAK_DRAIN_CONTINUOUS" in ps1 and "FAK_DRAIN_MAX" in ps1


# ---- crashed-duplicate tombstones written by `fak resume dedup` (#3146) --------
#
# The Go actuator (cmd/fak/resume_dedup.go) appends ONE manual_override row per crashed
# session whose (project, work-key) a LIVE session already owns. This file owns the OTHER
# half of that contract: the row it writes must make THIS watchdog's resume_blocked()
# return True -- the same way the 734355cc incident was settled by a hand-written row.
#
# DEDUP_TOMBSTONE_LINE is the byte-exact line `fak resume dedup --apply` emits. The Go test
# TestDedupTombstoneWireShapeIsTheWatchdogHonorShape pins the identical literal, so renaming
# a field on either side reds one of the two tests instead of silently un-blocking a
# relaunch the operator believes is tombstoned.
DEDUP_TOMBSTONE_LINE = (
    '{"ts":"2026-07-07T12:00:00Z","phase":"skipped","session":"734355cc-dead-beef",'
    '"account":".claude-w1","project":"C--work-fak","action":"dedup_tombstone",'
    '"manual_override":true,"reason":"duplicate of live session abcd1234-live owning the '
    'same work (loop:--lane claude)","work_key":"loop:--lane claude",'
    '"live_owner":"abcd1234-live","disp":"STOPPED_MIDTURN"}'
)


def test_dedup_tombstone_blocks_the_relaunch(tmp_path, monkeypatch):
    """The acceptance honor point: after `fak resume dedup --apply`, resume_blocked() is True.

    Staged as the incident actually ran -- the duplicate's last resume died recoverably, so
    the watchdog was still willing to relaunch it forever -- and then the tombstone lands.
    """
    wd = _reload({"FAK_MAX_ATTEMPTS": "8"})
    row = _jsonmod.loads(DEDUP_TOMBSTONE_LINE)
    sid = row["session"]
    p = _write_jsonl(tmp_path, "dup.jsonl",
                     _asst_err("You've hit your session limit · resets 6am (America/Los_Angeles)"))
    monkeypatch.setattr(wd, "_newest_transcript", lambda s: p)

    # Before the tombstone: a recoverable death under the cap -> the watchdog relaunches.
    attempts = [{"phase": "launched", "attempt": 1}]
    blocked, why = wd.resume_blocked(sid, attempts)
    assert not blocked, f"precondition: the duplicate is still relaunchable ({why})"

    # After `--apply`: the one appended row settles it at the existing honor point.
    blocked, why = wd.resume_blocked(sid, attempts + [row])
    assert blocked and "operator" in why.lower(), \
        f"a dedup tombstone must block the relaunch, got blocked={blocked} why={why!r}"

    # The tombstone is not itself a resume attempt and never reads as launch pressure, so
    # writing it can neither burn the retry cap nor trip a rate/spacing window.
    assert wd.is_resume_attempt_record(row) is False
    assert row["phase"] in wd.NON_LAUNCH_PHASES

    # A genuinely distinct session is untouched: no tombstone, no block.
    blocked, why = wd.resume_blocked("distinct-work-sid", attempts)
    assert not blocked, f"a session with no tombstone must stay relaunchable ({why})"


def test_dedup_tombstone_is_lifted_by_a_later_rearm():
    """Parity with every other settle: a rearm row AFTER the tombstone re-opens the session,
    so an operator reclaiming a lane is never permanently walled off by a stale dedup."""
    wd = _reload({})
    row = _jsonmod.loads(DEDUP_TOMBSTONE_LINE)
    blocked, _ = wd.resume_blocked(row["session"], [row])
    assert blocked
    blocked, _ = wd.resume_blocked(row["session"],
                                   [row, {"phase": "rearm", "reason": "lane reclaimed"}])
    assert not blocked, "a rearm after the dedup tombstone must lift it"


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
