#!/usr/bin/env python3
"""Regression tests for fleet_resume_watchdog.py's cross-account env wiring.

These pin the parity the .ps1 already had and the .py port had drifted from:
  * REG_DIR must follow FLEET_REG_DIR so the watchdog READS the registry/plan/ledger
    from the same dir the fleet_sessions.py refresh child WRITES (the silent-no-op /
    split-resume-once-ledger blocker when an ambient FLEET_REG_DIR is set by the
    control pane or an operator).
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


def test_reg_dir_defaults_to_local_registry():
    wd = _reload({"FLEET_REG_DIR": None})
    assert wd.REG_DIR == os.path.join(wd.HERE, "_registry")


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
    assert wd.resolve_probe_mode("auto", live=True) == "blocked"


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

import shutil as _shutil
import subprocess as _subprocess

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

    fak = tmp_path / "fak.cmd"
    fak.write_text(
        '@echo off\r\necho {"decision":{"reason":"%s"}}\r\nexit /b %d\r\n'
        % (fak_reason, fak_exit), encoding="ascii")

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
            "claude": claude, "fak": fak, "ledger": ledger}


def _run_ps1_live(tmp_path, paths, *, spacing=0, max_attempts=8):
    ps1 = Path(__file__).with_name("fleet_resume_watchdog.ps1")
    env = dict(os.environ)
    env["FLEET_STATE_DIR"] = str(tmp_path / "state")
    env["FAK_RESUME_SOURCE_POLICY"] = str(tmp_path / "policy.json")  # hermetic (missing = permissive)
    env.pop("FAK_EXE", None)
    return _subprocess.run(
        [_POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1),
         "-Live", "-Probe", "none",
         "-FleetDir", str(paths["fleet"]), "-RegistryDir", str(paths["reg"]),
         "-LogDir", str(paths["log"]), "-ClaudeExe", str(paths["claude"]),
         "-FakExe", str(paths["fak"]),
         "-LaunchSpacingSec", str(spacing), "-MaxAttempts", str(max_attempts)],
        capture_output=True, text=True, timeout=180, env=env)


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


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
