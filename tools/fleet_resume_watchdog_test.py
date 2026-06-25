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


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
