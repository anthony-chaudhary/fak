#!/usr/bin/env python3
"""Tests for fleet_supervisor_watchdog.py's Slack-event wiring.

The supervisor watchdog's only operator-notify seam is toast() on a RESPAWN; these
pin that --slack / FAK_DISPATCH_SLACK routes that event through tools/slack_post,
gated and best-effort, with no network and no real token/channel.

Run:  python -m pytest tools/fleet_supervisor_watchdog_test.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fleet_supervisor_watchdog as wd  # noqa: E402


def test_post_slack_event_disabled_is_noop():
    calls = []
    out = wd.post_slack_event("Fleet supervisor respawned", "pid=9", enabled=False,
                              transport=lambda *a: calls.append(1))
    assert out is None
    assert calls == []


def test_post_slack_event_posts_when_enabled(monkeypatch):
    for k in ("FAK_DISPATCH_TOKEN", "FAK_DISPATCH_CHANNEL", "FAK_SCOREBOARD_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("FAK_SCOREBOARD_TOKEN", "xoxb-test")
    monkeypatch.setenv("FAK_DISPATCH_CHANNEL", "C0SUP")
    calls = []

    def transport(url, body, headers, timeout):
        calls.append(json.loads(body.decode("utf-8")))
        return 200, json.dumps({"ok": True, "ts": "1.1", "channel": "C0SUP"})

    out = wd.post_slack_event("Fleet supervisor respawned",
                              "was READY; relaunched pid=42 target=4",
                              enabled=True, transport=transport)
    assert out["posted"] is True
    # a respawn is a warning-level event (the supervisor had been DOWN)
    assert calls[0]["text"].startswith("⚠️")
    assert "Fleet supervisor respawned" in calls[0]["text"]


def test_toast_routes_to_slack_when_module_flag_set(monkeypatch):
    monkeypatch.setattr(wd, "SLACK", True)
    monkeypatch.setattr(wd, "SLACK_DRY", False)
    posted = {}
    import slack_post

    def fake_event(title, detail="", *, level="info", **kw):
        posted["title"] = title
        posted["level"] = level
        return {"posted": True}

    monkeypatch.setattr(slack_post, "event", fake_event)
    # osascript is absent on the CI host; toast swallows that and still posts to Slack.
    wd.toast("Fleet supervisor respawned", "was READY; relaunched pid=42 target=4")
    assert posted["title"] == "Fleet supervisor respawned"
    assert posted["level"] == "warn"


def _aged(path, days):
    t = time.time() - days * 86400
    os.utime(path, (t, t))


def test_prune_supervisor_logs_removes_only_expired_pairs(tmp_path):
    # Mirror of the resume twin's retention test (#5346): a stale supervisor-<ts>
    # .log/.log.err pair must be pruned while a pair inside the window and the
    # differently-shaped tick log (watchdog.log) survive.
    old_log = tmp_path / "supervisor-20260101T000000Z.log"
    old_err = tmp_path / "supervisor-20260101T000000Z.log.err"
    fresh = tmp_path / "supervisor-20260701T000000Z.log"
    tick_log = tmp_path / "watchdog.log"  # wrong shape: never pruned here
    for p in (old_log, old_err, fresh, tick_log):
        p.write_text("x", encoding="utf-8")
    _aged(old_log, 20)
    _aged(old_err, 20)
    _aged(tick_log, 20)
    removed = wd.prune_supervisor_logs(str(tmp_path), retain_days=14)
    assert removed == 2
    assert not old_log.exists() and not old_err.exists(), "expired pair must be pruned"
    assert fresh.exists(), "a pair inside the window must survive"
    assert tick_log.exists(), "the tick log is not a per-tick pair -- never pruned"


def test_prune_supervisor_logs_zero_retention_disables(tmp_path):
    p = tmp_path / "supervisor-20260101T000000Z.log"
    p.write_text("x", encoding="utf-8")
    _aged(p, 365)
    assert wd.prune_supervisor_logs(str(tmp_path), retain_days=0) == 0
    assert p.exists()


if __name__ == "__main__":
    try:
        import pytest  # type: ignore
    except ModuleNotFoundError:
        import inspect

        class MiniMonkeyPatch:
            def __init__(self) -> None:
                self._undo = []

            def setenv(self, key: str, value: str) -> None:
                old = os.environ.get(key)
                present = key in os.environ
                self._undo.append(("env", key, old, present))
                os.environ[key] = value

            def delenv(self, key: str, *, raising: bool = True) -> None:
                if key not in os.environ:
                    if raising:
                        raise KeyError(key)
                    return
                old = os.environ[key]
                self._undo.append(("env", key, old, True))
                del os.environ[key]

            def setattr(self, obj, name: str, value) -> None:
                old = getattr(obj, name)
                self._undo.append(("attr", obj, name, old))
                setattr(obj, name, value)

            def undo(self) -> None:
                while self._undo:
                    kind, target, name, old = self._undo.pop()
                    if kind == "env":
                        if old:
                            os.environ[target] = name
                        else:
                            os.environ.pop(target, None)
                    else:
                        setattr(target, name, old)

        failed = 0
        tests = [(name, fn) for name, fn in sorted(globals().items())
                 if name.startswith("test_") and callable(fn)]
        for name, fn in tests:
            mp = MiniMonkeyPatch()
            try:
                params = inspect.signature(fn).parameters
                if "monkeypatch" in params:
                    fn(mp)
                else:
                    fn()
                print(f"ok   {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            finally:
                mp.undo()
        print(f"\n{len(tests) - failed}/{len(tests)} passed")
        sys.exit(1 if failed else 0)

    sys.exit(pytest.main([__file__, "-q"]))

def test_supervisor_alive_bounds_pgrep(monkeypatch):
    seen = {}

    def fake_run(*args, **kwargs):
        seen.update(kwargs)
        raise wd.subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(wd.subprocess, "run", fake_run)
    assert wd.supervisor_alive() == []
    assert seen["timeout"] == wd.PGREP_TIMEOUT_S


def test_main_bounds_unknown_verdict_probe(tmp_path, monkeypatch):
    job = tmp_path / "job"
    scripts = job / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "run_supervise_loop.py").write_text("", encoding="utf-8")
    (scripts / "supervise_now.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(wd, "JOB_DIR", str(job))
    monkeypatch.setattr(wd, "LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(wd, "ENABLED", True)
    monkeypatch.setattr(wd, "supervisor_alive", lambda: [])
    monkeypatch.setattr(wd, "toast", lambda *args: None)
    seen = []

    def fake_run(argv, **kwargs):
        seen.append((argv, kwargs))
        raise wd.subprocess.TimeoutExpired(argv, kwargs.get("timeout"))

    class FakeProc:
        pid = 123

    monkeypatch.setattr(wd.subprocess, "run", fake_run)
    monkeypatch.setattr(wd.subprocess, "Popen", lambda *args, **kwargs: FakeProc())
    assert wd.main() == 10
    assert seen[0][1]["timeout"] == wd.VERDICT_TIMEOUT_S

