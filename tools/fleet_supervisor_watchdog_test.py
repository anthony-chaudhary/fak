#!/usr/bin/env python3
"""Tests for fleet_supervisor_watchdog.py's Slack-event wiring.

The supervisor watchdog's only operator-notify seam is toast() on a RESPAWN; these
pin that --slack / FAK_DISPATCH_SLACK routes that event through tools/slack_post,
gated and best-effort, with no network and no real token/channel.

Run:  python -m pytest tools/fleet_supervisor_watchdog_test.py
"""
from __future__ import annotations

import json
import sys
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


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
