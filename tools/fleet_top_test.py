#!/usr/bin/env python3
"""Hermetic tests for fleet_top's fold + render.

These never touch disk or a subprocess: they feed a synthetic `fleet_sessions json`
doc straight into build_snapshot / render_frame, so the deterministic snapshot shape
and the operator-facing frame are pinned without a live fleet."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fleet_top  # noqa: E402


def _doc():
    """A fleet_sessions json doc shaped like a real mixed fleet."""
    return {
        "now": "2026-06-23T18:00:00+00:00",
        "throttle": {
            ".claude-bravo": {"reset": "Jun 26, 6pm", "age_min": 360.0},
        },
        "accounts": [
            {"account": ".claude-alpha", "tag": "alpha", "available": True,
             "blocked": False, "throttled": False, "block_kind": "", "block_reason": "",
             "config_dir": "/home/u/.claude-alpha"},
            {"account": ".claude-bravo", "tag": "bravo", "available": False,
             "blocked": True, "throttled": True, "block_kind": "throttle",
             "block_reason": "rate limited", "reset": "Jun 26, 6pm",
             "verdict_source": "passive", "verdict_age_min": 120.0,
             "config_dir": "/home/u/.claude-bravo"},
            {"account": ".claude-charlie", "tag": "charlie", "available": False,
             "blocked": True, "throttled": False, "block_kind": "auth",
             "block_reason": "please run /login", "config_dir": "/home/u/.claude-charlie"},
            {"account": ".claude-delta", "tag": "delta", "available": False,
             "blocked": True, "throttled": False, "block_kind": "access",
             "block_reason": "subscription disabled", "config_dir": "/home/u/.claude-delta"},
        ],
        "rows": [
            {"category": "LIVE", "disp": "LIVE", "action": "SKIP_DONE", "age_min": 1.0,
             "account": ".claude-alpha", "project": "C--work-fak", "session": "aaaaaaaa-1",
             "git": "main", "last": "", "resume_cmd": ""},
            {"category": "AGENT", "disp": "DONE", "action": "SKIP_DONE", "age_min": 12.0,
             "account": ".claude-alpha", "project": "C--work-fak", "session": "bbbbbbbb-2",
             "git": "main", "last": "", "resume_cmd": ""},
            {"category": "AGENT", "disp": "DEAD_MIDTOOL", "action": "AUTO_RESUME", "age_min": 7.0,
             "account": ".claude-alpha", "project": "C--work-fleet", "session": "cccccccc-3",
             "git": "main", "last": "", "resume_cmd": "claude --resume cccccccc-3 -p 'go'"},
            {"category": "INFRA", "disp": "STOPPED_LIMIT", "action": "DEFER_THROTTLED",
             "age_min": 30.0, "account": ".claude-bravo", "project": "C--work-fleet",
             "session": "dddddddd-4", "git": "main", "last": "", "resume_cmd": ""},
            {"category": "HANGING", "disp": "PARKED_WAIT", "action": "SURFACE", "age_min": 45.0,
             "account": ".claude-alpha", "project": "C--work-fak", "session": "eeeeeeee-5",
             "git": "main", "last": "awaiting task", "resume_cmd": ""},
        ],
    }


class BuildSnapshotTest(unittest.TestCase):
    def setUp(self):
        self.snap = fleet_top.build_snapshot(
            _doc(), workspace="C:/work/fak", window_h=10.0, now="2026-06-23T18:00:00Z"
        )

    def test_session_category_counts(self):
        cats = self.snap["sessions"]["by_category"]
        self.assertEqual(self.snap["sessions"]["total"], 5)
        self.assertEqual(cats, {"LIVE": 1, "AGENT": 2, "INFRA": 1, "HANGING": 1})

    def test_cause_breakdown(self):
        self.assertEqual(
            self.snap["sessions"]["causes"]["AGENT"],
            {"completed": 1, "crash_mid_tool": 1},
        )

    def test_account_partition(self):
        acc = self.snap["accounts"]
        self.assertEqual(acc["total"], 4)
        self.assertEqual(acc["usable"], 1)
        self.assertEqual(acc["available"], ["alpha"])
        self.assertEqual([t["tag"] for t in acc["throttled"]], ["bravo"])
        # auth + access walls are "blocked other", not throttled.
        self.assertEqual(
            sorted(b["tag"] for b in acc["blocked"]), ["charlie", "delta"]
        )

    def test_throttled_list_excludes_an_account_that_recovered(self):
        """The day24 stale-throttle case: an account still carried in the throttle MAP
        but now `available` in the accounts block (a newer successful turn cleared it)
        must NOT render as throttled -- the throttled list is driven off availability,
        not the raw map."""
        doc = _doc()
        # alpha is available; plant a stale throttle-map entry for it as if a 5-min-old
        # limit banner were still cached. It must be ignored because alpha.available=True.
        doc["throttle"][".claude-alpha"] = {"reset": "6pm", "age_min": 5.0}
        snap = fleet_top.build_snapshot(
            doc, workspace="C:/work/fak", window_h=10.0, now="2026-06-23T18:00:00Z")
        throttled_tags = [t["tag"] for t in snap["accounts"]["throttled"]]
        self.assertNotIn("alpha", throttled_tags)
        self.assertEqual(throttled_tags, ["bravo"])

    def test_throttled_entry_carries_freshness(self):
        snap = self.snap
        bravo = snap["accounts"]["throttled"][0]
        self.assertEqual(bravo["verdict_source"], "passive")
        self.assertEqual(bravo["verdict_age_min"], 120.0)

    def test_attention_ranks_resumable_first_and_carries_command(self):
        attn = self.snap["attention"]
        self.assertEqual(attn[0]["level"], "crit")
        self.assertIn("resumable", attn[0]["title"])
        self.assertEqual(attn[0]["command"], "claude --resume cccccccc-3 -p 'go'")

    def test_attention_flags_login_and_access_and_parked(self):
        titles = " | ".join(i["title"] for i in self.snap["attention"])
        self.assertIn("needs /login", titles)
        self.assertIn("access wall", titles)
        self.assertIn("parked/quiet", titles)


class RenderFrameTest(unittest.TestCase):
    def setUp(self):
        self.snap = fleet_top.build_snapshot(
            _doc(), workspace="C:/work/fak", window_h=10.0, now="2026-06-23T18:00:00Z"
        )

    def test_plain_frame_has_sections_and_command(self):
        frame = fleet_top.render_frame(self.snap, color=False, interval=None)
        self.assertIn("┌─ fleet top · C:/work/fak", frame)
        self.assertIn("SESSIONS  5 in 10.0h window", frame)
        self.assertIn("ACCOUNTS  1/4 usable", frame)
        self.assertIn("ATTENTION", frame)
        self.assertIn("claude --resume cccccccc-3", frame)
        self.assertIn("· snapshot", frame)
        # color off => no escape codes, so the frame stays diffable.
        self.assertNotIn("\x1b[", frame)

    def test_throttled_line_shows_freshness(self):
        frame = fleet_top.render_frame(self.snap, color=False, interval=None)
        # the throttled line carries how stale its evidence is, so a cached expired
        # throttle reads visibly differently from a live one.
        self.assertIn("throttled  bravo  resets Jun 26, 6pm", frame)
        self.assertIn("(passive, seen 120m ago)", frame)

    def test_live_footer_shows_cadence(self):
        frame = fleet_top.render_frame(self.snap, color=False, interval=5)
        self.assertIn("refresh 5s · Ctrl-C to quit", frame)

    def test_color_emits_escapes(self):
        frame = fleet_top.render_frame(self.snap, color=True, interval=None)
        self.assertIn("\x1b[", frame)


class EdgeCaseTest(unittest.TestCase):
    def test_error_doc_renders_clean_unavailable_line(self):
        snap = fleet_top.build_snapshot(
            {}, workspace="C:/work/fak", window_h=10.0, now="now", error="boom"
        )
        frame = fleet_top.render_frame(snap, color=False)
        self.assertIn("signal unavailable: boom", frame)

    def test_empty_fleet_is_quiet(self):
        snap = fleet_top.build_snapshot(
            {"rows": [], "accounts": [], "throttle": {}},
            workspace="C:/work/fak", window_h=10.0, now="now",
        )
        self.assertEqual(snap["sessions"]["total"], 0)
        self.assertEqual(snap["attention"], [])
        frame = fleet_top.render_frame(snap, color=False)
        self.assertIn("(no sessions in window)", frame)
        self.assertIn("fleet is quiet", frame)


class SlackPostTest(unittest.TestCase):
    """The --slack wiring: slack_text is pure; post_to_slack resolves + posts via the
    injected transport (no network, no real token/channel)."""

    SLACK_KEYS = ("FAK_DISPATCH_TOKEN", "FAK_DISPATCH_CHANNEL", "FAK_SCOREBOARD_TOKEN")

    def _clear_env(self):
        import os
        saved = {k: os.environ.pop(k, None) for k in self.SLACK_KEYS}
        self.addCleanup(self._restore_env, saved)

    def _restore_env(self, saved):
        import os
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_slack_text_headline_and_fenced_frame(self):
        snap = fleet_top.build_snapshot(
            _doc(), workspace="C:/work/fak", window_h=10.0, now="2026-06-23T18:00:00Z")
        text = fleet_top.slack_text(snap)
        self.assertIn("*fleet status:*", text)
        self.assertIn("session(s)", text)
        self.assertIn("accounts usable", text)
        self.assertIn("```", text)                # fenced for monospace
        self.assertIn("ATTENTION", text)          # the frame body is included

    def test_post_to_slack_via_injected_transport(self):
        import json as _json
        import os
        self._clear_env()
        os.environ["FAK_SCOREBOARD_TOKEN"] = "xoxb-test-tok"
        os.environ["FAK_DISPATCH_CHANNEL"] = "C0FLEET"
        calls = []

        def transport(url, body, headers, timeout):
            calls.append({"body": _json.loads(body.decode("utf-8")),
                          "auth": headers.get("Authorization")})
            return 200, _json.dumps({"ok": True, "ts": "1.2", "channel": "C0FLEET"})

        snap = fleet_top.build_snapshot(
            _doc(), workspace="C:/work/fak", window_h=10.0, now="now")
        verdict = fleet_top.post_to_slack(snap, transport=transport)
        self.assertTrue(verdict["posted"])
        self.assertEqual(verdict["channel"], "C0FLEET")
        self.assertEqual(calls[0]["auth"], "Bearer xoxb-test-tok")
        self.assertIn("fleet status", calls[0]["body"]["text"])

    def test_post_to_slack_dry_run_does_not_call_transport(self):
        import os
        self._clear_env()
        os.environ["FAK_SCOREBOARD_TOKEN"] = "xoxb-test-tok"
        os.environ["FAK_DISPATCH_CHANNEL"] = "C0FLEET"
        calls = []

        def transport(url, body, headers, timeout):
            calls.append(1)
            return 200, "{}"

        snap = fleet_top.build_snapshot(
            _doc(), workspace="C:/work/fak", window_h=10.0, now="now")
        verdict = fleet_top.post_to_slack(snap, dry_run=True, transport=transport)
        self.assertFalse(verdict["posted"])
        self.assertEqual(verdict["skipped"], "dry-run")
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
