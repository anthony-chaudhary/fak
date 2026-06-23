#!/usr/bin/env python3
"""Hermetic tests for the opencode transcript adapter.

These pin the contract from issue #423: an opencode classifier that shares
fleet_sessions' invariant table, anchors disposition to STRUCTURED stop
events (never prose), emits ``opencode run --session <id>`` for resume, and
keys the resume-once ledger on backend + session id.
"""
from __future__ import annotations

import datetime as dt
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fleet_sessions  # noqa: E402
import opencode_adapter as oa  # noqa: E402

# A fixed clock so liveness is deterministic. The session's updated_epoch is set
# relative to this in each test.
NOW = dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
NOW_EPOCH = NOW.timestamp()


def _session(messages, updated_epoch=None, **kw):
    """A normalized opencode session; defaults to long-stale so liveness never
    masks the disposition under test unless a test sets updated_epoch."""
    s = {
        "session": "ses_test",
        "updated_epoch": updated_epoch if updated_epoch is not None else NOW_EPOCH - 3600,
        "directory": "/repo",
        "messages": messages,
    }
    s.update(kw)
    return s


def _classify(messages, **kw):
    return oa.classify_session(_session(messages, **kw), now=NOW)


class TestInvariantTable(unittest.TestCase):
    def test_shares_fleet_sessions_table(self):
        # the adapter must NOT define its own table -- it imports the canonical one.
        self.assertIs(oa.CATEGORY, fleet_sessions.CATEGORY)
        self.assertIs(oa.DEAD, fleet_sessions.DEAD)

    def test_every_emitted_disp_is_in_the_table_with_matching_category(self):
        cases = [
            [{"role": "assistant", "finish": "stop", "text": "done"}],
            [{"role": "assistant", "finish": "error",
              "error": {"name": "ProviderRateLimitError"}, "text": ""}],
            [{"role": "assistant", "finish": "error",
              "error": {"name": "AuthenticationError"}, "text": ""}],
            [{"role": "assistant", "finish": "error",
              "error": {"name": "ConnectionResetError"}, "text": ""}],
            [{"role": "assistant", "finish": "aborted", "text": ""}],
            [{"role": "assistant", "finish": None,
              "tools": [{"name": "bash", "status": "running"}], "text": ""}],
            [{"role": "assistant", "finish": None, "text": "half a thought"}],
            [{"role": "user", "text": "hi"}],
        ]
        for msgs in cases:
            row = _classify(msgs)
            self.assertIn(row["disp"], fleet_sessions.CATEGORY)
            cat, cause = fleet_sessions.CATEGORY[row["disp"]]
            self.assertEqual(row["category"], cat)
            self.assertEqual(row["cause"], cause)
            self.assertEqual(row["backend"], "opencode")


class TestDispositions(unittest.TestCase):
    def test_clean_finish_is_done(self):
        self.assertEqual(_classify([{"role": "assistant", "finish": "stop",
                                     "text": "shipped"}])["disp"], "DONE")

    def test_rate_limit_error_name_is_stopped_limit(self):
        row = _classify([{"role": "assistant", "finish": "error",
                          "error": {"name": "ProviderRateLimitError",
                                    "message": "429 slow down"}, "text": ""}])
        self.assertEqual(row["disp"], "STOPPED_LIMIT")
        self.assertEqual(row["category"], "INFRA")

    def test_auth_error_name_is_infra_auth(self):
        row = _classify([{"role": "assistant", "finish": "error",
                          "error": {"name": "Unauthorized"}, "text": ""}])
        self.assertEqual(row["disp"], "INFRA_AUTH")

    def test_unknown_error_is_api_error(self):
        row = _classify([{"role": "assistant", "finish": "error",
                          "error": {"name": "WeirdTransportGlitch"}, "text": ""}])
        self.assertEqual(row["disp"], "STOPPED_APIERR")

    def test_aborted_finish_is_user_closed(self):
        self.assertEqual(_classify([{"role": "assistant", "finish": "aborted",
                                     "text": ""}])["disp"], "USER_CLOSED")

    def test_pending_tool_is_dead_midtool(self):
        row = _classify([{"role": "assistant", "finish": None, "text": "",
                          "tools": [{"name": "edit", "status": "running"}]}])
        self.assertEqual(row["disp"], "DEAD_MIDTOOL")
        self.assertIn("edit", row["reason"])

    def test_assistant_without_finish_is_dead_killed(self):
        self.assertEqual(_classify([{"role": "assistant", "finish": None,
                                     "text": "mid-thought"}])["disp"], "DEAD_KILLED")

    def test_recent_activity_is_live(self):
        row = _classify([{"role": "assistant", "finish": None, "text": "working"}],
                        updated_epoch=NOW_EPOCH - 30)  # 30s ago
        self.assertEqual(row["disp"], "LIVE")

    def test_no_assistant_message_is_quiet(self):
        self.assertEqual(_classify([{"role": "user", "text": "go"}])["disp"],
                         "STOPPED_QUIET")


class TestTalkAboutVsIsPitfall(unittest.TestCase):
    """The headline trap from #423: prose mentioning a stop reason must NOT drive
    the disposition -- only the structured stop event does."""

    def test_text_says_rate_limit_but_finish_is_clean_is_done(self):
        row = _classify([{"role": "assistant", "finish": "stop", "error": None,
                          "text": "I was worried about hitting the rate limit "
                                  "and getting throttled, but we finished."}])
        self.assertEqual(row["disp"], "DONE")
        self.assertNotEqual(row["disp"], "STOPPED_LIMIT")

    def test_text_says_auth_error_but_no_structured_error_is_not_auth(self):
        row = _classify([{"role": "assistant", "finish": "stop", "error": None,
                          "text": "earlier we hit an authentication error / 401, "
                                  "fixed it, then completed the task."}])
        self.assertEqual(row["disp"], "DONE")


class TestResumeAndLedger(unittest.TestCase):
    def test_resume_cmd_uses_opencode_run_session(self):
        cmd = oa.resume_cmd({"session": "ses_xyz"})
        self.assertIn("opencode run --session ses_xyz", cmd)
        self.assertNotIn("CLAUDE_CONFIG_DIR", cmd)
        self.assertNotIn("claude --resume", cmd)

    def test_ledger_key_is_backend_scoped(self):
        self.assertEqual(oa.ledger_key("opencode", "ses_1"), "opencode:ses_1")
        # a Claude uuid and an opencode id can never collide on a bare id
        self.assertNotEqual(oa.ledger_key("opencode", "abc"),
                            oa.ledger_key("claude", "abc"))

    def test_plan_entry_shape_tags_backend_and_carries_resume_cmd(self):
        row = _classify([{"role": "assistant", "finish": None, "text": "",
                          "tools": [{"name": "bash", "status": "running"}]}])
        entry = oa.plan_entry(row)
        self.assertEqual(entry["backend"], "opencode")
        self.assertEqual(entry["ledger_key"], "opencode:ses_test")
        self.assertIn("opencode run --session", entry["resume_cmd"])
        self.assertIsNone(entry["config_dir"])  # no CLAUDE_CONFIG_DIR for opencode

    def test_plan_entry_done_has_no_resume_cmd(self):
        row = _classify([{"role": "assistant", "finish": "stop", "text": "done"}])
        self.assertIsNone(oa.plan_entry(row)["resume_cmd"])


class TestAutonomy(unittest.TestCase):
    def test_explicit_flag_wins(self):
        s = _session([{"role": "user", "text": "just chatting"}], autonomous=True)
        self.assertTrue(oa.classify_session(s, now=NOW)["autonomous"])

    def test_directive_in_first_user_message_infers_autonomy(self):
        row = _classify([{"role": "user", "text": "Resolve GitHub issue #423"},
                         {"role": "assistant", "finish": "stop", "text": "ok"}])
        self.assertTrue(row["autonomous"])

    def test_plain_chat_is_not_autonomous(self):
        row = _classify([{"role": "user", "text": "what is 2+2"},
                         {"role": "assistant", "finish": "stop", "text": "4"}])
        self.assertFalse(row["autonomous"])


class TestFromStorage(unittest.TestCase):
    def test_normalizes_parts_and_ms_epoch(self):
        info = {"id": "ses_store", "directory": "/repo",
                "time": {"updated": int(NOW_EPOCH * 1000) - 3_600_000}}  # 1h ago, ms
        messages = [
            {"role": "user", "parts": [{"type": "text", "text": "go"}]},
            {"role": "assistant", "finish": "stop",
             "parts": [{"type": "tool", "tool": "bash",
                        "state": {"status": "completed"}},
                       {"type": "text", "text": "all set"}]},
        ]
        row = oa.from_storage(info, messages, now=NOW)
        self.assertEqual(row["session"], "ses_store")
        self.assertEqual(row["cwd"], "/repo")
        self.assertEqual(row["disp"], "DONE")

    def test_running_tool_part_is_dead_midtool(self):
        info = {"id": "ses_crash", "directory": "/repo",
                "time": {"updated": int(NOW_EPOCH * 1000) - 3_600_000}}
        messages = [
            {"role": "assistant", "finish": None,
             "parts": [{"type": "tool", "tool": "edit",
                        "state": {"status": "running"}}]},
        ]
        row = oa.from_storage(info, messages, now=NOW)
        self.assertEqual(row["disp"], "DEAD_MIDTOOL")


if __name__ == "__main__":
    unittest.main()
