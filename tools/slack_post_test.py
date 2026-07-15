#!/usr/bin/env python3
"""Hermetic tests for tools/slack_post.py.

Resolution is exercised against a temp .env.slack.local + a patched environment;
the transport is injected so a "post" records its call and returns a canned Slack
response — no network, no real token, no real channel. Mirrors the contract of
internal/slackenv + cmd/fak/slack.go's dispatch surface.
"""
from __future__ import annotations

import importlib.util
import json
import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "slack_post.py"


def load():
    spec = importlib.util.spec_from_file_location("slack_post", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sp = load()


class _Recorder:
    """An injectable transport that records the one call and returns a canned reply."""

    def __init__(self, reply: dict | None = None, status: int = 200):
        self.reply = reply if reply is not None else {"ok": True, "ts": "111.222",
                                                       "channel": "C0DISPATCH"}
        self.status = status
        self.calls: list[dict] = []

    def __call__(self, url, body, headers, timeout):
        self.calls.append({
            "url": url,
            "body": json.loads(body.decode("utf-8")),
            "headers": dict(headers),
            "timeout": timeout,
        })
        return self.status, json.dumps(self.reply)


class _EnvGuard:
    """Clear the Slack keys so a stray operator env never leaks into a test."""

    KEYS = ("FAK_DISPATCH_TOKEN", "FAK_DISPATCH_CHANNEL", "FAK_SCOREBOARD_TOKEN",
            "FAK_SCOREBOARD_CHANNEL", "FAK_SLACK_ROLE")

    def __enter__(self):
        self._saved = {k: os.environ.pop(k, None) for k in self.KEYS}
        return self

    def __exit__(self, *exc):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _write_env(tmp: Path, lines: list[str]) -> Path:
    (tmp / sp.ENV_FILE_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp


class ResolutionTests(unittest.TestCase):
    def test_file_value_walks_up_and_trims_export(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d, _EnvGuard():
            root = Path(d)
            # grammar mirrors internal/slackenv: `KEY=value`, an optional `export `
            # prefix, surrounding whitespace trimmed (no space *around* the `=`).
            _write_env(root, ["export FAK_DISPATCH_CHANNEL=C0FROMFILE  ",
                              "FAK_SCOREBOARD_TOKEN=xoxb-abc-123"])
            sub = root / "cmd" / "deep"
            sub.mkdir(parents=True)
            self.assertEqual(sp.file_value("FAK_DISPATCH_CHANNEL", start=sub), "C0FROMFILE")
            self.assertEqual(sp.file_value("FAK_SCOREBOARD_TOKEN", start=sub), "xoxb-abc-123")
            self.assertEqual(sp.file_value("FAK_MISSING", start=sub), "")

    def test_env_beats_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d, _EnvGuard():
            root = _write_env(Path(d), ["FAK_DISPATCH_CHANNEL=C0FILE"])
            os.environ["FAK_DISPATCH_CHANNEL"] = "C0ENV"
            val, src = sp.lookup("FAK_DISPATCH_CHANNEL", start=root)
            self.assertEqual((val, src), ("C0ENV", "env"))

    def test_token_falls_back_to_scoreboard(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d, _EnvGuard():
            root = _write_env(Path(d), ["FAK_SCOREBOARD_TOKEN=xoxb-shared-999"])
            val, src = sp.resolve_token(start=root)
            self.assertEqual(val, "xoxb-shared-999")
            self.assertIn("scoreboard-fallback", src)

    def test_channel_precedence_explicit_then_env_then_default(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d, _EnvGuard():
            root = Path(d)
            self.assertEqual(sp.resolve_channel("C0EXPLICIT", default="C0DEF")[0], "C0EXPLICIT")
            os.environ["FAK_DISPATCH_CHANNEL"] = "C0ENV"
            self.assertEqual(sp.resolve_channel("", default="C0DEF")[0], "C0ENV")
            del os.environ["FAK_DISPATCH_CHANNEL"]
            self.assertEqual(sp.resolve_channel("", default="C0DEF", start=root),
                             ("C0DEF", "built-in default"))
            self.assertEqual(sp.resolve_channel("", default="", start=root), ("", "unset"))


class SendTests(unittest.TestCase):
    def test_send_posts_with_resolved_token_and_channel(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d, _EnvGuard():
            root = _write_env(Path(d), ["FAK_SCOREBOARD_TOKEN=xoxb-shared-999",
                                        "FAK_DISPATCH_CHANNEL=C0DISPATCH"])
            rec = _Recorder()
            res = sp.send("dispatch is green", transport=rec, start=root)
            self.assertTrue(res["posted"])
            self.assertEqual(res["channel"], "C0DISPATCH")
            self.assertEqual(res["ts"], "111.222")
            self.assertIn("scoreboard-fallback", res["token_source"])
            # the call carried the bearer token and the right payload
            self.assertEqual(len(rec.calls), 1)
            call = rec.calls[0]
            self.assertTrue(call["url"].endswith("/chat.postMessage"))
            self.assertEqual(call["headers"]["Authorization"], "Bearer xoxb-shared-999")
            self.assertEqual(call["body"]["channel"], "C0DISPATCH")
            self.assertIn("dispatch is green", call["body"]["text"])
            self.assertIn("S/N self-score", call["body"]["text"])
            self.assertIn("signal_noise", res)

    def test_code_wraps_in_fence(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d, _EnvGuard():
            root = _write_env(Path(d), ["FAK_SCOREBOARD_TOKEN=t", "FAK_DISPATCH_CHANNEL=C0X"])
            rec = _Recorder()
            sp.send("│ card │", code=True, transport=rec, start=root)
            self.assertTrue(rec.calls[0]["body"]["text"].startswith("```\n"))
            self.assertIn("│ card │", rec.calls[0]["body"]["text"])
            self.assertIn("S/N self-score", rec.calls[0]["body"]["text"])

    def test_send_can_suppress_signal_noise_footer(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d, _EnvGuard():
            root = _write_env(Path(d), ["FAK_SCOREBOARD_TOKEN=t", "FAK_DISPATCH_CHANNEL=C0X"])
            rec = _Recorder()
            res = sp.send("dispatch card", transport=rec, start=root,
                          include_signal_noise=False)
            self.assertTrue(res["posted"])
            self.assertIn("signal_noise", res)
            self.assertNotIn("S/N self-score", rec.calls[0]["body"]["text"])

    def test_dry_run_resolves_but_never_calls_transport(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d, _EnvGuard():
            root = _write_env(Path(d), ["FAK_SCOREBOARD_TOKEN=t", "FAK_DISPATCH_CHANNEL=C0X"])
            rec = _Recorder()
            res = sp.send("x", dry_run=True, transport=rec, start=root)
            self.assertFalse(res["posted"])
            self.assertEqual(res["skipped"], "dry-run")
            self.assertEqual(res["channel"], "C0X")
            self.assertTrue(res["token_set"])
            self.assertEqual(rec.calls, [])

    def test_skips_cleanly_when_no_channel(self):
        with _EnvGuard():
            root = Path(os.getcwd())  # no .env here for these keys (guard cleared env)
            rec = _Recorder()
            res = sp.send("x", channel="", default_channel="", transport=rec,
                          start=Path("/nonexistent-zzz"))
            self.assertFalse(res["posted"])
            self.assertIn("no channel", res["skipped"])
            self.assertEqual(rec.calls, [])
            _ = root

    def test_skips_cleanly_when_no_token(self):
        with _EnvGuard():
            rec = _Recorder()
            res = sp.send("x", channel="C0X", transport=rec, start=Path("/nonexistent-zzz"))
            self.assertFalse(res["posted"])
            self.assertIn("no bot token", res["skipped"])
            self.assertEqual(rec.calls, [])

    def test_slack_error_becomes_verdict_not_exception(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d, _EnvGuard():
            root = _write_env(Path(d), ["FAK_SCOREBOARD_TOKEN=t", "FAK_DISPATCH_CHANNEL=C0X"])
            rec = _Recorder(reply={"ok": False, "error": "channel_not_found"}, status=200)
            res = sp.send("x", transport=rec, start=root)
            self.assertFalse(res["posted"])
            self.assertEqual(res["error"], "channel_not_found")

    def test_non_json_response_is_handled(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d, _EnvGuard():
            root = _write_env(Path(d), ["FAK_SCOREBOARD_TOKEN=t", "FAK_DISPATCH_CHANNEL=C0X"])

            def bad(url, body, headers, timeout):
                return 500, "<html>nope</html>"

            res = sp.send("x", transport=bad, start=root)
            self.assertFalse(res["posted"])
            self.assertIn("non-JSON", res["error"])


class EventTests(unittest.TestCase):
    """event_text formats one actionable line; event posts it through send."""

    def test_event_text_glyph_and_bold_title(self):
        self.assertEqual(sp.event_text("respawned", "pid=9", level="warn"),
                         "⚠️ *respawned* — pid=9")
        self.assertEqual(sp.event_text("ok"), "🟢 *ok*")  # default level, no detail
        self.assertTrue(sp.event_text("x", level="crit").startswith("🔴"))
        self.assertTrue(sp.event_text("x", level="resume").startswith("♻️"))

    def test_event_posts_formatted_line(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d, _EnvGuard():
            root = _write_env(Path(d), ["FAK_SCOREBOARD_TOKEN=t", "FAK_DISPATCH_CHANNEL=C0X"])
            rec = _Recorder()
            res = sp.event("supervisor respawned", "pid=42", level="warn",
                           transport=rec, start=root)
            self.assertTrue(res["posted"])
            self.assertIn("⚠️ *supervisor respawned* — pid=42", rec.calls[0]["body"]["text"])
            self.assertIn("S/N self-score", rec.calls[0]["body"]["text"])


class SignalNoiseTests(unittest.TestCase):
    def test_signal_noise_score_counts_frame_as_noise(self):
        score = sp.signal_noise_score("*headline*\n```\n┌ frame\nvalue\n└ end\n```")
        self.assertEqual(score["signal"], 2)
        self.assertEqual(score["noise"], 5)
        self.assertIn("S/N self-score", sp.signal_noise_line(score))

    def test_append_signal_noise_is_idempotent(self):
        once = sp.append_signal_noise("hello")
        twice = sp.append_signal_noise(once)
        self.assertEqual(once, twice)


class RedactTests(unittest.TestCase):
    def test_redact_keeps_only_last_four(self):
        self.assertEqual(sp.redact_token("xoxb-secret-tail"), "****tail")
        self.assertEqual(sp.redact_token(""), "(unset)")
        self.assertEqual(sp.redact_token("ab"), "****")


class OperatorChannelGuardTests(unittest.TestCase):
    """#4652: scheduler-test/scout/test verification traffic cannot post to the production
    operator channel (FAK_DISPATCH_CHANNEL). A loop declares its role via the role= kwarg
    or the FAK_SLACK_ROLE env stamp; a role in NON_OPERATOR_ROLES that resolves the
    operator channel is REFUSED before any post. Undeclared/operator traffic and the
    fake-transport unit tests are unaffected."""

    # ----- _effective_role -----
    def test_effective_role_explicit_beats_env_and_lowercases(self):
        with _EnvGuard():
            os.environ["FAK_SLACK_ROLE"] = "operator"
            self.assertEqual(sp._effective_role("SCOUT", transport=lambda *a: None), "scout")

    def test_effective_role_reads_env_stamp(self):
        with _EnvGuard():
            os.environ["FAK_SLACK_ROLE"] = "Scheduler-Test"
            self.assertEqual(sp._effective_role("", transport=lambda *a: None),
                             "scheduler-test")

    def test_effective_role_auto_test_only_on_live_transport(self):
        # under this unittest run: undeclared + a real (None) transport => "test"; but a
        # fake transport (the whole existing suite) stays undeclared so nothing is blocked.
        with _EnvGuard():
            self.assertEqual(sp._effective_role("", transport=None), "test")
            self.assertEqual(sp._effective_role("", transport=lambda *a: None), "")

    # ----- operator_channel_block (pure) -----
    def test_block_reason_for_non_operator_role_on_prod_channel(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d, _EnvGuard():
            root = _write_env(Path(d), ["FAK_DISPATCH_CHANNEL=C0PROD"])
            reason = sp.operator_channel_block("C0PROD", "scout", start=root)
            self.assertIn("may not post to the production operator channel C0PROD", reason)
            self.assertIn("#4652", reason)

    def test_no_block_for_operator_or_undeclared_role(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d, _EnvGuard():
            root = _write_env(Path(d), ["FAK_DISPATCH_CHANNEL=C0PROD"])
            self.assertEqual(sp.operator_channel_block("C0PROD", "operator", start=root), "")
            self.assertEqual(sp.operator_channel_block("C0PROD", "", start=root), "")

    def test_no_block_for_non_operator_role_on_a_different_channel(self):
        # an audit channel (any channel that is not the operator channel) is fine.
        import tempfile
        with tempfile.TemporaryDirectory() as d, _EnvGuard():
            root = _write_env(Path(d), ["FAK_DISPATCH_CHANNEL=C0PROD"])
            self.assertEqual(sp.operator_channel_block("C0AUDIT", "scout", start=root), "")

    def test_no_block_when_prod_channel_unresolved(self):
        # fail-safe: with no operator channel identifiable, we cannot claim a collision,
        # so we do not block an arbitrary channel.
        import tempfile
        with tempfile.TemporaryDirectory() as d, _EnvGuard():
            self.assertEqual(sp.operator_channel_block("C0ANY", "scout", start=Path(d)), "")

    # ----- send() integration -----
    def test_send_refuses_scout_role_to_prod_channel_via_kwarg(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d, _EnvGuard():
            root = _write_env(Path(d), ["FAK_SCOREBOARD_TOKEN=t",
                                        "FAK_DISPATCH_CHANNEL=C0PROD"])
            rec = _Recorder()
            res = sp.send("per-run dispatch result", transport=rec, start=root, role="scout")
            self.assertFalse(res["posted"])
            self.assertTrue(res["blocked"])
            self.assertEqual(res["role"], "scout")
            self.assertIn("C0PROD", res["skipped"])
            self.assertEqual(rec.calls, [])  # never reached the transport

    def test_send_refuses_via_env_role_stamp(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d, _EnvGuard():
            root = _write_env(Path(d), ["FAK_SCOREBOARD_TOKEN=t",
                                        "FAK_DISPATCH_CHANNEL=C0PROD"])
            os.environ["FAK_SLACK_ROLE"] = "scheduler-test"
            rec = _Recorder()
            res = sp.send("scheduler self-check", transport=rec, start=root)
            self.assertTrue(res["blocked"])
            self.assertEqual(res["role"], "scheduler-test")
            self.assertEqual(rec.calls, [])

    def test_send_allows_scout_role_to_a_separate_audit_channel(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d, _EnvGuard():
            root = _write_env(Path(d), ["FAK_SCOREBOARD_TOKEN=t",
                                        "FAK_DISPATCH_CHANNEL=C0PROD"])
            rec = _Recorder(reply={"ok": True, "ts": "9.9", "channel": "C0AUDIT"})
            res = sp.send("scout note", channel="C0AUDIT", transport=rec, start=root,
                          role="scout")
            self.assertTrue(res["posted"])
            self.assertIsNone(res["blocked"])
            self.assertEqual(len(rec.calls), 1)

    def test_send_allows_undeclared_role_to_prod_channel(self):
        # the real operator roll-up is undeclared (or role=operator) => unaffected.
        import tempfile
        with tempfile.TemporaryDirectory() as d, _EnvGuard():
            root = _write_env(Path(d), ["FAK_SCOREBOARD_TOKEN=t",
                                        "FAK_DISPATCH_CHANNEL=C0PROD"])
            rec = _Recorder(reply={"ok": True, "ts": "1.1", "channel": "C0PROD"})
            res = sp.send("fleet roll-up", transport=rec, start=root)  # no role
            self.assertTrue(res["posted"])
            self.assertIsNone(res["blocked"])

    def test_dry_run_reports_the_refusal_so_a_misconfig_is_visible(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d, _EnvGuard():
            root = _write_env(Path(d), ["FAK_SCOREBOARD_TOKEN=t",
                                        "FAK_DISPATCH_CHANNEL=C0PROD"])
            rec = _Recorder()
            res = sp.send("x", dry_run=True, transport=rec, start=root, role="scout")
            self.assertTrue(res["blocked"])
            self.assertFalse(res["posted"])
            self.assertEqual(rec.calls, [])
            self.assertEqual(sp._render(res), f"slack_post: REFUSED — {res['skipped']}")

    def test_auto_test_backstop_blocks_a_live_send_from_a_harness(self):
        # a test harness that issues a LIVE (transport=None) post to the resolved operator
        # channel is auto-classified "test" and refused BEFORE any network call — proving
        # the backstop needs no explicit role. (The block returns before urllib is used.)
        import tempfile
        with tempfile.TemporaryDirectory() as d, _EnvGuard():
            root = _write_env(Path(d), ["FAK_SCOREBOARD_TOKEN=t",
                                        "FAK_DISPATCH_CHANNEL=C0PROD"])
            res = sp.send("stray live post", start=root)  # transport=None, no role
            self.assertTrue(res["blocked"])
            self.assertEqual(res["role"], "test")


if __name__ == "__main__":
    unittest.main()
