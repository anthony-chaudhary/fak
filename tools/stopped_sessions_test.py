#!/usr/bin/env python3
"""Tests for stopped_sessions.py — deciding which stopped sessions may be resumed.

This tool's output drives an automated resume: a session it calls
``STOPPED_MIDTOOL`` gets relaunched, and one it calls ``STOPPED_LIMIT`` does not.
Misreading a transcript therefore either wastes a throttled account's remaining
budget or strands a session that could have finished, so what is pinned here is
the DISPOSITION LADDER and specifically its priority order:

* a live-looking mtime does NOT make a throttled or auth-blocked session
  resumable — those two are checked before the liveness window;
* an ``assistant`` tool_use with no following ``tool_result`` is the mid-tool
  signal, and a ``user`` turn carrying a tool_result CLEARS it. If the pairing
  ever stopped clearing, every completed session would read as mid-tool and be
  relaunched;
* a throttle banner seen earlier in the transcript but not last is remembered as
  ``throttle_seen`` yet must NOT set ``throttle_current`` — the account recovered.

One behaviour recorded here deliberately differs from the module docstring: the
docstring lists "Login interrupted" under `interrupt`, but ``is_auth_error``
matches that phrase and auth is tested first, so such a session is classified
``STOPPED_AUTH`` (deferred) rather than ``STOPPED_INTERRUPT`` (resumed). The test
pins the real behaviour; deferring a login-interrupted session is the safe side.

Every transcript is synthetic and written to a temp directory; no real account
directory is read and ``main()`` (which walks the live home directory) is not
exercised.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# stopped_sessions.py reads its window from sys.argv AT IMPORT TIME, so import it
# under a bare argv; otherwise a unittest flag like -v would be parsed as hours.
_argv = sys.argv[:]
sys.argv = sys.argv[:1]
try:
    import stopped_sessions as M  # noqa: E402
finally:
    sys.argv = _argv


def assistant(text=None, blocks=None, model=None, ts="2026-07-28T00:00:00Z"):
    msg = {"role": "assistant",
           "content": blocks if blocks is not None else [{"type": "text", "text": text or ""}]}
    if model:
        msg["model"] = model
    return {"type": "assistant", "message": msg, "timestamp": ts}


def user(text=None, blocks=None, ts="2026-07-28T00:00:00Z"):
    msg = {"role": "user",
           "content": blocks if blocks is not None else [{"type": "text", "text": text or ""}]}
    return {"type": "user", "message": msg, "timestamp": ts}


def tool_use(name="Bash"):
    return {"type": "tool_use", "name": name, "id": "tu_1", "input": {}}


def tool_result(text="ok"):
    return {"type": "tool_result", "tool_use_id": "tu_1", "content": text}


LIMIT_BANNER = ("Claude AI usage limit reached. Your limit . resets 9pm "
                "(America/Los_Angeles)")


class Session(unittest.TestCase):
    """Base class: write a synthetic transcript and classify it."""

    def transcript(self, records, age_min=60.0, name="0f8fbb4c-1111-2222-3333-444455556666"):
        path = Path(tempfile.mkdtemp()) / f"{name}.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
        stamp = M.NOW.timestamp() - age_min * 60.0
        os.utime(path, (stamp, stamp))
        return str(path)

    def disp(self, records, age_min=60.0):
        return M.classify(self.transcript(records, age_min))["disp"]


class TextOf(unittest.TestCase):
    def test_a_plain_string_is_returned_unchanged(self):
        self.assertEqual(M.text_of("hello"), "hello")

    def test_text_blocks_are_joined_in_order(self):
        blocks = [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]
        self.assertEqual(M.text_of(blocks), "one two")

    def test_a_tool_result_contributes_its_text(self):
        self.assertEqual(M.text_of([tool_result("stdout here")]), "stdout here")

    def test_a_nested_tool_result_payload_is_flattened_recursively(self):
        nested = [{"type": "tool_result", "content": [{"type": "text", "text": "deep"}]}]
        self.assertEqual(M.text_of(nested), "deep")

    def test_a_tool_use_block_contributes_nothing(self):
        self.assertEqual(M.text_of([tool_use(), {"type": "text", "text": "kept"}]), "kept")

    def test_an_unrecognised_shape_yields_the_empty_string(self):
        self.assertEqual(M.text_of(None), "")
        self.assertEqual(M.text_of({"type": "text"}), "")


class LastToolUseName(unittest.TestCase):
    def test_the_last_tool_use_in_the_turn_wins(self):
        self.assertEqual(M.last_tooluse_name([tool_use("Read"), tool_use("Bash")]), "Bash")

    def test_a_turn_with_no_tool_use_yields_none(self):
        self.assertIsNone(M.last_tooluse_name([{"type": "text", "text": "hi"}]))

    def test_a_non_list_content_yields_none(self):
        self.assertIsNone(M.last_tooluse_name("plain text turn"))


class ReadAndParse(Session):
    def test_only_the_tail_is_read_so_a_huge_transcript_stays_cheap(self):
        path = self.transcript([user(text="x" * 400), user(text="tail marker")])
        tail = "".join(M.read_lines(path, tail_bytes=200))
        self.assertNotIn("x" * 400, tail)              # the head was skipped
        self.assertIn("tail marker", tail)             # the newest turn survived
        self.assertIn("x" * 400, "".join(M.read_lines(path)))   # full read has both

    def test_a_missing_file_reads_as_empty_rather_than_raising(self):
        self.assertEqual(M.read_lines(str(Path(tempfile.mkdtemp()) / "nope.jsonl")), [])

    def test_blank_and_truncated_lines_are_skipped_not_fatal(self):
        path = Path(tempfile.mkdtemp()) / "s.jsonl"
        path.write_text('{"type": "user"}\n\n{"type": "assis\n{"type": "summary"}\n',
                        encoding="utf-8")
        got = M.parse(str(path))
        self.assertEqual(got, [{"type": "user"}, {"type": "summary"}])


class DispositionLadder(Session):
    def test_a_recently_appended_transcript_reads_as_live(self):
        self.assertEqual(self.disp([assistant(text="thinking about it")], age_min=1.0),
                         "LIVE")

    def test_a_transcript_older_than_the_liveness_window_is_not_live(self):
        self.assertNotEqual(self.disp([assistant(text="thinking about it")], age_min=10.0),
                            "LIVE")

    def test_a_throttle_banner_outranks_a_live_looking_mtime(self):
        # Resuming here would burn the account's remaining budget immediately.
        records = [assistant(text=LIMIT_BANNER, model="<synthetic>")]
        self.assertEqual(self.disp(records, age_min=0.5), "STOPPED_LIMIT")

    def test_an_auth_wall_outranks_a_live_looking_mtime(self):
        records = [assistant(text="API Error: 401 authentication_error")]
        self.assertEqual(self.disp(records, age_min=0.5), "STOPPED_AUTH")

    def test_an_unanswered_tool_use_means_the_process_died_mid_work(self):
        records = [user(text="go"), assistant(blocks=[tool_use("Bash")])]
        self.assertEqual(self.disp(records), "STOPPED_MIDTOOL")

    def test_a_following_tool_result_clears_the_mid_tool_signal(self):
        records = [assistant(blocks=[tool_use("Bash")]), user(blocks=[tool_result()])]
        self.assertNotEqual(self.disp(records), "STOPPED_MIDTOOL")

    def test_a_user_interrupt_is_recognised(self):
        records = [assistant(text="[Request interrupted by user] stopping now")]
        self.assertEqual(self.disp(records), "STOPPED_INTERRUPT")

    def test_a_login_interruption_defers_as_auth_rather_than_resuming(self):
        # Diverges from the module docstring's `interrupt` bullet on purpose:
        # is_auth_error() matches "Login interrupted" and auth is tested first.
        records = [assistant(text="Login interrupted; please run /login")]
        self.assertEqual(self.disp(records), "STOPPED_AUTH")

    def test_a_session_parked_on_a_background_task_is_not_resumed(self):
        records = [assistant(text="The suite is still running; I will wait for the "
                                  "harness to notify me when it completes.")]
        self.assertEqual(self.disp(records), "PARKED_WAIT")

    def test_a_wrap_up_reads_as_done(self):
        records = [assistant(text="Done. Committed and pushed the fix.")]
        self.assertEqual(self.disp(records), "DONE")

    def test_anything_else_falls_through_to_quiet(self):
        records = [assistant(text="The value of n is 42.")]
        self.assertEqual(self.disp(records), "STOPPED_QUIET")

    def test_mid_tool_is_only_consulted_after_the_interrupt_check(self):
        # Both signals present: the interrupt is the more specific explanation.
        records = [assistant(blocks=[tool_use("Bash")]),
                   assistant(text="[Request interrupted by user")]
        self.assertEqual(self.disp(records), "STOPPED_INTERRUPT")


class ThrottleBookkeeping(Session):
    def test_a_trailing_banner_is_both_seen_and_current(self):
        row = M.classify(self.transcript(
            [user(text="go"), assistant(text=LIMIT_BANNER, model="<synthetic>")]))
        self.assertTrue(row["throttle_current"])
        self.assertTrue(row["throttle_seen"])
        self.assertEqual(row["throttle_reset"], row["throttle_seen"])

    def test_a_banner_the_session_recovered_from_is_seen_but_not_current(self):
        # The account was throttled, waited it out, and kept working. Reporting
        # this as currently throttled would strand a resumable session.
        row = M.classify(self.transcript([
            assistant(text=LIMIT_BANNER, model="<synthetic>"),
            assistant(blocks=[tool_use("Bash")]),
        ]))
        self.assertTrue(row["throttle_seen"])
        self.assertFalse(row["throttle_current"])
        self.assertIsNone(row["throttle_reset"])
        self.assertEqual(row["disp"], "STOPPED_MIDTOOL")

    def test_a_banner_without_a_synthetic_model_is_not_a_throttle(self):
        # Only the harness's own <synthetic> record is authoritative; an agent
        # quoting the banner text must not throttle the account.
        row = M.classify(self.transcript([assistant(text=LIMIT_BANNER)]))
        self.assertFalse(row["throttle_current"])


class RowShape(Session):
    def test_session_metadata_is_carried_from_the_transcript(self):
        records = [dict(assistant(text="hi"), cwd="C:/work/fak", gitBranch="main",
                        version="2.1.5", sessionId="sid-123")]
        row = M.classify(self.transcript(records))
        self.assertEqual(row["cwd"], "C:/work/fak")
        self.assertEqual(row["git"], "main")
        self.assertEqual(row["version"], "2.1.5")
        self.assertEqual(row["session"], "sid-123")

    def test_the_session_id_falls_back_to_the_transcript_filename(self):
        name = "0f8fbb4c-1111-2222-3333-444455556666"
        row = M.classify(self.transcript([assistant(text="hi")], name=name))
        self.assertEqual(row["session"], name)

    def test_non_conversational_records_do_not_become_the_last_turn(self):
        records = [assistant(text="the real last turn"),
                   {"type": "summary", "summary": "a generated title"}]
        row = M.classify(self.transcript(records))
        self.assertEqual(row["last_role"], "assistant")
        self.assertIn("the real last turn", row["last"])

    def test_the_last_turn_excerpt_is_bounded_and_single_line(self):
        row = M.classify(self.transcript([assistant(text="a\nb " + "x" * 500)]))
        self.assertLessEqual(len(row["last"]), 300)
        self.assertNotIn("\n", row["last"])

    def test_the_row_reports_its_own_age_and_path(self):
        path = self.transcript([assistant(text="hi")], age_min=42.0)
        row = M.classify(path)
        self.assertAlmostEqual(row["age_min"], 42.0, delta=0.5)
        self.assertEqual(row["path"], path)
        self.assertTrue(row["seen_utc"].endswith("+00:00"))


class Constants(unittest.TestCase):
    def test_the_session_filename_pattern_is_a_uuid(self):
        self.assertTrue(M.UUID_RE.match("0f8fbb4c-1111-2222-3333-444455556666"))
        self.assertFalse(M.UUID_RE.match("not-a-uuid"))
        self.assertFalse(M.UUID_RE.match("0f8fbb4c-1111-2222-3333-44445555666"))

    def test_the_liveness_window_is_a_few_minutes_not_hours(self):
        # Too wide and every stopped session reads as live and is never resumed.
        self.assertGreater(M.LIVE_MIN, 0)
        self.assertLessEqual(M.LIVE_MIN, 15)

    def test_generated_record_types_are_excluded_from_the_conversation(self):
        for kind in ("summary", "ai-title", "file-history-snapshot", "system"):
            self.assertIn(kind, M.META_TYPES)


if __name__ == "__main__":
    unittest.main()
