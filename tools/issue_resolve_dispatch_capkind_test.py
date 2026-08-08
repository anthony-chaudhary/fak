#!/usr/bin/env python3
"""Cap-KIND fixtures for tools/issue_resolve_dispatch.py — the Claude 5-hour rolling cap
vs weekly limit question (#5890).

These live in their own file, not in issue_resolve_dispatch_test.py, because they are a
single-purpose corpus: one row per banner phrasing Claude actually emits, asserted at the
two seams that decide fleet behavior — ``_cap_hit_from_text`` (is it seen, and what kind
is it?) and ``_write_cap_hold`` (how long is the seat held?).

The regression they pin: the detector matched "hit your WEEKLY limit" but not "hit your
5-HOUR limit" (a `[\\w\\s]`-only class stops at the hyphen), saw neither of Claude's
"…limit reached." phrasings at all, read only "resets <when>" and not the modern "will
reset at/on <when>", and filed every unqualified cap under a `session` kind whose 90-minute
ceiling truncates a real 5-hour window. Nothing here is hermetic-hostile: the module is
loaded by path and every assertion is pure over an injected clock.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "issue_resolve_dispatch.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("issue_resolve_dispatch_capkind", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# A fixed instant with room on both sides: 2026-05-28 20:26:40 UTC == 13:26 PT, so a "5pm"
# PT reset is ~3.5h out (inside a real 5-hour window, well past the 90-minute session clamp)
# and a "6:10am" PT reset already passed today (the gem8 stale-time-of-day case).
NOW_TS = 1_780_000_000.0
NOW_UTC = dt.datetime(1970, 1, 1) + dt.timedelta(seconds=NOW_TS)


class CapKindClassificationTest(unittest.TestCase):
    """Every banner is SEEN, and is filed under the cap kind it actually names."""

    # (label, banner, expected kind). The first four were undetected outright before #5890.
    CASES = [
        ("5h modern",
         "Claude usage limit reached. Your limit will reset at 5pm (America/Los_Angeles).",
         "rolling_5h"),
        ("5h hit-your-hyphen",
         "You've hit your 5-hour limit. Your limit will reset at 5pm (America/Los_Angeles).",
         "rolling_5h"),
        ("weekly modern",
         "Claude usage limit reached. Your weekly limit will reset on Jun 2 at 9am "
         "(America/Los_Angeles).",
         "weekly"),
        ("weekly for-the-week",
         "You've hit your limit for the week. Try again Jun 2.",
         "weekly"),
        # These three were already detected; they must not regress.
        ("5h resets-form",
         "You've hit your usage limit; resets 5pm (America/Los_Angeles)",
         "rolling_5h"),
        ("weekly hit-your",
         "You've hit your weekly limit. Your limit will reset at Jun 2, 9am "
         "(America/Los_Angeles).",
         "weekly"),
        ("session legacy",
         "You've hit your session limit · resets 6:10am (America/Los_Angeles)",
         "session"),
    ]

    def test_every_banner_is_detected_and_correctly_kinded(self) -> None:
        mod = load()
        for label, banner, want_kind in self.CASES:
            with self.subTest(label):
                hit = mod._cap_hit_from_text(banner)
                self.assertIsNotNone(hit, f"{label}: banner not detected at all")
                self.assertEqual(hit["kind"], want_kind, label)

    def test_hyphen_does_not_hide_the_five_hour_banner(self) -> None:
        """The exact asymmetry that made the fleet see weekly caps and miss 5-hour caps:
        `hit your[\\w\\s]*limit` stops at the hyphen in "5-hour"."""
        mod = load()
        for phrase in ("hit your 5-hour limit", "hit your 5 hour limit",
                       "hit your weekly limit", "hit your usage limit"):
            with self.subTest(phrase):
                self.assertIsNotNone(mod._CAP_BANNER_RE.search(phrase), phrase)

    def test_spawn_header_is_still_not_a_cap_banner(self) -> None:
        """The widened detector must stay phrase-anchored: a worker's spawn header and
        ordinary agent prose that merely mention a limit are NOT cap banners."""
        mod = load()
        for benign in (
            "# fak-spawn issue=42 lane=docs backend=claude",
            "considering the rate limit design for the gateway",
            "the limit of this approach is that it needs a fixture",
        ):
            with self.subTest(benign):
                self.assertIsNone(mod._CAP_BANNER_RE.search(benign), benign)


class CapResetClauseTest(unittest.TestCase):
    """Claude's modern "will reset at/on <when>" clause is read, not dropped."""

    def test_will_reset_clause_is_captured(self) -> None:
        mod = load()
        cases = [
            ("Your limit will reset at 5pm (America/Los_Angeles).", "5pm"),
            ("Your weekly limit will reset on Jun 2 at 9am (America/Los_Angeles).",
             "Jun 2 at 9am"),
            ("Your limit will reset on Jun 2, 9am (America/Los_Angeles).", "Jun 2, 9am"),
        ]
        for text, want in cases:
            with self.subTest(text):
                hit = mod._cap_hit_from_text("You've hit your usage limit. " + text)
                self.assertIsNotNone(hit)
                self.assertEqual(hit["reset_text"], want)

    def test_legacy_resets_clause_still_wins(self) -> None:
        """The pre-existing "resets <when> (America/Los_Angeles)" form is unchanged."""
        mod = load()
        hit = mod._cap_hit_from_text(
            "You've hit your session limit · resets 6:10am (America/Los_Angeles)")
        self.assertEqual(hit["reset_text"], "6:10am")

    def test_announced_wait_still_wins_over_the_reset_clause(self) -> None:
        """#2610's gateway path is untouched: a relative announced_wait is still honored."""
        mod = load()
        hit = mod._cap_hit_from_text(
            "fak-turn trace=guard FAILED reason=rate_limited wire=anthropic_messages "
            "kind=weekly_limit announced_wait=1h7m0s")
        self.assertEqual(hit["kind"], "weekly")
        self.assertEqual(hit["reset_text"], "1h7m0s")


class CapWeekdayResetTest(unittest.TestCase):
    """A weekly cap names its reset by WEEKDAY far more often than by date, and the weekday
    word used to be ignored outright — the bare time-of-day branch resolved it to today or
    tomorrow. NOW_UTC is Thu 2026-05-28 20:26 UTC == Thu 13:26 PT, so the arithmetic below is
    checkable by hand."""

    def test_weekday_resolves_to_the_next_such_day(self) -> None:
        mod = load()
        # Thu 13:26 PT -> next Monday 09:00 PT is 4 days out (Jun 1), not tomorrow.
        got = mod._parse_reset_to_utc("Monday at 9am", NOW_UTC)
        self.assertEqual(got, dt.datetime(2026, 6, 1, 16, 0))

    def test_same_weekday_earlier_in_the_day_rolls_a_full_week(self) -> None:
        """Thursday 9am seen at Thursday 13:26 PT is NEXT Thursday, not four minutes ago
        and not "today" — a past instant would be dropped and fall to the blind fallback."""
        mod = load()
        got = mod._parse_reset_to_utc("Thursday at 9am", NOW_UTC)
        self.assertEqual(got, dt.datetime(2026, 6, 4, 16, 0))

    def test_weekday_under_hold_is_the_one_that_bit(self) -> None:
        """The regression in one number: "Wednesday at 3pm" read on a Thursday used to be
        parsed as TODAY 3pm — a 1.6-hour hold on a cap that has six days left to run."""
        mod = load()
        hit = mod._cap_hit_from_text(
            "You've hit your weekly limit. Your limit will reset on Wednesday at 3pm "
            "(America/Los_Angeles).")
        self.assertEqual(hit["kind"], "weekly")
        until = mod._parse_reset_to_utc(hit["reset_text"], NOW_UTC)
        self.assertGreater((until - NOW_UTC).total_seconds() / 3600.0, 24.0)

    def test_an_explicit_date_still_outranks_a_weekday(self) -> None:
        mod = load()
        # Jun 3 2026 is a WEDNESDAY, so a leading "Mon" token must lose to the date.
        self.assertEqual(mod._parse_reset_to_utc("Mon Jun 3 at 9am", NOW_UTC),
                         dt.datetime(2026, 6, 3, 16, 0))

    def test_a_day_lookalike_word_cannot_manufacture_a_reset(self) -> None:
        """The weekday branch only runs once a time-of-day is present, and the pattern is
        anchored to whole day words — "monitoring" and "summary" are not Monday/Sunday."""
        mod = load()
        self.assertIsNone(mod._parse_reset_to_utc("monitoring the summary", NOW_UTC))
        # A bare time-of-day beside a lookalike word keeps the today/tomorrow branch.
        self.assertEqual(mod._parse_reset_to_utc("monitoring resumes 5pm", NOW_UTC),
                         dt.datetime(2026, 5, 29, 0, 0))


class CapHoldLengthTest(unittest.TestCase):
    """Each cap kind is held for ITS OWN window, not one bucket's ceiling."""

    def _hold_hours(self, mod, hit) -> float:
        with tempfile.TemporaryDirectory() as d:
            out = mod._write_cap_hold(Path(d), product="claude", account_tag="a",
                                      hit=hit, now_ts=NOW_TS, fallback_min=60,
                                      source="test")
        until = dt.datetime.fromisoformat(out["until"].replace("Z", ""))
        return (until - NOW_UTC).total_seconds() / 3600.0

    def test_five_hour_cap_is_not_truncated_to_the_session_clamp(self) -> None:
        """A 5-hour cap naming a reset ~3.5h out is held ~3.5h. Before #5890 the
        `session` bucket's 90-minute ceiling cut it to 1.5h and the dispatcher respawned
        into the same wall two hours early."""
        mod = load()
        hit = mod._cap_hit_from_text(
            "You've hit your 5-hour limit. Your limit will reset at 5pm (America/Los_Angeles).")
        held = self._hold_hours(mod, hit)
        self.assertGreater(held, mod._SESSION_HOLD_MAX_MIN / 60.0,
                           "a real 5-hour reset must outlive the session clamp")
        self.assertLessEqual(held, mod._ROLLING_HOLD_MAX_MIN / 60.0)

    def test_rolling_cap_ceiling_still_bounds_a_stale_time_of_day(self) -> None:
        """The clamp's original motive survives: a bare time-of-day that already passed
        today must not become a ~24h wall, only a ~5h one."""
        mod = load()
        hit = {"kind": "rolling_5h", "reset_text": "6:10am"}
        self.assertLessEqual(self._hold_hours(mod, hit),
                             mod._ROLLING_HOLD_MAX_MIN / 60.0 + 0.01)

    def test_session_limit_hold_is_still_bounded_to_ninety_minutes(self) -> None:
        """The gem8 false-cap guard is untouched for the kind it was written for."""
        mod = load()
        hit = mod._cap_hit_from_text(
            "You've hit your session limit · resets 6:10am (America/Los_Angeles)")
        self.assertEqual(hit["kind"], "session")
        self.assertLessEqual(self._hold_hours(mod, hit),
                             mod._SESSION_HOLD_MAX_MIN / 60.0 + 0.02)

    def test_unannounced_weekly_cap_outlasts_the_rolling_fallback(self) -> None:
        """A weekly cap whose banner names no parseable reset must not take the short
        fallback: its real window is days, so a 1-hour hold re-admits the seat ~168 times
        before it can serve."""
        mod = load()
        hit = mod._cap_hit_from_text("You've hit your limit for the week.")
        self.assertEqual(hit["kind"], "weekly")
        self.assertAlmostEqual(self._hold_hours(mod, hit),
                               mod._WEEKLY_FALLBACK_MIN / 60.0, places=2)

    def test_dated_weekly_cap_keeps_its_full_reset(self) -> None:
        """A weekly cap naming a real dated reset is held to that reset — no kind ceiling
        applies to weekly."""
        mod = load()
        hit = mod._cap_hit_from_text(
            "You've hit your weekly limit. Your limit will reset at Jun 2, 9am "
            "(America/Los_Angeles).")
        self.assertEqual(hit["kind"], "weekly")
        # Jun 2 09:00 PDT == Jun 2 16:00 UTC, ~4.8 days past NOW_UTC (May 28 20:26 UTC).
        self.assertGreater(self._hold_hours(mod, hit), 24.0)


if __name__ == "__main__":
    unittest.main()
