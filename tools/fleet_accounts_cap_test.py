#!/usr/bin/env python3
"""Unit tests for tools/fleet_accounts_cap.py."""
from __future__ import annotations

import datetime as dt
import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_TOOLS_DIR = str(Path(__file__).resolve().parent)


def _cap():
    if _TOOLS_DIR not in sys.path:
        sys.path.insert(0, _TOOLS_DIR)
    return importlib.import_module("fleet_accounts_cap")


class FleetAccountsCapTest(unittest.TestCase):
    def test_product_and_tag_classification(self) -> None:
        cap = _cap()
        self.assertEqual(cap.account_product(".claude-gem8-acct"), "claude")
        self.assertEqual(cap.account_product("opencode-glm"), "opencode")
        self.assertEqual(cap.account_tag(".claude"), "default")
        self.assertEqual(cap.account_tag(".claude-gem8-acct"), "gem8")
        self.assertEqual(cap.account_tag("opencode"), "default")
        self.assertEqual(cap.account_tag("opencode-glm"), "glm")

    def test_session_cap_env_override_and_fallback(self) -> None:
        cap = _cap()
        saved = os.environ.get(cap.SESSIONS_PER_ACCOUNT_ENV)
        try:
            os.environ[cap.SESSIONS_PER_ACCOUNT_ENV] = "7"
            self.assertEqual(cap._session_cap({"product": "claude"}), 7)
            os.environ[cap.SESSIONS_PER_ACCOUNT_ENV] = "bad"
            self.assertEqual(cap._session_cap({"product": "claude"}), 4)
            self.assertEqual(cap._session_cap({"product": "codex"}), 20)
            self.assertEqual(cap._session_cap({"product": "opencode"}), 1)
        finally:
            if saved is None:
                os.environ.pop(cap.SESSIONS_PER_ACCOUNT_ENV, None)
            else:
                os.environ[cap.SESSIONS_PER_ACCOUNT_ENV] = saved

    def test_reset_is_future_dated_and_bare_daily_window(self) -> None:
        cap = _cap()
        now = dt.datetime(2026, 6, 25, 14, 0, tzinfo=dt.timezone.utc)
        self.assertTrue(cap._reset_is_future("Jun 25, 4pm", now=now))
        self.assertFalse(cap._reset_is_future("Jun 24, 4pm", now=now))
        self.assertFalse(cap._reset_is_future("12:30am", now=now))
        self.assertTrue(cap._reset_is_future("5pm", now=now))
        late_night = dt.datetime(2026, 6, 25, 22, 0, tzinfo=dt.timezone.utc)
        self.assertTrue(cap._reset_is_future("1am", now=late_night))
        self.assertIsNone(cap._reset_is_future("later", now=now))
        self.assertIsNotNone(cap._reset_is_future("3pm (America/Los_Angeles)", now=now))

    def test_disambiguate_cap_aging_and_probe_override(self) -> None:
        cap = _cap()
        now = dt.datetime(2026, 6, 25, 14, 0, tzinfo=dt.timezone.utc)
        info = {"reset": "11am", "weekly": "Jun 30, 1pm"}
        base = cap._disambiguate_cap(info, now=now)
        self.assertTrue(base.active)
        self.assertTrue(base.weekly_active)

        aged_obs = cap._CapObservation(
            first_seen=now - dt.timedelta(days=8),
            ok_streak=0,
        )
        aged = cap._disambiguate_cap(info, obs=aged_obs, now=now)
        self.assertTrue(aged.aged_out)
        self.assertFalse(aged.weekly_active)
        self.assertFalse(aged.active)

        override_obs = cap._CapObservation(first_seen=None, ok_streak=2)
        overridden = cap._disambiguate_cap(info, obs=override_obs, now=now)
        self.assertEqual(overridden.overridden_by, 2)
        self.assertFalse(overridden.active)
        self.assertTrue(cap.throttle_is_active("5pm", now=now))
        self.assertTrue(cap._weekly_throttle_is_active(info, now=now))

    def test_cap_observation_from_ledger_tail(self) -> None:
        cap = _cap()
        entries = [
            {"account": ".claude-a", "status": "LIMIT", "ts": "2026-06-20T00:00:00Z"},
            {"account": ".claude-a", "status": "LIMIT", "ts": "2026-06-21T00:00:00Z"},
            {"account": ".claude-a", "status": "OK", "ts": "2026-06-22T00:00:00Z"},
            {"account": ".claude-a", "status": "OK", "ts": "2026-06-23T00:00:00Z"},
        ]
        by_acct = cap._derive_cap_observations(entries)
        self.assertEqual(by_acct[".claude-a"].ok_streak, 2)
        self.assertIsNone(by_acct[".claude-a"].first_seen)

        blocked_obs = cap._cap_observation_from([
            {"status": "OK", "ts": "2026-06-20T00:00:00Z"},
            {"status": "LIMIT", "ts": "2026-06-21T01:00:00Z"},
            {"status": "AUTH", "ts": "2026-06-22T02:00:00Z"},
        ])
        self.assertEqual(blocked_obs.ok_streak, 0)
        self.assertEqual(blocked_obs.first_seen, dt.datetime(2026, 6, 21, 1, 0, tzinfo=dt.timezone.utc))

    def test_active_account_cap_throttles_sidecars(self) -> None:
        cap = _cap()
        now = dt.datetime(2026, 6, 25, 14, 0, tzinfo=dt.timezone.utc)
        rows = [
            {"kind": "worker", "product": "claude", "account": ".claude-a", "tag": "a"},
            {"kind": "worker", "product": "claude", "account": ".claude-b", "tag": "b"},
        ]
        with tempfile.TemporaryDirectory() as td:
            Path(td, "account-cap-a.json").write_text(json.dumps({
                "product": "claude",
                "account": "a",
                "kind": "weekly",
                "until": "2026-06-26T12:00:00Z",
                "detected": "2026-06-25T13:00:00Z",
                "reset_text": "Jun 26, 12pm",
            }), encoding="utf-8")
            Path(td, "account-cap-b.json").write_text(json.dumps({
                "product": "claude",
                "account": "b",
                "until": "2026-06-25T13:00:00Z",
            }), encoding="utf-8")
            Path(td, "account-cap-bad.json").write_text("{broken", encoding="utf-8")
            holds = cap.active_account_cap_throttles(rows, runs_dir=td, now=now)
        self.assertEqual(set(holds), {".claude-a"})
        self.assertEqual(holds[".claude-a"]["weekly"], "2026-06-26T12:00:00Z")

    def test_oauth_token_and_routable_worker_predicate(self) -> None:
        cap = _cap()
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(cap.read_oauth_token(td))
            Path(td, ".oauth-token").write_text(" tok-123 \n", encoding="utf-8")
            self.assertEqual(cap.read_oauth_token(td), "tok-123")
        self.assertTrue(cap.routable_worker({"kind": "worker", "identity_role": "canonical"}))
        self.assertFalse(cap.routable_worker({"kind": "worker", "identity_role": "duplicate"}))
        self.assertFalse(cap.routable_worker({"kind": "excluded"}))


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FleetAccountsCapTest)
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if res.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
