#!/usr/bin/env python3
"""Hermetic tests for tools/dos_supervisor_canary.py."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "dos_supervisor_canary.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("dos_supervisor_canary", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def readiness() -> dict:
    return {
        "schema": "fleet-dos-supervisor-status/1",
        "ok": True,
        "verdict": "READY_TO_CANARY",
        "why": "test readiness",
        "next_action": "test next",
        "supervise": {
            "target": 3,
            "alive": 0,
            "spawn": ["adjudicator"],
            "reap": [],
            "flag": [],
        },
        "plans": {
            "total_plans": 1,
            "total_units": 5,
            "done_units": 0,
            "drift_count": 0,
        },
    }


def clean_safety() -> dict:
    return {"ok": True, "blockers": [], "git": {"dirty": False, "relation": "in_sync"}}


class DosSupervisorCanaryTest(unittest.TestCase):
    def test_dry_run_does_not_call_runner(self) -> None:
        mod = load()

        def fail_runner(_cmd, _cwd, _timeout_s):
            raise AssertionError("dry run must not call runner")

        got = mod.run_canary(
            workspace=Path("C:/work/fleet"),
            live=False,
            target=1,
            max_ticks=1,
            timeout_s=120,
            readiness=readiness(),
            safety=clean_safety(),
            runner=fail_runner,
        )

        self.assertTrue(got["ok"])
        self.assertEqual(got["action"], "dry_run")
        self.assertFalse(got["live"])
        self.assertIsNone(got["post_audit"])
        self.assertEqual(got["launch"]["action"], "would_enact")

    def test_live_refusal_keeps_post_audit_and_does_not_call_runner(self) -> None:
        mod = load()
        dirty = {
            "ok": False,
            "blockers": [{"kind": "dirty", "detail": "worktree has 1 dirty path(s)"}],
            "git": {"dirty": True, "dirty_count": 1},
        }

        def fail_runner(_cmd, _cwd, _timeout_s):
            raise AssertionError("refused live run must not call runner")

        got = mod.run_canary(
            workspace=Path("C:/work/fleet"),
            live=True,
            target=1,
            max_ticks=1,
            timeout_s=120,
            readiness=readiness(),
            safety=dirty,
            runner=fail_runner,
            post_audit={"ok": False, "verdict": "BLOCKED", "finding": "typed_blocker", "reason": "dirty"},
        )

        self.assertFalse(got["ok"])
        self.assertEqual(got["action"], "live_refused")
        self.assertEqual(got["launch"]["action"], "refuse")
        self.assertEqual(got["post_audit"]["verdict"], "BLOCKED")

    def test_live_success_calls_runner_and_returns_post_audit(self) -> None:
        mod = load()
        calls = []

        def runner(cmd, cwd, timeout_s):
            calls.append((cmd, cwd, timeout_s))
            return {"returncode": 0, "stdout": "{}", "stderr": ""}

        post = {
            "ok": True,
            "verdict": "CANARY_OBSERVED",
            "finding": "live_worker_observed",
            "reason": "1 live worker counted",
            "next_action": "inspect worker liveness",
        }
        got = mod.run_canary(
            workspace=Path("C:/work/fleet"),
            live=True,
            target=1,
            max_ticks=1,
            timeout_s=17,
            readiness=readiness(),
            safety=clean_safety(),
            runner=runner,
            post_audit=post,
        )

        self.assertTrue(got["ok"])
        self.assertEqual(got["action"], "live_enacted")
        self.assertTrue(got["live"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(got["post_audit"]["verdict"], "CANARY_OBSERVED")

    def test_record_payload_writes_compact_history_and_latest(self) -> None:
        mod = load()
        got = mod.run_canary(
            workspace=Path("C:/work/fleet"),
            live=False,
            target=1,
            max_ticks=1,
            timeout_s=120,
            readiness=readiness(),
            safety=clean_safety(),
        )

        with tempfile.TemporaryDirectory() as td:
            record = mod.record_payload(
                got,
                workspace=Path("C:/work/fleet"),
                record_dir=Path(td),
                now_utc="2026-06-19T00:00:00Z",
            )
            history_path = Path(record["history_path"])
            latest_path = Path(record["latest_path"])

            rows = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()]
            latest = json.loads(latest_path.read_text(encoding="utf-8"))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["action"], "dry_run")
            self.assertFalse(rows[0]["live"])
            self.assertEqual(rows[0]["audit"]["verdict"], "PRE_CANARY_READY")
            self.assertEqual(latest["event"]["recorded_utc"], "2026-06-19T00:00:00Z")
            self.assertEqual(record["history"]["verdict"], "HISTORY_OK")

    def test_history_summary_reports_recurring_blocker_with_route_command(self) -> None:
        mod = load()
        event = {
            "schema": mod.RECORD_SCHEMA,
            "recorded_utc": "2026-06-19T00:00:00Z",
            "workspace": "C:/work/fleet",
            "ok": False,
            "action": "dry_run",
            "live": False,
            "reason": "worktree has 1 dirty path(s)",
            "next_action": "commit or stash",
            "audit": {"verdict": "BLOCKED", "finding": "typed_blocker"},
            "launch": {"action": "would_enact"},
            "blockers": ["workspace_safety:dirty"],
        }

        with tempfile.TemporaryDirectory() as td:
            history_path = Path(td) / "history.jsonl"
            history_path.write_text(
                json.dumps(event) + "\n" + json.dumps({**event, "recorded_utc": "2026-06-19T00:01:00Z"}) + "\n",
                encoding="utf-8",
            )

            got = mod.summarize_history(Path("C:/work/fleet"), record_dir=Path(td))

            self.assertFalse(got["ok"])
            self.assertEqual(got["verdict"], "RECURRING_BLOCKER")
            self.assertEqual(got["recurring_blockers"][0]["blocker"], "workspace_safety:dirty")
            self.assertIn("--history --json", got["recurring_blockers"][0]["command"])

    def test_history_without_records_is_ok_and_names_record_command(self) -> None:
        mod = load()

        with tempfile.TemporaryDirectory() as td:
            got = mod.summarize_history(Path("C:/work/fleet"), record_dir=Path(td))

            self.assertTrue(got["ok"])
            self.assertEqual(got["verdict"], "NO_HISTORY")
            self.assertIn("--record --json", got["next_action"])


if __name__ == "__main__":
    unittest.main()
