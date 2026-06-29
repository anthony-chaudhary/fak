#!/usr/bin/env python3
"""Hermetic tests for tools/dispatch_status.py.

build_payload() is a pure FOLD over five already-collected sub-tool dicts
(preflight, supervisor, watchdog, backlog, closure). We feed it synthetic dicts
and assert the overall verdict, the watchdog reason line, and the backlog/closure
"na" degradation — no subprocess, no gh, no schtasks. render() is exercised on a
minimal payload to prove it does not raise.
"""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "dispatch_status.py"


def load():
    spec = importlib.util.spec_from_file_location("dispatch_status", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pre(verdict: str = "SPAWN_OK", *, host_safe: bool = True, cap: int = 2,
        live: int = 0) -> dict:
    return {
        "verdict": verdict,
        "reason": f"synthetic {verdict}",
        "cap": cap,
        "live": live,
        "host": {"safe": host_safe},
        "account": {"tag": "worker-a", "tier": 1, "model": "claude", "available": True},
    }


def sup(verdict: str = "READY_TO_CANARY") -> dict:
    return {
        "verdict": verdict,
        "supervise": {"target": 3, "alive": 1},
        "plans": {"total_plans": 2, "total_units": 17},
    }


def backlog_ok() -> dict:
    return {
        "lanes": {"docs": {"issues": [1, 2, 3]}, "agent": {"issues": [4]}},
        "counts": {"open": 4, "routed": 4, "unrouted": 0},
    }


def closure_ok() -> dict:
    return {
        "closure_rate": 0.8,
        "counts": {"TRUE_RESOLVED": 8, "CLAIMED_CLOSED": 10, "OPEN_WITNESSED": 2},
    }


def build(mod, **over):
    kw = dict(
        root=ROOT, pre=pre(), sup=sup(), wd={"installed": True, "status": "Ready"},
        backlog=backlog_ok(), closure=closure_ok(), max_workers=2, fast=False)
    kw.update(over)
    return mod.build_payload(**kw)


class VerdictTest(unittest.TestCase):
    def test_ready_to_grow_when_safe_to_spawn(self) -> None:
        mod = load()
        p = build(mod, pre=pre("SPAWN_OK"))
        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], "READY_TO_GROW")
        self.assertEqual(p["dispatcher"]["headroom"], 2)

    def test_host_flagged_fails_the_card(self) -> None:
        mod = load()
        p = build(mod, pre=pre("SPAWN_OK", host_safe=False))
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "HOST_FLAGGED")
        self.assertTrue(any("host resource guard flagged" in r for r in p["reasons"]))

    def test_at_cap_is_a_healthy_steady_state(self) -> None:
        mod = load()
        p = build(mod, pre=pre("REFUSE_AT_CAP", cap=2, live=2))
        self.assertTrue(p["ok"])  # at cap is normal, not breakage
        self.assertEqual(p["verdict"], "AT_CAP")
        self.assertEqual(p["dispatcher"]["headroom"], 0)

    def test_blocked_on_account_is_a_healthy_steady_state(self) -> None:
        mod = load()
        p = build(mod, pre=pre("REFUSE_NO_ACCOUNT"))
        self.assertTrue(p["ok"])
        self.assertEqual(p["verdict"], "BLOCKED_ON_ACCOUNT")

    def test_inspect_fails_the_card(self) -> None:
        mod = load()
        p = build(mod, pre=pre("REFUSE_INSPECT"))
        self.assertFalse(p["ok"])
        self.assertEqual(p["verdict"], "INSPECT")


class BacklogClosureNaTest(unittest.TestCase):
    def test_backlog_na_on_skipped_and_closure_na_on_skipped(self) -> None:
        mod = load()
        p = build(mod, backlog={"_skipped": "fast"}, closure={"_skipped": "fast"}, fast=True)
        self.assertTrue(p["backlog"]["na"])
        self.assertIsNone(p["backlog"]["open_issues"])
        self.assertTrue(p["closure"]["na"])
        self.assertIsNone(p["closure"]["closure_rate"])

    def test_backlog_na_on_error_with_no_lanes(self) -> None:
        mod = load()
        p = build(mod, backlog={"_error": "gh timed out"})
        self.assertTrue(p["backlog"]["na"])

    def test_backlog_present_folds_lane_counts(self) -> None:
        mod = load()
        p = build(mod)
        self.assertFalse(p["backlog"]["na"])
        self.assertEqual(p["backlog"]["open_issues"], 4)
        self.assertEqual(p["backlog"]["by_lane"], {"docs": 3, "agent": 1})
        self.assertEqual(p["backlog"]["routed"], 4)

    def test_closure_present_surfaces_rate_and_open_witnessed(self) -> None:
        mod = load()
        p = build(mod)
        self.assertFalse(p["closure"]["na"])
        self.assertEqual(p["closure"]["closure_rate"], 0.8)
        self.assertEqual(p["closure"]["open_witnessed_closable"], 2)


class WatchdogReasonTest(unittest.TestCase):
    def test_watchdog_installed_reason_line(self) -> None:
        mod = load()
        p = build(mod, wd={"installed": True, "status": "Ready"})
        self.assertTrue(any("watchdog installed (Ready)" in r for r in p["reasons"]))
        self.assertEqual(p["dispatcher"]["watchdog"]["installed"], True)

    def test_watchdog_not_installed_reason_line(self) -> None:
        mod = load()
        p = build(mod, wd={"installed": False, "status": None})
        self.assertTrue(any("watchdog NOT installed" in r for r in p["reasons"]))

    def test_watchdog_unknown_emits_no_install_line(self) -> None:
        mod = load()
        # installed is None (schtasks couldn't run) -> neither install line appears.
        p = build(mod, wd={"installed": None, "error": "schtasks missing"})
        self.assertFalse(any("watchdog" in r for r in p["reasons"]))


class RunStatusDigestTest(unittest.TestCase):
    def test_loop_ledger_run_ids_are_recent_rids_only(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            ledger = Path(d) / "loops.jsonl"
            ledger.write_text("\n".join([
                '{"loop_id":"issue-resolve-dispatch/claude","run_id":"legacy-1"}',
                '{"loop_id":"other","run_id":"RID-OTHER1"}',
                '{"loop_id":"issue-resolve-progress","run_id":"RID-PROGRESS1"}',
                '{"loop_id":"issue-resolve-dispatch/codex","run_id":"RID-DISPATCH1"}',
                '{"loop_id":"issue-resolve-dispatch/codex","run_id":"RID-DISPATCH1"}',
            ]) + "\n", encoding="utf-8")
            self.assertEqual(
                mod.run_ids_from_loop_ledger(ledger),
                ["RID-DISPATCH1", "RID-PROGRESS1"])

    def test_claimed_key_detector_is_recursive(self) -> None:
        mod = load()
        self.assertTrue(mod.has_key_named({"liveness": [{"claimed": "done"}]}, "claimed"))
        self.assertFalse(mod.has_key_named({"liveness": [{"verdict": "STALLED"}]}, "claimed"))

    def test_build_payload_summarizes_dos_status_digests(self) -> None:
        mod = load()
        p = build(mod, run_status=[
            {"run_id": "RID-DISPATCH1", "liveness": {"verdict": "ADVANCING"}},
            {"run_id": "RID-PROGRESS1", "_error": "dos unavailable"},
        ])
        self.assertEqual(p["run_status"]["source"], "dos status")
        self.assertEqual(p["run_status"]["count"], 2)
        self.assertEqual(p["run_status"]["liveness"], {"ADVANCING": 1})
        self.assertEqual(p["run_status"]["errors"], 1)
        self.assertTrue(any("dos status digest" in r for r in p["reasons"]))
        self.assertIn("run truth", mod.render(p))


class RenderTest(unittest.TestCase):
    def test_render_does_not_raise_on_minimal_payload(self) -> None:
        mod = load()
        p = build(mod)
        text = mod.render(p)
        self.assertIn("DISPATCHER", text)
        self.assertIn("READY_TO_GROW", text)

    def test_render_does_not_raise_on_na_payload(self) -> None:
        mod = load()
        p = build(mod, backlog={"_skipped": "fast"}, closure={"_skipped": "fast"}, fast=True)
        text = mod.render(p)
        self.assertIn("n/a", text)

    def test_render_surfaces_silent_workers_line(self) -> None:
        mod = load()
        p = build(mod, silent=[{"issue": 465, "stamp": "20260621-232003",
                                "log": "resolve-465-20260621-232003.log", "pid": 39688}])
        text = mod.render(p)
        self.assertIn("1 silent", text)
        self.assertIn("#465", text)


class SilentWorkersFoldTest(unittest.TestCase):
    """build_payload folds the injected silent list into payload['workers'] and a reason."""

    def test_silent_workers_fold_and_reason(self) -> None:
        mod = load()
        silent = [{"issue": 465, "stamp": "20260621-232003",
                   "log": "resolve-465-20260621-232003.log", "pid": 39688}]
        p = build(mod, silent=silent)
        self.assertEqual(p["workers"]["silent_count"], 1)
        self.assertEqual(p["workers"]["silent"], silent)
        self.assertTrue(any("exited producing nothing" in r for r in p["reasons"]))

    def test_no_silent_workers_emits_no_reason(self) -> None:
        mod = load()
        p = build(mod, silent=[])
        self.assertEqual(p["workers"]["silent_count"], 0)
        self.assertFalse(any("producing nothing" in r for r in p["reasons"]))

    def test_silent_defaults_to_empty_when_omitted(self) -> None:
        mod = load()
        p = build(mod)  # build() does not pass silent -> defaults to None -> []
        self.assertEqual(p["workers"]["silent_count"], 0)


class SilentWorkersScanTest(unittest.TestCase):
    """silent_workers() classification — hermetic: a tmp runs-dir + injected alive set."""

    def _mk(self, runs: Path, issue: int, stamp: str, *, size: int, pid: int,
            sidecar_mtime: float | None = None) -> None:
        import os
        log = runs / f"resolve-{issue}-{stamp}.log"
        log.write_bytes(b"x" * size)
        pid_file = runs / f"resolve-{issue}-{stamp}.pid"
        pid_file.write_text(str(pid), encoding="utf-8")
        if sidecar_mtime is not None:
            os.utime(pid_file, (sidecar_mtime, sidecar_mtime))

    def test_zero_byte_dead_pid_is_silent(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 465, "20260621-232003", size=0, pid=39688)
            out = mod.silent_workers(runs, alive=set())  # nothing alive
            self.assertEqual([w["issue"] for w in out], [465])
            self.assertEqual(out[0]["pid"], 39688)

    def test_zero_byte_live_pid_is_excluded(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 465, "20260621-232003", size=0, pid=39688)
            probe = lambda pid: {
                "alive": True,
                "cmdline": "claude -p resolve GitHub issue #465",
            }
            out = mod.silent_workers(runs, alive={39688}, probe=probe)  # still running
            self.assertEqual(out, [])

    def test_zero_byte_reused_pid_is_silent(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            now = 1_000_000.0
            self._mk(runs, 465, "20260621-232003", size=0, pid=39688,
                     sidecar_mtime=now)
            probe = lambda pid: {
                "alive": True,
                "create_time": now + 60 * 60,
                "cmdline": "chrome.exe --type=renderer",
            }
            out = mod.silent_workers(runs, alive={39688}, probe=probe)
            self.assertEqual([w["issue"] for w in out], [465])

    def test_non_empty_log_is_excluded(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 464, "20260621-232247", size=2366, pid=18796)
            out = mod.silent_workers(runs, alive=set())
            self.assertEqual(out, [])

    def test_missing_runs_dir_is_empty(self) -> None:
        mod = load()
        out = mod.silent_workers(Path("does-not-exist-xyz"), alive=set())
        self.assertEqual(out, [])

    def test_no_liveness_oracle_reports_nothing(self) -> None:
        # alive=None with psutil unavailable must NOT false-positive a silent worker
        # (we cannot prove the pid dead, so we report nothing rather than a false alarm).
        mod = load()
        import builtins

        real_import = builtins.__import__

        def no_psutil(name, *a, **k):
            if name == "psutil":
                raise ImportError("psutil disabled for this test")
            return real_import(name, *a, **k)

        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 465, "20260621-232003", size=0, pid=39688)
            builtins.__import__ = no_psutil
            try:
                out = mod.silent_workers(runs)  # alive=None -> tries psutil -> ImportError
            finally:
                builtins.__import__ = real_import
            self.assertEqual(out, [])

    def test_newest_first_ordering(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 465, "20260621-225432", size=0, pid=36580)
            self._mk(runs, 465, "20260621-232003", size=0, pid=39688)
            out = mod.silent_workers(runs, alive=set())
            self.assertEqual([w["stamp"] for w in out],
                             ["20260621-232003", "20260621-225432"])


class RenderMdTest(unittest.TestCase):
    """render_md is pure: a hand-built payload -> the committed-doc markdown."""

    def _payload(self, mod, **over):
        return build(mod, **over)

    def test_md_has_lane_table_and_closure_table(self) -> None:
        mod = load()
        md = mod.render_md(self._payload(mod), date="2026-06-21")
        self.assertIn("# Issue dispatch status — 2026-06-21", md)
        self.assertIn("## Backlog by lane", md)
        self.assertIn("| docs | 3 |", md)          # from backlog_ok(): docs has 3
        self.assertIn("| agent | 1 |", md)
        self.assertIn("## Closure honesty", md)
        self.assertIn("`closure_rate` = **0.8**", md)
        self.assertIn("| TRUE_RESOLVED | 8 |", md)

    def test_md_silent_section_lists_workers(self) -> None:
        mod = load()
        silent = [{"issue": 465, "stamp": "20260621-232003",
                   "log": "resolve-465-20260621-232003.log", "pid": 39688}]
        md = mod.render_md(self._payload(mod, silent=silent), date="2026-06-21")
        self.assertIn("## Workers that produced nothing", md)
        self.assertIn("| #465 | 20260621-232003 |", md)

    def test_md_silent_section_says_none_when_clean(self) -> None:
        mod = load()
        md = mod.render_md(self._payload(mod, silent=[]), date="2026-06-21")
        self.assertIn("## Workers that produced nothing", md)
        self.assertIn("None — every spawned worker", md)

    def test_md_handles_na_folds(self) -> None:
        mod = load()
        p = build(mod, backlog={"_skipped": "fast"}, closure={"_skipped": "fast"}, fast=True)
        md = mod.render_md(p, date="2026-06-21")
        self.assertIn("Backlog n/a", md)
        self.assertIn("Closure audit n/a", md)


if __name__ == "__main__":
    unittest.main()
