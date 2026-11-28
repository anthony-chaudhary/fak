#!/usr/bin/env python3
"""Hermetic tests for worker command building, liveness, probing, and sidecars."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from issue_resolve_dispatch_test import load  # noqa: E402


class OpencodeWorkerEnvTest(unittest.TestCase):
    def test_pinning_xdg_preserves_windows_gh_config(self) -> None:
        import os
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            appdata = root / "AppData" / "Roaming"
            gh_config = appdata / "GitHub CLI"
            gh_config.mkdir(parents=True)
            account = root / "opencode"
            account.mkdir()
            with mock.patch.dict(os.environ, {
                "APPDATA": str(appdata),
                "USERPROFILE": str(root / "home"),
            }, clear=True):
                env = mod.opencode_worker_env(
                    str(account), "contract-repair", root, root / ".dispatch-runs")
        self.assertEqual(env["XDG_CONFIG_HOME"], str(root))
        self.assertEqual(env["GH_CONFIG_DIR"], str(gh_config))

    def test_explicit_gh_config_dir_wins(self) -> None:
        import os
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            explicit = root / "custom-gh"
            account = root / "opencode"
            account.mkdir()
            with mock.patch.dict(os.environ, {
                "GH_CONFIG_DIR": str(explicit),
                "APPDATA": str(root / "AppData" / "Roaming"),
            }, clear=True):
                env = mod.opencode_worker_env(
                    str(account), "contract-repair", root, root / ".dispatch-runs")
        self.assertEqual(env["GH_CONFIG_DIR"], str(explicit))

    def test_temp_vars_are_inside_dispatch_runs(self) -> None:
        import os
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            account = root / "opencode"
            account.mkdir()
            runs = root / ".dispatch-runs"
            with mock.patch.dict(os.environ, {
                "TEMP": str(root / "external-temp"),
                "TMP": str(root / "external-tmp"),
                "TMPDIR": str(root / "external-tmpdir"),
            }, clear=True):
                env = mod.opencode_worker_env(str(account), "docs/tools", root, runs)

            expected = root / ".dispatch-runs" / ".opencode-tmp" / "docs-tools"
            self.assertTrue(expected.is_dir())
            self.assertEqual(env["TEMP"], str(expected))
            self.assertEqual(env["TMP"], str(expected))
            self.assertEqual(env["TMPDIR"], str(expected))


class CooldownTest(unittest.TestCase):
    def test_recent_log_is_in_cooldown_old_one_is_not(self) -> None:
        import os
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            now = 1_000_000.0
            fresh = runs / "resolve-465-20260621-225432.log"
            stale = runs / "resolve-450-20260621-100000.log"
            fresh.write_text("", encoding="utf-8")
            stale.write_text("", encoding="utf-8")
            os.utime(fresh, (now - 10 * 60, now - 10 * 60))    # 10 min ago
            os.utime(stale, (now - 300 * 60, now - 300 * 60))  # 5 h ago
            cooled = mod.recently_attempted_issues(runs, cooldown_min=120, now_ts=now)
        self.assertEqual(cooled, {465})        # 465 fresh (10m < 120m), 450 stale

    def test_recent_witness_only_attempt_is_in_cooldown(self) -> None:
        import os
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            now = 1_000_000.0
            fresh = runs / "resolve-2898-20260705-214923.witness"
            fresh.write_text('{"claim": "CLAIM_NO_COMMIT", "issue": 2898}',
                             encoding="utf-8")
            os.utime(fresh, (now - 5 * 60, now - 5 * 60))
            cooled = mod.recently_attempted_issues(runs, cooldown_min=120, now_ts=now)
        self.assertEqual(cooled, {2898})

    def test_cooldown_zero_disables(self) -> None:
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            (runs / "resolve-465-x.log").write_text("", encoding="utf-8")
            self.assertEqual(mod.recently_attempted_issues(runs, cooldown_min=0), set())


class LiveResolutionIssuesTest(unittest.TestCase):
    def _mk(self, runs: Path, issue: int, stamp: str, *, pid: int,
            sidecar_mtime: float) -> None:
        import os
        log = runs / f"resolve-{issue}-{stamp}.log"
        log.write_text("", encoding="utf-8")
        pid_file = log.with_suffix(".pid")
        pid_file.write_text(str(pid), encoding="utf-8")
        os.utime(pid_file, (sidecar_mtime, sidecar_mtime))

    def test_counts_issue_when_sidecar_process_matches_spawn_time(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 717, "20260625-062210", pid=101, sidecar_mtime=now)
            # OS hid the cmdline, but the image is a real worker backend → counts.
            def probe(pid):
                return {"alive": True, "create_time": now - 1,
                                             "name": "claude.exe", "cmdline": ""}
            self.assertEqual(mod.live_resolution_issues(runs, probe=probe), {717})

    def test_rejects_issue_when_sidecar_pid_was_reused(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 718, "20260625-060712", pid=102, sidecar_mtime=now)
            def probe(pid):
                return {
                            "alive": True,
                            "create_time": now + 60 * 60,
                            "name": "chrome.exe",
                            "cmdline": "chrome.exe --type=renderer",
                        }
            self.assertEqual(mod.live_resolution_issues(runs, probe=probe), set())


class LiveResolutionLanesTest(unittest.TestCase):
    """The pre-spawn lane-lease set (#1310): a lane is HELD when it already has a
    live worker. Read from the ``# fak-spawn ... lane=<L>`` header
    ``spawn_issue_worker`` flushes, gated by the SAME sidecar-pid liveness check
    as the live-issue set, so a recycled PID never falsely holds a lane."""

    def _mk(self, runs: Path, issue: int, stamp: str, lane: str, *, pid: int,
            sidecar_mtime: float, header: bool = True) -> None:
        import os
        log = runs / f"resolve-{issue}-{stamp}.log"
        body = (f"# fak-spawn {stamp} issue={issue} lane={lane} backend=claude "
                f"argv0=claude.exe\n") if header else ""
        log.write_text(body, encoding="utf-8")
        pid_file = log.with_suffix(".pid")
        pid_file.write_text(str(pid), encoding="utf-8")
        os.utime(pid_file, (sidecar_mtime, sidecar_mtime))

    def test_held_lane_when_live_worker_header_names_it(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 717, "20260625-062210", "docs", pid=101, sidecar_mtime=now)
            def probe(pid):
                return {"alive": True, "create_time": now - 1,
                        "name": "claude.exe", "cmdline": ""}
            self.assertEqual(mod.live_resolution_lanes(runs, probe=probe), {"docs"})

    def test_recycled_pid_does_not_hold_its_lane(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 718, "20260625-060712", "gateway", pid=102, sidecar_mtime=now)
            def probe(pid):  # a different process owns the PID now → worker is gone
                return {"alive": True, "create_time": now + 60 * 60,
                        "name": "chrome.exe", "cmdline": "chrome.exe --type=renderer"}
            self.assertEqual(mod.live_resolution_lanes(runs, probe=probe), set())

    def test_headerless_log_contributes_no_lane(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            # live pid but a 0-byte log (died before exec wrote the header) → its
            # lane is unknowable, so it holds nothing (best effort, fail-open).
            self._mk(runs, 719, "20260625-070000", "docs", pid=103,
                     sidecar_mtime=now, header=False)
            def probe(pid):
                return {"alive": True, "create_time": now - 1,
                        "name": "claude.exe", "cmdline": ""}
            self.assertEqual(mod.live_resolution_lanes(runs, probe=probe), set())

    def test_banner_noop_log_does_not_hold_its_lane(self) -> None:
        """A worker whose log is a terminal banner no-op (#1275) holds no lane even
        when a recycled pid passes the weak liveness gate (#1398). An exited opencode
        worker runs as a ``node`` image, so a recycled ``node`` pid landing in the
        sidecar's spawn window passes the create-time fallback and would otherwise pin
        ``docs`` at LANE_BUSY forever behind a dead 122-byte no-op."""
        import os
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            log = runs / "resolve-1398-20260629-101010.log"
            # header + the documented opencode/glm banner-only no-op, well under the
            # 512-byte stub floor (the real #1398 holders were 122 bytes).
            log.write_text(
                "# fak-spawn 20260629-101010 issue=1398 lane=docs backend=opencode "
                "argv0=node\n> build · glm-4.5-air\n",
                encoding="utf-8")
            pid_file = log.with_suffix(".pid")
            pid_file.write_text("777", encoding="utf-8")
            os.utime(pid_file, (now, now))
            def probe(pid):  # a recycled `node` pid in the spawn window → weak-live
                return {"alive": True, "create_time": now - 1,
                        "name": "node.exe", "cmdline": ""}
            self.assertEqual(mod.live_resolution_lanes(runs, probe=probe), set())

    def test_two_live_workers_hold_two_lanes(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 717, "20260625-062210", "docs", pid=101, sidecar_mtime=now)
            self._mk(runs, 720, "20260625-063000", "gateway", pid=104, sidecar_mtime=now)
            def probe(pid):
                return {"alive": True, "create_time": now - 1,
                        "name": "claude.exe", "cmdline": ""}
            self.assertEqual(mod.live_resolution_lanes(runs, probe=probe),
                             {"docs", "gateway"})


class TimedOutWorkerReapTest(unittest.TestCase):
    def _mk(self, runs: Path, issue: int, stamp: str, *, pid: int,
            sidecar_mtime: float) -> Path:
        import os
        log = runs / f"resolve-{issue}-{stamp}.log"
        log.write_text("", encoding="utf-8")
        pid_file = log.with_suffix(".pid")
        pid_file.write_text(str(pid), encoding="utf-8")
        os.utime(pid_file, (sidecar_mtime, sidecar_mtime))
        return pid_file

    def test_dry_run_reports_would_reap_without_killing(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        killed: list[int] = []
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 717, "20260625-062210", pid=101, sidecar_mtime=now - 4000)
            def probe(pid):
                return {"alive": True, "create_time": now - 4001,
                                             "name": "claude.exe", "cmdline": ""}
            out = mod.reap_timed_out_workers(
                runs, timeout_s=1800, live=False, now_ts=now, probe=probe,
                killer=lambda pid: killed.append(pid))
        self.assertEqual([r["pid"] for r in out["would_reap"]], [101])
        self.assertEqual(out["reaped"], [])
        self.assertEqual(killed, [])

    def test_live_reaps_matching_timed_out_worker(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        killed: list[int] = []
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 718, "20260625-060712", pid=102, sidecar_mtime=now - 4000)
            def probe(pid):
                return {"alive": True, "create_time": now - 4001,
                                             "name": "claude.exe", "cmdline": ""}
            out = mod.reap_timed_out_workers(
                runs, timeout_s=1800, live=True, now_ts=now, probe=probe,
                killer=lambda pid: (killed.append(pid) or {"ok": True, "returncode": 0}))
        self.assertEqual([r["pid"] for r in out["reaped"]], [102])
        self.assertEqual(killed, [102])
        self.assertTrue(out["reaped"][0]["kill"]["ok"])

    def test_fresh_worker_is_not_reaped(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 719, "20260625-055209", pid=103, sidecar_mtime=now - 60)
            def probe(pid):
                return {"alive": True, "create_time": now - 61, "cmdline": ""}
            out = mod.reap_timed_out_workers(
                runs, timeout_s=1800, live=True, now_ts=now, probe=probe,
                killer=lambda pid: {"ok": True})
        self.assertEqual(out["candidates"], [])
        self.assertEqual(out["reaped"], [])

    def test_reused_pid_is_not_reaped(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 720, "20260625-055210", pid=104, sidecar_mtime=now - 4000)
            def probe(pid):
                return {
                            "alive": True,
                            "create_time": now + 60,
                            "cmdline": "chrome.exe --type=renderer",
                        }
            out = mod.reap_timed_out_workers(
                runs, timeout_s=1800, live=True, now_ts=now, probe=probe,
                killer=lambda pid: {"ok": True})
        self.assertEqual(out["candidates"], [])
        self.assertEqual(out["reaped"], [])

    def test_timed_out_repair_worker_is_reaped_too(self) -> None:
        # A runaway contract-repair worker is subject to the SAME wall-clock cap
        # as a resolution worker — the reaper sweeps both log prefixes.
        import os
        import tempfile
        mod = load()
        now = 1_000_000.0
        killed: list[int] = []
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            log = runs / "repair-1207-20260702-180000.log"
            log.write_text("", encoding="utf-8")
            pid_file = log.with_suffix(".pid")
            pid_file.write_text("105", encoding="utf-8")
            os.utime(pid_file, (now - 4000, now - 4000))
            def probe(pid):
                return {"alive": True, "create_time": now - 4001,
                        "name": "claude.exe", "cmdline": ""}
            out = mod.reap_timed_out_workers(
                runs, timeout_s=1800, live=True, now_ts=now, probe=probe,
                killer=lambda pid: (killed.append(pid) or {"ok": True, "returncode": 0}))
        self.assertEqual([r["pid"] for r in out["reaped"]], [105])
        self.assertEqual(killed, [105])


class PruneDeadSidecarsTest(unittest.TestCase):
    def _mk(self, runs: Path, issue: int, stamp: str, *, pid: int,
            mtime: float, siblings: tuple[str, ...] = ()) -> Path:
        import os
        pid_file = runs / f"resolve-{issue}-{stamp}.pid"
        pid_file.write_text(str(pid), encoding="utf-8")
        stem = pid_file.with_suffix("")
        for suf in siblings:
            stem.with_suffix(suf).write_text("", encoding="utf-8")
        os.utime(pid_file, (mtime, mtime))
        return pid_file

    def test_live_prunes_dead_sidecar_and_siblings(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 825, "20260625-213720", pid=58752, mtime=now - 4000,
                     siblings=(".log", ".backend", ".wave", ".account"))
            log = runs / "resolve-825-20260625-213720.log"
            log.write_text("worker transcript\n", encoding="utf-8")
            def probe(pid):
                return {"alive": False}
            out = mod.prune_dead_sidecars(runs, live=True, now_ts=now, probe=probe)
            self.assertEqual(out["pruned"], ["resolve-825-20260625-213720.pid"])
            self.assertTrue(log.exists())
            self.assertEqual(log.read_text(encoding="utf-8"), "worker transcript\n")
            self.assertEqual(
                sorted(p.name for p in runs.glob("resolve-825-*")),
                ["resolve-825-20260625-213720.log"],
            )

    def test_live_prune_keeps_witness_sidecar_for_cooldown(self) -> None:
        import os
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            pid_file = self._mk(
                runs, 826, "20260625-213721", pid=58752, mtime=now - 4000,
                siblings=(".log", mod.WITNESS_SIDECAR_SUFFIX))
            witness = pid_file.with_suffix(mod.WITNESS_SIDECAR_SUFFIX)
            witness.write_text('{"claim": "CLAIM_NO_COMMIT", "issue": 826}',
                               encoding="utf-8")
            os.utime(witness, (now - 10 * 60, now - 10 * 60))
            def probe(pid):
                return {"alive": False}
            out = mod.prune_dead_sidecars(runs, live=True, now_ts=now, probe=probe)
            cooled = mod.recently_attempted_issues(
                runs, cooldown_min=120, now_ts=now)
            self.assertEqual(out["pruned"], ["resolve-826-20260625-213721.pid"])
            self.assertFalse(pid_file.exists())
            self.assertTrue(witness.exists())
            self.assertEqual(cooled, {826})

    def test_dead_repair_sidecar_and_issues_sibling_are_pruned(self) -> None:
        # Repair corpses (pid + the .issues batch sidecar) are swept like resolve
        # corpses, so stale repair sidecars never pin the seat count or the
        # repair-admission scan's directory.
        import os
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            pid_file = runs / "repair-1207-20260702-180000.pid"
            pid_file.write_text("58753", encoding="utf-8")
            stem = pid_file.with_suffix("")
            for suf in (".log", ".backend", mod.REPAIR_ISSUES_SIDECAR_SUFFIX):
                stem.with_suffix(suf).write_text("", encoding="utf-8")
            os.utime(pid_file, (now - 4000, now - 4000))
            def probe(pid):
                return {"alive": False}
            out = mod.prune_dead_sidecars(runs, live=True, now_ts=now, probe=probe)
            self.assertEqual(out["pruned"], ["repair-1207-20260702-180000.pid"])
            self.assertEqual(sorted(p.name for p in runs.glob("repair-1207-*")), [])

    def test_repair_cooldown_outlives_the_sidecar_sweep(self) -> None:
        """The repair cooldown must survive the corpse it was read off.

        Regression: `recently_repaired_issues` read the `repair-*.log` mtime and
        its `.issues` batch sidecar -- both of which `prune_dead_sidecars` unlinks
        the moment the groomer exits. So the 360-min anti-churn window silently
        collapsed to the groomer's OWN LIFETIME, which the live-repair scan already
        covers, and the same un-repairable heads were re-groomed every ~40 min
        (measured on this host: 62/186 grooming events inside the window, median
        gap 43 min). The durable ledger must still cool the whole batch after the
        sweep has taken every file the old reader depended on.
        """
        import os
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            mod.record_repair_attempt(runs, [3238, 5255, 4318],
                                      live=True, now_ts=now - 40 * 60)
            pid_file = runs / "repair-3238-20260702-180000.pid"
            pid_file.write_text("58753", encoding="utf-8")
            stem = pid_file.with_suffix("")
            stem.with_suffix(".log").write_text("", encoding="utf-8")
            stem.with_suffix(mod.REPAIR_ISSUES_SIDECAR_SUFFIX).write_text(
                "3238,5255,4318", encoding="utf-8")
            os.utime(pid_file, (now - 40 * 60, now - 40 * 60))

            def probe(pid):
                return {"alive": False}

            mod.prune_dead_sidecars(runs, live=True, now_ts=now, probe=probe)
            # Every file the OLD reader used is gone -- that is the bug's premise.
            self.assertEqual(sorted(runs.glob("repair-3238-*")), [])
            cooled = mod.recently_repaired_issues(
                runs, cooldown_min=360, now_ts=now)

        self.assertEqual(cooled, {3238, 5255, 4318})

    def test_repair_cooldown_expires_with_its_window(self) -> None:
        """...and clears itself: past the window the batch is groomable again with
        no reset verb, so an issue is never pinned out permanently."""
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            mod.record_repair_attempt(runs, [3238, 5255],
                                      live=True, now_ts=now - 361 * 60)
            expired = mod.recently_repaired_issues(
                runs, cooldown_min=360, now_ts=now)
            # A dry run stays side-effect-free, so it never cools anything.
            mod.record_repair_attempt(runs, [999], live=False, now_ts=now)
            dry = mod.recently_repaired_issues(runs, cooldown_min=360, now_ts=now)
            disabled = mod.recently_repaired_issues(
                runs, cooldown_min=0, now_ts=now)

        self.assertEqual(expired, set())
        self.assertEqual(dry, set())
        self.assertEqual(disabled, set())

    def test_recycled_shell_in_window_is_pruned(self) -> None:
        # The exact ghost: a recycled cmd.exe whose create time lands inside the
        # stale sidecar's spawn window. resolve_sidecar_pid_is_live rejects a bare
        # shell image, so the sidecar is correctly seen as dead and swept.
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 825, "20260625-213720", pid=58752, mtime=now - 4000)
            def probe(pid):
                return {"alive": True, "create_time": now - 4030,
                                             "name": "cmd.exe", "cmdline": ""}
            out = mod.prune_dead_sidecars(runs, live=True, now_ts=now, probe=probe)
            self.assertEqual(out["pruned"], ["resolve-825-20260625-213720.pid"])

    def test_live_worker_sidecar_is_kept(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 822, "20260625-220538", pid=8416, mtime=now - 4000)
            def probe(pid):
                return {"alive": True, "create_time": now - 4001,
                                             "name": "claude.exe",
                                             "cmdline": "claude -p resolve GitHub issue #822"}
            out = mod.prune_dead_sidecars(runs, live=True, now_ts=now, probe=probe)
            self.assertEqual(out["pruned"], [])
            self.assertTrue((runs / "resolve-822-20260625-220538.pid").exists())

    def test_too_fresh_dead_sidecar_survives_min_age(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self._mk(runs, 830, "20260625-210713", pid=999, mtime=now - 5)
            def probe(pid):
                return {"alive": False}
            out = mod.prune_dead_sidecars(runs, live=True, min_age_s=60.0,
                                          now_ts=now, probe=probe)
            self.assertEqual(out["pruned"], [])
            self.assertTrue((runs / "resolve-830-20260625-210713.pid").exists())

    def test_dry_run_reports_would_prune_without_unlinking(self) -> None:
        import tempfile
        mod = load()
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            pid_file = self._mk(runs, 831, "20260625-205217", pid=111, mtime=now - 4000)
            def probe(pid):
                return {"alive": False}
            out = mod.prune_dead_sidecars(runs, live=False, now_ts=now, probe=probe)
            self.assertEqual(out["would_prune"], ["resolve-831-20260625-205217.pid"])
            self.assertEqual(out["pruned"], [])
            self.assertTrue(pid_file.exists())

    def test_timeout_zero_disables_reap(self) -> None:
        mod = load()
        out = mod.reap_timed_out_workers(Path("does-not-matter"), timeout_s=None, live=True)
        self.assertTrue(out["disabled"])


class BuildWorkerCommandTest(unittest.TestCase):
    def test_claude_command_shape(self) -> None:
        mod = load()
        self.assertEqual(
            mod.build_worker_command("claude", "PROMPT", None),
            ["claude", "-p", "--permission-mode", "bypassPermissions"])

    def test_claude_command_pins_model_and_effort(self) -> None:
        mod = load()
        self.assertEqual(
            mod.build_worker_command("claude", "PROMPT", "claude-opus-4-8", "xhigh"),
            ["claude", "-p", "--permission-mode", "bypassPermissions",
             "--model", "claude-opus-4-8", "--effort", "xhigh"])

    def test_claude_command_effort_only_when_model_absent(self) -> None:
        # A model-less claude worker still carries its reasoning effort flag.
        mod = load()
        self.assertEqual(
            mod.build_worker_command("claude", "PROMPT", None, "xhigh"),
            ["claude", "-p", "--permission-mode", "bypassPermissions",
             "--effort", "xhigh"])

    def test_worker_model_effort_pins_claude_to_opus_xhigh(self) -> None:
        # A cloud claude account is pinned to Opus at xhigh regardless of the
        # (live-rotated) per-account settings.json model.
        mod = load()
        self.assertEqual(
            mod.worker_model_effort("claude", {"model": "claude-fable-5"}),
            (mod.CLAUDE_WORKER_MODEL, mod.CLAUDE_WORKER_EFFORT))
        self.assertEqual(mod.CLAUDE_WORKER_MODEL, "claude-opus-5")
        self.assertEqual(mod.CLAUDE_WORKER_EFFORT, "xhigh")

    def test_worker_model_effort_keeps_local_claude_and_opencode(self) -> None:
        mod = load()
        # A 'local' claude account is not forced onto cloud Opus, and takes no effort.
        self.assertEqual(mod.worker_model_effort("claude", {"model": "local"}),
                         ("local", None))
        # opencode keeps its own model and takes no effort knob.
        self.assertEqual(mod.worker_model_effort("opencode", {"model": "glm-5.2"}),
                         ("glm-5.2", None))
        # codex is unpinned.
        self.assertEqual(mod.worker_model_effort("codex", {"model": "x"}), (None, None))

    def test_opencode_command_pins_model_and_skips_permissions(self) -> None:
        mod = load()
        self.assertEqual(
            mod.build_worker_command("opencode", "PROMPT", "zai-coding-plan/glm-5.2"),
            ["opencode", "run", "--pure", "--print-logs",
             "--dangerously-skip-permissions",
             "-m", "zai-coding-plan/glm-5.2", mod.OPENCODE_PROMPT_NOTICE])

    def test_opencode_command_without_model(self) -> None:
        mod = load()
        self.assertEqual(
            mod.build_worker_command("opencode", "PROMPT", None),
            ["opencode", "run", "--pure", "--print-logs",
             "--dangerously-skip-permissions",
             mod.OPENCODE_PROMPT_NOTICE])

    def test_opencode_command_keeps_full_prompt_out_of_argv(self) -> None:
        mod = load()
        full_prompt = "your goal: resolve GitHub issue #2588 with lots of detail"
        got = mod.build_worker_command("opencode", full_prompt, "glm")
        self.assertNotIn(full_prompt, got)
        self.assertIn("resolve github issue #", mod.OPENCODE_PROMPT_NOTICE.lower())
        self.assertLessEqual(len(mod.OPENCODE_PROMPT_NOTICE), 96)

    def test_opencode_prompt_file_is_attached_before_notice(self) -> None:
        mod = load()
        got = mod.attach_opencode_prompt_file(
            ["opencode", "run", "--pure", "--print-logs", "-m", "glm",
             mod.OPENCODE_PROMPT_NOTICE],
            r"C:\work\fak\.dispatch-runs\resolve-2588.prompt.txt")
        self.assertEqual(got, [
            "opencode", "run", "--pure", "--print-logs", "-m", "glm",
            "--file", r"C:\work\fak\.dispatch-runs\resolve-2588.prompt.txt",
            "--", mod.OPENCODE_PROMPT_NOTICE])

    def test_opencode_prompt_file_survives_guard_wrapper(self) -> None:
        mod = load()
        got = mod.attach_opencode_prompt_file(
            ["fak", "guard", "--provider", "openai", "--", "opencode", "run",
             mod.OPENCODE_PROMPT_NOTICE],
            "resolve-2588.prompt.txt")
        self.assertEqual(got[-4:], ["--file", "resolve-2588.prompt.txt",
                                    "--", mod.OPENCODE_PROMPT_NOTICE])

    def test_unwrap_opencode_npm_shim_targets_real_executable(self) -> None:
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            npm = Path(d)
            real = npm / "node_modules" / "opencode-ai" / "bin" / "opencode.exe"
            real.parent.mkdir(parents=True)
            real.write_text("fake exe", encoding="utf-8")
            self.assertEqual(
                mod.unwrap_opencode_npm_shim(str(npm / "opencode.cmd")), str(real))
            self.assertEqual(mod.unwrap_opencode_npm_shim(str(npm / "claude.cmd")), "")

    def test_codex_command_shape(self) -> None:
        mod = load()
        self.assertEqual(
            mod.build_worker_command("codex", "PROMPT", None),
            ["codex", "exec", "--dangerously-bypass-approvals-and-sandbox",
             "--skip-git-repo-check"])

    def test_codex_command_pins_model(self) -> None:
        mod = load()
        self.assertEqual(
            mod.build_worker_command("codex", "PROMPT", "gpt-5.5"),
            ["codex", "exec", "--dangerously-bypass-approvals-and-sandbox",
             "--skip-git-repo-check", "-m", "gpt-5.5"])

    def test_codex_env_uses_codex_home_not_claude_vars(self) -> None:
        mod = load()
        env = mod.codex_worker_env("/some/.codex", "tools", Path("/ws"))
        self.assertEqual(env.get("CODEX_HOME"), "/some/.codex")
        self.assertNotIn("CLAUDE_CONFIG_DIR", env)
        self.assertNotIn("CLAUDE_CODE_OAUTH_TOKEN", env)

    def test_unknown_backend_rejected(self) -> None:
        mod = load()
        with self.assertRaises(ValueError):
            mod.build_worker_command("gpt", "x", None)

    def test_windows_fak_bin_env_keeps_backslashes(self) -> None:
        mod = load()
        with mock.patch.object(mod.os, "name", "nt"):
            self.assertEqual(
                mod.split_command_env(r"C:\work\fak\fak.exe"),
                [r"C:\work\fak\fak.exe"])


class WinCreationFlagsTest(unittest.TestCase):
    def test_worker_spawn_allocates_no_console_for_native_and_batch_launchers(self) -> None:
        mod = load()
        want = mod._DETACHED_PROCESS if os.name == "nt" else 0
        for exe in (r"C:\npm\opencode.CMD", "opencode.cmd", "wrap.bat",
                    r"C:\bin\claude.exe", "/usr/bin/claude"):
            with self.subTest(exe=exe):
                self.assertEqual(mod.win_creationflags(exe), want)
                self.assertEqual(mod.win_creationflags(exe) & mod._CREATE_NO_WINDOW, 0)


class SpawnProbeTest(unittest.TestCase):
    def test_probe_reports_alive_when_timeout_expires(self) -> None:
        import tempfile
        mod = load()

        class _Alive:
            def wait(self, timeout):
                raise mod.subprocess.TimeoutExpired("worker", timeout)

        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "resolve-1.log"
            log.write_text("", encoding="utf-8")
            out = mod.probe_spawned_worker(_Alive(), log, wait_s=0.1)
        self.assertEqual(out, {"checked": True, "alive": True, "wait_s": 0.1})

    def test_probe_reports_silent_exit(self) -> None:
        import tempfile
        mod = load()

        class _Exited:
            def wait(self, timeout):
                return 0

        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "resolve-1.log"
            log.write_text("", encoding="utf-8")
            out = mod.probe_spawned_worker(_Exited(), log, wait_s=0.1)
        self.assertFalse(out["alive"])
        self.assertTrue(out["silent"])
        self.assertEqual(out["log_bytes"], 0)

    def test_spawn_writes_preexec_header_under_stub_floor(self) -> None:
        """A spawned worker's log opens with a flushed spawn header, so a later
        0-byte log means 'died before exec' (not 'exec'd then silent'). The header
        must stay under the stub floor and be banner-free so it never trips the
        stub/cap-banner classifiers."""
        import tempfile
        import sys
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            # A trivially-exiting real process (no output of its own).
            res = mod.spawn_issue_worker([sys.executable, "-c", "pass"], env={},
                                         cwd=Path(d), log_dir=runs, issue=42,
                                         lane="docs", backend="claude",
                                         spawn_probe_s=2.0)
            log = Path(res["log"])
            body = log.read_text(encoding="utf-8")
        self.assertTrue(body.startswith("# fak-spawn "))
        self.assertIn("issue=42", body)
        self.assertIn("lane=docs", body)
        self.assertLess(len(body), mod._STUB_LOG_MAX_BYTES)   # header alone stays a stub
        self.assertIsNone(mod._CAP_BANNER_RE.search(body))    # never a false cap banner


class OpencodeConfigHomeTest(unittest.TestCase):
    def test_canonical_dir_uses_parent_as_xdg_home(self) -> None:
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            acct = Path(d) / "opencode"
            acct.mkdir()
            home = mod._opencode_config_home(str(acct), Path(d) / "runs")
            self.assertEqual(Path(home), Path(d))   # parent is the XDG config home


class WaveMembershipTest(unittest.TestCase):
    """A wave is a typed group, not N anonymous lanes: each worker is stamped with
    its rank/wave/size into child_env + a .wave sidecar, so an auditor enumerates
    the whole group from the filesystem and reads the honest under-fill."""

    def test_wave_membership_env_shape(self) -> None:
        mod = load()
        env = mod.wave_membership_env(rank=2, wave_id="wave-abc", size=4, shortfall=1)
        self.assertEqual(env, {
            "FLEET_WAVE_ID": "wave-abc", "FLEET_WAVE_RANK": "2",
            "FLEET_WAVE_SIZE": "4", "FLEET_WAVE_SHORTFALL": "1"})

    def test_enumerate_wave_reads_ranks_and_shortfall_from_sidecars(self) -> None:
        # Simulate a granted-3 wave (requested 5 -> shortfall 2) by writing the same
        # .wave sidecars spawn_issue_worker writes, then enumerate from disk alone.
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            for rank in range(3):
                out_log = runs / f"resolve-{600 + rank}-2026{rank}.log"
                mod.write_wave_sidecar(out_log, rank=rank, wave_id="wave-xyz",
                                       size=3, shortfall=2)
            # A worker from a DIFFERENT wave must not bleed into this enumeration.
            mod.write_wave_sidecar(runs / "resolve-999-other.log", rank=0,
                                   wave_id="wave-other", size=1, shortfall=0)
            grp = mod.enumerate_wave(runs, "wave-xyz")
            self.assertEqual(grp["wave_id"], "wave-xyz")
            self.assertEqual(grp["granted"], 3)
            self.assertEqual(grp["size"], 3)
            self.assertEqual(grp["shortfall"], 2)
            self.assertEqual(grp["ranks"], [0, 1, 2])   # ranks 0..granted-1

    def test_spawn_stamps_membership_env_and_sidecar(self) -> None:
        # End-to-end: spawn_issue_worker stamps FLEET_WAVE_* into the child env AND
        # writes the .wave sidecar — proven without launching a real process.
        import tempfile
        from unittest import mock
        mod = load()
        captured: dict[str, object] = {}

        class _FakePopen:
            def __init__(self, argv, **kwargs):
                captured["env"] = kwargs.get("env")
                kwargs.get("stdout").close()  # the parent's log fh (no real child)
                self.pid = 31337

        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            membership = {"rank": 1, "wave_id": "wave-zzz", "size": 2, "shortfall": 0}
            with mock.patch.object(mod.subprocess, "Popen", _FakePopen):
                res = mod.spawn_issue_worker(
                    ["true"], {"DISPATCH_LANE": "tools"}, runs, runs,
                    issue=42, lane="tools", backend="claude", membership=membership)
            env = captured["env"]
            self.assertEqual(env["FLEET_WAVE_RANK"], "1")
            self.assertEqual(env["FLEET_WAVE_ID"], "wave-zzz")
            self.assertEqual(env["FLEET_WAVE_SIZE"], "2")
            self.assertEqual(env["FLEET_WAVE_SHORTFALL"], "0")
            self.assertEqual(env["DISPATCH_LANE"], "tools")  # base env preserved
            self.assertEqual(res["membership"]["rank"], 1)
            # The wave is enumerable from the sidecar that spawn wrote.
            grp = mod.enumerate_wave(runs, "wave-zzz")
            self.assertEqual(grp["ranks"], [1])
            self.assertEqual(grp["shortfall"], 0)

    def test_spawn_stamps_account_sidecar(self) -> None:
        # A silent 0-byte worker log is only useful if it can be attributed back to
        # the selected switcher account. The account sidecar is written at spawn time,
        # before the child has to produce any output.
        import json
        import tempfile
        from unittest import mock
        mod = load()

        class _FakePopen:
            def __init__(self, argv, **kwargs):
                self.pid = 9

        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            account = {"tag": "gem8-netra", "tier": 1, "model": "opus", "dir": "/acct/gem8"}
            with mock.patch.object(mod.subprocess, "Popen", _FakePopen):
                res = mod.spawn_issue_worker(["true"], {}, runs, runs,
                                             issue=2, lane="tools", backend="claude",
                                             account=account)
            self.assertEqual(res["account"]["tag"], "gem8-netra")
            sidecars = list(runs.glob("*.account"))
            self.assertEqual(len(sidecars), 1)
            self.assertEqual(json.loads(sidecars[0].read_text(encoding="utf-8")), account)

    def test_spawn_stamps_lease_sidecar(self) -> None:
        import json
        import tempfile
        from unittest import mock
        mod = load()

        class _FakePopen:
            def __init__(self, argv, **kwargs):
                self.pid = 10

        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            lease = {
                "acquired": True,
                "id": "resolve-tools",
                "holder": "session-1",
                "generation": 4,
                "session_id": "sess-1",
                "tree": ["tools/**"],
            }
            with mock.patch.object(mod.subprocess, "Popen", _FakePopen):
                res = mod.spawn_issue_worker(["true"], {}, runs, runs,
                                             issue=2, lane="tools", backend="claude",
                                             lease=lease)
            self.assertEqual(res["lease"]["id"], "resolve-tools")
            sidecars = list(runs.glob("*.lease"))
            self.assertEqual(len(sidecars), 1)
            self.assertEqual(json.loads(sidecars[0].read_text(encoding="utf-8")),
                             {"id": "resolve-tools", "holder": "session-1",
                              "generation": 4, "session_id": "sess-1",
                              "tree": ["tools/**"]})

    def test_opencode_spawn_stages_prompt_and_unwraps_npm_shim(self) -> None:
        import tempfile
        from unittest import mock
        mod = load()
        captured: dict[str, object] = {}

        class _FakePopen:
            def __init__(self, argv, **kwargs):
                captured["argv"] = argv
                captured["env"] = kwargs.get("env")
                self.pid = 2588

        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            npm = runs / "npm"
            real = npm / "node_modules" / "opencode-ai" / "bin" / "opencode.exe"
            real.parent.mkdir(parents=True)
            real.write_text("fake exe", encoding="utf-8")
            full_prompt = "your goal: resolve GitHub issue #2588 with private details"
            command = mod.build_worker_command("opencode", full_prompt, "glm")

            def _which(name):
                return str(npm / "opencode.cmd") if name == "opencode" else None

            with mock.patch.object(mod.shutil, "which", _which), \
                    mock.patch.object(mod.os, "name", "nt"), \
                    mock.patch.object(mod.subprocess, "Popen", _FakePopen):
                res = mod.spawn_issue_worker(
                    command, {"DISPATCH_LANE": "dispatch"}, runs, runs,
                    issue=2588, lane="dispatch", backend="opencode",
                    prompt_payload=full_prompt)

            argv = captured["argv"]
            self.assertEqual(argv[0], str(real))
            self.assertIn("--file", argv)
            self.assertIn("--", argv)
            self.assertIn(mod.OPENCODE_PROMPT_NOTICE, argv)
            self.assertNotIn(full_prompt, argv)
            prompt_file = Path(res["prompt_file"])
            self.assertEqual(prompt_file.read_text(encoding="utf-8"), full_prompt)

    def test_codex_long_prompt_is_streamed_via_stdin_file(self) -> None:
        mod = load()
        captured = {}
        full_prompt = "resolve #4779 " + ("x" * 40000)

        class _FakePopen:
            pid = 1234
            def __init__(self, argv, **kwargs):
                captured["argv"] = argv
                captured["stdin_text"] = kwargs["stdin"].read()
            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(mod.subprocess, "Popen", _FakePopen):
            runs = Path(d)
            command = mod.build_worker_command("codex", full_prompt, None)
            res = mod.spawn_issue_worker(
                command, {}, runs, runs, issue=4779, lane="tools",
                backend="codex", prompt_payload=full_prompt)
            prompt_copy = Path(res["prompt_file"]).read_text(encoding="utf-8")

        self.assertNotIn(full_prompt, captured["argv"])
        self.assertEqual(captured["stdin_text"], full_prompt)
        self.assertEqual(prompt_copy, full_prompt)

    def test_claude_long_prompt_is_streamed_via_stdin_file(self) -> None:
        mod = load()
        captured = {}
        full_prompt = "resolve #4779 " + ("x" * 40000)

        class _FakePopen:
            pid = 1234
            def __init__(self, argv, **kwargs):
                captured["argv"] = argv
                captured["stdin_text"] = kwargs["stdin"].read()
            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(mod.subprocess, "Popen", _FakePopen):
            runs = Path(d)
            command = mod.build_worker_command("claude", full_prompt, None)
            res = mod.spawn_issue_worker(
                command, {}, runs, runs, issue=4779, lane="tools",
                backend="claude", prompt_payload=full_prompt)
            prompt_copy = Path(res["prompt_file"]).read_text(encoding="utf-8")

        self.assertNotIn(full_prompt, captured["argv"])
        self.assertEqual(captured["stdin_text"], full_prompt)
        self.assertEqual(prompt_copy, full_prompt)

    def test_spawn_without_membership_writes_no_wave_sidecar(self) -> None:
        # The default single-worker path is unchanged: no membership -> no .wave file.
        import tempfile
        from unittest import mock
        mod = load()

        class _FakePopen:
            def __init__(self, argv, **kwargs):
                kwargs.get("stdout").close()  # the parent's log fh (no real child)
                self.pid = 7

        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            with mock.patch.object(mod.subprocess, "Popen", _FakePopen):
                mod.spawn_issue_worker(["true"], {}, runs, runs,
                                       issue=1, lane="tools", backend="claude")
            self.assertEqual(list(runs.glob("*.wave")), [])


class DispatchAccountStampTest(unittest.TestCase):
    """#5870: the long-``Retry-After`` goal park is account-scoped
    (``goalpark.Record.Blocks``), and a record whose ``account`` is blank blocks
    NOBODY. Only the Go spine ever stamped ``DISPATCH_ACCOUNT``; this dispatcher —
    the live producer of the fleet's ``resolve-*.log`` units — stamped nothing, so
    every park its workers wrote was unattributed and the park was inert. These
    assert the stamp AND that its format matches the Go spine's
    ``dispatchAccountID`` exactly, because ``SameAccount`` is a plain string
    compare that fails silently on a mismatch."""

    def test_identity_prefers_the_tag_then_the_dir_basename(self) -> None:
        mod = load()
        self.assertEqual(
            mod.dispatch_account_id({"tag": "aug5-netra",
                                     "dir": r"C:\Users\u\.claude-aug5-netra"}),
            "aug5-netra")
        # No tag: the config dir's BASE NAME, exactly as Go's
        # filepath.Base(filepath.Clean(dir)) resolves it. A trailing separator must
        # not shift the answer to "".
        self.assertEqual(
            mod.dispatch_account_id({"dir": "/home/u/.claude-july16-netra"}),
            ".claude-july16-netra")
        self.assertEqual(
            mod.dispatch_account_id({"dir": "/home/u/.claude-july16-netra/"}),
            ".claude-july16-netra")
        # Genuinely anonymous, and the blank/absent forms of both fields.
        self.assertEqual(mod.dispatch_account_id({}), "")
        self.assertEqual(mod.dispatch_account_id(None), "")
        self.assertEqual(mod.dispatch_account_id({"tag": "  ", "dir": " "}), "")

    def test_stamp_never_mutates_the_caller_env(self) -> None:
        mod = load()
        base = {"DISPATCH_LANE": "tools"}
        out = mod.stamp_dispatch_account(base, {"tag": "seat-a"})
        self.assertEqual(out["DISPATCH_ACCOUNT"], "seat-a")
        self.assertEqual(out["DISPATCH_LANE"], "tools")
        self.assertNotIn("DISPATCH_ACCOUNT", base)

    def test_anonymous_seat_drops_an_inherited_identity(self) -> None:
        # An unattributable spawn must NOT inherit this dispatcher's own ambient
        # DISPATCH_ACCOUNT: that would file someone else's wall under our seat.
        mod = load()
        out = mod.stamp_dispatch_account(
            {"DISPATCH_ACCOUNT": "stale-dispatcher-identity"}, None)
        self.assertNotIn("DISPATCH_ACCOUNT", out)

    def test_spawn_stamps_the_serving_account_into_the_child_env(self) -> None:
        # End-to-end over the real spawn seam, no live launch: the child env the
        # dispatcher hands `fak guard` carries a NON-EMPTY account, which is the
        # exact string guard writes into the park record and reads back through
        # Blocks. Proven by capturing Popen's env.
        import tempfile
        from unittest import mock
        mod = load()
        captured: dict[str, object] = {}

        class _FakePopen:
            def __init__(self, argv, **kwargs):
                captured["env"] = kwargs.get("env")
                kwargs.get("stdout").close()  # the parent's log fh (no real child)
                self.pid = 4805

        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            acct = {"tag": "aug5-netra", "tier": 1, "model": "opus",
                    "dir": r"C:\Users\u\.claude-aug5-netra"}
            with mock.patch.object(mod.subprocess, "Popen", _FakePopen):
                res = mod.spawn_issue_worker(
                    ["true"], {"DISPATCH_LANE": "gateway"}, runs, runs,
                    issue=5870, lane="gateway", backend="claude", account=acct)
            env = captured["env"]
            self.assertEqual(env["DISPATCH_ACCOUNT"], "aug5-netra")
            self.assertEqual(env["DISPATCH_LANE"], "gateway")  # base env preserved
            # The env stamp and the on-disk account sidecar must name ONE seat.
            self.assertEqual(res["account"]["tag"], env["DISPATCH_ACCOUNT"])

    def test_spawn_without_an_account_leaves_no_stale_identity(self) -> None:
        import tempfile
        from unittest import mock
        mod = load()
        captured: dict[str, object] = {}

        class _FakePopen:
            def __init__(self, argv, **kwargs):
                captured["env"] = kwargs.get("env")
                kwargs.get("stdout").close()
                self.pid = 11

        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            with mock.patch.object(mod.subprocess, "Popen", _FakePopen):
                mod.spawn_issue_worker(
                    ["true"], {"DISPATCH_ACCOUNT": "stale-dispatcher-identity"},
                    runs, runs, issue=2, lane="tools", backend="claude")
            self.assertNotIn("DISPATCH_ACCOUNT", captured["env"])


class TickExitCodeTest(unittest.TestCase):
    """A scheduled-task LastResult must flag only a genuine malfunction, never a
    benign no-work / backpressure tick (the chronic-0x1 durability bug)."""

    def test_benign_no_work_and_backpressure_exit_zero(self) -> None:
        mod = load()
        for action in ("no_issue", "no_lane", "lane_busy", "lane_leased",
                       "same_issue_wip", "dirty_path_collision",
                       "refused", "weekly_capped", "backend_unhealthy"):
            with self.subTest(action=action):
                # ok=False on every benign verdict, yet the tick ran correctly.
                self.assertEqual(mod.tick_exit_code({"action": action, "ok": False}), 0)

    def test_dispatched_work_exits_zero(self) -> None:
        mod = load()
        self.assertEqual(mod.tick_exit_code({"action": "spawned", "ok": True}), 0)
        self.assertEqual(mod.tick_exit_code({"action": "would_spawn", "ok": True}), 0)

    def test_single_transient_spawn_failed_is_not_red(self) -> None:
        # #2636: a lone SPAWN_FAILED (child died <5s on an otherwise-correct tick) is the
        # ~1-in-25 self-healing baseline — failover retries it, so it must NOT redden
        # LastTaskResult. An un-stamped payload (or streak 1) reads green.
        mod = load()
        self.assertEqual(mod.tick_exit_code({"action": "spawn_failed", "ok": False}), 0)
        self.assertEqual(mod.tick_exit_code(
            {"action": "spawn_failed", "ok": False, "spawn_failed_streak": 1}), 0)
        # just below the threshold stays green (still plausibly transient noise)
        self.assertEqual(mod.tick_exit_code(
            {"action": "spawn_failed", "ok": False,
             "spawn_failed_streak": mod.SPAWN_FAILED_RED_STREAK - 1}), 0)

    def test_repeated_same_target_spawn_failed_is_red(self) -> None:
        # #2636: a SPAWN_FAILED that keeps recurring on the SAME target (failover not
        # healing it) reaches SPAWN_FAILED_RED_STREAK and DOES go red.
        mod = load()
        self.assertEqual(mod.tick_exit_code(
            {"action": "spawn_failed", "ok": False,
             "spawn_failed_streak": mod.SPAWN_FAILED_RED_STREAK}), 1)
        self.assertEqual(mod.tick_exit_code(
            {"action": "spawn_failed", "ok": False,
             "spawn_failed_streak": mod.SPAWN_FAILED_RED_STREAK + 1}), 1)

    def test_spawn_failure_streak_bumps_per_target_and_clears_on_success(self) -> None:
        # The production wiring behind the done condition: the same-target run-length is
        # per-target and per-backend, and a successful spawn resets it (#2636).
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self.assertEqual(mod.bump_spawn_failure_streak(runs, 1515, "claude"), 1)
            self.assertEqual(mod.bump_spawn_failure_streak(runs, 1515, "claude"), 2)
            # a different target keeps its own independent run-length
            self.assertEqual(mod.bump_spawn_failure_streak(runs, 42, "claude"), 1)
            # a different backend never shares the counter
            self.assertEqual(mod.bump_spawn_failure_streak(runs, 1515, "opencode"), 1)
            # a successful spawn of the same target resets it to zero
            mod.clear_spawn_failure_streak(runs, 1515, "claude")
            self.assertEqual(mod.bump_spawn_failure_streak(runs, 1515, "claude"), 1)

    def test_unknown_or_malformed_action_exits_nonzero(self) -> None:
        mod = load()
        # An unrecognised/missing action fails loud rather than masking a new mode.
        self.assertEqual(mod.tick_exit_code({"action": "kaboom"}), 1)
        self.assertEqual(mod.tick_exit_code({}), 1)
        self.assertEqual(mod.tick_exit_code(None), 1)  # type: ignore[arg-type]

    def test_benign_actions_cover_every_non_failure_verdict(self) -> None:
        # Guard against a new evaluate() action silently defaulting to "failure":
        # spawn_failed is the ONLY emitted action outside the benign set.
        mod = load()
        emitted = {"spawned", "would_spawn", "no_issue", "no_lane", "lane_busy",
                   "lane_leased", "same_issue_wip", "dirty_path_collision",
                   "multi_lane_scope", "refused", "weekly_capped",
                   "backend_unhealthy", "backend_health_skip", "seat_cooled",
                   "spawn_failed"}
        self.assertEqual(emitted - mod.BENIGN_ACTIONS, {"spawn_failed"})


class SeatKeyedSpawnFailureStreakTest(unittest.TestCase):
    """#4591 keying fix: the target-keyed pair lets a dead needs_login seat
    cycling across DIFFERENT issues evade cooldown (every target rotation
    restarts that counter at 1); the seat-keyed pair accrues ONE run-length per
    seat regardless of which issue burned the spawn."""

    def test_seat_cycling_across_distinct_issues_accrues_one_streak(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            # The EVASION this test pins: 3 failures on 3 DISTINCT targets never
            # lift any TARGET-keyed run-length above 1 ...
            for target in (101, 202, 303):
                self.assertEqual(
                    mod.bump_spawn_failure_streak(runs, target, "claude"), 1)
            # ... while the SEAT-keyed run-length sees one seat fail 3x in a row.
            self.assertEqual(
                mod.bump_spawn_failure_streak_seat(runs, "july17", "claude"), 1)
            self.assertEqual(
                mod.bump_spawn_failure_streak_seat(runs, "july17", "claude"), 2)
            self.assertEqual(
                mod.bump_spawn_failure_streak_seat(runs, "july17", "claude"), 3)
            state = mod.seat_spawn_failure_state(runs, "july17", "claude")
            self.assertEqual(state["streak"], 3)
            self.assertTrue(state["cooled"])
            self.assertFalse(state["reprobe_due"])  # failure just recorded

    def test_streaks_are_per_seat_and_per_backend(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self.assertEqual(mod.bump_spawn_failure_streak_seat(runs, "a", "claude"), 1)
            self.assertEqual(mod.bump_spawn_failure_streak_seat(runs, "a", "claude"), 2)
            # a different seat keeps its own independent run-length
            self.assertEqual(mod.bump_spawn_failure_streak_seat(runs, "b", "claude"), 1)
            # a different backend never shares the counter
            self.assertEqual(mod.bump_spawn_failure_streak_seat(runs, "a", "opencode"), 1)

    def test_clean_launch_on_the_seat_clears_it_whatever_the_target(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            for _ in range(3):
                mod.bump_spawn_failure_streak_seat(runs, "a", "claude")
            # The clear is SEAT-scoped, not same-target-scoped: any clean launch
            # on the seat breaks the streak.
            mod.clear_spawn_failure_streak_seat(runs, "a", "claude")
            self.assertEqual(mod.bump_spawn_failure_streak_seat(runs, "a", "claude"), 1)

    def test_blank_tag_records_nothing_and_never_cools(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            self.assertEqual(mod.bump_spawn_failure_streak_seat(runs, "", "claude"), 0)
            self.assertEqual(mod.bump_spawn_failure_streak_seat(runs, None, "claude"), 0)
            state = mod.seat_spawn_failure_state(runs, "", "claude")
            self.assertEqual(state["streak"], 0)
            self.assertFalse(state["cooled"])

    def test_cooled_seat_earns_a_reprobe_after_the_window(self) -> None:
        # A permanently-skipped seat could never clear its own streak, so the
        # cool admits ONE probe spawn every SEAT_STREAK_REPROBE_MIN minutes.
        mod = load()
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            t0 = 1_000_000.0
            for _ in range(mod.SPAWN_FAILED_RED_STREAK):
                mod.bump_spawn_failure_streak_seat(runs, "s", "claude", now_ts=t0)
            fresh = mod.seat_spawn_failure_state(runs, "s", "claude", now_ts=t0 + 60)
            self.assertTrue(fresh["cooled"])
            self.assertFalse(fresh["reprobe_due"])
            later = mod.seat_spawn_failure_state(
                runs, "s", "claude",
                now_ts=t0 + mod.SEAT_STREAK_REPROBE_MIN * 60 + 1)
            self.assertTrue(later["cooled"])
            self.assertTrue(later["reprobe_due"])


class SpawnIssueWorkerWorktreeTest(unittest.TestCase):
    """#3181: spawn_issue_worker edits in a per-worker worktree pinned at base_sha
    when the flag is on, fails open to the shared-trunk cwd when off."""

    def _fake_git(self):
        def git(root, args):
            if args and args[0] == "rev-parse":
                return 0, "deadbeef\n"
            return 0, ""
        return git

    def _spawn(self, env_overrides, td):
        m = load()
        seen: dict[str, object] = {}

        class FakeProc:
            pid = 5150

        def fake_popen(argv, cwd=None, env=None, **kw):
            seen["cwd"] = cwd
            return FakeProc()

        runs = Path(td) / "runs"
        with mock.patch.object(m.subprocess, "Popen", fake_popen), \
             mock.patch.dict(os.environ, env_overrides, clear=False):
            res = m.spawn_issue_worker(
                ["python", "-c", "pass"], {"PATH": os.environ.get("PATH", "")},
                Path(td), runs, 3181, "tools", "claude",
                lease={"acquired": True, "id": "L1", "holder": "h",
                       "tree": ["tools/**"]},
                base_sha="basecafe", worktree_git=self._fake_git())
        return m, seen, res, runs

    def test_worktree_spawn_off_uses_root_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _, seen, res, runs = self._spawn({"FLEET_WORKER_WORKTREE": "0"}, td)
            self.assertEqual(seen["cwd"], str(Path(td)))
            self.assertNotIn("worktree", res)
            self.assertEqual(list(runs.glob("*.worktree")), [])  # off -> no sidecar

    def test_worktree_spawn_on_uses_worktree_cwd_pinned_at_base_sha(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            m, seen, res, runs = self._spawn(
                {"FLEET_WORKER_WORKTREE": "1",
                 "FLEET_WORKER_WORKTREE_ROOT": str(Path(td) / "wt")}, td)
            self.assertNotEqual(seen["cwd"], str(Path(td)))
            self.assertTrue(m.worker_worktree.is_worker_worktree(str(seen["cwd"])))
            self.assertEqual(res.get("worktree"), seen["cwd"])
            # Sidecar contract the shared witness sweep consumes: a PLAIN worktree path,
            # the pinned base, and the lease tree for a scoped land.
            wt_side = list(runs.glob("*.worktree"))
            self.assertEqual(len(wt_side), 1)
            self.assertEqual(wt_side[0].read_text(encoding="utf-8"), seen["cwd"])
            base_side = list(runs.glob("*.basesha"))
            self.assertEqual(base_side[0].read_text(encoding="utf-8"), "basecafe")
            tree_side = list(runs.glob("*.lease-tree.json"))
            self.assertEqual(json.loads(tree_side[0].read_text(encoding="utf-8")),
                             ["tools/**"])


class ImplementationFirstProgressTests(unittest.TestCase):
    def test_process_only_trace_stalls_and_preserves_compact_summary(self) -> None:
        mod = load()
        trace = "\n".join([
            'tool_call functions.update_plan',
            'tool_call mcp__fak__fak_capabilities',
            'tool_call web_search',
            'finding: existing reaper is the correct integration seam',
        ])
        progress = mod.bounded_worker_progress(
            trace, age_s=301, first_edit_deadline_s=300)
        self.assertEqual(progress["status"], "stalled")
        self.assertEqual(progress["terminal_state"], "STALLED_NO_EDIT")
        self.assertEqual(progress["time_to_first_edit_s"], None)
        self.assertGreater(progress["planning_tool_ratio"], 0)
        self.assertIn("integration seam", mod._compact_investigation_summary(trace)[-1])

    def test_short_investigation_then_edit_test_commit_remains_productive(self) -> None:
        mod = load()
        trace = "\n".join([
            'tool_call functions.exec_command Select-String declared.py',
            'tool_call functions.apply_patch declared.py',
            'tool_call functions.exec_command python tools/declared_test.py',
            'tool_call functions.exec_command git commit -s',
        ])
        progress = mod.bounded_worker_progress(
            trace, age_s=90, first_edit_deadline_s=300)
        self.assertEqual(progress["status"], "committing")
        self.assertIsNone(progress["terminal_state"])
        self.assertEqual(progress["time_to_first_edit_s"], 90)
        self.assertEqual(progress["planning_tool_ratio"], 0)

    def test_no_edit_reaper_stops_worker_before_wall_timeout(self) -> None:
        import tempfile
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            runs = Path(td)
            pid_file = runs / "resolve-8816-20260824-120000.pid"
            pid_file.write_text("4321\n", encoding="utf-8")
            pid_file.with_suffix(".log").write_text(
                "tool_call functions.update_plan\nfinding: reaper seam\n",
                encoding="utf-8")
            os.utime(pid_file, (1000.0, 1000.0))
            killed: list[int] = []
            report = mod.reap_timed_out_workers(
                runs, timeout_s=3600, first_edit_deadline_s=300, live=True,
                now_ts=1301.0, probe=lambda _pid: {"alive": True, "cmdline": ["fak", "guard", "--", "codex", "resolve GitHub issue #8816"]},
                killer=lambda pid: killed.append(pid) or {"ok": True})
            self.assertEqual(killed, [4321])
            self.assertEqual(report["reaped"][0]["reason"], "STALLED_NO_EDIT")
            self.assertEqual(report["reaped"][0]["progress"]["status"], "stalled")
            self.assertIn("reaper seam", report["reaped"][0]["investigation_summary"][-1])

    def test_escalation_requires_a_closed_typed_need(self) -> None:
        mod = load()
        admitted = mod.bounded_worker_progress(
            "ESCALATION_NEED: missing-path", age_s=10, first_edit_deadline_s=300)
        prose = mod.bounded_worker_progress(
            "I need another planner", age_s=10, first_edit_deadline_s=300)
        self.assertEqual(admitted["typed_escalation_need"], "missing-path")
        self.assertIsNone(prose["typed_escalation_need"])


if __name__ == "__main__":
    unittest.main()
