#!/usr/bin/env python3
"""Hermetic tests for tools/proc_resource_guard.py (no real process scan)."""
from __future__ import annotations

import ast
import importlib.util
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "proc_resource_guard.py"


def load():
    spec = importlib.util.spec_from_file_location("proc_resource_guard", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The incident: one llama-cli process with ~129k threads, plus benign processes
# including the NT "System" kernel at ~613 threads (must NOT be flagged at the
# 2000 default).
INCIDENT = [
    {"pid": 38264, "name": "llama-cli", "threads": 129427, "handles": 293, "ws_mb": 9253},
    {"pid": 4, "name": "System", "threads": 613, "handles": 9087, "ws_mb": 14},
    {"pid": 19728, "name": "WindowsTerminal", "threads": 328, "handles": 7526, "ws_mb": 1152},
    {"pid": 113628, "name": "python", "threads": 90, "handles": 719, "ws_mb": 101},
]


class ClassifyTests(unittest.TestCase):
    def setUp(self):
        self.mod = load()

    def test_flags_only_the_runaway(self):
        flagged = self.mod.classify(INCIDENT)
        self.assertEqual([r["pid"] for r in flagged], [38264])
        self.assertIn("threads 129427 > 2000", flagged[0]["reasons"])
        self.assertFalse(flagged[0]["protected"])

    def test_clean_host_flags_nothing(self):
        self.assertEqual(self.mod.classify(INCIDENT[1:]), [])

    def test_missing_thread_dimension_is_not_a_breach(self):
        # macOS-style row where ps could not report nlwp -> threads is None.
        rows = [{"pid": 1, "name": "x", "threads": None, "handles": None, "ws_mb": 50}]
        self.assertEqual(self.mod.classify(rows), [])

    def test_handles_and_ws_dimensions_opt_in(self):
        rows = [{"pid": 7, "name": "leaky", "threads": 10, "handles": 50000, "ws_mb": 40000}]
        self.assertEqual(self.mod.classify(rows), [])  # disabled by default
        by_handles = self.mod.classify(rows, max_handles=10000)
        self.assertEqual(by_handles[0]["pid"], 7)
        self.assertTrue(any("handles" in r for r in by_handles[0]["reasons"]))
        by_ws = self.mod.classify(rows, max_ws_mb=8000)
        self.assertTrue(any("ws_mb" in r for r in by_ws[0]["reasons"]))

    def test_protected_name_marked_but_still_listed(self):
        rows = [{"pid": 4, "name": "System", "threads": 999999, "handles": 1, "ws_mb": 1}]
        flagged = self.mod.classify(rows)
        self.assertTrue(flagged[0]["protected"])

    def test_terminal_host_over_ceiling_is_protected(self):
        # #2227: WindowsTerminal's threads scale with live panes; a busy fleet
        # host crosses the ceiling legitimately. It must surface as a flag but
        # carry protected=True so --enact skips it and the dispatch preflight
        # does not wedge on it.
        rows = [{"pid": 85884, "name": "WindowsTerminal", "threads": 2320,
                 "handles": 44156, "ws_mb": 1741}]
        flagged = self.mod.classify(rows)
        self.assertEqual([r["pid"] for r in flagged], [85884])
        self.assertTrue(flagged[0]["protected"])

    def test_allowlist_exempts_by_name(self):
        rows = [{"pid": 9, "name": "BigDB", "threads": 50000, "handles": 1, "ws_mb": 1}]
        self.assertEqual(self.mod.classify(rows, allow_names=frozenset({"bigdb"})), [])

    def test_protected_pid_set(self):
        rows = [{"pid": 123, "name": "worker", "threads": 50000, "handles": 1, "ws_mb": 1}]
        flagged = self.mod.classify(rows, protected_pids=frozenset({123}))
        self.assertTrue(flagged[0]["protected"])


class CpuPinTests(unittest.TestCase):
    """The opt-in CPU-pin dimension: a single-threaded process pinning one core
    (normal thread/handle count -> invisible to every level dimension)."""

    def setUp(self):
        self.mod = load()

    def test_delta_is_per_core_top_style(self):
        # One core fully used over a 3s window accrues 3 CPU-seconds -> 100%.
        self.assertEqual(self.mod.cpu_pct_delta(10.0, 13.0, 3.0), 100.0)
        # Four cores -> 400%.
        self.assertEqual(self.mod.cpu_pct_delta(10.0, 22.0, 3.0), 400.0)
        # Half a core -> 50%.
        self.assertEqual(self.mod.cpu_pct_delta(0.0, 1.5, 3.0), 50.0)

    def test_delta_guards(self):
        # PID reuse: the counter went backwards -> refuse to attribute (None).
        self.assertIsNone(self.mod.cpu_pct_delta(50.0, 1.0, 3.0))
        # Missing sample on either side -> None (dimension skipped, never a breach).
        self.assertIsNone(self.mod.cpu_pct_delta(None, 5.0, 3.0))
        self.assertIsNone(self.mod.cpu_pct_delta(5.0, None, 3.0))
        # Non-positive window -> None.
        self.assertIsNone(self.mod.cpu_pct_delta(0.0, 3.0, 0.0))

    def test_sustained_is_min_over_windows(self):
        # pid 1 pins both windows (pin); pid 2 pins window-1 only then goes quiet
        # (a legit burst) -> its sustained score is the QUIET window, not the spike.
        snaps = [
            {1: 0.0, 2: 0.0},
            {1: 3.0, 2: 3.0},   # window 1: both at 100%
            {1: 6.0, 2: 3.3},   # window 2: pid1 100%, pid2 ~10%
        ]
        out = self.mod.cpu_pct_sustained(snaps, 3.0)
        self.assertAlmostEqual(out[1], 100.0)
        self.assertAlmostEqual(out[2], 10.0)  # min(100, 10) -> not a pin

    def test_sustained_omits_pid_missing_from_a_window(self):
        # A pid absent from any snapshot (born/died mid-measurement) is omitted,
        # never guessed.
        snaps = [{1: 0.0}, {1: 3.0, 2: 9.0}]
        out = self.mod.cpu_pct_sustained(snaps, 3.0)
        self.assertIn(1, out)
        self.assertNotIn(2, out)  # pid 2 missing from the first snapshot

    def test_sustained_needs_two_samples(self):
        self.assertEqual(self.mod.cpu_pct_sustained([{1: 0.0}], 3.0), {})
        self.assertEqual(self.mod.cpu_pct_sustained([], 3.0), {})

    def test_classify_cpu_dimension_opt_in(self):
        rows = [{"pid": 7, "name": "spinner", "threads": 1, "handles": 10,
                 "ws_mb": 20, "cpu_pct": 140.0}]
        self.assertEqual(self.mod.classify(rows), [])  # disabled by default
        flagged = self.mod.classify(rows, max_cpu_pct=90)
        self.assertEqual(flagged[0]["pid"], 7)
        self.assertTrue(any("cpu" in r for r in flagged[0]["reasons"]))
        self.assertEqual(flagged[0]["cpu_pct"], 140.0)
        self.assertFalse(flagged[0]["protected"])

    def test_classify_missing_cpu_is_not_a_breach(self):
        # A process whose CPU could not be sampled (cpu_pct None) is skipped even
        # with the dimension enabled -- never flagged on absence of evidence.
        rows = [{"pid": 7, "name": "x", "threads": 1, "handles": 1, "ws_mb": 1, "cpu_pct": None}]
        self.assertEqual(self.mod.classify(rows, max_cpu_pct=90), [])

    def test_classify_cpu_protected_marked(self):
        rows = [{"pid": 4, "name": "System", "threads": 1, "handles": 1,
                 "ws_mb": 1, "cpu_pct": 300.0}]
        flagged = self.mod.classify(rows, max_cpu_pct=90)
        self.assertTrue(flagged[0]["protected"])

    def test_cpu_pin_sorts_above_thread_breach(self):
        # A live core-burner outranks a high static thread count for attention.
        rows = [
            {"pid": 1, "name": "manythreads", "threads": 9000, "handles": 1, "ws_mb": 1},
            {"pid": 2, "name": "pin", "threads": 1, "handles": 1, "ws_mb": 1, "cpu_pct": 150.0},
        ]
        flagged = self.mod.classify(rows, max_threads=2000, max_cpu_pct=90)
        self.assertEqual(flagged[0]["pid"], 2)  # the CPU pin first

    def test_build_payload_reaps_cpu_pin(self):
        killed = []
        rows = [{"pid": 5150, "name": "spinner", "threads": 1, "handles": 9,
                 "ws_mb": 12, "cpu_pct": 99.0}]
        payload = self.mod.build_payload(
            rows, max_threads=2000, max_handles=0, max_ws_mb=0, max_cpu_pct=90,
            protected_pids=frozenset(), allow_names=frozenset(),
            enact=True, killer=lambda pid: (killed.append(pid), (True, "SIGKILL sent"))[1],
        )
        self.assertEqual(killed, [5150])
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["thresholds"]["max_cpu_pct"], 90)
        self.assertEqual(payload["flagged"][0]["action"], "killed")

    def test_collect_processes_cpu_enriches_from_samples(self):
        # Hermetic: stub the underlying scan with scripted cumulative-CPU snapshots
        # and a no-op sleeper; the LAST snapshot is returned, annotated with cpu_pct.
        m = self.mod
        snaps = [
            [{"pid": 1, "name": "a", "cpu_s": 0.0, "threads": 1},
             {"pid": 2, "name": "b", "cpu_s": 0.0, "threads": 1}],
            [{"pid": 1, "name": "a", "cpu_s": 3.0, "threads": 1},
             {"pid": 2, "name": "b", "cpu_s": 0.3, "threads": 1}],
            [{"pid": 1, "name": "a", "cpu_s": 6.0, "threads": 1},
             {"pid": 2, "name": "b", "cpu_s": 0.6, "threads": 1}],
        ]
        state = {"i": 0}

        def fake_collect():
            i = state["i"]
            state["i"] += 1
            return snaps[i], ""

        orig = m.collect_processes
        m.collect_processes = fake_collect
        try:
            procs, err = m.collect_processes_cpu(window_sec=3.0, samples=3, sleeper=lambda _s: None)
        finally:
            m.collect_processes = orig
        self.assertEqual(err, "")
        by = {p["pid"]: p["cpu_pct"] for p in procs}
        self.assertAlmostEqual(by[1], 100.0)  # pinned a core both windows
        self.assertAlmostEqual(by[2], 10.0)   # 10% both windows -> not a pin

    def test_collect_processes_cpu_propagates_scan_error(self):
        m = self.mod
        orig = m.collect_processes
        m.collect_processes = lambda: ([], "scan boom")
        try:
            procs, err = m.collect_processes_cpu(window_sec=1.0, samples=2, sleeper=lambda _s: None)
        finally:
            m.collect_processes = orig
        self.assertEqual(err, "scan boom")


class PayloadTests(unittest.TestCase):
    def setUp(self):
        self.mod = load()

    def test_report_only_does_not_kill(self):
        killed = []
        payload = self.mod.build_payload(
            INCIDENT,
            max_threads=2000,
            max_handles=0,
            max_ws_mb=0,
            protected_pids=frozenset(),
            allow_names=frozenset(),
            enact=False,
            killer=lambda pid: (killed.append(pid), (True, "x"))[1],
        )
        self.assertFalse(payload["ok"])  # a runaway is present -> ACTION
        self.assertEqual(killed, [])
        self.assertEqual(payload["flagged"][0]["action"], "report")
        self.assertEqual(payload["enacted"], [])

    def test_enact_kills_non_protected_only(self):
        killed = []

        def killer(pid):
            killed.append(pid)
            return True, "SIGKILL sent"

        rows = INCIDENT + [{"pid": 4, "name": "System", "threads": 999999, "handles": 1, "ws_mb": 1}]
        payload = self.mod.build_payload(
            rows,
            max_threads=2000,
            max_handles=0,
            max_ws_mb=0,
            protected_pids=frozenset(),
            allow_names=frozenset(),
            enact=True,
            killer=killer,
        )
        self.assertEqual(killed, [38264])  # llama-cli killed; System skipped
        actions = {r["name"]: r["action"] for r in payload["flagged"]}
        self.assertEqual(actions["llama-cli"], "killed")
        self.assertEqual(actions["System"], "protected-skip")
        self.assertEqual(payload["enacted"], [{"pid": 38264, "name": "llama-cli", "ok": True, "detail": "SIGKILL sent"}])

    def test_clean_host_is_ok(self):
        payload = self.mod.build_payload(
            INCIDENT[1:],
            max_threads=2000,
            max_handles=0,
            max_ws_mb=0,
            protected_pids=frozenset(),
            allow_names=frozenset(),
            enact=False,
            killer=lambda pid: (True, ""),
        )
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["flagged_count"], 0)

    def test_protected_only_flag_is_not_action(self):
        # A protected process (NT `System`) over the ceiling is reported but can
        # never be reaped, so it must NOT raise ACTION -- otherwise the kernel
        # thread pool transiently crossing 2000 on a busy many-session host
        # produces a perpetual false ACTION in the control pane.
        rows = INCIDENT[2:] + [
            {"pid": 4, "name": "System", "threads": 9000, "handles": 1, "ws_mb": 1}
        ]
        payload = self.mod.build_payload(
            rows,
            max_threads=2000,
            max_handles=0,
            max_ws_mb=0,
            protected_pids=frozenset(),
            allow_names=frozenset(),
            enact=False,
            killer=lambda pid: (True, ""),
        )
        self.assertTrue(payload["ok"])  # protected-only breach is non-actionable
        self.assertEqual(payload["actionable_flagged_count"], 0)
        flagged = {r["name"]: r for r in payload["flagged"]}
        self.assertIn("System", flagged)          # still reported...
        self.assertTrue(flagged["System"]["protected"])  # ...marked protected

    def test_collect_error_is_not_clean(self):
        payload = self.mod.build_payload(
            [],
            max_threads=2000,
            max_handles=0,
            max_ws_mb=0,
            protected_pids=frozenset(),
            allow_names=frozenset(),
            enact=False,
            killer=lambda pid: (True, ""),
            collect_error="boom",
        )
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["collect_error"], "boom")


class ParserTests(unittest.TestCase):
    def setUp(self):
        self.mod = load()

    def test_windows_json_array(self):
        text = (
            '[{"pid":38264,"name":"llama-cli","threads":129427,"handles":293,"ws":9701888000},'
            '{"pid":4,"name":"System","threads":613,"handles":9087,"ws":14680064}]'
        )
        rows = self.mod._parse_windows_json(text)
        self.assertEqual(rows[0]["threads"], 129427)
        self.assertEqual(rows[0]["ws_mb"], 9701888000 // (1024 * 1024))

    def test_windows_json_single_object(self):
        # ConvertTo-Json emits a bare object (not an array) for one process.
        rows = self.mod._parse_windows_json('{"pid":1,"name":"x","threads":5,"handles":2,"ws":1048576}')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["ws_mb"], 1)

    def test_windows_json_empty(self):
        self.assertEqual(self.mod._parse_windows_json(""), [])

    def test_windows_json_includes_cpu_seconds(self):
        rows = self.mod._parse_windows_json(
            '{"pid":1,"name":"x","threads":5,"handles":2,"ws":1048576,"cpu":42.5}'
        )
        self.assertEqual(rows[0]["cpu_s"], 42.5)
        # A row without a cpu field (older scan) -> cpu_s None, not a crash.
        rows2 = self.mod._parse_windows_json('{"pid":1,"name":"x","threads":5,"handles":2,"ws":1048576}')
        self.assertIsNone(rows2[0]["cpu_s"])

    def test_posix_ps(self):
        # Current 5-column format: pid nlwp rss cputimes comm.
        text = "  38264 129427 9474048 8123 llama-cli\n      4   613    14336 0 systemd\n"
        rows = self.mod._parse_posix_ps(text)
        self.assertEqual(rows[0]["pid"], 38264)
        self.assertEqual(rows[0]["threads"], 129427)
        self.assertEqual(rows[0]["ws_mb"], 9474048 // 1024)
        self.assertEqual(rows[0]["cpu_s"], 8123.0)
        self.assertEqual(rows[1]["name"], "systemd")

    def test_posix_ps_backward_compat_four_columns(self):
        # A ps without cputimes (4 columns) still parses; cpu_s is simply absent.
        text = "  38264 129427 9474048 llama-cli\n"
        rows = self.mod._parse_posix_ps(text)
        self.assertEqual(rows[0]["pid"], 38264)
        self.assertEqual(rows[0]["threads"], 129427)
        self.assertEqual(rows[0]["name"], "llama-cli")
        self.assertIsNone(rows[0]["cpu_s"])

    def test_posix_ps_space_in_comm(self):
        # cputimes stays its own field even when comm carries a space (a kernel
        # thread like "my worker"): split(None, 4) keeps the trailing comm whole.
        text = "  10 2 4096 5 my worker\n"
        rows = self.mod._parse_posix_ps(text)
        self.assertEqual(rows[0]["cpu_s"], 5.0)
        self.assertEqual(rows[0]["threads"], 2)
        self.assertEqual(rows[0]["name"], "my worker")  # basename keeps the whole comm

    def test_kill_pid_rejects_bad_pid(self):
        ok, detail = self.mod.kill_pid(0)
        self.assertFalse(ok)


class OrphanClassifyTests(unittest.TestCase):
    def setUp(self):
        self.mod = load()

    # An orphaned dos_mcp.server (owner pid 999 dead) next to a live-owned one
    # (owner pid 100 alive). Only the orphan should flag.
    ORPHANS = [
        {"pid": 20044, "name": "python", "ppid": 100, "cmdline": "python -m dos_mcp.server", "age_sec": 600},
        {"pid": 36252, "name": "python", "ppid": 999, "cmdline": "python -m dos_mcp.server", "age_sec": 600},
        {"pid": 100, "name": "claude", "ppid": 50, "cmdline": "claude", "age_sec": 600},
    ]

    def test_flags_orphaned_mcp_only(self):
        flagged = self.mod.classify_orphans(
            self.ORPHANS,
            live_pids=frozenset({20044, 36252, 100, 50}),
            child_counts=self.mod._child_counts(self.ORPHANS),
            orphan_patterns=("dos_mcp.server",),
        )
        self.assertEqual([r["pid"] for r in flagged], [36252])
        self.assertEqual(flagged[0]["kind"], "orphan-helper")
        self.assertIn("owner pid 999 not alive", flagged[0]["reasons"][0])

    def test_pattern_miss_flags_nothing(self):
        rows = [{"pid": 7, "name": "python", "ppid": 999, "cmdline": "python -m something_else", "age_sec": 1}]
        self.assertEqual(
            self.mod.classify_orphans(rows, live_pids=frozenset(), orphan_patterns=("dos_mcp.server",)),
            [],
        )

    def test_reparented_to_init_is_orphan(self):
        # POSIX: owner died, init (pid 1) adopted the helper -> ppid 1 == orphaned.
        rows = [{"pid": 7, "name": "python", "ppid": 1, "cmdline": "python -m dos_mcp.server", "age_sec": 9}]
        flagged = self.mod.classify_orphans(rows, live_pids=frozenset({1, 7}), orphan_patterns=("dos_mcp.server",))
        self.assertEqual([r["pid"] for r in flagged], [7])

    def test_pid_reuse_spares_helper(self):
        # ppid 100 is alive (reused by an unrelated proc) -> conservatively spared.
        rows = [{"pid": 7, "name": "python", "ppid": 100, "cmdline": "python -m dos_mcp.server", "age_sec": 9}]
        self.assertEqual(
            self.mod.classify_orphans(rows, live_pids=frozenset({100, 7}), orphan_patterns=("dos_mcp.server",)),
            [],
        )

    def test_idle_shell_opt_in_and_aged(self):
        rows = [{"pid": 31736, "name": "pwsh", "ppid": 9, "cmdline": "pwsh", "age_sec": 4000}]
        # disabled by default
        self.assertEqual(
            self.mod.classify_orphans(rows, live_pids=frozenset({9}), child_counts={}),
            [],
        )
        flagged = self.mod.classify_orphans(
            rows, live_pids=frozenset({9}), child_counts={}, reap_idle_shells=True,
            idle_shell_names=self.mod.DEFAULT_IDLE_SHELL_NAMES, min_age_sec=1800,
        )
        self.assertEqual(flagged[0]["pid"], 31736)
        self.assertEqual(flagged[0]["kind"], "idle-shell")

    def test_idle_shell_with_children_spared(self):
        rows = [{"pid": 31736, "name": "pwsh", "ppid": 9, "cmdline": "pwsh", "age_sec": 4000}]
        flagged = self.mod.classify_orphans(
            rows, live_pids=frozenset({9}), child_counts={31736: 1}, reap_idle_shells=True,
            idle_shell_names=self.mod.DEFAULT_IDLE_SHELL_NAMES, min_age_sec=1800,
        )
        self.assertEqual(flagged, [])

    def test_idle_shell_too_young_spared(self):
        rows = [{"pid": 31736, "name": "pwsh", "ppid": 9, "cmdline": "pwsh", "age_sec": 60}]
        flagged = self.mod.classify_orphans(
            rows, live_pids=frozenset({9}), child_counts={}, reap_idle_shells=True,
            idle_shell_names=self.mod.DEFAULT_IDLE_SHELL_NAMES, min_age_sec=1800,
        )
        self.assertEqual(flagged, [])

    def test_idle_shell_parented_by_terminal_spared(self):
        rows = [
            {"pid": 9, "name": "WindowsTerminal", "ppid": 1, "cmdline": "wt", "age_sec": 5000},
            {"pid": 31736, "name": "pwsh", "ppid": 9, "cmdline": "pwsh", "age_sec": 4000},
        ]
        flagged = self.mod.classify_orphans(
            rows, live_pids=frozenset({1, 9, 31736}), child_counts=self.mod._child_counts(rows),
            parent_names=self.mod._parent_names(rows), reap_idle_shells=True,
            idle_shell_names=self.mod.DEFAULT_IDLE_SHELL_NAMES, min_age_sec=1800,
        )
        self.assertEqual(flagged, [])

    def test_idle_shell_parented_by_background_launcher_flags(self):
        rows = [
            {"pid": 9, "name": "codex", "ppid": 1, "cmdline": "codex", "age_sec": 5000},
            {"pid": 31736, "name": "pwsh", "ppid": 9, "cmdline": "pwsh", "age_sec": 4000},
        ]
        flagged = self.mod.classify_orphans(
            rows, live_pids=frozenset({1, 9, 31736}), child_counts=self.mod._child_counts(rows),
            parent_names=self.mod._parent_names(rows), reap_idle_shells=True,
            idle_shell_names=self.mod.DEFAULT_IDLE_SHELL_NAMES, min_age_sec=1800,
        )
        self.assertEqual([r["pid"] for r in flagged], [31736])
        self.assertEqual(flagged[0]["parent_name"], "codex")

    def test_orphan_console_shell_with_only_conhost_child_flags(self):
        rows = [
            {"pid": 10, "name": "cmd", "ppid": 999, "cmdline": "", "age_sec": 4000},
            {"pid": 11, "name": "conhost", "ppid": 10, "cmdline": "", "age_sec": 4000},
        ]
        flagged = self.mod.classify_orphans(
            rows, live_pids=frozenset({10, 11}), child_counts=self.mod._child_counts(rows),
            child_names=self.mod._child_names(rows), parent_names=self.mod._parent_names(rows),
            reap_idle_shells=True, idle_shell_names=self.mod.DEFAULT_IDLE_SHELL_NAMES,
            min_age_sec=1800,
        )
        self.assertEqual([r["pid"] for r in flagged], [10])
        self.assertEqual(flagged[0]["kind"], "orphan-console-shell")

    def test_orphan_console_shell_with_real_child_spared(self):
        rows = [
            {"pid": 10, "name": "cmd", "ppid": 999, "cmdline": "", "age_sec": 4000},
            {"pid": 11, "name": "python", "ppid": 10, "cmdline": "python worker.py", "age_sec": 4000},
        ]
        flagged = self.mod.classify_orphans(
            rows, live_pids=frozenset({10, 11}), child_counts=self.mod._child_counts(rows),
            child_names=self.mod._child_names(rows), parent_names=self.mod._parent_names(rows),
            reap_idle_shells=True, idle_shell_names=self.mod.DEFAULT_IDLE_SHELL_NAMES,
            min_age_sec=1800,
        )
        self.assertEqual(flagged, [])

    def test_orphan_protected_name_marked(self):
        rows = [{"pid": 4, "name": "csrss", "ppid": 999, "cmdline": "dos_mcp.server", "age_sec": 9}]
        flagged = self.mod.classify_orphans(rows, live_pids=frozenset(), orphan_patterns=("dos_mcp.server",))
        self.assertTrue(flagged[0]["protected"])

    def test_orphan_allowlist_exempts(self):
        rows = [{"pid": 7, "name": "python", "ppid": 999, "cmdline": "dos_mcp.server", "age_sec": 9}]
        self.assertEqual(
            self.mod.classify_orphans(
                rows, live_pids=frozenset(), orphan_patterns=("dos_mcp.server",),
                allow_names=frozenset({"python"}),
            ),
            [],
        )


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.mod = load()

    def test_merge_unions_reasons_by_pid(self):
        resource = [{"pid": 5, "name": "x", "threads": 9000, "handles": None, "ws_mb": None,
                     "reasons": ["threads 9000 > 2000"], "protected": False}]
        orphan = [{"pid": 5, "name": "x", "ppid": 1, "threads": 9000, "handles": None, "ws_mb": None,
                   "reasons": ["orphaned helper: owner pid 1 not alive"], "protected": False, "kind": "orphan-helper"}]
        merged = self.mod._merge_flagged(resource, orphan)
        self.assertEqual(len(merged), 1)
        self.assertEqual(len(merged[0]["reasons"]), 2)
        self.assertEqual(merged[0]["kind"], "orphan-helper")

    def test_merge_protected_is_or(self):
        resource = [{"pid": 5, "name": "x", "threads": 1, "handles": None, "ws_mb": None,
                     "reasons": ["a"], "protected": True}]
        orphan = [{"pid": 5, "name": "x", "threads": 1, "handles": None, "ws_mb": None,
                   "reasons": ["b"], "protected": False, "kind": "idle-shell"}]
        self.assertTrue(self.mod._merge_flagged(resource, orphan)[0]["protected"])

    def test_build_payload_reaps_orphan(self):
        killed = []
        orphan = self.mod.classify_orphans(
            [{"pid": 36252, "name": "python", "ppid": 999, "cmdline": "python -m dos_mcp.server", "age_sec": 9}],
            live_pids=frozenset(), orphan_patterns=("dos_mcp.server",),
        )
        payload = self.mod.build_payload(
            [], max_threads=2000, max_handles=0, max_ws_mb=0,
            protected_pids=frozenset(), allow_names=frozenset(),
            enact=True, killer=lambda pid: (killed.append(pid), (True, "SIGKILL sent"))[1],
            orphan_rows=orphan,
        )
        self.assertEqual(killed, [36252])
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["flagged"][0]["action"], "killed")


class RelationsParserTests(unittest.TestCase):
    def setUp(self):
        self.mod = load()

    def test_windows_relations_strips_exe_and_age(self):
        text = (
            '[{"pid":36252,"ppid":999,"name":"python.exe","cmd":"python -m dos_mcp.server","age":600},'
            '{"pid":4,"ppid":0,"name":"System","cmd":null,"age":-1}]'
        )
        rows = self.mod._parse_windows_relations(text)
        self.assertEqual(rows[0]["name"], "python")
        self.assertEqual(rows[0]["age_sec"], 600)
        self.assertEqual(rows[0]["cmdline"], "python -m dos_mcp.server")
        self.assertEqual(rows[1]["name"], "System")
        self.assertIsNone(rows[1]["age_sec"])  # -1 sentinel -> None

    def test_windows_relations_single_object(self):
        rows = self.mod._parse_windows_relations('{"pid":1,"ppid":0,"name":"x.exe","cmd":"x","age":5}')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["name"], "x")

    def test_posix_relations(self):
        text = "20044 100 600 python python -m dos_mcp.server\n  4 0 14000 systemd /sbin/init\n"
        rows = self.mod._parse_posix_ps_relations(text)
        self.assertEqual(rows[0]["pid"], 20044)
        self.assertEqual(rows[0]["ppid"], 100)
        self.assertEqual(rows[0]["age_sec"], 600)
        self.assertIn("dos_mcp.server", rows[0]["cmdline"])
        self.assertEqual(rows[1]["name"], "systemd")

    def test_child_counts(self):
        rows = [{"pid": 1, "ppid": 0}, {"pid": 2, "ppid": 1}, {"pid": 3, "ppid": 1}]
        self.assertEqual(self.mod._child_counts(rows), {0: 1, 1: 2})


class CpuReapConfirmTests(unittest.TestCase):
    """Cross-tick confirmation: a CPU-ONLY pin is auto-reaped only after it persists
    across N consecutive runs, keyed by (pid+start) so a recycled pid cannot inherit a
    dead process's streak. The safety proofs for the standing auto-killer."""

    def setUp(self):
        self.mod = load()

    def _kpayload(self, rows, *, confirm, prev, orphan_rows=None):
        killed = []
        payload = self.mod.build_payload(
            rows, max_threads=2000, max_handles=0, max_ws_mb=0, max_cpu_pct=90,
            cpu_reap_confirm=confirm, cpu_streaks_prev=prev,
            protected_pids=frozenset(), allow_names=frozenset(),
            enact=True, killer=lambda pid: (killed.append(pid), (True, "SIGKILL sent"))[1],
            orphan_rows=orphan_rows,
        )
        return killed, payload

    def test_streak_key_includes_start(self):
        self.assertEqual(self.mod.cpu_streak_key(100, "2026-01-01T00:00:00Z"), "100:2026-01-01T00:00:00Z")
        self.assertEqual(self.mod.cpu_streak_key(100, None), "100:")
        self.assertEqual(self.mod.cpu_streak_key(100, ""), "100:")

    def test_bump_streaks_increments_and_drops(self):
        prev = {"10:A": 2, "11:A": 5}
        out = self.mod._bump_cpu_streaks(prev, ["10:A", "12:A"])
        self.assertEqual(out, {"10:A": 3, "12:A": 1})  # 10 bumped, 12 new, 11 dropped

    def test_cpu_only_unconfirmed_is_not_killed(self):
        rows = [{"pid": 50, "name": "spin", "threads": 1, "handles": 9, "ws_mb": 12, "cpu_pct": 99.0, "start": "S1"}]
        killed, payload = self._kpayload(rows, confirm=2, prev={})  # streak -> 1 < 2
        self.assertEqual(killed, [])
        row = payload["flagged"][0]
        self.assertEqual(row["action"], "cpu-unconfirmed")
        self.assertEqual(row["cpu_streak"], 1)
        self.assertFalse(payload["ok"])  # still ACTION -- a pin is present
        self.assertEqual(payload["cpu_streaks"], {"50:S1": 1})

    def test_cpu_only_confirmed_is_killed(self):
        rows = [{"pid": 50, "name": "spin", "threads": 1, "handles": 9, "ws_mb": 12, "cpu_pct": 99.0, "start": "S1"}]
        killed, payload = self._kpayload(rows, confirm=2, prev={"50:S1": 1})  # streak -> 2 >= 2
        self.assertEqual(killed, [50])
        self.assertEqual(payload["flagged"][0]["action"], "killed")

    def test_pid_reuse_does_not_inherit_streak(self):
        # pid 50 was confirmed (streak 1) for a process started at S1; it died and pid 50
        # is reused by a NEW process started at S2. The new process must NOT inherit the
        # old streak -- it gets a fresh key "50:S2" -> streak 1 < 2 -> NOT killed.
        rows = [{"pid": 50, "name": "spin", "threads": 1, "handles": 9, "ws_mb": 12, "cpu_pct": 99.0, "start": "S2"}]
        killed, payload = self._kpayload(rows, confirm=2, prev={"50:S1": 1})
        self.assertEqual(killed, [])
        self.assertEqual(payload["flagged"][0]["action"], "cpu-unconfirmed")
        self.assertEqual(payload["cpu_streaks"], {"50:S2": 1})  # old "50:S1" dropped

    def test_cpu_plus_thread_breach_reaps_immediately(self):
        # CPU *and* a thread runaway is unambiguous -> killed now, even with high confirm
        # and no prior streak (not cpu-only).
        rows = [{"pid": 51, "name": "bad", "threads": 9000, "handles": 9, "ws_mb": 12, "cpu_pct": 150.0, "start": "S"}]
        killed, payload = self._kpayload(rows, confirm=99, prev={})
        self.assertEqual(killed, [51])
        self.assertEqual(payload["flagged"][0]["action"], "killed")

    def test_orphan_reaps_immediately_regardless_of_confirm(self):
        orphan = self.mod.classify_orphans(
            [{"pid": 36252, "name": "python", "ppid": 999, "cmdline": "python -m dos_mcp.server", "age_sec": 9}],
            live_pids=frozenset(), orphan_patterns=("dos_mcp.server",),
        )
        killed, payload = self._kpayload([], confirm=99, prev={}, orphan_rows=orphan)
        self.assertEqual(killed, [36252])  # orphan is unambiguous -> not gated by confirm

    def test_protected_cpu_pin_never_killed_even_confirmed(self):
        rows = [{"pid": 4, "name": "System", "threads": 1, "handles": 1, "ws_mb": 1, "cpu_pct": 300.0, "start": "S"}]
        killed, payload = self._kpayload(rows, confirm=1, prev={"4:S": 9})
        self.assertEqual(killed, [])
        self.assertEqual(payload["flagged"][0]["action"], "protected-skip")

    def test_confirm_default_one_kills_on_first_detection(self):
        rows = [{"pid": 52, "name": "spin", "threads": 1, "handles": 1, "ws_mb": 1, "cpu_pct": 99.0, "start": "S"}]
        killed, _ = self._kpayload(rows, confirm=1, prev={})
        self.assertEqual(killed, [52])  # confirm=1 == legacy first-detection reap

    def test_report_mode_annotates_streak_without_killing(self):
        rows = [{"pid": 53, "name": "spin", "threads": 1, "handles": 1, "ws_mb": 1, "cpu_pct": 99.0, "start": "S"}]
        payload = self.mod.build_payload(
            rows, max_threads=2000, max_handles=0, max_ws_mb=0, max_cpu_pct=90,
            cpu_reap_confirm=2, cpu_streaks_prev={"53:S": 4},
            protected_pids=frozenset(), allow_names=frozenset(),
            enact=False, killer=lambda pid: (True, ""),
        )
        self.assertEqual(payload["flagged"][0]["action"], "report")
        self.assertEqual(payload["flagged"][0]["cpu_streak"], 5)

    def test_ledger_round_trip_and_corruption_safe(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            self.assertEqual(self.mod.load_cpu_streaks(p), {})  # absent -> empty
            self.mod.save_cpu_streaks(p, {"7:S": 3})
            self.assertEqual(self.mod.load_cpu_streaks(p), {"7:S": 3})
            (p / self.mod.CPU_STREAK_LEDGER).write_text("{not json", encoding="utf-8")
            self.assertEqual(self.mod.load_cpu_streaks(p), {})  # corrupt -> empty, no raise


# --------------------------------------------------------------------------- #
# `ps` dialect gate (#5541)
# --------------------------------------------------------------------------- #
# The Python half of internal/architest/ps_dialect_test.go. THAT gate is an AST gate
# over Go source and structurally cannot see Python, which is why three Python `ps`
# call sites still shipped the pre-#5385 argv verbatim long after the Go half was
# fixed. Repairing them by hand repairs today; this is what stops the fourth copy.
#
# The defect being gated is NOT "the keyword is unportable". It is that an
# UNMEASURABLE dimension came back indistinguishable from a measured zero: a `ps` that
# rejects one requested keyword rejects the whole invocation, the collector returned
# rows=[] paired with an EMPTY error string, and the guard printed scanned=0 / ok=true
# / "no runaway or orphaned process; no action" for a host it had not measured at all.

# Keywords procps-ng defines and BSD `ps` does not, with the reason each has no
# portable spelling. Kept in step with ps_dialect_test.go's procpsOnlyKeywords.
PROCPS_ONLY_KEYWORDS = {
    "nlwp": "thread count; BSD ps has no thread-count keyword at all",
    "etimes": "elapsed seconds; BSD spells it etime, FORMATTED [[dd-]hh:]mm:ss",
    "cputimes": "cumulative CPU seconds; BSD spells it time, likewise formatted",
}

# Files allowed to hand `ps` a procps-only keyword ALONGSIDE portable ones. An entry
# is EARNED by a test that pins the non-procps behaviour, exactly as internal/procguard
# earns its exemption in the Go gate; a bare entry with nothing pinning it is how this
# gate would rot into a rubber stamp.
PS_DIALECT_AWARE = {
    "proc_resource_guard.py":
        "branches on platform.system(): PsDialectSpecTests pins the BSD argv "
        "keyword-for-keyword and PsCensusErrorTests pins that a rejected `ps` reports "
        "a named error instead of a clean host",
}

_PS_COLUMN_TOKEN = re.compile(r"^[a-z_]+=$")


def ps_column_literals(source: str) -> list[dict]:
    """Every `ps -o` column-list string literal in a Python source that names a
    procps-ng-only keyword.

    String LITERALS only (walked with ast), so a `#` comment discussing the two
    dialects is not a finding -- a gate that argues with prose gets switched off.

    ``fragment`` marks a literal that starts or ends with a comma: a piece of a
    concatenated argv whose remaining columns live in a sibling literal. Such a piece
    can never be judged "a lone column" on its own, which is what stops the rule below
    being evaded by splitting one argv across two strings.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # A peer's file caught mid-edit is not this gate's business.
        return []
    hits: list[dict] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        raw = node.value
        tokens = [t for t in raw.split(",") if t]
        if not tokens or not all(_PS_COLUMN_TOKEN.match(t) for t in tokens):
            continue
        columns = [t[:-1] for t in tokens]
        procps = sorted(c for c in columns if c in PROCPS_ONLY_KEYWORDS)
        if not procps:
            continue
        fragment = raw.startswith(",") or raw.endswith(",")
        hits.append({
            "literal": raw,
            "line": node.lineno,
            "columns": columns,
            "procps_only": procps,
            "fragment": fragment,
            # A lone, complete column list loses ONLY the dimension its keyword names
            # when a `ps` rejects it. Mixed with portable columns, that same rejection
            # takes the portable readings down too and turns a one-dimension gap into
            # a whole-census silent zero -- the actual bug.
            "lone": (not fragment) and len(columns) == 1,
        })
    return hits


class PsDialectSourceGateTests(unittest.TestCase):
    def test_no_tools_script_mixes_a_procps_only_keyword_with_portable_columns(self):
        offenders = []
        for path in sorted((ROOT / "tools").glob("*.py")):
            try:
                source = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if path.name in PS_DIALECT_AWARE:
                continue
            for hit in ps_column_literals(source):
                if hit["lone"]:
                    continue
                offenders.append("%s:%d %r names %s" % (
                    path.name, hit["line"], hit["literal"], ",".join(hit["procps_only"])))
        self.assertEqual(offenders, [], "\n".join([
            "a `ps` argv mixes a procps-ng-only keyword with portable columns:",
            *offenders,
            "cure: branch the argv on platform.system() the way",
            "tools/proc_resource_guard.py::_ps_census_spec does, leave the column the",
            "other dialect lacks as PS_NO_COLUMN (None, never 0), and add the file to",
            "PS_DIALECT_AWARE with the test that pins its non-procps behaviour.",
        ]))

    def test_every_dialect_aware_entry_still_has_something_to_excuse(self):
        # An allowlist that outlives the call site it excuses is how the next copy gets
        # waved through.
        for name in PS_DIALECT_AWARE:
            hits = ps_column_literals((ROOT / "tools" / name).read_text(encoding="utf-8"))
            self.assertTrue(
                any(not h["lone"] for h in hits),
                "%s no longer mixes a procps-only keyword with portable columns; drop "
                "its PS_DIALECT_AWARE entry" % name)

    def test_detector_catches_a_mixed_argv_and_clears_a_portable_one(self):
        # Non-vacuity for the detector itself. The offending column list is ASSEMBLED at
        # runtime so this test file does not itself contain the literal it hunts for.
        mixed = "pid=," + "nlwp" + "=,rss=,comm="
        hits = ps_column_literals('ARGS = ["ps", "-eo", %r]' % mixed)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["procps_only"], ["nlwp"])
        self.assertFalse(hits[0]["lone"])
        # The BSD spelling of the same census is clean.
        self.assertEqual(ps_column_literals('ARGS = ["ps", "-eo", "pid=,rss=,time=,comm="]'), [])

    def test_detector_treats_a_lone_column_and_a_concatenated_piece_differently(self):
        lone = ps_column_literals('ARGS = ["ps", "-eo", %r]' % ("nlwp" + "="))
        self.assertTrue(lone[0]["lone"])  # only the thread dimension can be lost
        piece = ps_column_literals('ARGS = ["ps", "-eo", "pid=,ppid=," + %r + "comm="]'
                                   % ("etimes" + "=,"))
        self.assertTrue(piece and not piece[0]["lone"])  # a fragment is never "lone"


class PsDialectSpecTests(unittest.TestCase):
    """The BSD argv must name no keyword BSD `ps` cannot answer, and the Linux argv
    must keep the one that matters. Reverting _ps_census_spec to a single unconditional
    procps argv fails the first of these; "fixing" BSD by dropping nlwp everywhere
    fails the second."""

    def setUp(self):
        self.mod = load()

    def _args(self, spec):
        return " ".join(spec["args"])

    def test_module_keyword_list_matches_the_gate(self):
        # The module names the same three keywords the gate hunts for. A list that
        # drifts from the gate's is a gate that has stopped seeing what the code does.
        self.assertEqual(sorted(self.mod.PS_PROCPS_ONLY_KEYWORDS),
                         sorted(PROCPS_ONLY_KEYWORDS))

    def test_bsd_census_argv_names_no_procps_only_keyword(self):
        args = self._args(self.mod._ps_census_spec("Darwin"))
        for keyword, why in PROCPS_ONLY_KEYWORDS.items():
            self.assertNotIn(keyword, args, "BSD census argv names %s -- %s" % (keyword, why))

    def test_bsd_relations_argv_names_no_procps_only_keyword(self):
        args = self._args(self.mod._ps_relations_spec("Darwin"))
        for keyword, why in PROCPS_ONLY_KEYWORDS.items():
            self.assertNotIn(keyword, args, "BSD relations argv names %s -- %s" % (keyword, why))

    def test_linux_census_argv_keeps_the_thread_keyword(self):
        # The control. Fixing BSD by blinding Linux would disable the thread dimension
        # this guard exists for (the incident was one process at ~129,427 threads).
        self.assertIn("nlwp", self._args(self.mod._ps_census_spec("Linux")))
        self.assertIn("etimes", self._args(self.mod._ps_relations_spec("Linux")))

    def test_unwitnessed_platform_keeps_the_invocation_known_to_work(self):
        # Only Darwin was actually witnessed rejecting the keywords; an unwitnessed
        # POSIX name keeps the procps argv, and _census_error is what stops such a host
        # reading as an empty machine in the meantime.
        self.assertFalse(self.mod._ps_bsd("Linux"))
        self.assertTrue(self.mod._ps_bsd("Darwin"))

    def test_bsd_row_leaves_threads_unmeasured_not_zero(self):
        # BSD census table: pid rss time comm.
        rows = self.mod._parse_posix_ps("  1 2048 01:23 launchd\n",
                                        self.mod._ps_census_spec("Darwin"))
        self.assertEqual(rows[0]["pid"], 1)
        self.assertEqual(rows[0]["ws_mb"], 2)
        self.assertIsNone(rows[0]["threads"])       # named absent ...
        self.assertNotEqual(rows[0]["threads"], 0)  # ... never a measurement of zero
        self.assertEqual(rows[0]["cpu_s"], 83.0)    # mm:ss read as seconds, not dropped

    def test_bsd_relations_age_is_parsed_not_zeroed(self):
        rows = self.mod._parse_posix_ps_relations(
            "20044 100 2-03:04:05 python python -m dos_mcp.server\n",
            self.mod._ps_relations_spec("Darwin"))
        self.assertEqual(rows[0]["age_sec"], 2 * 86400 + 3 * 3600 + 4 * 60 + 5)
        self.assertIn("dos_mcp.server", rows[0]["cmdline"])

    def test_unreadable_duration_column_is_absent_not_zero(self):
        # A `-` placeholder, a leaked header, an unexpected dialect. Zero age means
        # "started this instant" and zero CPU means "never ran"; both are claims.
        for bad in ("-", "", "  ", "nan", "1e9", "1:2:3:4", "-5"):
            self.assertIsNone(self.mod._parse_ps_duration(bad), repr(bad))
        self.assertEqual(self.mod._parse_ps_duration("8123"), 8123.0)  # procps: bare seconds


class PsCensusErrorTests(unittest.TestCase):
    """A `ps` that yielded nothing must be reported as NO CENSUS, never as a quiet host.

    The witnessed shape (a real non-procps `ps`, the MSYS2 build shipped with Git for
    Windows, which does not implement -o at all):

        $ ps -eo pid=,nlwp=,rss=,cputimes=,comm=
        ps: unknown option -- o          # on stderr
        ; exit 1, stdout empty

    Before #5541 that reached the payload as rows=[] with collect_error="", so ok was
    True, scanned was 0 and next_action was "no runaway or orphaned process; no
    action". Reverting _census_error to an unconditional "" fails every test here.
    """

    REJECTED = ("", "exit status 1: ps: unknown option -- o")

    def setUp(self):
        self.mod = load()

    def _with_tool(self, stdout, error, fn, *args):
        original = self.mod._run_tool
        self.mod._run_tool = lambda *_a, **_k: (stdout, error)
        try:
            return fn(*args)
        finally:
            self.mod._run_tool = original

    def test_rejected_census_is_a_named_error_not_a_clean_host(self):
        rows, err = self._with_tool(*self.REJECTED, self.mod._collect_posix, "Linux")
        self.assertEqual(rows, [])
        self.assertTrue(err, "a `ps` that produced no census must say so")
        self.assertIn("unknown option", err)  # the sentence that names the bug on sight

    def test_rejected_relations_is_a_named_error_not_a_quiet_host(self):
        rows, err = self._with_tool(*self.REJECTED, self.mod._collect_posix_relations, "Linux")
        self.assertEqual(rows, [])
        self.assertTrue(err)

    def test_payload_from_a_rejected_census_is_not_ok(self):
        _rows, err = self._with_tool(*self.REJECTED, self.mod._collect_posix, "Linux")
        payload = self.mod.build_payload(
            [], max_threads=2000, max_handles=0, max_ws_mb=0, max_cpu_pct=0,
            cpu_reap_confirm=1, cpu_streaks_prev={}, protected_pids=frozenset(),
            allow_names=frozenset(), enact=False, killer=lambda pid: (True, ""),
            collect_error=err, orphan_rows=[],
        )
        self.assertFalse(payload["ok"])
        self.assertIsNotNone(payload["collect_error"])
        self.assertIn("scan failed", payload["next_action"])

    def test_a_ps_that_printed_rows_then_failed_is_still_a_census(self):
        # The opposite direction, and it matters just as much: ok is computed as
        # `not collect_error`, so keeping the error here would turn a host whose census
        # WORKED into a permanent ACTION.
        rows, err = self._with_tool(
            "  38264 129427 9474048 8123 llama-cli\n", "exit status 1: some moan",
            self.mod._collect_posix, "Linux")
        self.assertEqual(err, "")
        self.assertEqual(rows[0]["threads"], 129427)

    def test_a_usage_message_on_stdout_does_not_count_as_a_census(self):
        # Some tools print their complaint on stdout. Those lines must not become
        # phantom rows, or they would suppress the very error that explains the
        # empty result.
        rows, err = self._with_tool(
            "usage: ps [-AaCcEefhjlMmrSTvwXx] [-O fmt | -o fmt]\n",
            "exit status 1: bad usage", self.mod._collect_posix, "Linux")
        self.assertEqual(rows, [])
        self.assertTrue(err)

    def test_a_working_ps_reports_no_error(self):
        # The green control: nothing above is achieved by calling every scan broken.
        rows, err = self._with_tool(
            "  38264 129427 9474048 8123 llama-cli\n      4   613    14336 0 systemd\n",
            "", self.mod._collect_posix, "Linux")
        self.assertEqual(err, "")
        self.assertEqual([r["pid"] for r in rows], [38264, 4])

    def test_missing_ps_binary_is_reported_not_swallowed(self):
        # _run_tool's own failure path, exercised for real: a program that does not
        # exist must come back as a named error, not an empty successful scan.
        out, err = self.mod._run_tool(30, "fak-no-such-census-tool-5541", "-eo")
        self.assertEqual(out, "")
        self.assertTrue(err)


if __name__ == "__main__":
    unittest.main()
