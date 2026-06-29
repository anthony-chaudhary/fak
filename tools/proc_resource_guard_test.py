#!/usr/bin/env python3
"""Hermetic tests for tools/proc_resource_guard.py (no real process scan)."""
from __future__ import annotations

import importlib.util
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


if __name__ == "__main__":
    unittest.main()
