#!/usr/bin/env python3
"""Tests for amd_gpu_facts — the AMD GPU-state probe. The probe shells PowerShell to read
Windows GPU perf counters, so these tests exercise the pure-Python parsing/contract surface
(name filtering, error shaping, the MiB derivation) by stubbing the PowerShell call, plus one
opportunistic live smoke check that SKIPS off-Windows or when the counters are unavailable."""
import json
import sys
import unittest
from unittest import mock

import amd_gpu_facts as g


_GOOD = json.dumps({
    "name": "AMD Radeon RX 7600",
    "driver_version": "32.0.31019.2002",
    "adapter_ram": 4293918720,
    "vram_used_bytes": 6_400_000_000,
    "compute_util_pct": 0.0,
    "total_util_pct": 97.5,
    "engines": [{"type": "Graphics", "util_pct": 88.0}, {"type": "Compute", "util_pct": 0.0}],
})


class AmdFactsContract(unittest.TestCase):
    def test_success_shapes_and_derives_mib(self):
        with mock.patch.object(g, "_pwsh", return_value=(True, _GOOD, "")):
            f = g.amd_facts()
        self.assertTrue(f["available"])
        self.assertEqual(f["name"], "AMD Radeon RX 7600")
        # 6.4e9 bytes -> ~6103.5 MiB, the exact live VRAM figure (not the WMI-capped adapter_ram)
        self.assertAlmostEqual(f["vram_used_mib"], 6_400_000_000 / (1024 * 1024), places=1)
        self.assertIn("WMI-WORD-capped", f["note"])

    def test_busiest_engine_is_derived(self):
        # On AMD, Vulkan compute lands under the '3d'/Graphics engine, NOT engtype_Compute, so
        # compute_util_pct reads ~0 mid-decode. The probe must surface the busiest engine as the
        # honest "is it working" signal — here Graphics(88) over Compute(0).
        with mock.patch.object(g, "_pwsh", return_value=(True, _GOOD, "")):
            f = g.amd_facts()
        self.assertEqual(f["busiest_engine"], "Graphics")
        self.assertEqual(f["busiest_util_pct"], 88.0)
        # the note must warn that the headline compute_util_pct is misleading on AMD
        self.assertIn("compute_util_pct", f["note"])
        self.assertIn("3d", f["note"])

    def test_name_filter_matches(self):
        with mock.patch.object(g, "_pwsh", return_value=(True, _GOOD, "")):
            self.assertTrue(g.amd_facts("7600")["available"])

    def test_name_filter_rejects_other_device(self):
        with mock.patch.object(g, "_pwsh", return_value=(True, _GOOD, "")):
            f = g.amd_facts("a100")
        self.assertFalse(f["available"])
        self.assertEqual(f["saw"], "AMD Radeon RX 7600")

    def test_no_powershell_reports_unavailable_not_crash(self):
        with mock.patch.object(g, "_pwsh", return_value=(False, "", "no PowerShell")):
            f = g.amd_facts()
        self.assertFalse(f["available"])
        self.assertIn("PowerShell", f["error"])

    def test_garbage_output_is_handled(self):
        with mock.patch.object(g, "_pwsh", return_value=(True, "not json", "")):
            f = g.amd_facts()
        self.assertFalse(f["available"])
        self.assertIn("parse failed", f["error"])

    def test_one_shot_probes_once_and_exit_matches_snapshot(self):
        # one-shot mode must probe exactly once so the printed JSON and the exit code describe
        # the SAME sample (no TOCTOU) and the PowerShell cost isn't paid twice.
        probe = mock.Mock(return_value=(True, _GOOD, ""))
        with mock.patch.object(g, "_pwsh", probe):
            rc = g.main([])
        self.assertEqual(probe.call_count, 1)
        self.assertEqual(rc, 0)

    def test_one_shot_unavailable_exits_nonzero(self):
        probe = mock.Mock(return_value=(False, "", "no PowerShell"))
        with mock.patch.object(g, "_pwsh", probe):
            rc = g.main([])
        self.assertEqual(probe.call_count, 1)
        self.assertEqual(rc, 1)

    def test_live_smoke_optional(self):
        """If we're really on this Windows box with the AMD driver, the probe should resolve a
        device. Otherwise SKIP — this is observability, not a hard gate."""
        if sys.platform != "win32":
            self.skipTest("not Windows")
        f = g.amd_facts()
        if not f.get("available"):
            self.skipTest(f"GPU counters unavailable: {f.get('error')}")
        self.assertIn("vram_used_mib", f)
        self.assertGreaterEqual(f["vram_used_mib"], 0)


if __name__ == "__main__":
    unittest.main()
