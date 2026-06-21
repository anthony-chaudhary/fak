#!/usr/bin/env python3
"""Tests for crash_audit — the Windows crash forensics probe. The probe shells PowerShell to read
the System event log + CIM hardware classes, so these tests exercise the pure analysis surface
(`decode_bugcheck` and `audit`) against synthetic probe dicts, plus one opportunistic live smoke
that SKIPS off-Windows. The verdict logic is where the value is, so it gets the most coverage."""
import sys
import unittest
from unittest import mock

import crash_audit as ca


def _probe(crashes, *, ecc=3, dimms=4, configured=3600, total_gb=256,
           whea=None, rex=None, disk=None, dumps=None, dirty=None,
           now="2026-06-19 00:00:00"):
    """Build a minimal probe dict shaped like the PowerShell output."""
    return {
        "available": True,
        "now": now,
        "last_boot": "2026-06-18 19:52:51",
        "crashes": crashes,
        "dumps": dumps if dumps is not None else [
            {"time": c["time"], "dump": f"C:\\Windows\\Minidump\\{c['time'][:10]}.dmp"}
            for c in crashes
        ],
        "dirty_shutdowns": dirty or [],
        "whea": whea or [],
        "resource_exhaustion": rex or [],
        "disk_errors": disk or [],
        "ram_total_gb": total_gb,
        "dimms": [{"gb": total_gb // dimms, "speed": configured, "configured": configured,
                   "part": "CP64G56C46U5", "loc": f"DIMM{i}"} for i in range(dimms)],
        "memory_error_correction": ecc,
        "pagefile_auto": True,
        "cpu": "AMD Ryzen 9 9950X 16-Core Processor",
        "board": "Micro-Star International Co., Ltd. MAG X870E TOMAHAWK WIFI (MS-7E59)",
        "bios_version": "2.A90",
        "bios_date": "2025-08-27",
        "crash_dump_enabled": 3,
        "minidump_accessible": False,
    }


def _crash(time, code, p1="0x6"):
    return {"time": time, "bugcheck_code": code, "p1": p1, "p2": "0x0", "p3": "0x0", "p4": "0x0",
            "power_button": "0"}


class DecodeBugcheck(unittest.TestCase):
    def test_known_codes_classified(self):
        self.assertEqual(ca.decode_bugcheck(0x4E)["name"], "PFN_LIST_CORRUPT")
        self.assertEqual(ca.decode_bugcheck(0x4E)["fault_class"], "memory")
        self.assertEqual(ca.decode_bugcheck(0x0A)["name"], "IRQL_NOT_LESS_OR_EQUAL")
        self.assertEqual(ca.decode_bugcheck(0xD1)["fault_class"], "driver")
        self.assertEqual(ca.decode_bugcheck(0x124)["fault_class"], "cpu_hw")

    def test_code_zero_is_power_loss(self):
        self.assertEqual(ca.decode_bugcheck(0)["fault_class"], "power_loss")

    def test_hex_formatting(self):
        self.assertEqual(ca.decode_bugcheck(0x4E)["code_hex"], "0x0000004e")

    def test_unknown_code_does_not_raise(self):
        d = ca.decode_bugcheck(0xDEAD)
        self.assertEqual(d["name"], "UNKNOWN_BUGCHECK")
        self.assertEqual(d["fault_class"], "unknown")


class AuditVerdict(unittest.TestCase):
    def test_the_actual_2026_06_18_crash_reads_as_memory_subsystem(self):
        """The real incident: 0x4E PFN_LIST_CORRUPT + 0x0A on a non-ECC 4x64GB @3600 box."""
        probe = _probe([_crash("2026-06-18 19:21:17", 0x4E),
                        _crash("2026-06-03 10:34:29", 0x0A)])
        a = ca.audit(probe)
        self.assertTrue(a["available"])
        self.assertEqual(a["bugcheck_count"], 2)
        self.assertIn("MEMORY SUBSYSTEM", a["verdict"])
        self.assertEqual(a["confidence"], "high")          # non-ECC + 4 DIMMs + down-clocked
        self.assertEqual(a["latest_crash"]["name"], "PFN_LIST_CORRUPT")
        self.assertTrue(a["memory"]["downclocked_below_jedec"])
        self.assertFalse(a["memory"]["is_ecc"])
        self.assertTrue(any("MemTest86" in x for x in a["actions"]))

    def test_ecc_two_dimm_memory_fault_is_lower_confidence(self):
        probe = _probe([_crash("2026-06-18 19:21:17", 0x4E)],
                       ecc=6, dimms=2, configured=5600)
        a = ca.audit(probe)
        self.assertIn("MEMORY SUBSYSTEM", a["verdict"])
        self.assertEqual(a["confidence"], "medium")        # ECC + 2 DIMMs + not down-clocked
        self.assertFalse(a["memory"]["downclocked_below_jedec"])

    def test_machine_check_outranks_memory(self):
        probe = _probe([_crash("2026-06-18 19:21:17", 0x124)])
        a = ca.audit(probe)
        self.assertIn("HARDWARE", a["verdict"])
        self.assertEqual(a["confidence"], "high")

    def test_whea_error_triggers_hardware_verdict(self):
        probe = _probe([_crash("2026-06-18 19:21:17", 0x50)],
                       whea=[{"time": "2026-06-18 19:21:00", "id": 18, "level": "Error"}])
        a = ca.audit(probe)
        self.assertIn("HARDWARE", a["verdict"])

    def test_driver_verdict(self):
        probe = _probe([_crash("2026-06-18 19:21:17", 0xD1)], ecc=6, dimms=2, configured=5600)
        a = ca.audit(probe)
        self.assertIn("DRIVER", a["verdict"])

    def test_pure_power_loss_verdict(self):
        probe = _probe([_crash("2026-05-06 13:00:00", 0)])
        a = ca.audit(probe)
        self.assertIn("POWER-LOSS", a["verdict"])
        self.assertEqual(a["powerloss_count"], 1)
        self.assertEqual(a["bugcheck_count"], 0)

    def test_no_crashes_is_stable(self):
        a = ca.audit(_probe([]))
        self.assertIn("stable", a["verdict"].lower())

    def test_window_filters_old_crashes(self):
        probe = _probe([_crash("2026-06-18 19:21:17", 0x4E),
                        _crash("2025-10-17 16:07:43", 0x4E)])
        a = ca.audit(probe, window_days=30)
        self.assertEqual(a["crash_count"], 1)              # the 2025 one is outside 30 days

    def test_side_signals_appended(self):
        probe = _probe([_crash("2026-06-18 19:21:17", 0x4E)],
                       rex=["2026-06-18 18:00:00"],
                       disk=[{"time": "2026-06-18 18:00:00", "id": 51, "provider": "disk"}])
        a = ca.audit(probe)
        joined = " ".join(a["actions"])
        self.assertIn("exhaustion", joined)
        self.assertIn("SMART", joined)

    def test_dump_path_joined_onto_crash(self):
        probe = _probe(
            [_crash("2026-06-18 19:52:56", 0x4E)],
            dumps=[{"time": "2026-06-18 19:53:08", "dump": "C:\\Windows\\Minidump\\061826.dmp"}],
        )
        a = ca.audit(probe)
        self.assertTrue(a["latest_crash"]["dump"].endswith(".dmp"))

    def test_eventlog_6008_shutdown_time_is_used_for_bugcheck_time(self):
        probe = _probe(
            [_crash("2026-06-18 19:52:56", 0x4E)],
            dirty=[{
                "logged": "2026-06-18 19:53:09",
                "shutdown_time": "2026-06-18 19:21:17",
                "detail": "previous shutdown was unexpected",
            }],
        )
        a = ca.audit(probe)
        self.assertEqual(a["latest_crash"]["time"], "2026-06-18 19:21:17")
        self.assertEqual(a["latest_crash"]["logged_time"], "2026-06-18 19:52:56")

    def test_stale_eventlog_6008_shutdown_time_is_ignored(self):
        probe = _probe(
            [_crash("2026-05-06 13:46:59", 0)],
            dirty=[{
                "logged": "2026-05-06 13:47:13",
                "shutdown_time": "2026-01-28 04:33:04",
                "detail": "previous shutdown was unexpected",
            }],
        )
        a = ca.audit(probe)
        self.assertEqual(a["latest_crash"]["time"], "2026-05-06 13:46:59")

    def test_volmgr_162_dump_write_is_not_disk_health_signal(self):
        probe = _probe(
            [_crash("2026-06-18 19:21:17", 0x4E)],
            disk=[{"time": "2026-06-18 19:52:55", "id": 162, "provider": "volmgr"}],
        )
        a = ca.audit(probe)
        self.assertEqual(a["signals"]["disk_errors"], 0)
        self.assertNotIn("SMART", " ".join(a["actions"]))


class BlindSpot(unittest.TestCase):
    """The audit's named blind spot (CRASH-AUDIT §4): a memory verdict cannot exclude a driver
    fault without the minidump stack. These pin that the audit always says whether the leading
    bugcheck is dump-attributable, and that an attribution closes it."""

    def test_blind_spot_present_when_dump_not_analyzed(self):
        # The real incident: 0x4E with a dump path, but attribution not attempted.
        probe = _probe([_crash("2026-06-18 19:21:17", 0x4E)])
        a = ca.audit(probe)  # no dump_attributions
        bs = a["blind_spot"]
        self.assertTrue(bs["present"])
        self.assertIn("not attempted", bs["reason"])
        self.assertIn("cdb", bs["command"])

    def test_blind_spot_closed_when_dump_attributed(self):
        probe = _probe([_crash("2026-06-18 19:21:17", 0x4E)])
        dump = ca.audit(probe)["latest_crash"]["dump"]
        attributions = {dump: {"attributed": True, "faulting_module": "myfault.sys",
                               "tool": "cdb", "error": "", "raw_excerpt": ""}}
        a = ca.audit(probe, dump_attributions=attributions)
        self.assertFalse(a["blind_spot"]["present"])
        self.assertIn("myfault.sys", a["blind_spot"]["reason"])
        self.assertEqual(a["latest_crash"]["attribution"]["faulting_module"], "myfault.sys")

    def test_blind_spot_when_attribution_failed(self):
        probe = _probe([_crash("2026-06-18 19:21:17", 0x4E)])
        dump = ca.audit(probe)["latest_crash"]["dump"]
        attributions = {dump: {"attributed": False, "faulting_module": "", "tool": "",
                               "error": "no Microsoft debugger on PATH", "raw_excerpt": ""}}
        a = ca.audit(probe, dump_attributions=attributions)
        self.assertTrue(a["blind_spot"]["present"])
        self.assertIn("no Microsoft debugger", a["blind_spot"]["reason"])

    def test_blind_spot_when_no_dump_captured(self):
        probe = _probe([_crash("2026-06-18 19:21:17", 0x4E)], dumps=[])
        a = ca.audit(probe)
        self.assertTrue(a["blind_spot"]["present"])
        self.assertIn("CrashControl", a["blind_spot"]["command"])

    def test_blind_spot_absent_when_only_power_loss(self):
        probe = _probe([_crash("2026-05-06 13:00:00", 0)], dumps=[])
        a = ca.audit(probe)
        self.assertFalse(a["blind_spot"]["present"])


class DumpAttribution(unittest.TestCase):
    def test_extract_faulting_module_parses_cdb_output(self):
        cdb_out = (
            "MODULE_NAME: myfault\n"
            "IMAGE_NAME:  myfault.sys\n"
            "FAILURE_BUCKET_ID: 0x4E_99_myfault!Bad\n"
        )
        self.assertEqual(ca._extract_faulting_module(cdb_out), "myfault.sys")

    def test_extract_faulting_module_returns_empty_on_miss(self):
        self.assertEqual(ca._extract_faulting_module("no recognizable lines here"), "")

    def test_analyze_dump_no_debugger_is_safe(self):
        with mock.patch.object(ca, "_find_debugger", return_value=""):
            r = ca.analyze_dump("C:\\Windows\\Minidump\\x.dmp")
        self.assertFalse(r["attributed"])
        self.assertIn("debugger", r["error"])

    def test_analyze_dump_parses_attributed_result(self):
        cdb_out = "MODULE_NAME: nt\nIMAGE_NAME: ntoskrnl.exe\n"
        fake = mock.MagicMock()
        fake.stdout = cdb_out
        fake.stderr = ""
        with mock.patch.object(ca, "_find_debugger", return_value="C:\\dbg\\cdb.exe"), \
             mock.patch.object(ca.os.path, "isfile", return_value=True), \
             mock.patch.object(ca.subprocess, "run", return_value=fake):
            r = ca.analyze_dump("C:\\Windows\\Minidump\\x.dmp")
        self.assertTrue(r["attributed"])
        self.assertEqual(r["faulting_module"], "ntoskrnl.exe")

    def test_collect_attributions_dedupes_and_skips_empty(self):
        crashes = [{"dump": "C:\\a.dmp"}, {"dump": "C:\\a.dmp"}, {"dump": ""}]
        seen = []
        def fake_analyze(path, timeout=30.0):
            seen.append(path)
            return {"attributed": True, "faulting_module": "x.sys", "tool": "cdb",
                    "error": "", "raw_excerpt": ""}
        with mock.patch.object(ca, "analyze_dump", side_effect=fake_analyze):
            out = ca.collect_attributions(crashes)
        self.assertEqual(list(out.keys()), ["C:\\a.dmp"])
        self.assertEqual(seen, ["C:\\a.dmp"])  # deduped, empty skipped


class Disposition(unittest.TestCase):
    """The recorded-disposition surface (#111): a forensic conclusion ships as data so a recurring
    host fault is reported with its KNOWN cause + remediation status instead of re-deriving an
    `cause: unclear`. These pin matching, the no-record default, and the report rendering."""

    _PFN_DISP = {
        "id": "host-pfn-list-corrupt-2026-06-18",
        "match": {"fault_class": "memory", "board": "MAG X870E TOMAHAWK WIFI"},
        "cause": "host_memory_subsystem",
        "is_fak_bug": False,
        "recurring": True,
        "summary": "0x4E is a host memory-subsystem fault, not a fak code bug.",
        "remediation_status": "operator_action_pending",
        "evidence_doc": "CRASH-AUDIT-2026-06-18-pfn-list-corrupt.md",
        "tracking_issues": [111, 75],
    }

    def test_recorded_disposition_surfaces_for_matching_host_fault(self):
        probe = _probe([_crash("2026-06-18 19:21:17", 0x4E)])
        a = ca.audit(probe, dispositions=[self._PFN_DISP])
        d = a["disposition"]
        self.assertIsNotNone(d)
        self.assertEqual(d["cause"], "host_memory_subsystem")
        self.assertIs(d["is_fak_bug"], False)
        self.assertTrue(d["recurring"])

    def test_no_disposition_when_record_empty(self):
        probe = _probe([_crash("2026-06-18 19:21:17", 0x4E)])
        a = ca.audit(probe, dispositions=[])
        self.assertIsNone(a["disposition"])

    def test_disposition_does_not_match_other_board(self):
        probe = _probe([_crash("2026-06-18 19:21:17", 0x4E)])
        probe["board"] = "Some Other Board X99"
        a = ca.audit(probe, dispositions=[self._PFN_DISP])
        self.assertIsNone(a["disposition"])

    def test_disposition_does_not_match_different_fault_class(self):
        # A driver-class bugcheck (0xD1) on the same host must NOT pick up the memory disposition.
        probe = _probe([_crash("2026-06-18 19:21:17", 0xD1)], ecc=6, dimms=2, configured=5600)
        a = ca.audit(probe, dispositions=[self._PFN_DISP])
        self.assertIsNone(a["disposition"])

    def test_empty_match_block_is_not_a_catch_all(self):
        probe = _probe([_crash("2026-06-18 19:21:17", 0x4E)])
        a = ca.audit(probe, dispositions=[{"id": "x", "match": {}}])
        self.assertIsNone(a["disposition"])

    def test_no_disposition_when_no_bugchecks(self):
        probe = _probe([_crash("2026-05-06 13:00:00", 0)], dumps=[])
        a = ca.audit(probe, dispositions=[self._PFN_DISP])
        self.assertIsNone(a["disposition"])

    def test_report_renders_recorded_disposition(self):
        probe = _probe([_crash("2026-06-18 19:21:17", 0x4E)])
        txt = ca._report(ca.audit(probe, dispositions=[self._PFN_DISP]))
        self.assertIn("RECORDED DISPOSITION", txt)
        self.assertIn("host_memory_subsystem", txt)
        self.assertIn("not a fak", txt)
        self.assertIn("#111", txt)

    def test_load_dispositions_missing_file_is_empty(self):
        self.assertEqual(ca.load_dispositions("C:\\no\\such\\dispositions.json"), [])

    def test_shipped_dispositions_file_records_pfn_fault(self):
        """The shipped record (tools/crash_dispositions.json) must contain the 0x4E disposition so
        the live audit on this host reports it instead of re-deriving an unexplained cause."""
        disps = ca.load_dispositions()
        probe = _probe([_crash("2026-06-18 19:21:17", 0x4E)])
        d = ca.match_disposition(ca.audit(probe, dispositions=[])["latest_crash"], probe, disps)
        self.assertIsNotNone(d, "shipped crash_dispositions.json should match the 0x4E host fault")
        self.assertIs(d["is_fak_bug"], False)
        self.assertEqual(d["cause"], "host_memory_subsystem")


class ProbeContract(unittest.TestCase):
    def test_unavailable_probe_short_circuits(self):
        a = ca.audit({"available": False, "error": "no PowerShell"})
        self.assertFalse(a["available"])
        self.assertIn("PowerShell", a["error"])

    def test_collect_handles_no_powershell(self):
        with mock.patch.object(ca, "_pwsh", return_value=(False, "", "no PowerShell")):
            p = ca.collect()
        self.assertFalse(p["available"])

    def test_collect_handles_garbage(self):
        with mock.patch.object(ca, "_pwsh", return_value=(True, "not json", "")):
            p = ca.collect()
        self.assertFalse(p["available"])
        self.assertIn("parse failed", p["error"])

    def test_report_renders_without_error(self):
        probe = _probe([_crash("2026-06-18 19:21:17", 0x4E)])
        txt = ca._report(ca.audit(probe))
        self.assertIn("CRASH AUDIT", txt)
        self.assertIn("VERDICT", txt)
        self.assertIn("PFN_LIST_CORRUPT", txt)


class LiveSmoke(unittest.TestCase):
    def test_live_optional(self):
        """On this Windows box the probe should resolve real crash forensics. Otherwise SKIP —
        this is observability, not a hard gate."""
        if sys.platform != "win32":
            self.skipTest("not Windows")
        p = ca.collect()
        if not p.get("available"):
            self.skipTest(f"probe unavailable: {p.get('error')}")
        a = ca.audit(p)
        self.assertTrue(a["available"])
        self.assertIn("verdict", a)
        self.assertIn(a["confidence"], ("n/a", "low", "medium", "high"))


if __name__ == "__main__":
    unittest.main()
