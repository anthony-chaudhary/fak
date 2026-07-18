#!/usr/bin/env python3
"""Hermetic tests for tools/fak_phase0_gate.py.

The gate is a fold of pure validators over synthetic manifest/bench dicts that
append human-readable strings to a ``failures`` list. These tests pin the
load-bearing coercion helpers (bool is NOT an int/float; junk coerces to None)
and drive the node-info + modelbench validators with both a canonical-clean
document (zero failures) and specifically-corrupted ones (exactly the expected
failure text), so a regression in any single rule is caught by name.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "fak_phase0_gate.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("fak_phase0_gate", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = load()


class TestCoercion(unittest.TestCase):
    def test_as_int_rejects_bool(self):
        self.assertIsNone(MOD.as_int(True))
        self.assertIsNone(MOD.as_int(False))

    def test_as_int_parses_numbers_and_strings(self):
        self.assertEqual(MOD.as_int(32), 32)
        self.assertEqual(MOD.as_int("64"), 64)
        self.assertIsNone(MOD.as_int("not-a-number"))
        self.assertIsNone(MOD.as_int(None))

    def test_as_float_rejects_bool(self):
        self.assertIsNone(MOD.as_float(True))
        self.assertEqual(MOD.as_float("1.5"), 1.5)

    def test_positive_number(self):
        self.assertTrue(MOD.positive_number(0.1))
        self.assertFalse(MOD.positive_number(0))
        self.assertFalse(MOD.positive_number(-3))
        self.assertFalse(MOD.positive_number("x"))
        self.assertFalse(MOD.positive_number(True))  # bool is not a number here

    def test_require_only_appends_on_false(self):
        failures = []
        MOD.require(failures, True, "should not appear")
        self.assertEqual(failures, [])
        MOD.require(failures, False, "boom")
        self.assertEqual(failures, ["boom"])


class TestValidateNodeInfo(unittest.TestCase):
    def _clean(self):
        return {"host": "da33", "os": "linux", "arch": "amd64", "cpu": "epyc",
                "cores": 96, "go": "1.24", "git": "abc123"}

    def test_clean_node_info_no_failures(self):
        failures = []
        info = self._clean()
        MOD.validate_node_info(info, {"host": "da33", "git": "abc123"}, failures)
        self.assertEqual(failures, [])

    def test_missing_key_is_flagged(self):
        failures = []
        info = self._clean()
        del info["cpu"]
        MOD.validate_node_info(info, {}, failures)
        self.assertIn("node-info.json missing cpu", failures)

    def test_host_mismatch_flagged(self):
        failures = []
        MOD.validate_node_info(self._clean(), {"host": "other", "git": "abc123"}, failures)
        self.assertIn("manifest host does not match node-info.json", failures)


class TestValidateModelbench(unittest.TestCase):
    def _clean(self):
        return {
            "precision": "Q8_0",
            "prefill": [{"tokens": 256, "tok_per_sec": 10.0}],
            "decode": {"tok_per_sec": 5.0},
            "workload": {"schema": "fak.agent-workload.v1", "prefill_cap": 0,
                         "decode_steps_cap": 32, "cases": 1, "path": "w.json"},
            "workload_prefill": [{"tokens": 100, "recorded_tokens": 100,
                                  "tok_per_sec": 9.0}],
            "workload_decode": [{"recorded_prompt_tokens": 100, "prompt_tokens": 100,
                                 "decode_steps": 32, "tok_per_sec": 4.0}],
        }

    def test_clean_modelbench_no_failures(self):
        failures = []
        MOD.validate_modelbench(self._clean(), {"workload": "w.json"}, failures)
        self.assertEqual(failures, [], failures)

    def test_wrong_precision_flagged(self):
        failures = []
        mb = self._clean()
        mb["precision"] = "Q4_0"
        MOD.validate_modelbench(mb, {}, failures)
        self.assertIn("modelbench-q8.json precision must be Q8_0", failures)

    def test_capped_decode_prompt_tokens_flagged(self):
        failures = []
        mb = self._clean()
        mb["workload_decode"][0]["prompt_tokens"] = 50  # capped below recorded 100
        MOD.validate_modelbench(mb, {}, failures)
        self.assertTrue(any("is capped" in f for f in failures), failures)

    def test_case_count_mismatch_flagged(self):
        failures = []
        mb = self._clean()
        mb["workload"]["cases"] = 2  # but only one row present
        MOD.validate_modelbench(mb, {}, failures)
        self.assertTrue(
            any("must contain every workload case" in f for f in failures), failures)


if __name__ == "__main__":
    unittest.main()
