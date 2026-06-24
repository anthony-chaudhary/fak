#!/usr/bin/env python3
"""Hermetic tests for tools/dispositions.py.

No live dos calls, no network, no shared dos state — every subprocess call
is injected via the ``runner`` parameter.

Run: ``python -m pytest tools/dispositions_test.py``
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "dispositions.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("dispositions", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class EnumerateUnitsTest(unittest.TestCase):
    def test_parses_units_from_json_stdout(self) -> None:
        mod = load()

        def runner(argv):
            assert "enumerate" in argv
            return {
                "returncode": 0,
                "stdout": json.dumps({"units": ["AUTH1", "AUTH2"], "series": "AUTH"}),
                "stderr": "",
            }

        units = mod.enumerate_units(Path("plan.md"), runner=runner)
        self.assertEqual(units, ["AUTH1", "AUTH2"])

    def test_exit3_driftnote_still_returns_units(self) -> None:
        mod = load()

        def runner(argv):
            return {
                "returncode": 3,
                "stdout": json.dumps({"units": ["U1"], "drift": []}),
                "stderr": "",
            }

        units = mod.enumerate_units(Path("plan.md"), runner=runner)
        self.assertEqual(units, ["U1"])

    def test_exit4_empty_returns_empty_list(self) -> None:
        mod = load()

        def runner(argv):
            return {"returncode": 4, "stdout": json.dumps({"units": []}), "stderr": ""}

        self.assertEqual(mod.enumerate_units(Path("plan.md"), runner=runner), [])

    def test_error_exit_raises_runtime_error(self) -> None:
        mod = load()

        def runner(argv):
            return {"returncode": 2, "stdout": "", "stderr": "contract error"}

        with self.assertRaises(RuntimeError):
            mod.enumerate_units(Path("missing.md"), runner=runner)

    def test_workspace_arg_forwarded(self) -> None:
        mod = load()
        seen: list[list[str]] = []

        def runner(argv):
            seen.append(list(argv))
            return {"returncode": 0, "stdout": json.dumps({"units": []}), "stderr": ""}

        ws = Path("/ws")
        mod.enumerate_units(Path("p.md"), workspace=ws, runner=runner)
        self.assertIn("--workspace", seen[0])
        self.assertEqual(seen[0][seen[0].index("--workspace") + 1], str(ws))


class PickableRowTest(unittest.TestCase):
    def test_offerable_unit_returns_held_false(self) -> None:
        mod = load()

        def runner(argv):
            assert "pickable" in argv
            assert "AUTH1" in argv
            return {
                "returncode": 0,
                "stdout": json.dumps({"held": False, "reason": None, "unit": "AUTH1"}),
                "stderr": "",
            }

        row = mod.pickable_row("AUTH1", {}, runner=runner)
        self.assertFalse(row["held"])
        self.assertIsNone(row["reason"])

    def test_held_unit_returns_reason(self) -> None:
        mod = load()

        def runner(argv):
            return {
                "returncode": 10,
                "stdout": json.dumps({"held": True, "reason": "DRAFT_CLASS", "unit": "U1"}),
                "stderr": "",
            }

        row = mod.pickable_row("U1", {"plan_class": "DRAFT"}, runner=runner)
        self.assertTrue(row["held"])
        self.assertEqual(row["reason"], "DRAFT_CLASS")

    def test_state_json_forwarded_in_argv(self) -> None:
        mod = load()
        seen: list[list[str]] = []

        def runner(argv):
            seen.append(list(argv))
            return {"returncode": 0, "stdout": json.dumps({"held": False, "reason": None}), "stderr": ""}

        mod.pickable_row("X", {"in_flight": True}, runner=runner)
        state_idx = seen[0].index("--state") + 1
        self.assertEqual(json.loads(seen[0][state_idx]), {"in_flight": True})

    def test_unexpected_exit_raises_runtime_error(self) -> None:
        mod = load()

        def runner(argv):
            return {"returncode": 99, "stdout": "", "stderr": "bad"}

        with self.assertRaises(RuntimeError):
            mod.pickable_row("X", {}, runner=runner)

    def test_all_curable_hold_exits_accepted(self) -> None:
        mod = load()
        for rc in (20, 21, 22, 23, 24, 25):
            def runner(argv, _rc=rc):
                return {
                    "returncode": _rc,
                    "stdout": json.dumps({"held": True, "reason": "COOLDOWN"}),
                    "stderr": "",
                }

            row = mod.pickable_row("U", {}, runner=runner)
            self.assertTrue(row["held"])


class BuildDispositionsTest(unittest.TestCase):
    def test_live_and_held_rows(self) -> None:
        mod = load()

        responses = {
            "UNIT1": {"returncode": 0, "stdout": json.dumps({"held": False, "reason": None}), "stderr": ""},
            "UNIT2": {"returncode": 20, "stdout": json.dumps({"held": True, "reason": "IN_FLIGHT"}), "stderr": ""},
        }

        def runner(argv):
            for unit in ("UNIT1", "UNIT2"):
                if unit in argv:
                    return responses[unit]
            raise AssertionError(f"unexpected argv: {argv}")

        rows = mod.build_dispositions(["UNIT1", "UNIT2"], {}, runner=runner)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], {"phase": "UNIT1", "live": True, "drop_reason": ""})
        self.assertEqual(rows[1], {"phase": "UNIT2", "live": False, "drop_reason": "IN_FLIGHT"})

    def test_state_map_forwarded_per_unit(self) -> None:
        mod = load()
        seen_states: list[dict] = []

        def runner(argv):
            if "pickable" in argv:
                idx = argv.index("--state") + 1
                seen_states.append(json.loads(argv[idx]))
            return {"returncode": 0, "stdout": json.dumps({"held": False, "reason": None}), "stderr": ""}

        state_map = {"A": {"shipped": True}, "B": {"in_flight": True}}
        mod.build_dispositions(["A", "B"], state_map, runner=runner)
        self.assertEqual(seen_states[0], {"shipped": True})
        self.assertEqual(seen_states[1], {"in_flight": True})

    def test_missing_state_defaults_to_empty(self) -> None:
        mod = load()
        seen_states: list[dict] = []

        def runner(argv):
            if "pickable" in argv:
                idx = argv.index("--state") + 1
                seen_states.append(json.loads(argv[idx]))
            return {"returncode": 0, "stdout": json.dumps({"held": False, "reason": None}), "stderr": ""}

        mod.build_dispositions(["UNKNOWN_UNIT"], {}, runner=runner)
        self.assertEqual(seen_states[0], {})

    def test_empty_unit_list_returns_empty_rows(self) -> None:
        mod = load()
        rows = mod.build_dispositions([], {}, runner=lambda a: {})
        self.assertEqual(rows, [])


class WriteSidecarTest(unittest.TestCase):
    def test_round_trip(self) -> None:
        mod = load()
        rows = [
            {"phase": "A", "live": True, "drop_reason": ""},
            {"phase": "B", "live": False, "drop_reason": "IN_FLIGHT"},
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / ".dispositions-test.json"
            mod.write_sidecar(rows, path)
            self.assertTrue(path.exists())
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data, rows)

    def test_atomic_write_no_tmp_left_behind(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / ".dispositions-x.json"
            mod.write_sidecar([], path)
            leftovers = [p for p in Path(td).iterdir() if p.suffix == ".tmp"]
            self.assertEqual(leftovers, [])


class ClassifyTest(unittest.TestCase):
    def _sidecar(self, td: str, rows: list | None = None) -> Path:
        path = Path(td) / ".dispositions-tag.json"
        path.write_text(json.dumps(rows or []), encoding="utf-8")
        return path

    def test_live_verdict(self) -> None:
        mod = load()

        def runner(argv):
            assert "gate" in argv
            return {
                "returncode": 0,
                "stdout": json.dumps({"verdict": "LIVE", "reason": "1 live pick(s)", "evidence": []}),
                "stderr": "",
            }

        with tempfile.TemporaryDirectory() as td:
            result = mod.classify(self._sidecar(td), runner=runner)
        self.assertEqual(result["verdict"], "LIVE")

    def test_drain_verdict(self) -> None:
        mod = load()

        def runner(argv):
            return {
                "returncode": 3,
                "stdout": json.dumps({"verdict": "DRAIN", "reason": "no live picks", "evidence": []}),
                "stderr": "",
            }

        with tempfile.TemporaryDirectory() as td:
            result = mod.classify(self._sidecar(td), runner=runner)
        self.assertEqual(result["verdict"], "DRAIN")

    def test_stale_stamp_verdict(self) -> None:
        mod = load()

        def runner(argv):
            return {
                "returncode": 4,
                "stdout": json.dumps({"verdict": "STALE-STAMP", "reason": "shipped but unstamped", "evidence": []}),
                "stderr": "",
            }

        with tempfile.TemporaryDirectory() as td:
            result = mod.classify(self._sidecar(td), runner=runner)
        self.assertEqual(result["verdict"], "STALE-STAMP")

    def test_blocked_verdict(self) -> None:
        mod = load()

        def runner(argv):
            return {
                "returncode": 5,
                "stdout": json.dumps({"verdict": "BLOCKED", "reason": "claim held", "evidence": []}),
                "stderr": "",
            }

        with tempfile.TemporaryDirectory() as td:
            result = mod.classify(self._sidecar(td), runner=runner)
        self.assertEqual(result["verdict"], "BLOCKED")

    def test_race_verdict(self) -> None:
        mod = load()

        def runner(argv):
            return {
                "returncode": 6,
                "stdout": json.dumps({"verdict": "RACE", "reason": "cache lock lost", "evidence": []}),
                "stderr": "",
            }

        with tempfile.TemporaryDirectory() as td:
            result = mod.classify(self._sidecar(td), runner=runner)
        self.assertEqual(result["verdict"], "RACE")

    def test_contract_error_raises(self) -> None:
        mod = load()

        def runner(argv):
            return {"returncode": 2, "stdout": "", "stderr": "contract error"}

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(RuntimeError):
                mod.classify(self._sidecar(td), runner=runner)

    def test_sidecar_path_passed_to_gate(self) -> None:
        mod = load()
        seen: list[list[str]] = []

        def runner(argv):
            seen.append(list(argv))
            return {
                "returncode": 0,
                "stdout": json.dumps({"verdict": "LIVE", "reason": "", "evidence": []}),
                "stderr": "",
            }

        with tempfile.TemporaryDirectory() as td:
            sc = self._sidecar(td)
            mod.classify(sc, runner=runner)
        self.assertIn(str(sc), seen[0])


class SidecarPathTest(unittest.TestCase):
    def test_naming_convention(self) -> None:
        mod = load()
        path = mod.sidecar_path("myplan", Path("/tmp"))
        self.assertEqual(path.name, ".dispositions-myplan.json")
        self.assertEqual(path.parent, Path("/tmp"))


class BuildPayloadTest(unittest.TestCase):
    def test_ok_iff_live(self) -> None:
        mod = load()
        for verdict, expected in [("LIVE", True), ("DRAIN", False), ("BLOCKED", False), ("RACE", False)]:
            payload = mod.build_payload(
                plan_doc=Path("plan.md"),
                tag="t",
                dispositions=[],
                sidecar_path=Path("/tmp/.dispositions-t.json"),
                gate={"verdict": verdict, "reason": "test"},
            )
            self.assertEqual(payload["ok"], expected, f"verdict={verdict}")
            self.assertEqual(payload["verdict"], verdict)
            self.assertEqual(payload["schema"], mod.SCHEMA)


if __name__ == "__main__":
    unittest.main()
