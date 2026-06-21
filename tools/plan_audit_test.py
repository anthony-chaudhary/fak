#!/usr/bin/env python3
"""Hermetic tests for tools/plan_audit.py."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "plan_audit.py"


def load():
    spec = importlib.util.spec_from_file_location("plan_audit", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PlanAuditTest(unittest.TestCase):
    def test_counts_table_rows_and_numbered_headings(self) -> None:
        pa = load()
        lines = [
            "# Plan",
            "| N | Work |",
            "|---|---|",
            "| 1 | table unit |",
            "| 2 | table unit |",
            "## 3. heading unit",
            "### 3.1 heading subunit",
            "## Not a unit",
        ]

        self.assertEqual(pa.count_units(lines), 4)

    def test_public_heading_plan_contributes_to_task_weighted_floor(self) -> None:
        pa = load()
        report = pa.build_report([
            {
                "id": "PLAN-x",
                "name": "Plan X",
                "file": "PLAN-x.md",
                "total_units": 5,
                "signal": "none",
                "percent_complete": 0,
                "status": "not_started",
            }
        ])

        task = report["work_units"]["task_weighted"]
        self.assertEqual(task["total_units"], 5)
        self.assertEqual(task["done_units"], 0)
        self.assertEqual(task["coverage_plans"], 1)


if __name__ == "__main__":
    unittest.main()
