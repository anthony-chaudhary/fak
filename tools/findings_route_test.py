#!/usr/bin/env python3
"""Hermetic tests for tools/findings_route.py — the defect-STOP closure appender.

Pure stdlib, no network/binary/GPU, so gated_tool_tests classifies this HERMETIC and
runs it on every push. Every case writes into a TemporaryDirectory sink so the repo
tree is never touched, and pins ``now`` for determinism.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "findings_route.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("findings_route", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # register so dataclass annotation resolution works
    spec.loader.exec_module(mod)
    return mod


fr = load()


def _route(tmp: Path, **kw):
    base = {
        "key": kw.pop("key", "k1"),
        "sev": kw.pop("sev", "P2"),
        "pattern": kw.pop("pattern", "recurring-blocker"),
        "item": kw.pop("item", "dispatch loop stalled on lane tools"),
        "source": kw.pop("source", "fanout/run-1"),
        "owning_plan": kw.pop("owning_plan", "RSI"),
        "queue": str(tmp / "queue.md"),
        "now": kw.pop("now", "2026-06-28T00:00:00+00:00"),
        "env": kw.pop("env", {}),  # {} => enabled, host env can't disable the test
    }
    base.update(kw)
    return fr.route_finding(**base)


class RouteFindingTest(unittest.TestCase):
    def test_first_route_creates_one_open_row(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            v = _route(tmp, key="k1", cause_key="C")
            self.assertEqual(v["action"], "routed")
            self.assertTrue(v["routed"])
            self.assertEqual(v["n"], 1)
            snap = fr.load_queue(queue=str(tmp / "queue.md"))
            self.assertEqual(snap["total"], 1)
            self.assertEqual(snap["open"], 1)

    def test_idempotent_same_key_does_not_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            _route(tmp, key="dup", cause_key="C")
            v2 = _route(tmp, key="dup", cause_key="C")
            self.assertEqual(v2["action"], "noop")
            self.assertFalse(v2["routed"])
            snap = fr.load_queue(queue=str(tmp / "queue.md"))
            self.assertEqual(snap["total"], 1)
            self.assertEqual(snap["rows"][0]["hits"], 1)

    def test_damping_folds_concurrent_cause_onto_one_row(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            _route(tmp, key="k1", cause_key="C")
            v2 = _route(tmp, key="k2", cause_key="C")
            self.assertEqual(v2["action"], "damped")
            self.assertEqual(v2["hits"], 2)
            snap = fr.load_queue(queue=str(tmp / "queue.md"))
            self.assertEqual(snap["total"], 1)
            self.assertEqual(snap["rows"][0]["hits"], 2)

    def test_distinct_causes_get_distinct_rows(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            a = _route(tmp, key="k1", cause_key="A")
            b = _route(tmp, key="k2", cause_key="B")
            self.assertEqual(a["n"], 1)
            self.assertEqual(b["n"], 2)
            self.assertEqual(fr.load_queue(queue=str(tmp / "queue.md"))["total"], 2)

    def test_recurrence_after_close_reopens_and_escalates(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            _route(tmp, key="k1", cause_key="C", sev="P2")
            closed = fr.close_finding(cause_key="C", sha="abc1234", queue=str(tmp / "queue.md"),
                                      now="2026-06-28T01:00:00+00:00")
            self.assertEqual(closed["action"], "closed")
            self.assertEqual(fr.load_queue(queue=str(tmp / "queue.md"))["open"], 0)
            v = _route(tmp, key="k2", cause_key="C", sev="P2")
            self.assertEqual(v["action"], "escalated")
            self.assertEqual(v["sev"], "P1")  # P2 bumped one rung on recurrence
            self.assertEqual(v["recurrences"], 1)
            snap = fr.load_queue(queue=str(tmp / "queue.md"))
            self.assertEqual(snap["open"], 1)  # reopened

    def test_escalation_clamps_at_p0(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            _route(tmp, key="k1", cause_key="C", sev="P1")
            fr.close_finding(cause_key="C", sha="s1", queue=str(tmp / "queue.md"))
            v1 = _route(tmp, key="k2", cause_key="C")  # P1 -> P0
            self.assertEqual(v1["sev"], "P0")
            fr.close_finding(cause_key="C", sha="s2", queue=str(tmp / "queue.md"))
            v2 = _route(tmp, key="k3", cause_key="C")  # P0 clamps at P0
            self.assertEqual(v2["sev"], "P0")
            self.assertEqual(v2["recurrences"], 2)

    def test_cause_key_defaults_to_key(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            v = _route(tmp, key="lonely")  # no cause_key
            self.assertEqual(v["action"], "routed")
            snap = fr.load_queue(queue=str(tmp / "queue.md"))
            self.assertEqual(snap["rows"][0]["cause_key"], "lonely")

    def test_kill_switch_disables_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            v = _route(tmp, key="k1", cause_key="C", env={"FLEET_FINDINGS_ROUTE_ENABLED": "0"})
            self.assertEqual(v["action"], "disabled")
            self.assertFalse(v["routed"])
            self.assertFalse((tmp / "queue.md").exists())
            self.assertFalse((tmp / "findings-followup-queue.history.jsonl").exists())

    def test_kill_switch_unset_is_enabled(self) -> None:
        self.assertTrue(fr.enabled_from_env({}))
        self.assertTrue(fr.enabled_from_env({"FLEET_FINDINGS_ROUTE_ENABLED": "1"}))
        self.assertFalse(fr.enabled_from_env({"FLEET_FINDINGS_ROUTE_ENABLED": "off"}))
        self.assertFalse(fr.enabled_from_env({"FLEET_FINDINGS_ROUTE_ENABLED": ""}))

    def test_never_raises_on_internal_error(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            original = fr._append_event
            fr._append_event = lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
            try:
                v = _route(tmp, key="k1", cause_key="C")  # must NOT raise
            finally:
                fr._append_event = original
            self.assertFalse(v["ok"])
            self.assertEqual(v["action"], "error")
            self.assertFalse(v["routed"])

    def test_close_unknown_row_is_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            v = fr.close_finding(cause_key="nope", sha="x", queue=str(tmp / "queue.md"))
            self.assertEqual(v["action"], "not_found")
            self.assertFalse(v["ok"])

    def test_markdown_renders_row_and_closed_strikethrough(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            _route(tmp, key="k1", cause_key="C", item="loop wall on lane tools")
            text = (tmp / "queue.md").read_text(encoding="utf-8")
            self.assertIn("| 1 |", text)
            self.assertIn("loop wall on lane tools", text)
            self.assertIn("open", text)
            fr.close_finding(cause_key="C", sha="deadbee", queue=str(tmp / "queue.md"))
            text2 = (tmp / "queue.md").read_text(encoding="utf-8")
            self.assertIn("~~1~~", text2)
            self.assertIn("CLOSED deadbee", text2)

    def test_markdown_pipe_in_item_is_escaped(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            _route(tmp, key="k1", cause_key="C", item="a | b | c")
            text = (tmp / "queue.md").read_text(encoding="utf-8")
            self.assertIn("a \\| b \\| c", text)

    def test_history_is_append_only_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            _route(tmp, key="k1", cause_key="C")
            _route(tmp, key="k2", cause_key="C")  # damped -> second event
            hist = tmp / "findings-followup-queue.history.jsonl"
            lines = [ln for ln in hist.read_text(encoding="utf-8").splitlines() if ln.strip()]
            self.assertEqual(len(lines), 2)  # append-only: one event per write
            for ln in lines:
                json.loads(ln)  # every line is valid JSON

    def test_concurrent_writers_serialize_with_no_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            n = 16
            barrier = threading.Barrier(n)

            def worker(i: int) -> None:
                barrier.wait()
                _route(tmp, key=f"k{i}", cause_key="C", now=None)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            snap = fr.load_queue(queue=str(tmp / "queue.md"))
            self.assertEqual(snap["total"], 1)          # one cause -> one row
            self.assertEqual(snap["rows"][0]["hits"], n)  # every concurrent stop folded in
            hist = tmp / "findings-followup-queue.history.jsonl"
            lines = [ln for ln in hist.read_text(encoding="utf-8").splitlines() if ln.strip()]
            self.assertEqual(len(lines), n)
            for ln in lines:
                json.loads(ln)  # no torn/interleaved writes


if __name__ == "__main__":
    unittest.main()
