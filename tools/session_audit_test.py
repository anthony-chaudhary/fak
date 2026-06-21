#!/usr/bin/env python3
"""Hermetic tests for tools/session_audit.py.

Locks in the de-dup invariant: Claude Code writes MULTIPLE transcript lines per
billed assistant turn (streaming events / retries / sidechain re-serialization),
all carrying the SAME message.usage. The auditor must fold each billed turn ONCE
(keyed on message.id), or every token/cost/turn total runs ~2x high. A regression
here silently doubles every reported number, so this test is the witness that the
fix from 2026-06-20 (heaviest session 093ca0fc: 901->455 turns, $634->$323) holds.
"""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "session_audit.py"


def load():
    spec = importlib.util.spec_from_file_location("session_audit", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _assistant(msg_id, *, out, cread, ccreate, inp=0, tool=None, model="claude-opus-4-8"):
    """One assistant transcript record with a given message.id and usage."""
    content = []
    if tool:
        content.append({"type": "tool_use", "name": tool, "input": {}})
    return {
        "type": "assistant",
        "timestamp": "2026-06-20T00:00:00.000Z",
        "uuid": f"uuid-{msg_id}-{out}-{cread}",   # per-LINE, intentionally unique
        "message": {
            "id": msg_id,
            "model": model,
            "usage": {
                "input_tokens": inp,
                "output_tokens": out,
                "cache_read_input_tokens": cread,
                "cache_creation_input_tokens": ccreate,
            },
            "content": content,
        },
    }


def _write_transcript(records):
    tmp = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    for r in records:
        tmp.write(json.dumps(r) + "\n")
    tmp.close()
    return tmp.name


class DedupTest(unittest.TestCase):
    def test_duplicate_billed_turn_counted_once(self) -> None:
        sa = load()
        # The same billed turn re-serialized 4x, then a distinct second turn 2x.
        recs = (
            [_assistant("msg-A", out=400, cread=50_000, ccreate=6_000)] * 4
            + [_assistant("msg-B", out=500, cread=60_000, ccreate=7_000, tool="Bash")] * 2
        )
        s = sa.analyze(_write_transcript(recs))

        self.assertEqual(s["assistant_turns"], 2, "two distinct message.ids = two turns")
        self.assertEqual(s["dup_assistant_lines"], 4, "the 6 lines hold 4 duplicates")
        self.assertEqual(s["tokens"]["output"], 900, "400 + 500, not multiplied")
        self.assertEqual(s["tokens"]["cache_read"], 110_000)
        self.assertEqual(s["tokens"]["cache_create"], 13_000)
        self.assertEqual(s["n_tool_use"], 1, "the duplicated tool_use is not re-counted")
        self.assertEqual(s["tools"].get("Bash"), 1)

    def test_no_duplicates_is_a_noop(self) -> None:
        sa = load()
        recs = [
            _assistant("msg-1", out=100, cread=10_000, ccreate=1_000),
            _assistant("msg-2", out=200, cread=20_000, ccreate=2_000),
        ]
        s = sa.analyze(_write_transcript(recs))
        self.assertEqual(s["assistant_turns"], 2)
        self.assertEqual(s["dup_assistant_lines"], 0)
        self.assertEqual(s["tokens"]["output"], 300)

    def test_idless_lines_each_count(self) -> None:
        sa = load()
        # Defensive: a record with no message.id must NOT collapse into one bucket.
        r = _assistant("x", out=50, cread=5_000, ccreate=500)
        del r["message"]["id"]
        s = sa.analyze(_write_transcript([dict(r), dict(r)]))
        self.assertEqual(s["assistant_turns"], 2, "id-less lines are counted individually")
        self.assertEqual(s["tokens"]["output"], 100)

    def test_cost_is_per_deduped_turn(self) -> None:
        sa = load()
        # Opus output @ $75/MTok: 1000 out tok = $0.075, regardless of dup lines.
        recs = [_assistant("msg-only", out=1_000, cread=0, ccreate=0)] * 3
        s = sa.analyze(_write_transcript(recs))
        self.assertAlmostEqual(s["cost_usd"], 1_000 * 75.0 / 1e6, places=9)


if __name__ == "__main__":
    unittest.main()
