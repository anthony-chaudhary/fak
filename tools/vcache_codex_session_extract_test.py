#!/usr/bin/env python3
"""Hermetic tests for tools/vcache_codex_session_extract.py."""
from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "vcache_codex_session_extract.py"


def load():
    spec = importlib.util.spec_from_file_location("vcache_codex_session_extract", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class SanitizeTest(unittest.TestCase):
    def test_token_count_row_is_stripped_to_usage_only(self) -> None:
        mod = load()
        row = {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "last_token_usage": {
                        "input_tokens": 2006,
                        "cached_input_tokens": 1920,
                        "output_tokens": 12,
                    },
                    "prompt_text": "must not be copied",
                },
            },
            "extra": "must not be copied",
        }
        self.assertEqual(
            mod.sanitize_row(row),
            {
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "last_token_usage": {
                            "input_tokens": 2006,
                            "cached_input_tokens": 1920,
                        }
                    },
                },
            },
        )

    def test_turn_completed_usage_is_preserved(self) -> None:
        mod = load()
        row = {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 24763,
                "cached_input_tokens": 24448,
                "output_tokens": 122,
            },
            "item": {"text": "must not be copied"},
        }
        self.assertEqual(
            mod.sanitize_row(row),
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 24763,
                    "cached_input_tokens": 24448,
                },
            },
        )


class RunTest(unittest.TestCase):
    def test_extracts_only_token_rows_from_explicit_session(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / "raw.jsonl"
            out = Path(td) / "sanitized.jsonl"
            raw.write_text(
                "\n".join(
                    [
                        json.dumps({"type": "response_item", "payload": {"content": "drop"}}),
                        json.dumps(
                            {
                                "type": "event_msg",
                                "payload": {
                                    "type": "token_count",
                                    "info": {
                                        "last_token_usage": {
                                            "input_tokens": 100,
                                            "cached_input_tokens": 50,
                                        }
                                    },
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "type": "turn.completed",
                                "usage": {"input_tokens": 200, "cached_input_tokens": 180},
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            err = io.StringIO()
            code = mod.run(["--session", str(raw), "--out", str(out)], stderr=err, stdout=io.StringIO())
            self.assertEqual(code, 0)
            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["payload"]["info"]["last_token_usage"]["cached_input_tokens"], 50)
            self.assertEqual(rows[1]["usage"]["cached_input_tokens"], 180)
            self.assertIn("wrote 2 sanitized token rows", err.getvalue())

    def test_discovers_session_by_thread_id(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            home = Path(td) / "codex-home"
            session_dir = home / "sessions" / "2026" / "06"
            session_dir.mkdir(parents=True)
            raw = session_dir / "thread-abc123.jsonl"
            out = Path(td) / "out.jsonl"
            raw.write_text(
                json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10, "cached_input_tokens": 9}}),
                encoding="utf-8",
            )

            code = mod.run(
                ["--thread-id", "abc123", "--codex-home", str(home), "--out", str(out)],
                env={},
                stderr=io.StringIO(),
                stdout=io.StringIO(),
            )
            self.assertEqual(code, 0)
            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["usage"]["input_tokens"], 10)

    def test_no_token_rows_is_refuted(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / "raw.jsonl"
            raw.write_text(json.dumps({"type": "response_item"}) + "\n", encoding="utf-8")
            err = io.StringIO()
            code = mod.run(["--session", str(raw), "--out", str(Path(td) / "out.jsonl")], stderr=err)
            self.assertEqual(code, 1)
            self.assertIn("no token usage rows", err.getvalue())


if __name__ == "__main__":
    unittest.main()
