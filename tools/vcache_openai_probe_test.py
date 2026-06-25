#!/usr/bin/env python3
"""Hermetic tests for tools/vcache_openai_probe.py."""
from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "vcache_openai_probe.py"


def load():
    spec = importlib.util.spec_from_file_location("vcache_openai_probe", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PayloadTest(unittest.TestCase):
    def test_chat_payload_keeps_anchor_in_first_message(self) -> None:
        mod = load()
        payload = mod.chat_payload("m", "stable-prefix", "suffix-one", 8)
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "stable-prefix"})
        self.assertEqual(payload["messages"][1]["content"], "Reply exactly: suffix-one")
        self.assertEqual(payload["max_tokens"], 8)

    def test_cached_token_extractors_accept_openai_shapes(self) -> None:
        mod = load()
        self.assertEqual(
            mod.cached_tokens({"input_tokens_details": {"cached_tokens": 1920}}),
            1920,
        )
        self.assertEqual(
            mod.cached_tokens({"prompt_tokens_details": {"cached_tokens": 1024}}),
            1024,
        )
        self.assertEqual(mod.prompt_tokens({"input_tokens": 2006}), 2006)
        self.assertEqual(mod.prompt_tokens({"prompt_tokens": 2006}), 2006)


class RunTest(unittest.TestCase):
    def test_missing_api_key_skips_without_writing(self) -> None:
        mod = load()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "probe.jsonl"
            err = io.StringIO()
            code = mod.run(["--out", str(out)], env={}, stderr=err, stdout=io.StringIO())
            self.assertEqual(code, mod.SKIP)
            self.assertFalse(out.exists())
            self.assertIn("OPENAI_API_KEY is not set", err.getvalue())

    def test_dry_run_prints_plan_without_api_key(self) -> None:
        mod = load()
        out = io.StringIO()
        code = mod.run(["--out", "x.jsonl", "--dry-run"], env={}, stderr=io.StringIO(), stdout=out)
        self.assertEqual(code, 0)
        plan = json.loads(out.getvalue())
        self.assertEqual(plan["schema"], mod.SCHEMA)
        self.assertFalse(plan["api_key_present"])

    def test_writes_provider_rows_with_usage_top_level(self) -> None:
        mod = load()
        calls = []

        def fake_post(url, api_key, payload, timeout):
            calls.append((url, api_key, payload, timeout))
            idx = len(calls)
            return {
                "id": f"resp_{idx}",
                "usage": {
                    "prompt_tokens": 2006,
                    "prompt_tokens_details": {"cached_tokens": 0 if idx == 1 else 1920},
                },
            }

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "probe.jsonl"
            err = io.StringIO()
            code = mod.run(
                ["--out", str(out), "--requests", "2", "--model", "test-model"],
                post=fake_post,
                env={"OPENAI_API_KEY": "sk-test"},
                stderr=err,
                stdout=io.StringIO(),
            )
            self.assertEqual(code, 0)
            self.assertEqual(len(calls), 2)
            self.assertTrue(calls[0][0].endswith("/v1/chat/completions"))
            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["schema"], mod.SCHEMA)
            self.assertEqual(rows[1]["usage"]["prompt_tokens_details"]["cached_tokens"], 1920)
            self.assertIn("cached_tokens=1920", err.getvalue())


if __name__ == "__main__":
    unittest.main()
