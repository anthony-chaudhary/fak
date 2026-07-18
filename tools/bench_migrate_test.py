#!/usr/bin/env python3
"""Hermetic tests for tools/bench_migrate.py.

The migrator's load-bearing pure helpers are a JSON reader that swallows bad
input, an order-independent config hash, and an mtime->UTC-stamp deriver. These
tests exercise each on real temp files with meaningful assertions (never just
"it runs"): the hash is stable across key reordering yet changes with content,
the reader returns None on malformed/absent files, and the stamp is a parseable
20-char ``YYYYmmddTHHMMSSZ`` token pinned to the file's mtime.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "bench_migrate.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("bench_migrate", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = load()


class TestConfigHash(unittest.TestCase):
    def test_deterministic_and_key_order_independent(self):
        a = MOD.config_hash({"model": "q8", "batch": 4})
        b = MOD.config_hash({"batch": 4, "model": "q8"})
        self.assertEqual(a, b, "hash must be invariant to key order (sort_keys)")

    def test_length_is_eight_hex(self):
        h = MOD.config_hash({"x": 1})
        self.assertEqual(len(h), 8)
        int(h, 16)  # raises if not hex

    def test_content_change_changes_hash(self):
        self.assertNotEqual(
            MOD.config_hash({"model": "q8"}),
            MOD.config_hash({"model": "q4"}),
        )


class TestLoadJson(unittest.TestCase):
    def test_reads_valid_json(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.json"
            p.write_text(json.dumps({"host": "da33", "cores": 96}), encoding="utf-8")
            self.assertEqual(MOD.load_json(p), {"host": "da33", "cores": 96})

    def test_missing_file_returns_none(self):
        self.assertIsNone(MOD.load_json(Path("does-not-exist-xyz.json")))

    def test_malformed_json_returns_none(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.json"
            p.write_text("{not: valid", encoding="utf-8")
            self.assertIsNone(MOD.load_json(p))


class TestTimestampFromPath(unittest.TestCase):
    def test_derives_utc_stamp_from_mtime(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.json"
            p.write_text("{}", encoding="utf-8")
            # pin mtime to a known epoch (2021-01-02T03:04:05Z)
            fixed = datetime(2021, 1, 2, 3, 4, 5, tzinfo=timezone.utc).timestamp()
            os.utime(p, (fixed, fixed))
            stamp = MOD.timestamp_from_path(p)
            self.assertEqual(stamp, "20210102T030405Z")

    def test_stamp_is_parseable_token(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.json"
            p.write_text("{}", encoding="utf-8")
            stamp = MOD.timestamp_from_path(p)
            self.assertIsNotNone(stamp)
            # round-trips through the same format string
            datetime.strptime(stamp, "%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    unittest.main()
