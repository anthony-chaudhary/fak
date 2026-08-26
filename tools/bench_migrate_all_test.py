#!/usr/bin/env python3
"""Behavior tests for the all-artifact benchmark migrator helpers."""

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import bench_migrate_all as migrate


def test_save_load_and_timestamp_round_trip():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "result-20260826-123456.json"
        payload = {"z": 1, "nested": {"ok": True}}
        assert migrate.save_json(path, payload)
        assert migrate.load_json(path) == payload
        assert migrate.get_timestamp_from_path(path) == "20260826T123456Z"
        assert not path.with_suffix(".tmp").exists()


def test_timestamp_falls_back_to_file_mtime_and_machine_hints():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "mac-benchmark.json"
        path.write_text("{}", encoding="utf-8")
        fixed = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc).timestamp()
        os.utime(path, (fixed, fixed))
        assert migrate.get_timestamp_from_path(path) == "20250102T030405Z"
        assert migrate.infer_machine_from_path(path) == "node-macos-a"
        assert migrate.infer_machine_from_path(Path("rtx4070.json")) == "anthony"


def test_load_json_rejects_malformed_input():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "bad.json"
        path.write_text("{bad", encoding="utf-8")
        assert migrate.load_json(path) is None
