#!/usr/bin/env python3
"""Behavior tests for comprehensive benchmark migration helpers."""

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import bench_migrate_comprehensive as migrate


def test_atomic_json_write_and_invalid_read():
    with tempfile.TemporaryDirectory() as td:
        good = Path(td) / "nested" / "result.json"
        assert migrate.save_json(good, {"rows": [1, 2], "kind": "fleet"})
        assert migrate.load_json(good) == {"kind": "fleet", "rows": [1, 2]}
        good.write_text("not json", encoding="utf-8")
        assert migrate.load_json(good) is None


def test_timestamp_and_parent_machine_inference():
    with tempfile.TemporaryDirectory() as td:
        parent = Path(td) / "node-macos-a"
        parent.mkdir()
        path = parent / "result.json"
        path.write_text("{}", encoding="utf-8")
        fixed = datetime(2024, 2, 3, 4, 5, 6, tzinfo=timezone.utc).timestamp()
        os.utime(path, (fixed, fixed))
        assert migrate.get_timestamp_from_path(path) == "20240203T040506Z"
        assert migrate.infer_machine_from_path(path) == "node-macos-a"


def test_filename_timestamp_wins_over_mtime():
    assert (
        migrate.get_timestamp_from_path(Path("qwen-20260701T112233.json"))
        == "20260701T112233Z"
    )

