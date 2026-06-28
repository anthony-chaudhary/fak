#!/usr/bin/env python3
"""Tests for cache-value ledger (issue #1075).

Exercises the ledger path, regression check, and JSONL persistence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import cache_value_ledger as ledger  # noqa: E402


def test_extract_metrics_from_score():
    payload = {
        "schema": "fak.vcache.score.v1",
        "active_multiplier": 3.76,
        "two_x_better": True,
        "grade": "A",
        "economics": {
            "hit_rate": 0.9553,
            "cache_read_tokens": 10163712,
            "rebate_token_equiv": 9147340.8,
            "cost_token_equiv": 1491490.2,
            "baseline_token_equiv": 10638831,
            "multiplier": 7.13,
        },
    }
    entry = ledger.extract_metrics(payload)
    assert entry.active_multiplier == 3.76
    assert entry.two_x_better is True
    assert entry.grade == "A"
    assert entry.economics_hit_rate == 0.9553
    assert entry.economics_multiplier == 7.13


def test_append_and_read_ledger():
    with NamedTemporaryFile(mode="w+", suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    try:
        entry = ledger.CacheValueEntry(
            utc="2026-06-27T12:00:00Z",
            active_multiplier=3.76,
            two_x_better=True,
            grade="A",
            economics_hit_rate=0.9553,
            economics_cache_read_tokens=10163712,
            economics_rebate_token_equiv=9147340.8,
            economics_cost_token_equiv=1491490.2,
            economics_baseline_token_equiv=10638831,
            economics_multiplier=7.13,
        )
        ledger.append_entry(path, entry)
        entries = ledger.read_ledger(path)
        assert len(entries) == 1
        assert entries[0].active_multiplier == 3.76
        assert entries[0].two_x_better is True
    finally:
        path.unlink(missing_ok=True)


def test_regression_check_pass():
    entry = ledger.CacheValueEntry(
        utc="2026-06-27T12:00:00Z",
        active_multiplier=3.76,
        two_x_better=True,
        grade="A",
        economics_hit_rate=0.9553,
        economics_cache_read_tokens=10163712,
        economics_rebate_token_equiv=9147340.8,
        economics_cost_token_equiv=1491490.2,
        economics_baseline_token_equiv=10638831,
        economics_multiplier=7.13,
    )
    ok, reason = ledger.check_regression(entry, 2.0)
    assert ok is True
    assert "3.76 >= 2x" in reason


def test_regression_check_fail():
    entry = ledger.CacheValueEntry(
        utc="2026-06-27T12:00:00Z",
        active_multiplier=1.5,
        two_x_better=False,
        grade="F",
        economics_hit_rate=0.5,
        economics_cache_read_tokens=1000,
        economics_rebate_token_equiv=500,
        economics_cost_token_equiv=500,
        economics_baseline_token_equiv=1000,
        economics_multiplier=2.0,
    )
    ok, reason = ledger.check_regression(entry, 2.0)
    assert ok is False
    assert "1.50 < 2x" in reason


def test_ledger_ignore_malformed_rows():
    with NamedTemporaryFile(mode="w+", suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    try:
        path.write_text('{"active_multiplier":3.76,"two_x_better":true}\nnot json\n')
        entries = ledger.read_ledger(path)
        assert len(entries) == 1
        assert entries[0].active_multiplier == 3.76
    finally:
        path.unlink(missing_ok=True)


def _run() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run())