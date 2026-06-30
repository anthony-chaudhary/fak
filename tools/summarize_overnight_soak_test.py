#!/usr/bin/env python3
"""Tests for the overnight-soak summarizer's pure core (tools/summarize_overnight_soak.py).

Covers the cell formatters (`fmt_num`, `fmt_bool`) and `tick_rows` — the pure
transform from a raw soak summary into flat per-tick rows, including the
missing-metric defaults and the empty-input case. No disk / run-tree needed.

Run: `python tools/summarize_overnight_soak_test.py`  (exit 0 = all pass),
or `python -m pytest tools/summarize_overnight_soak_test.py -q`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_overnight_soak as sos  # noqa: E402


# --- fmt_num ---------------------------------------------------------------

def test_fmt_num_none_is_dash() -> None:
    assert sos.fmt_num(None) == "-"


def test_fmt_num_rounds_to_digits() -> None:
    assert sos.fmt_num(3.14159, digits=2) == "3.14"
    assert sos.fmt_num(2, digits=1) == "2.0"


def test_fmt_num_non_numeric_passthrough() -> None:
    assert sos.fmt_num("n/a") == "n/a"


# --- fmt_bool --------------------------------------------------------------

def test_fmt_bool_tristate() -> None:
    assert sos.fmt_bool(None) == "-"
    assert sos.fmt_bool(True) == "yes"
    assert sos.fmt_bool(False) == "no"
    assert sos.fmt_bool(1) == "yes" and sos.fmt_bool(0) == "no"


# --- tick_rows -------------------------------------------------------------

def test_tick_rows_empty_summary() -> None:
    assert sos.tick_rows({}) == []
    assert sos.tick_rows({"ticks": None}) == []


def test_tick_rows_extracts_nested_metrics() -> None:
    summary = {"ticks": [{
        "tick": "h01", "failures": 0, "duration_sec": 12.5, "boundary_heavy": True,
        "metrics": {
            "fakbench": {"gate": "PASS", "p50_ns": 100, "speedup_x": 2.0},
            "turntax": {"turns_saved": 3, "tokens_saved": 500},
            "node_compare": {"nodes": 2, "hosts": ["a", "b"]},
        },
    }]}
    rows = sos.tick_rows(summary)
    assert len(rows) == 1
    r = rows[0]
    assert r["tick"] == "h01" and r["fak_gate"] == "PASS" and r["speedup_x"] == 2.0
    assert r["turns_saved"] == 3 and r["nodes"] == 2 and r["hosts"] == ["a", "b"]


def test_tick_rows_defaults_missing_metrics_to_none() -> None:
    # a tick with no metrics block: every metric field is None, hosts is [].
    rows = sos.tick_rows({"ticks": [{"tick": "h02"}]})
    assert len(rows) == 1
    r = rows[0]
    assert r["tick"] == "h02"
    assert r["fak_gate"] is None and r["model_tok_s"] is None
    assert r["hosts"] == []


def _run_all() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
