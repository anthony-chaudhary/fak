#!/usr/bin/env python3
"""Input and break-even lookup tests for the turn-tax plotter."""

import json
import tempfile
from pathlib import Path

import turntax_plot as plot


def test_self_host_sessions_selects_named_regime_only():
    point = {
        "regimes": [
            {"name": "provider-ships", "sessions_to_break_even": 1},
            {"name": "self-host-fork", "sessions_to_break_even": 1234},
        ]
    }
    assert plot.self_host_sessions(point) == 1234
    assert plot.self_host_sessions({"regimes": []}) is None


def test_load_round_trips_report_without_mutation():
    report = {"real_world_hit_rate": 0.007, "points": [{"hit_rate": 0.1}]}
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "turntax.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        assert plot.load(path) == report


def test_airline_reference_is_exact_nine_of_fourteen():
    assert plot.AIRLINE_H == 9.0 / 14.0

