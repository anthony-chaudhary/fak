#!/usr/bin/env python3
"""Series and lookup tests for Mac-vs-baseline plotting."""

from pathlib import Path

import plot_turn_agent_mac_vs_baseline as plot


def test_series_group_by_turns_and_sort_by_agent_count():
    mac = {
        "points": [
            {"turns": 80, "concurrency": 20, "reuse_agent_turns_per_sec": 3.0},
            {"turns": 80, "concurrency": 8, "reuse_agent_turns_per_sec": 1.5},
            {"turns": 120, "concurrency": 8, "reuse_agent_turns_per_sec": 1.0},
        ]
    }
    baseline = {
        "cells": [
            {"turns": 80, "agents": 20, "fak_agent_turns_per_sec": 2.0},
            {"turns": 80, "agents": 8, "fak_agent_turns_per_sec": 1.0},
        ]
    }
    assert plot.mac_series(mac)[80] == [(8, 1.5), (20, 3.0)]
    assert plot.baseline_series(baseline, "fak_agent_turns_per_sec")[80] == [
        (8, 1.0),
        (20, 2.0),
    ]


def test_lookups_accept_concurrency_or_agents_and_return_none_when_absent():
    points = [{"turns": 10, "concurrency": 4, "rate": "2.5"}]
    cells = [{"turns": 10, "agents": 4, "rate": 3}]
    assert plot.point_lookup(points, 10, 4, "rate") == 2.5
    assert plot.baseline_lookup(cells, 10, 4, "rate") == 3.0
    assert plot.point_lookup(points, 20, 4, "rate") is None


def test_artifact_label_is_relative_inside_repo_and_safe_outside():
    inside = plot.ROOT / "experiments" / "sample.json"
    outside = Path("D:/bench/sample.json")
    assert plot._artifact_label(inside) == str(Path("experiments") / "sample.json")
    assert plot._artifact_label(outside) == str(outside)

