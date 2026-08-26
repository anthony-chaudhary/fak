#!/usr/bin/env python3
"""Profile arithmetic tests for realistic turn-agent sweeps."""

import run_turn_agent_realistic_sweep as sweep


def test_parse_ints_ignores_empty_fields_and_whitespace():
    assert sweep.parse_ints("8, 12,,20") == [8, 12, 20]


def test_profile_estimates_compute_context_and_kv_geometry_per_cell():
    profile = {
        "prefix": 100,
        "turns": "2,4",
        "agents": "1,3",
        "decode": 10,
        "result": 5,
    }
    saved = sweep.appversion.app_version
    sweep.appversion.app_version = lambda: "test-version"
    try:
        rows = sweep.profile_estimates(profile)
    finally:
        sweep.appversion.app_version = saved

    assert len(rows) == 4
    row = next(r for r in rows if r["turns"] == 2 and r["agents"] == 3)
    assert row == {
        "version": "test-version",
        "turns": 2,
        "agents": 3,
        "per_agent_context_tokens": 125,
        "unified_kv_cells": 175,
        "llama_n_ctx_with_slack": 239,
        "agent_turns": 6,
    }


def test_shipped_profiles_stay_within_named_context_envelope():
    rows = sweep.profile_estimates(sweep.PROFILES["interactive"])
    assert max(row["per_agent_context_tokens"] for row in rows) <= 8192
    assert {row["agents"] for row in rows} == {8, 12, 16, 20}

