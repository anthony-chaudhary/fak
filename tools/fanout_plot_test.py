#!/usr/bin/env python3
"""Data-selection tests for fan-out plot generation."""

import tempfile
from pathlib import Path

import numpy as np

import fanout_plot as plot


def test_dashboard_slice_prefers_2048_prefix_and_masks_every_column():
    cols = {
        "prefix_tokens": np.array([1024.0, 2048.0, 1024.0, 2048.0]),
        "agents": np.array([1.0, 1.0, 2.0, 2.0]),
        "saved": np.array([10.0, 20.0, 30.0, 40.0]),
    }

    selected, prefix = plot.dashboard_slice(cols)

    assert prefix == 2048
    np.testing.assert_array_equal(selected["agents"], [1.0, 2.0])
    np.testing.assert_array_equal(selected["saved"], [20.0, 40.0])


def test_load_csv_converts_all_values_to_numeric_arrays():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "fanout.csv"
        path.write_text("agents,saved\n1,0.5\n8,4.25\n", encoding="utf-8")
        cols = plot.load_csv(path)
        np.testing.assert_array_equal(cols["agents"], [1.0, 8.0])
        np.testing.assert_allclose(cols["saved"], [0.5, 4.25])

