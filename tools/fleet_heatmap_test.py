#!/usr/bin/env python3
"""Grid-construction tests for fleet heatmap rendering."""

import tempfile
from pathlib import Path

import numpy as np

import fleet_heatmap as heatmap


def test_grid_sorts_axes_and_places_sparse_cells():
    cols = {
        "turns": np.array([20.0, 10.0, 20.0]),
        "agents": np.array([4.0, 2.0, 2.0]),
        "saved": np.array([8.0, 3.0, 5.0]),
    }
    turns, agents, values = heatmap.grid(cols, "saved")

    np.testing.assert_array_equal(turns, [10.0, 20.0])
    np.testing.assert_array_equal(agents, [2.0, 4.0])
    assert values[0, 0] == 3.0 and values[0, 1] == 5.0
    assert np.isnan(values[1, 0]) and values[1, 1] == 8.0


def test_load_csv_empty_and_populated_contracts():
    with tempfile.TemporaryDirectory() as td:
        empty = Path(td) / "empty.csv"
        empty.write_text("turns,agents,saved\n", encoding="utf-8")
        assert heatmap.load_csv(empty) is None

        full = Path(td) / "full.csv"
        full.write_text("turns,agents,saved\n10,2,3.5\n", encoding="utf-8")
        cols = heatmap.load_csv(full)
        assert cols["saved"].tolist() == [3.5]

