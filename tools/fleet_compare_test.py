#!/usr/bin/env python3
"""Tests for fleet_compare — the "fleet vs isolated baseline, absolute terms" figure.

These exercise the PURE slice function (slice_fixed) on a synthetic in-memory
column dict, so the load-bearing invariant the figure rests on — isolated =
shared - cross, and the OTHER axis comes back sorted ascending — is pinned without
rendering a plot or reading a CSV. matplotlib is import-guarded to Agg in the
module, so importing it here is headless-safe."""
import unittest

import numpy as np

import fleet_compare as fc


def cols():
    # Two agents rows (50) at turns {30,10,20} deliberately OUT of order, so the
    # sort inside slice_fixed is actually exercised. cross < shared everywhere.
    return {
        "agents": np.array([50.0, 50.0, 50.0, 20.0]),
        "turns": np.array([30.0, 10.0, 20.0, 30.0]),
        "shared_saved_mean": np.array([90.0, 40.0, 65.0, 30.0]),
        "cross_uplift_mean": np.array([25.0, 10.0, 18.0, 8.0]),
    }


class SliceFixed(unittest.TestCase):
    def test_isolated_is_shared_minus_cross(self):
        xs, sh, iso, cr = fc.slice_fixed(cols(), "agents", 50.0)
        # only the three agents==50 cells, sorted by turns ascending
        np.testing.assert_array_equal(xs, np.array([10.0, 20.0, 30.0]))
        np.testing.assert_array_equal(sh, np.array([40.0, 65.0, 90.0]))
        np.testing.assert_array_equal(cr, np.array([10.0, 18.0, 25.0]))
        # the invariant the shaded gap depends on
        np.testing.assert_array_equal(iso, sh - cr)
        np.testing.assert_array_equal(iso, np.array([30.0, 47.0, 65.0]))

    def test_fixing_turns_sweeps_the_agents_axis(self):
        xs, sh, iso, cr = fc.slice_fixed(cols(), "turns", 30.0)
        # turns==30 appears at agents 50 and 20 -> sorted by agents ascending
        np.testing.assert_array_equal(xs, np.array([20.0, 50.0]))
        np.testing.assert_array_equal(sh, np.array([30.0, 90.0]))
        np.testing.assert_array_equal(iso, sh - cr)


if __name__ == "__main__":
    unittest.main()
