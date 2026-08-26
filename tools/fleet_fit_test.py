#!/usr/bin/env python3
"""Numerical behavior tests for fleet scaling-law helpers."""

import numpy as np

import fleet_fit as fit


def test_error_metrics_distinguish_exact_and_biased_predictions():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert fit.rmse(y, y) == 0.0
    assert fit.adj_r2(y, y, 1) == 1.0
    assert fit.rmse(y, y + 2.0) == 2.0
    assert fit.adj_r2(y, np.repeat(y.mean(), len(y)), 1) < 1.0


def test_coupon_form_matches_first_agent_zero_and_grows_with_coverage():
    cols = {
        "turns": np.array([1.0, 10.0, 10.0]),
        "agents": np.array([1.0, 2.0, 4.0]),
    }
    pred, capacity, tau = fit.coupon_form(cols, pool=8, p_shared=0.5)

    assert pred[0] == 0.0
    assert pred[2] > pred[1] > 0.0
    assert capacity == 8 and tau == 16.0
    assert fit.coupon_form(cols, pool=0, p_shared=0.5) is None


def test_slope_in_agents_recovers_known_through_origin_slope():
    cols = {
        "turns": np.array([10.0, 10.0, 10.0, 20.0]),
        "agents": np.array([1.0, 2.0, 4.0, 2.0]),
        "cross_uplift_mean": np.array([0.0, 3.0, 9.0, 99.0]),
    }
    assert fit.slope_in_A(cols, 10) == 3.0
    assert fit.slope_in_A(cols, 30) is None

