#!/usr/bin/env python3
"""Planning tests for deficit-only account top-ups."""

import dispatch_account_topup as topup


def test_plan_caps_to_deficit_and_skips_busy_or_recent_issues():
    saved = (
        topup.live_by_account,
        topup.ird.lane_issue_numbers,
        topup.ird.live_resolution_issues,
        topup.ird.recently_attempted_issues,
    )
    topup.live_by_account = lambda _account: 1
    topup.ird.lane_issue_numbers = lambda _repo, _lane: {
        "lane": "docs",
        "numbers": [101, 102, 103, 104],
    }
    topup.ird.live_resolution_issues = lambda _runs: [101]
    topup.ird.recently_attempted_issues = lambda _runs, cooldown_min: [102]
    try:
        plan = topup.plan("seat-a", target=3, lane_arg="docs")
    finally:
        (
            topup.live_by_account,
            topup.ird.lane_issue_numbers,
            topup.ird.live_resolution_issues,
            topup.ird.recently_attempted_issues,
        ) = saved

    assert plan == {
        "account": "seat-a",
        "live": 1,
        "target": 3,
        "deficit": 2,
        "lane": "docs",
        "targets": [103, 104],
    }


def test_plan_at_target_selects_nothing():
    saved = (
        topup.live_by_account,
        topup.ird.lane_issue_numbers,
        topup.ird.live_resolution_issues,
        topup.ird.recently_attempted_issues,
    )
    topup.live_by_account = lambda _account: 4
    topup.ird.lane_issue_numbers = lambda _repo, _lane: {"lane": "docs", "numbers": [7]}
    topup.ird.live_resolution_issues = lambda _runs: []
    topup.ird.recently_attempted_issues = lambda _runs, cooldown_min: []
    try:
        plan = topup.plan("seat-a", target=2, lane_arg=None)
    finally:
        (
            topup.live_by_account,
            topup.ird.lane_issue_numbers,
            topup.ird.live_resolution_issues,
            topup.ird.recently_attempted_issues,
        ) = saved

    assert plan["deficit"] == 0 and plan["targets"] == []

