#!/usr/bin/env python3
"""Journal-fold and prose tests for the session-effectiveness SVG generator."""

import gen_session_effectiveness_svg as svg


def test_analyze_folds_verdicts_chain_workmix_and_duration():
    rows = [
        {
            "ts_unix_nano": 1,
            "hash": "a",
            "prev_hash": "",
            "verdict": "ALLOW",
            "tool": "Read",
            "by": "monitor",
        },
        {
            "ts_unix_nano": 3_600_000_000_001,
            "hash": "b",
            "prev_hash": "a",
            "verdict": "DENY",
            "tool": "Write",
            "by": "gitgate",
            "reason": "POLICY_BLOCK",
        },
    ]

    summary = svg.analyze(rows)

    assert summary["total"] == 2
    assert summary["allow"] == 1 and summary["deny"] == 1
    assert summary["dur_h"] == 1 and summary["dur_m"] == 0
    assert summary["breaks"] == 0
    assert summary["reads"] == 1 and summary["writes"] == 1
    assert summary["workmix"]["explore"] == 1
    assert summary["workmix"]["author"] == 1
    assert sum(summary["bins"]) == 2


def test_duration_and_denial_prose_handle_short_and_guarded_runs():
    assert svg.run_length(0, 5) == "5-minute"
    assert svg.run_clock(1, 0) == "1 hour"
    assert svg.run_straight(2, 0) == "two straight hours"
    assert svg.deny_prose("Write", "POLICY_BLOCK", "gitgate") == (
        "off-policy git action — refused at the git gate"
    )
    assert "tainted data" in svg.deny_prose("SendMessage", "IFC", "ifc-sink")


def test_svg_escape_covers_all_xml_metacharacters_it_accepts():
    assert svg.esc("a&b<c>d") == "a&amp;b&lt;c&gt;d"
