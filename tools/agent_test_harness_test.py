#!/usr/bin/env python3
"""Behavior tests for the deterministic agent workflow harness."""

import agent_test_harness as harness


def test_booking_workflow_executes_and_replays_exactly():
    fixture = harness.booking_fixture()
    transcript = harness.run(fixture)

    harness.assert_tool_sequence(
        transcript,
        [
            "get_user_details",
            "fetch_policy",
            "search_direct_flight",
            "convert_currency",
            "book_flight",
        ],
    )
    harness.assert_tool_args(transcript, "book_flight", {"flight_id": "UA123"})
    harness.assert_final_contains(transcript, "220.80 EUR")
    harness.assert_reproducible(fixture, transcript)


def test_transcript_diff_names_first_divergent_message():
    recorded = [harness.msg_user("original")]
    replayed = [harness.msg_user("changed")]

    diff = harness.diff_transcripts(recorded, replayed)
    assert diff is not None
    assert "message 0" in diff
    assert "original" in diff and "changed" in diff

