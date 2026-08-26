#!/usr/bin/env python3
"""Contract tests for the combined compaction/guard dogfood payload."""

import compact_guard_combined_dogfood as dogfood


def test_body_preserves_cache_prefix_and_alternates_later_turns():
    body = dogfood.build_body(6)

    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert body["messages"][0]["content"][0]["cache_control"] == {
        "type": "ephemeral"
    }
    assert [m["role"] for m in body["messages"]] == [
        "user",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    texts = [m["content"][0]["text"] for m in body["messages"][1:]]
    assert len(texts) == len(set(texts))
    assert body["model"] == "claude-mock" and body["max_tokens"] == 64


def test_recorder_instances_do_not_share_bodies():
    left = dogfood.Recorder()
    right = dogfood.Recorder()
    left.bodies.append(b"one")
    assert right.bodies == []

