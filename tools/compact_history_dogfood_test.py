#!/usr/bin/env python3
"""Contract tests for the long-history dogfood request builder."""

import json

import compact_history_dogfood as dogfood


def test_large_body_is_unique_valid_and_token_estimate_tracks_bytes():
    body = dogfood.build_body(8)
    raw = json.dumps(body).encode("utf-8")

    assert dogfood.is_valid_messages(raw)
    assert not dogfood.is_valid_messages(b"not-json")
    assert dogfood.est_tokens(raw) == len(raw) // 4
    assert body["system"][0]["cache_control"]["type"] == "ephemeral"
    assert body["messages"][0]["content"][0]["cache_control"]["type"] == "ephemeral"
    later = [m["content"][0]["text"] for m in body["messages"][1:]]
    assert len(later) == 7 and len(set(later)) == 7


def test_empty_message_list_is_not_a_valid_forwarded_body():
    assert not dogfood.is_valid_messages(json.dumps({"messages": []}).encode("utf-8"))

