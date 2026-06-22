#!/usr/bin/env python3
"""Smoke test for transcript_workload.py — the per-message turn aggregation.

The load-bearing correctness property: Claude Code splits ONE assistant API response into
several JSONL records (one per content block) that all carry the SAME message.usage. The
profiler must collapse them into one logical turn keyed by message.id, counting usage once —
otherwise turns ~3× over-count and the tool-call fraction is diluted by thinking/text-only
split records.

Run: python tools/transcript_workload_smoke_test.py   (exits non-zero on failure)
Also importable as a pytest module (test_* functions).
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transcript_workload as tw


def _rec(mid, block, usage, name=None):
    content = [{"type": block}] if block != "tool_use" else [{"type": "tool_use", "name": name}]
    return {"type": "assistant", "isSidechain": False,
            "message": {"id": mid, "usage": usage, "content": content}}


def _tool_result(chars):
    return {"type": "user", "isSidechain": False,
            "message": {"content": [{"type": "tool_result", "content": "x" * chars}]}}


def _write_transcript(path):
    u1 = {"output_tokens": 100, "input_tokens": 2, "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 50}
    u2 = {"output_tokens": 200, "input_tokens": 2, "cache_read_input_tokens": 1500, "cache_creation_input_tokens": 80}
    u3 = {"output_tokens": 60, "input_tokens": 2, "cache_read_input_tokens": 1600, "cache_creation_input_tokens": 10}
    recs = [
        # response 1: thinking + text + 1 tool_use, all same id m1, duplicated usage
        _rec("m1", "thinking", u1), _rec("m1", "text", u1), _rec("m1", "tool_use", u1, "Bash"),
        _tool_result(400),  # -> 100 tok attributed to m1
        # response 2: thinking + text + 2 parallel tool_use (4 split records), same id m2
        _rec("m2", "thinking", u2), _rec("m2", "text", u2),
        _rec("m2", "tool_use", u2, "Read"), _rec("m2", "tool_use", u2, "Read"),
        _tool_result(800),  # -> 200 tok attributed to m2
        # response 3: thinking + text only, NO tool (a non-tool turn), id m3
        _rec("m3", "thinking", u3), _rec("m3", "text", u3),
    ]
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def _analyze():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "sess.jsonl")
    _write_transcript(p)
    return tw.analyze_turns(p)


def test_collapses_split_records_into_logical_turns():
    s = _analyze()
    assert s is not None
    # 10 records but 3 logical responses
    assert s["n_turns"] == 3, s["n_turns"]
    # decode NOT tripled: per-turn output is 100/200/60, total 360 (not 1080)
    assert s["decode_per_turn"] == [100, 200, 60], s["decode_per_turn"]


def test_tool_call_fraction_is_per_response():
    s = _analyze()
    # 2 of 3 responses make a tool call -> 0.667, NOT diluted by split text/thinking records
    assert abs(s["tool_call_fraction"] - 2 / 3) < 1e-6, s["tool_call_fraction"]
    assert s["n_tool_turns"] == 2, s["n_tool_turns"]
    # response 2 made 2 parallel tool calls -> 3 total tool calls across 2 tool turns
    assert s["n_tool_calls"] == 3, s["n_tool_calls"]
    assert s["tools_per_tool_turn"] == [1, 2], s["tools_per_tool_turn"]


def test_prefix_from_first_turn_cache_read():
    s = _analyze()
    assert s["prefix_tokens"] == 1000, s["prefix_tokens"]


def test_result_attributed_to_issuing_turn():
    s = _analyze()
    # 400 chars/4=100 for m1, 800/4=200 for m2; only tool turns carry result
    assert s["result_per_tool_turn"] == [100, 200], s["result_per_tool_turn"]


def test_build_profile_globals_consistent():
    s = _analyze()
    # n_turns=3 < the >=3 keep threshold passes; prefix 1000 >= 256 keep
    prof = tw.build_profile([s], replay_tracks=1)
    assert prof["schema"] == "fak.workload.v1"
    assert prof["app_version"]
    # build_profile rounds the global fraction to 4 dp (0.6667); tolerance reflects that
    assert abs(prof["tool_call_fraction"] - 2 / 3) < 1e-3, prof["tool_call_fraction"]
    # one replay track, 3-turn track, 2 tool turns
    assert len(prof["replay"]) == 1
    assert prof["replay"][0]["version"] == prof["app_version"]
    track = prof["replay"][0]["track"]
    assert len(track) == 3
    assert all(row["version"] == prof["app_version"] for row in track)
    assert sum(1 for x in track if x["tool"]) == 2


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
