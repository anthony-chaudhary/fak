#!/usr/bin/env python3
"""Tests for ctxwin.py — the context-window baseline + self-reducer.

The load-bearing correctness property is PAIRING: a tool_result must be matched to the
tool_use that produced it by the UNIQUE tool_use_id, NOT by deduping assistant lines on
message.id. Claude Code streams an assistant turn as several JSONL snapshots that SHARE
one message.id and accumulate tool_use blocks; deduping by message.id keeps only the
first snapshot and loses later tool_use blocks, so most tool_results fail to resolve their
tool name/path. (That bug faked an "88% superseded" reduction headroom during development.)

Run: python tools/ctxwin_test.py     (exits non-zero on failure)
Also importable as a pytest module (test_* functions).
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctxwin as C


def _assistant_tooluse(mid, blocks):
    """blocks: list of (toolu_id, name, input_dict)."""
    content = [{"type": "tool_use", "id": tu, "name": nm, "input": inp} for tu, nm, inp in blocks]
    return {"type": "assistant", "message": {"id": mid, "content": content}}


def _user_results(results):
    """results: list of (toolu_id, content_str)."""
    content = [{"type": "tool_result", "tool_use_id": tu, "content": s} for tu, s in results]
    return {"type": "user", "message": {"content": content}}


def _write(recs):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "sess.jsonl")
    with open(p, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    return p


def test_pairing_survives_streamed_message_id_snapshots():
    # Same assistant message.id 'm1' written as TWO growing snapshots (stream), then the
    # user results for BOTH tool_use ids. Deduping by message.id would drop the 2nd block.
    recs = [
        _assistant_tooluse("m1", [("t1", "Read", {"file_path": "/a"})]),
        _assistant_tooluse("m1", [("t1", "Read", {"file_path": "/a"}),
                                  ("t2", "Bash", {"command": "ls"})]),
        _user_results([("t1", "AAAA" * 50), ("t2", "BBBB" * 50)]),
    ]
    items, meta = C.parse_window(_write(recs))
    results = [it for it in items if it.kind == "result"]
    assert meta["n_results"] == 2, meta
    tools = sorted(it.tool for it in results)
    # both resolved to their real tool name — NOT "?" (the artifact)
    assert tools == ["Bash", "Read"], tools
    assert all(it.tool != "?" for it in results)


def test_stale_read_flagged_when_later_write_same_path():
    recs = [
        _assistant_tooluse("m1", [("t1", "Read", {"file_path": "/x"})]),
        _user_results([("t1", "x" * 4000)]),
        _assistant_tooluse("m2", [("t2", "Edit", {"file_path": "/x"})]),
        _user_results([("t2", "ok")]),
        # a Read of a DIFFERENT path with no later write -> not stale
        _assistant_tooluse("m3", [("t3", "Read", {"file_path": "/y"})]),
        _user_results([("t3", "y" * 4000)]),
    ]
    items, _ = C.parse_window(_write(recs))
    by_path = {it.path: it for it in items if it.kind == "result" and it.tool == "Read"}
    assert by_path[os.path.normcase("/x")].stale is True
    assert by_path[os.path.normcase("/y")].stale is False


def test_stale_not_flagged_when_write_precedes_read():
    # Edit /z THEN Read /z -> the read is the CURRENT truth, not stale.
    recs = [
        _assistant_tooluse("m1", [("t1", "Edit", {"file_path": "/z"})]),
        _user_results([("t1", "ok")]),
        _assistant_tooluse("m2", [("t2", "Read", {"file_path": "/z"})]),
        _user_results([("t2", "z" * 4000)]),
    ]
    items, _ = C.parse_window(_write(recs))
    rd = [it for it in items if it.kind == "result" and it.tool == "Read"][0]
    assert rd.stale is False


def test_reduce_is_recoverable_and_bounded():
    recs = [
        _assistant_tooluse("m1", [("t1", "Read", {"file_path": "/big"})]),
        _user_results([("t1", "Z" * 40000)]),     # 10000 tok
        _assistant_tooluse("m2", [("t2", "Edit", {"file_path": "/big"})]),  # makes /big stale
        _user_results([("t2", "ok")]),
    ]
    items, _ = C.parse_window(_write(recs))
    rep = C.reduce_window(items, 1000)
    assert rep["reduced_tok"] <= rep["baseline_tok"]
    assert rep["ratio"] >= 1.0
    assert rep["recoverable_frac"] == 1.0
    # the big read is stale -> collapsed by the stale tier, not the window tier
    assert rep["tiers"]["stale"]["n"] == 1
    assert rep["tiers"]["stale"]["removed_tok"] > 0


def test_no_fabrication_below_floor():
    # 20 small unique results; cannot reach 2x without windowing below MIN_BUDGET.
    recs = []
    for i in range(20):
        recs.append(_assistant_tooluse(f"m{i}", [(f"t{i}", "Bash", {"command": f"c{i}"})]))
        recs.append(_user_results([(f"t{i}", f"out{i} " + "k" * 380)]))
    items, _ = C.parse_window(_write(recs))
    rep, budget = C.autotune(items, 2.0)
    assert budget >= C.MIN_BUDGET
    assert rep["ratio"] < 1.5, rep["ratio"]   # honest: cannot fabricate 2x


def test_noise_filter_strips_cruft_keeps_signal():
    rep = "this is a long repeated log line that just wastes context tokens"
    raw = "\x1b[31mERR\x1b[0m\n" + ((rep + "\n") * 6) + "unique-signal-tail"
    out = C.filter_noise(raw)
    assert "\x1b[" not in out                                  # ANSI stripped
    assert "unique-signal-tail" in out and "ERR" in out         # signal kept
    assert "identical line above" in out                        # long run collapsed
    assert len(out) < len(raw)                                  # never inflates


def test_noise_filter_never_inflates_short_runs():
    # a SHORT repeated line whose collapse marker would cost more than it saves: keep as-is
    raw = "ab\n" * 3
    assert len(C.filter_noise(raw)) <= len(raw)


def test_tool_use_mass_from_final_snapshot_not_first():
    # Claude Code streams a turn as growing snapshots sharing message.id; the FIRST is
    # often empty and the big Write `content` lands in a LATER snapshot. parse_window must
    # count the tool_use input mass from the FULL (last) snapshot, not the empty first.
    big = "B" * 8000
    recs = [
        {"type": "assistant", "message": {"id": "m1", "content": []}},  # empty first snapshot
        _assistant_tooluse("m1", [("t1", "Write", {"file_path": "/f", "content": big})]),
        _user_results([("t1", "ok")]),
    ]
    items, _ = C.parse_window(_write(recs))
    tu_mass = sum(it.toks for it in items if it.kind == "tool_use")
    # full input (~8000 chars / 4 = ~2000 tok) must be counted, not ~0 from the empty first
    assert tu_mass > 1500, tu_mass


def test_multi_result_in_one_message_orders_by_tool_use_id():
    # Two tool_results delivered in ONE user message, out of tool_use order: Read /p (old),
    # Edit /p, then BOTH results in one message. The Read must be flagged stale (write is
    # later), the ordering bound by tool_use_id — not a name+path delivery cursor.
    recs = [
        _assistant_tooluse("m1", [("tr", "Read", {"file_path": "/p"})]),
        _assistant_tooluse("m2", [("te", "Edit", {"file_path": "/p"})]),
        # one user message carrying both results, Edit result FIRST (out of order)
        _user_results([("te", "ok"), ("tr", "p" * 4000)]),
    ]
    items, _ = C.parse_window(_write(recs))
    rd = [it for it in items if it.kind == "result" and it.tool == "Read"][0]
    assert rd.stale is True


def test_window_content_materializes_head_tail_pointer():
    s = "HEAD " + "z" * 6000 + " TAIL"
    out = C.window_content(s, 400, path="/dir/file.py")
    assert len(out) < len(s)
    assert out.startswith("HEAD") and out.endswith("TAIL")
    assert "ctxwin: elided" in out and "file.py" in out


def test_recoverable_frac_is_measured_not_constant():
    # windowed NON-file result -> middle has no re-fetch handle -> frac < 1.0
    rb = C.reduce_window(C._seq([C._mk("result", "Bash", 8000, salt="b")]), 1000)
    # windowed FILE-backed result -> re-fetchable -> 1.0
    rf = C.reduce_window(C._seq([C._mk("result", "Read", 8000, salt="r", path="/x")]), 1000)
    assert rb["recoverable_frac"] < 1.0
    assert rf["recoverable_frac"] == 1.0


def test_selfcheck_passes():
    assert C.runselfcheck() == 0


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
