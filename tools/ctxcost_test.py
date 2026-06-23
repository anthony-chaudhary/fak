#!/usr/bin/env python3
"""Tests for ctxcost.py — the per-turn O(1)-context vs append-only-with-cache cost replay.

The load-bearing correctness properties:
  1) PARSING: streaming snapshots that SHARE one assistant message.id are ONE turn, not many
     (Claude Code writes a turn as several growing JSONL snapshots; counting them as separate
     turns inflates every total ~2.4x). We keep one record per message.id, the most-complete.
  2) ANTI-OVERCLAIM: at a budget >= the largest prompt, reconstruct is a no-op, so C must equal
     A exactly. The tool cannot fabricate a saving.
  3) THE CROSSOVER IS THE CACHE'S EFFECTIVE DISCOUNT: C beats B exactly when the window drops
     below the cache's effective input multiplier as a fraction of the full context. This is the
     headline finding; we pin it numerically on a synthetic warm session.

Run: python tools/ctxcost_test.py     (exits non-zero on failure)
Also importable as a pytest module (test_* functions).
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctxcost as C


def _assistant(mid, fresh, write, read, out):
    return {"type": "assistant",
            "message": {"id": mid, "usage": {
                "input_tokens": fresh, "cache_creation_input_tokens": write,
                "cache_read_input_tokens": read, "output_tokens": out}}}


def _write(recs):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "sess.jsonl")
    with open(p, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    return p


def _warm_session(n=50, delta=1000, out=100):
    """A perfectly-warm append-only session: each turn adds `delta`, all prior is cache-read."""
    turns = []
    prefix = 0
    for _ in range(n):
        turns.append(C.Turn(fresh=2, write=delta, read=prefix, out=out))
        prefix += delta
    return turns


def test_parse_dedups_streamed_snapshots_to_one_turn():
    # message.id 'm1' written as THREE growing snapshots (out grows 50->100->150); same
    # input accounting. Must collapse to ONE turn keeping the most-complete (out=150).
    recs = [
        _assistant("m1", 2, 8000, 40000, 50),
        _assistant("m1", 2, 8000, 40000, 100),
        _assistant("m1", 2, 8000, 40000, 150),
        _assistant("m2", 2, 1000, 48000, 80),
    ]
    turns = C.parse_turns(_write(recs))
    assert len(turns) == 2, f"snapshots not deduped: {len(turns)} turns"
    assert turns[0].out == 150, f"kept a partial snapshot: out={turns[0].out}"
    assert turns[0].prompt == 48002, turns[0].prompt   # 2 + 8000 + 40000


def test_no_overclaim_C_equals_A_at_full_budget():
    turns = _warm_session()
    big = max(t.prompt for t in turns) + 1
    r = C.replay_session(turns, big)
    assert abs(r["C"]["cost"] - r["A"]["cost"]) < 1e-6, (r["C"]["cost"], r["A"]["cost"])


def test_cache_never_costs_more_than_no_cache():
    turns = _warm_session()
    r = C.replay_session(turns, 4000)
    assert r["B"]["cost"] <= r["A"]["cost"] + 1e-6


def test_thesis_is_a_crossover_not_unconditional():
    turns = _warm_session()
    small = C.replay_session(turns, 2000)["C"]["cost"]
    large = C.replay_session(turns, 8000)["C"]["cost"]
    b = C.replay_session(turns, 2000)["B"]["cost"]
    assert small < b < large, (small, b, large)


def test_crossover_equals_cache_effective_input_multiplier():
    # The headline identity: the budget fraction at which C ties B equals B's effective
    # input multiplier (its input bill / the full-price bill on the same input tokens).
    turns = _warm_session(n=80)
    fresh = sum(t.fresh for t in turns)
    write = sum(t.write for t in turns)
    read = sum(t.read for t in turns)
    prompt = fresh + write + read
    eff = (fresh * C.M_FRESH + write * C.M_WRITE_5M + read * C.M_READ) / prompt
    _, frac = C.crossover([turns], "B", "C")
    assert frac is not None
    assert abs(frac - eff) < 0.01, f"crossover_frac={frac} != effective_mult={eff}"


def test_eviction_degrades_B_toward_A():
    turns = _warm_session()
    warm = C.replay_session(turns, 4000, evict_frac=0.0)["B"]["cost"]
    cold = C.replay_session(turns, 4000, evict_frac=1.0)["B"]["cost"]
    a = C.replay_session(turns, 4000)["A"]["cost"]
    assert warm < cold
    assert abs(cold - a) < 1e-6, (cold, a)


def test_C_monotonic_in_budget():
    turns = _warm_session()
    c_small = C.replay_session(turns, 2000)["C"]["cost"]
    c_big = C.replay_session(turns, 20000)["C"]["cost"]
    assert c_big >= c_small - 1e-6


def test_trace_is_deterministic_and_flags_truncation():
    # A session where some turns add more NEW context than an 8k window can hold: the trace
    # must FLAG those as observability events (holds_new_context=False), not hide them, and
    # must be byte-identical on a replay (deterministic function of (turns, budget)).
    turns = [
        C.Turn(fresh=2, write=3000, read=0, out=100),       # new=3002 fits 8k
        C.Turn(fresh=2, write=20000, read=3000, out=100),   # new=20002 does NOT fit 8k
        C.Turn(fresh=2, write=1000, read=23000, out=100),   # new=1002 fits 8k
    ]
    a = C.trace_session(turns, 8000)
    b = C.trace_session(turns, 8000)
    assert a == b, "trace is not deterministic"
    assert a[0]["reconstruct"]["holds_new_context"] is True
    assert a[1]["reconstruct"]["holds_new_context"] is False   # the truncation event is visible
    assert a[2]["reconstruct"]["holds_new_context"] is True
    summ = C.trace_summary(a)
    assert summ["turns_window_cannot_hold_new_context"] == 1
    assert summ["max_new_context_tok"] == 20002
    # every turn carries the full observable record (prompt 23002, window 8000 -> pruned 15002)
    assert a[1]["new_context_tok"] == 20002 and a[1]["reconstruct"]["pruned_tok"] == 15002


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
