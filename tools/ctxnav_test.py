#!/usr/bin/env python3
"""Tests for ctxnav.py — agent-navigable context via memory(ref, expand/contract, budget).

The load-bearing properties: a tombstone view is O(1) (independent of stored content size),
expand/contract honor the budget and go both directions, expansion recurses (drill into a
node's middle child), a full expand is lossless, and the exploration journal replays
deterministically — so an agent's self-exploration path is reproducible and learnable.

Run: python tools/ctxnav_test.py     (exits non-zero on failure)
Also importable as a pytest module (test_* functions).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctxnav as N


def _store():
    st = N.Store()
    big = st.add("Read", "A" * 40000)     # 10000 tok
    small = st.add("Bash", "B" * 4000)    # 1000 tok
    return st, big, small


def test_tombstone_resident_is_O1_independent_of_content_size():
    # The resident view of N tombstones must not grow with the stored bytes — that is the
    # whole O(1) claim. Compare a store of small nodes vs huge nodes: resident is ~equal.
    a = N.Store()
    a.add("Read", "x" * 4000)
    a.add("Read", "y" * 4000)
    b = N.Store()
    b.add("Read", "x" * 4_000_000)
    b.add("Read", "y" * 4_000_000)
    # near-constant: the only dependence on content size is the digit-count of the displayed
    # token total in the marker (logarithmic), never linear in the bytes.
    assert abs(a.resident_tok() - b.resident_tok()) <= 6, (a.resident_tok(), b.resident_tok())
    assert a.resident_tok() < b.full_tok() / 1000   # tombstones are nearly free vs full


def test_expand_respects_budget():
    st, big, _ = _store()
    st.memory(big, "expand", 1000)
    rb = N.itok(st.render_node(big))
    assert 800 <= rb <= 1200, rb


def test_recursion_drills_and_grows_resident():
    st, big, _ = _store()
    base = st.resident_tok()
    st.memory(big, "expand", 1000)
    after_parent = st.resident_tok()
    cid = st.child_of(big)
    assert cid is not None and cid in st.nodes
    st.memory(cid, "expand", 1000)
    after_child = st.resident_tok()
    # drilling into the middle child must INCREASE the resident view (it inlines deeper content)
    assert base < after_parent < after_child, (base, after_parent, after_child)


def test_contract_frees_budget():
    st, big, _ = _store()
    st.memory(big, "expand", 2000)
    expanded = st.resident_tok()
    st.memory(big, "contract")
    contracted = st.resident_tok()
    assert contracted < expanded
    assert N.itok(st.render_node(big)) <= N.TOMBSTONE_TOK + 2


def test_full_expand_is_lossless():
    st, big, _ = _store()
    st.memory(big, "expand", 10_000_000)   # budget >= full -> full resolution
    assert st.render_node(big) == st.nodes[big].content


def test_journal_replay_is_deterministic():
    st, big, _ = _store()
    st.memory(big, "expand", 1000)
    cid = st.child_of(big)
    st.memory(cid, "expand", 1000)
    st.memory(big, "expand", 4000)
    rp = st.replay()
    assert rp.resident_tok() == st.resident_tok(), (rp.resident_tok(), st.resident_tok())
    # the rebuilt store renders the top-level node byte-identically
    assert rp.render_node(big) == st.render_node(big)


def test_selfcheck_passes():
    assert N.runselfcheck() == 0


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
