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
    """results: list of (toolu_id, content_str) or (toolu_id, content_str, is_error)."""
    content = []
    for r in results:
        tu, s = r[0], r[1]
        block = {"type": "tool_result", "tool_use_id": tu, "content": s}
        if len(r) > 2 and r[2]:
            block["is_error"] = True
        content.append(block)
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


def test_is_error_flag_captured_from_transcript():
    # a failed Bash call carries is_error:true on its tool_result; parse_window must surface it
    # so the error-collapse tier can key on the structural signal (not a guessy regex).
    recs = [
        _assistant_tooluse("m1", [("t1", "Bash", {"command": "boom"})]),
        _user_results([("t1", "Traceback (most recent call last):\n" + "x" * 4000, True)]),
        _assistant_tooluse("m2", [("t2", "Bash", {"command": "ok"})]),
        _user_results([("t2", "fine")]),
    ]
    items, _ = C.parse_window(_write(recs))
    by_cmd = {it.content[:10]: it for it in items if it.kind == "result"}
    err = [it for it in items if it.kind == "result" and it.is_error]
    assert len(err) == 1 and err[0].tool == "Bash"
    assert C.looks_like_error(err[0]) is True
    ok = [it for it in items if it.kind == "result" and not it.is_error][0]
    assert C.looks_like_error(ok) is False         # "fine" is not error-shaped


def test_profile_off_disables_reduction():
    # the "disabled" choice — even a windowable corpus reduces exactly 1.0x / removes nothing.
    items = C._seq([C._mk("result", "Read", 8000, salt=f"r{i}", path=f"/r{i}") for i in range(6)])
    rep = C.reduce_window(items, 1000, profile=C.make_profile("off"))
    assert rep["ratio"] == 1.0
    assert rep["removed_tok"] == 0
    assert rep["profile"] == "off"


def test_profile_conservative_skips_windowing():
    # lossless + recoverable only: a stale read collapses (re-fetchable) but a big unique
    # non-file result is kept verbatim — no bounded-loss windowing, recoverable_frac == 1.0.
    items = C._seq(
        [C._mk("result", "Read", 16000, salt="s", path="/f", stale=True),
         C._mk("result", "Edit", 200, salt="e", path="/f"),
         C._mk("result", "Bash", 16000, salt="big")])
    rep = C.reduce_window(items, 1000, profile=C.make_profile("conservative"))
    assert rep["tiers"]["window"]["n"] == 0
    assert rep["tiers"]["stale"]["n"] == 1
    assert rep["recoverable_frac"] == 1.0
    assert rep["lossy_removed_tok"] == 0
    assert rep["ratio"] > 1.0


def test_error_collapse_oneline_is_lossy_but_honest():
    # the user's example: a 30k error result collapsed to a one-line marker. Big reduction,
    # but recoverable_frac drops (no file handle) and lossy_removed_tok > 0 — honestly lossy.
    items = C._seq([C._mk("result", "Bash", 30000, salt="Error: boom\n", is_error=True)])
    rep = C.reduce_window(items, 200, profile=C.make_profile("hyper"))
    assert rep["tiers"]["error"]["n"] == 1
    assert rep["lossy_removed_tok"] > 0
    assert rep["ratio"] > 10.0
    assert rep["recoverable_frac"] < 0.5            # not file-recoverable -> honest low frac


def test_error_collapse_head_keeps_message_line():
    err = C._mk("result", "Bash", 30000, salt="Error: connection refused\n", is_error=True)
    head, recov = C.collapse_error(err, "head")
    one, recov2 = C.collapse_error(err, "oneline")
    assert head.startswith("Error: connection refused")
    assert "error body elided" in head and len(head) < len(err.content)
    assert "error" in one.lower() and "elided" in one and len(one) < 64
    assert recov is False and recov2 is False        # no path -> not re-fetchable


def test_error_collapse_off_in_default_profile():
    # the lossy lever is OFF unless asked: a 30k error under `balanced` is WINDOWED (head+tail),
    # never collapsed to a marker. The error tier stays empty; nothing is lossy.
    items = C._seq([C._mk("result", "Bash", 30000, salt="Error: boom\n", is_error=True)])
    rep = C.reduce_window(items, 1000, profile=C.make_profile("balanced"))
    assert rep["tiers"]["error"]["n"] == 0
    assert rep["lossy_removed_tok"] == 0


def test_error_collapse_never_false_positive_or_inflates():
    # never collapse a non-error result, even under hyper + a huge budget...
    clean = C._seq([C._mk("result", "Bash", 8000, salt="ordinary log output here", path="/n")])
    r_clean = C.reduce_window(clean, 100000, profile=C.make_profile("hyper"))
    assert r_clean["tiers"]["error"]["n"] == 0
    # ...and never inflate a tiny error whose marker would cost more than the body.
    tiny = C._seq([C._mk("result", "Bash", 30, salt="Error: x\n", is_error=True)])
    r_tiny = C.reduce_window(tiny, 1000, profile=C.make_profile("hyper"))
    assert r_tiny["reduced_tok"] <= r_tiny["baseline_tok"]


def test_per_tool_exempt_and_tight_cap():
    # exempt: an exempted tool's big payload is left verbatim (nothing removed).
    fe = C._seq([C._mk("result", "Bash", 8000, salt="bigexempt")])
    r_ex = C.reduce_window(fe, 1000, profile=C.Profile("ex", per_tool={"Bash": None}))
    assert r_ex["ratio"] == 1.0 and r_ex["removed_tok"] == 0
    # tight per-tool cap windows that tool harder than the default budget.
    fr = C._seq([C._mk("result", "Read", 8000, salt="rr", path="/x")])
    r_def = C.reduce_window(fr, 1000, profile=C.Profile("d"))
    r_tight = C.reduce_window(fr, 1000, profile=C.Profile("t", per_tool={"Read": 100}))
    assert r_tight["removed_tok"] > r_def["removed_tok"]


def test_resolve_profile_parses_overrides():
    import argparse
    ns = argparse.Namespace(profile="aggressive", no_noise=True, no_stale=False,
                            error_collapse="oneline",
                            per_tool=["Bash=150", "TodoWrite=off", "Read=none"])
    p = C.resolve_profile(ns)
    assert p.name == "aggressive"
    assert p.noise is False                          # --no-noise override applied
    assert p.error_collapse == "oneline"             # override beat the preset's "head"
    assert p.per_tool["Bash"] == 150
    assert p.per_tool["TodoWrite"] is None           # exempt
    assert p.per_tool["Read"] is None                # "none" -> exempt
    assert p.is_exempt("TodoWrite") is True


def test_profiles_are_a_monotonic_dial():
    # the "how aggressive" dial must be monotonic: each profile reduces at least as hard as the
    # previous one on a reducible corpus. Guards a preset edit from inverting the order.
    items = C._seq(
        [C._mk("result", "Read", 8000, salt=f"rd{i}", path=f"/rd{i}") for i in range(8)] +
        [C._mk("result", "Bash", 8000, salt=f"e{i} Error: boom", is_error=True) for i in range(3)] +
        [C._mk("result", "Read", 8000, salt="st", path="/st", stale=True),
         C._mk("result", "Edit", 100, salt="ed", path="/st")] +
        [C._mk("result", "Grep", 8000, salt="dup"), C._mk("result", "Grep", 8000, salt="dup")])

    def ratio(name):
        p = C.make_profile(name)
        if not p.window:
            return C.reduce_window(items, 0, profile=p)["ratio"] or 1.0
        if p.budget is None:
            return C.autotune(items, p.target, profile=p)[0]["ratio"] or 1.0
        return C.reduce_window(items, p.budget, profile=p)["ratio"] or 1.0

    r = [ratio(n) for n in ("off", "conservative", "balanced", "aggressive", "hyper")]
    assert r[0] == 1.0                                    # off is the identity
    for i in range(4):
        assert r[i] <= r[i + 1] + 1e-9, (i, r)            # non-decreasing aggression


def test_make_profile_rejects_unknown():
    try:
        C.make_profile("turbo")
    except SystemExit:
        return
    raise AssertionError("make_profile should reject an unknown profile name")


def test_default_profile_matches_legacy_path():
    # the `balanced` preset must reproduce the pre-profile reducer byte-for-byte.
    items = C._seq([C._mk("result", "Read", 8000, salt=f"r{i}", path=f"/r{i}") for i in range(8)])
    legacy = C.reduce_window(items, 800)
    balanced = C.reduce_window(items, 800, profile=C.make_profile("balanced"))
    assert legacy["ratio"] == balanced["ratio"]
    assert legacy["removed_tok"] == balanced["removed_tok"]
    assert legacy["recoverable_frac"] == balanced["recoverable_frac"]
    assert legacy["tiers"]["window"]["n"] == balanced["tiers"]["window"]["n"]


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
