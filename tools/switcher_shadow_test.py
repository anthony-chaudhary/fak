#!/usr/bin/env python3
"""Tests for switcher_shadow -- the shadow-mode account-switcher savings observer.

The load-bearing property is ANTI-OVERCLAIM: the tool must be structurally unable to
fabricate a saving from real work. These tests pin that (any hard/dev prompt or any
write-tool => $0 claimed), the honest column separation, ledger idempotency, and the
fail-open Stop-hook contract.

Run:  python tools/switcher_shadow_test.py     (or: pytest tools/switcher_shadow_test.py)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import switcher_shadow as ss
import fleet_accounts


POL = fleet_accounts.load_policy()


# --------------------------------------------------------------------------- #
# classify_session -- the whole-session trivial gate
# --------------------------------------------------------------------------- #
def test_hard_prompt_never_qualifies():
    s = {"prompts": [(None, "implement the gateway adjudicator and ship it")],
         "tools": {}, "tokens": {"input": 100, "output": 100}}
    d = ss.classify_session(s, tier2_available=True, policy=POL)
    assert d["would_tier"] == 1
    assert d["qualifies_tier2"] is False


def test_any_hard_prompt_in_a_session_disqualifies_the_whole_session():
    # A session with a trivial turn AND a dev turn must NOT qualify -- gating on ANY light
    # prompt would claim engineering work could have run on GLM.
    s = {"prompts": [(None, "hi"), (None, "now refactor the kvmmu backend")],
         "tools": {}, "tokens": {"input": 100, "output": 100}}
    d = ss.classify_session(s, tier2_available=True, policy=POL)
    assert d["qualifies_tier2"] is False
    assert "non-trivial" in d["reason"]


def test_write_tool_disqualifies_even_with_trivial_prompts():
    s = {"prompts": [(None, "hi")], "tools": {"Write": 1, "Read": 3},
         "tokens": {"input": 100, "output": 100}}
    d = ss.classify_session(s, tier2_available=True, policy=POL)
    assert d["qualifies_tier2"] is False
    assert "side-effecting" in d["reason"]


def test_read_only_tools_do_not_disqualify():
    s = {"prompts": [(None, "hi")], "tools": {"Read": 5, "Grep": 2},
         "tokens": {"input": 100, "output": 100}}
    d = ss.classify_session(s, tier2_available=True, policy=POL)
    assert d["qualifies_tier2"] is True


def test_no_tier2_seat_disqualifies():
    s = {"prompts": [(None, "hi")], "tools": {}, "tokens": {"input": 100, "output": 100}}
    d = ss.classify_session(s, tier2_available=False, policy=POL)
    assert d["qualifies_tier2"] is False
    assert "up-shift" in d["reason"]


def test_genuinely_trivial_session_qualifies():
    s = {"prompts": [(None, "hi")], "tools": {}, "tokens": {"input": 100, "output": 100}}
    d = ss.classify_session(s, tier2_available=True, policy=POL)
    assert d["would_tier"] == 2
    assert d["qualifies_tier2"] is True


def test_no_prompts_is_tier1():
    s = {"prompts": [], "tools": {}, "tokens": {"input": 100, "output": 100}}
    d = ss.classify_session(s, tier2_available=True, policy=POL)
    assert d["would_tier"] == 1
    assert d["qualifies_tier2"] is False


# --------------------------------------------------------------------------- #
# savings math -- honest unit, never fabricated
# --------------------------------------------------------------------------- #
def test_tier_downgrade_zero_for_non_qualifying():
    s = {"prompts": [(None, "implement X")], "tools": {},
         "tokens": {"input": 10_000, "output": 5_000}}
    d = ss.classify_session(s, tier2_available=True, policy=POL)
    td = ss.tier_downgrade_savings(s, d)
    assert td["qualifies"] is False
    assert td["movable_total_tokens"] == 0
    assert td["hypothetical_usd_opus_upper_bound"] == 0.0


def test_tier_downgrade_zeroes_cache_columns():
    # The dollar upper bound prices only FRESH input + output, never the cache tokens
    # (the GLM counterfactual has no comparable cache regime).
    s = {"prompts": [(None, "hi")], "tools": {},
         "tokens": {"input": 1_000_000, "output": 0, "cache_read": 9_999_999,
                    "cache_create": 9_999_999}}
    d = ss.classify_session(s, tier2_available=True, policy=POL)
    td = ss.tier_downgrade_savings(s, d)
    assert td["qualifies"] is True
    # only the 1e6 fresh-input tokens at opus input rate; cache ignored.
    expected = 1_000_000 * ss._OPUS_INPUT / 1e6
    assert abs(td["hypothetical_usd_opus_upper_bound"] - round(expected, 4)) < 1e-6
    assert td["movable_total_tokens"] == 1_000_000  # input + output(0)


def test_cache_reuse_is_realized_and_nonneg():
    assert ss.cache_reuse_savings({"cache_read": 0, "input": 50})["realized_usd_opus"] == 0.0
    big = ss.cache_reuse_savings({"cache_read": 1_000_000})
    assert big["realized_usd_opus"] > 0.0
    # realized = cache_read * (full_input - cache_read_rate)
    expected = 1_000_000 * (ss._OPUS_INPUT - ss._OPUS_CREAD) / 1e6
    assert abs(big["realized_usd_opus"] - round(expected, 4)) < 1e-6


def test_caveat_present_on_every_surface():
    assert ss.CAVEAT.strip()
    roll = ss.rollup([{"cache_reuse": {"realized_usd_opus": 1.0},
                       "tier_downgrade": {"qualifies": False, "movable_total_tokens": 0,
                                          "hypothetical_usd_opus_upper_bound": 0.0},
                       "actual_cost_usd": 2.0}])
    assert roll["caveat"] == ss.CAVEAT
    assert ss.CAVEAT in ss.report_md(roll)


def test_columns_never_summed():
    # rollup keeps the two axes as distinct keys; there is no fused total.
    roll = ss.rollup([])
    keys = set(roll.keys())
    assert "cache_reuse_realized_usd_opus" in keys
    assert "tier_downgrade_hypothetical_usd_opus_upper_bound" in keys
    assert not any("total_savings" in k or "combined" in k for k in keys)


# --------------------------------------------------------------------------- #
# ledger -- idempotent, atomic, bounded
# --------------------------------------------------------------------------- #
def _row(session, fp, **kw):
    base = {"session": session, "fingerprint": fp,
            "cache_reuse": {"realized_usd_opus": 1.0},
            "tier_downgrade": {"qualifies": False, "movable_total_tokens": 0,
                               "hypothetical_usd_opus_upper_bound": 0.0},
            "actual_cost_usd": 1.0}
    base.update(kw)
    return base


def test_ledger_insert_then_idempotent_skip():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ledger.jsonl")
        r = _row("sess-1", "3:10")
        assert ss.upsert_ledger(r, path)["action"] == "inserted"
        assert ss.upsert_ledger(r, path)["action"] == "skipped"  # same content -> no dup
        assert len(ss._read_ledger(path)) == 1


def test_ledger_update_on_new_content_same_session():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ledger.jsonl")
        ss.upsert_ledger(_row("sess-1", "3:10"), path)
        # same session, NEW fingerprint (more turns) -> a different key -> a second row,
        # which is correct (a resumed session that grew is a new content state).
        ss.upsert_ledger(_row("sess-1", "5:20"), path)
        rows = ss._read_ledger(path)
        assert len(rows) == 2
        # same session + same fingerprint but changed payload -> in-place update, no dup.
        res = ss.upsert_ledger(_row("sess-1", "5:20", actual_cost_usd=9.0), path)
        assert res["action"] == "updated"
        assert len(ss._read_ledger(path)) == 2


def test_ledger_rotation_when_over_cap(monkeypatch=None):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ledger.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write("x" * 2048)
        old_cap = ss.LEDGER_CAP_BYTES
        try:
            ss.LEDGER_CAP_BYTES = 1024
            ss._rotate_if_needed(path, cap=1024)
            assert os.path.exists(path + ".1")  # prior generation kept, bounded
        finally:
            ss.LEDGER_CAP_BYTES = old_cap


# --------------------------------------------------------------------------- #
# Stop-hook -- fail-open, non-blocking, never emits a decision
# --------------------------------------------------------------------------- #
def test_hook_failopen_on_garbage_stdin(capsys=None):
    assert ss.run_hook("not json at all {{{") == 0


def test_hook_failopen_on_missing_transcript():
    payload = json.dumps({"session_id": "x", "transcript_path": "/no/such/file.jsonl",
                          "stop_hook_active": False})
    assert ss.run_hook(payload) == 0


def test_hook_shortcircuits_on_stop_hook_active():
    payload = json.dumps({"transcript_path": "/no/such/file.jsonl", "stop_hook_active": True})
    assert ss.run_hook(payload) == 0


def test_hook_never_prints_a_decision_object(capsys=None):
    # A Stop hook that prints {"decision":"block"} would block the session ending.
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ss.run_hook(json.dumps({"transcript_path": "/no/such.jsonl"}))
    out = buf.getvalue()
    assert "decision" not in out
    assert "block" not in out


def test_hook_appends_one_row_for_a_real_transcript():
    # Build a minimal trivial transcript and verify the hook upserts exactly one ledger row.
    with tempfile.TemporaryDirectory() as d:
        tpath = os.path.join(d, "sess.jsonl")
        lines = [
            {"type": "user", "timestamp": "2026-06-24T00:00:00Z",
             "message": {"content": "hi"}},
            {"type": "assistant", "timestamp": "2026-06-24T00:00:01Z",
             "message": {"id": "m1", "model": "claude-haiku-4-5",
                         "usage": {"input_tokens": 50, "output_tokens": 10,
                                   "cache_read_input_tokens": 0,
                                   "cache_creation_input_tokens": 0},
                         "content": [{"type": "text", "text": "hello"}]}},
        ]
        with open(tpath, "w", encoding="utf-8") as f:
            for ln in lines:
                f.write(json.dumps(ln) + "\n")
        ledger = os.path.join(d, "ledger.jsonl")
        old = ss.LEDGER_PATH
        try:
            ss.LEDGER_PATH = ledger
            payload = json.dumps({"session_id": "sess", "transcript_path": tpath,
                                  "stop_hook_active": False})
            assert ss.run_hook(payload) == 0
            assert ss.run_hook(payload) == 0  # re-fire -> idempotent
            rows = ss._read_ledger(ledger)
            assert len(rows) == 1
            assert rows[0]["session"] == "sess"
        finally:
            ss.LEDGER_PATH = old


# --------------------------------------------------------------------------- #
# the bundled anti-overclaim selfcheck must pass
# --------------------------------------------------------------------------- #
def test_selfcheck_passes():
    assert ss.runselfcheck() == 0


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERR   {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
