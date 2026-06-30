#!/usr/bin/env python3
"""Tests for qwen36_parity_witness_gate.py -- the witness-grading regression gate."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import qwen36_parity_witness_gate as gate


def witness(fak_ids, *, oracle=None, div=None, host="Darwin 24.5.0 arm64 M3Pro-Metal",
            metal_pass=False, prefill=0.0, decode=1.84, commit="deadbeef"):
    o = list(gate.ORACLE_IDS) if oracle is None else oracle
    if div is None:
        div = gate.first_divergence(fak_ids, o)
    return {
        "host": host, "commit": commit, "captured_at": "20260628T120000Z",
        "prompt_ids": [248045, 8678, 198],
        "llamacpp_token_ids": o, "fak_token_ids": fak_ids,
        "first_divergence_index": div,
        "correctness_parity": div == -1,
        "metal_gate_pass": metal_pass,
        "fak_prefill_tok_s": prefill, "fak_decode_tok_s": decode,
        "bar_prefill_tok_s": gate.BAR_PREFILL, "bar_decode_tok_s": gate.BAR_DECODE,
        "overall_verdict": "PASS" if (div == -1 and metal_pass) else "FAIL",
    }


def test_known_drift_is_expected_and_passes():
    # Today's recorded state: the exact token-3 near-tie flip. Not a regression -> green.
    r = gate.grade_witness(witness(list(gate.KNOWN_FAK_IDS)))
    assert r["correctness"]["verdict"] == "KNOWN_DRIFT"
    assert r["correctness"]["regression"] is False
    assert r["passed"] is True


def test_parity_closed_drift_passes_and_celebrates():
    # The GOAL: fak reproduces the oracle exactly -> PARITY, gate stays green and says so.
    r = gate.grade_witness(witness(list(gate.ORACLE_IDS)))
    assert r["correctness"]["verdict"] == "PARITY"
    assert r["correctness"]["first_divergence_index"] == -1
    assert r["passed"] is True
    assert "CLOSED" in r["correctness"]["note"]


def test_regression_earlier_divergence_fails_closed():
    # Diverging at index 1 (earlier than the known 2) is strictly worse -> FAIL.
    r = gate.grade_witness(witness([248068, 999, 8160]))
    assert r["correctness"]["verdict"] == "REGRESSION"
    assert r["correctness"]["regression"] is True
    assert r["passed"] is False
    assert any("REGRESSION" in f for f in r["failures"])


def test_progress_later_divergence_passes_with_review():
    # A 4th compared token that matches further before diverging = partial improvement.
    w = witness([248068, 198, 90700, 111], oracle=[248068, 198, 90700, 222])
    r = gate.grade_witness(w)
    assert r["correctness"]["verdict"] == "PROGRESS"
    assert r["correctness"]["regression"] is False
    assert r["passed"] is True


def test_drift_changed_same_index_different_ids_passes_with_review():
    r = gate.grade_witness(witness([248068, 198, 12345]))
    assert r["correctness"]["verdict"] == "DRIFT_CHANGED"
    assert r["correctness"]["regression"] is False
    assert r["passed"] is True


def test_oracle_drift_fails_closed():
    # The llama.cpp reference itself changed -> a setup/measurement bug, fail closed.
    w = witness([248068, 198, 8160], oracle=[248068, 198, 99999], div=2)
    r = gate.grade_witness(w)
    assert r["correctness"]["verdict"] == "ORACLE_DRIFT"
    assert r["passed"] is False


def test_reported_index_disagreeing_with_ids_is_malformed():
    w = witness(list(gate.KNOWN_FAK_IDS))
    w["first_divergence_index"] = 0  # lie: the ids say 2
    r = gate.grade_witness(w)
    assert r["correctness"]["verdict"] == "MALFORMED"
    assert r["passed"] is False


def test_missing_ids_is_malformed():
    w = witness(list(gate.KNOWN_FAK_IDS))
    del w["fak_token_ids"]
    r = gate.grade_witness(w)
    assert r["correctness"]["verdict"] == "MALFORMED"
    assert r["passed"] is False


def test_scrub_catches_leaked_host_ip_and_ssh():
    for field, val in [
        ("host", "my-macbook.tail9abc.ts.net"),
        ("host", "192.168.1.42"),
        ("host", "user@studio.local"),
    ]:
        w = witness(list(gate.KNOWN_FAK_IDS))
        w[field] = val
        r = gate.grade_witness(w)
        assert r["scrub"]["clean"] is False, val
        assert r["passed"] is False, val
        assert any("leak" in f for f in r["failures"]), val


def test_scrub_allows_the_sanctioned_placeholder_and_scrubbed_uname():
    for host in ("Darwin 24.5.0 arm64 M3Pro-Metal", "node-macos-a.local"):
        r = gate.grade_witness(witness(list(gate.KNOWN_FAK_IDS), host=host))
        assert r["scrub"]["clean"] is True, host
        assert r["passed"] is True, host


def test_speed_reported_not_gated_by_default():
    r = gate.grade_witness(witness(list(gate.KNOWN_FAK_IDS), decode=1.2))
    assert r["speed"]["gated"] is False
    assert abs(r["speed"]["decode_ratio"] - (1.2 / gate.BAR_DECODE)) < 1e-3
    assert r["passed"] is True  # under-bar speed never fails when min_ratio == 0


def test_speed_gates_when_min_ratio_set():
    r = gate.grade_witness(witness(list(gate.KNOWN_FAK_IDS), decode=1.2), min_ratio=0.5)
    assert r["speed"]["gated"] is True
    assert r["passed"] is False
    assert any("decode" in f for f in r["speed"]["failures"])


def test_report_carries_open_m4_issue_links():
    r = gate.grade_witness(witness(list(gate.KNOWN_FAK_IDS)))
    assert r["issues"]["correctness"] == [64, 1458]
    assert r["issues"]["metal_gate"] == [71, 1458]
    assert r["issues"]["speed"] == [64, 1382]
    assert r["issues"]["q6k_fused_mlp"] == [1381]


def test_markdown_names_issue_71_not_stale_116():
    md = gate.render_markdown(gate.grade_witness(witness(list(gate.KNOWN_FAK_IDS))))
    assert "Metal hybrid-prefill gate (#71)" in md
    assert "#116" not in md


def test_no_witness_is_not_a_pass_when_required():
    soft = gate.no_witness_report(require=False)
    hard = gate.no_witness_report(require=True)
    assert soft["passed"] is True and soft["status"] == "NO_WITNESS"
    assert hard["passed"] is False
    assert hard["issues"]["metal_gate"] == [71, 1458]


def test_find_latest_witness_picks_newest_by_timestamp():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "qwen36-mac-parity-gate-20260628T100000Z.json").write_text("{}", encoding="utf-8")
        newest = d / "qwen36-mac-parity-gate-20260628T180000Z.json"
        newest.write_text("{}", encoding="utf-8")
        (d / "unrelated.json").write_text("{}", encoding="utf-8")
        assert gate.find_latest_witness(d) == newest


def test_main_grades_an_explicit_witness_file_and_renders():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "qwen36-mac-parity-gate-20260628T120000Z.json"
        p.write_text(json.dumps(witness(list(gate.KNOWN_FAK_IDS))), encoding="utf-8")
        rc = gate.main(["--witness", str(p)])
        assert rc == 0  # known drift -> green


def test_main_fails_closed_on_a_regression_witness():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "qwen36-mac-parity-gate-20260628T120000Z.json"
        p.write_text(json.dumps(witness([248068, 5, 8160])), encoding="utf-8")
        assert gate.main(["--witness", str(p)]) == 1


if __name__ == "__main__":
    import inspect
    mod = inspect.getmembers(__import__(__name__), inspect.isfunction)
    fns = [f for name, f in mod if name.startswith("test_")]
    for fn in fns:
        fn()
    print(f"PASS qwen36_parity_witness_gate_test ({len(fns)} tests)")
