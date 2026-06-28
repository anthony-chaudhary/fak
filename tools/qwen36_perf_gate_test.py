#!/usr/bin/env python3
"""Smoke tests for qwen36_perf_gate.py."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import qwen36_perf_gate as gate


def write_json(path: Path, data) -> str:
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def fak_artifact(prefill, decode_tps=2.0, decode_steps=4):
    return {
        "engine": "fak",
        "model": "test",
        "prefill": [
            {"tokens": tokens, "tok_per_sec": tps}
            for tokens, tps in prefill.items()
        ],
        "decode": {
            "prompt_tokens": 16,
            "decode_steps": decode_steps,
            "tok_per_sec": decode_tps,
        },
    }


def llama_artifact(prefill, decode_tps=1.0, n_gen=1):
    rows = [
        {"n_prompt": tokens, "n_gen": 0, "avg_ts": tps, "backends": "Vulkan"}
        for tokens, tps in prefill.items()
    ]
    rows.append({"n_prompt": 0, "n_gen": n_gen, "avg_ts": decode_tps, "backends": "Vulkan"})
    return rows


def test_gate_passes_when_fak_meets_every_ratio():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        fak = write_json(root / "fak.json", fak_artifact({16: 12.0, 64: 20.0}, 1.2))
        llama = write_json(root / "llama.json", llama_artifact({16: 6.0, 64: 10.0}, 1.0))

        report = gate.build_report([("case", fak, llama)], min_ratio=1.0)

        assert report["passed"] is True
        assert not report["failures"]
        assert len(report["cases"][0]["rows"]) == 3


def test_gate_fails_when_any_ratio_is_below_threshold():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        fak = write_json(root / "fak.json", fak_artifact({16: 4.0, 64: 20.0}, 1.2))
        llama = write_json(root / "llama.json", llama_artifact({16: 6.0, 64: 10.0}, 1.0))

        report = gate.build_report([("case", fak, llama)], min_ratio=1.0)

        assert report["passed"] is False
        assert any("P16" in failure for failure in report["failures"])


def test_default_artifacts_pass_current_committed_gate():
    report = gate.build_report(gate.DEFAULT_CASES, min_ratio=1.0)

    assert report["passed"], report["failures"]
    assert len(report["cases"]) == 2
    metrics = {row["metric"] for case in report["cases"] for row in case["rows"]}
    assert {"prefill_P16", "prefill_P64", "prefill_P256", "prefill_P512", "prefill_P1024", "decode"} <= metrics


def test_metal_arm_records_the_open_decode_gap_at_the_within_2x_bar():
    # #64/#300: the Metal arm asserts fak-Metal decode vs the llama.cpp-Metal bar
    # at the within-2x acceptance (min_ratio 0.5). fak is ~0.16x today, so the gate
    # is fail-closed -- it RECORDS the open gap; it does not fabricate a pass.
    report = gate.build_report(gate.METAL_CASES, min_ratio=gate.METAL_TARGET_RATIO)

    assert report["passed"] is False, "Metal arm must be fail-closed until fak reaches the bar"
    decode = next(
        row for case in report["cases"] for row in case["rows"] if row["metric"] == "decode"
    )
    assert decode["fak_tok_per_sec"] == 1.2
    assert decode["llama_tok_per_sec"] == 7.29
    assert abs(decode["ratio"] - (1.2 / 7.29)) < 1e-9
    assert any("decode" in failure for failure in report["failures"])


def test_metal_arm_passes_once_fak_clears_the_checked_ratio():
    # Proves the comparison math + threshold both work: at a ratio below today's
    # 0.16x the same artifacts PASS, so a future fak-Metal speedup flips the gate green.
    report = gate.build_report(gate.METAL_CASES, min_ratio=0.16)

    assert report["passed"] is True, report["failures"]


def test_metal_bar_provenance_caveat_is_surfaced():
    # The bar is observed-external (#459/#452) -- the gate must carry the caveat
    # through so it is never silently asserted as a fak-controlled witness.
    report = gate.build_report(gate.METAL_CASES, min_ratio=gate.METAL_TARGET_RATIO)
    llama = report["cases"][0]["llama"]
    assert llama.get("provenance") == "observed-external"
    assert "#459" in llama.get("caveat", "")
    assert "#452" in llama.get("caveat", "")
    assert "provenance" in gate.render_markdown(report).lower()


if __name__ == "__main__":
    test_gate_passes_when_fak_meets_every_ratio()
    test_gate_fails_when_any_ratio_is_below_threshold()
    test_metal_arm_records_the_open_decode_gap_at_the_within_2x_bar()
    test_metal_arm_passes_once_fak_clears_the_checked_ratio()
    test_metal_bar_provenance_caveat_is_surfaced()
    test_default_artifacts_pass_current_committed_gate()
    print("PASS qwen36_perf_gate_test")
