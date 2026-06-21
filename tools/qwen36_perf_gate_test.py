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


if __name__ == "__main__":
    test_gate_passes_when_fak_meets_every_ratio()
    test_gate_fails_when_any_ratio_is_below_threshold()
    test_default_artifacts_pass_current_committed_gate()
    print("PASS qwen36_perf_gate_test")
