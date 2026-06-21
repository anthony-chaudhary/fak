#!/usr/bin/env python3
"""Smoke tests for fak_llama_turn_agent_compare.py."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fak_llama_turn_agent_compare as cmp


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def test_tuned_reference_and_projection_fields():
    with tempfile.TemporaryDirectory() as td:
        fak_path = os.path.join(td, "fak.json")
        llama_path = os.path.join(td, "llama.json")
        out_path = os.path.join(td, "compare.json")
        write_json(fak_path, {
            "engine": "fak test",
            "prefix_len": 1024,
            "decode_steps_per_turn": 32,
            "result_tokens_between_turns": 48,
            "points": [{
                "turns": 40,
                "concurrency": 8,
                "reuse_total_ms": 1000,
                "reuse_agent_turns_per_sec": 2.0,
                "reuse_agents_per_sec": 0.05,
                "reuse_speedup_vs_noreuse": 3.5,
            }],
        })
        write_json(llama_path, {
            "engine": "llama tuned test",
            "points": [{
                "turns": 40,
                "agents": 8,
                "total_ms": 500,
                "agent_turns_per_sec": 4.0,
                "agents_per_sec": 0.1,
            }],
        })

        rc = cmp.main([
            "--fak", fak_path,
            "--llama", llama_path,
            "--baseline-label", "tuned-test",
            "--candidate-tuning-mult", "3",
            "--candidate-tuning-label", "projected-3x",
            "--out", out_path,
        ])
        assert rc == 0
        with open(out_path, encoding="utf-8") as f:
            summary = json.load(f)
        policy = summary["baseline_policy"]
        assert policy["headline_reference"] == "tuned-test"
        assert policy["baseline_role"] == "tuned_external_reference"
        assert policy["candidate_tuning_multiplier"] == 3.0
        assert policy["naive_or_noreuse_role"] == "internal_ablation_only"

        row = summary["cells"][0]
        assert row["fak_current_vs_tuned_reference"] == 0.5
        assert row["fak_current_gap_to_tuned_reference"] == 2.0
        assert row["fak_projected_agent_turns_per_sec"] == 6.0
        assert row["fak_projected_vs_tuned_reference"] == 1.5
        assert row["fak_reuse_speedup_vs_noreuse"] == 3.5


if __name__ == "__main__":
    test_tuned_reference_and_projection_fields()
    print("PASS fak_llama_turn_agent_compare_test")
