#!/usr/bin/env python3
"""Tests for glm_throughput_record.py — the glm-throughput/1 record assembler.

The interesting promises of this tool are not "it parses JSON"; they are the
HONESTY rules layered on top of the parse, and each one is a place where a
silent regression would publish a misleading number:

* a run that lost its ``scope`` caveat is REFUSED, so a tok/s figure can never be
  quoted without the "synthetic weights, not the 753B" qualifier that bounds it;
* a ``--public`` rollup drops the private ``node`` id while keeping the content
  hash identical, so the public and private records are provably the same log;
* a modelbench report whose checkpoint is not a GLM-DSA model is REFUSED instead
  of being recorded as one;
* the fold is deterministic — the same log and stamps always produce the same
  bytes — which is what makes the catalog diffable.

Everything here runs on synthetic in-memory logs: no disk, no network, no GPU.
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glm_throughput_record as M  # noqa: E402


def tput_line(**over):
    """One GLMTPUT_JSON log line with sane defaults, overridable per test."""
    run = {
        "schema": M.SCHEMA,
        "backend": "cuda (tier=sm_80 class=approx)",
        "config": {"heads": 16, "hidden": 2048, "layers": 8},
        "decode_ms_tok": 4.0,
        "decode_tok_s": 250.0,
        "model": "glm_moe_dsa",
        "prefill_tok_s": 6400.0,
        "scope": "synthetic-weights;NOT-the-753B",
    }
    run.update(over)
    return "GLMTPUT_JSON " + json.dumps(run, sort_keys=True)


def log_bytes(*lines, node="node-dgx-a", head="f39796e", arch="sm_80"):
    head_lines = [f"=== node {node} gpu=0 arch={arch} nonce=deadbeef ===",
                  f"=== HEAD {head} ==="]
    return ("\n".join(head_lines + list(lines)) + "\n").encode("utf-8")


class ParseLog(unittest.TestCase):
    def test_pulls_every_run_plus_the_run_header_fields(self):
        parsed = M.parse_log(log_bytes(tput_line(), tput_line(decode_tok_s=111.0)).decode())
        self.assertEqual(len(parsed["runs"]), 2)
        self.assertEqual(parsed["head_sha"], "f39796e")
        self.assertEqual(parsed["node"], "node-dgx-a")
        self.assertEqual(parsed["arch"], "sm_80")

    def test_missing_header_is_unknown_not_a_crash(self):
        # A log captured without the `=== HEAD ... ===` preamble must still record;
        # the provenance is reported as unknown rather than being invented.
        parsed = M.parse_log(tput_line() + "\n")
        self.assertEqual(parsed["head_sha"], "unknown")
        self.assertEqual(parsed["node"], "")
        self.assertEqual(parsed["arch"], "")
        self.assertEqual(len(parsed["runs"]), 1)

    def test_ignores_surrounding_noise_lines(self):
        text = "some runner chatter\n" + tput_line() + "\nmore chatter\n"
        self.assertEqual(len(M.parse_log(text)["runs"]), 1)


class ScopeGuard(unittest.TestCase):
    def test_refuses_a_run_with_no_scope(self):
        with self.assertRaises(ValueError):
            M.build_record(log_bytes(tput_line(scope="")), "utc", "dgx", public=False)

    def test_single_shared_scope_is_promoted_to_the_record(self):
        rec = M.build_record(log_bytes(tput_line(), tput_line(decode_tok_s=111.0)),
                             "2026-06-24T21:00:00Z", "dgx", public=False)
        self.assertEqual(rec["scope"], "synthetic-weights;NOT-the-753B")

    def test_disagreeing_scopes_are_not_collapsed_into_one_claim(self):
        rec = M.build_record(log_bytes(tput_line(), tput_line(scope="real-checkpoint")),
                             "2026-06-24T21:00:00Z", "dgx", public=False)
        self.assertEqual(rec["scope"], "mixed;see-runs")


class PublicRollup(unittest.TestCase):
    def test_public_drops_node_private_keeps_it_and_the_hash_is_unchanged(self):
        raw = log_bytes(tput_line())
        priv = M.build_record(raw, "2026-06-24T21:00:00Z", "dgx", public=False)
        pub = M.build_record(raw, "2026-06-24T21:00:00Z", "a100", public=True)
        self.assertEqual(priv["node"], "node-dgx-a")
        self.assertNotIn("node", pub)
        # the scrub must not change WHICH log was recorded
        self.assertEqual(pub["content_sha256"], priv["content_sha256"])
        self.assertEqual(pub["machine_id"], "a100")


class Verdict(unittest.TestCase):
    def test_positive_decode_rates_pass_and_the_best_row_is_surfaced(self):
        rec = M.build_record(log_bytes(tput_line(decode_tok_s=111.0),
                                       tput_line(decode_tok_s=250.0,
                                                 config={"layers": 8})),
                             "2026-06-24T21:00:00Z", "dgx", public=False)
        self.assertEqual(rec["verdict"], "PASS")
        self.assertEqual(rec["rc"], 0)
        self.assertEqual(rec["n_configs"], 2)
        self.assertEqual(rec["best_decode_tok_s"], 250.0)
        self.assertEqual(rec["best_decode_config"], {"layers": 8})

    def test_a_zero_rate_run_fails_the_record(self):
        rec = M.build_record(log_bytes(tput_line(decode_tok_s=0)),
                             "2026-06-24T21:00:00Z", "dgx", public=False)
        self.assertEqual(rec["verdict"], "FAIL")
        self.assertEqual(rec["rc"], 1)

    def test_a_log_with_no_runs_fails_rather_than_reporting_an_empty_pass(self):
        rec = M.build_record(log_bytes(), "2026-06-24T21:00:00Z", "dgx", public=False)
        self.assertEqual(rec["n_configs"], 0)
        self.assertEqual(rec["verdict"], "FAIL")
        self.assertIsNone(rec["best_decode_tok_s"])


class ModelbenchNormalisation(unittest.TestCase):
    def report(self, **over):
        rep = {
            "backend": {"class": "approx", "selected": "cuda", "tier": "sm_80"},
            "decode": {"decode_steps": 64, "per_token_median_ms": 12.5,
                       "prompt_tokens": 512, "tok_per_sec": 80.0},
            "model": "glm52-tiny (glm_moe_dsa) [lean]",
            "model_config": {"architectures": ["GlmMoeDsaForCausalLM"],
                             "model_type": "glm_moe_dsa"},
            "precision": "Q8_0",
            "prefill": [{"median_ms": 20.0, "tokens": 128, "tok_per_sec": 6400.0},
                        {"median_ms": 100.0, "tokens": 512, "tok_per_sec": 5120.0}],
            "source": "/models/glm52/model.safetensors",
        }
        rep.update(over)
        return rep

    def test_refuses_a_checkpoint_that_is_not_glm_dsa(self):
        bad = self.report(model_config={"architectures": ["LlamaForCausalLM"],
                                        "model_type": "llama"})
        with self.assertRaises(ValueError):
            M.normalize_modelbench_report(bad)

    def test_normalised_row_carries_the_real_checkpoint_scope(self):
        row = M.normalize_modelbench_report(self.report())
        self.assertEqual(row["schema"], M.SCHEMA)
        self.assertEqual(row["harness"], "cmd/modelbench")
        self.assertEqual(row["scope"], "real-checkpoint;arch-blind-modelbench;not-synthetic")
        self.assertEqual(row["decode_tok_s"], 80.0)
        self.assertEqual(row["decode_ms_tok"], 12.5)

    def test_prefill_headline_is_the_longest_prompt_not_the_first_row(self):
        # The 128-token row is listed first and has the FLATTERING tok/s; the
        # headline must still come from the 512-token row.
        row = M.normalize_modelbench_report(self.report())
        self.assertEqual(row["prefill_ms"], 100.0)
        self.assertEqual(row["prefill_tok_s"], 5120.0)

    def test_backend_label_folds_the_dict_into_one_string(self):
        row = M.normalize_modelbench_report(self.report())
        self.assertEqual(row["backend"], "cuda (tier=sm_80 class=approx)")


class Determinism(unittest.TestCase):
    def test_same_log_and_stamps_fold_to_identical_bytes(self):
        raw = log_bytes(tput_line(), tput_line(decode_tok_s=111.0))
        a = M.build_record(raw, "2026-06-24T21:00:00Z", "dgx", public=False)
        b = M.build_record(raw, "2026-06-24T21:00:00Z", "dgx", public=False)
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
        self.assertEqual(a["log_bytes"], len(raw))
        self.assertEqual(a["record_tool"], M.RECORD_TOOL)


class NumberCoercion(unittest.TestCase):
    def test_bools_are_not_treated_as_numbers(self):
        # `True` is an int in Python; a report field that is a bool must not be
        # silently recorded as the rate 1.0.
        self.assertIsNone(M._number(True))
        self.assertIsNone(M._number("not-a-number"))
        self.assertEqual(M._number("12.5"), 12.5)
        self.assertEqual(M._number(3), 3.0)


if __name__ == "__main__":
    unittest.main()
