#!/usr/bin/env python3
"""Tests for glm_witness_record.py — the GPU-witness log -> record parser.

This parser is the seam where a human-readable GPU run log becomes a machine
artifact other tools assert on, so its FAILURE modes matter more than its happy
path (which the module's own ``--self-test`` golden already covers). What is
pinned here is everything that decides whether a run gets to claim PASS:

* a test whose PASS/FAIL line never arrived is recorded FAIL, not "unknown" —
  fail-loud, because a truncated log must never read as a green witness;
* the runner's own summary rc WINS over the per-test aggregation, so a log whose
  tests all printed PASS but whose runner exited non-zero is still FAIL;
* an empty or unparseable log is FAIL rather than a vacuous PASS over zero tests;
* the ``--public`` rollup drops the private node name while keeping the content
  hash identical, so the public record provably came from the same bytes.

Pure stdlib over synthetic in-memory logs: no GPU, no network, no disk.
"""
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glm_witness_record as M  # noqa: E402


HEADER = ("=== [13:53:08] node gpu-node.local gpu=0 arch=sm_80 ===\n"
          "=== [13:53:11] HEAD 7889a5b ===\n")


def test_block(name, cosine="1.000000", verdict="PASS", device_key="cuda",
               desc="a described path"):
    lines = [f"=== [13:53:16] go test -tags cuda -run {name} ({desc}) ===",
             f"=== RUN   {name}",
             f"    x_test.go:63: some detail: cosine={cosine} argmax cpu=40 "
             f"{device_key}=40 tier=sm_80 class=approx"]
    if verdict:
        lines.append(f"--- {verdict}: {name} (0.16s)")
    return "\n".join(lines) + "\n"


def summary(head="7889a5b", rc1=0, rc2=0, rc3=0, rc=0):
    return (f"=== [13:53:27] GLM GPU WITNESS DONE head={head} "
            f"rc1={rc1} rc2={rc2} rc3={rc3} -> rc={rc} ===\n")


def record(log_text, **over):
    kwargs = {"utc": "2026-06-24T13:53:08Z", "machine_id": "dgx",
              "model_name": "glm_moe_dsa", "precision": "q8", "public": False}
    kwargs.update(over)
    return M.build_record(log_text.encode("utf-8"), **kwargs)


class ParseLog(unittest.TestCase):
    def test_node_and_arch_come_from_the_first_node_banner(self):
        text = HEADER + "=== [14:00:00] node other-node gpu=3 arch=sm_90 ===\n"
        parsed = M.parse_log(text)
        self.assertEqual(parsed["node"], "gpu-node.local")
        self.assertEqual(parsed["arch"], "sm_80")

    def test_tests_keep_the_order_the_runner_ran_them_in(self):
        text = HEADER + test_block("TestA") + test_block("TestB") + test_block("TestC")
        names = [t["test_name"] for t in M.parse_log(text)["tests"]]
        self.assertEqual(names, ["TestA", "TestB", "TestC"])

    def test_the_banner_description_is_kept_as_the_exercised_path(self):
        text = HEADER + test_block("TestA", desc="all-device GLM-5.2 DSA forward")
        self.assertEqual(M.parse_log(text)["tests"][0]["path_exercised"],
                         "all-device GLM-5.2 DSA forward")

    def test_the_cosine_argmax_tier_and_class_are_all_captured(self):
        text = HEADER + test_block("TestA", cosine="0.999123")
        t = M.parse_log(text)["tests"][0]
        self.assertEqual(t["cosine"], 0.999123)
        self.assertEqual((t["argmax_cpu"], t["argmax_device"]), (40, 40))
        self.assertEqual(t["tier"], "sm_80")
        self.assertEqual(t["class"], "approx")

    def test_the_hybrid_and_device_argmax_spellings_are_understood(self):
        # The offload test prints `hybrid=`, not `cuda=`; both are the device side.
        for key in ("cuda", "hybrid", "device"):
            text = HEADER + test_block("TestA", device_key=key)
            self.assertEqual(M.parse_log(text)["tests"][0]["argmax_device"], 40, key)

    def test_build_evidence_is_recorded_when_the_runner_printed_it(self):
        text = (HEADER + "[cuda] built 217836 byte libfakcuda.a\n"
                "[cuda] OK build\n" + test_block("TestA"))
        build = M.parse_log(text)["build"]
        self.assertEqual(build["libfakcuda_bytes"], 217836)
        self.assertTrue(build["ok"])

    def test_no_build_lines_leaves_build_unset_rather_than_faking_one(self):
        self.assertIsNone(M.parse_log(HEADER + test_block("TestA"))["build"])

    def test_the_runner_summary_is_parsed_into_its_component_rcs(self):
        parsed = M.parse_log(HEADER + summary(rc1=0, rc2=1, rc3=0, rc=1))
        self.assertEqual(parsed["summary"],
                         {"head": "7889a5b", "rc1": 0, "rc2": 1, "rc3": 0, "rc": 1})

    def test_head_falls_back_to_the_summary_when_the_banner_is_missing(self):
        parsed = M.parse_log(test_block("TestA") + summary(head="deadbee"))
        self.assertEqual(parsed["head_sha"], "deadbee")

    def test_an_empty_log_parses_to_empty_fields_not_an_exception(self):
        parsed = M.parse_log("")
        self.assertEqual(parsed["tests"], [])
        self.assertIsNone(parsed["node"])
        self.assertEqual(parsed["summary"], {})


class FailLoud(unittest.TestCase):
    def test_a_test_with_no_verdict_line_is_recorded_as_failed(self):
        # A truncated log (the runner died mid-test) must never read as green.
        text = HEADER + test_block("TestA", verdict=None)
        t = M.parse_log(text)["tests"][0]
        self.assertEqual(t["verdict"], "FAIL")
        self.assertEqual(t["rc"], 1)

    def test_an_explicit_fail_line_is_carried_through(self):
        text = HEADER + test_block("TestA", verdict="FAIL")
        t = M.parse_log(text)["tests"][0]
        self.assertEqual(t["verdict"], "FAIL")
        self.assertEqual(t["rc"], 1)

    def test_an_empty_log_records_fail_not_a_vacuous_pass(self):
        rec = record("")
        self.assertEqual(rec["rc"], 1)
        self.assertEqual(rec["verdict"], "FAIL")
        self.assertEqual(rec["tests"], [])

    def test_missing_measurements_are_recorded_as_null_not_dropped(self):
        # The schema expects the keys to exist so a consumer can tell "not
        # measured" from "key absent because the parser changed".
        text = HEADER + "=== RUN   TestA\n--- PASS: TestA (0.1s)\n"
        t = M.parse_log(text)["tests"][0]
        for key in ("cosine", "gate", "argmax_cpu", "argmax_device", "set_equality"):
            self.assertIn(key, t)
            self.assertIsNone(t[key])


class OverallVerdict(unittest.TestCase):
    def test_the_runner_summary_rc_outranks_the_per_test_aggregation(self):
        # Every test printed PASS but the runner exited non-zero (a post-test
        # step failed). The record must follow the RUNNER, not the test lines.
        text = HEADER + test_block("TestA") + test_block("TestB") + summary(rc3=1, rc=1)
        rec = record(text)
        self.assertEqual([t["verdict"] for t in rec["tests"]], ["PASS", "PASS"])
        self.assertEqual(rec["rc"], 1)
        self.assertEqual(rec["verdict"], "FAIL")

    def test_without_a_summary_all_tests_passing_is_a_pass(self):
        rec = record(HEADER + test_block("TestA") + test_block("TestB"))
        self.assertEqual(rec["rc"], 0)
        self.assertEqual(rec["verdict"], "PASS")

    def test_without_a_summary_one_failing_test_fails_the_record(self):
        rec = record(HEADER + test_block("TestA") + test_block("TestB", verdict="FAIL"))
        self.assertEqual(rec["rc"], 1)
        self.assertEqual(rec["verdict"], "FAIL")

    def test_a_summary_rc_zero_passes_the_record(self):
        rec = record(HEADER + test_block("TestA") + summary())
        self.assertEqual(rec["verdict"], "PASS")


class PublicRollup(unittest.TestCase):
    def test_public_drops_the_node_and_keeps_the_content_hash(self):
        text = HEADER + test_block("TestA") + summary()
        priv = record(text)
        pub = record(text, machine_id="a100", public=True)
        self.assertEqual(priv["node"], "gpu-node.local")
        self.assertNotIn("node", pub)
        self.assertEqual(pub["content_sha256"], priv["content_sha256"])
        self.assertEqual(pub["machine_id"], "a100")

    def test_a_log_with_no_node_line_simply_has_no_node_field(self):
        rec = record(test_block("TestA") + summary())
        self.assertNotIn("node", rec)


class RecordEnvelope(unittest.TestCase):
    def setUp(self):
        self.text = HEADER + test_block("TestA") + summary()

    def test_the_hash_is_over_the_raw_bytes_and_the_length_is_recorded(self):
        rec = record(self.text)
        self.assertIsNotNone(re.fullmatch(r"[a-f0-9]{64}", rec["content_sha256"]))
        self.assertEqual(rec["log_bytes"], len(self.text.encode("utf-8")))

    def test_one_changed_byte_changes_the_hash(self):
        a = record(self.text)["content_sha256"]
        b = record(self.text.replace("cosine=1.000000", "cosine=0.999999"))["content_sha256"]
        self.assertNotEqual(a, b)

    def test_caller_stamped_fields_are_carried_verbatim(self):
        rec = record(self.text, utc="2026-01-02T03:04:05Z", model_name="glm_moe_dsa",
                     precision="q4")
        self.assertEqual(rec["utc"], "2026-01-02T03:04:05Z")
        self.assertEqual(rec["model"], {"name": "glm_moe_dsa", "precision": "q4"})
        self.assertEqual(rec["schema"], M.SCHEMA)
        self.assertEqual(rec["record_tool"], M.RECORD_TOOL)

    def test_optional_provenance_is_included_only_when_supplied(self):
        bare = record(self.text)
        self.assertNotIn("head_subject", bare)
        self.assertNotIn("environment", bare)
        rich = record(self.text, head_subject="fix(model): ...",
                      environment={"driver": "550.54"})
        self.assertEqual(rich["head_subject"], "fix(model): ...")
        self.assertEqual(rich["environment"], {"driver": "550.54"})

    def test_the_same_log_and_stamps_produce_the_same_record(self):
        self.assertEqual(record(self.text), record(self.text))


class ToInt(unittest.TestCase):
    def test_unparseable_values_become_none_rather_than_raising(self):
        self.assertEqual(M._to_int("42"), 42)
        self.assertIsNone(M._to_int("n/a"))
        self.assertIsNone(M._to_int(None))


class GoldenSelfTest(unittest.TestCase):
    def test_the_shipped_golden_log_still_parses_to_three_passing_tests(self):
        rec = record(M.GOLDEN_LOG)
        self.assertEqual(rec["verdict"], "PASS")
        self.assertEqual(len(rec["tests"]), 3)
        self.assertTrue(all(t["cosine"] == 1.0 for t in rec["tests"]))
        self.assertEqual(rec["head_sha"], "7889a5b")
        self.assertEqual(rec["arch"], "sm_80")


if __name__ == "__main__":
    unittest.main()
