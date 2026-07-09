#!/usr/bin/env python3
"""Hermetic tests for tools/glm52_prefill_sweep.py (the L9 prefill-sweep driver).

GPU-free and network-free by construction: every test exercises the pure planner
or the pure ledger-land helper. The one network guard test patches urllib +
subprocess to RAISE, then proves --dry-run touches neither — so "produces no
prefill number, only enables the measurement" is enforced, not just documented.

Stdlib unittest (like glm52_serving_witness_test.py) so the gated hermetic
runner runs it directly; no pytest dependency, no quarantine entry."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import glm52_prefill_sweep as sweep  # noqa: E402


class PlanShape(unittest.TestCase):
    def test_plan_covers_all_five_lengths_in_order(self) -> None:
        plan = sweep.build_plan("zai-org/GLM-5.2", "land/root")
        self.assertEqual([s["prompt_len"] for s in plan], [128, 512, 2048, 4096, 8192])
        self.assertEqual(len(plan), 5)

    def test_request_bodies_are_prefill_dominant_and_well_formed(self) -> None:
        plan = sweep.build_plan("my-model", "land/root", max_tokens=1, stream=True)
        for step in plan:
            body = step["request_body"]
            self.assertEqual(body["model"], "my-model")
            self.assertEqual(body["temperature"], 0)
            self.assertLessEqual(body["max_tokens"], 4)  # prefill-dominant
            self.assertTrue(body["stream"])
            self.assertEqual(body["stream_options"], {"include_usage": True})
            msgs = body["messages"]
            self.assertEqual(msgs[0]["role"], "user")
            content = msgs[0]["content"]
            self.assertTrue(content)
            # The synthetic prompt must scale with the target length (prefill-dominant
            # at scale) — a fixed tiny prompt would defeat the whole sweep.
            self.assertEqual(len(content.split()), step["prompt_len"])

    def test_land_paths_are_templated_per_length(self) -> None:
        plan = sweep.build_plan("m", "experiments/benchmark/runs/by-machine/nodeA/STAMP-glm52-prefill-sweep")
        got = {s["prompt_len"]: s["land_subdir"] for s in plan}
        self.assertEqual(
            got[512],
            "experiments/benchmark/runs/by-machine/nodeA/STAMP-glm52-prefill-sweep/p512",
        )
        self.assertEqual(
            got[8192],
            "experiments/benchmark/runs/by-machine/nodeA/STAMP-glm52-prefill-sweep/p8192",
        )

    def test_two_largest_lengths_flagged_fragile_on_sm80(self) -> None:
        plan = sweep.build_plan("m", "root")
        fragile = {s["prompt_len"] for s in plan if s["fragile_on_sm80"]}
        self.assertEqual(fragile, {4096, 8192})

    def test_blocking_mode_omits_stream_options(self) -> None:
        plan = sweep.build_plan("m", "root", stream=False)
        for step in plan:
            self.assertFalse(step["request_body"]["stream"])
            self.assertNotIn("stream_options", step["request_body"])


class DryRunIsPure(unittest.TestCase):
    def test_dry_run_writes_plan_and_touches_no_network_or_gpu(self) -> None:
        # Any HTTP or subprocess (nvidia-smi / go / git shell-out) inside --dry-run
        # is a contract violation: the planner must be pure. Patch both to raise.
        def boom(*_a, **_k):  # noqa: ANN002, ANN003
            raise AssertionError("dry-run must not perform network/GPU/subprocess I/O")

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "plan.json"
            with mock.patch.object(sweep.urllib.request, "urlopen", boom), \
                 mock.patch.object(sweep.subprocess, "run", boom), \
                 redirect_stdout(StringIO()):
                code = sweep.main(["--dry-run", "--out", str(out)])
            self.assertEqual(code, 0)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(report["dry_run"])
            self.assertEqual(report["mode"], "plan")
            self.assertEqual(len(report["plan"]), 5)
            # Honesty fence surfaced in the artifact, not just the docstring.
            self.assertIn("no prefill number is produced", " ".join(report["notes"]).lower())

    def test_no_endpoint_falls_back_to_plan(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "plan.json"
            with redirect_stdout(StringIO()):
                code = sweep.main(["--out", str(out)])  # no --endpoint, no --dry-run
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(out.read_text(encoding="utf-8"))["mode"], "plan")


class LedgerArtifactIsDiscoverable(unittest.TestCase):
    """The landed manifest must satisfy the benchcli.DecodeArtifact contract so
    BuildLineageIndex folds it and `dos verify` can bind it — reproduced here in
    Python and checked structurally (GPU-free)."""

    def _fake_record(self) -> dict:
        measurement = {"ok": True, "http_status": 200, "ttft_s": 0.25,
                       "total_s": 0.26, "prompt_tokens": 512, "completion_tokens": 1,
                       "source": "stream-ttft"}
        return sweep.record_for_length(
            measurement, model="zai-org/GLM-5.2", endpoint="http://n:8000/v1",
            prompt_len=512, max_tokens=1, stream=True,
        )

    def test_manifest_carries_lineage_and_artifact_result_stays_raw(self) -> None:
        lineage = {
            "lineage_schema": sweep.LINEAGE_SCHEMA,
            "app_version": "test",
            "utc": "2026-07-08T00:00:00Z",
            "git_commit": "deadbeefcafe1234",
            "go_version": "python-driver",
            "node": "nodeA",
        }
        with tempfile.TemporaryDirectory() as td:
            land = Path(td) / "p512"
            manifest_path = sweep.write_ledger_artifact(str(land), self._fake_record(), lineage)
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            # DecodeArtifact keys on either a lineage block (git_commit/utc) ...
            self.assertEqual(manifest["lineage"]["git_commit"], "deadbeefcafe1234")
            self.assertTrue(manifest["lineage"]["utc"])
            # ... or a benchmark_artifact with a non-empty run_id.
            self.assertTrue(manifest["benchmark_artifact"]["run_id"])
            # The load-bearing scope survives at the manifest top level.
            self.assertIn("NOT the 753B", manifest["scope"])
            self.assertEqual(manifest["prefill_tok_s"], round(512 / 0.25, 3))
            # result.json is the RAW record with NO lineage/envelope, so
            # BuildLineageIndex does not double-count it.
            result = json.loads((land / "result.json").read_text(encoding="utf-8"))
            self.assertNotIn("lineage", result)
            self.assertNotIn("benchmark_artifact", result)
            self.assertTrue((land / "RESULTS.md").exists())

    def test_failed_length_records_fail_without_number(self) -> None:
        measurement = {"ok": False, "http_status": 500, "error": "CUDA illegal memory access"}
        rec = sweep.record_for_length(
            measurement, model="m", endpoint="http://n/v1",
            prompt_len=8192, max_tokens=1, stream=True,
        )
        self.assertEqual(rec["status"], "FAIL")
        self.assertIsNone(rec["prefill_tok_s"])
        self.assertIn("illegal memory access", rec["error"])


class LandOptOut(unittest.TestCase):
    def test_glm_land_dir_empty_disables_land(self) -> None:
        with mock.patch.dict(sweep.os.environ, {"GLM_LAND_DIR": ""}, clear=False):
            with tempfile.TemporaryDirectory() as td:
                out = Path(td) / "plan.json"
                with redirect_stdout(StringIO()):
                    sweep.main(["--dry-run", "--out", str(out)])
                report = json.loads(out.read_text(encoding="utf-8"))
                self.assertFalse(report["land_enabled"])
                self.assertEqual(report["plan"][0]["land_subdir"], "")


if __name__ == "__main__":
    unittest.main()
