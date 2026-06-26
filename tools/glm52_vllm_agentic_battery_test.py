"""Hermetic tests for tools/glm52_vllm_agentic_battery.py."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import glm52_vllm_agentic_battery as bat  # noqa: E402


def write_floor_artifacts(out: Path) -> None:
    (out / "swebench-20-fak-floor.json").write_text(json.dumps({
        "schema": "fak-swebench-compare/1",
        "dataset": {"instances": 20},
        "workers": [1, 2, 4, 8],
        "families": [{}, {}, {}, {}],
    }), encoding="utf-8")
    (out / "turntax-airline.json").write_text(json.dumps({
        "provenance": {"slice_id": "turntax-airline"},
        "consistency_check": "ok",
        "net": {"turns_saved": 9},
        "safety_floor": {
            "injections_admitted_fak": 0,
            "destructive_executed_fak": 0,
        },
    }), encoding="utf-8")
    (out / "sessionbench-synthetic.json").write_text(json.dumps({
        "engine": "fak sessionbench (multi-agent session value stack, Q8=true)",
        "model": "smollm2-135m [synthetic]",
        "prefix": 2048,
        "decode_per_turn": 32,
        "result_per_turn": 64,
        "cells": [{"turns": 50, "agents": 5, "prefix": 2048, "decode": 32, "result": 64}],
    }), encoding="utf-8")
    (out / "fanbench-research.json").write_text(json.dumps({
        "profile": {"name": "research-goal"},
        "trials": 12,
        "agent_grid": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        "prefix_grid": [2048],
        "cells": [{"agents": 1024}],
    }), encoding="utf-8")
    (out / "radixbench-synthetic.json").write_text(json.dumps({
        "engine": "fak radixbench — RadixAttention-style prefix cache",
        "model": "synthetic-llama (64h/4L/8q-2kv, vocab 256)",
        "policy_eviction": {"demonstrated": True},
        "workloads": [{"name": "agents", "cache_hit_rate": 0.866}],
    }), encoding="utf-8")


def write_complete_artifacts(root: Path, *, raw_resolved: int = 7, gateway_resolved: int = 8) -> None:
    out = root / "out"
    serving = root / "serving"
    swe = root / "swe"
    out.mkdir()
    serving.mkdir()
    swe.mkdir()
    instance_ids = [f"swebench__verified-{i:02d}" for i in range(20)]
    (out / "glm52-vllm-preflight.json").write_text(json.dumps({
        "model": "zai-org/GLM-5.2",
        "summary": {"node_verdict": "READY"},
        "engines": [{"engine": "vllm", "ready": True, "quant": "fp8"}],
    }), encoding="utf-8")
    (serving / "full-size-serving-witness.json").write_text(json.dumps({
        "model": "glm-5.2",
        "base_url": "http://127.0.0.1:8000/v1",
        "context_length": 131072,
        "engine_cache": {"engine": "vllm"},
        "summary": {"full_size_serving_witness": "PASS"},
    }), encoding="utf-8")
    (out / "adjudication-tax-witness.json").write_text(json.dumps({
        "model": "glm-5.2",
        "base_url": "http://127.0.0.1:8000/v1",
        "engine_cache_engine": "vllm",
        "summary": {
            "vllm_adjudication_tax_witness": "PASS",
            "latency_tax": 1.12,
            "decode_tps_tax": 1.08,
        },
        "raw_vllm": {"ok_samples": 8},
        "gateway": {"ok_samples": 8},
    }), encoding="utf-8")
    (swe / "COMPARE-PREFLIGHT.json").write_text(json.dumps({
        "schema": "fak.dgx-swebench-compare-preflight.v1",
        "ok": True,
        "model": "zai-org/GLM-5.2-FP8",
        "served_as": "glm-5.2",
        "engine": "vllm",
        "selection": "slice 0:20",
        "arms": ["fak-gateway", "raw-vllm"],
        "config": {
            "raw_base_url": "http://127.0.0.1:8000/v1",
            "gateway_base_url": "http://127.0.0.1:8080/v1",
            "skip_engine_serve": True,
            "skip_gateway_serve": False,
            "require_gpu_name": "H200",
            "tp": 8,
            "context_length": 131072,
        },
        "runtime": {
            "mini_swe_agent": {"ok": True, "detail": "mini-extra 1.0"},
            "swebench_python": {"ok": True, "detail": "3.11.9"},
            "swebench_version": {"ok": True, "detail": "4.0.0"},
            "vllm_version": {"ok": True, "detail": "0.10.0"},
        },
        "checks": [
            {"name": "mini-swe-agent", "ok": True, "detail": "/root/venvs/mini-swe-agent/bin/mini-extra"},
            {"name": "swebench-harness-import", "ok": True, "detail": "ok"},
            {"name": "raw-vllm-endpoint", "ok": True, "detail": "http://127.0.0.1:8000/v1/models"},
            {"name": "fak-bin", "ok": True, "detail": "/srv/fleet/tools/.bin/fak"},
        ],
    }), encoding="utf-8")
    (swe / "compare.json").write_text(json.dumps({
        "model": "zai-org/GLM-5.2-FP8",
        "served_as": "glm-5.2",
        "engine": "vllm",
        "dataset": "princeton-nlp/SWE-bench_Verified",
        "selection": "slice 0:20",
        "selection_instance_ids": instance_ids,
        "selection_instance_ids_match": True,
        "tool_call_probes": [
            {"arm": "raw-vllm", "endpoint": "http://127.0.0.1:8000/v1",
             "ok": True, "status": "PASS", "tool_calls": 1},
            {"arm": "fak-gateway", "endpoint": "http://127.0.0.1:8080/v1",
             "ok": True, "status": "PASS", "tool_calls": 1},
        ],
        "arms": [
            {"agent": {"arm": "raw-vllm", "instances": 20, "completed": 20,
                       "instance_ids": instance_ids,
                       "endpoint": "http://127.0.0.1:8000/v1"},
             "grade": {"available": True, "grade_rc": 0, "run_id": "swecmp-raw-vllm",
                       "submitted": 20, "report_path": "preds_raw-vllm/report.json",
                       "grade_log": "grade_raw-vllm.log",
                       "resolved_ids": instance_ids[:raw_resolved],
                       "total": 20, "resolved": raw_resolved,
                       "resolve_pct": round(100 * raw_resolved / 20, 1)}},
            {"agent": {"arm": "fak-gateway", "instances": 20, "completed": 20,
                       "instance_ids": list(instance_ids),
                       "endpoint": "http://127.0.0.1:8080/v1"},
             "grade": {"available": True, "grade_rc": 0, "run_id": "swecmp-fak-gateway",
                       "submitted": 20, "report_path": "preds_fak-gateway/report.json",
                       "grade_log": "grade_fak-gateway.log",
                       "resolved_ids": instance_ids[:gateway_resolved],
                       "total": 20, "resolved": gateway_resolved,
                       "resolve_pct": round(100 * gateway_resolved / 20, 1)}},
        ],
    }), encoding="utf-8")
    (swe / "DONE.rc").write_text("0\n", encoding="utf-8")
    write_floor_artifacts(out)


def base_args(tmp: Path) -> list[str]:
    return [
        "--out-dir", str(tmp / "out"),
        "--serving-out-dir", str(tmp / "serving"),
        "--swe-run-dir", str(tmp / "swe"),
        "--python", "python",
        "--go", "go",
        "--swebench-difficulty", "",
        "--swebench-dataset", "",
    ]


class ManifestTest(unittest.TestCase):
    def test_default_artifact_paths_match_runbook_contract(self) -> None:
        parser = bat.argparse.ArgumentParser()
        bat.add_common_args(parser)
        report = bat.build_report(parser.parse_args([]))
        steps = {row["id"]: row for row in report["steps"]}

        self.assertEqual(
            steps["preflight"]["primary_artifact"],
            "experiments/vllm/glm52-vllm-preflight.json",
        )
        self.assertEqual(
            steps["serving_witness"]["primary_artifact"],
            "experiments/glm52/full-size-serving-witness.json",
        )
        self.assertEqual(
            steps["vllm_tax"]["primary_artifact"],
            "experiments/vllm/adjudication-tax-witness.json",
        )
        self.assertEqual(
            steps["swebench_floor_20"]["primary_artifact"],
            "experiments/vllm/swebench-20-fak-floor.json",
        )
        self.assertEqual(
            steps["swebench_compare_preflight"]["primary_artifact"],
            "/tmp/swe-glm52-vllm-20/COMPARE-PREFLIGHT.json",
        )
        self.assertIn(
            "/tmp/swe-glm52-vllm-20/COMPARE-PREFLIGHT.json",
            steps["swebench_verified_20"]["artifacts"],
        )

    def test_dry_manifest_contains_20_task_vllm_compare_command(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ns = bat.argparse.ArgumentParser()
            bat.add_common_args(ns)
            parsed = ns.parse_args(base_args(Path(td)))
            report = bat.build_report(parsed)

        steps = {row["id"]: row for row in report["steps"]}
        preflight_cmd = steps["swebench_compare_preflight"]["command"]["argv"]
        self.assertIn("--preflight-only", preflight_cmd)
        self.assertIn("--skip-engine-serve", preflight_cmd)
        self.assertIn("--require-gpu-name", preflight_cmd)
        cmd = steps["swebench_verified_20"]["command"]["argv"]
        self.assertIn("tools/dgx_swebench_compare.py", cmd)
        self.assertIn("--engine", cmd)
        self.assertIn("vllm", cmd)
        self.assertIn("--verified-count", cmd)
        self.assertIn("20", cmd)
        self.assertIn("zai-org/GLM-5.2-FP8", cmd)
        self.assertIn("--skip-engine-serve", cmd)
        self.assertIn("--require-tool-calls", cmd)
        self.assertIn("--require-grade", cmd)
        self.assertNotIn("--preflight-only", cmd)
        self.assertEqual(report["summary"]["status"], "PENDING_MEASUREMENT")
        self.assertIn("swebench_compare_preflight", report["summary"]["required_missing"])
        self.assertIn("swebench_verified_20", report["summary"]["required_missing"])

    def test_run_contract_is_written_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            contract = root / "out" / "glm52-agentic-battery" / "raw-vllm-vs-fak-gateway-contract.json"
            rc = bat.main(base_args(root) + [
                "--run-contract", str(contract),
                "--contract-only",
            ])

            self.assertEqual(rc, 0)
            doc = json.loads(contract.read_text(encoding="utf-8"))
            parser = bat.argparse.ArgumentParser()
            bat.add_common_args(parser)
            parsed = parser.parse_args(base_args(root) + ["--run-contract", str(contract)])
            report = bat.build_report(parsed)
            steps = {row["id"]: row for row in report["steps"]}
            final_cmd = bat.final_gate_command(report)

        self.assertEqual(doc["schema"], bat.RUN_CONTRACT_SCHEMA)
        self.assertEqual(doc["status"], "PENDING_MEASUREMENT")
        self.assertFalse(doc["result_claim_allowed"])
        self.assertEqual(
            {row["id"] for row in doc["arms"]},
            {"raw-vllm", "fak-gateway"},
        )
        self.assertEqual(doc["shared_config"]["checkpoint_model"], "zai-org/GLM-5.2-FP8")
        self.assertEqual(doc["shared_config"]["selection"], "slice 0:20")
        self.assertEqual(doc["shared_config"]["budget"]["implicit_retries"], 0)
        self.assertIn("tools/dgx_swebench_compare.py", doc["commands"]["compare_run"])
        self.assertEqual(steps["raw_fak_run_contract"]["artifact_status"], "PASS")
        self.assertIn("--run-contract", final_cmd)
        self.assertIn(str(contract).replace("\\", "/"), final_cmd)
        self.assertIn(
            "result_claim_allowed=false",
            steps["raw_fak_run_contract"]["claim_boundary"],
        )

    def test_main_writes_contract_before_manifest_check(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "manifest.json"
            contract = root / "contract.json"
            rc = bat.main(base_args(root) + [
                "--out", str(manifest),
                "--run-contract", str(contract),
                "--allow-pending",
            ])

            self.assertEqual(rc, 0)
            report = json.loads(manifest.read_text(encoding="utf-8"))
            steps = {row["id"]: row for row in report["steps"]}

        self.assertEqual(report["run_contract_artifact"], str(contract).replace("\\", "/"))
        self.assertEqual(steps["raw_fak_run_contract"]["artifact_status"], "PASS")

    def test_artifact_checker_marks_complete_for_minimal_passing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_complete_artifacts(root)

            parser = bat.argparse.ArgumentParser()
            bat.add_common_args(parser)
            parsed = parser.parse_args(base_args(root))
            report = bat.build_report(parsed)

        self.assertTrue(report["summary"]["complete"])
        self.assertEqual(report["summary"]["required_missing"], [])
        self.assertEqual(report["summary"]["required_failed"], [])

    def test_stale_swebench_artifact_keeps_battery_pending(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out = root / "out"
            swe = root / "swe"
            out.mkdir()
            swe.mkdir()
            (swe / "compare.json").write_text(json.dumps({
                "arms": [
                    {"agent": {"arm": "raw-vllm", "instances": 3},
                     "grade": {"available": True, "total": 3}},
                    {"agent": {"arm": "fak-gateway", "instances": 3},
                     "grade": {"available": True, "total": 3}},
                ]
            }), encoding="utf-8")
            parser = bat.argparse.ArgumentParser()
            bat.add_common_args(parser)
            parsed = parser.parse_args(base_args(root))
            report = bat.build_report(parsed)

        by_id = {row["id"]: row for row in report["steps"]}
        self.assertFalse(report["summary"]["complete"])
        self.assertEqual(by_id["swebench_verified_20"]["artifact_status"], "FAIL")
        self.assertIn("instances=3", by_id["swebench_verified_20"]["artifact_detail"])

    def test_nonzero_swebench_grade_rc_keeps_battery_pending(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_complete_artifacts(root)
            compare_path = root / "swe" / "compare.json"
            doc = json.loads(compare_path.read_text(encoding="utf-8"))
            doc["arms"][0]["grade"]["grade_rc"] = 1
            compare_path.write_text(json.dumps(doc), encoding="utf-8")

            parser = bat.argparse.ArgumentParser()
            bat.add_common_args(parser)
            parsed = parser.parse_args(base_args(root))
            report = bat.build_report(parsed)

        by_id = {row["id"]: row for row in report["steps"]}
        self.assertFalse(report["summary"]["complete"])
        self.assertEqual(by_id["swebench_verified_20"]["artifact_status"], "FAIL")
        self.assertIn("raw-vllm grade.grade_rc=1", by_id["swebench_verified_20"]["artifact_detail"])

    def test_mismatched_swebench_instance_ids_keep_battery_pending(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_complete_artifacts(root)
            compare_path = root / "swe" / "compare.json"
            doc = json.loads(compare_path.read_text(encoding="utf-8"))
            doc["selection_instance_ids_match"] = False
            doc["arms"][1]["agent"]["instance_ids"][-1] = "different__task-999"
            compare_path.write_text(json.dumps(doc), encoding="utf-8")

            parser = bat.argparse.ArgumentParser()
            bat.add_common_args(parser)
            parsed = parser.parse_args(base_args(root))
            report = bat.build_report(parsed)

        by_id = {row["id"]: row for row in report["steps"]}
        self.assertFalse(report["summary"]["complete"])
        self.assertEqual(by_id["swebench_verified_20"]["artifact_status"], "FAIL")
        self.assertIn("selection_instance_ids_match=False",
                      by_id["swebench_verified_20"]["artifact_detail"])
        self.assertIn("raw-vllm and fak-gateway instance_ids differ",
                      by_id["swebench_verified_20"]["artifact_detail"])

    def test_stale_grade_resolved_ids_keep_battery_pending(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_complete_artifacts(root)
            compare_path = root / "swe" / "compare.json"
            doc = json.loads(compare_path.read_text(encoding="utf-8"))
            resolved_ids = doc["arms"][0]["grade"]["resolved_ids"]
            resolved_ids[0] = "stale__task-999"
            compare_path.write_text(json.dumps(doc), encoding="utf-8")

            parser = bat.argparse.ArgumentParser()
            bat.add_common_args(parser)
            parsed = parser.parse_args(base_args(root))
            report = bat.build_report(parsed)

        by_id = {row["id"]: row for row in report["steps"]}
        self.assertFalse(report["summary"]["complete"])
        self.assertEqual(by_id["swebench_verified_20"]["artifact_status"], "FAIL")
        self.assertIn("grade.resolved_ids outside selection",
                      by_id["swebench_verified_20"]["artifact_detail"])

    def test_failed_swebench_compare_preflight_keeps_battery_pending(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_complete_artifacts(root)
            preflight_path = root / "swe" / "COMPARE-PREFLIGHT.json"
            doc = json.loads(preflight_path.read_text(encoding="utf-8"))
            doc["checks"][0]["ok"] = False
            doc["checks"][0]["name"] = "mini-swe-agent"
            preflight_path.write_text(json.dumps(doc), encoding="utf-8")

            parser = bat.argparse.ArgumentParser()
            bat.add_common_args(parser)
            parsed = parser.parse_args(base_args(root))
            report = bat.build_report(parsed)

        by_id = {row["id"]: row for row in report["steps"]}
        self.assertFalse(report["summary"]["complete"])
        self.assertEqual(by_id["swebench_compare_preflight"]["artifact_status"], "FAIL")
        self.assertIn("failed checks: mini-swe-agent",
                      by_id["swebench_compare_preflight"]["artifact_detail"])
        self.assertEqual(by_id["swebench_verified_20"]["artifact_status"], "FAIL")

    def test_swebench_preflight_without_runtime_metadata_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_complete_artifacts(root)
            preflight_path = root / "swe" / "COMPARE-PREFLIGHT.json"
            doc = json.loads(preflight_path.read_text(encoding="utf-8"))
            doc.pop("runtime", None)
            preflight_path.write_text(json.dumps(doc), encoding="utf-8")

            parser = bat.argparse.ArgumentParser()
            bat.add_common_args(parser)
            parsed = parser.parse_args(base_args(root))
            report = bat.build_report(parsed)

        by_id = {row["id"]: row for row in report["steps"]}
        self.assertFalse(report["summary"]["complete"])
        self.assertEqual(by_id["swebench_compare_preflight"]["artifact_status"], "FAIL")
        self.assertIn("runtime missing", by_id["swebench_compare_preflight"]["artifact_detail"])

    def test_wrong_model_swebench_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out = root / "out"
            swe = root / "swe"
            out.mkdir()
            swe.mkdir()
            (swe / "compare.json").write_text(json.dumps({
                "model": "Qwen/Qwen3.6-27B",
                "served_as": "qwen36-27b",
                "engine": "vllm",
                "selection": "slice 0:20",
                "arms": [
                    {"agent": {"arm": "raw-vllm", "instances": 20,
                               "endpoint": "http://127.0.0.1:8000/v1"},
                     "grade": {"available": True, "total": 20}},
                    {"agent": {"arm": "fak-gateway", "instances": 20,
                               "endpoint": "http://127.0.0.1:8080/v1"},
                     "grade": {"available": True, "total": 20}},
                ],
            }), encoding="utf-8")
            parser = bat.argparse.ArgumentParser()
            bat.add_common_args(parser)
            parsed = parser.parse_args(base_args(root))
            report = bat.build_report(parsed)

        by_id = {row["id"]: row for row in report["steps"]}
        self.assertEqual(by_id["swebench_verified_20"]["artifact_status"], "FAIL")
        self.assertIn("Qwen/Qwen3.6-27B", by_id["swebench_verified_20"]["artifact_detail"])

    def test_generic_floor_json_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out = root / "out"
            out.mkdir()
            (out / "turntax-airline.json").write_text('{"ok": true}\n', encoding="utf-8")
            parser = bat.argparse.ArgumentParser()
            bat.add_common_args(parser)
            parsed = parser.parse_args(base_args(root))
            report = bat.build_report(parsed)

        by_id = {row["id"]: row for row in report["steps"]}
        self.assertEqual(by_id["turntax_airline"]["artifact_status"], "FAIL")
        self.assertIn("slice_id", by_id["turntax_airline"]["artifact_detail"])

    def test_markdown_keeps_pending_measurement_framing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            parser = bat.argparse.ArgumentParser()
            bat.add_common_args(parser)
            parsed = parser.parse_args(base_args(Path(td)))
            report = bat.build_report(parsed)
            md = bat.render_markdown(report)

        self.assertIn("PENDING_MEASUREMENT", md)
        self.assertIn("Pending measurement", md)
        self.assertIn("not benchmark results", md)

    def test_script_renderer_has_parser_guard_and_twenty_task_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            parser = bat.argparse.ArgumentParser()
            bat.add_common_args(parser)
            parsed = parser.parse_args(base_args(Path(td)) + ["--python", "python3"])
            report = bat.build_report(parsed)
            script = bat.render_script(report)

        self.assertIn("set -euo pipefail", script)
        self.assertIn("mkdir -p", script)
        self.assertIn("glm52-agentic-battery", script)
        self.assertIn("GLM52_TOOL_CALL_PARSER:?", script)
        self.assertIn("python3 tools/glm52_vllm_agentic_battery.py", script)
        self.assertIn("--verified-count 20", script)
        self.assertIn("zai-org/GLM-5.2-FP8", script)
        self.assertIn("tools/dgx_swebench_compare.py", script)
        self.assertIn("--preflight-only", script)
        self.assertIn("FINAL-CHECK.md", script)
        self.assertIn("--authority-draft", script)
        self.assertIn("BENCHMARK-AUTHORITY-DRAFT.md", script)
        self.assertIn("FAK_SWEBENCH_DIFFICULTY", script)
        self.assertIn("FAK_SWEBENCH_DATASET", script)
        self.assertNotIn("--allow-pending", script)

    def test_write_text_uses_lf_line_endings_for_generated_runner(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "runner.sh"
            bat.write_text(path, "if true; then\n  echo ok\nfi\n")
            data = path.read_bytes()

        self.assertIn(b"fi\n", data)
        self.assertNotIn(b"\r\n", data)

    def test_swebench_source_flag_is_recorded_and_preserved_in_final_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            diff = "/bench/swebench_verified_difficulty.json"
            parser = bat.argparse.ArgumentParser()
            bat.add_common_args(parser)
            parsed = parser.parse_args(base_args(Path(td)) + ["--swebench-difficulty", diff])
            report = bat.build_report(parsed)
            steps = {row["id"]: row for row in report["steps"]}
            script = bat.render_script(report)
            final_cmd = bat.final_gate_command(report)

        self.assertEqual(report["swebench_difficulty"], diff)
        self.assertIn("--difficulty", steps["swebench_floor_20"]["command"]["argv"])
        self.assertIn(diff, steps["swebench_floor_20"]["command"]["argv"])
        self.assertIn("--swebench-difficulty", final_cmd)
        self.assertIn(diff, final_cmd)
        self.assertNotIn("before swebench_floor_20", script)

    def test_authority_draft_is_written_only_from_complete_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            write_complete_artifacts(root, raw_resolved=7, gateway_resolved=8)
            manifest = root / "manifest.json"
            draft = root / "authority-draft.md"
            rc = bat.main(base_args(root) + [
                "--out", str(manifest),
                "--authority-draft", str(draft),
            ])

            self.assertEqual(rc, 0)
            text = draft.read_text(encoding="utf-8")

        self.assertIn("BENCHMARK-AUTHORITY Draft", text)
        self.assertIn("raw-vLLM 7/20 (35.0%)", text)
        self.assertIn("fak-gateway 8/20 (40.0%)", text)
        self.assertIn("gateway latency tax 1.12x", text)
        self.assertIn("adjudication-tax-witness.json", text)
        self.assertIn("COMPARE-PREFLIGHT.json", text)
        self.assertIn("DONE.rc", text)
        self.assertIn("SWE-bench instance IDs", text)
        self.assertIn("swebench__verified-19", text)
        self.assertIn("preds_raw-vllm/report.json", text)
        self.assertIn("grade_fak-gateway.log", text)
        self.assertIn("Compare preflight raw base", text)
        self.assertIn("SWE-bench `4.0.0`", text)
        self.assertIn("vLLM `0.10.0`", text)
        self.assertIn("turntax turns_saved 9", text)
        self.assertIn("Fak-native SWE-bench floor", text)
        self.assertIn("workers `1,2,4,8`", text)
        self.assertIn("Sessionbench floor", text)
        self.assertIn("Fanbench floor", text)
        self.assertIn("Radixbench floor", text)
        self.assertIn("cache_hit_rate `0.866`", text)

    def test_authority_draft_refuses_pending_even_with_allow_pending(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "manifest.json"
            draft = root / "authority-draft.md"
            rc = bat.main(base_args(root) + [
                "--out", str(manifest),
                "--authority-draft", str(draft),
                "--allow-pending",
            ])

            self.assertEqual(rc, 2)
            self.assertTrue(manifest.exists())
            self.assertFalse(draft.exists())


if __name__ == "__main__":
    unittest.main()
