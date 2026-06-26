"""Hermetic helper tests for tools/dgx_swebench_compare.py."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dgx_swebench_compare as cmp  # noqa: E402


def args(**overrides):
    base = {
        "engine": "sglang",
        "model": cmp.DEFAULT_MODEL,
        "served_model_name": cmp.DEFAULT_SERVED,
        "tp": cmp.DEFAULT_TP,
        "mem_fraction": cmp.DEFAULT_MEM_FRACTION,
        "context_length": 0,
        "engine_port": cmp.DEFAULT_ENGINE_PORT,
        "fak_port": cmp.DEFAULT_FAK_PORT,
        "raw_base_url": "",
        "gateway_base_url": "",
        "sglang_python": "python-sglang",
        "vllm_command": "vllm",
        "fak_bin": "fak",
        "tool_call_parser": "qwen3_coder",
        "engine_args": "",
        "filter": "astropy__astropy-12907",
        "slice": "0:1",
        "verified_count": 0,
        "workers": 1,
        "agent_timeout": 1,
        "max_iterations": 0,
        "redo": False,
        "grade_workers": 1,
        "grade_timeout": 1,
        "skip_serve": False,
        "skip_engine_serve": False,
        "skip_gateway_serve": False,
        "preflight_only": False,
        "stop_serve": False,
        "require_tool_calls": False,
        "require_grade": False,
        "require_gpu_name": "",
        "arms": "",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class CommandBuilderTest(unittest.TestCase):
    def test_default_sglang_command_preserves_qwen_tool_parser(self) -> None:
        cmd = cmp.build_engine_command(args())
        self.assertEqual(cmd[:3], ["python-sglang", "-m", "sglang.launch_server"])
        self.assertIn("--model-path", cmd)
        self.assertIn(cmp.DEFAULT_MODEL, cmd)
        self.assertIn("--served-model-name", cmd)
        self.assertIn(cmp.DEFAULT_SERVED, cmd)
        self.assertIn("--tool-call-parser", cmd)
        self.assertIn("qwen3_coder", cmd)

    def test_vllm_command_carries_glm52_pass_through_flags(self) -> None:
        cmd = cmp.build_engine_command(args(
            engine="vllm",
            model="zai-org/GLM-5.2",
            served_model_name="glm-5.2",
            vllm_command="/opt/vllm/bin/vllm",
            context_length=131072,
            engine_args="--enable-auto-tool-choice --tool-call-parser glm45",
        ))
        self.assertEqual(cmd[:3], ["/opt/vllm/bin/vllm", "serve", "zai-org/GLM-5.2"])
        self.assertIn("--served-model-name", cmd)
        self.assertIn("glm-5.2", cmd)
        self.assertIn("--max-model-len", cmd)
        self.assertIn("131072", cmd)
        self.assertIn("--enable-auto-tool-choice", cmd)
        self.assertIn("--tool-call-parser", cmd)
        self.assertIn("glm45", cmd)


class EndpointAndSelectionTest(unittest.TestCase):
    def test_vllm_arms_use_raw_vllm_and_gateway_v1(self) -> None:
        a = args(engine="vllm", raw_base_url="http://node:8000/v1",
                 gateway_base_url="http://node:8080")
        self.assertEqual(cmp.arm_specs(a), [
            ("raw-vllm", "http://node:8000/v1"),
            ("fak-gateway", "http://node:8080/v1"),
        ])

    def test_verified_count_is_slice_shortcut(self) -> None:
        a = args(verified_count=20)
        cmp.apply_selection_shortcuts(a)
        self.assertEqual(a.filter, "")
        self.assertEqual(a.slice, "0:20")


class ReportRenderingTest(unittest.TestCase):
    def test_write_compare_renders_raw_vllm_without_sglang_wording(self) -> None:
        a = args(engine="vllm", model="zai-org/GLM-5.2", served_model_name="glm-5.2",
                 filter="", slice="0:20")
        instance_ids = [f"case-{i:02d}" for i in range(20)]
        results = [
            {
                "agent": {
                    "arm": "raw-vllm",
                    "endpoint": "http://node:8000/v1",
                    "instances": 20,
                    "instance_ids": instance_ids,
                    "completed": 12,
                    "patch_bytes": 4096,
                    "agent_sec": 100.0,
                },
                "grade": {"available": True, "resolved": 8, "total": 20,
                          "resolve_pct": 40.0, "grade_sec": 30.0},
            },
            {
                "agent": {
                    "arm": "fak-gateway",
                    "endpoint": "http://node:8080/v1",
                    "instances": 20,
                    "instance_ids": list(instance_ids),
                    "completed": 11,
                    "patch_bytes": 3900,
                    "agent_sec": 110.0,
                },
                "grade": {"available": True, "resolved": 7, "total": 20,
                          "resolve_pct": 35.0, "grade_sec": 31.0},
            },
        ]
        probes = [
            {"arm": "raw-vllm", "endpoint": "http://node:8000/v1",
             "ok": True, "status": "PASS", "tool_calls": 1, "finish_reason": "tool_calls"},
            {"arm": "fak-gateway", "endpoint": "http://node:8080/v1",
             "ok": True, "status": "PASS", "tool_calls": 1, "finish_reason": "tool_calls"},
        ]
        with tempfile.TemporaryDirectory() as td:
            cmp.write_compare(Path(td), {"host": "gpu-test"}, results, a, probes)
            payload = json.loads((Path(td) / "compare.json").read_text(encoding="utf-8"))
            md = (Path(td) / "COMPARE.md").read_text(encoding="utf-8")
        self.assertEqual(payload["engine"], "vllm")
        self.assertEqual(payload["selection"], "slice 0:20")
        self.assertEqual(payload["selection_instance_ids"], instance_ids)
        self.assertTrue(payload["selection_instance_ids_match"])
        self.assertEqual(payload["tool_call_probes"][0]["arm"], "raw-vllm")
        self.assertIn("raw-vllm", md)
        self.assertIn("SWE-bench Instances", md)
        self.assertIn("Tool-Call Self-Test", md)
        self.assertIn("zai-org/GLM-5.2", md)
        self.assertNotIn("raw SGLang", md)


class GateHelpersTest(unittest.TestCase):
    def test_write_done_rc_records_detached_completion_status(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cmp.write_done_rc(Path(td), 6)
            got = (Path(td) / "DONE.rc").read_text(encoding="utf-8")
        self.assertEqual(got, "6\n")

    def test_grade_has_resolve_requires_official_count_and_denominator(self) -> None:
        self.assertTrue(cmp.grade_has_resolve({
            "available": True,
            "resolved": 8,
            "resolved_ids": [f"case-{i}" for i in range(8)],
            "total": 20,
            "submitted": 20,
            "resolve_pct": 40.0,
            "report_path": "report.json",
            "grade_log": "grade.log",
        }, 20))
        self.assertFalse(cmp.grade_has_resolve({
            "available": True,
            "resolved_ids": [],
            "total": 20,
            "submitted": 20,
            "resolve_pct": 0.0,
            "report_path": "report.json",
            "grade_log": "grade.log",
        }, 20))
        self.assertFalse(cmp.grade_has_resolve({
            "available": True,
            "resolved": 8,
            "resolved_ids": [f"case-{i}" for i in range(8)],
            "total": 500,
            "submitted": 20,
            "resolve_pct": 1.6,
            "report_path": "report.json",
            "grade_log": "grade.log",
        }, 20))
        self.assertFalse(cmp.grade_has_resolve({
            "available": True,
            "resolved": 8,
            "resolved_ids": [f"case-{i}" for i in range(8)],
            "total": 20,
            "submitted": 20,
            "resolve_pct": 40.0,
            "grade_rc": 1,
            "report_path": "report.json",
            "grade_log": "grade.log",
        }, 20))
        self.assertFalse(cmp.grade_has_resolve({
            "available": True,
            "resolved": 8,
            "resolved_ids": [f"case-{i}" for i in range(8)],
            "total": 20,
            "submitted": 20,
            "resolve_pct": 40.0,
            "grade_log": "grade.log",
        }, 20))

    def test_grade_launch_failure_is_unavailable(self) -> None:
        old_vpy = cmp.VPY
        try:
            cmp.VPY = str(Path("definitely-missing-python-exe"))
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                preds = root / "preds.json"
                preds.write_text(json.dumps([
                    {"instance_id": "x__y-1", "model_patch": "diff --git a/a b/a\n"}
                ]), encoding="utf-8")
                result = cmp.grade("raw-vllm", str(preds), root, args(), submitted=1)
        finally:
            cmp.VPY = old_vpy

        self.assertFalse(result["available"])
        self.assertIn("grade harness failed", result["reason"])
        self.assertEqual(result["total"], 1)

    def test_compare_preflight_writes_failure_for_missing_grader_python(self) -> None:
        old_mini = cmp.MINI
        old_vpy = cmp.VPY
        try:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                fake_mini = root / "mini-extra"
                fake_sglang = root / "python-sglang"
                fake_fak = root / "fak"
                for path in (fake_mini, fake_sglang, fake_fak):
                    path.write_text("#!/bin/sh\n", encoding="utf-8")
                cmp.MINI = str(fake_mini)
                cmp.VPY = str(root / "missing-python")

                a = args(sglang_python=str(fake_sglang), fak_bin=str(fake_fak))
                payload = cmp.compare_preflight(
                    root, a, {"gpu0": "test-gpu"}, cmp.selected_arms(a)
                )
                saved = json.loads((root / "COMPARE-PREFLIGHT.json").read_text(encoding="utf-8"))
        finally:
            cmp.MINI = old_mini
            cmp.VPY = old_vpy

        self.assertFalse(payload["ok"])
        self.assertEqual(saved["schema"], "fak.dgx-swebench-compare-preflight.v1")
        self.assertEqual(saved["config"]["raw_base_url"], "http://127.0.0.1:30000/v1")
        self.assertIn("swebench_python", saved["runtime"])
        self.assertIn("swebench_version", saved["runtime"])
        by_name = {row["name"]: row for row in saved["checks"]}
        self.assertFalse(by_name["swebench-python"]["ok"])
        self.assertFalse(by_name["swebench-harness-import"]["ok"])
        self.assertTrue(by_name["mini-swe-agent"]["ok"])

    def test_compare_preflight_requires_sglang_flashinfer_ninja(self) -> None:
        old_command_exists = cmp.command_exists
        old_check_python_import = cmp.check_python_import
        old_preflight_runtime = cmp.preflight_runtime
        old_check_ninja = cmp.check_sglang_flashinfer_ninja
        try:
            cmp.command_exists = lambda _cmd: True
            cmp.check_python_import = lambda _python, _module: (True, "ok")
            cmp.preflight_runtime = lambda _args: {}
            cmp.check_sglang_flashinfer_ninja = lambda _python: {
                "ok": False,
                "detail": "ninja_package=no; ninja_executable=; install with pinned requirements",
            }
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                payload = cmp.compare_preflight(
                    root, args(), {"gpu0": "NVIDIA A100"}, cmp.selected_arms(args())
                )
                saved = json.loads((root / "COMPARE-PREFLIGHT.json").read_text(encoding="utf-8"))
        finally:
            cmp.command_exists = old_command_exists
            cmp.check_python_import = old_check_python_import
            cmp.preflight_runtime = old_preflight_runtime
            cmp.check_sglang_flashinfer_ninja = old_check_ninja

        self.assertFalse(payload["ok"])
        by_name = {row["name"]: row for row in saved["checks"]}
        self.assertFalse(by_name["sglang-flashinfer-jit-ninja"]["ok"])
        self.assertIn("pinned requirements", by_name["sglang-flashinfer-jit-ninja"]["detail"])

    def test_format_toolcall_probe_names_failures(self) -> None:
        text = cmp.format_toolcall_probe({
            "endpoint": "http://node:8000/v1",
            "status": "FAIL",
            "finish_reason": "stop",
            "tool_calls": 0,
            "content_len": 12,
            "completion_tokens": 3,
        })
        self.assertIn("tool_calls=NONE", text)
        self.assertIn("finish=stop", text)

    def test_run_agent_launch_failure_is_structured(self) -> None:
        old_mini = cmp.MINI
        old_served_models = cmp.served_models
        try:
            cmp.MINI = str(Path("definitely-missing-mini-swe-agent-exe"))
            cmp.served_models = lambda _api_base: "{}"
            with tempfile.TemporaryDirectory() as td:
                result = cmp.run_agent(
                    "raw-vllm",
                    "http://node:8000/v1",
                    Path(td),
                    args(workers=1, agent_timeout=1, max_iterations=0, redo=False),
                )
        finally:
            cmp.MINI = old_mini
            cmp.served_models = old_served_models

        self.assertEqual(result["agent_rc"], -127)
        self.assertEqual(result["instances"], 0)
        self.assertIn("launch failed", result["agent_error"])


if __name__ == "__main__":
    unittest.main()
