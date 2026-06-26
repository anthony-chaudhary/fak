#!/usr/bin/env python3
"""GLM-5.2/vLLM agentic benchmark battery manifest and artifact checker.

This is the executable run sheet for the GLM-5.2-on-vLLM comparison:

* raw vLLM versus the same endpoint behind ``fak serve`` for serving witness,
  adjudication tax, and a 20-task SWE-bench Verified slice;
* fak-native agentic floor artifacts for SWE-bench geometry, turn-tax,
  sessionbench, fanbench, and radixbench.

The default mode is deliberately non-invasive. It does not start a serving
engine, run the official SWE-bench harness, or touch model weights. It writes a
machine-readable manifest, optionally a Markdown run sheet, and checks whether
the required artifacts already exist and pass the honesty gates.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

SCHEMA = "fak.glm52-vllm-agentic-battery.v1"
DEFAULT_MODEL = "zai-org/GLM-5.2"
DEFAULT_CHECKPOINT_MODEL = "zai-org/GLM-5.2-FP8"
DEFAULT_SERVED_MODEL = "glm-5.2"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_GATEWAY_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_OUT_DIR = Path("experiments/vllm")
DEFAULT_SERVING_OUT_DIR = Path("experiments/glm52")
DEFAULT_SWE_RUN_DIR = Path("/tmp/swe-glm52-vllm-20")
REQUIRED_PREFLIGHT_VERDICTS = {"READY", "READY_PENDING_INSTALL"}
RUN_CONTRACT_SCHEMA = "fak.glm52-vllm-raw-fak-run-contract.v1"
RUN_CONTRACT_REQUIRED_ARTIFACTS = [
    "serving_witness",
    "vllm_tax",
    "compare_preflight",
    "compare_result",
    "compare_markdown",
    "done_rc",
    "final_check",
]
RUN_CONTRACT_REQUIRED_METRICS = [
    "completed",
    "pass_rate",
    "resolved",
    "resolve_pct",
    "safe_completion_rate",
    "tool_call_counts",
    "token_counts.prompt_tokens",
    "token_counts.completion_tokens",
    "token_counts.total_tokens",
    "latency.agent_sec",
    "latency.grade_sec",
    "latency.gateway_tax",
    "cost_proxy.total_tokens",
    "cost_proxy.wall_seconds",
    "policy_blocks",
    "evidence_completeness",
]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def slash(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def shell_join(argv: list[str]) -> str:
    """Render argv as a shell-friendly command without changing the argv witness."""
    return " ".join(shlex.quote(str(part)) for part in argv)


def read_json(path: Path) -> tuple[dict[str, Any] | None, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "missing"
    except (OSError, ValueError) as exc:
        return None, f"unreadable: {exc}"
    if not isinstance(data, dict):
        return None, "not a JSON object"
    return data, "ok"


def nested(doc: dict[str, Any], path: list[str]) -> Any:
    cur: Any = doc
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def require_list_values(problems: list[str], label: str, value: Any, required: list[str]) -> None:
    if not isinstance(value, list):
        problems.append(f"{label} missing")
        return
    missing = [want for want in required if want not in value]
    if missing:
        problems.append(f"{label} missing {', '.join(missing)}")


def json_present(path: Path) -> tuple[bool, str]:
    doc, detail = read_json(path)
    if doc is None:
        return False, detail
    return True, "parseable JSON artifact"


def norm_url(value: str) -> str:
    return str(value or "").rstrip("/")


def expectation(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "preflight_model": DEFAULT_MODEL,
        "checkpoint_model": args.model,
        "served_model_name": args.served_model_name,
        "base_url": norm_url(args.base_url),
        "context_length": args.context_length,
        "quant": args.quant,
        "verified_count": args.verified_count,
        "require_gpu_name": args.require_gpu_name,
    }


def default_run_contract_path(out_dir: Path) -> Path:
    return out_dir / "glm52-agentic-battery" / "raw-vllm-vs-fak-gateway-contract.json"


def compare_budget() -> dict[str, Any]:
    return {
        "workers": 1,
        "max_iterations": 0,
        "agent_timeout_sec": 3600,
        "grade_workers": 2,
        "grade_timeout_sec": 2400,
        "implicit_retries": 0,
    }


def build_run_contract(args: argparse.Namespace) -> dict[str, Any]:
    p = paths(args.out_dir, args.serving_out_dir, args.swe_run_dir)
    run_contract = args.run_contract or default_run_contract_path(args.out_dir)
    selection = f"slice 0:{args.verified_count}"
    raw_base = norm_url(args.base_url)
    gateway_base = DEFAULT_GATEWAY_BASE_URL
    swe_common = [
        args.python, "tools/dgx_swebench_compare.py",
        "--engine", "vllm",
        "--model", args.model,
        "--served-model-name", args.served_model_name,
        "--raw-base-url", args.base_url,
        "--verified-count", str(args.verified_count),
        "--skip-engine-serve",
        "--require-tool-calls",
        "--require-grade",
        "--run-dir", slash(args.swe_run_dir),
    ]
    if args.require_gpu_name:
        swe_common += ["--require-gpu-name", args.require_gpu_name]
    return {
        "schema": RUN_CONTRACT_SCHEMA,
        "generated_at": utc_now(),
        "status": "PENDING_MEASUREMENT",
        "result_claim_allowed": False,
        "result_claim_allowed_when": [
            "preflight artifact passes",
            "raw-vLLM endpoint and fak-gateway endpoint both pass tool-call probes",
            "both arms run the same selected SWE-bench instance IDs",
            "official SWE-bench harness grades both arms",
            "DONE.rc is 0",
            "GLM-5.2/vLLM battery final-check status is COMPLETE",
        ],
        "shared_config": {
            "checkpoint_model": args.model,
            "served_model_name": args.served_model_name,
            "engine": "vllm",
            "selection": selection,
            "verified_count": args.verified_count,
            "same_task_ids_required": True,
            "tool_calls_required": True,
            "official_grade_required": True,
            "raw_vllm_base_url": raw_base,
            "gateway_base_url": gateway_base,
            "context_length": args.context_length,
            "quant": args.quant,
            "require_gpu_name": args.require_gpu_name,
            "tax_count": args.tax_count,
            "budget": compare_budget(),
        },
        "arms": [
            {
                "id": "raw-vllm",
                "model": args.served_model_name,
                "endpoint": raw_base,
                "engine": "vllm",
                "request_path": "direct OpenAI-compatible /v1 endpoint",
            },
            {
                "id": "fak-gateway",
                "model": args.served_model_name,
                "endpoint": gateway_base,
                "upstream": raw_base,
                "engine": "fak serve fronting the same raw vLLM endpoint",
                "request_path": "OpenAI-compatible /v1 through fak adjudication",
            },
        ],
        "claim_frame": {
            "allowed": "same-model harness gain",
            "must_not_claim": [
                "GLM-5.2 superiority over other model families",
                "fak-native reduced-scale GLM kernel superiority over full-size vLLM serving",
                "vLLM throughput result before the live artifacts pass",
            ],
        },
        "metrics_required": RUN_CONTRACT_REQUIRED_METRICS,
        "measurement_requirements": {
            "per_arm": [
                {
                    "name": "pass_rate",
                    "source": "compare.json arms[].grade.resolve_pct from the official SWE-bench harness",
                    "required_for": ["raw-vllm", "fak-gateway"],
                },
                {
                    "name": "safe_completion_rate",
                    "source": "serving witness plus compare artifact safety/completion fields",
                    "required_for": ["raw-vllm", "fak-gateway"],
                },
                {
                    "name": "token_counts",
                    "source": "tool-call probes and tax witness usage prompt/completion/total tokens",
                    "required_for": ["raw-vllm", "fak-gateway"],
                },
                {
                    "name": "tool_call_counts",
                    "source": "compare.json tool_call_probes[].tool_calls",
                    "required_for": ["raw-vllm", "fak-gateway"],
                },
                {
                    "name": "latency",
                    "source": "compare.json arms[].agent.agent_sec, arms[].grade.grade_sec, and tax witness latency",
                    "required_for": ["raw-vllm", "fak-gateway"],
                },
                {
                    "name": "cost_proxy",
                    "source": "total tokens plus wall seconds; no dollar claim without a separate price file",
                    "required_for": ["raw-vllm", "fak-gateway"],
                },
                {
                    "name": "policy_blocks",
                    "source": "fak-gateway serving/tax/compare evidence; raw-vllm records zero-or-not-applicable explicitly",
                    "required_for": ["raw-vllm", "fak-gateway"],
                },
            ],
            "cross_arm": [
                {
                    "name": "same_task_ids",
                    "source": "compare.json selection_instance_ids and selection_instance_ids_match",
                },
                {
                    "name": "evidence_completeness",
                    "source": "final-check.json summary.required_passed/required_artifacts with no missing or failed required artifacts",
                },
            ],
        },
        "evidence_completeness": {
            "final_check": slash(run_contract.with_name("final-check.json")),
            "complete_when": "summary.complete == true and required_missing/required_failed are empty",
            "authority_link_required": True,
        },
        "required_artifacts": {
            "serving_witness": slash(p["serving_json"]),
            "vllm_tax": slash(p["tax_json"]),
            "compare_preflight": slash(p["swe_compare_preflight"]),
            "compare_result": slash(p["swe_compare_json"]),
            "compare_markdown": slash(p["swe_compare_md"]),
            "done_rc": slash(p["swe_compare_done"]),
            "final_check": slash(run_contract.with_name("final-check.json")),
        },
        "commands": {
            "compare_preflight": shell_join(swe_common + ["--preflight-only"]),
            "compare_run": shell_join(swe_common),
        },
        "honesty": (
            "This is a run contract, not a benchmark result. It records the identity, "
            "arms, budgets, required metrics, and artifact boundaries for the future "
            "live GLM-5.2/vLLM comparison."
        ),
    }


def check_run_contract(path: Path, expected: dict[str, Any]) -> tuple[bool, str]:
    doc, detail = read_json(path)
    if doc is None:
        return False, detail
    problems: list[str] = []
    if doc.get("schema") != RUN_CONTRACT_SCHEMA:
        problems.append(f"schema={doc.get('schema')!r}")
    if doc.get("status") != "PENDING_MEASUREMENT":
        problems.append(f"status={doc.get('status')!r}")
    if doc.get("result_claim_allowed") is not False:
        problems.append(f"result_claim_allowed={doc.get('result_claim_allowed')!r}")
    shared = doc.get("shared_config")
    if not isinstance(shared, dict):
        problems.append("shared_config missing")
        shared = {}
    else:
        checks = {
            "checkpoint_model": expected["checkpoint_model"],
            "served_model_name": expected["served_model_name"],
            "engine": "vllm",
            "selection": f"slice 0:{expected['verified_count']}",
            "verified_count": expected["verified_count"],
            "raw_vllm_base_url": expected["base_url"],
            "gateway_base_url": DEFAULT_GATEWAY_BASE_URL,
            "context_length": expected["context_length"],
            "quant": expected["quant"],
            "require_gpu_name": expected["require_gpu_name"],
        }
        for key, want in checks.items():
            if shared.get(key) != want:
                problems.append(f"shared_config.{key}={shared.get(key)!r}")
        for key in ("same_task_ids_required", "tool_calls_required", "official_grade_required"):
            if shared.get(key) is not True:
                problems.append(f"shared_config.{key}={shared.get(key)!r}")
        budget = shared.get("budget") if isinstance(shared.get("budget"), dict) else {}
        if budget.get("implicit_retries") != 0:
            problems.append(f"budget.implicit_retries={budget.get('implicit_retries')!r}")
    arms = doc.get("arms") if isinstance(doc.get("arms"), list) else []
    by_arm = {row.get("id"): row for row in arms if isinstance(row, dict)}
    raw = by_arm.get("raw-vllm")
    gateway = by_arm.get("fak-gateway")
    if not raw:
        problems.append("missing raw-vllm arm")
    else:
        if raw.get("endpoint") != expected["base_url"]:
            problems.append(f"raw-vllm.endpoint={raw.get('endpoint')!r}")
        if raw.get("model") != expected["served_model_name"]:
            problems.append(f"raw-vllm.model={raw.get('model')!r}")
    if not gateway:
        problems.append("missing fak-gateway arm")
    else:
        if gateway.get("endpoint") != DEFAULT_GATEWAY_BASE_URL:
            problems.append(f"fak-gateway.endpoint={gateway.get('endpoint')!r}")
        if gateway.get("upstream") != expected["base_url"]:
            problems.append(f"fak-gateway.upstream={gateway.get('upstream')!r}")
        if gateway.get("model") != expected["served_model_name"]:
            problems.append(f"fak-gateway.model={gateway.get('model')!r}")
    required = doc.get("required_artifacts") if isinstance(doc.get("required_artifacts"), dict) else {}
    for key in RUN_CONTRACT_REQUIRED_ARTIFACTS:
        if not isinstance(required.get(key), str) or not required.get(key):
            problems.append(f"required_artifacts.{key} missing")
    require_list_values(problems, "metrics_required", doc.get("metrics_required"),
                        RUN_CONTRACT_REQUIRED_METRICS)
    claim_frame = doc.get("claim_frame")
    if not isinstance(claim_frame, dict):
        problems.append("claim_frame missing")
    else:
        if claim_frame.get("allowed") != "same-model harness gain":
            problems.append(f"claim_frame.allowed={claim_frame.get('allowed')!r}")
        forbidden = claim_frame.get("must_not_claim")
        if not isinstance(forbidden, list) or not any("model families" in str(row) for row in forbidden):
            problems.append("claim_frame.must_not_claim missing model-family guard")
    measurement = doc.get("measurement_requirements")
    if not isinstance(measurement, dict):
        problems.append("measurement_requirements missing")
    else:
        per_arm = measurement.get("per_arm")
        if not isinstance(per_arm, list):
            problems.append("measurement_requirements.per_arm missing")
        else:
            names = [str(row.get("name")) for row in per_arm if isinstance(row, dict)]
            for want in ("pass_rate", "safe_completion_rate", "token_counts",
                         "tool_call_counts", "latency", "cost_proxy", "policy_blocks"):
                if want not in names:
                    problems.append(f"measurement_requirements.per_arm missing {want}")
        cross_arm = measurement.get("cross_arm")
        if not isinstance(cross_arm, list):
            problems.append("measurement_requirements.cross_arm missing")
        else:
            names = [str(row.get("name")) for row in cross_arm if isinstance(row, dict)]
            for want in ("same_task_ids", "evidence_completeness"):
                if want not in names:
                    problems.append(f"measurement_requirements.cross_arm missing {want}")
    completeness = doc.get("evidence_completeness")
    if not isinstance(completeness, dict):
        problems.append("evidence_completeness missing")
    else:
        if not isinstance(completeness.get("final_check"), str) or not completeness.get("final_check"):
            problems.append("evidence_completeness.final_check missing")
        if completeness.get("authority_link_required") is not True:
            problems.append(
                f"evidence_completeness.authority_link_required={completeness.get('authority_link_required')!r}"
            )
    commands = doc.get("commands") if isinstance(doc.get("commands"), dict) else {}
    if "tools/dgx_swebench_compare.py" not in str(commands.get("compare_run", "")):
        problems.append("commands.compare_run missing dgx_swebench_compare.py")
    if problems:
        return False, "; ".join(problems) + "; expected raw-vLLM/fak-gateway run contract"
    return True, "raw-vLLM/fak-gateway same-model run contract present; result claims disabled"


def check_preflight(path: Path, expected: dict[str, Any]) -> tuple[bool, str]:
    doc, detail = read_json(path)
    if doc is None:
        return False, detail
    problems: list[str] = []
    if doc.get("model") != expected["preflight_model"]:
        problems.append(f"model={doc.get('model')!r}")
    verdict = nested(doc, ["summary", "node_verdict"])
    if verdict not in REQUIRED_PREFLIGHT_VERDICTS:
        problems.append(f"node_verdict={verdict!r}")
    engines = doc.get("engines") if isinstance(doc.get("engines"), list) else []
    vllm = next((row for row in engines if isinstance(row, dict) and row.get("engine") == "vllm"), None)
    if not vllm:
        problems.append("missing vllm engine row")
    else:
        if vllm.get("ready") is not True:
            problems.append(f"vllm.ready={vllm.get('ready')!r}")
        if vllm.get("quant") != expected["quant"]:
            problems.append(f"vllm.quant={vllm.get('quant')!r}")
    if problems:
        return False, "; ".join(problems) + "; expected GLM-5.2 vLLM READY/PENDING preflight"
    return True, f"model={expected['preflight_model']} vllm quant={expected['quant']} node_verdict={verdict}"


def check_serving_witness(path: Path, expected: dict[str, Any]) -> tuple[bool, str]:
    doc, detail = read_json(path)
    if doc is None:
        return False, detail
    problems: list[str] = []
    if doc.get("model") != expected["served_model_name"]:
        problems.append(f"model={doc.get('model')!r}")
    if norm_url(doc.get("base_url")) != expected["base_url"]:
        problems.append(f"base_url={doc.get('base_url')!r}")
    if int(doc.get("context_length") or 0) != int(expected["context_length"]):
        problems.append(f"context_length={doc.get('context_length')!r}")
    if nested(doc, ["engine_cache", "engine"]) != "vllm":
        problems.append(f"engine_cache.engine={nested(doc, ['engine_cache', 'engine'])!r}")
    state = nested(doc, ["summary", "full_size_serving_witness"])
    if state != "PASS":
        problems.append(f"full_size_serving_witness={state!r}")
    if problems:
        return False, "; ".join(problems) + "; expected GLM-5.2 served-model witness PASS"
    return True, f"model={expected['served_model_name']} base_url={expected['base_url']} PASS"


def check_tax_witness(path: Path, expected: dict[str, Any]) -> tuple[bool, str]:
    doc, detail = read_json(path)
    if doc is None:
        return False, detail
    problems: list[str] = []
    if doc.get("model") != expected["served_model_name"]:
        problems.append(f"model={doc.get('model')!r}")
    if norm_url(doc.get("base_url")) != expected["base_url"]:
        problems.append(f"base_url={doc.get('base_url')!r}")
    if doc.get("engine_cache_engine") != "vllm":
        problems.append(f"engine_cache_engine={doc.get('engine_cache_engine')!r}")
    state = nested(doc, ["summary", "vllm_adjudication_tax_witness"])
    latency_tax = nested(doc, ["summary", "latency_tax"])
    raw_ok = nested(doc, ["raw_vllm", "ok_samples"])
    gateway_ok = nested(doc, ["gateway", "ok_samples"])
    if state != "PASS":
        problems.append(f"state={state!r}")
    if latency_tax is None:
        problems.append("latency_tax=None")
    if not raw_ok:
        problems.append(f"raw_ok={raw_ok!r}")
    if not gateway_ok:
        problems.append(f"gateway_ok={gateway_ok!r}")
    if problems:
        return False, "; ".join(problems) + "; expected measured GLM-5.2 vLLM tax PASS"
    return True, f"model={expected['served_model_name']} latency_tax={latency_tax}"


def check_swebench_compare_preflight(path: Path, expected: dict[str, Any]) -> tuple[bool, str]:
    doc, detail = read_json(path)
    if doc is None:
        return False, detail
    count = int(expected["verified_count"])
    problems: list[str] = []
    if doc.get("schema") != "fak.dgx-swebench-compare-preflight.v1":
        problems.append(f"schema={doc.get('schema')!r}")
    if doc.get("ok") is not True:
        problems.append(f"ok={doc.get('ok')!r}")
    if doc.get("model") != expected["checkpoint_model"]:
        problems.append(f"model={doc.get('model')!r}")
    if doc.get("served_as") != expected["served_model_name"]:
        problems.append(f"served_as={doc.get('served_as')!r}")
    if doc.get("engine") != "vllm":
        problems.append(f"engine={doc.get('engine')!r}")
    if doc.get("selection") != f"slice 0:{count}":
        problems.append(f"selection={doc.get('selection')!r}")
    preflight_arms = set(doc.get("arms") or [])
    for arm in ("raw-vllm", "fak-gateway"):
        if arm not in preflight_arms:
            problems.append(f"missing {arm}")
    config = doc.get("config")
    if not isinstance(config, dict):
        problems.append("config missing")
        config = {}
    else:
        if norm_url(config.get("raw_base_url")) != expected["base_url"]:
            problems.append(f"config.raw_base_url={config.get('raw_base_url')!r}")
        if config.get("skip_engine_serve") is not True:
            problems.append(f"config.skip_engine_serve={config.get('skip_engine_serve')!r}")
        if config.get("require_gpu_name") != expected["require_gpu_name"]:
            problems.append(f"config.require_gpu_name={config.get('require_gpu_name')!r}")
    runtime = doc.get("runtime")
    if not isinstance(runtime, dict):
        problems.append("runtime missing")
        runtime = {}
    else:
        for key in ("mini_swe_agent", "swebench_python", "swebench_version", "vllm_version"):
            if not isinstance(runtime.get(key), dict):
                problems.append(f"runtime.{key} missing")
    checks = doc.get("checks")
    if not isinstance(checks, list) or not checks:
        problems.append("checks missing")
    else:
        failed = [
            str(row.get("name") or "?")
            for row in checks
            if isinstance(row, dict) and row.get("ok") is not True
        ]
        if failed:
            problems.append("failed checks: " + ", ".join(failed))
    if problems:
        return False, "; ".join(problems) + "; expected SWE-bench compare preflight PASS"
    return True, f"SWE-bench compare preflight passed for {count} GLM-5.2/vLLM tasks"


def check_swebench_compare(path: Path, expected: dict[str, Any]) -> tuple[bool, str]:
    doc, detail = read_json(path)
    if doc is None:
        return False, detail
    count = int(expected["verified_count"])
    problems: list[str] = []
    done_path = path.with_name("DONE.rc")
    try:
        done_rc = int(done_path.read_text(encoding="utf-8").strip())
    except FileNotFoundError:
        problems.append("DONE.rc missing")
    except (OSError, ValueError) as exc:
        problems.append(f"DONE.rc unreadable: {exc}")
    else:
        if done_rc != 0:
            problems.append(f"DONE.rc={done_rc!r}")
    preflight_ok, preflight_detail = check_swebench_compare_preflight(
        path.with_name("COMPARE-PREFLIGHT.json"), expected
    )
    if not preflight_ok:
        problems.append(f"COMPARE-PREFLIGHT.json {preflight_detail}")
    if doc.get("model") != expected["checkpoint_model"]:
        problems.append(f"model={doc.get('model')!r}")
    if doc.get("served_as") != expected["served_model_name"]:
        problems.append(f"served_as={doc.get('served_as')!r}")
    if doc.get("engine") != "vllm":
        problems.append(f"engine={doc.get('engine')!r}")
    if doc.get("selection") != f"slice 0:{count}":
        problems.append(f"selection={doc.get('selection')!r}")
    probes = doc.get("tool_call_probes")
    if not isinstance(probes, list):
        problems.append("tool_call_probes missing")
        probes = []
    by_probe: dict[str, dict[str, Any]] = {}
    for row in probes:
        if isinstance(row, dict) and isinstance(row.get("arm"), str):
            by_probe[row["arm"]] = row
    for arm in ("raw-vllm", "fak-gateway"):
        probe = by_probe.get(arm)
        if not probe:
            problems.append(f"{arm} tool-call probe missing")
            continue
        if probe.get("ok") is not True:
            problems.append(f"{arm} tool-call probe ok={probe.get('ok')!r}")
        try:
            tool_calls = int(probe.get("tool_calls"))
        except (TypeError, ValueError):
            problems.append(f"{arm} tool_calls={probe.get('tool_calls')!r}")
        else:
            if tool_calls <= 0:
                problems.append(f"{arm} tool_calls={tool_calls!r}")
    results = doc.get("arms") or doc.get("results")
    if not isinstance(results, list):
        return False, "missing arms/results list"
    by_arm: dict[str, dict[str, Any]] = {}
    for row in results:
        if not isinstance(row, dict):
            continue
        arm = nested(row, ["agent", "arm"])
        if isinstance(arm, str):
            by_arm[arm] = row
    missing = [arm for arm in ("raw-vllm", "fak-gateway") if arm not in by_arm]
    if missing:
        problems.append(f"missing arms: {', '.join(missing)}")
    selection_ids = doc.get("selection_instance_ids")
    if not isinstance(selection_ids, list):
        problems.append("selection_instance_ids missing")
        selection_ids = []
    elif len(selection_ids) != count:
        problems.append(f"selection_instance_ids len={len(selection_ids)!r}")
    elif not all(isinstance(instance_id, str) and instance_id for instance_id in selection_ids):
        problems.append("selection_instance_ids contains non-string/empty ids")
    if doc.get("selection_instance_ids_match") is not True:
        problems.append(f"selection_instance_ids_match={doc.get('selection_instance_ids_match')!r}")
    arm_instance_ids: dict[str, list[str]] = {}
    for arm, row in by_arm.items():
        instances = nested(row, ["agent", "instances"])
        endpoint = nested(row, ["agent", "endpoint"])
        completed = nested(row, ["agent", "completed"])
        instance_ids = nested(row, ["agent", "instance_ids"])
        grade_total = nested(row, ["grade", "total"])
        grade_available = nested(row, ["grade", "available"])
        grade_rc = nested(row, ["grade", "grade_rc"])
        grade_run_id = nested(row, ["grade", "run_id"])
        grade_submitted = nested(row, ["grade", "submitted"])
        grade_report_path = nested(row, ["grade", "report_path"])
        grade_log = nested(row, ["grade", "grade_log"])
        resolved_ids = nested(row, ["grade", "resolved_ids"])
        resolved = nested(row, ["grade", "resolved"])
        resolve_pct = nested(row, ["grade", "resolve_pct"])
        if arm == "raw-vllm" and norm_url(endpoint) != expected["base_url"]:
            problems.append(f"raw-vllm endpoint={endpoint!r}")
        if instances != count:
            problems.append(f"{arm} instances={instances!r}")
        if not isinstance(instance_ids, list):
            problems.append(f"{arm} instance_ids missing")
        elif len(instance_ids) != count:
            problems.append(f"{arm} instance_ids len={len(instance_ids)!r}")
        elif not all(isinstance(instance_id, str) and instance_id for instance_id in instance_ids):
            problems.append(f"{arm} instance_ids contains non-string/empty ids")
        else:
            arm_instance_ids[arm] = instance_ids
            if selection_ids and instance_ids != selection_ids:
                problems.append(f"{arm} instance_ids differ from selection_instance_ids")
        if completed is None:
            problems.append(f"{arm} completed missing")
        if grade_total != count:
            problems.append(f"{arm} grade.total={grade_total!r}")
        if grade_available is not True:
            problems.append(f"{arm} grade.available={grade_available!r}")
        if grade_rc != 0:
            problems.append(f"{arm} grade.grade_rc={grade_rc!r}")
        if grade_run_id != f"swecmp-{arm}":
            problems.append(f"{arm} grade.run_id={grade_run_id!r}")
        if grade_submitted != count:
            problems.append(f"{arm} grade.submitted={grade_submitted!r}")
        if not isinstance(grade_report_path, str) or not grade_report_path:
            problems.append(f"{arm} grade.report_path missing")
        if not isinstance(grade_log, str) or not grade_log:
            problems.append(f"{arm} grade.grade_log missing")
        if resolved is None:
            problems.append(f"{arm} grade.resolved missing")
        else:
            try:
                resolved_int = int(resolved)
            except (TypeError, ValueError):
                problems.append(f"{arm} grade.resolved={resolved!r}")
            else:
                if not 0 <= resolved_int <= count:
                    problems.append(f"{arm} grade.resolved={resolved!r}")
                if isinstance(resolved_ids, list) and len(resolved_ids) != resolved_int:
                    problems.append(f"{arm} grade.resolved_ids len={len(resolved_ids)!r}")
        if not isinstance(resolved_ids, list):
            problems.append(f"{arm} grade.resolved_ids missing")
        elif not all(isinstance(instance_id, str) and instance_id for instance_id in resolved_ids):
            problems.append(f"{arm} grade.resolved_ids contains non-string/empty ids")
        elif arm in arm_instance_ids:
            unknown_resolved = [instance_id for instance_id in resolved_ids
                                if instance_id not in arm_instance_ids[arm]]
            if unknown_resolved:
                problems.append(
                    f"{arm} grade.resolved_ids outside selection: {', '.join(unknown_resolved[:3])}"
                )
        if resolve_pct is None:
            problems.append(f"{arm} grade.resolve_pct missing")
    if {"raw-vllm", "fak-gateway"} <= set(arm_instance_ids):
        if arm_instance_ids["raw-vllm"] != arm_instance_ids["fak-gateway"]:
            problems.append("raw-vllm and fak-gateway instance_ids differ")
    if problems:
        return False, "; ".join(problems)
    return True, (
        f"model={expected['checkpoint_model']} engine=vllm raw-vllm and fak-gateway "
        f"each have the same {count} graded instance IDs"
    )


def check_swebench_floor(path: Path, expected: dict[str, Any]) -> tuple[bool, str]:
    doc, detail = read_json(path)
    if doc is None:
        return False, detail
    count = int(expected["verified_count"])
    problems: list[str] = []
    if doc.get("schema") != "fak-swebench-compare/1":
        problems.append(f"schema={doc.get('schema')!r}")
    if nested(doc, ["dataset", "instances"]) != count:
        problems.append(f"dataset.instances={nested(doc, ['dataset', 'instances'])!r}")
    if doc.get("workers") != [1, 2, 4, 8]:
        problems.append(f"workers={doc.get('workers')!r}")
    families = doc.get("families")
    if not isinstance(families, list) or len(families) < 4:
        problems.append("families missing or incomplete")
    if problems:
        return False, "; ".join(problems) + "; expected fak SWE-bench 20-task floor"
    return True, f"fak SWE-bench floor: {count} instances workers=[1,2,4,8]"


def check_turntax_floor(path: Path) -> tuple[bool, str]:
    doc, detail = read_json(path)
    if doc is None:
        return False, detail
    problems: list[str] = []
    if nested(doc, ["provenance", "slice_id"]) != "turntax-airline":
        problems.append(f"slice_id={nested(doc, ['provenance', 'slice_id'])!r}")
    if doc.get("consistency_check") != "ok":
        problems.append(f"consistency_check={doc.get('consistency_check')!r}")
    if nested(doc, ["net", "turns_saved"]) is None:
        problems.append("net.turns_saved missing")
    if nested(doc, ["safety_floor", "injections_admitted_fak"]) != 0:
        problems.append("injections_admitted_fak not zero")
    if nested(doc, ["safety_floor", "destructive_executed_fak"]) != 0:
        problems.append("destructive_executed_fak not zero")
    if problems:
        return False, "; ".join(problems) + "; expected turntax-airline floor"
    return True, f"turntax-airline turns_saved={nested(doc, ['net', 'turns_saved'])}"


def check_sessionbench_floor(path: Path) -> tuple[bool, str]:
    doc, detail = read_json(path)
    if doc is None:
        return False, detail
    problems: list[str] = []
    if not str(doc.get("engine") or "").startswith("fak sessionbench"):
        problems.append(f"engine={doc.get('engine')!r}")
    if doc.get("model") != "smollm2-135m [synthetic]":
        problems.append(f"model={doc.get('model')!r}")
    if doc.get("prefix") != 2048:
        problems.append(f"prefix={doc.get('prefix')!r}")
    if doc.get("decode_per_turn") != 32:
        problems.append(f"decode_per_turn={doc.get('decode_per_turn')!r}")
    if doc.get("result_per_turn") != 64:
        problems.append(f"result_per_turn={doc.get('result_per_turn')!r}")
    cells = doc.get("cells") if isinstance(doc.get("cells"), list) else []
    want_cell = any(
        isinstance(row, dict)
        and row.get("turns") == 50
        and row.get("agents") == 5
        and row.get("prefix") == 2048
        and row.get("decode") == 32
        and row.get("result") == 64
        for row in cells
    )
    if not want_cell:
        problems.append("missing T=50 C=5 P=2048 D=32 R=64 cell")
    timing_mode = doc.get("timing_mode")
    if problems:
        return False, "; ".join(problems) + "; expected sessionbench synthetic floor"
    if timing_mode == "deterministic_prefill_token_counts_only":
        return True, "sessionbench synthetic T=50 C=5 deterministic token-count floor present"
    return True, "sessionbench synthetic T=50 C=5 floor present"


def check_fanbench_floor(path: Path) -> tuple[bool, str]:
    doc, detail = read_json(path)
    if doc is None:
        return False, detail
    problems: list[str] = []
    if nested(doc, ["profile", "name"]) != "research-goal":
        problems.append(f"profile.name={nested(doc, ['profile', 'name'])!r}")
    if doc.get("trials") != 12:
        problems.append(f"trials={doc.get('trials')!r}")
    if 1024 not in (doc.get("agent_grid") or []):
        problems.append("agent_grid missing 1024")
    if 2048 not in (doc.get("prefix_grid") or []):
        problems.append("prefix_grid missing 2048")
    if not isinstance(doc.get("cells"), list) or not doc.get("cells"):
        problems.append("cells missing")
    if problems:
        return False, "; ".join(problems) + "; expected fanbench research floor"
    return True, "fanbench research grid/trials present"


def check_radixbench_floor(path: Path) -> tuple[bool, str]:
    doc, detail = read_json(path)
    if doc is None:
        return False, detail
    problems: list[str] = []
    if "radixbench" not in str(doc.get("engine") or ""):
        problems.append(f"engine={doc.get('engine')!r}")
    if not str(doc.get("model") or "").startswith("synthetic-llama"):
        problems.append(f"model={doc.get('model')!r}")
    if nested(doc, ["policy_eviction", "demonstrated"]) is not True:
        problems.append("policy_eviction.demonstrated not true")
    workloads = doc.get("workloads") if isinstance(doc.get("workloads"), list) else []
    agents = next((row for row in workloads if isinstance(row, dict) and row.get("name") == "agents"), None)
    if not agents or agents.get("cache_hit_rate") is None:
        problems.append("agents workload cache_hit_rate missing")
    if problems:
        return False, "; ".join(problems) + "; expected radixbench synthetic floor"
    return True, f"radixbench agents hit_rate={agents.get('cache_hit_rate')}"


def artifact_check(step: dict[str, Any], expected: dict[str, Any]) -> tuple[str, str]:
    path = Path(step["primary_artifact"])
    kind = step.get("check")
    if kind == "preflight":
        ok, detail = check_preflight(path, expected)
    elif kind == "serving_witness":
        ok, detail = check_serving_witness(path, expected)
    elif kind == "tax_witness":
        ok, detail = check_tax_witness(path, expected)
    elif kind == "swebench_compare":
        ok, detail = check_swebench_compare(path, expected)
    elif kind == "swebench_compare_preflight":
        ok, detail = check_swebench_compare_preflight(path, expected)
    elif kind == "swebench_floor":
        ok, detail = check_swebench_floor(path, expected)
    elif kind == "turntax_floor":
        ok, detail = check_turntax_floor(path)
    elif kind == "sessionbench_floor":
        ok, detail = check_sessionbench_floor(path)
    elif kind == "fanbench_floor":
        ok, detail = check_fanbench_floor(path)
    elif kind == "radixbench_floor":
        ok, detail = check_radixbench_floor(path)
    elif kind == "run_contract":
        ok, detail = check_run_contract(path, expected)
    elif kind == "json_present":
        ok, detail = json_present(path)
    else:
        ok, detail = False, f"unknown check {kind!r}"
    if ok:
        return "PASS", detail
    return "MISSING" if detail == "missing" else "FAIL", detail


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model", default=DEFAULT_CHECKPOINT_MODEL,
                    help="HF checkpoint/model path served by vLLM and recorded in compare artifacts")
    ap.add_argument("--served-model-name", default=DEFAULT_SERVED_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--context-length", type=int, default=131072)
    ap.add_argument("--quant", default="fp8")
    ap.add_argument("--require-gpu-name", default="H200",
                    help='GPU-name guard for the live SWE-bench driver; pass "" to disable')
    ap.add_argument("--verified-count", type=int, default=20)
    ap.add_argument("--tax-count", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help="artifact root for vLLM preflight/tax and fak-native floor outputs")
    ap.add_argument("--serving-out-dir", type=Path, default=DEFAULT_SERVING_OUT_DIR,
                    help="artifact root for the full-size serving witness")
    ap.add_argument("--swe-run-dir", type=Path, default=DEFAULT_SWE_RUN_DIR)
    ap.add_argument("--python", default="python")
    ap.add_argument("--go", default="go")
    ap.add_argument("--out", type=Path, default=None, help="write manifest JSON here")
    ap.add_argument("--markdown", type=Path, default=None, help="write Markdown run sheet here")
    ap.add_argument("--script", type=Path, default=None,
                    help="write a bash runner for the planned benchmark battery")
    ap.add_argument("--run-contract", type=Path, default=None,
                    help="write and check the standalone raw-vLLM/fak-gateway run contract")
    ap.add_argument("--contract-only", action="store_true",
                    help="write --run-contract and exit without emitting the full manifest")
    ap.add_argument("--authority-draft", type=Path, default=None,
                    help="write a BENCHMARK-AUTHORITY.md draft entry; requires COMPLETE artifacts")
    ap.add_argument("--allow-pending", action="store_true",
                    help="exit 0 even when required artifacts are still missing")
    ap.add_argument("--swebench-difficulty", default=os.environ.get("FAK_SWEBENCH_DIFFICULTY", ""),
                    help="fak SWE-bench floor difficulty map; defaults to FAK_SWEBENCH_DIFFICULTY")
    ap.add_argument("--swebench-dataset", default=os.environ.get("FAK_SWEBENCH_DATASET", ""),
                    help="fak SWE-bench floor dataset JSON/JSONL; defaults to FAK_SWEBENCH_DATASET")


def paths(out_dir: Path, serving_out_dir: Path, swe_run_dir: Path) -> dict[str, Path]:
    return {
        "preflight_json": out_dir / "glm52-vllm-preflight.json",
        "preflight_md": out_dir / "glm52-vllm-preflight.md",
        "serving_json": serving_out_dir / "full-size-serving-witness.json",
        "serving_md": serving_out_dir / "full-size-serving-witness.md",
        "tax_json": out_dir / "adjudication-tax-witness.json",
        "tax_md": out_dir / "adjudication-tax-witness.md",
        "swe_compare_preflight": swe_run_dir / "COMPARE-PREFLIGHT.json",
        "swe_compare_json": swe_run_dir / "compare.json",
        "swe_compare_md": swe_run_dir / "COMPARE.md",
        "swe_compare_done": swe_run_dir / "DONE.rc",
        "swe_floor_json": out_dir / "swebench-20-fak-floor.json",
        "swe_floor_md": out_dir / "swebench-20-fak-floor.md",
        "turntax_json": out_dir / "turntax-airline.json",
        "session_json": out_dir / "sessionbench-synthetic.json",
        "fanbench_json": out_dir / "fanbench-research.json",
        "fanbench_csv": out_dir / "fanbench-research.csv",
        "radix_json": out_dir / "radixbench-synthetic.json",
    }


def step(
    *,
    step_id: str,
    kind: str,
    description: str,
    argv: list[str],
    primary_artifact: Path,
    artifacts: list[Path],
    check: str,
    required: bool = True,
    status: str = "planned",
    notes: str = "",
) -> dict[str, Any]:
    return {
        "id": step_id,
        "kind": kind,
        "description": description,
        "required": required,
        "status": status,
        "command": {"argv": argv, "shell": shell_join(argv)},
        "primary_artifact": slash(primary_artifact),
        "artifacts": [slash(p) for p in artifacts],
        "check": check,
        "claim_boundary": notes,
    }


def build_steps(args: argparse.Namespace) -> list[dict[str, Any]]:
    p = paths(args.out_dir, args.serving_out_dir, args.swe_run_dir)
    run_contract = args.run_contract or default_run_contract_path(args.out_dir)

    contract_cmd = [
        args.python, "tools/glm52_vllm_agentic_battery.py",
        "--model", args.model,
        "--served-model-name", args.served_model_name,
        "--base-url", args.base_url,
        "--context-length", str(args.context_length),
        "--quant", args.quant,
        "--verified-count", str(args.verified_count),
        "--tax-count", str(args.tax_count),
        "--out-dir", slash(args.out_dir),
        "--serving-out-dir", slash(args.serving_out_dir),
        "--swe-run-dir", slash(args.swe_run_dir),
        "--run-contract", slash(run_contract),
        "--contract-only",
    ]
    if args.require_gpu_name:
        contract_cmd += ["--require-gpu-name", args.require_gpu_name]

    preflight = [
        args.python, "tools/glm52_serve_preflight.py",
        "--engine", "vllm",
        "--quant", args.quant,
        "--require-ready",
        "--out", slash(p["preflight_json"]),
        "--markdown", slash(p["preflight_md"]),
    ]

    serve_shell = (
        ': "${{GLM52_TOOL_CALL_PARSER:?set GLM52_TOOL_CALL_PARSER to the vLLM parser name}}" && '
        'ENGINE=vllm SERVED_NAME={served} PORT=8000 '
        'ENGINE_ARGS="--enable-auto-tool-choice --tool-call-parser ${{GLM52_TOOL_CALL_PARSER}}" '
        'bash tools/glm52_sglang_vllm_serve.sh'
    ).format(served=shlex.quote(args.served_model_name))

    serving = [
        args.python, "tools/glm52_serving_witness.py",
        "--base-url", args.base_url,
        "--model", args.served_model_name,
        "--engine-cache-engine", "vllm",
        "--context-length", str(args.context_length),
        "--out", slash(p["serving_json"]),
        "--markdown", slash(p["serving_md"]),
    ]

    tax = [
        args.python, "tools/vllm_tax_witness.py",
        "--base-url", args.base_url,
        "--model", args.served_model_name,
        "--count", str(args.tax_count),
        "--record",
        "--out", slash(p["tax_json"]),
        "--markdown", slash(p["tax_md"]),
    ]

    swe_compare = [
        args.python, "tools/dgx_swebench_compare.py",
        "--engine", "vllm",
        "--model", args.model,
        "--served-model-name", args.served_model_name,
        "--raw-base-url", args.base_url,
        "--verified-count", str(args.verified_count),
        "--skip-engine-serve",
        "--require-tool-calls",
        "--require-grade",
        "--run-dir", slash(args.swe_run_dir),
    ]
    if args.require_gpu_name:
        swe_compare += ["--require-gpu-name", args.require_gpu_name]
    swe_compare_preflight = swe_compare + ["--preflight-only"]

    swe_floor = [
        args.go, "run", "./cmd/fak", "swebench", "compare",
        "--workers", "1,2,4,8",
        "--limit", str(args.verified_count),
        "--with-adjudication",
        "--out", slash(p["swe_floor_json"]),
        "--md", slash(p["swe_floor_md"]),
    ]
    if args.swebench_dataset:
        swe_floor += ["--dataset", args.swebench_dataset]
    if args.swebench_difficulty:
        swe_floor += ["--difficulty", args.swebench_difficulty]

    steps: list[dict[str, Any]] = []
    if args.run_contract:
        steps.append(step(
            step_id="raw_fak_run_contract",
            kind="run-contract",
            description="Record the raw-vLLM vs fak-gateway same-model run contract.",
            argv=contract_cmd,
            primary_artifact=run_contract,
            artifacts=[run_contract],
            check="run_contract",
            notes="Contract only; result_claim_allowed=false until the live artifacts pass.",
        ))
    steps += [
        step(
            step_id="preflight",
            kind="live-readiness",
            description="Fail-closed GLM-5.2/vLLM node readiness gate.",
            argv=preflight,
            primary_artifact=p["preflight_json"],
            artifacts=[p["preflight_json"], p["preflight_md"]],
            check="preflight",
            notes="Readiness only; no resolve-rate or speed number comes from this rung.",
        ),
        {
            "id": "serve_raw_vllm",
            "kind": "manual-live-serving",
            "description": "Start raw vLLM for GLM-5.2 before the live witnesses.",
            "required": True,
            "status": "manual",
            "command": {"argv": [], "shell": serve_shell},
            "primary_artifact": "",
            "artifacts": [],
            "check": "manual",
            "claim_boundary": "This step is witnessed by the serving and tax artifacts, not by this manifest.",
        },
        step(
            step_id="serving_witness",
            kind="live-serving",
            description="Direct + fak-gateway + quarantine serving witness.",
            argv=serving,
            primary_artifact=p["serving_json"],
            artifacts=[p["serving_json"], p["serving_md"]],
            check="serving_witness",
            notes="Counts only external full-size serving; native reduced-scale evidence is not folded here.",
        ),
        step(
            step_id="vllm_tax",
            kind="live-serving",
            description="Measure fak gateway adjudication tax over raw vLLM.",
            argv=tax,
            primary_artifact=p["tax_json"],
            artifacts=[p["tax_json"], p["tax_md"]],
            check="tax_witness",
            notes="Tax >= 1 is expected; this is a cost witness, not a fak speed win.",
        ),
        step(
            step_id="swebench_compare_preflight",
            kind="live-agentic-preflight",
            description="Fail fast before the long SWE-bench agent run.",
            argv=swe_compare_preflight,
            primary_artifact=p["swe_compare_preflight"],
            artifacts=[p["swe_compare_preflight"], p["swe_compare_done"]],
            check="swebench_compare_preflight",
            notes="Dependency/raw-endpoint/GPU gate only; resolve claims come from swebench_verified_20.",
        ),
        step(
            step_id="swebench_verified_20",
            kind="live-agentic",
            description="Run raw-vLLM vs fak-gateway on a 20-task SWE-bench Verified slice.",
            argv=swe_compare,
            primary_artifact=p["swe_compare_json"],
            artifacts=[
                p["swe_compare_preflight"],
                p["swe_compare_json"],
                p["swe_compare_md"],
                p["swe_compare_done"],
            ],
            check="swebench_compare",
            notes="Completion/resolve claims require official harness grading for both arms.",
        ),
        step(
            step_id="swebench_floor_20",
            kind="fak-native-floor",
            description="Record the deterministic fak-native SWE-bench geometry floor at the same 20-task scale.",
            argv=swe_floor,
            primary_artifact=p["swe_floor_json"],
            artifacts=[p["swe_floor_json"], p["swe_floor_md"]],
            check="swebench_floor",
            notes="This is deterministic mechanism evidence, not a raw vLLM head-to-head.",
        ),
        step(
            step_id="turntax_airline",
            kind="fak-native-floor",
            description="Measure the turn-tax safety/control floor.",
            argv=[args.go, "run", "./cmd/fak", "turntax", "--suite", "turntax-airline",
                  "--out", slash(p["turntax_json"])],
            primary_artifact=p["turntax_json"],
            artifacts=[p["turntax_json"]],
            check="turntax_floor",
            notes="Engine-agnostic safety floor; do not present as vLLM throughput.",
        ),
        step(
            step_id="sessionbench_synthetic",
            kind="fak-native-floor",
            description="Record the deterministic multi-agent session token-work floor.",
            argv=[args.go, "run", "./cmd/sessionbench", "-synthetic", "smollm2-135m",
                  "-turns", "50", "-agents", "5", "-prefix", "2048",
                  "-decode", "32", "-result", "64", "-counts-only",
                  "-out", slash(p["session_json"])],
            primary_artifact=p["session_json"],
            artifacts=[p["session_json"]],
            check="sessionbench_floor",
            notes="Deterministic token-work floor; live wall-clock requires a separate measured artifact.",
        ),
        step(
            step_id="fanbench_research",
            kind="fak-native-floor",
            description="Measure fan-out prefix sharing and cost geometry.",
            argv=[args.go, "run", "./cmd/fanbench", "-profile", "research", "-trials", "12",
                  "-out", slash(p["fanbench_json"]), "-csv", slash(p["fanbench_csv"])],
            primary_artifact=p["fanbench_json"],
            artifacts=[p["fanbench_json"], p["fanbench_csv"]],
            check="fanbench_floor",
            notes="Cost geometry and kernel reuse evidence; not a raw serving-engine comparison.",
        ),
        step(
            step_id="radixbench_synthetic",
            kind="fak-native-floor",
            description="Measure RadixAttention-style prefix-cache hit-rate evidence.",
            argv=[args.go, "run", "./cmd/radixbench", "-live=false",
                  "-out", slash(p["radix_json"])],
            primary_artifact=p["radix_json"],
            artifacts=[p["radix_json"]],
            check="radixbench_floor",
            notes="Hardware-independent hit-rate evidence; live tok/s needs a separate serving run.",
        ),
    ]
    return steps


def annotate_artifacts(steps: list[dict[str, Any]], expected: dict[str, Any]) -> None:
    for row in steps:
        if row.get("check") == "manual":
            row["artifact_status"] = "MANUAL"
            row["artifact_detail"] = "manual serving step; witnessed by downstream artifacts"
            continue
        status, detail = artifact_check(row, expected)
        row["artifact_status"] = status
        row["artifact_detail"] = detail
        if status == "PASS":
            row["status"] = "artifact-pass"
        elif status == "FAIL":
            row["status"] = "artifact-fail"
        else:
            row["status"] = "planned"


def summarize(steps: list[dict[str, Any]]) -> dict[str, Any]:
    required = [s for s in steps if s.get("required") and s.get("check") != "manual"]
    passed = [s for s in required if s.get("artifact_status") == "PASS"]
    failed = [s for s in required if s.get("artifact_status") == "FAIL"]
    missing = [s for s in required if s.get("artifact_status") == "MISSING"]
    complete = len(passed) == len(required)
    return {
        "complete": complete,
        "status": "COMPLETE" if complete else "PENDING_MEASUREMENT",
        "required_artifacts": len(required),
        "required_passed": len(passed),
        "required_failed": [s["id"] for s in failed],
        "required_missing": [s["id"] for s in missing],
        "honesty": (
            "No GLM-5.2/vLLM benchmark number is quotable until every required "
            "artifact passes and any copied number is linked from BENCHMARK-AUTHORITY.md."
        ),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    steps = build_steps(args)
    expected = expectation(args)
    annotate_artifacts(steps, expected)
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "model_family": DEFAULT_MODEL,
        "checkpoint_model": args.model,
        "served_model_name": args.served_model_name,
        "raw_vllm_base_url": args.base_url,
        "context_length": args.context_length,
        "quant": args.quant,
        "verified_count": args.verified_count,
        "tax_count": args.tax_count,
        "python": args.python,
        "out_dir": slash(args.out_dir),
        "serving_out_dir": slash(args.serving_out_dir),
        "swe_run_dir": slash(args.swe_run_dir),
        "swebench_difficulty": args.swebench_difficulty,
        "swebench_dataset": args.swebench_dataset,
        "run_contract_artifact": slash(args.run_contract) if args.run_contract else "",
        "steps": steps,
    }
    report["summary"] = summarize(steps)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# GLM-5.2 vLLM Agentic Benchmark Battery",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Model family: `{report['model_family']}`",
        f"- Checkpoint: `{report['checkpoint_model']}` served as `{report['served_model_name']}`",
        f"- Raw vLLM endpoint: `{report['raw_vllm_base_url']}`",
        f"- Status: **`{summary['status']}`**",
        f"- Required artifacts passed: `{summary['required_passed']}/{summary['required_artifacts']}`",
        "",
        "> Pending measurement: this manifest contains commands and artifact checks, not benchmark results.",
        "",
        "| step | kind | artifact | status | detail |",
        "|---|---|---|---|---|",
    ]
    for row in report["steps"]:
        artifact = row.get("primary_artifact") or "(manual)"
        detail = row.get("artifact_detail", "")
        lines.append(
            f"| `{row['id']}` | {row['kind']} | `{artifact}` | "
            f"{row.get('artifact_status', row.get('status'))} | {detail} |"
        )
    lines += [
        "",
        "## Commands",
        "",
    ]
    for row in report["steps"]:
        lines += [
            f"### {row['id']}",
            "",
            f"{row['description']}",
            "",
            "```bash",
            row["command"]["shell"],
            "```",
            "",
        ]
    if summary["required_missing"]:
        lines.append("Missing required artifacts: `" + "`, `".join(summary["required_missing"]) + "`.")
    if summary["required_failed"]:
        lines.append("Failed required artifacts: `" + "`, `".join(summary["required_failed"]) + "`.")
    lines += ["", summary["honesty"], ""]
    return "\n".join(lines)


def step_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in report["steps"]}


def load_step_json(report: dict[str, Any], step_id: str) -> dict[str, Any]:
    row = step_map(report)[step_id]
    doc, detail = read_json(Path(row["primary_artifact"]))
    if doc is None:
        raise ValueError(f"{step_id}: {detail}")
    return doc


def arm_result(compare: dict[str, Any], arm: str) -> dict[str, Any]:
    for row in compare.get("arms") or compare.get("results") or []:
        if isinstance(row, dict) and nested(row, ["agent", "arm"]) == arm:
            return row
    raise ValueError(f"missing {arm} arm in SWE-bench compare artifact")


def fmt_resolve(row: dict[str, Any]) -> str:
    grade = row.get("grade") if isinstance(row.get("grade"), dict) else {}
    resolved = grade.get("resolved")
    total = grade.get("total")
    pct = grade.get("resolve_pct")
    if pct is None and isinstance(resolved, (int, float)) and isinstance(total, (int, float)) and total:
        pct = round(100 * float(resolved) / float(total), 1)
    return f"{resolved}/{total} ({pct}%)"


def fmt_completed(row: dict[str, Any]) -> str:
    agent = row.get("agent") if isinstance(row.get("agent"), dict) else {}
    return f"{agent.get('completed')}/{agent.get('instances')}"


def target_session_cell(doc: dict[str, Any]) -> dict[str, Any]:
    cells = doc.get("cells") if isinstance(doc.get("cells"), list) else []
    for row in cells:
        if (
            isinstance(row, dict)
            and row.get("turns") == 50
            and row.get("agents") == 5
            and row.get("prefix") == 2048
            and row.get("decode") == 32
            and row.get("result") == 64
        ):
            return row
    return {}


def first_agents_workload(doc: dict[str, Any]) -> dict[str, Any]:
    workloads = doc.get("workloads") if isinstance(doc.get("workloads"), list) else []
    for row in workloads:
        if isinstance(row, dict) and row.get("name") == "agents":
            return row
    return {}


def render_authority_draft(report: dict[str, Any]) -> str:
    compare = load_step_json(report, "swebench_verified_20")
    compare_preflight = load_step_json(report, "swebench_compare_preflight")
    tax = load_step_json(report, "vllm_tax")
    serving = load_step_json(report, "serving_witness")
    swe_floor = load_step_json(report, "swebench_floor_20")
    turntax = load_step_json(report, "turntax_airline")
    session = load_step_json(report, "sessionbench_synthetic")
    fan = load_step_json(report, "fanbench_research")
    radix = load_step_json(report, "radixbench_synthetic")
    steps = step_map(report)
    raw = arm_result(compare, "raw-vllm")
    gateway = arm_result(compare, "fak-gateway")
    latency_tax = nested(tax, ["summary", "latency_tax"])
    decode_tax = nested(tax, ["summary", "decode_tps_tax"])
    artifact_refs = list(steps["swebench_verified_20"].get("artifacts") or [])
    artifact_refs += [
        steps["vllm_tax"]["primary_artifact"],
        steps["serving_witness"]["primary_artifact"],
        steps["swebench_floor_20"]["primary_artifact"],
        steps["turntax_airline"]["primary_artifact"],
        steps["sessionbench_synthetic"]["primary_artifact"],
        steps["fanbench_research"]["primary_artifact"],
        steps["radixbench_synthetic"]["primary_artifact"],
    ]
    artifact_cell = " + ".join(f"`{ref}`" for ref in artifact_refs)
    selection_ids = compare.get("selection_instance_ids")
    number = (
        f"SWE-bench Verified raw-vLLM {fmt_resolve(raw)} vs fak-gateway "
        f"{fmt_resolve(gateway)}; completion raw {fmt_completed(raw)} vs "
        f"gateway {fmt_completed(gateway)}; gateway latency tax {latency_tax}x"
    )
    if decode_tax is not None:
        number += f", decode-tps tax {decode_tax}x"
    turntax_saved = nested(turntax, ["net", "turns_saved"])
    radix_agents = first_agents_workload(radix)
    if turntax_saved is not None:
        number += f"; turntax turns_saved {turntax_saved}"
    if radix_agents.get("cache_hit_rate") is not None:
        number += f"; radix agents hit_rate {radix_agents.get('cache_hit_rate')}"
    lines = [
        "# BENCHMARK-AUTHORITY Draft - GLM-5.2/vLLM Agentic Battery",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "This draft was emitted only after the strict GLM-5.2/vLLM battery checker "
        "saw every required artifact pass. Review it, commit the referenced artifacts, "
        "then paste the table row into `BENCHMARK-AUTHORITY.md`.",
        "",
        "## Quick Reference Row Draft",
        "",
        "| Claim | Number | Model | Baseline | Commit | Artifact |",
        "|---|---|---|---|---|---|",
        (
            "| **GLM-5.2/vLLM agentic battery, 20-task SWE-bench Verified** | "
            f"**{number}** | `{report['checkpoint_model']}` served as "
            f"`{report['served_model_name']}` on vLLM | raw vLLM, same checkpoint "
            f"and selection | _this commit_ | {artifact_cell} |"
        ),
        "",
        "## Checked Artifacts",
        "",
        "| Gate | Artifact | Detail |",
        "|---|---|---|",
    ]
    for row in report["steps"]:
        if row.get("check") == "manual":
            continue
        lines.append(
            f"| `{row['id']}` | `{row['primary_artifact']}` | "
            f"{row.get('artifact_detail', '')} |"
        )
    lines += [
        "",
        "## Source Facts",
        "",
        f"- Serving model: `{serving.get('model')}` at `{serving.get('base_url')}`; "
        f"context length `{serving.get('context_length')}`; engine cache "
        f"`{nested(serving, ['engine_cache', 'engine'])}`.",
        f"- SWE-bench selection: `{compare.get('selection')}`; dataset "
        f"`{compare.get('dataset')}`; engine `{compare.get('engine')}`.",
        f"- Raw endpoint: `{nested(raw, ['agent', 'endpoint'])}`.",
        f"- Gateway endpoint: `{nested(gateway, ['agent', 'endpoint'])}`.",
        f"- Raw grade report: `{nested(raw, ['grade', 'report_path'])}`; log "
        f"`{nested(raw, ['grade', 'grade_log'])}`.",
        f"- Gateway grade report: `{nested(gateway, ['grade', 'report_path'])}`; log "
        f"`{nested(gateway, ['grade', 'grade_log'])}`.",
        f"- Fak-native SWE-bench floor: `{nested(swe_floor, ['dataset', 'instances'])}` "
        f"instances; workers `{','.join(str(w) for w in (swe_floor.get('workers') or []))}`.",
        f"- Turntax floor: turns_saved `{turntax_saved}`; injections admitted "
        f"`{nested(turntax, ['safety_floor', 'injections_admitted_fak'])}`; destructive "
        f"executed `{nested(turntax, ['safety_floor', 'destructive_executed_fak'])}`.",
    ]
    session_cell = target_session_cell(session)
    lines.append(
        f"- Sessionbench floor: model `{session.get('model')}`; T `{session_cell.get('turns')}`; "
        f"C `{session_cell.get('agents')}`; P `{session_cell.get('prefix')}`; "
        f"D `{session_cell.get('decode')}`; R `{session_cell.get('result')}`."
    )
    lines.append(
        f"- Fanbench floor: profile `{nested(fan, ['profile', 'name'])}`; trials "
        f"`{fan.get('trials')}`; max agents `{max(fan.get('agent_grid') or [None])}`; "
        f"prefix grid `{','.join(str(p) for p in (fan.get('prefix_grid') or []))}`."
    )
    lines.append(
        f"- Radixbench floor: agents cache_hit_rate `{radix_agents.get('cache_hit_rate')}`; "
        f"policy eviction `{nested(radix, ['policy_eviction', 'demonstrated'])}`."
    )
    if isinstance(selection_ids, list) and selection_ids:
        lines.append(
            "- SWE-bench instance IDs: "
            + ", ".join(f"`{instance_id}`" for instance_id in selection_ids)
            + "."
        )
    config = compare_preflight.get("config") if isinstance(compare_preflight.get("config"), dict) else {}
    runtime = compare_preflight.get("runtime") if isinstance(compare_preflight.get("runtime"), dict) else {}
    lines += [
        f"- Compare preflight raw base: `{config.get('raw_base_url')}`; gateway base "
        f"`{config.get('gateway_base_url')}`; GPU guard `{config.get('require_gpu_name')}`.",
        f"- Compare runtime: mini-swe-agent `{nested(runtime, ['mini_swe_agent', 'detail'])}`; "
        f"SWE-bench `{nested(runtime, ['swebench_version', 'detail'])}`; "
        f"vLLM `{nested(runtime, ['vllm_version', 'detail'])}`.",
    ]
    lines.append("")
    return "\n".join(lines)


def final_gate_command(report: dict[str, Any]) -> list[str]:
    final_dir = Path(str(report["out_dir"])) / "glm52-agentic-battery"
    cmd = [
        str(report.get("python") or "python"), "tools/glm52_vllm_agentic_battery.py",
        "--model", str(report["checkpoint_model"]),
        "--served-model-name", str(report["served_model_name"]),
        "--base-url", str(report["raw_vllm_base_url"]),
        "--context-length", str(report["context_length"]),
        "--quant", str(report["quant"]),
        "--verified-count", str(report["verified_count"]),
        "--tax-count", str(report["tax_count"]),
        "--out-dir", str(report["out_dir"]),
        "--serving-out-dir", str(report["serving_out_dir"]),
        "--swe-run-dir", str(report["swe_run_dir"]),
        "--out", slash(final_dir / "final-check.json"),
        "--markdown", slash(final_dir / "FINAL-CHECK.md"),
        "--authority-draft", slash(final_dir / "BENCHMARK-AUTHORITY-DRAFT.md"),
    ]
    if report.get("swebench_difficulty"):
        cmd += ["--swebench-difficulty", str(report["swebench_difficulty"])]
    if report.get("swebench_dataset"):
        cmd += ["--swebench-dataset", str(report["swebench_dataset"])]
    if report.get("run_contract_artifact"):
        cmd += ["--run-contract", str(report["run_contract_artifact"])]
    return cmd


def render_script(report: dict[str, Any]) -> str:
    final_dir = Path(str(report["out_dir"])) / "glm52-agentic-battery"
    setup_dirs = sorted({
        slash(report["out_dir"]),
        slash(report["serving_out_dir"]),
        slash(report["swe_run_dir"]),
        slash(final_dir),
    })
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"# Generated by tools/glm52_vllm_agentic_battery.py at {report['generated_at']}",
        "# Runs the GLM-5.2/vLLM agentic benchmark battery described in the manifest.",
        "# Set GLM52_TOOL_CALL_PARSER to the parser name required by your vLLM build/model card.",
        "",
        "# Prepare artifact directories.",
        shell_join(["mkdir", "-p", *setup_dirs]),
        "",
    ]
    needs_swe_source_guard = not report.get("swebench_difficulty") and not report.get("swebench_dataset")
    for row in report["steps"]:
        if row["id"] == "swebench_floor_20" and needs_swe_source_guard:
            lines += [
                "# fak swebench compare needs an official difficulty map or dataset.",
                'if [[ -z "${FAK_SWEBENCH_DIFFICULTY:-}" && -z "${FAK_SWEBENCH_DATASET:-}" ]]; then',
                '  echo "set FAK_SWEBENCH_DIFFICULTY or FAK_SWEBENCH_DATASET before swebench_floor_20" >&2',
                "  exit 2",
                "fi",
            ]
        lines += [
            f"# {row['id']}: {row['description']}",
            row["command"]["shell"],
            "",
        ]
    lines += [
        "# Final strict gate: exits nonzero unless every required artifact exists",
        "# and matches the GLM-5.2/vLLM benchmark identity.",
        shell_join(final_gate_command(report)),
        "",
    ]
    return "\n".join(lines)


def write_text(path: Path, text: str) -> None:
    if path.parent and str(path.parent) not in ("", "."):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    args = ap.parse_args(argv)

    if args.contract_only:
        if not args.run_contract:
            sys.stderr.write("--contract-only requires --run-contract\n")
            return 2
        write_text(args.run_contract, json.dumps(build_run_contract(args), indent=2) + "\n")
        return 0
    if args.run_contract:
        write_text(args.run_contract, json.dumps(build_run_contract(args), indent=2) + "\n")

    report = build_report(args)
    payload = json.dumps(report, indent=2) + "\n"
    if args.out:
        write_text(args.out, payload)
    else:
        sys.stdout.write(payload)
    if args.markdown:
        write_text(args.markdown, render_markdown(report))
    if args.script:
        write_text(args.script, render_script(report))
        try:
            args.script.chmod(args.script.stat().st_mode | 0o111)
        except OSError:
            pass
    if args.authority_draft:
        if not report["summary"]["complete"]:
            sys.stderr.write(
                "authority draft refused: GLM-5.2/vLLM battery is not COMPLETE\n"
            )
            return 2
        try:
            write_text(args.authority_draft, render_authority_draft(report))
        except ValueError as exc:
            sys.stderr.write(f"authority draft refused: {exc}\n")
            return 2
    if report["summary"]["complete"] or args.allow_pending:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
