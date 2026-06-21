#!/usr/bin/env python3
"""Resolve the API-host bridge proof back to source witnesses."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

import fleet_version


SCHEMA = "fak.api-host-bridge-matrix.v1"
ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PROVIDERS = ["openai-compatible", "xai", "anthropic", "gemini"]


WITNESSES: list[dict[str, Any]] = [
    {
        "id": "policy_manifest_floor",
        "claim": "Reviewable manifests narrow risky allowed tools by argument without prompt tax.",
        "required": True,
        "cwd": "fak",
        "command": "go test ./internal/policy ./cmd/fak -run 'Test(ArgRulesAreLoadBearing|LoadedPolicyIsLoadBearing|ApplyRuntimeInstallsIFCManifestPolicy)$'",
        "argv": ["go", "test", "./internal/policy", "./cmd/fak", "-run", "Test(ArgRulesAreLoadBearing|LoadedPolicyIsLoadBearing|ApplyRuntimeInstallsIFCManifestPolicy)$", "-count=1"],
        "checks": [
            {"type": "go_symbol", "path": "fak/internal/policy/policy_test.go", "name": "TestArgRulesAreLoadBearing"},
            {"type": "go_symbol", "path": "fak/internal/policy/policy_test.go", "name": "TestLoadedPolicyIsLoadBearing"},
            {"type": "go_symbol", "path": "fak/cmd/fak/main_test.go", "name": "TestApplyRuntimeInstallsIFCManifestPolicy"},
        ],
    },
    {
        "id": "openai_compatible_gateway",
        "claim": "OpenAI-compatible clients can route through the gateway, receive filtered/repaired tool calls, and have tool-result bytes quarantined before upstream model send.",
        "required": True,
        "cwd": "fak",
        "command": "go test ./internal/gateway -run 'Test(ChatProxyFiltersAndRepairs|ChatProxyOpenAICompatibleToolResultsAreQuarantinedPreSend|HTTPModelsAndHealth)$'",
        "argv": ["go", "test", "./internal/gateway", "-run", "Test(ChatProxyFiltersAndRepairs|ChatProxyOpenAICompatibleToolResultsAreQuarantinedPreSend|HTTPModelsAndHealth)$", "-count=1"],
        "checks": [
            {"type": "go_symbol", "path": "fak/internal/gateway/gateway_test.go", "name": "TestChatProxyFiltersAndRepairs"},
            {"type": "go_symbol", "path": "fak/internal/gateway/gateway_test.go", "name": "TestChatProxyOpenAICompatibleToolResultsAreQuarantinedPreSend"},
            {"type": "go_symbol", "path": "fak/internal/gateway/gateway_test.go", "name": "TestHTTPModelsAndHealth"},
            {"type": "source_token", "path": "fak/internal/gateway/gateway_test.go", "token": "/v1/chat/completions"},
            {"type": "source_token", "path": "fak/internal/gateway/gateway_test.go", "token": "/v1/models"},
            {"type": "source_token", "path": "fak/internal/gateway/gateway_test.go", "token": "_quarantined"},
            {"type": "source_token", "path": "fak/internal/gateway/gateway_test.go", "token": "pre_send"},
        ],
    },
    {
        "id": "native_provider_adapters",
        "claim": "Provider-specific transcript adapters preserve pre-send result quarantine for native host shapes.",
        "required": True,
        "providers": EXPECTED_PROVIDERS,
        "cwd": "fak",
        "command": "go test ./internal/agent -run 'Test(PreSendQuarantineRedactsToolResultsAcrossAdapters|ProviderAdaptersMarshalNativeToolShapes|ProviderAdaptersOmitToolChoiceWithoutTools|ProviderAdaptersParseToolCalls|HTTPPlannerUsesProviderAdapterAndPreSendQuarantine)$'",
        "argv": ["go", "test", "./internal/agent", "-run", "Test(PreSendQuarantineRedactsToolResultsAcrossAdapters|ProviderAdaptersMarshalNativeToolShapes|ProviderAdaptersOmitToolChoiceWithoutTools|ProviderAdaptersParseToolCalls|HTTPPlannerUsesProviderAdapterAndPreSendQuarantine)$", "-count=1"],
        "checks": [
            {"type": "go_symbol", "path": "fak/internal/agent/adapters_test.go", "name": "TestPreSendQuarantineRedactsToolResultsAcrossAdapters"},
            {"type": "go_symbol", "path": "fak/internal/agent/adapters_test.go", "name": "TestProviderAdaptersMarshalNativeToolShapes"},
            {"type": "go_symbol", "path": "fak/internal/agent/adapters_test.go", "name": "TestProviderAdaptersOmitToolChoiceWithoutTools"},
            {"type": "go_symbol", "path": "fak/internal/agent/adapters_test.go", "name": "TestProviderAdaptersParseToolCalls"},
            {"type": "go_symbol", "path": "fak/internal/agent/adapters_test.go", "name": "TestHTTPPlannerUsesProviderAdapterAndPreSendQuarantine"},
            {"type": "source_token", "path": "fak/internal/agent/adapters_test.go", "token": "ProviderOpenAI"},
            {"type": "source_token", "path": "fak/internal/agent/adapters_test.go", "token": "ProviderXAI"},
            {"type": "source_token", "path": "fak/internal/agent/adapters_test.go", "token": "ProviderAnthropic"},
            {"type": "source_token", "path": "fak/internal/agent/adapters_test.go", "token": "ProviderGemini"},
            {"type": "source_token", "path": "fak/internal/agent/adapters_test.go", "token": "_quarantined"},
        ],
    },
    {
        "id": "provider_proxy_end_to_end",
        "claim": "An OpenAI-compatible client can call FAK while FAK fronts each covered upstream provider wire and filters proposed tool calls.",
        "required": True,
        "providers": EXPECTED_PROVIDERS,
        "cwd": "fak",
        "command": "go test ./internal/gateway -run 'TestChatProxyProviderAdaptersEndToEnd$'",
        "argv": ["go", "test", "./internal/gateway", "-run", "TestChatProxyProviderAdaptersEndToEnd$", "-count=1"],
        "checks": [
            {"type": "go_symbol", "path": "fak/internal/gateway/gateway_test.go", "name": "TestChatProxyProviderAdaptersEndToEnd"},
            {"type": "source_token", "path": "fak/internal/gateway/gateway_test.go", "token": "/v1/chat/completions"},
            {"type": "source_token", "path": "fak/internal/gateway/gateway_test.go", "token": "openai"},
            {"type": "source_token", "path": "fak/internal/gateway/gateway_test.go", "token": "xai"},
            {"type": "source_token", "path": "fak/internal/gateway/gateway_test.go", "token": "anthropic"},
            {"type": "source_token", "path": "fak/internal/gateway/gateway_test.go", "token": "gemini"},
            {"type": "source_token", "path": "fak/internal/gateway/gateway_test.go", "token": '{"redacted":true}'},
        ],
    },
    {
        "id": "host_agnostic_openai_compatible",
        "claim": "Arbitrary OpenAI-compatible API hosts can sit behind FAK using compatible aliases, configured base URLs, opaque model ids, optional auth, ignored vendor extension fields, and stream=true client requests synthesized only after adjudication.",
        "required": True,
        "providers": ["openai-compatible"],
        "cwd": "fak",
        "command": "go test ./internal/gateway -run 'TestChatProxyOpenAICompatible(AliasIsHostAgnostic|ObjectArgumentsAreHostAgnostic|StreamModeStreamsAdjudicatedCalls)$'",
        "argv": ["go", "test", "./internal/gateway", "-run", "TestChatProxyOpenAICompatible(AliasIsHostAgnostic|ObjectArgumentsAreHostAgnostic|StreamModeStreamsAdjudicatedCalls)$", "-count=1"],
        "checks": [
            {"type": "go_symbol", "path": "fak/internal/gateway/gateway_test.go", "name": "TestChatProxyOpenAICompatibleAliasIsHostAgnostic"},
            {"type": "go_symbol", "path": "fak/internal/gateway/gateway_test.go", "name": "TestChatProxyOpenAICompatibleObjectArgumentsAreHostAgnostic"},
            {"type": "go_symbol", "path": "fak/internal/gateway/gateway_test.go", "name": "TestChatProxyOpenAICompatibleStreamModeStreamsAdjudicatedCalls"},
            {"type": "source_token", "path": "fak/internal/gateway/gateway_test.go", "token": "openai-compatible"},
            {"type": "source_token", "path": "fak/internal/gateway/gateway_test.go", "token": "chat-completions"},
            {"type": "source_token", "path": "fak/internal/gateway/gateway_test.go", "token": "empty-provider-default-trailing-base"},
            {"type": "source_token", "path": "fak/internal/gateway/gateway_test.go", "token": "/tenant/acme/openai"},
            {"type": "source_token", "path": "fak/internal/gateway/gateway_test.go", "token": "/gateway/compat/"},
            {"type": "source_token", "path": "fak/internal/gateway/gateway_test.go", "token": "/v42/compatible"},
            {"type": "source_token", "path": "fak/internal/gateway/gateway_test.go", "token": "opaque-host/model:v1"},
            {"type": "source_token", "path": "fak/internal/gateway/gateway_test.go", "token": "opaque-object-args:model"},
            {"type": "source_token", "path": "fak/internal/gateway/gateway_test.go", "token": "vendor_extra"},
            {"type": "source_token", "path": "fak/internal/gateway/gateway_test.go", "token": "gateway must not ask upstream for raw streaming deltas"},
            {"type": "source_token", "path": "fak/internal/gateway/gateway_test.go", "token": "denied tool call reached streamed delta"},
            {"type": "source_token", "path": "fak/internal/gateway/gateway_test.go", "token": "text/event-stream"},
            {"type": "source_token", "path": "fak/internal/gateway/gateway_test.go", "token": '{"redacted":true}'},
        ],
    },
    {
        "id": "openai_compatible_host_profiles",
        "claim": "OpenAI-compatible host-profile drift is covered across null arguments, legacy function_call, typed content parts, extra fields, omitted tool_choice without advertised tools, rogue tool calls, multichoice responses, and content-only replies.",
        "required": True,
        "providers": ["openai-compatible"],
        "cwd": "fak",
        "command": "go test ./internal/gateway -run 'TestChatProxyOpenAICompatibleHostProfileConformance$'",
        "argv": ["go", "test", "./internal/gateway", "-run", "TestChatProxyOpenAICompatibleHostProfileConformance$", "-count=1"],
        "checks": [
            {"type": "go_symbol", "path": "fak/internal/gateway/gateway_test.go", "name": "TestChatProxyOpenAICompatibleHostProfileConformance"},
            {"type": "source_token", "path": "fak/internal/gateway/gateway_test.go", "token": "profile-null-arguments-extra-fields"},
            {"type": "source_token", "path": "fak/internal/gateway/gateway_test.go", "token": "profile-legacy-function-call"},
            {"type": "source_token", "path": "fak/internal/gateway/gateway_test.go", "token": "profile-content-parts-with-tool-call"},
            {"type": "source_token", "path": "fak/internal/gateway/gateway_test.go", "token": "profile-no-tools-rogue-tool-call"},
            {"type": "source_token", "path": "fak/internal/gateway/gateway_test.go", "token": "tool_choice must be omitted when no tools are sent"},
            {"type": "source_token", "path": "fak/internal/gateway/gateway_test.go", "token": "profile-multichoice-mixed-arguments"},
            {"type": "source_token", "path": "fak/internal/gateway/gateway_test.go", "token": "profile-content-only-no-fak-extension"},
            {"type": "source_token", "path": "fak/internal/gateway/gateway_test.go", "token": "\"arguments\":null"},
            {"type": "source_token", "path": "fak/internal/gateway/gateway_test.go", "token": "\"finish_reason\":\"function_call\""},
            {"type": "source_token", "path": "fak/internal/gateway/gateway_test.go", "token": "client part one"},
            {"type": "source_token", "path": "fak/internal/gateway/gateway_test.go", "token": "profile_extra"},
            {"type": "source_token", "path": "fak/internal/gateway/gateway_test.go", "token": "second choice must not leak"},
        ],
    },
    {
        "id": "direct_http_syscall",
        "claim": "Any-language clients can bypass provider quirks and call the kernel over native HTTP.",
        "required": True,
        "cwd": "fak",
        "command": "go test ./internal/gateway -run 'Test(HTTPSyscallAllow|HTTPAdjudicateTransformRepairsArgs|HTTPQuarantineSurfaced|HTTPSyscallWitnessFailsClosed)$'",
        "argv": ["go", "test", "./internal/gateway", "-run", "Test(HTTPSyscallAllow|HTTPAdjudicateTransformRepairsArgs|HTTPQuarantineSurfaced|HTTPSyscallWitnessFailsClosed)$", "-count=1"],
        "checks": [
            {"type": "go_symbol", "path": "fak/internal/gateway/gateway_test.go", "name": "TestHTTPSyscallAllow"},
            {"type": "go_symbol", "path": "fak/internal/gateway/gateway_test.go", "name": "TestHTTPAdjudicateTransformRepairsArgs"},
            {"type": "go_symbol", "path": "fak/internal/gateway/gateway_test.go", "name": "TestHTTPQuarantineSurfaced"},
            {"type": "go_symbol", "path": "fak/internal/gateway/gateway_test.go", "name": "TestHTTPSyscallWitnessFailsClosed"},
        ],
    },
    {
        "id": "direct_mcp_syscall",
        "claim": "Any-language clients can call the same kernel over MCP stdio or MCP-over-HTTP.",
        "required": True,
        "cwd": "fak",
        "command": "go test ./internal/gateway -run 'Test(MCPStdioRoundtrip|MCPOverHTTP)$'",
        "argv": ["go", "test", "./internal/gateway", "-run", "Test(MCPStdioRoundtrip|MCPOverHTTP)$", "-count=1"],
        "checks": [
            {"type": "go_symbol", "path": "fak/internal/gateway/gateway_test.go", "name": "TestMCPStdioRoundtrip"},
            {"type": "go_symbol", "path": "fak/internal/gateway/gateway_test.go", "name": "TestMCPOverHTTP"},
        ],
    },
    {
        "id": "dos_style_turnbench",
        "claim": "DOS-style proof remains available at the bridge: safety, vDSO path swap, and no fake turn-tax win.",
        "required": True,
        "cwd": "fak",
        "command": "go test ./internal/turnbench -run 'TestRun_(AirlineClassesAreLiveKernelEvents|VDSOAblationIsARealPathSwap|HappyPathSavesNothing)$'",
        "argv": ["go", "test", "./internal/turnbench", "-run", "TestRun_(AirlineClassesAreLiveKernelEvents|VDSOAblationIsARealPathSwap|HappyPathSavesNothing)$", "-count=1"],
        "checks": [
            {"type": "go_symbol", "path": "fak/internal/turnbench/turnbench_test.go", "name": "TestRun_AirlineClassesAreLiveKernelEvents"},
            {"type": "go_symbol", "path": "fak/internal/turnbench/turnbench_test.go", "name": "TestRun_VDSOAblationIsARealPathSwap"},
            {"type": "go_symbol", "path": "fak/internal/turnbench/turnbench_test.go", "name": "TestRun_HappyPathSavesNothing"},
        ],
    },
    {
        "id": "live_api_host_inventory",
        "claim": "Committed live artifacts include a Gemini OpenAI-compatible success, local OpenAI-compatible shims, and typed auth/billing blockers.",
        "required": False,
        "command": "python tools/api_host_live_inventory.py --out fak/experiments/api-host-bridge/api-host-live-inventory.json --markdown fak/experiments/api-host-bridge/api-host-live-inventory.md",
        "checks": [
            {"type": "file_exists", "path": "tools/api_host_live_inventory.py"},
            {"type": "file_exists", "path": "tools/api_host_live_inventory_test.py"},
        ],
    },
    {
        "id": "current_api_host_readiness",
        "claim": "Current compatible-host /models probes classify reachable, missing-auth, auth, billing, access-denied, and transport states without running chat completions.",
        "required": False,
        "command": "python tools/api_host_readiness_probe.py --out fak/experiments/api-host-bridge/api-host-readiness.json --markdown fak/experiments/api-host-bridge/api-host-readiness.md",
        "checks": [
            {"type": "file_exists", "path": "tools/api_host_readiness_probe.py"},
            {"type": "file_exists", "path": "tools/api_host_readiness_probe_test.py"},
            {"type": "file_exists", "path": "fak/experiments/api-host-bridge/api-host-readiness.json"},
            {"type": "source_token", "path": "fak/experiments/api-host-bridge/api-host-readiness.json", "token": "\"readiness_gate\": true"},
        ],
    },
]


def read_text(root: Path, rel_path: str) -> str:
    return (root / rel_path).read_text(encoding="utf-8-sig")


def line_for(body: str, offset: int) -> int:
    return body.count("\n", 0, offset) + 1


def resolve_check(root: Path, check: dict[str, str]) -> dict[str, Any]:
    path = check["path"]
    full = root / path
    out: dict[str, Any] = {"type": check["type"], "path": path}
    if "name" in check:
        out["name"] = check["name"]
    if "token" in check:
        out["token"] = check["token"]
    if not full.exists():
        return {**out, "status": "missing_file"}
    if check["type"] == "file_exists":
        return {**out, "status": "resolved"}
    body = read_text(root, path)
    if check["type"] == "go_symbol":
        match = re.search(rf"(?m)^func\s+{re.escape(check['name'])}\s*\(", body)
        if not match:
            return {**out, "status": "missing_symbol"}
        return {**out, "status": "resolved", "line": line_for(body, match.start())}
    if check["type"] == "source_token":
        offset = body.find(check["token"])
        if offset < 0:
            return {**out, "status": "missing_token"}
        return {**out, "status": "resolved", "line": line_for(body, offset)}
    return {**out, "status": "unknown_check_type"}


def resolve_witness(root: Path, witness: dict[str, Any]) -> dict[str, Any]:
    evidence = [resolve_check(root, check) for check in witness["checks"]]
    ok = all(item["status"] == "resolved" for item in evidence)
    return {
        "version": fleet_version.app_version(),
        "id": witness["id"],
        "claim": witness["claim"],
        "required": witness["required"],
        "status": "resolved" if ok else ("missing_required" if witness["required"] else "missing_supporting"),
        "command": witness["command"],
        "cwd": witness.get("cwd", ""),
        "argv": witness.get("argv", []),
        "providers": witness.get("providers", []),
        "evidence": evidence,
    }


def build_report(root: Path | None = None) -> dict[str, Any]:
    root = root or ROOT
    app_ver = fleet_version.app_version(root)
    witnesses = [resolve_witness(root, item) for item in WITNESSES]
    required = [w for w in witnesses if w["required"]]
    resolved = [w for w in required if w["status"] == "resolved"]
    providers = sorted({p for w in witnesses for p in w["providers"]})
    return {
        "schema": SCHEMA,
        "app_version": app_ver,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "bridge_claim": "A model can stay behind any compatible API host while FAK/DOS owns the tool-call boundary.",
        "summary": {
            "required_witnesses": len(required),
            "resolved_required_witnesses": len(resolved),
            "bridge_covered": len(required) == len(resolved),
            "provider_shapes_covered": providers,
        },
        "witnesses": witnesses,
    }


def markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# API-Host Bridge Matrix",
        "",
        f"- Required witnesses resolved: {s['resolved_required_witnesses']}/{s['required_witnesses']}",
        f"- Bridge covered: {'yes' if s['bridge_covered'] else 'no'}",
        f"- Provider shapes covered: {', '.join(s['provider_shapes_covered'])}",
        "",
        "| witness | required | status | command |",
        "|---|:---:|---|---|",
    ]
    for w in report["witnesses"]:
        lines.append(f"| `{w['id']}` | {'yes' if w['required'] else 'no'} | {w['status']} | `{w['command']}` |")
    lines.append("")
    return "\n".join(lines)


def write_text(path: str, body: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Resolve API-host bridge witnesses")
    ap.add_argument("--out", default="")
    ap.add_argument("--markdown", default="")
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args(argv)
    report = build_report(Path(args.root))
    body = json.dumps(report, indent=2) + "\n"
    if args.out:
        write_text(args.out, body)
    else:
        print(body, end="")
    if args.markdown:
        write_text(args.markdown, markdown(report))
    return 0 if report["summary"]["bridge_covered"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
