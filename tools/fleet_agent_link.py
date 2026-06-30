#!/usr/bin/env python3
"""Fleet Agent Link stdio JSON-RPC adapter.

This module is intentionally small: it exposes reviewed Fleet method names over
JSON-RPC and maps those methods to local in-process handlers. It is safe to run
through SSH/Tailscale SSH because it has no generic shell execution method.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform as host_platform
from dataclasses import dataclass
from pathlib import Path
import shlex
import shutil
import subprocess
from dispatch_worker import install_no_window_subprocess_defaults
import sys
import time
from typing import Any, Callable
install_no_window_subprocess_defaults(subprocess)


JSONRPC = "2.0"
CALL_SCHEMA = "fleet.call.v1"
RESULT_SCHEMA = "fleet.result.v1"
LINK_SCHEMA = "fleet.agent-link.v1"
A2A_PROTOCOL_VERSION = "1.0"
A2A_DEFAULT_URL = "https://fleet.example.com/a2a"
DEFAULT_TIMEOUT_S = 7200
OUTPUT_TAIL_CHARS = 20000
STATUS_SAMPLE_LIMIT = 20

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

Handler = Callable[[dict[str, Any], Path], dict[str, Any]]


class JsonRpcFault(Exception):
    """A JSON-RPC error that should be returned to the caller."""

    def __init__(self, code: int, message: str, data: Any | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


@dataclass(frozen=True)
class MethodSpec:
    name: str
    scope: str
    description: str
    handler: Handler

    def public(self) -> dict[str, str]:
        return {
            "name": self.name,
            "scope": self.scope,
            "description": self.description,
        }


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def tail_text(text: str, limit: int = OUTPUT_TAIL_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def git_output(root: Path, *args: str) -> tuple[bool, str]:
    try:
        cp = subprocess.run(
            ("git", *args),
            cwd=str(root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, ""
    return cp.returncode == 0, cp.stdout.strip()


def host_report(root: Path) -> dict[str, Any]:
    return {
        "node": host_platform.node() or "unknown",
        "system": host_platform.system() or "unknown",
        "release": host_platform.release() or "unknown",
        "machine": host_platform.machine() or "unknown",
        "python": sys.executable or "unknown",
        "repo_root": str(root),
    }


def repo_report(root: Path) -> dict[str, Any]:
    ok_head, head = git_output(root, "rev-parse", "HEAD")
    ok_branch, branch = git_output(root, "branch", "--show-current")
    if not branch:
        ok_branch, branch = git_output(root, "rev-parse", "--abbrev-ref", "HEAD")
    ok_status, status = git_output(root, "status", "--porcelain", "--untracked-files=all")
    ok_upstream, upstream = git_output(root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    status_lines = [line for line in status.splitlines() if line]
    return {
        "git_available": bool(ok_head and ok_status),
        "head": head or "unknown",
        "branch": branch or ("unknown" if ok_branch else "unknown"),
        "upstream": upstream if ok_upstream else "",
        "dirty": bool(status_lines),
        "status_count": len(status_lines) if ok_status else -1,
        "status_sample": status_lines[:STATUS_SAMPLE_LIMIT] if ok_status else [],
        "status_truncated": len(status_lines) > STATUS_SAMPLE_LIMIT if ok_status else False,
    }


def tool_report() -> dict[str, str]:
    names = ("git", "python", "python3", "py", "powershell", "pwsh", "wsl.exe", "tailscale", "nvidia-smi")
    return {name: shutil.which(name) or "" for name in names}


def bool_param(params: dict[str, Any], key: str, default: bool = False) -> bool:
    value = params.get(key, default)
    if not isinstance(value, bool):
        raise JsonRpcFault(INVALID_PARAMS, f"params.{key} must be boolean")
    return value


def str_param(params: dict[str, Any], key: str, default: str = "") -> str:
    value = params.get(key, default)
    if not isinstance(value, str):
        raise JsonRpcFault(INVALID_PARAMS, f"params.{key} must be a string")
    return value


def timeout_param(params: dict[str, Any]) -> int:
    value = params.get("timeout_s", DEFAULT_TIMEOUT_S)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise JsonRpcFault(INVALID_PARAMS, "params.timeout_s must be a positive integer")
    return value


def call_envelope(method: str, params: dict[str, Any], spec: MethodSpec, root: Path) -> dict[str, Any]:
    return {
        "schema": CALL_SCHEMA,
        "id": params.get("call_id") if isinstance(params.get("call_id"), str) else "",
        "generated_utc": utc_now(),
        "target": {"checkout": str(root), "machine": host_platform.node() or "unknown"},
        "method": method,
        "policy": {"scope": spec.scope, "requires_confirmation": spec.scope == "act"},
    }


def result_envelope(
    method: str,
    params: dict[str, Any],
    spec: MethodSpec,
    root: Path,
    status: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA,
        "link_schema": LINK_SCHEMA,
        "generated_utc": utc_now(),
        "call": call_envelope(method, params, spec, root),
        "status": status,
        "result": result,
    }


def manifest(registry: dict[str, MethodSpec]) -> dict[str, Any]:
    return {
        "schema": LINK_SCHEMA,
        "jsonrpc": JSONRPC,
        "transport": "stdio-jsonrpc",
        "methods": [registry[name].public() for name in sorted(registry)],
        "policy_scopes": {
            "read": "Non-mutating inspection/status methods.",
            "act": "Methods that run proof lanes, launch sessions, or change local process state.",
        },
        "adapters": {
            "in_memory": True,
            "stdio_jsonrpc": True,
            "a2a": "agent-card-projection-only",
            "a2a_http": "planned-edge-adapter",
            "mcp": "planned-tool-context-adapter",
        },
        "non_goals": ["generic_exec", "os_kernel_protocol", "always_on_daemon_required"],
    }


def canonical_digest(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def fleet_version(root: Path) -> str:
    version_file = root / "VERSION"
    try:
        version = version_file.read_text(encoding="utf-8").strip()
    except OSError:
        return "0.0.0-local"
    return version or "0.0.0-local"


def skill_id(method: str) -> str:
    safe = "".join(char if char.isalnum() else "_" for char in method.lower()).strip("_")
    return "fleet_" + safe


def skill_name(method: str) -> str:
    return " ".join(part.capitalize() for part in method.replace(".", " ").replace("_", " ").split())


def a2a_agent_card(
    registry: dict[str, MethodSpec],
    root: Path,
    *,
    url: str = A2A_DEFAULT_URL,
    name: str = "Fleet Coordinator",
    description: str = "Policy-filtered Fleet control-plane agent.",
    tenant: str = "",
    protocol_binding: str = "HTTP+JSON",
    protocol_version: str = A2A_PROTOCOL_VERSION,
    api_key_header: str = "",
    allowed_methods: set[str] | None = None,
    allowed_scopes: set[str] | None = None,
) -> dict[str, Any]:
    fleet_manifest = manifest(registry)
    manifest_digest = canonical_digest(fleet_manifest)
    interface: dict[str, str] = {
        "url": url,
        "protocolBinding": protocol_binding,
        "protocolVersion": protocol_version,
    }
    if tenant:
        interface["tenant"] = tenant

    card: dict[str, Any] = {
        "name": name,
        "description": description,
        "supportedInterfaces": [interface],
        "version": fleet_version(root),
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "extendedAgentCard": False,
        },
        "defaultInputModes": ["application/json", "text/plain"],
        "defaultOutputModes": ["application/json"],
        "skills": [],
    }
    if api_key_header:
        card["securitySchemes"] = {
            "fleetApiKey": {
                "apiKeySecurityScheme": {
                    "name": api_key_header,
                    "in": "header",
                }
            }
        }
        card["security"] = [{"fleetApiKey": []}]

    for method_name in sorted(registry):
        spec = registry[method_name]
        if allowed_methods is not None and method_name not in allowed_methods:
            continue
        if allowed_scopes is not None and spec.scope not in allowed_scopes:
            continue
        family = method_name.split(".", 1)[0]
        card["skills"].append(
            {
                "id": skill_id(method_name),
                "name": skill_name(method_name),
                "description": spec.description,
                "tags": ["fleet", "agent-link", family, spec.scope],
                "inputModes": ["application/json"],
                "outputModes": ["application/json"],
                "metadata": {
                    "fleet_method": method_name,
                    "fleet_policy_scope": spec.scope,
                    "fleet_requires_confirmation": spec.scope == "act",
                    "fleet_manifest_schema": LINK_SCHEMA,
                    "fleet_manifest_digest": manifest_digest,
                    "fleet_evidence_method": "protocol.manifest",
                },
            }
        )
    return card


def lint_check(checks: list[dict[str, str]], passed: bool, name: str, message: str) -> None:
    checks.append({"name": name, "status": "pass" if passed else "fail", "message": message})


def lint_a2a_agent_card(
    card: Any,
    registry: dict[str, MethodSpec] | None = None,
    *,
    require_signature: bool = False,
    require_auth: bool = False,
) -> dict[str, Any]:
    checks: list[dict[str, str]] = []
    if not isinstance(card, dict):
        return {
            "schema": "fleet.a2a-card-lint.v1",
            "ok": False,
            "checks": [{"name": "card_object", "status": "fail", "message": "Agent Card must be a JSON object."}],
        }

    registry = registry or build_registry()
    fleet_manifest = manifest(registry)
    manifest_digest = canonical_digest(fleet_manifest)
    registry_by_name = {method["name"]: method for method in fleet_manifest["methods"]}
    non_goal_blob = json.dumps(card, sort_keys=True).lower()

    lint_check(checks, isinstance(card.get("name"), str) and bool(card["name"]), "card_name", "Agent Card has a name.")
    interfaces = card.get("supportedInterfaces")
    lint_check(
        checks,
        isinstance(interfaces, list) and bool(interfaces),
        "supported_interfaces_present",
        "Agent Card declares at least one supported interface.",
    )
    if isinstance(interfaces, list):
        for index, interface in enumerate(interfaces):
            ok = (
                isinstance(interface, dict)
                and isinstance(interface.get("url"), str)
                and bool(interface.get("url"))
                and isinstance(interface.get("protocolBinding"), str)
                and bool(interface.get("protocolBinding"))
                and isinstance(interface.get("protocolVersion"), str)
                and bool(interface.get("protocolVersion"))
            )
            lint_check(checks, ok, f"supported_interface_{index}", "Interface has url, protocolBinding, and protocolVersion.")
            if isinstance(interface, dict) and "tenant" in interface:
                tenant_ok = isinstance(interface["tenant"], str) and bool(interface["tenant"])
                lint_check(checks, tenant_ok, f"supported_interface_{index}_tenant", "Tenant is a non-empty string when present.")

    has_security = isinstance(card.get("securitySchemes"), dict) and bool(card.get("securitySchemes")) and isinstance(card.get("security"), list) and bool(card.get("security"))
    if require_auth:
        lint_check(checks, has_security, "auth_required", "Card declares securitySchemes and security.")
    elif "security" in card or "securitySchemes" in card:
        lint_check(checks, has_security, "auth_consistency", "Security declarations include both securitySchemes and security.")

    has_signature = isinstance(card.get("signatures"), list) and bool(card.get("signatures"))
    if require_signature:
        lint_check(checks, has_signature, "signature_required", "Card has at least one signature.")

    skills = card.get("skills")
    lint_check(checks, isinstance(skills, list) and bool(skills), "skills_present", "Agent Card declares at least one skill.")
    skill_ids: set[str] = set()
    advertised_methods: set[str] = set()
    if isinstance(skills, list):
        for index, skill in enumerate(skills):
            if not isinstance(skill, dict):
                lint_check(checks, False, f"skill_{index}_object", "Skill must be a JSON object.")
                continue
            sid = skill.get("id")
            lint_check(checks, isinstance(sid, str) and bool(sid), f"skill_{index}_id", "Skill has an id.")
            if isinstance(sid, str):
                lint_check(checks, sid not in skill_ids, f"skill_{index}_unique_id", "Skill id is unique.")
                skill_ids.add(sid)
            metadata = skill.get("metadata")
            metadata_ok = isinstance(metadata, dict)
            lint_check(checks, metadata_ok, f"skill_{index}_metadata", "Skill has Fleet metadata.")
            if not metadata_ok:
                continue
            method_name = metadata.get("fleet_method")
            scope = metadata.get("fleet_policy_scope")
            known = isinstance(method_name, str) and method_name in registry_by_name
            lint_check(checks, known, f"skill_{index}_method_known", "Skill maps to a registered Fleet method.")
            if known:
                advertised_methods.add(method_name)
                expected_scope = registry_by_name[method_name]["scope"]
                lint_check(checks, scope == expected_scope, f"skill_{index}_scope", "Skill policy scope matches the registry.")
            lint_check(checks, metadata.get("fleet_manifest_schema") == LINK_SCHEMA, f"skill_{index}_schema", "Skill names the Fleet manifest schema.")
            lint_check(checks, metadata.get("fleet_manifest_digest") == manifest_digest, f"skill_{index}_digest", "Skill digest matches the current Fleet manifest.")

    for non_goal in fleet_manifest["non_goals"]:
        lint_check(checks, non_goal.lower() not in non_goal_blob, f"non_goal_{non_goal}", f"Card does not advertise {non_goal}.")

    return {
        "schema": "fleet.a2a-card-lint.v1",
        "ok": all(check["status"] == "pass" for check in checks),
        "advertised_methods": sorted(advertised_methods),
        "checks": checks,
    }


def handle_agent_ping(params: dict[str, Any], root: Path) -> dict[str, Any]:
    return {
        "ok": True,
        "monotonic_ms": int(time.monotonic() * 1000),
        "host": host_report(root),
    }


def handle_agent_info(params: dict[str, Any], root: Path) -> dict[str, Any]:
    registry = build_registry()
    return {
        "host": host_report(root),
        "repo": repo_report(root),
        "tools": tool_report(),
        "manifest": manifest(registry),
    }


def handle_protocol_manifest(params: dict[str, Any], root: Path) -> dict[str, Any]:
    return manifest(build_registry())


def run_command(argv: list[str], cwd: Path, timeout_s: int) -> dict[str, Any]:
    start = time.monotonic()
    try:
        cp = subprocess.run(
            argv,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )
        return {
            "argv": argv,
            "cwd": str(cwd),
            "exit_code": cp.returncode,
            "duration_ms": int((time.monotonic() - start) * 1000),
            "stdout_tail": tail_text(cp.stdout),
            "stderr_tail": tail_text(cp.stderr),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return {
            "argv": argv,
            "cwd": str(cwd),
            "exit_code": 124,
            "duration_ms": int((time.monotonic() - start) * 1000),
            "stdout_tail": tail_text(stdout),
            "stderr_tail": tail_text(stderr or f"command timed out after {timeout_s}s"),
            "timed_out": True,
        }
    except OSError as exc:
        return {
            "argv": argv,
            "cwd": str(cwd),
            "exit_code": 127,
            "duration_ms": int((time.monotonic() - start) * 1000),
            "stdout_tail": "",
            "stderr_tail": str(exc),
            "timed_out": False,
        }


def laptop_runner(root: Path) -> list[str]:
    return [sys.executable, str(root / "tools" / "fak_laptop_test.py")]


def append_common_laptop_flags(argv: list[str], params: dict[str, Any]) -> None:
    if bool_param(params, "cpu_only"):
        argv.append("--cpu-only")
    if bool_param(params, "full_cpu"):
        argv.append("--full-cpu")
    if bool_param(params, "fast"):
        argv.append("--fast")
    wsl_distro = str_param(params, "wsl_distro")
    if wsl_distro:
        argv.extend(["--wsl-distro", wsl_distro])
    check_report = str_param(params, "check_report")
    if check_report:
        argv.extend(["--check-report", check_report])
    run_report = str_param(params, "run_report")
    if run_report:
        argv.extend(["--run-report", run_report])


def laptop_argv(method: str, params: dict[str, Any], root: Path) -> list[str]:
    lane = method.split(".", 1)[1]
    argv = [*laptop_runner(root), lane]

    if method == "laptop.check":
        if bool_param(params, "require_nvidia"):
            argv.append("--require-nvidia")
        if bool_param(params, "require_cuda_toolchain"):
            argv.append("--require-cuda-toolchain")
        if bool_param(params, "cpu_only"):
            argv.append("--cpu-only")
        if bool_param(params, "fast"):
            argv.append("--fast")
        wsl_distro = str_param(params, "wsl_distro")
        if wsl_distro:
            argv.extend(["--wsl-distro", wsl_distro])
        out = str_param(params, "out")
        if out:
            argv.extend(["--out", out])
        return argv

    append_common_laptop_flags(argv, params)
    if bool_param(params, "allow_stale_repo"):
        argv.append("--allow-stale-repo")
    if method == "laptop.accept":
        precheck_report = str_param(params, "precheck_report")
        if precheck_report:
            argv.extend(["--precheck-report", precheck_report])
    return argv


def handle_laptop(method: str) -> Handler:
    def _handler(params: dict[str, Any], root: Path) -> dict[str, Any]:
        command = run_command(laptop_argv(method, params, root), root, timeout_param(params))
        return {
            "ok": command["exit_code"] == 0,
            "command": command,
        }

    return _handler


def build_registry() -> dict[str, MethodSpec]:
    return {
        "agent.info": MethodSpec("agent.info", "read", "Return host, repo, tool, and method metadata.", handle_agent_info),
        "agent.ping": MethodSpec("agent.ping", "read", "Cheap liveness check.", handle_agent_ping),
        "protocol.manifest": MethodSpec(
            "protocol.manifest",
            "read",
            "Return the Fleet Agent Link method manifest.",
            handle_protocol_manifest,
        ),
        "laptop.check": MethodSpec(
            "laptop.check",
            "act",
            "Run tools/fak_laptop_test.py check with reviewed parameters.",
            handle_laptop("laptop.check"),
        ),
        "laptop.status": MethodSpec(
            "laptop.status",
            "read",
            "Run tools/fak_laptop_test.py status against saved proof reports.",
            handle_laptop("laptop.status"),
        ),
        "laptop.verify": MethodSpec(
            "laptop.verify",
            "act",
            "Run tools/fak_laptop_test.py verify against saved proof reports.",
            handle_laptop("laptop.verify"),
        ),
        "laptop.accept": MethodSpec(
            "laptop.accept",
            "act",
            "Run tools/fak_laptop_test.py accept.",
            handle_laptop("laptop.accept"),
        ),
    }


def require_params(params: Any) -> dict[str, Any]:
    if params is None:
        return {}
    if not isinstance(params, dict):
        raise JsonRpcFault(INVALID_PARAMS, "params must be an object")
    return params


def dispatch(method: str, params: Any, root: Path | None = None) -> dict[str, Any]:
    params_obj = require_params(params)
    fleet_root = root or repo_root()
    registry = build_registry()
    spec = registry.get(method)
    if spec is None:
        raise JsonRpcFault(METHOD_NOT_FOUND, f"unknown method: {method}")
    result = spec.handler(params_obj, fleet_root)
    ok = result.get("ok", True) is True
    return result_envelope(method, params_obj, spec, fleet_root, "completed" if ok else "failed", result)


def read_request(text: str) -> dict[str, Any]:
    if not text.strip():
        raise JsonRpcFault(PARSE_ERROR, "empty JSON-RPC request")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise JsonRpcFault(PARSE_ERROR, "invalid JSON", {"line": exc.lineno, "column": exc.colno}) from exc
    if not isinstance(data, dict):
        raise JsonRpcFault(INVALID_REQUEST, "request must be a JSON object")
    if data.get("jsonrpc") != JSONRPC:
        raise JsonRpcFault(INVALID_REQUEST, "jsonrpc must be '2.0'")
    if not isinstance(data.get("method"), str) or not data["method"]:
        raise JsonRpcFault(INVALID_REQUEST, "method must be a non-empty string")
    return data


def jsonrpc_response(request_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": JSONRPC, "id": request_id, "result": result}


def jsonrpc_error(request_id: Any, code: int, message: str, data: Any | None = None) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": JSONRPC, "id": request_id, "error": error}


def handle_request(data: dict[str, Any], root: Path | None = None) -> dict[str, Any] | None:
    request_id = data.get("id")
    try:
        result = dispatch(data["method"], data.get("params", {}), root)
        if "id" not in data:
            return None
        return jsonrpc_response(request_id, result)
    except JsonRpcFault as exc:
        return jsonrpc_error(request_id, exc.code, exc.message, exc.data)
    except Exception as exc:  # pragma: no cover - defensive RPC boundary.
        return jsonrpc_error(request_id, INTERNAL_ERROR, "internal error", str(exc))


def handle_text(text: str, root: Path | None = None) -> str:
    request_id: Any = None
    try:
        data = read_request(text)
        request_id = data.get("id")
        response = handle_request(data, root)
    except JsonRpcFault as exc:
        response = jsonrpc_error(request_id, exc.code, exc.message, exc.data)
    return "" if response is None else json.dumps(response, sort_keys=True) + "\n"


def parse_params(raw: str) -> dict[str, Any]:
    try:
        data = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--params must be JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit("--params must decode to a JSON object")
    return data


def parse_json_document(raw: str, label: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{label} must be JSON: {exc}") from exc


def request_object(method: str, params: dict[str, Any], request_id: str | None = None) -> dict[str, Any]:
    return {
        "jsonrpc": JSONRPC,
        "id": request_id if request_id is not None else method,
        "method": method,
        "params": params,
    }


def ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def remote_command(cwd: str, shell: str) -> str:
    if shell == "powershell":
        return f"Set-Location {ps_quote(cwd)}; py -3 tools\\fleet_agent_link.py serve-once"
    return f"cd {shlex.quote(cwd)} && python3 tools/fleet_agent_link.py serve-once"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fleet Agent Link JSON-RPC stdio adapter")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("serve-once", help="read one JSON-RPC request from stdin and write one response")

    request = sub.add_parser("request", help="print a JSON-RPC request object")
    request.add_argument("method")
    request.add_argument("--params", default="{}", help="JSON object params")
    request.add_argument("--id", default=None, help="request id; defaults to method")

    call_local = sub.add_parser("call-local", help="dispatch a method in this checkout")
    call_local.add_argument("method")
    call_local.add_argument("--params", default="{}", help="JSON object params")
    call_local.add_argument("--id", default=None, help="request id; defaults to method")

    remote = sub.add_parser("remote-command", help="print remote command for SSH stdin piping")
    remote.add_argument("--cwd", required=True, help="remote Fleet checkout path")
    remote.add_argument("--shell", choices=("powershell", "posix"), default="powershell")

    card = sub.add_parser("a2a-card", help="print an A2A Agent Card projected from the Fleet method registry")
    card.add_argument("--url", default=A2A_DEFAULT_URL, help="A2A endpoint URL advertised in supportedInterfaces")
    card.add_argument("--name", default="Fleet Coordinator", help="Agent Card name")
    card.add_argument("--description", default="Policy-filtered Fleet control-plane agent.", help="Agent Card description")
    card.add_argument("--tenant", default="", help="optional tenant value advertised on the interface")
    card.add_argument("--protocol-binding", default="HTTP+JSON", help="A2A protocol binding")
    card.add_argument("--protocol-version", default=A2A_PROTOCOL_VERSION, help="A2A protocol version")
    card.add_argument("--api-key-header", default="", help="optional API key header to declare in securitySchemes")
    card.add_argument("--method", action="append", default=[], help="registered Fleet method to advertise; repeatable")
    card.add_argument("--scope", action="append", choices=("read", "act"), default=[], help="policy scope to advertise; repeatable")

    lint = sub.add_parser("a2a-lint", help="lint an A2A Agent Card against the Fleet method registry")
    lint.add_argument("--card", default="-", help="Agent Card JSON file, or '-' for stdin")
    lint.add_argument("--require-signature", action="store_true", help="fail if the card has no signatures")
    lint.add_argument("--require-auth", action="store_true", help="fail if the card has no security declaration")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(argv if argv is not None else sys.argv[1:]))
    if args.command == "serve-once":
        sys.stdout.write(handle_text(sys.stdin.read()))
        return 0
    if args.command == "request":
        print(json.dumps(request_object(args.method, parse_params(args.params), args.id), sort_keys=True))
        return 0
    if args.command == "call-local":
        request = request_object(args.method, parse_params(args.params), args.id)
        sys.stdout.write(handle_text(json.dumps(request)))
        return 0
    if args.command == "remote-command":
        print(remote_command(args.cwd, args.shell))
        return 0
    if args.command == "a2a-card":
        registry = build_registry()
        allowed_methods = set(args.method) if args.method else None
        if allowed_methods is not None:
            unknown = sorted(allowed_methods.difference(registry))
            if unknown:
                raise SystemExit(f"unknown method for --method: {', '.join(unknown)}")
        card = a2a_agent_card(
            registry,
            repo_root(),
            url=args.url,
            name=args.name,
            description=args.description,
            tenant=args.tenant,
            protocol_binding=args.protocol_binding,
            protocol_version=args.protocol_version,
            api_key_header=args.api_key_header,
            allowed_methods=allowed_methods,
            allowed_scopes=set(args.scope) if args.scope else None,
        )
        print(json.dumps(card, indent=2, sort_keys=True))
        return 0
    if args.command == "a2a-lint":
        raw = sys.stdin.read() if args.card == "-" else Path(args.card).read_text(encoding="utf-8")
        report = lint_a2a_agent_card(
            parse_json_document(raw, args.card),
            require_signature=args.require_signature,
            require_auth=args.require_auth,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["ok"] else 1
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
