#!/usr/bin/env python3
"""Summarize Qwen3.6 DGX and standalone bench readiness.

This is a driver-side audit. It does not claim a DGX or lab node is ready unless
there is imported evidence for that surface; missing live hardware remains an
explicit external gate in the report.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import qwen36_node_reports as node_reports


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "fak.qwen36-standalone-readiness.v1"


def _module_base(root: Path) -> Path:
    """Resolve the fak module base across both supported checkouts.

    The operator's private fleet superrepo nests the module under
    ``<root>/fak/`` (artifacts at ``<root>/fak/experiments/...``), while the
    standalone public checkout has the module at the repo root
    (``<root>/experiments/...``). Older defaults hard-coded the ``fak/``
    prefix, which made this audit look in a non-existent tree on the
    standalone checkout and silently report every artifact missing. Prefer
    whichever layout actually exists on disk; fall back to the module root.
    """
    if (root / "fak" / "experiments").is_dir():
        return root / "fak"
    return root


BASE = _module_base(ROOT)
DEFAULT_EXPERIMENT_DIR = BASE / "experiments" / "qwen36"
DEFAULT_NODE_REPORT_DIR = DEFAULT_EXPERIMENT_DIR / "node-reports"
DEFAULT_SLACK_HELPERS_DIR = ROOT.parent / "slack-helpers"
# experiments/dgx/ is operator-private lab infra, excluded from the public
# copy (see PUBLIC-SCRUB-POLICY.md). discover_dgx_runs() no-ops when it is absent.
DEFAULT_DGX_DIR = BASE / "experiments" / "dgx"
DEFAULT_PACKET_DIRS = [
    ROOT / "tools" / "_registry" / "qwen36-watch-packets",
    ROOT / "tools" / "_registry" / "qwen36-node-packet",
]
DEFAULT_SLACK_WORKDIR = "/srv/fleet"
DEFAULT_SLACK_STATE_FILE = "/var/lib/slack-control/state.json"
DEFAULT_SLACK_LOCK_FILE = "/var/lib/slack-control/state.json.lock"
DEFAULT_SLACK_TRANSCRIPT_FILE = "/var/lib/slack-control/state.transcript.jsonl"
SLACK_ENV_NAMES = [
    "SLACK_BOT_TOKEN",
    "SLACK_USER_TOKEN",
    "SLACK_CONTROL_USERS",
    "SLACK_CONTROL_COMMAND",
]
ACTION_PRIORITY = {
    "run_node_launcher": 10,
    "packet_delivery": 20,
    "send_packet": 30,
    "ssh_remote_start_available": 70,
    "ssh_auth": 80,
}
TAIL_CHARS = 2000
DGX_REQUIRED_ARTIFACTS = [
    "DGX_RUN.json",
    "DGX_RUNBOOK.md",
    "PREFLIGHT.static.json",
    "PREFLIGHT.endpoints.json",
    "compare.json",
    "COMPARE.md",
    "MATRIX.json",
    "GATE.json",
    "RUN_GATE.json",
]
STANDALONE_PROFILE_LABELS = {
    "mac": "macOS standalone",
    "linux-nvidia": "Linux/NVIDIA or DGX-adjacent standalone",
    "nvidia": "Windows/NVIDIA standalone",
    "vulkan": "Windows/AMD Vulkan standalone",
}

Runner = Callable[[Sequence[str], Path, Mapping[str, str], float], subprocess.CompletedProcess[str]]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def env_presence(env: Mapping[str, str], names: Sequence[str]) -> dict[str, dict[str, Any]]:
    return {
        name: {"set": bool(env.get(name)), "length": len(env.get(name, ""))}
        for name in names
    }


def bash_quote(value: str) -> str:
    return shlex.quote(value)


def slack_lock_file_for(state_file: str) -> str:
    return f"{state_file}.lock"


def slack_transcript_file_for(state_file: str) -> str:
    if "/" in state_file:
        directory, filename = state_file.rsplit("/", 1)
        prefix = f"{directory}/"
    else:
        filename = state_file
        prefix = ""
    if "." in filename:
        filename = filename.rsplit(".", 1)[0]
    return f"{prefix}{filename}.transcript.jsonl"


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"path": str(path), "parsed": False, "error": str(exc)}
    if not isinstance(data, dict):
        return {"path": str(path), "parsed": False, "error": "JSON root is not an object"}
    data = dict(data)
    data["_path"] = str(path)
    return data


def file_sha256(path: Path) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return ""


def newest(paths: Sequence[Path], limit: int) -> list[Path]:
    existing = [path for path in paths if path.exists()]
    return sorted(existing, key=lambda path: (path.stat().st_mtime_ns, path.as_posix()), reverse=True)[:max(0, limit)]


def json_artifact_sort_key(path: Path) -> tuple[str, int, str]:
    data = load_json_object(path)
    generated_at = data.get("generated_at") if data.get("parsed", True) else ""
    return (
        generated_at if isinstance(generated_at, str) else "",
        path.stat().st_mtime_ns,
        path.as_posix(),
    )


def newest_json_artifacts(paths: Sequence[Path], limit: int) -> list[Path]:
    existing = [path for path in paths if path.exists()]
    return sorted(existing, key=json_artifact_sort_key, reverse=True)[:max(0, limit)]


def discover_watch_reports(experiment_dir: Path, limit: int) -> list[Path]:
    if not experiment_dir.exists():
        return []
    paths = [path for path in experiment_dir.glob("*watch*.json") if path.is_file()]
    return newest_json_artifacts(paths, limit)


def discover_surface_smokes(experiment_dir: Path, limit: int) -> list[Path]:
    if not experiment_dir.exists():
        return []
    candidates = []
    for path in experiment_dir.glob("*.json"):
        data = load_json_object(path)
        if data.get("schema") == "fak.qwen36-surface-smoke.v1":
            candidates.append(path)
    return newest_json_artifacts(candidates, limit)


def summarize_watch_report(path: Path) -> dict[str, Any]:
    data = load_json_object(path)
    if not data.get("parsed", True):
        return data
    rows = data.get("nodes") if isinstance(data.get("nodes"), list) else []
    nodes = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        actions = row.get("next_actions") if isinstance(row.get("next_actions"), list) else []
        nodes.append({
            "node": row.get("node", ""),
            "state": row.get("state", ""),
            "status": row.get("status", ""),
            "detail": row.get("detail", ""),
            "smoke_ran": bool(row.get("smoke_ran")),
            "next_actions": actions,
        })
    packet_dispatch = data.get("packet_dispatch") if isinstance(data.get("packet_dispatch"), dict) else {}
    node_report = data.get("node_report") if isinstance(data.get("node_report"), dict) else {}
    latest_preflight = node_report.get("latest_preflight") if isinstance(node_report.get("latest_preflight"), dict) else {}
    node_report_summary = {}
    if node_report:
        node_report_summary = {
            "imported": bool(node_report.get("imported")),
            "status": node_report.get("status", ""),
            "archive": node_report.get("archive", ""),
            "report_dir": node_report.get("report_dir", ""),
            "latest_preflight": {
                "path": latest_preflight.get("path", ""),
                "parsed": latest_preflight.get("parsed"),
                "ok": latest_preflight.get("ok"),
                "profile": latest_preflight.get("profile", ""),
                "base_url": latest_preflight.get("base_url", ""),
                "llama_server_found": latest_preflight.get("llama_server_found"),
                "failed_checks": latest_preflight.get("failed_checks", []),
                "nvidia_smi": latest_preflight.get("nvidia_smi") if isinstance(latest_preflight.get("nvidia_smi"), dict) else None,
            },
        }
    return {
        "path": rel(path),
        "parsed": True,
        "schema": data.get("schema", ""),
        "generated_at": data.get("generated_at", ""),
        "summary": data.get("summary") if isinstance(data.get("summary"), dict) else {},
        "nodes": nodes,
        "packet_sent": bool(packet_dispatch.get("sent")),
        "packet_report_target": packet_dispatch.get("report_target", ""),
        "node_report": node_report_summary,
    }


def is_qwen36_model(model: Any) -> bool:
    text = str(model or "").lower().replace("-", "").replace("_", "")
    return "qwen3.6" in text or "qwen36" in text


def summarize_surface_smoke(path: Path) -> dict[str, Any]:
    data = load_json_object(path)
    if not data.get("parsed", True):
        return data
    surfaces = data.get("surfaces") if isinstance(data.get("surfaces"), list) else []
    surface_rows = []
    for row in surfaces:
        if not isinstance(row, dict):
            continue
        chat_perf = row.get("chat_perf") if isinstance(row.get("chat_perf"), dict) else {}
        surface_rows.append({
            "surface": row.get("surface", ""),
            "status": row.get("status", ""),
            "decode_tps": chat_perf.get("decode_tps"),
            "decode_vs_baseline": chat_perf.get("decode_vs_baseline"),
        })
    pass_count = len([row for row in surface_rows if row.get("status") == "PASS"])
    fail_count = len([row for row in surface_rows if row.get("status") == "FAIL"])
    planned_count = len([row for row in surface_rows if row.get("status") == "PLANNED"])
    passed = bool(surface_rows) and fail_count == 0 and pass_count == len(surface_rows)
    model = data.get("model", "")
    endpoint = data.get("endpoint") if isinstance(data.get("endpoint"), dict) else {}
    return {
        "path": rel(path),
        "parsed": True,
        "schema": data.get("schema", ""),
        "generated_at": data.get("generated_at", ""),
        "node_name": data.get("node_name", ""),
        "model": model,
        "qwen36_model": is_qwen36_model(model),
        "base_url": data.get("base_url", ""),
        "endpoint": {
            "name": endpoint.get("name", ""),
            "source": endpoint.get("source", ""),
            "state": endpoint.get("state", ""),
            "tailscale_ip": endpoint.get("tailscale_ip", ""),
            "base_url": endpoint.get("base_url", ""),
        } if endpoint else {},
        "summary": data.get("summary") if isinstance(data.get("summary"), dict) else {},
        "surfaces": surface_rows,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "planned_count": planned_count,
        "passed": passed,
    }


def discover_node_report_dirs(node_report_dir: Path, limit: int) -> list[Path]:
    if not node_report_dir.exists():
        return []
    paths = [
        path for path in node_report_dir.iterdir()
        if path.is_dir() and any(path.glob("**/preflight-*.json"))
    ]
    return newest(paths, limit)


def summarize_node_report_dir(path: Path) -> dict[str, Any]:
    summary = node_reports.summarize_dir(path)
    latest_preflight = summary.get("latest_preflight") if isinstance(summary.get("latest_preflight"), dict) else {}
    return {
        "path": rel(path),
        "status": summary.get("status", ""),
        "preflight_count": summary.get("preflight_count", 0),
        "server_log_count": summary.get("server_log_count", 0),
        "latest_preflight": {
            "path": latest_preflight.get("path", ""),
            "parsed": latest_preflight.get("parsed"),
            "ok": latest_preflight.get("ok"),
            "profile": latest_preflight.get("profile", ""),
            "base_url": latest_preflight.get("base_url", ""),
            "llama_server_found": latest_preflight.get("llama_server_found"),
            "failed_checks": latest_preflight.get("failed_checks", []),
            "nvidia_smi": latest_preflight.get("nvidia_smi") if isinstance(latest_preflight.get("nvidia_smi"), dict) else None,
        },
    }


def discover_dgx_runs(dgx_dir: Path, limit: int) -> list[Path]:
    if not dgx_dir.exists():
        return []
    run_dirs = [
        path
        for path in dgx_dir.iterdir()
        if path.is_dir() and (path / "DGX_RUN.json").exists()
    ]
    return sorted(
        run_dirs,
        key=lambda path: (*json_artifact_sort_key(path / "DGX_RUN.json"), path.as_posix()),
        reverse=True,
    )[:max(0, limit)]


def json_passed(path: Path) -> bool | None:
    data = load_json_object(path)
    if not data.get("parsed", True):
        return None
    value = data.get("passed")
    return value if isinstance(value, bool) else None


def count_endpoint_reports(run_dir: Path) -> int:
    count = 0
    for path in run_dir.glob("*.json"):
        data = load_json_object(path)
        if data.get("schema") == "fak.dgx-endpoint-bench.v1":
            count += 1
    return count


def count_monitor_reports(run_dir: Path) -> int:
    return len([path for path in (run_dir / "benchmark-monitor").glob("**/results_*.json") if path.is_file()])


def summarize_dgx_handoff(run_dir: Path) -> dict[str, Any]:
    handoff_dir = run_dir / "handoff"
    archives = newest([path for path in handoff_dir.glob("fleet-*.tgz") if path.is_file()], 1)
    archive = archives[0] if archives else None
    script = handoff_dir / "RUN_ON_DGX.sh"
    readme = handoff_dir / "DGX_HANDOFF.md"
    return {
        "path": rel(handoff_dir),
        "complete": bool(archive and script.exists() and readme.exists()),
        "archive": rel(archive) if archive else "",
        "archive_bytes": archive.stat().st_size if archive and archive.exists() else 0,
        "archive_sha256": file_sha256(archive) if archive else "",
        "script": rel(script) if script.exists() else "",
        "readme": rel(readme) if readme.exists() else "",
    }


def summarize_dgx_remote_probe(run_dir: Path) -> dict[str, Any]:
    probes = newest_json_artifacts([path for path in run_dir.glob("REMOTE_PROBE*.json") if path.is_file()], 1)
    if not probes:
        return {"path": "", "status": "MISSING", "generated_at": ""}
    path = probes[0]
    data = load_json_object(path)
    if not data.get("parsed", True):
        return {"path": rel(path), "status": "UNPARSEABLE", "generated_at": "", "error": data.get("error", "")}
    target_dns = data.get("target_dns") if isinstance(data.get("target_dns"), dict) else {}
    ssh = data.get("ssh") if isinstance(data.get("ssh"), dict) else {}
    if ssh.get("attempted") and ssh.get("returncode") == 0:
        status = "SSH_PASS"
    elif ssh.get("attempted"):
        status = "SSH_FAILED"
    elif target_dns.get("resolved") is True:
        status = "DNS_RESOLVED"
    else:
        status = "DNS_FAILED"
    return {
        "path": rel(path),
        "status": status,
        "generated_at": data.get("generated_at", ""),
        "ssh_target": data.get("ssh_target", ""),
        "proxy_jump": data.get("proxy_jump", ""),
        "target_dns_resolved": target_dns.get("resolved"),
        "target_dns_addresses": target_dns.get("addresses", []) if isinstance(target_dns.get("addresses"), list) else [],
        "target_dns_error": target_dns.get("error", ""),
        "ssh_attempted": ssh.get("attempted"),
        "ssh_returncode": ssh.get("returncode"),
        "ssh_stderr": ssh.get("stderr", ""),
    }


def summarize_dgx_run(run_dir: Path) -> dict[str, Any]:
    plan = load_json_object(run_dir / "DGX_RUN.json")
    missing = [name for name in DGX_REQUIRED_ARTIFACTS if not (run_dir / name).exists()]
    gate_passed = json_passed(run_dir / "GATE.json")
    run_gate_passed = json_passed(run_dir / "RUN_GATE.json")
    endpoint_reports = count_endpoint_reports(run_dir)
    monitor_reports = count_monitor_reports(run_dir)
    has_monitor_manifest = (
        (run_dir / "benchmark-monitor" / "_csv_manifest.json").exists()
        or (run_dir / "benchmark-monitor" / "csv" / "_csv_manifest.json").exists()
    )
    failures: list[str] = []
    if missing:
        failures.append(f"missing required artifacts: {', '.join(missing)}")
    if gate_passed is not True:
        failures.append("GATE.json passed is not true")
    if run_gate_passed is not True:
        failures.append("RUN_GATE.json passed is not true")
    if endpoint_reports < 2:
        failures.append("fewer than two endpoint bench reports")
    if monitor_reports == 0 and not has_monitor_manifest:
        failures.append("missing Benchmark monitor result or CSV manifest")

    if not (run_dir / "PREFLIGHT.static.json").exists() and not (run_dir / "PREFLIGHT.endpoints.json").exists():
        status = "PREP_ONLY"
    elif failures:
        status = "INCOMPLETE"
    else:
        status = "PASS"

    plan_obj = plan if isinstance(plan, dict) and plan.get("parsed", True) else {}
    benchmark = plan_obj.get("benchmark") if isinstance(plan_obj.get("benchmark"), dict) else {}
    return {
        "path": rel(run_dir),
        "status": status,
        "run_id": plan_obj.get("run_id") or run_dir.name,
        "generated_at": plan_obj.get("generated_at", ""),
        "model": plan_obj.get("model", ""),
        "hardware": plan_obj.get("hardware", ""),
        "benchmark_root": benchmark.get("root", ""),
        "plan_sha256": file_sha256(run_dir / "DGX_RUN.json"),
        "runbook_sha256": file_sha256(run_dir / "DGX_RUNBOOK.md"),
        "handoff": summarize_dgx_handoff(run_dir),
        "remote_probe": summarize_dgx_remote_probe(run_dir),
        "missing_artifacts": missing,
        "gate_passed": gate_passed,
        "run_gate_passed": run_gate_passed,
        "endpoint_report_count": endpoint_reports,
        "benchmark_monitor_reports": monitor_reports,
        "benchmark_monitor_manifest": has_monitor_manifest,
        "failures": failures,
    }


def summarize_packet_manifest(path: Path) -> dict[str, Any]:
    data = load_json_object(path)
    if not data.get("parsed", True):
        return {
            "path": rel(path),
            "parsed": False,
            "error": data.get("error", ""),
            "profiles": [],
            "mtime": path.stat().st_mtime if path.exists() else 0,
        }
    profiles = [str(item) for item in data.get("profiles", []) if isinstance(item, str)]
    out_dir = path.parent.parent
    archive = newest(list(out_dir.glob("qwen36-node-packet-*.zip")), 1)
    commands = {
        key: data.get(key) if isinstance(data.get(key), dict) else {}
        for key in (
            "bootstrap_command",
            "preflight_command",
            "start_command",
            "install_command",
            "report_command",
            "node_command",
        )
    }
    receive = data.get("receive") if isinstance(data.get("receive"), dict) else {}
    return {
        "path": rel(path),
        "parsed": True,
        "schema": data.get("schema", ""),
        "generated_at": data.get("generated_at", ""),
        "model": data.get("model", ""),
        "serve_port": data.get("serve_port"),
        "profiles": profiles,
        "report_target": data.get("report_target", ""),
        "payload_dir": rel(path.parent),
        "packet_dir": rel(out_dir),
        "archive": rel(archive[0]) if archive else "",
        "archive_present": bool(archive),
        "commands": commands,
        "receive": {
            profile: [str(line) for line in lines]
            for profile, lines in receive.items()
            if isinstance(profile, str) and isinstance(lines, list)
        },
        "driver_command": data.get("driver_command", ""),
        "mtime": path.stat().st_mtime if path.exists() else 0,
    }


def packet_profile_matrix(manifests: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    parsed = sorted(
        [manifest for manifest in manifests if manifest.get("parsed") is True],
        key=lambda manifest: float(manifest.get("mtime") or 0),
        reverse=True,
    )
    for profile, label in STANDALONE_PROFILE_LABELS.items():
        manifest = next(
            (row for row in parsed if profile in (row.get("profiles") if isinstance(row.get("profiles"), list) else [])),
            None,
        )
        if not manifest:
            rows.append({
                "profile": profile,
                "label": label,
                "status": "MISSING",
                "evidence": "no parsed packet manifest includes this profile",
            })
            continue
        commands = manifest.get("commands") if isinstance(manifest.get("commands"), dict) else {}
        def command(name: str) -> str:
            mapping = commands.get(name) if isinstance(commands.get(name), dict) else {}
            return str(mapping.get(profile) or "")

        receive = manifest.get("receive") if isinstance(manifest.get("receive"), dict) else {}
        status = "PREPARED" if manifest.get("archive_present") else "MANIFEST_ONLY"
        rows.append({
            "profile": profile,
            "label": label,
            "status": status,
            "evidence": manifest.get("archive") or manifest.get("path", ""),
            "manifest": manifest.get("path", ""),
            "archive": manifest.get("archive", ""),
            "packet_dir": manifest.get("packet_dir", ""),
            "payload_dir": manifest.get("payload_dir", ""),
            "generated_at": manifest.get("generated_at", ""),
            "model": manifest.get("model", ""),
            "serve_port": manifest.get("serve_port"),
            "report_target": manifest.get("report_target", ""),
            "bootstrap_command": command("bootstrap_command"),
            "preflight_command": command("preflight_command"),
            "start_command": command("start_command"),
            "install_command": command("install_command"),
            "report_command": command("report_command"),
            "node_command": command("node_command"),
            "receive": [str(line) for line in receive.get(profile, [])] if isinstance(receive.get(profile), list) else [],
            "driver_command": manifest.get("driver_command", ""),
        })
    return rows


def packet_artifacts(packet_dirs: Sequence[Path]) -> dict[str, Any]:
    archives: list[Path] = []
    manifests: list[Path] = []
    for base in packet_dirs:
        if not base.exists():
            continue
        archives.extend(path for path in base.glob("**/qwen36-node-packet-*.zip") if path.is_file())
        manifests.extend(path for path in base.glob("**/manifest.json") if path.is_file())
    latest_archives = newest(archives, 5)
    latest_manifests = newest(manifests, 5)
    manifest_summaries = [summarize_packet_manifest(path) for path in newest(manifests, 20)]
    profile_matrix = packet_profile_matrix(manifest_summaries)
    return {
        "archive_count": len(archives),
        "manifest_count": len(manifests),
        "latest_archives": [rel(path) for path in latest_archives],
        "latest_manifests": [rel(path) for path in latest_manifests],
        "profiles_prepared": len([row for row in profile_matrix if row.get("status") == "PREPARED"]),
        "profile_count": len(profile_matrix),
        "profile_matrix": profile_matrix,
    }


def action_priority(action: dict[str, Any]) -> int:
    return ACTION_PRIORITY.get(str(action.get("kind") or ""), 50)


def dedupe_actions(actions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    rows: list[dict[str, Any]] = []
    for action in actions:
        key = (
            action.get("kind"),
            action.get("detail"),
            action.get("host"),
            action.get("port"),
            tuple(action.get("bootstrap_files", []) if isinstance(action.get("bootstrap_files"), list) else []),
            action.get("packet_dir"),
            action.get("archive"),
        )
        if key in seen:
            continue
        seen.add(key)
        rows.append(dict(action))
    return rows


def prune_stale_optional_actions(actions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(actions)
    kinds = {str(action.get("kind") or "") for action in rows}
    if "ssh_remote_start_available" in kinds:
        rows = [action for action in rows if action.get("kind") != "ssh_auth"]
    return rows


def node_slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-") or "node"


def launcher_snippets(action: dict[str, Any], node: str) -> dict[str, Any]:
    bootstrap_files = action.get("bootstrap_files") if isinstance(action.get("bootstrap_files"), list) else []
    commands: list[str] = []
    recover: list[str] = []
    if any(str(name).endswith(".cmd") for name in bootstrap_files):
        commands.extend(f".\\{name}" for name in bootstrap_files if str(name).endswith(".cmd"))
    if any(str(name).endswith(".ps1") for name in bootstrap_files):
        commands.extend(
            f"powershell -NoProfile -ExecutionPolicy Bypass -File .\\{name}"
            for name in bootstrap_files
            if str(name).endswith(".ps1")
        )
    if any(str(name).endswith(".sh") for name in bootstrap_files):
        commands.extend(f"bash {name}" for name in bootstrap_files if str(name).endswith(".sh"))
    if any(str(name).endswith(".command") for name in bootstrap_files):
        commands.extend(f"bash {name}" for name in bootstrap_files if str(name).endswith(".command"))

    if any("NVIDIA" in str(name).upper() for name in bootstrap_files):
        recover = [
            "New-Item -ItemType Directory -Force $HOME\\qwen36-node-packet | Out-Null",
            "Set-Location $HOME\\qwen36-node-packet",
            "tailscale file get .",
            "Expand-Archive -Force .\\qwen36-node-packet-*.zip .",
            "Set-Location .\\qwen36-node-packet",
            ".\\RUN-NVIDIA.ps1 --preflight",
            ".\\INSTALL-NVIDIA.cmd",
        ]
    elif any("VULKAN" in str(name).upper() for name in bootstrap_files):
        recover = [
            "New-Item -ItemType Directory -Force $HOME\\qwen36-node-packet | Out-Null",
            "Set-Location $HOME\\qwen36-node-packet",
            "tailscale file get .",
            "Expand-Archive -Force .\\qwen36-node-packet-*.zip .",
            "Set-Location .\\qwen36-node-packet",
            ".\\RUN-VULKAN.ps1 --preflight",
            ".\\INSTALL-VULKAN.cmd",
        ]
    elif any("LINUX-NVIDIA" in str(name).upper() for name in bootstrap_files):
        recover = [
            "mkdir -p ~/qwen36-node-packet",
            "cd ~/qwen36-node-packet",
            "tailscale file get .",
            "unzip -o qwen36-node-packet-*.zip",
            "cd qwen36-node-packet",
            "bash RUN-LINUX-NVIDIA.sh --preflight",
            "bash INSTALL-LINUX-NVIDIA.sh",
        ]
    elif any("MAC" in str(name).upper() for name in bootstrap_files):
        recover = [
            "mkdir -p ~/qwen36-node-packet",
            "cd ~/qwen36-node-packet",
            "tailscale file get .",
            "unzip -o qwen36-node-packet-*.zip",
            "cd qwen36-node-packet",
            "bash RUN-MAC.sh --preflight",
            "bash INSTALL-MAC.command",
        ]

    verify = [
        "python tools\\qwen36_watch_nodes.py `",
        f"  --node {node} `",
        "  --gateway-chat `",
        "  --import-reports `",
        "  --perf-decode-baseline-tps 7.29 `",
        f"  --out fak\\experiments\\qwen36\\{node_slug(node)}-qwen36-watch-live.json",
    ]
    return {
        "target_commands": commands,
        "target_recover_commands": recover,
        "driver_verify_commands": verify,
    }


def send_packet_snippets(node: str) -> dict[str, Any]:
    return {
        "driver_send_commands": [
            "python tools\\qwen36_watch_nodes.py `",
            f"  --node {node} `",
            "  --send-packet `",
            "  --packet-profile auto `",
            "  --gateway-chat `",
            "  --import-reports `",
            "  --perf-decode-baseline-tps 7.29",
        ]
    }


def action_snippets(action: dict[str, Any] | None, node: str) -> dict[str, Any]:
    if not isinstance(action, dict):
        return {}
    kind = action.get("kind")
    if kind == "run_node_launcher":
        return launcher_snippets(action, node)
    if kind == "send_packet":
        return send_packet_snippets(node)
    return {}


def collect_target_next_actions(watches: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_node: dict[str, dict[str, Any]] = {}
    for watch in watches:
        source_report = str(watch.get("path") or "")
        for node in watch.get("nodes", []):
            if not isinstance(node, dict):
                continue
            node_name = str(node.get("node") or "")
            if not node_name:
                continue
            entry = by_node.setdefault(node_name, {
                "node": node_name,
                "state": node.get("state", ""),
                "status": node.get("status", ""),
                "detail": node.get("detail", ""),
                "reports": [],
                "required_actions": [],
                "optional_actions": [],
            })
            if source_report and source_report not in entry["reports"]:
                entry["reports"].append(source_report)
            for action in node.get("next_actions", []):
                if not isinstance(action, dict):
                    continue
                enriched = dict(action)
                enriched["source_report"] = source_report
                enriched["source_state"] = node.get("state", "")
                enriched["source_status"] = node.get("status", "")
                if enriched.get("required") is True:
                    entry["required_actions"].append(enriched)
                else:
                    entry["optional_actions"].append(enriched)

    rows = []
    for entry in by_node.values():
        required = sorted(dedupe_actions(entry["required_actions"]), key=action_priority)
        optional = sorted(
            prune_stale_optional_actions(dedupe_actions(entry["optional_actions"])),
            key=action_priority,
        )
        row = {
            "node": entry["node"],
            "state": entry["state"],
            "status": entry["status"],
            "detail": entry["detail"],
            "reports": entry["reports"],
            "primary_action": required[0] if required else None,
            "all_required_actions": required,
            "optional_actions": optional,
        }
        row["snippets"] = action_snippets(row["primary_action"], entry["node"])
        rows.append(row)
    return sorted(rows, key=lambda row: (0 if row.get("primary_action") else 1, row["node"]))


def target_surface_pass_nodes(surface_smokes: Sequence[dict[str, Any]]) -> set[str]:
    nodes: set[str] = set()
    for smoke in surface_smokes:
        if not (
            smoke.get("passed") is True
            and smoke.get("qwen36_model") is True
            and isinstance(smoke.get("endpoint"), dict)
            and smoke.get("endpoint")
        ):
            continue
        endpoint = smoke.get("endpoint") if isinstance(smoke.get("endpoint"), dict) else {}
        endpoint_name = str(endpoint.get("name") or "").strip()
        if endpoint_name:
            nodes.add(endpoint_name)
            continue
        node_name = str(smoke.get("node_name") or "").strip()
        if node_name:
            nodes.add(node_name)
    return nodes


def run_subprocess(
    command: Sequence[str],
    cwd: Path,
    env: Mapping[str, str],
    timeout_s: float,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=str(cwd),
        env=dict(env),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=max(1.0, timeout_s),
        check=False,
    )


def slack_helpers_status(
    slack_helpers_dir: Path,
    env: Mapping[str, str],
    *,
    run_dry_run: bool,
    run_local_demo: bool,
    run_live_probe: bool,
    channel: str,
    command_text: str,
    timeout_s: float,
    runner: Runner = run_subprocess,
) -> dict[str, Any]:
    components = {
        "readme": slack_helpers_dir / "README.md",
        "installer": slack_helpers_dir / "install-slack-control-dgx.sh",
        "cli": slack_helpers_dir / "slack_helpers" / "cli.py",
        "control": slack_helpers_dir / "slack_helpers" / "control.py",
        "local_demo": slack_helpers_dir / "examples" / "slack_control_local_demo.py",
    }
    result: dict[str, Any] = {
        "path": str(slack_helpers_dir),
        "exists": slack_helpers_dir.exists(),
        "components": {name: path.exists() for name, path in components.items()},
        "env": env_presence(env, SLACK_ENV_NAMES),
        "dry_run_requested": run_dry_run,
        "local_demo_requested": run_local_demo,
        "live_probe_requested": run_live_probe,
        "dry_run_command": [
            sys.executable,
            "-m",
            "slack_helpers.cli",
            "control",
            "--channel",
            channel,
            "--command",
            command_text,
            "--dry-run",
        ],
        "live_probe_command": [
            sys.executable,
            "-m",
            "slack_helpers.cli",
            "control",
            "--channel",
            channel,
            "--probe",
        ],
        "local_demo_command": [
            sys.executable,
            "examples/slack_control_local_demo.py",
        ],
    }
    result["complete_checkout"] = bool(result["exists"] and all(result["components"].values()))
    if run_dry_run and not result["complete_checkout"]:
        result["dry_run"] = {"ok": False, "skipped": True, "reason": "slack-helpers checkout is incomplete"}
    elif run_dry_run:
        dry_env = dict(env)
        dry_env.setdefault("SLACK_CONTROL_USERS", "UTEST")
        try:
            proc = runner(result["dry_run_command"], slack_helpers_dir, dry_env, timeout_s)
        except subprocess.TimeoutExpired as exc:
            result["dry_run"] = {
                "ok": False,
                "exit_code": 124,
                "stdout": (exc.stdout or "")[-TAIL_CHARS:] if isinstance(exc.stdout, str) else "",
                "stderr": (exc.stderr or "")[-TAIL_CHARS:] if isinstance(exc.stderr, str) else "",
                "error": f"timed out after {timeout_s:g}s",
            }
        except OSError as exc:
            result["dry_run"] = {"ok": False, "exit_code": None, "stdout": "", "stderr": "", "error": str(exc)}
        else:
            result["dry_run"] = {
                "ok": proc.returncode == 0,
                "exit_code": proc.returncode,
                "stdout": proc.stdout[-TAIL_CHARS:],
                "stderr": proc.stderr[-TAIL_CHARS:],
            }

    local_demo_path = components["local_demo"]
    if run_local_demo and not result["complete_checkout"]:
        result["local_demo"] = {"ok": False, "skipped": True, "reason": "slack-helpers checkout is incomplete"}
    elif run_local_demo and not local_demo_path.exists():
        result["local_demo"] = {"ok": False, "skipped": True, "reason": "examples/slack_control_local_demo.py is missing"}
    elif run_local_demo:
        try:
            proc = runner(result["local_demo_command"], slack_helpers_dir, env, timeout_s)
        except subprocess.TimeoutExpired as exc:
            result["local_demo"] = {
                "ok": False,
                "exit_code": 124,
                "stdout": (exc.stdout or "")[-TAIL_CHARS:] if isinstance(exc.stdout, str) else "",
                "stderr": (exc.stderr or "")[-TAIL_CHARS:] if isinstance(exc.stderr, str) else "",
                "error": f"timed out after {timeout_s:g}s",
            }
        except OSError as exc:
            result["local_demo"] = {"ok": False, "exit_code": None, "stdout": "", "stderr": "", "error": str(exc)}
        else:
            result["local_demo"] = {
                "ok": proc.returncode == 0 and "local slack-control demo: OK" in proc.stdout,
                "exit_code": proc.returncode,
                "stdout": proc.stdout[-TAIL_CHARS:],
                "stderr": proc.stderr[-TAIL_CHARS:],
            }

    env_status = result["env"]
    has_token = bool(
        env_status.get("SLACK_BOT_TOKEN", {}).get("set")
        or env_status.get("SLACK_USER_TOKEN", {}).get("set")
    )
    if run_live_probe and not result["complete_checkout"]:
        result["live_probe"] = {"ok": False, "skipped": True, "reason": "slack-helpers checkout is incomplete"}
    elif run_live_probe and not has_token:
        result["live_probe"] = {
            "ok": False,
            "skipped": True,
            "reason": "SLACK_BOT_TOKEN or SLACK_USER_TOKEN is required for live probe",
        }
    elif run_live_probe:
        try:
            proc = runner(result["live_probe_command"], slack_helpers_dir, env, timeout_s)
        except subprocess.TimeoutExpired as exc:
            result["live_probe"] = {
                "ok": False,
                "exit_code": 124,
                "stdout": (exc.stdout or "")[-TAIL_CHARS:] if isinstance(exc.stdout, str) else "",
                "stderr": (exc.stderr or "")[-TAIL_CHARS:] if isinstance(exc.stderr, str) else "",
                "error": f"timed out after {timeout_s:g}s",
            }
        except OSError as exc:
            result["live_probe"] = {"ok": False, "exit_code": None, "stdout": "", "stderr": "", "error": str(exc)}
        else:
            result["live_probe"] = {
                "ok": proc.returncode == 0,
                "exit_code": proc.returncode,
                "stdout": proc.stdout[-TAIL_CHARS:],
                "stderr": proc.stderr[-TAIL_CHARS:],
            }
    return result


def slack_next_actions(
    slack: dict[str, Any],
    *,
    channel: str,
    workdir: str,
    state_file: str,
    lock_file: str,
    transcript_file: str,
) -> dict[str, Any]:
    env = slack.get("env") if isinstance(slack.get("env"), dict) else {}
    has_token = bool(
        env.get("SLACK_BOT_TOKEN", {}).get("set")
        or env.get("SLACK_USER_TOKEN", {}).get("set")
    )
    has_users = bool(env.get("SLACK_CONTROL_USERS", {}).get("set"))
    has_command = bool(env.get("SLACK_CONTROL_COMMAND", {}).get("set"))
    missing = []
    if not has_token:
        missing.append("SLACK_BOT_TOKEN or SLACK_USER_TOKEN")
    if not has_users:
        missing.append("SLACK_CONTROL_USERS")
    if not has_command:
        missing.append("SLACK_CONTROL_COMMAND")
    if not slack.get("complete_checkout"):
        missing.append("complete slack-helpers checkout")
    live_probe = slack.get("live_probe") if isinstance(slack.get("live_probe"), dict) else {}
    live_probe_requested = bool(slack.get("live_probe_requested"))
    if live_probe.get("ok") is True:
        live_probe_status = "PASS"
    elif live_probe.get("skipped") is True:
        live_probe_status = "SKIPPED"
    elif live_probe_requested:
        live_probe_status = "FAIL"
    else:
        live_probe_status = "NOT_REQUESTED"

    command = f"bash -lc 'cd {workdir} && exec bash -li'"
    foreground_command = (
        f"python -m slack_helpers.cli control --channel {channel} "
        f"--cwd {bash_quote(workdir)} --resume --state-file {bash_quote(state_file)} "
        f"--lock-file {bash_quote(lock_file)} --transcript-file {bash_quote(transcript_file)} "
        '--command "$SLACK_CONTROL_COMMAND"'
    )
    setup_commands = [
        'export SLACK_BOT_TOKEN="xoxb-..."',
        'export SLACK_CONTROL_USERS="U12345678,U23456789"',
        f'export SLACK_CONTROL_COMMAND="{command}"',
        f"python -m slack_helpers.cli control --channel {channel} --probe",
    ]
    service_install_commands = [
        f"sudo -E bash ./install-slack-control-dgx.sh --channel {channel} --workdir {workdir} "
        f"--state-file {state_file} --lock-file {lock_file} --transcript-file {transcript_file}",
    ]
    dry_run_commands = [
        f'SLACK_BOT_TOKEN=xoxb-test SLACK_CONTROL_USERS=UTEST SLACK_CONTROL_COMMAND="python --version" '
        f"bash install-slack-control-dgx.sh --dry-run --no-start --no-probe --channel {channel} --workdir {workdir} "
        f"--state-file {state_file} --lock-file {lock_file} --transcript-file {transcript_file}",
    ]
    live_probe_commands = [
        f"python -m slack_helpers.cli control --channel {channel} --probe",
        "sudo journalctl -u slack-control -f",
        f"sudo tail -f {transcript_file}",
    ]
    if missing:
        status = "NEEDS_CONFIGURATION"
    elif live_probe_status == "PASS":
        status = "READY_FOR_SERVICE_INSTALL"
    elif live_probe_status == "FAIL":
        status = "PROBE_FAILED"
    else:
        status = "READY_FOR_LIVE_PROBE"
    return {
        "status": status,
        "channel": channel,
        "workdir": workdir,
        "state_file": state_file,
        "lock_file": lock_file,
        "transcript_file": transcript_file,
        "live_probe_status": live_probe_status,
        "missing_requirements": missing,
        "suggested_control_command": command,
        "foreground_control_command": foreground_command,
        "setup_commands": setup_commands,
        "service_install_commands": service_install_commands,
        "dry_run_commands": dry_run_commands,
        "live_probe_commands": live_probe_commands,
        "completion_evidence": [
            "slack control --probe succeeds on the DGX/lab host",
            "a live Slack thread accepts !status and reports the target process",
            "target-side Qwen/DGX evidence is imported back into fak/experiments",
        ],
    }


def check_rows(
    *,
    prep_doc: Path,
    readiness_doc: Path,
    watches: Sequence[dict[str, Any]],
    node_report_summaries: Sequence[dict[str, Any]],
    surface_smokes: Sequence[dict[str, Any]],
    dgx_runs: Sequence[dict[str, Any]],
    target_next_actions: Sequence[dict[str, Any]],
    packets: dict[str, Any],
    slack: dict[str, Any],
) -> list[dict[str, str]]:
    raw_watch_action = any(
        watch.get("summary", {}).get("action_required") is True
        for watch in watches
        if isinstance(watch.get("summary"), dict)
    )
    target_action_count = len([row for row in target_next_actions if row.get("primary_action")])
    watcher_evidence = f"{len(watches)} watch report(s) parsed; {target_action_count} unsuppressed target action node(s)"
    if raw_watch_action and target_action_count == 0 and watches:
        watcher_evidence += "; historical watch actions cleared by target evidence"
    any_watch_pass = any(
        node.get("status") == "PASS"
        for watch in watches
        for node in watch.get("nodes", [])
        if isinstance(node, dict)
    )
    passing_watch_preflight = next(
        (
            watch
            for watch in watches
            if isinstance(watch.get("node_report"), dict)
            and watch["node_report"].get("status") == "PREFLIGHT_OK"
            and isinstance(watch["node_report"].get("latest_preflight"), dict)
            and watch["node_report"]["latest_preflight"].get("ok") is True
        ),
        None,
    )
    passing_report_preflight = next(
        (
            report
            for report in node_report_summaries
            if report.get("status") == "PREFLIGHT_OK"
            and isinstance(report.get("latest_preflight"), dict)
            and report["latest_preflight"].get("ok") is True
        ),
        None,
    )
    if passing_watch_preflight:
        latest_preflight = passing_watch_preflight["node_report"]["latest_preflight"]
        preflight_evidence = (
            f"{passing_watch_preflight.get('path', '')} imported {passing_watch_preflight['node_report'].get('archive', '')}; "
            f"profile {latest_preflight.get('profile', '')}, base_url {latest_preflight.get('base_url', '')}"
        )
    elif passing_report_preflight:
        latest_preflight = passing_report_preflight["latest_preflight"]
        preflight_evidence = (
            f"{passing_report_preflight.get('path', '')}; "
            f"profile {latest_preflight.get('profile', '')}, base_url {latest_preflight.get('base_url', '')}"
        )
    else:
        preflight_evidence = "no imported target node preflight report has status PREFLIGHT_OK"
    local_surface_pass = next(
        (
            smoke for smoke in surface_smokes
            if smoke.get("passed") is True
            and smoke.get("qwen36_model") is True
            and not smoke.get("endpoint")
        ),
        None,
    )
    target_surface_pass = next(
        (
            smoke for smoke in surface_smokes
            if smoke.get("passed") is True
            and smoke.get("qwen36_model") is True
            and isinstance(smoke.get("endpoint"), dict)
            and bool(smoke.get("endpoint"))
        ),
        None,
    )
    surface_evidence = next(
        (
            str(smoke.get("path", ""))
            for smoke in surface_smokes
            if smoke is local_surface_pass
        ),
        "",
    )
    if any_watch_pass:
        target_smoke_evidence = "at least one watched target node passed"
    elif target_surface_pass:
        target_smoke_evidence = str(target_surface_pass.get("path", ""))
    else:
        target_smoke_evidence = "no parsed watch node or target surface-smoke artifact has PASS status"
    passing_dgx_run = next((run for run in dgx_runs if run.get("status") == "PASS"), None)
    latest_dgx_run = dgx_runs[0] if dgx_runs else None
    handoff_dgx_run = next(
        (
            run for run in dgx_runs
            if isinstance(run.get("handoff"), dict)
            and run["handoff"].get("complete") is True
        ),
        None,
    )
    latest_remote_probe_run = next(
        (
            run for run in dgx_runs
            if isinstance(run.get("remote_probe"), dict)
            and run["remote_probe"].get("status") != "MISSING"
        ),
        None,
    )
    if passing_dgx_run:
        dgx_evidence = f"{passing_dgx_run.get('path', '')} passed GATE.json and RUN_GATE.json"
    elif latest_dgx_run:
        dgx_evidence = (
            f"latest {latest_dgx_run.get('path', '')} is {latest_dgx_run.get('status', '')}: "
            f"{'; '.join(str(item) for item in latest_dgx_run.get('failures', [])[:3])}"
        ).rstrip(": ")
    else:
        dgx_evidence = "no DGX run directory with DGX_RUN.json was parsed"
    if handoff_dgx_run and isinstance(handoff_dgx_run.get("handoff"), dict):
        handoff_evidence = (
            f"{handoff_dgx_run['handoff'].get('archive', '')}; "
            f"script {handoff_dgx_run['handoff'].get('script', '')}; "
            f"readme {handoff_dgx_run['handoff'].get('readme', '')}"
        )
    else:
        handoff_evidence = "no DGX run has handoff/fleet-*.tgz plus RUN_ON_DGX.sh and DGX_HANDOFF.md"
    if latest_remote_probe_run and isinstance(latest_remote_probe_run.get("remote_probe"), dict):
        probe = latest_remote_probe_run["remote_probe"]
        remote_probe_evidence = (
            f"{probe.get('path', '')}: {probe.get('status', '')}; "
            f"target {probe.get('ssh_target', '')}; "
            f"dns_error {probe.get('target_dns_error', '')}"
        )
        remote_probe_status = "PASS" if probe.get("status") in {"DNS_RESOLVED", "SSH_PASS"} else "WARN"
    else:
        remote_probe_evidence = "no REMOTE_PROBE*.json artifact was parsed"
        remote_probe_status = "WARN"
    rows = [
        {
            "name": "Operator prep doc",
            "status": "PASS" if prep_doc.exists() and readiness_doc.exists() else "FAIL",
            "evidence": f"{rel(prep_doc)} and {rel(readiness_doc)}",
        },
        {
            "name": "Standalone packet artifacts",
            "status": "PASS" if packets.get("archive_count", 0) > 0 else "FAIL",
            "evidence": f"{packets.get('archive_count', 0)} packet archives, {packets.get('manifest_count', 0)} manifests",
        },
        {
            "name": "Standalone install profiles",
            "status": (
                "PASS"
                if packets.get("profile_count", 0) > 0
                and packets.get("profiles_prepared") == packets.get("profile_count")
                else "WARN"
            ),
            "evidence": f"{packets.get('profiles_prepared', 0)}/{packets.get('profile_count', 0)} profiles have packet archives",
        },
        {
            "name": "Watcher evidence",
            "status": "ACTION_REQUIRED" if target_action_count > 0 else ("PASS" if watches else "FAIL"),
            "evidence": watcher_evidence,
        },
        {
            "name": "Target standalone preflight",
            "status": "PASS" if passing_watch_preflight or passing_report_preflight else "UNPROVEN",
            "evidence": preflight_evidence,
        },
        {
            "name": "Local standalone endpoint smoke",
            "status": "PASS" if local_surface_pass else "UNPROVEN",
            "evidence": surface_evidence or "no parsed surface-smoke artifact has all surfaces PASS",
        },
        {
            "name": "Target standalone endpoint smoke",
            "status": "PASS" if any_watch_pass or target_surface_pass else "UNPROVEN",
            "evidence": target_smoke_evidence,
        },
        {
            "name": "Slack helper checkout",
            "status": "PASS" if slack.get("complete_checkout") else "FAIL",
            "evidence": str(slack.get("path", "")),
        },
        {
            "name": "Slack dry-run",
            "status": (
                "PASS"
                if slack.get("dry_run", {}).get("ok") is True
                else ("FAIL" if slack.get("dry_run_requested") else "WARN")
            ),
            "evidence": (
                "dry-run completed"
                if slack.get("dry_run", {}).get("ok") is True
                else ("not requested" if not slack.get("dry_run_requested") else str(slack.get("dry_run", {}).get("error", "dry-run failed")))
            ),
        },
        {
            "name": "Slack local-control demo",
            "status": (
                "PASS"
                if slack.get("local_demo", {}).get("ok") is True
                else ("FAIL" if slack.get("local_demo_requested") else "WARN")
            ),
            "evidence": (
                "local slack-control demo completed"
                if slack.get("local_demo", {}).get("ok") is True
                else (
                    "not requested"
                    if not slack.get("local_demo_requested")
                    else str(slack.get("local_demo", {}).get("error") or slack.get("local_demo", {}).get("reason", "local demo failed"))
                )
            ),
        },
        {
            "name": "DGX handoff bundle",
            "status": "PASS" if handoff_dgx_run else "WARN",
            "evidence": handoff_evidence,
        },
        {
            "name": "DGX remote probe",
            "status": remote_probe_status,
            "evidence": remote_probe_evidence,
        },
        {
            "name": "Slack control thread",
            "status": "UNPROVEN",
            "evidence": "requires a live target-side slack control probe and !status thread",
        },
        {
            "name": "Live DGX run",
            "status": "PASS" if passing_dgx_run else "UNPROVEN",
            "evidence": dgx_evidence,
        },
    ]
    return rows


def build_report(
    *,
    experiment_dir: Path = DEFAULT_EXPERIMENT_DIR,
    watch_reports: Sequence[Path] = (),
    watch_limit: int = 3,
    node_report_dir: Path = DEFAULT_NODE_REPORT_DIR,
    node_report_dirs: Sequence[Path] = (),
    node_report_limit: int = 5,
    surface_reports: Sequence[Path] = (),
    surface_limit: int = 10,
    dgx_dir: Path = DEFAULT_DGX_DIR,
    dgx_runs: Sequence[Path] = (),
    dgx_limit: int = 5,
    slack_helpers_dir: Path = DEFAULT_SLACK_HELPERS_DIR,
    slack_workdir: str = DEFAULT_SLACK_WORKDIR,
    slack_state_file: str = DEFAULT_SLACK_STATE_FILE,
    slack_lock_file: str | None = None,
    slack_transcript_file: str | None = None,
    env: Mapping[str, str] | None = None,
    run_slack_dry_run: bool = False,
    run_slack_local_demo: bool = False,
    run_slack_live_probe: bool = False,
    slack_channel: str = "dgx-control",
    slack_command: str = "python --version",
    slack_timeout_s: float = 20.0,
    runner: Runner = run_subprocess,
) -> dict[str, Any]:
    env_map = dict(os.environ if env is None else env)
    selected_watch_reports = list(watch_reports) or discover_watch_reports(experiment_dir, watch_limit)
    watches = [summarize_watch_report(path) for path in selected_watch_reports]
    effective_node_report_dir = node_report_dir
    if node_report_dir == DEFAULT_NODE_REPORT_DIR and experiment_dir != DEFAULT_EXPERIMENT_DIR:
        effective_node_report_dir = experiment_dir / "node-reports"
    selected_node_report_dirs = list(node_report_dirs) or discover_node_report_dirs(effective_node_report_dir, node_report_limit)
    node_report_summaries = [summarize_node_report_dir(path) for path in selected_node_report_dirs]
    selected_surface_reports = list(surface_reports) or discover_surface_smokes(experiment_dir, surface_limit)
    surface_smokes = [summarize_surface_smoke(path) for path in selected_surface_reports]
    selected_dgx_runs = list(dgx_runs) or discover_dgx_runs(dgx_dir, dgx_limit)
    dgx_run_summaries = [summarize_dgx_run(path) for path in selected_dgx_runs]
    cleared_target_nodes = target_surface_pass_nodes(surface_smokes)
    target_next_actions = [
        row for row in collect_target_next_actions(watches)
        if row.get("node") not in cleared_target_nodes
    ]
    packets = packet_artifacts(DEFAULT_PACKET_DIRS)
    slack = slack_helpers_status(
        slack_helpers_dir,
        env_map,
        run_dry_run=run_slack_dry_run,
        run_local_demo=run_slack_local_demo,
        run_live_probe=run_slack_live_probe,
        channel=slack_channel,
        command_text=slack_command,
        timeout_s=slack_timeout_s,
        runner=runner,
    )
    effective_slack_lock_file = slack_lock_file or slack_lock_file_for(slack_state_file)
    effective_slack_transcript_file = slack_transcript_file or slack_transcript_file_for(slack_state_file)
    slack_actions = slack_next_actions(
        slack,
        channel=slack_channel,
        workdir=slack_workdir,
        state_file=slack_state_file,
        lock_file=effective_slack_lock_file,
        transcript_file=effective_slack_transcript_file,
    )
    checks = check_rows(
        prep_doc=BASE / "QWEN36-DGX-STANDALONE-PREP.md",
        readiness_doc=DEFAULT_EXPERIMENT_DIR / "QWEN36-DGX-STANDALONE-READINESS.md",
        watches=watches,
        node_report_summaries=node_report_summaries,
        surface_smokes=surface_smokes,
        dgx_runs=dgx_run_summaries,
        target_next_actions=target_next_actions,
        packets=packets,
        slack=slack,
    )
    blocking_statuses = {"FAIL", "ACTION_REQUIRED"}
    handoff_failure_checks = {
        "Operator prep doc",
        "Standalone packet artifacts",
        "Slack helper checkout",
        "Slack dry-run",
    }
    external_unproven = [row for row in checks if row["status"] == "UNPROVEN"]
    action_required = any(row["status"] in blocking_statuses for row in checks) or bool(external_unproven)
    ready_for_handoff = not any(
        row["status"] == "FAIL" and row["name"] in handoff_failure_checks
        for row in checks
    )
    qwen36_surface_passes = len([
        smoke for smoke in surface_smokes
        if smoke.get("passed") is True and smoke.get("qwen36_model") is True
    ])
    target_preflight_passes = len([
        watch for watch in watches
        if isinstance(watch.get("node_report"), dict)
        and watch["node_report"].get("status") == "PREFLIGHT_OK"
    ]) + len([report for report in node_report_summaries if report.get("status") == "PREFLIGHT_OK"])
    dgx_run_passes = len([run for run in dgx_run_summaries if run.get("status") == "PASS"])
    dgx_handoff_bundles = len([
        run for run in dgx_run_summaries
        if isinstance(run.get("handoff"), dict)
        and run["handoff"].get("complete") is True
    ])
    dgx_remote_probes = len([
        run for run in dgx_run_summaries
        if isinstance(run.get("remote_probe"), dict)
        and run["remote_probe"].get("status") != "MISSING"
    ])
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "summary": {
            "ready_for_handoff": ready_for_handoff,
            "operator_action_required": action_required,
            "unproven_external_gates": [row["name"] for row in external_unproven],
            "watch_reports": len(watches),
            "node_reports": len(node_report_summaries),
            "surface_smokes": len(surface_smokes),
            "qwen36_surface_passes": qwen36_surface_passes,
            "target_preflight_passes": target_preflight_passes,
            "dgx_runs": len(dgx_run_summaries),
            "dgx_run_passes": dgx_run_passes,
            "dgx_handoff_bundles": dgx_handoff_bundles,
            "dgx_remote_probes": dgx_remote_probes,
            "latest_dgx_run_status": dgx_run_summaries[0].get("status") if dgx_run_summaries else None,
            "target_action_nodes": len([row for row in target_next_actions if row.get("primary_action")]),
            "packet_archives": packets.get("archive_count", 0),
            "standalone_profiles_prepared": packets.get("profiles_prepared", 0),
            "standalone_profile_count": packets.get("profile_count", 0),
            "slack_helpers": bool(slack.get("complete_checkout")),
            "slack_live_status": slack_actions.get("status"),
            "slack_live_probe_status": slack_actions.get("live_probe_status"),
            "slack_dry_run_ok": slack.get("dry_run", {}).get("ok") if slack.get("dry_run_requested") else None,
            "slack_local_demo_ok": slack.get("local_demo", {}).get("ok") if slack.get("local_demo_requested") else None,
        },
        "checks": checks,
        "watch_reports": watches,
        "node_reports": node_report_summaries,
        "surface_smokes": surface_smokes,
        "dgx_runs": dgx_run_summaries,
        "target_next_actions": target_next_actions,
        "target_surface_pass_nodes": sorted(cleared_target_nodes),
        "packet_artifacts": packets,
        "slack_control": slack,
        "slack_next_actions": slack_actions,
    }


def table_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Qwen3.6 DGX/standalone readiness",
        "",
        f"Generated: `{report.get('generated_at', '')}`",
        "",
        "## Summary",
        "",
        f"- Ready for handoff: `{summary.get('ready_for_handoff')}`",
        f"- Operator action required: `{summary.get('operator_action_required')}`",
        f"- Packet archives: `{summary.get('packet_archives')}`",
        f"- Standalone profiles prepared: `{summary.get('standalone_profiles_prepared')}/{summary.get('standalone_profile_count')}`",
        f"- Watch reports: `{summary.get('watch_reports')}`",
        f"- Node reports: `{summary.get('node_reports')}`",
        f"- Surface smokes: `{summary.get('surface_smokes')}`",
        f"- Qwen3.6 surface-smoke passes: `{summary.get('qwen36_surface_passes')}`",
        f"- Target standalone preflight passes: `{summary.get('target_preflight_passes')}`",
        f"- DGX runs parsed: `{summary.get('dgx_runs')}`",
        f"- DGX run passes: `{summary.get('dgx_run_passes')}`",
        f"- DGX handoff bundles: `{summary.get('dgx_handoff_bundles')}`",
        f"- DGX remote probes: `{summary.get('dgx_remote_probes')}`",
        f"- Latest DGX run status: `{summary.get('latest_dgx_run_status')}`",
        f"- Target action nodes: `{summary.get('target_action_nodes')}`",
        f"- Slack live status: `{summary.get('slack_live_status')}`",
        f"- Slack live probe: `{summary.get('slack_live_probe_status')}`",
        f"- Slack local demo: `{summary.get('slack_local_demo_ok')}`",
        f"- Unproven external gates: `{', '.join(summary.get('unproven_external_gates', []))}`",
        "",
        "## Checks",
        "",
        "| Check | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for row in report.get("checks", []):
        lines.append(f"| {table_cell(row.get('name', ''))} | {table_cell(row.get('status', ''))} | {table_cell(row.get('evidence', ''))} |")
    lines.extend(["", "## Standalone Install Benches", ""])
    packets = report.get("packet_artifacts") if isinstance(report.get("packet_artifacts"), dict) else {}
    profile_rows = packets.get("profile_matrix") if isinstance(packets.get("profile_matrix"), list) else []
    if not profile_rows:
        lines.append("No standalone packet profile manifests were parsed.")
    for row in profile_rows:
        lines.append(
            f"- `{row.get('profile', '')}` ({row.get('label', '')}): `{row.get('status', '')}` "
            f"packet `{row.get('archive', '')}` report_target `{row.get('report_target', '')}`"
        )
        commands = [
            ("bootstrap", row.get("bootstrap_command")),
            ("preflight", row.get("preflight_command")),
            ("install", row.get("install_command")),
            ("start", row.get("start_command")),
            ("report", row.get("report_command")),
        ]
        command_text = "; ".join(f"{name}: `{value}`" for name, value in commands if value)
        if command_text:
            lines.append(f"  - Commands: {command_text}")
        receive = row.get("receive") if isinstance(row.get("receive"), list) else []
        if receive:
            lines.append("  - Receive/install smoke:")
            lines.append("")
            shell = "powershell" if row.get("profile") in {"nvidia", "vulkan"} else "bash"
            lines.append(f"```{shell}")
            lines.extend(str(command) for command in receive)
            lines.append("```")
    lines.extend(["", "## Watch Reports", ""])
    watches = report.get("watch_reports", [])
    cleared_target_nodes = set(
        str(node) for node in report.get("target_surface_pass_nodes", [])
        if str(node).strip()
    )
    if not watches:
        lines.append("No watch reports were parsed.")
    for watch in watches:
        summary_row = watch.get("summary") if isinstance(watch.get("summary"), dict) else {}
        lines.append(f"- `{watch.get('path', '')}`: passed `{summary_row.get('passed', 0)}`, waiting `{summary_row.get('waiting', 0)}`, failed `{summary_row.get('failed', 0)}`, action_required `{summary_row.get('action_required')}`")
        for node in watch.get("nodes", []):
            node_name = str(node.get("node", "")).strip()
            if node_name in cleared_target_nodes:
                lines.append(f"  - `{node_name}`: cleared by imported target surface smoke; historical watch actions suppressed")
                continue
            actions = node.get("next_actions") if isinstance(node.get("next_actions"), list) else []
            action_kinds = ", ".join(str(action.get("kind", "")) for action in actions if isinstance(action, dict)) or "none"
            lines.append(f"  - `{node_name}`: `{node.get('status', '')}` / `{node.get('state', '')}`; next actions `{action_kinds}`")
        node_report = watch.get("node_report") if isinstance(watch.get("node_report"), dict) else {}
        latest_preflight = node_report.get("latest_preflight") if isinstance(node_report.get("latest_preflight"), dict) else {}
        if node_report.get("status") or latest_preflight.get("path"):
            lines.append(
                f"  - Node report: `{node_report.get('status', '')}`; "
                f"preflight_ok `{latest_preflight.get('ok')}`; "
                f"profile `{latest_preflight.get('profile', '')}`; "
                f"base_url `{latest_preflight.get('base_url', '')}`"
            )
    lines.extend(["", "## Node Reports", ""])
    node_reports_rows = report.get("node_reports", []) if isinstance(report.get("node_reports"), list) else []
    if not node_reports_rows:
        lines.append("No extracted node-report directories were parsed.")
    for node_report in node_reports_rows:
        latest_preflight = node_report.get("latest_preflight") if isinstance(node_report.get("latest_preflight"), dict) else {}
        lines.append(
            f"- `{node_report.get('path', '')}`: `{node_report.get('status', '')}`, "
            f"preflight_ok `{latest_preflight.get('ok')}`, "
            f"profile `{latest_preflight.get('profile', '')}`, "
            f"base_url `{latest_preflight.get('base_url', '')}`"
        )
    lines.extend(["", "## Target Next Actions", ""])
    target_actions = report.get("target_next_actions", [])
    if not target_actions:
        lines.append("No target next actions were found in parsed watch reports.")
    for row in target_actions:
        primary = row.get("primary_action") if isinstance(row.get("primary_action"), dict) else None
        if primary:
            lines.append(f"- `{row.get('node', '')}`: `{primary.get('kind', '')}` from `{primary.get('source_report', '')}`")
            detail = primary.get("detail")
            if detail:
                lines.append(f"  - {detail}")
            bootstrap_files = primary.get("bootstrap_files")
            if isinstance(bootstrap_files, list) and bootstrap_files:
                lines.append(f"  - Bootstrap files: `{', '.join(str(item) for item in bootstrap_files)}`")
            if primary.get("packet_dir"):
                lines.append(f"  - Packet dir: `{primary.get('packet_dir')}`")
            if primary.get("report_target"):
                lines.append(f"  - Report target: `{primary.get('report_target')}`")
            snippets = row.get("snippets") if isinstance(row.get("snippets"), dict) else {}
            target_commands = snippets.get("target_commands") if isinstance(snippets.get("target_commands"), list) else []
            if target_commands:
                lines.append("  - Target command:")
                lines.append("")
                lines.append("```powershell")
                lines.extend(str(command) for command in target_commands)
                lines.append("```")
            recover_commands = snippets.get("target_recover_commands") if isinstance(snippets.get("target_recover_commands"), list) else []
            if recover_commands:
                lines.append("  - If Taildrop files are not visible on the target:")
                lines.append("")
                lines.append("```powershell")
                lines.extend(str(command) for command in recover_commands)
                lines.append("```")
            verify_commands = snippets.get("driver_verify_commands") if isinstance(snippets.get("driver_verify_commands"), list) else []
            if verify_commands:
                lines.append("  - Driver verification command after the target starts:")
                lines.append("")
                lines.append("```powershell")
                lines.extend(str(command) for command in verify_commands)
                lines.append("```")
            send_commands = snippets.get("driver_send_commands") if isinstance(snippets.get("driver_send_commands"), list) else []
            if send_commands:
                lines.append("  - Driver send command:")
                lines.append("")
                lines.append("```powershell")
                lines.extend(str(command) for command in send_commands)
                lines.append("```")
        else:
            lines.append(f"- `{row.get('node', '')}`: no required action in parsed watch reports")
        optional = row.get("optional_actions") if isinstance(row.get("optional_actions"), list) else []
        for action in optional:
            lines.append(f"  - Optional `{action.get('kind', '')}`: {action.get('detail', '')}")
    lines.extend(["", "## Surface Smokes", ""])
    surface_smokes = report.get("surface_smokes", [])
    if not surface_smokes:
        lines.append("No surface-smoke artifacts were parsed.")
    for smoke in surface_smokes:
        lines.append(
            f"- `{smoke.get('path', '')}`: node `{smoke.get('node_name', '')}`, "
            f"model `{smoke.get('model', '')}`, qwen36 `{smoke.get('qwen36_model')}`, "
            f"verified_pass `{smoke.get('passed')}`, pass `{smoke.get('pass_count', 0)}`, "
            f"planned `{smoke.get('planned_count', 0)}`, failed `{smoke.get('fail_count', 0)}`"
        )
        for surface in smoke.get("surfaces", []):
            perf = ""
            if surface.get("decode_tps") is not None:
                perf = f", decode_tps `{surface.get('decode_tps')}`"
            lines.append(f"  - `{surface.get('surface', '')}`: `{surface.get('status', '')}`{perf}")
    lines.extend(["", "## DGX Runs", ""])
    dgx_runs = report.get("dgx_runs", [])
    if not dgx_runs:
        lines.append("No DGX run directories with `DGX_RUN.json` were parsed.")
    for run in dgx_runs:
        plan_sha = str(run.get("plan_sha256", ""))
        runbook_sha = str(run.get("runbook_sha256", ""))
        lines.append(
            f"- `{run.get('path', '')}`: status `{run.get('status', '')}`, "
            f"run_id `{run.get('run_id', '')}`, model `{run.get('model', '')}`, "
            f"gate `{run.get('gate_passed')}`, run_gate `{run.get('run_gate_passed')}`, "
            f"endpoint_reports `{run.get('endpoint_report_count', 0)}`, "
            f"benchmark_monitor_reports `{run.get('benchmark_monitor_reports', 0)}`, "
            f"monitor_manifest `{run.get('benchmark_monitor_manifest')}`"
        )
        if run.get("generated_at") or plan_sha or runbook_sha:
            lines.append(
                f"  - Packet: generated `{run.get('generated_at', '')}`, "
                f"plan_sha `{plan_sha[:12]}`, runbook_sha `{runbook_sha[:12]}`"
            )
        handoff = run.get("handoff") if isinstance(run.get("handoff"), dict) else {}
        if handoff:
            lines.append(
                f"  - Handoff: complete `{handoff.get('complete')}`, "
                f"archive `{handoff.get('archive', '')}`, "
                f"bytes `{handoff.get('archive_bytes', 0)}`, "
                f"sha `{str(handoff.get('archive_sha256', ''))[:12]}`"
            )
        remote_probe = run.get("remote_probe") if isinstance(run.get("remote_probe"), dict) else {}
        if remote_probe and remote_probe.get("status") != "MISSING":
            addresses = remote_probe.get("target_dns_addresses") if isinstance(remote_probe.get("target_dns_addresses"), list) else []
            lines.append(
                f"  - Remote probe: status `{remote_probe.get('status', '')}`, "
                f"target `{remote_probe.get('ssh_target', '')}`, "
                f"dns_resolved `{remote_probe.get('target_dns_resolved')}`, "
                f"addresses `{', '.join(str(item) for item in addresses)}`, "
                f"error `{remote_probe.get('target_dns_error', '')}`"
            )
        missing = run.get("missing_artifacts") if isinstance(run.get("missing_artifacts"), list) else []
        if missing:
            lines.append(f"  - Missing: `{', '.join(str(item) for item in missing)}`")
        failures = run.get("failures") if isinstance(run.get("failures"), list) else []
        for failure in failures[:5]:
            lines.append(f"  - Gate gap: {failure}")
    slack = report.get("slack_control", {})
    slack_actions = report.get("slack_next_actions") if isinstance(report.get("slack_next_actions"), dict) else {}
    lines.extend([
        "",
        "## Slack Control",
        "",
        f"- Checkout: `{slack.get('path', '')}`",
        f"- Complete checkout: `{slack.get('complete_checkout')}`",
        f"- Dry-run requested: `{slack.get('dry_run_requested')}`",
        f"- Local demo requested: `{slack.get('local_demo_requested')}`",
        f"- Live probe requested: `{slack.get('live_probe_requested')}`",
    ])
    if isinstance(slack.get("dry_run"), dict):
        dry = slack["dry_run"]
        lines.append(f"- Dry-run ok: `{dry.get('ok')}` exit `{dry.get('exit_code')}`")
    if isinstance(slack.get("local_demo"), dict):
        demo = slack["local_demo"]
        lines.append(f"- Local demo ok: `{demo.get('ok')}` exit `{demo.get('exit_code')}`")
        if demo.get("stdout"):
            first_line = str(demo.get("stdout", "")).splitlines()[0] if str(demo.get("stdout", "")).splitlines() else ""
            lines.append(f"- Local demo marker: `{first_line}`")
    lines.extend([
        f"- Live status: `{slack_actions.get('status', '')}`",
        f"- Live probe status: `{slack_actions.get('live_probe_status', '')}`",
        f"- Control state file: `{slack_actions.get('state_file', '')}`",
        f"- Control lock file: `{slack_actions.get('lock_file', '')}`",
        f"- Control transcript file: `{slack_actions.get('transcript_file', '')}`",
        f"- Missing live requirements: `{', '.join(slack_actions.get('missing_requirements', []))}`",
    ])
    if isinstance(slack.get("live_probe"), dict):
        probe = slack["live_probe"]
        lines.append(f"- Live probe ok: `{probe.get('ok')}` exit `{probe.get('exit_code')}`")
        if probe.get("skipped"):
            lines.append(f"- Live probe skipped: `{probe.get('reason', '')}`")
    setup_commands = slack_actions.get("setup_commands") if isinstance(slack_actions.get("setup_commands"), list) else []
    if setup_commands:
        lines.extend(["", "DGX/lab host setup:", "", "```bash"])
        lines.extend(str(command) for command in setup_commands)
        lines.append("```")
    foreground_command = slack_actions.get("foreground_control_command")
    if foreground_command:
        lines.extend([
            "",
            "Foreground bridge smoke before systemd:",
            "",
            "```bash",
            str(foreground_command),
            "```",
        ])
    service_install_commands = (
        slack_actions.get("service_install_commands")
        if isinstance(slack_actions.get("service_install_commands"), list)
        else []
    )
    if service_install_commands:
        lines.extend(["", "Install as a systemd service:", "", "```bash"])
        lines.extend(str(command) for command in service_install_commands)
        lines.append("```")
    dry_run_commands = slack_actions.get("dry_run_commands") if isinstance(slack_actions.get("dry_run_commands"), list) else []
    if dry_run_commands:
        lines.extend(["", "Installer dry-run on the DGX/lab host:", "", "```bash"])
        lines.extend(str(command) for command in dry_run_commands)
        lines.append("```")
    live_probe_commands = slack_actions.get("live_probe_commands") if isinstance(slack_actions.get("live_probe_commands"), list) else []
    if live_probe_commands:
        lines.extend(["", "After install:", "", "```bash"])
        lines.extend(str(command) for command in live_probe_commands)
        lines.append("```")
    lines.extend([
        "",
        "## External Gates",
        "",
        f"Remaining external gates: `{', '.join(summary.get('unproven_external_gates', [])) or 'none'}`. "
        "Imported target or DGX evidence is required only for the corresponding unproven target surface.",
        "",
    ])
    return "\n".join(lines)


def write_text(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8", newline="\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--watch-report", type=Path, action="append", default=[], help="watch JSON to include; defaults to latest *watch*.json files")
    parser.add_argument("--watch-limit", type=int, default=3)
    parser.add_argument("--node-report-dir", type=Path, default=DEFAULT_NODE_REPORT_DIR)
    parser.add_argument("--node-report", type=Path, action="append", default=[], help="extracted node-report directory to include; defaults to latest under --node-report-dir")
    parser.add_argument("--node-report-limit", type=int, default=5)
    parser.add_argument("--surface-report", type=Path, action="append", default=[], help="surface-smoke JSON to include; defaults to latest schema-matching files")
    parser.add_argument("--surface-limit", type=int, default=10)
    parser.add_argument("--dgx-dir", type=Path, default=DEFAULT_DGX_DIR)
    parser.add_argument("--dgx-run", type=Path, action="append", default=[], help="DGX run directory to include; defaults to latest directories under --dgx-dir")
    parser.add_argument("--dgx-limit", type=int, default=5)
    parser.add_argument("--slack-helpers-dir", type=Path, default=DEFAULT_SLACK_HELPERS_DIR)
    parser.add_argument("--run-slack-dry-run", action="store_true", help="run slack_helpers.cli control --dry-run")
    parser.add_argument("--run-slack-local-demo", action="store_true", help="run slack-helpers local SlackControlBridge demo without Slack credentials")
    parser.add_argument("--run-slack-live-probe", action="store_true", help="run slack_helpers.cli control --probe when a Slack token is configured")
    parser.add_argument("--slack-channel", default="dgx-control")
    parser.add_argument("--slack-workdir", default=DEFAULT_SLACK_WORKDIR)
    parser.add_argument("--slack-state-file", default=DEFAULT_SLACK_STATE_FILE)
    parser.add_argument("--slack-lock-file", default=None)
    parser.add_argument("--slack-transcript-file", default=None)
    parser.add_argument("--slack-command", default="python --version")
    parser.add_argument("--slack-timeout-s", type=float, default=20.0)
    parser.add_argument("--out", type=Path, help="write JSON report")
    parser.add_argument("--markdown", type=Path, help="write Markdown report")
    parser.add_argument("--fail-on-action-required", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(
        experiment_dir=args.experiment_dir,
        watch_reports=args.watch_report,
        watch_limit=max(1, args.watch_limit),
        node_report_dir=args.node_report_dir,
        node_report_dirs=args.node_report,
        node_report_limit=max(1, args.node_report_limit),
        surface_reports=args.surface_report,
        surface_limit=max(1, args.surface_limit),
        dgx_dir=args.dgx_dir,
        dgx_runs=args.dgx_run,
        dgx_limit=max(1, args.dgx_limit),
        slack_helpers_dir=args.slack_helpers_dir,
        slack_workdir=args.slack_workdir,
        slack_state_file=args.slack_state_file,
        slack_lock_file=args.slack_lock_file,
        slack_transcript_file=args.slack_transcript_file,
        run_slack_dry_run=args.run_slack_dry_run,
        run_slack_local_demo=args.run_slack_local_demo,
        run_slack_live_probe=args.run_slack_live_probe,
        slack_channel=args.slack_channel,
        slack_command=args.slack_command,
        slack_timeout_s=max(1.0, args.slack_timeout_s),
    )
    print(json.dumps(report["summary"], indent=2))
    if args.out:
        write_text(args.out, json.dumps(report, indent=2) + "\n")
    if args.markdown:
        write_text(args.markdown, render_markdown(report))
    if args.fail_on_action_required and report["summary"].get("operator_action_required"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
