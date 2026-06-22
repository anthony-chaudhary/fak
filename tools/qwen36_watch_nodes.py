#!/usr/bin/env python3
"""Watch Qwen3.6 test-bed nodes and run the surface smoke when ready."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import qwen36_node_packet as node_packet
import qwen36_node_reports as node_reports
import qwen36_surface_smoke as smoke


SCHEMA = "fak.qwen36-node-watch.v1"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M"


@dataclass(frozen=True)
class WatchOptions:
    nodes: list[str]
    model: str
    serve_port: int
    out_dir: Path
    http_timeout_s: float
    agent_max_turns: int
    agent_timeout_s: float
    gateway_chat: bool
    poll_interval_s: float
    max_wait_s: float
    mode: str
    perf_decode_baseline_tps: float = 0.0
    min_decode_tps: float = 0.0
    import_reports: bool = False
    report_inbox: Path = node_reports.DEFAULT_INBOX
    report_out_dir: Path = node_reports.DEFAULT_OUT_DIR
    report_archive: Path | None = None
    skip_report_taildrop: bool = False
    report_log_tail_lines: int = 60
    send_packet: bool = False
    packet_out_dir: Path = ROOT / "tools" / "_registry" / "qwen36-watch-packets"
    packet_report_target: str = node_packet.AUTO_REPORT_TARGET
    packet_taildrop_retries: int = 1
    packet_taildrop_retry_delay_s: float = 5.0
    packet_taildrop_timeout_s: float = 15.0
    packet_bootstrap: bool = True
    packet_profile: str = "auto"
    registry: str = ""


Resolver = Callable[[str, int, bool, float], dict[str, Any]]
Runner = Callable[[list[str]], int]
Sleeper = Callable[[float], None]
ReportImporter = Callable[[WatchOptions], dict[str, Any]]
PacketDispatcher = Callable[[WatchOptions], dict[str, Any]]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return text or "node"


def smoke_paths(out_dir: Path, node: str) -> tuple[Path, Path]:
    node_slug = slug(node)
    return (
        out_dir / f"qwen36-surface-smoke-{node_slug}.json",
        out_dir / f"qwen36-surface-smoke-{node_slug}.md",
    )


def smoke_argv(node: str, opts: WatchOptions) -> list[str]:
    out_json, out_md = smoke_paths(opts.out_dir, node)
    argv = [
        "--tailscale-node", node,
        "--serve-port", str(opts.serve_port),
        "--model", opts.model,
        "--node-name", slug(node),
        "--http-timeout-s", str(opts.http_timeout_s),
        "--agent-max-turns", str(opts.agent_max_turns),
        "--agent-timeout-s", str(opts.agent_timeout_s),
        "--out-dir", str(opts.out_dir),
        "--out", str(out_json),
        "--markdown", str(out_md),
    ]
    if opts.gateway_chat:
        argv.append("--gateway-chat")
    if opts.perf_decode_baseline_tps > 0:
        argv.extend(["--perf-decode-baseline-tps", str(opts.perf_decode_baseline_tps)])
    if opts.min_decode_tps > 0:
        argv.extend(["--min-decode-tps", str(opts.min_decode_tps)])
    return argv


def registry_endpoint(node: str, opts: WatchOptions) -> dict[str, Any] | None:
    try:
        endpoint = smoke.resolve_endpoint(node, opts.registry, False, opts.http_timeout_s)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if endpoint.get("state") == "MISSING":
        return None
    return endpoint


def resolve_watch_endpoint(
    node: str,
    opts: WatchOptions,
    resolver: Resolver,
) -> dict[str, Any]:
    endpoint = registry_endpoint(node, opts)
    if endpoint is not None:
        return endpoint
    return resolver(node, opts.serve_port, False, opts.http_timeout_s)


def poll_once(
    opts: WatchOptions,
    *,
    resolver: Resolver = smoke.resolve_tailscale_node,
    runner: Runner = smoke.main,
    prior: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for node in opts.nodes:
        previous = (prior or {}).get(node)
        if previous and previous.get("status") in {"PASS", "FAIL"}:
            rows.append(previous)
            continue
        endpoint = resolve_watch_endpoint(node, opts, resolver)
        row: dict[str, Any] = {
            "node": node,
            "state": endpoint.get("state"),
            "detail": endpoint.get("detail"),
            "endpoint": endpoint,
            "smoke_ran": False,
        }
        if endpoint.get("state") == "READY":
            argv = smoke_argv(node, opts)
            rc = runner(argv)
            out_json, out_md = smoke_paths(opts.out_dir, node)
            row.update({
                "smoke_ran": True,
                "smoke_exit": rc,
                "smoke_json": str(out_json),
                "smoke_markdown": str(out_md),
                "status": "PASS" if rc == 0 else "FAIL",
            })
        else:
            row["status"] = "WAIT"
        rows.append(row)
    return rows


def complete(rows: list[dict[str, Any]], mode: str) -> bool:
    passed = [row for row in rows if row.get("status") == "PASS"]
    return bool(passed) if mode == "any" else len(passed) == len(rows)


def import_node_report_once(opts: WatchOptions) -> dict[str, Any]:
    args = argparse.Namespace(
        inbox=opts.report_inbox,
        out_dir=opts.report_out_dir,
        archive=opts.report_archive,
        wait=False,
        skip_taildrop=opts.skip_report_taildrop or opts.report_archive is not None,
        replace=True,
        log_tail_lines=opts.report_log_tail_lines,
    )
    try:
        return node_reports.import_report_bundle(args)
    except Exception as exc:
        return {"schema": node_reports.SCHEMA, "imported": False, "error": str(exc)}


def packet_profile(endpoint: dict[str, Any]) -> str:
    explicit = str(
        endpoint.get("packet_profile")
        or endpoint.get("qwen36_profile")
        or endpoint.get("profile")
        or ""
    ).strip().lower()
    if explicit:
        node_packet.parse_profiles(explicit)
        return explicit

    facts = " ".join(
        str(value or "")
        for value in (
            endpoint.get("os"),
            endpoint.get("gpu"),
            endpoint.get("accelerator"),
            endpoint.get("hardware"),
            endpoint.get("notes"),
            " ".join(str(v) for v in (endpoint.get("roles") or [])),
        )
    ).lower()
    os_name = str(endpoint.get("os") or "").lower()
    if "mac" in os_name or "darwin" in os_name:
        return "mac"
    if "vulkan" in facts or "amd" in facts or "radeon" in facts:
        return "vulkan"
    if "linux" in os_name:
        return "linux-nvidia"
    if "nvidia" in facts or "cuda" in facts:
        return "nvidia"
    if "windows" in os_name or "nvidia" in os_name:
        return "nvidia"
    return "both"


def dispatch_packets_once(
    opts: WatchOptions,
    *,
    resolver: Resolver = smoke.resolve_tailscale_node,
) -> dict[str, Any]:
    rows = []
    generated_at = utc_now()
    report_target = ""
    try:
        report_target = node_packet.resolve_report_target(opts.packet_report_target)
    except ValueError as exc:
        return {
            "generated_at": generated_at,
            "sent": False,
            "error": str(exc),
            "nodes": [],
        }

    for node in opts.nodes:
        endpoint = resolve_watch_endpoint(node, opts, resolver)
        profile_value = opts.packet_profile if opts.packet_profile != "auto" else packet_profile(endpoint)
        profiles = node_packet.parse_profiles(profile_value)
        out_dir = opts.packet_out_dir / f"{utc_now().replace(':', '').replace('.', '-')}-{slug(node)}"
        manifest = node_packet.write_packet(ROOT, out_dir, profiles, opts.model, opts.serve_port, report_target=report_target)
        archive = out_dir / f"qwen36-node-packet-{node_packet.utc_stamp()}.zip"
        node_packet.archive_packet(Path(manifest["payload_dir"]), archive)
        manifest["archive"] = str(archive)
        paths = node_packet.taildrop_send_paths(
            archive,
            out_dir,
            manifest,
            include_bootstrap=opts.packet_bootstrap,
        )
        send = node_packet.taildrop_files(
            paths,
            node,
            dry_run=False,
            retries=opts.packet_taildrop_retries,
            retry_delay_s=opts.packet_taildrop_retry_delay_s,
            attempt_timeout_s=opts.packet_taildrop_timeout_s,
        )
        rows.append({
            "node": node,
            "profile": profile_value,
            "endpoint_state": endpoint.get("state"),
            "endpoint_detail": endpoint.get("detail"),
            "packet_dir": str(out_dir),
            "archive": str(archive),
            "bootstrap_files": manifest.get("bootstrap_files", []),
            "taildrop": send,
        })
    return {
        "generated_at": generated_at,
        "report_target": report_target,
        "sent": bool(rows) and all(row.get("taildrop", {}).get("sent") for row in rows),
        "nodes": rows,
    }


def packet_dispatch_for_node(packet_dispatch: dict[str, Any] | None, node: str) -> dict[str, Any] | None:
    if not isinstance(packet_dispatch, dict):
        return None
    rows = packet_dispatch.get("nodes")
    if not isinstance(rows, list):
        return None
    return next((row for row in rows if isinstance(row, dict) and row.get("node") == node), None)


def next_actions_for_row(
    row: dict[str, Any],
    opts: WatchOptions,
    packet_dispatch: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if row.get("status") != "WAIT":
        return []

    actions: list[dict[str, Any]] = []
    node = str(row.get("node") or "")
    endpoint = row.get("endpoint") if isinstance(row.get("endpoint"), dict) else {}
    state = row.get("state")
    dispatch_row = packet_dispatch_for_node(packet_dispatch, node)
    taildrop = dispatch_row.get("taildrop") if isinstance(dispatch_row, dict) else {}

    if state == "ONLINE_NO_SERVE":
        if isinstance(taildrop, dict) and taildrop.get("sent") is True:
            bootstrap_files = dispatch_row.get("bootstrap_files") if isinstance(dispatch_row, dict) else []
            actions.append({
                "kind": "run_node_launcher",
                "required": True,
                "detail": (
                    f"Packet reached {node}; run one received bootstrap launcher on the node, "
                    "then rerun this watcher or import returned reports."
                ),
                "bootstrap_files": bootstrap_files if isinstance(bootstrap_files, list) else [],
                "packet_dir": dispatch_row.get("packet_dir") if isinstance(dispatch_row, dict) else "",
                "archive": dispatch_row.get("archive") if isinstance(dispatch_row, dict) else "",
                "report_target": packet_dispatch.get("report_target", "") if isinstance(packet_dispatch, dict) else "",
            })
        elif opts.send_packet:
            actions.append({
                "kind": "packet_delivery",
                "required": True,
                "detail": "Packet delivery did not complete; inspect packet_dispatch before starting the node.",
            })
        else:
            actions.append({
                "kind": "send_packet",
                "required": True,
                "detail": "Send a node packet with --send-packet or start the OpenAI-compatible model server manually.",
            })

    ssh = endpoint.get("ssh") if isinstance(endpoint, dict) and isinstance(endpoint.get("ssh"), dict) else {}
    if ssh.get("tcp_open") is True and ssh.get("auth_verified") is not True:
        actions.append({
            "kind": "ssh_auth",
            "required": False,
            "detail": "SSH TCP is open, but public-key auth is not verified for this driver; configure SSH auth to enable remote starts.",
            "host": next((probe.get("host") for probe in ssh.get("tcp", []) if isinstance(probe, dict) and probe.get("open")), ""),
            "port": ssh.get("port"),
        })
    elif ssh.get("tcp_open") is True and ssh.get("auth_verified") is True:
        actions.append({
            "kind": "ssh_remote_start_available",
            "required": False,
            "detail": "SSH auth is verified for this driver; use the registered SSH route for remote preflight/start if manual node access is unavailable.",
            "host": next((probe.get("host") for probe in ssh.get("tcp", []) if isinstance(probe, dict) and probe.get("open")), ""),
            "port": ssh.get("port"),
        })

    return actions


def rows_with_next_actions(
    rows: list[dict[str, Any]],
    opts: WatchOptions,
    packet_dispatch: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    enriched = []
    for row in rows:
        row_out = dict(row)
        actions = next_actions_for_row(row, opts, packet_dispatch)
        if actions:
            row_out["next_actions"] = actions
        enriched.append(row_out)
    return enriched


def build_report(
    rows: list[dict[str, Any]],
    opts: WatchOptions,
    started_at: str,
    timed_out: bool,
    node_report: dict[str, Any] | None = None,
    packet_dispatch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report_rows = rows_with_next_actions(rows, opts, packet_dispatch)
    summary = {
        "nodes": len(report_rows),
        "passed": len([r for r in report_rows if r.get("status") == "PASS"]),
        "waiting": len([r for r in report_rows if r.get("status") == "WAIT"]),
        "failed": len([r for r in report_rows if r.get("status") == "FAIL"]),
        "timed_out": timed_out,
        "action_required": any(
            action.get("required") is True
            for row in report_rows
            for action in row.get("next_actions", [])
            if isinstance(action, dict)
        ),
    }
    if opts.import_reports:
        summary["node_report_imported"] = bool(node_report and node_report.get("imported"))
    if opts.send_packet:
        summary["packet_sent"] = bool(packet_dispatch and packet_dispatch.get("sent"))
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "started_at": started_at,
        "model": opts.model,
        "serve_port": opts.serve_port,
        "mode": opts.mode,
        "summary": summary,
        "nodes": report_rows,
    }
    if opts.import_reports:
        report["node_report"] = node_report
    if opts.send_packet:
        report["packet_dispatch"] = packet_dispatch
    return report


def watch(
    opts: WatchOptions,
    *,
    resolver: Resolver = smoke.resolve_tailscale_node,
    runner: Runner = smoke.main,
    sleeper: Sleeper = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    report_importer: ReportImporter = import_node_report_once,
    packet_dispatcher: PacketDispatcher = dispatch_packets_once,
) -> tuple[int, dict[str, Any]]:
    started_at = utc_now()
    deadline = monotonic() + opts.max_wait_s
    rows: list[dict[str, Any]] = []
    terminal: dict[str, dict[str, Any]] = {}
    node_report: dict[str, Any] | None = None
    packet_dispatch: dict[str, Any] | None = packet_dispatcher(opts) if opts.send_packet else None
    while True:
        rows = poll_once(opts, resolver=resolver, runner=runner, prior=terminal)
        if opts.import_reports:
            node_report = report_importer(opts)
        terminal.update({
            str(row["node"]): row for row in rows
            if row.get("status") in {"PASS", "FAIL"}
        })
        if complete(rows, opts.mode):
            return 0, build_report(rows, opts, started_at, timed_out=False, node_report=node_report, packet_dispatch=packet_dispatch)
        if rows and all(row.get("status") in {"PASS", "FAIL"} for row in rows):
            return 1, build_report(rows, opts, started_at, timed_out=False, node_report=node_report, packet_dispatch=packet_dispatch)
        if opts.max_wait_s <= 0 or monotonic() >= deadline:
            return 1, build_report(rows, opts, started_at, timed_out=opts.max_wait_s > 0, node_report=node_report, packet_dispatch=packet_dispatch)
        sleeper(min(opts.poll_interval_s, max(0.0, deadline - monotonic())))


def write_report(path: str, report: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Poll Qwen3.6 Tailscale test beds and smoke them once ready")
    ap.add_argument("--node", action="append", required=True, help="Tailscale node name/DNS/IP; repeatable")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--serve-port", type=int, default=8131)
    ap.add_argument("--out-dir", default="fak/experiments/qwen36")
    ap.add_argument("--out", default="", help="watch summary JSON path")
    ap.add_argument("--http-timeout-s", type=float, default=5.0)
    ap.add_argument("--agent-max-turns", type=int, default=1)
    ap.add_argument("--agent-timeout-s", type=float, default=900.0)
    ap.add_argument("--gateway-chat", action="store_true")
    ap.add_argument("--perf-decode-baseline-tps", type=float, default=0.0, help="annotate gateway chat decode tok/s as a ratio against this baseline")
    ap.add_argument("--min-decode-tps", type=float, default=0.0, help="fail gateway chat if measured decode tok/s is below this threshold")
    ap.add_argument("--poll-interval-s", type=float, default=30.0)
    ap.add_argument("--max-wait-s", type=float, default=0.0, help="0 means poll once")
    ap.add_argument("--mode", choices=["any", "all"], default="all")
    ap.add_argument("--import-reports", action="store_true", help="receive and attach returned node report bundles each poll")
    ap.add_argument("--report-inbox", type=Path, default=node_reports.DEFAULT_INBOX)
    ap.add_argument("--report-out-dir", type=Path, default=node_reports.DEFAULT_OUT_DIR)
    ap.add_argument("--report-archive", type=Path, default=None, help="specific qwen36-node-reports-*.zip to import")
    ap.add_argument("--skip-report-taildrop", action="store_true", help="import reports without running tailscale file get first")
    ap.add_argument("--report-log-tail-lines", type=int, default=60)
    ap.add_argument("--registry", default="", help="endpoint registry path; defaults to tools/fleet_endpoints.local.json/json/example")
    ap.add_argument("--send-packet", action="store_true", help="Taildrop node packet/bootstrap files before polling")
    ap.add_argument("--packet-out-dir", type=Path, default=ROOT / "tools" / "_registry" / "qwen36-watch-packets")
    ap.add_argument("--packet-report-target", default=node_packet.AUTO_REPORT_TARGET)
    ap.add_argument("--packet-taildrop-retries", type=int, default=1)
    ap.add_argument("--packet-taildrop-retry-delay-s", type=float, default=5.0)
    ap.add_argument("--packet-taildrop-timeout-s", type=float, default=15.0)
    ap.add_argument("--no-packet-bootstrap", dest="packet_bootstrap", action="store_false", default=True)
    ap.add_argument(
        "--packet-profile",
        choices=["auto", "mac", "linux", "linux-nvidia", "dgx", "nvidia", "vulkan", "windows", "both", "all"],
        default="auto",
        help="override node-packet profile selection when --send-packet is used",
    )
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    report_inbox = args.report_inbox
    if not report_inbox.is_absolute():
        report_inbox = ROOT / report_inbox
    report_out_dir = args.report_out_dir
    if not report_out_dir.is_absolute():
        report_out_dir = ROOT / report_out_dir
    report_archive = args.report_archive
    if report_archive is not None and not report_archive.is_absolute():
        report_archive = ROOT / report_archive
    packet_out_dir = args.packet_out_dir
    if not packet_out_dir.is_absolute():
        packet_out_dir = ROOT / packet_out_dir
    opts = WatchOptions(
        nodes=args.node,
        model=args.model,
        serve_port=args.serve_port,
        out_dir=out_dir,
        http_timeout_s=args.http_timeout_s,
        agent_max_turns=args.agent_max_turns,
        agent_timeout_s=args.agent_timeout_s,
        gateway_chat=args.gateway_chat,
        perf_decode_baseline_tps=max(0.0, args.perf_decode_baseline_tps),
        min_decode_tps=max(0.0, args.min_decode_tps),
        poll_interval_s=max(0.1, args.poll_interval_s),
        max_wait_s=max(0.0, args.max_wait_s),
        mode=args.mode,
        import_reports=args.import_reports,
        report_inbox=report_inbox,
        report_out_dir=report_out_dir,
        report_archive=report_archive,
        skip_report_taildrop=args.skip_report_taildrop,
        report_log_tail_lines=max(1, args.report_log_tail_lines),
        registry=args.registry,
        send_packet=args.send_packet,
        packet_out_dir=packet_out_dir,
        packet_report_target=args.packet_report_target,
        packet_taildrop_retries=max(1, args.packet_taildrop_retries),
        packet_taildrop_retry_delay_s=max(0.0, args.packet_taildrop_retry_delay_s),
        packet_taildrop_timeout_s=max(1.0, args.packet_taildrop_timeout_s),
        packet_bootstrap=args.packet_bootstrap,
        packet_profile=args.packet_profile,
    )
    rc, report = watch(opts)
    body = json.dumps(report["summary"], indent=2)
    print(body)
    if args.out:
        write_report(args.out, report)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
