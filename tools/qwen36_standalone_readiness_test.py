#!/usr/bin/env python3
"""Smoke tests for qwen36_standalone_readiness.py."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import qwen36_standalone_readiness as readiness


def fake_slack_helpers(root: Path) -> Path:
    slack = root / "slack-helpers"
    (slack / "slack_helpers").mkdir(parents=True)
    (slack / "examples").mkdir(parents=True)
    (slack / "README.md").write_text("slack helpers\n", encoding="utf-8")
    (slack / "install-slack-control-dgx.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (slack / "slack_helpers" / "cli.py").write_text("# cli\n", encoding="utf-8")
    (slack / "slack_helpers" / "control.py").write_text("# control\n", encoding="utf-8")
    (slack / "examples" / "slack_control_local_demo.py").write_text("# demo\n", encoding="utf-8")
    return slack


def write_watch(path: Path) -> None:
    path.write_text(json.dumps({
        "schema": "fak.qwen36-node-watch.v1",
        "generated_at": "2026-06-19T00:00:00Z",
        "summary": {
            "nodes": 1,
            "passed": 0,
            "waiting": 1,
            "failed": 0,
            "timed_out": False,
            "action_required": True,
        },
        "nodes": [{
            "node": "anthony",
            "state": "ONLINE_NO_SERVE",
            "status": "WAIT",
            "detail": "serve port closed",
            "smoke_ran": False,
            "next_actions": [{
                "kind": "run_node_launcher",
                "required": True,
                "detail": "run the bootstrap launcher",
            }],
        }],
        "packet_dispatch": {
            "sent": True,
            "report_target": "node-desktop-b",
        },
    }), encoding="utf-8")


def write_send_watch(path: Path) -> None:
    path.write_text(json.dumps({
        "schema": "fak.qwen36-node-watch.v1",
        "generated_at": "2026-06-19T00:01:00Z",
        "summary": {
            "nodes": 1,
            "passed": 0,
            "waiting": 1,
            "failed": 0,
            "timed_out": False,
            "action_required": True,
        },
        "nodes": [{
            "node": "anthony",
            "state": "ONLINE_NO_SERVE",
            "status": "WAIT",
            "detail": "serve port closed",
            "smoke_ran": False,
            "next_actions": [{
                "kind": "send_packet",
                "required": True,
                "detail": "send a node packet",
            }],
        }],
    }), encoding="utf-8")


def write_verified_ssh_watch(path: Path) -> None:
    path.write_text(json.dumps({
        "schema": "fak.qwen36-node-watch.v1",
        "generated_at": "2026-06-19T00:02:00Z",
        "summary": {
            "nodes": 1,
            "passed": 0,
            "waiting": 1,
            "failed": 0,
            "timed_out": False,
            "action_required": True,
        },
        "nodes": [{
            "node": "anthony",
            "state": "ONLINE_NO_SERVE",
            "status": "WAIT",
            "detail": "node accepts SSH, serve port closed",
            "smoke_ran": False,
            "next_actions": [
                {
                    "kind": "send_packet",
                    "required": True,
                    "detail": "send a node packet",
                },
                {
                    "kind": "ssh_remote_start_available",
                    "required": False,
                    "detail": "SSH auth is verified for this driver",
                    "host": "100.64.0.10",
                    "port": 22,
                },
            ],
        }],
    }), encoding="utf-8")


def write_preflight_ok_watch(path: Path) -> None:
    path.write_text(json.dumps({
        "schema": "fak.qwen36-node-watch.v1",
        "generated_at": "2026-06-19T00:03:00Z",
        "summary": {
            "nodes": 1,
            "passed": 0,
            "waiting": 1,
            "failed": 0,
            "timed_out": False,
            "action_required": True,
            "node_report_imported": True,
        },
        "nodes": [{
            "node": "anthony",
            "state": "ONLINE_NO_SERVE",
            "status": "WAIT",
            "detail": "preflight ok, serve port closed",
            "smoke_ran": False,
            "next_actions": [],
        }],
        "node_report": {
            "imported": True,
            "status": "PREFLIGHT_OK",
            "archive": r"C:\Users\USER\Downloads\qwen36-node-reports-nvidia.zip",
            "report_dir": r"C:\work\fleet\fak\experiments\qwen36\node-reports\qwen36-node-reports-nvidia",
            "latest_preflight": {
                "path": r"C:\work\fleet\fak\experiments\qwen36\node-reports\qwen36-node-reports-nvidia\preflight-nvidia.json",
                "parsed": True,
                "ok": True,
                "profile": "nvidia",
                "base_url": "http://100.64.0.10:8131/v1",
                "llama_server_found": True,
                "failed_checks": [],
                "nvidia_smi": {
                    "name": "nvidia_smi",
                    "ok": True,
                    "gpus": [{"name": "NVIDIA GeForce RTX 4070 Laptop GPU"}],
                },
            },
        },
    }), encoding="utf-8")


def write_node_report_dir(path: Path) -> None:
    reports_dir = path / "qwen36-reports"
    reports_dir.mkdir(parents=True)
    (reports_dir / "preflight-nvidia-remote-20260619-141154.json").write_text(json.dumps({
        "ok": True,
        "profile": "nvidia",
        "base_url": "http://100.64.0.10:8131/v1",
        "llama_server_found": True,
        "failures": [],
        "checks": [
            {"name": "llama_server", "ok": True},
            {
                "name": "nvidia_smi",
                "ok": True,
                "gpus": [{"name": "NVIDIA GeForce RTX 4070 Laptop GPU"}],
            },
        ],
    }), encoding="utf-8")


def write_packeted_watch(path: Path) -> None:
    path.write_text(json.dumps({
        "schema": "fak.qwen36-node-watch.v1",
        "generated_at": "2026-06-19T00:00:00Z",
        "summary": {
            "nodes": 1,
            "passed": 0,
            "waiting": 1,
            "failed": 0,
            "timed_out": False,
            "action_required": True,
            "packet_sent": True,
        },
        "nodes": [{
            "node": "anthony",
            "state": "ONLINE_NO_SERVE",
            "status": "WAIT",
            "detail": "serve port closed",
            "smoke_ran": False,
            "next_actions": [
                {
                    "kind": "run_node_launcher",
                    "required": True,
                    "detail": "run the received bootstrap launcher",
                    "bootstrap_files": ["START-QWEN36-NVIDIA.cmd", "START-QWEN36-NVIDIA.ps1"],
                    "packet_dir": r"C:\packets\anthony",
                    "report_target": "node-desktop-b",
                },
                {
                    "kind": "ssh_auth",
                    "required": False,
                    "detail": "configure SSH auth",
                    "host": "100.64.0.10",
                    "port": 22,
                },
            ],
        }],
        "packet_dispatch": {
            "sent": True,
            "report_target": "node-desktop-b",
        },
    }), encoding="utf-8")


def write_surface_smoke(
    path: Path,
    model: str = "lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M",
    endpoint: dict | None = None,
) -> None:
    body = {
        "schema": "fak.qwen36-surface-smoke.v1",
        "generated_at": "2026-06-19T00:00:00Z",
        "node_name": "mac-local-qwen36",
        "model": model,
        "base_url": "http://127.0.0.1:8131/v1",
        "summary": {"surfaces": 3, "passed": 3, "failed": 0},
        "surfaces": [
            {"surface": "agent", "status": "PASS"},
            {
                "surface": "gateway-openai",
                "status": "PASS",
                "chat_perf": {"decode_tps": 6.036, "decode_vs_baseline": 0.828},
            },
            {"surface": "mcp-http", "status": "PASS"},
        ],
    }
    if endpoint is not None:
        body["endpoint"] = endpoint
    path.write_text(json.dumps(body), encoding="utf-8")


def write_dgx_prep_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "DGX_RUN.json").write_text(json.dumps({
        "schema": "fak.dgx-run-plan.v1",
        "generated_at": "2026-06-19T00:03:00Z",
        "run_id": "prep-001",
        "model": "Qwen3.6-27B",
        "hardware": "gpu-server",
        "benchmark": {"root": "/srv/Benchmark"},
    }), encoding="utf-8")
    (run_dir / "DGX_RUNBOOK.md").write_text("# prep\n", encoding="utf-8")
    (run_dir / "DGX_ONE_TOUCH.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")


def write_dgx_handoff(run_dir: Path) -> None:
    handoff = run_dir / "handoff"
    handoff.mkdir(parents=True)
    (handoff / f"fleet-{run_dir.name}.tgz").write_bytes(b"fake-tarball")
    (handoff / "RUN_ON_DGX.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (handoff / "DGX_HANDOFF.md").write_text("# handoff\n", encoding="utf-8")


def write_dgx_remote_probe(run_dir: Path, resolved: bool = False) -> None:
    (run_dir / "REMOTE_PROBE.dns.json").write_text(json.dumps({
        "schema": "fak.dgx-one-touch.v1.remote-probe",
        "generated_at": "2026-06-19T00:04:00Z",
        "ssh_target": "<ssh-password>-dgx1.example.lab",
        "proxy_jump": "",
        "target_dns": {
            "host": "<ssh-password>-dgx1.example.lab",
            "resolved": resolved,
            "addresses": ["10.0.0.1"] if resolved else [],
            "error": "" if resolved else "[Errno 11001] getaddrinfo failed",
        },
        "jump_dns": None,
        "ssh_argv": ["ssh", "<ssh-password>-dgx1.example.lab", "hostname && nvidia-smi -L"],
        "ssh": {"attempted": False, "returncode": None, "stdout": "", "stderr": ""},
    }), encoding="utf-8")


def write_dgx_completed_run(run_dir: Path) -> None:
    write_dgx_prep_run(run_dir)
    for name in ("PREFLIGHT.static.json", "PREFLIGHT.endpoints.json"):
        (run_dir / name).write_text(json.dumps({"schema": "fak.dgx-preflight.v1", "passed": True}), encoding="utf-8")
    for name in ("raw-sglang.json", "fak-gateway.json"):
        (run_dir / name).write_text(json.dumps({"schema": "fak.dgx-endpoint-bench.v1", "passed": True}), encoding="utf-8")
    (run_dir / "compare.json").write_text(json.dumps([]), encoding="utf-8")
    (run_dir / "COMPARE.md").write_text("# compare\n", encoding="utf-8")
    (run_dir / "MATRIX.json").write_text(json.dumps({"schema": "fak.dgx-run-matrix.v1", "results": []}), encoding="utf-8")
    (run_dir / "GATE.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
    (run_dir / "RUN_GATE.json").write_text(json.dumps({"passed": True, "runs": []}), encoding="utf-8")
    monitor_csv = run_dir / "benchmark-monitor" / "csv"
    monitor_csv.mkdir(parents=True)
    (monitor_csv / "_csv_manifest.json").write_text(json.dumps({"files": []}), encoding="utf-8")


def write_packet_manifest(root: Path, name: str, profiles: list[str], archive: bool = True) -> Path:
    packet_dir = root / name
    payload = packet_dir / "qwen36-node-packet"
    payload.mkdir(parents=True)
    if archive:
        (packet_dir / f"qwen36-node-packet-{name}.zip").write_bytes(b"zip")
    command_maps = {
        "bootstrap_command": {
            "mac": "bash START-QWEN36-MAC.command",
            "linux-nvidia": "bash START-QWEN36-LINUX-NVIDIA.sh",
            "nvidia": ".\\START-QWEN36-NVIDIA.cmd",
            "vulkan": ".\\START-QWEN36-VULKAN.cmd",
        },
        "preflight_command": {
            "mac": "bash RUN-MAC.sh --preflight",
            "linux-nvidia": "bash RUN-LINUX-NVIDIA.sh --preflight",
            "nvidia": ".\\RUN-NVIDIA.ps1 --preflight",
            "vulkan": ".\\RUN-VULKAN.ps1 --preflight",
        },
        "start_command": {
            "mac": "bash START-MAC.command",
            "linux-nvidia": "bash START-LINUX-NVIDIA.sh",
            "nvidia": ".\\START-NVIDIA.cmd",
            "vulkan": ".\\START-VULKAN.cmd",
        },
        "install_command": {
            "mac": "bash INSTALL-MAC.command",
            "linux-nvidia": "bash INSTALL-LINUX-NVIDIA.sh",
            "nvidia": ".\\INSTALL-NVIDIA.cmd",
            "vulkan": ".\\INSTALL-VULKAN.cmd",
        },
        "report_command": {
            "mac": "bash SEND-REPORTS-MAC.sh <driver-tailnet-name>",
            "linux-nvidia": "bash SEND-REPORTS-LINUX-NVIDIA.sh <driver-tailnet-name>",
            "nvidia": ".\\SEND-REPORTS-NVIDIA.ps1 -Target <driver-tailnet-name>",
            "vulkan": ".\\SEND-REPORTS-VULKAN.ps1 -Target <driver-tailnet-name>",
        },
        "node_command": {
            "mac": "bash RUN-MAC.sh",
            "linux-nvidia": "bash RUN-LINUX-NVIDIA.sh",
            "nvidia": ".\\RUN-NVIDIA.ps1",
            "vulkan": ".\\RUN-VULKAN.ps1",
        },
    }
    payload_commands = {
        key: {profile: value for profile, value in mapping.items() if profile in profiles}
        for key, mapping in command_maps.items()
    }
    manifest = {
        "schema": "fak.qwen36-node-packet.v1",
        "generated_at": "2026-06-19T00:00:00Z",
        "model": "Qwen3.6",
        "serve_port": 8131,
        "profiles": profiles,
        "report_target": "driver-node",
        **payload_commands,
        "receive": {profile: [f"receive {profile}", f"install {profile}"] for profile in profiles},
        "driver_command": ".\\SMOKE-FROM-DRIVER.ps1 -Node <tailscale-node-name> -NodeName qwen36-testbed",
    }
    path = payload / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_watch_report_action_required_is_preserved():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        watch_path = base / "node-watch.json"
        write_watch(watch_path)
        report = readiness.build_report(
            experiment_dir=base,
            watch_reports=[watch_path],
            slack_helpers_dir=fake_slack_helpers(base),
            env={},
        )

    watch = report["watch_reports"][0]
    checks = {row["name"]: row for row in report["checks"]}
    assert report["summary"]["operator_action_required"] is True
    assert report["summary"]["target_action_nodes"] == 1
    assert checks["Watcher evidence"]["status"] == "ACTION_REQUIRED"
    assert "1 unsuppressed target action node(s)" in checks["Watcher evidence"]["evidence"]
    assert watch["packet_sent"] is True
    assert watch["packet_report_target"] == "node-desktop-b"
    assert watch["nodes"][0]["next_actions"][0]["kind"] == "run_node_launcher"


def test_discover_watch_reports_prefers_embedded_generated_at_over_mtime():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        older = base / "node-old-watch.json"
        newer = base / "node-new-watch.json"
        write_watch(older)
        write_send_watch(newer)
        os.utime(older, (200, 200))
        os.utime(newer, (100, 100))

        selected = readiness.discover_watch_reports(base, 1)

    assert selected == [newer]


def test_target_next_actions_prefer_packeted_launcher_over_resend():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        send_watch = base / "node-current-watch.json"
        packeted_watch = base / "node-packeted-watch.json"
        write_send_watch(send_watch)
        write_packeted_watch(packeted_watch)
        report = readiness.build_report(
            experiment_dir=base,
            watch_reports=[send_watch, packeted_watch],
            slack_helpers_dir=fake_slack_helpers(base),
            env={},
        )

    assert report["summary"]["target_action_nodes"] == 1
    action_row = report["target_next_actions"][0]
    assert action_row["node"] == "anthony"
    assert action_row["primary_action"]["kind"] == "run_node_launcher"
    assert action_row["primary_action"]["bootstrap_files"] == ["START-QWEN36-NVIDIA.cmd", "START-QWEN36-NVIDIA.ps1"]
    assert action_row["primary_action"]["report_target"] == "node-desktop-b"
    assert action_row["snippets"]["target_commands"] == [
        ".\\START-QWEN36-NVIDIA.cmd",
        "powershell -NoProfile -ExecutionPolicy Bypass -File .\\START-QWEN36-NVIDIA.ps1",
    ]
    assert action_row["snippets"]["target_recover_commands"][-2:] == [
        ".\\RUN-NVIDIA.ps1 --preflight",
        ".\\INSTALL-NVIDIA.cmd",
    ]
    assert action_row["snippets"]["driver_verify_commands"][-1] == "  --out fak\\experiments\\qwen36\\node-qwen36-watch-live.json"
    assert action_row["optional_actions"][0]["kind"] == "ssh_auth"


def test_verified_ssh_action_suppresses_stale_ssh_auth_advice():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        verified_watch = base / "node-current-watch.json"
        packeted_watch = base / "node-packeted-watch.json"
        write_verified_ssh_watch(verified_watch)
        write_packeted_watch(packeted_watch)
        report = readiness.build_report(
            experiment_dir=base,
            watch_reports=[verified_watch, packeted_watch],
            slack_helpers_dir=fake_slack_helpers(base),
            env={},
        )

    action_row = report["target_next_actions"][0]
    assert action_row["primary_action"]["kind"] == "run_node_launcher"
    assert [action["kind"] for action in action_row["optional_actions"]] == ["ssh_remote_start_available"]


def test_imported_target_preflight_has_its_own_readiness_row():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        watch_path = base / "node-preflight-watch.json"
        write_preflight_ok_watch(watch_path)
        report = readiness.build_report(
            experiment_dir=base,
            watch_reports=[watch_path],
            slack_helpers_dir=fake_slack_helpers(base),
            env={},
        )

    checks = {row["name"]: row for row in report["checks"]}
    assert report["summary"]["target_preflight_passes"] == 1
    assert checks["Target standalone preflight"]["status"] == "PASS"
    assert "PREFLIGHT_OK" not in checks["Target standalone endpoint smoke"]["evidence"]
    markdown = readiness.render_markdown(report)
    assert "Target standalone preflight passes: `1`" in markdown
    assert "Node report: `PREFLIGHT_OK`" in markdown


def test_extracted_node_report_dir_can_clear_target_preflight():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        report_dir = base / "node-reports" / "qwen36-node-reports-nvidia"
        write_node_report_dir(report_dir)
        report = readiness.build_report(
            experiment_dir=base,
            watch_reports=[],
            node_report_dir=base / "node-reports",
            slack_helpers_dir=fake_slack_helpers(base),
            env={},
        )

    checks = {row["name"]: row for row in report["checks"]}
    assert report["summary"]["node_reports"] == 1
    assert report["summary"]["target_preflight_passes"] == 1
    assert checks["Target standalone preflight"]["status"] == "PASS"
    markdown = readiness.render_markdown(report)
    assert "Node Reports" in markdown
    assert "qwen36-node-reports-nvidia" in markdown


def test_surface_smoke_proves_local_not_target_endpoint_smoke():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        surface_path = base / "mac-local-qwen36-surfaces.json"
        write_surface_smoke(surface_path)
        report = readiness.build_report(
            experiment_dir=base,
            surface_reports=[surface_path],
            watch_reports=[],
            slack_helpers_dir=fake_slack_helpers(base),
            env={},
        )

    checks = {row["name"]: row for row in report["checks"]}
    assert checks["Local standalone endpoint smoke"]["status"] == "PASS"
    assert checks["Target standalone endpoint smoke"]["status"] == "UNPROVEN"
    assert report["summary"]["surface_smokes"] == 1
    assert report["summary"]["qwen36_surface_passes"] == 1
    assert report["surface_smokes"][0]["passed"] is True
    assert report["surface_smokes"][0]["qwen36_model"] is True
    assert report["surface_smokes"][0]["surfaces"][1]["decode_tps"] == 6.036


def test_target_surface_smoke_can_clear_target_endpoint_smoke():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        surface_path = base / "node-qwen36-surfaces.json"
        write_surface_smoke(
            surface_path,
            model="qwen36-testbed",
            endpoint={
                "name": "anthony",
                "source": "registry:tools/fleet_endpoints.local.json",
                "state": "READY",
                "tailscale_ip": "100.64.0.10",
                "base_url": "http://100.64.0.10:8131/v1",
            },
        )
        report = readiness.build_report(
            experiment_dir=base,
            surface_reports=[surface_path],
            watch_reports=[],
            slack_helpers_dir=fake_slack_helpers(base),
            env={},
        )

    checks = {row["name"]: row for row in report["checks"]}
    assert checks["Local standalone endpoint smoke"]["status"] == "UNPROVEN"
    assert checks["Target standalone endpoint smoke"]["status"] == "PASS"
    assert checks["Target standalone endpoint smoke"]["evidence"] == str(surface_path)
    assert report["surface_smokes"][0]["endpoint"]["name"] == "anthony"


def test_target_surface_smoke_suppresses_stale_target_next_actions():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        watch_path = base / "node-watch.json"
        surface_path = base / "node-qwen36-surfaces.json"
        write_send_watch(watch_path)
        write_surface_smoke(
            surface_path,
            model="qwen36-testbed",
            endpoint={
                "name": "anthony",
                "source": "tailscale-status",
                "state": "READY",
                "tailscale_ip": "100.64.0.10",
                "base_url": "http://100.64.0.10:8131/v1",
            },
        )
        report = readiness.build_report(
            experiment_dir=base,
            surface_reports=[surface_path],
            watch_reports=[watch_path],
            slack_helpers_dir=fake_slack_helpers(base),
            env={},
        )

    assert report["summary"]["target_action_nodes"] == 0
    assert report["target_next_actions"] == []
    assert "anthony" in report["target_surface_pass_nodes"]
    checks = {row["name"]: row for row in report["checks"]}
    assert checks["Watcher evidence"]["status"] == "PASS"
    assert "0 unsuppressed target action node(s)" in checks["Watcher evidence"]["evidence"]
    assert "historical watch actions cleared by target evidence" in checks["Watcher evidence"]["evidence"]
    assert report["summary"]["operator_action_required"] is True
    assert "Slack control thread" in report["summary"]["unproven_external_gates"]
    assert "Live DGX run" in report["summary"]["unproven_external_gates"]
    rendered = readiness.render_markdown(report)
    assert "cleared by imported target surface smoke" in rendered
    assert "ONLINE_NO_SERVE" not in rendered


def test_other_model_surface_smoke_does_not_prove_qwen36():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        surface_path = base / "local-qwen25-surfaces.json"
        write_surface_smoke(surface_path, model="local-qwen2.5-15b")
        report = readiness.build_report(
            experiment_dir=base,
            surface_reports=[surface_path],
            watch_reports=[],
            slack_helpers_dir=fake_slack_helpers(base),
            env={},
        )

    checks = {row["name"]: row for row in report["checks"]}
    assert checks["Local standalone endpoint smoke"]["status"] == "UNPROVEN"
    assert report["summary"]["qwen36_surface_passes"] == 0
    assert report["surface_smokes"][0]["passed"] is True
    assert report["surface_smokes"][0]["qwen36_model"] is False


def test_dgx_prep_only_run_is_reported_but_does_not_clear_live_gate():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        run_dir = base / "dgx" / "prep-only"
        write_dgx_prep_run(run_dir)
        report = readiness.build_report(
            experiment_dir=base,
            watch_reports=[],
            dgx_runs=[run_dir],
            slack_helpers_dir=fake_slack_helpers(base),
            env={},
        )

    checks = {row["name"]: row for row in report["checks"]}
    assert report["summary"]["dgx_runs"] == 1
    assert report["summary"]["dgx_run_passes"] == 0
    assert report["summary"]["latest_dgx_run_status"] == "PREP_ONLY"
    assert report["dgx_runs"][0]["status"] == "PREP_ONLY"
    assert report["dgx_runs"][0]["generated_at"] == "2026-06-19T00:03:00Z"
    assert len(report["dgx_runs"][0]["plan_sha256"]) == 64
    assert len(report["dgx_runs"][0]["runbook_sha256"]) == 64
    assert "PREFLIGHT.static.json" in report["dgx_runs"][0]["missing_artifacts"]
    assert checks["Live DGX run"]["status"] == "UNPROVEN"
    assert "prep-only" in checks["Live DGX run"]["evidence"]
    assert "Packet: generated `2026-06-19T00:03:00Z`" in readiness.render_markdown(report)


def test_dgx_handoff_bundle_is_reported_as_ready_for_manual_transfer():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        run_dir = base / "dgx" / "prep-only"
        write_dgx_prep_run(run_dir)
        write_dgx_handoff(run_dir)
        report = readiness.build_report(
            experiment_dir=base,
            watch_reports=[],
            dgx_runs=[run_dir],
            slack_helpers_dir=fake_slack_helpers(base),
            env={},
        )

    checks = {row["name"]: row for row in report["checks"]}
    handoff = report["dgx_runs"][0]["handoff"]
    assert report["summary"]["dgx_handoff_bundles"] == 1
    assert checks["DGX handoff bundle"]["status"] == "PASS"
    assert handoff["complete"] is True
    assert handoff["archive"].endswith("fleet-prep-only.tgz")
    assert len(handoff["archive_sha256"]) == 64
    rendered = readiness.render_markdown(report)
    assert "DGX handoff bundles: `1`" in rendered
    assert "Handoff: complete `True`" in rendered


def test_dgx_remote_probe_dns_failure_is_reported_without_blocking_handoff():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        run_dir = base / "dgx" / "prep-only"
        write_dgx_prep_run(run_dir)
        write_dgx_remote_probe(run_dir, resolved=False)
        report = readiness.build_report(
            experiment_dir=base,
            watch_reports=[],
            dgx_runs=[run_dir],
            slack_helpers_dir=fake_slack_helpers(base),
            env={},
        )

    checks = {row["name"]: row for row in report["checks"]}
    probe = report["dgx_runs"][0]["remote_probe"]
    assert report["summary"]["dgx_remote_probes"] == 1
    assert checks["DGX remote probe"]["status"] == "WARN"
    assert probe["status"] == "DNS_FAILED"
    assert probe["ssh_target"] == "<ssh-password>-dgx1.example.lab"
    assert probe["target_dns_resolved"] is False
    rendered = readiness.render_markdown(report)
    assert "DGX remote probes: `1`" in rendered
    assert "Remote probe: status `DNS_FAILED`" in rendered


def test_dgx_completed_run_clears_live_dgx_gate():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        run_dir = base / "dgx" / "complete"
        write_dgx_completed_run(run_dir)
        report = readiness.build_report(
            experiment_dir=base,
            watch_reports=[],
            dgx_runs=[run_dir],
            slack_helpers_dir=fake_slack_helpers(base),
            env={},
        )

    checks = {row["name"]: row for row in report["checks"]}
    assert report["summary"]["dgx_runs"] == 1
    assert report["summary"]["dgx_run_passes"] == 1
    assert report["summary"]["latest_dgx_run_status"] == "PASS"
    assert report["dgx_runs"][0]["status"] == "PASS"
    assert report["dgx_runs"][0]["endpoint_report_count"] == 2
    assert report["dgx_runs"][0]["benchmark_monitor_manifest"] is True
    assert checks["Live DGX run"]["status"] == "PASS"
    assert "Live DGX run" not in report["summary"]["unproven_external_gates"]


def test_packet_profile_matrix_summarizes_install_bench_commands():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        write_packet_manifest(base, "all", ["mac", "linux-nvidia", "nvidia"], archive=True)
        write_packet_manifest(base, "vulkan-manifest-only", ["vulkan"], archive=False)
        packets = readiness.packet_artifacts([base])

    rows = {row["profile"]: row for row in packets["profile_matrix"]}
    assert packets["profile_count"] == 4
    assert packets["profiles_prepared"] == 3
    assert rows["mac"]["status"] == "PREPARED"
    assert rows["mac"]["bootstrap_command"] == "bash START-QWEN36-MAC.command"
    assert rows["linux-nvidia"]["install_command"] == "bash INSTALL-LINUX-NVIDIA.sh"
    assert rows["nvidia"]["preflight_command"] == ".\\RUN-NVIDIA.ps1 --preflight"
    assert rows["vulkan"]["status"] == "MANIFEST_ONLY"
    assert rows["vulkan"]["archive"] == ""
    assert rows["vulkan"]["receive"] == ["receive vulkan", "install vulkan"]


def test_slack_dry_run_uses_safe_default_user_without_exposing_tokens():
    calls = []

    def runner(command, cwd, env, timeout_s):
        calls.append((list(command), cwd, dict(env), timeout_s))
        return subprocess.CompletedProcess(command, 0, stdout="dry ok\n", stderr="")

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        report = readiness.build_report(
            experiment_dir=base,
            watch_reports=[],
            slack_helpers_dir=fake_slack_helpers(base),
            env={},
            run_slack_dry_run=True,
            run_slack_live_probe=True,
            runner=runner,
        )

    assert report["slack_control"]["dry_run"]["ok"] is True
    assert report["slack_control"]["live_probe"]["skipped"] is True
    assert report["slack_control"]["live_probe"]["reason"] == "SLACK_BOT_TOKEN or SLACK_USER_TOKEN is required for live probe"
    assert calls[0][2]["SLACK_CONTROL_USERS"] == "UTEST"
    assert "SLACK_BOT_TOKEN" not in calls[0][2]
    assert report["slack_control"]["env"]["SLACK_BOT_TOKEN"]["set"] is False
    assert report["slack_next_actions"]["status"] == "NEEDS_CONFIGURATION"
    assert report["slack_next_actions"]["live_probe_status"] == "SKIPPED"
    assert "SLACK_BOT_TOKEN or SLACK_USER_TOKEN" in report["slack_next_actions"]["missing_requirements"]
    assert "SLACK_CONTROL_USERS" in report["slack_next_actions"]["missing_requirements"]
    assert "SLACK_CONTROL_COMMAND" in report["slack_next_actions"]["missing_requirements"]
    assert report["slack_next_actions"]["setup_commands"][0] == 'export SLACK_BOT_TOKEN="xoxb-..."'
    assert "--channel dgx-control --probe" in report["slack_next_actions"]["setup_commands"][3]
    assert "--state-file /var/lib/slack-control/state.json" in report["slack_next_actions"]["service_install_commands"][0]
    assert "--lock-file /var/lib/slack-control/state.json.lock" in report["slack_next_actions"]["service_install_commands"][0]
    assert "--transcript-file /var/lib/slack-control/state.transcript.jsonl" in report["slack_next_actions"]["service_install_commands"][0]


def test_slack_local_control_demo_is_recorded_without_live_credentials():
    calls = []

    def runner(command, cwd, env, timeout_s):
        calls.append((list(command), cwd, dict(env), timeout_s))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="local slack-control demo: OK\nthread_ts=1.000000\n",
            stderr="",
        )

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        report = readiness.build_report(
            experiment_dir=base,
            watch_reports=[],
            slack_helpers_dir=fake_slack_helpers(base),
            env={},
            run_slack_local_demo=True,
            runner=runner,
        )

    checks = {row["name"]: row for row in report["checks"]}
    assert report["summary"]["slack_local_demo_ok"] is True
    assert checks["Slack local-control demo"]["status"] == "PASS"
    assert report["slack_control"]["local_demo"]["ok"] is True
    assert report["slack_control"]["local_demo_command"][-1] == "examples/slack_control_local_demo.py"
    assert calls[0][0][-1] == "examples/slack_control_local_demo.py"
    assert "local slack-control demo: OK" in readiness.render_markdown(report)


def test_slack_next_actions_ready_when_live_env_is_present():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        report = readiness.build_report(
            experiment_dir=base,
            watch_reports=[],
            slack_helpers_dir=fake_slack_helpers(base),
            slack_workdir="/srv/fleet",
            env={
                "SLACK_BOT_TOKEN": "xoxb-redacted",
                "SLACK_CONTROL_USERS": "UTEST",
                "SLACK_CONTROL_COMMAND": "bash -lc 'cd /srv/fleet && exec bash -li'",
            },
        )

    actions = report["slack_next_actions"]
    assert actions["status"] == "READY_FOR_LIVE_PROBE"
    assert actions["missing_requirements"] == []
    assert actions["suggested_control_command"] == "bash -lc 'cd /srv/fleet && exec bash -li'"
    assert actions["state_file"] == readiness.DEFAULT_SLACK_STATE_FILE
    assert actions["lock_file"] == readiness.DEFAULT_SLACK_LOCK_FILE
    assert actions["transcript_file"] == readiness.DEFAULT_SLACK_TRANSCRIPT_FILE
    assert "--resume --state-file /var/lib/slack-control/state.json" in actions["foreground_control_command"]
    assert "--lock-file /var/lib/slack-control/state.json.lock" in actions["foreground_control_command"]
    assert "--transcript-file /var/lib/slack-control/state.transcript.jsonl" in actions["foreground_control_command"]
    assert "--cwd /srv/fleet" in actions["foreground_control_command"]
    assert '--command "$SLACK_CONTROL_COMMAND"' in actions["foreground_control_command"]
    assert actions["live_probe_commands"][0] == "python -m slack_helpers.cli control --channel dgx-control --probe"
    assert actions["live_probe_commands"][1] == "sudo journalctl -u slack-control -f"
    assert actions["live_probe_commands"][2] == "sudo tail -f /var/lib/slack-control/state.transcript.jsonl"


def test_slack_live_probe_pass_is_recorded_without_token_output():
    calls = []

    def runner(command, cwd, env, timeout_s):
        calls.append((list(command), cwd, dict(env), timeout_s))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="control probe: dgx-control (C0EXAMPLE00)\nauth: OK team=T team_id=T123 user=bot user_id=U123 bot_id=B123\nchannel: OK\nmembership: OK\nhistory-read: OK\n",
            stderr="",
        )

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        report = readiness.build_report(
            experiment_dir=base,
            watch_reports=[],
            slack_helpers_dir=fake_slack_helpers(base),
            env={
                "SLACK_BOT_TOKEN": "xoxb-redacted-secret",
                "SLACK_CONTROL_USERS": "UTEST",
                "SLACK_CONTROL_COMMAND": "bash -lc 'cd /srv/fleet && exec bash -li'",
            },
            run_slack_live_probe=True,
            runner=runner,
        )

    assert calls[0][0][-1] == "--probe"
    assert calls[0][2]["SLACK_BOT_TOKEN"] == "xoxb-redacted-secret"
    assert report["slack_control"]["live_probe"]["ok"] is True
    assert "xoxb-redacted-secret" not in report["slack_control"]["live_probe"]["stdout"]
    assert report["slack_next_actions"]["live_probe_status"] == "PASS"
    assert report["slack_next_actions"]["status"] == "READY_FOR_SERVICE_INSTALL"


def test_slack_live_probe_failure_blocks_service_install_status():
    def runner(command, cwd, env, timeout_s):
        return subprocess.CompletedProcess(command, 1, stdout="control probe: dgx-control\nchannel: FAIL\n", stderr="")

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        report = readiness.build_report(
            experiment_dir=base,
            watch_reports=[],
            slack_helpers_dir=fake_slack_helpers(base),
            env={
                "SLACK_BOT_TOKEN": "xoxb-redacted",
                "SLACK_CONTROL_USERS": "UTEST",
                "SLACK_CONTROL_COMMAND": "bash -lc 'cd /srv/fleet && exec bash -li'",
            },
            run_slack_live_probe=True,
            runner=runner,
        )

    assert report["slack_control"]["live_probe"]["ok"] is False
    assert report["slack_next_actions"]["live_probe_status"] == "FAIL"
    assert report["slack_next_actions"]["status"] == "PROBE_FAILED"


def test_markdown_keeps_external_gates_visible():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        report = readiness.build_report(
            experiment_dir=base,
            watch_reports=[],
            slack_helpers_dir=fake_slack_helpers(base),
            env={},
        )
    markdown = readiness.render_markdown(report)
    assert "Slack control thread" in markdown
    assert "Target standalone endpoint smoke" in markdown
    assert "Live DGX run" in markdown
    assert "Standalone Install Benches" in markdown
    assert "DGX/lab host setup" in markdown
    assert "Foreground bridge smoke before systemd" in markdown
    assert "Install as a systemd service" in markdown
    assert "External Gates" in markdown


if __name__ == "__main__":
    test_watch_report_action_required_is_preserved()
    test_discover_watch_reports_prefers_embedded_generated_at_over_mtime()
    test_target_next_actions_prefer_packeted_launcher_over_resend()
    test_verified_ssh_action_suppresses_stale_ssh_auth_advice()
    test_imported_target_preflight_has_its_own_readiness_row()
    test_extracted_node_report_dir_can_clear_target_preflight()
    test_surface_smoke_proves_local_not_target_endpoint_smoke()
    test_target_surface_smoke_can_clear_target_endpoint_smoke()
    test_target_surface_smoke_suppresses_stale_target_next_actions()
    test_other_model_surface_smoke_does_not_prove_qwen36()
    test_dgx_prep_only_run_is_reported_but_does_not_clear_live_gate()
    test_dgx_handoff_bundle_is_reported_as_ready_for_manual_transfer()
    test_dgx_remote_probe_dns_failure_is_reported_without_blocking_handoff()
    test_dgx_completed_run_clears_live_dgx_gate()
    test_packet_profile_matrix_summarizes_install_bench_commands()
    test_slack_dry_run_uses_safe_default_user_without_exposing_tokens()
    test_slack_local_control_demo_is_recorded_without_live_credentials()
    test_slack_next_actions_ready_when_live_env_is_present()
    test_slack_live_probe_pass_is_recorded_without_token_output()
    test_slack_live_probe_failure_blocks_service_install_status()
    test_markdown_keeps_external_gates_visible()
    print("PASS qwen36_standalone_readiness_test")
