#!/usr/bin/env python3
"""Smoke tests for qwen36_node_packet.py."""
from __future__ import annotations

import json
import subprocess
import tempfile
import zipfile
from pathlib import Path

import qwen36_node_packet as packet


def test_packet_contains_launchers_and_manifest():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)

        manifest = packet.write_packet(packet.ROOT, out, ["mac", "nvidia", "vulkan"], packet.DEFAULT_MODEL, 8131)

        payload = Path(manifest["payload_dir"])
        assert (payload / "qwen36_node_server.py").exists()
        assert (payload / "RUN-MAC.sh").exists()
        assert (payload / "START-MAC.command").exists()
        assert (payload / "INSTALL-MAC.command").exists()
        assert (payload / "SEND-REPORTS-MAC.sh").exists()
        assert (payload / "RUN-NVIDIA.ps1").exists()
        assert (payload / "START-NVIDIA.cmd").exists()
        assert (payload / "INSTALL-NVIDIA.cmd").exists()
        assert (payload / "SEND-REPORTS-NVIDIA.ps1").exists()
        assert (payload / "RUN-VULKAN.ps1").exists()
        assert (payload / "START-VULKAN.cmd").exists()
        assert (payload / "INSTALL-VULKAN.cmd").exists()
        assert (payload / "SEND-REPORTS-VULKAN.ps1").exists()
        assert (payload / "SMOKE-FROM-DRIVER.ps1").exists()
        assert (out / "START-QWEN36-MAC.command").exists()
        assert (out / "START-QWEN36-NVIDIA.cmd").exists()
        assert (out / "START-QWEN36-NVIDIA.ps1").exists()
        assert (out / "START-QWEN36-VULKAN.cmd").exists()
        assert (out / "START-QWEN36-VULKAN.ps1").exists()
        data = json.loads((payload / "manifest.json").read_text(encoding="utf-8"))
        assert data["schema"] == packet.SCHEMA
        assert data["serve_port"] == 8131
        assert data["profiles"] == ["mac", "nvidia", "vulkan"]
        assert data["bootstrap_files"] == [
            "START-QWEN36-MAC.command",
            "START-QWEN36-NVIDIA.cmd",
            "START-QWEN36-NVIDIA.ps1",
            "START-QWEN36-VULKAN.cmd",
            "START-QWEN36-VULKAN.ps1",
        ]
        assert data["bootstrap_command"]["mac"] == "bash START-QWEN36-MAC.command"
        assert data["bootstrap_command"]["nvidia"] == ".\\START-QWEN36-NVIDIA.cmd"
        assert data["bootstrap_command"]["vulkan"] == ".\\START-QWEN36-VULKAN.cmd"
        assert "tailscale file get" in " ".join(data["receive"]["mac"])
        assert "tailscale file get" in " ".join(data["receive"]["nvidia"])
        assert "tailscale file get" in " ".join(data["receive"]["vulkan"])
        assert data["preflight_command"]["mac"] == "bash RUN-MAC.sh --preflight"
        assert data["preflight_command"]["nvidia"] == ".\\RUN-NVIDIA.ps1 --preflight"
        assert data["preflight_command"]["vulkan"] == ".\\RUN-VULKAN.ps1 --preflight"
        assert data["start_command"]["mac"] == "bash START-MAC.command"
        assert data["start_command"]["nvidia"] == ".\\START-NVIDIA.cmd"
        assert data["start_command"]["vulkan"] == ".\\START-VULKAN.cmd"
        assert data["install_command"]["mac"] == "bash INSTALL-MAC.command"
        assert data["install_command"]["nvidia"] == ".\\INSTALL-NVIDIA.cmd"
        assert data["install_command"]["vulkan"] == ".\\INSTALL-VULKAN.cmd"
        assert data["report_command"]["mac"] == "bash SEND-REPORTS-MAC.sh <driver-tailnet-name>"
        assert data["report_command"]["nvidia"] == ".\\SEND-REPORTS-NVIDIA.ps1 -Target <driver-tailnet-name>"
        assert data["report_command"]["vulkan"] == ".\\SEND-REPORTS-VULKAN.ps1 -Target <driver-tailnet-name>"
        assert "--bind tailnet" in (payload / "RUN-MAC.sh").read_text(encoding="utf-8")
        assert "--profile nvidia" in (payload / "RUN-NVIDIA.ps1").read_text(encoding="utf-8")
        assert "--profile vulkan" in (payload / "RUN-VULKAN.ps1").read_text(encoding="utf-8")
        mac_start = (payload / "START-MAC.command").read_text(encoding="utf-8")
        nvidia_start = (payload / "START-NVIDIA.cmd").read_text(encoding="utf-8")
        vulkan_start = (payload / "START-VULKAN.cmd").read_text(encoding="utf-8")
        assert "bash RUN-MAC.sh --preflight" in mac_start
        assert "qwen36-reports/preflight-mac-" in mac_start
        assert "tee \"$server_log\"" in mac_start
        assert 'caffeinate -dimsu -w "$$"' in mac_start
        assert "ExecutionPolicy Bypass" in nvidia_start
        assert "qwen36-reports\\preflight-nvidia-" in nvidia_start
        assert "Tee-Object" in nvidia_start
        assert "RUN-VULKAN.ps1" in vulkan_start
        assert "qwen36-reports\\preflight-vulkan-" in vulkan_start
        assert "Tee-Object" in vulkan_start
        mac_install = (payload / "INSTALL-MAC.command").read_text(encoding="utf-8")
        assert "brew install llama.cpp" in mac_install
        assert 'caffeinate -dimsu -w "$$"' in mac_install
        assert "winget install llama.cpp" in (payload / "INSTALL-NVIDIA.cmd").read_text(encoding="utf-8")
        nvidia_install = (payload / "INSTALL-NVIDIA.cmd").read_text(encoding="utf-8")
        vulkan_install = (payload / "INSTALL-VULKAN.cmd").read_text(encoding="utf-8")
        assert "winget install llama.cpp" in nvidia_install
        assert "winget install llama.cpp" in vulkan_install
        assert "rerun START-NVIDIA.cmd" in nvidia_install
        assert "rerun START-VULKAN.cmd" in vulkan_install
        assert "rerun START-NVIDIA.cmd" not in vulkan_install
        assert "tailscale file cp" in (payload / "SEND-REPORTS-MAC.sh").read_text(encoding="utf-8")
        assert "Compress-Archive" in (payload / "SEND-REPORTS-NVIDIA.ps1").read_text(encoding="utf-8")
        assert "Compress-Archive" in (payload / "SEND-REPORTS-VULKAN.ps1").read_text(encoding="utf-8")
        readme = (payload / "README.md").read_text(encoding="utf-8")
        assert "tailscale file get" in readme
        assert "--preflight" in readme
        assert "START-" in readme
        assert "INSTALL-" in readme
        assert "qwen36-reports" in readme
        assert "SEND-REPORTS" in readme
        assert "SEND-REPORTS-LINUX-NVIDIA.sh" in readme
        assert "SEND-REPORTS-VULKAN.ps1" in readme
        assert "nvidia-smi" in readme
        mac_bootstrap = (out / "START-QWEN36-MAC.command").read_text(encoding="utf-8")
        nvidia_cmd = (out / "START-QWEN36-NVIDIA.cmd").read_text(encoding="utf-8")
        nvidia_ps1 = (out / "START-QWEN36-NVIDIA.ps1").read_text(encoding="utf-8")
        vulkan_cmd = (out / "START-QWEN36-VULKAN.cmd").read_text(encoding="utf-8")
        vulkan_ps1 = (out / "START-QWEN36-VULKAN.ps1").read_text(encoding="utf-8")
        assert "qwen36-node-packet-*.zip" in mac_bootstrap
        assert "INSTALL-MAC.command" in mac_bootstrap
        assert "START-QWEN36-NVIDIA.ps1" in nvidia_cmd
        assert "INSTALL-NVIDIA.cmd" in nvidia_ps1
        assert "START-QWEN36-VULKAN.ps1" in vulkan_cmd
        assert "INSTALL-VULKAN.cmd" in vulkan_ps1


def test_archive_uses_single_top_level_packet_directory():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        manifest = packet.write_packet(packet.ROOT, out, ["mac"], packet.DEFAULT_MODEL, 8131)
        archive = out / "packet.zip"

        packet.archive_packet(Path(manifest["payload_dir"]), archive)

        with zipfile.ZipFile(archive) as zf:
            names = set(zf.namelist())
        assert "qwen36-node-packet/qwen36_node_server.py" in names
        assert "qwen36-node-packet/RUN-MAC.sh" in names
        assert "qwen36-node-packet/START-MAC.command" in names
        assert "qwen36-node-packet/INSTALL-MAC.command" in names
        assert "qwen36-node-packet/SEND-REPORTS-MAC.sh" in names
        assert "qwen36-node-packet/RUN-NVIDIA.ps1" not in names
        assert "qwen36-node-packet/START-NVIDIA.cmd" not in names
        assert "qwen36-node-packet/INSTALL-NVIDIA.cmd" not in names
        assert "qwen36-node-packet/SEND-REPORTS-NVIDIA.ps1" not in names


def test_linux_nvidia_profile_contains_bash_launchers_and_manifest_rows():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)

        manifest = packet.write_packet(packet.ROOT, out, ["linux-nvidia"], packet.DEFAULT_MODEL, 8131)

        payload = Path(manifest["payload_dir"])
        assert (payload / "RUN-LINUX-NVIDIA.sh").exists()
        assert (payload / "START-LINUX-NVIDIA.sh").exists()
        assert (payload / "INSTALL-LINUX-NVIDIA.sh").exists()
        assert (payload / "SEND-REPORTS-LINUX-NVIDIA.sh").exists()
        assert (out / "START-QWEN36-LINUX-NVIDIA.sh").exists()
        data = json.loads((payload / "manifest.json").read_text(encoding="utf-8"))
        assert data["profiles"] == ["linux-nvidia"]
        assert data["preflight_command"]["linux-nvidia"] == "bash RUN-LINUX-NVIDIA.sh --preflight"
        assert data["start_command"]["linux-nvidia"] == "bash START-LINUX-NVIDIA.sh"
        assert data["install_command"]["linux-nvidia"] == "bash INSTALL-LINUX-NVIDIA.sh"
        assert data["report_command"]["linux-nvidia"] == "bash SEND-REPORTS-LINUX-NVIDIA.sh <driver-tailnet-name>"
        assert data["bootstrap_command"]["linux-nvidia"] == "bash START-QWEN36-LINUX-NVIDIA.sh"
        assert data["receive"]["linux-nvidia"][-2] == "bash RUN-LINUX-NVIDIA.sh --preflight"
        run_script = (payload / "RUN-LINUX-NVIDIA.sh").read_text(encoding="utf-8")
        start_script = (payload / "START-LINUX-NVIDIA.sh").read_text(encoding="utf-8")
        install_script = (payload / "INSTALL-LINUX-NVIDIA.sh").read_text(encoding="utf-8")
        report_script = (payload / "SEND-REPORTS-LINUX-NVIDIA.sh").read_text(encoding="utf-8")
        bootstrap = (out / "START-QWEN36-LINUX-NVIDIA.sh").read_text(encoding="utf-8")
        assert "--profile nvidia" in run_script
        assert "--bind tailnet" in run_script
        assert "preflight-linux-nvidia-" in start_script
        assert "llama-server is required" in install_script
        assert "qwen36-node-reports-linux-nvidia-" in report_script
        assert "INSTALL-LINUX-NVIDIA.sh" in bootstrap


def test_reused_output_directory_drops_stale_profile_files():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)

        packet.write_packet(packet.ROOT, out, ["mac", "linux-nvidia", "nvidia", "vulkan"], packet.DEFAULT_MODEL, 8131)
        manifest = packet.write_packet(packet.ROOT, out, ["mac"], packet.DEFAULT_MODEL, 8131)

        payload = Path(manifest["payload_dir"])
        assert (payload / "RUN-MAC.sh").exists()
        assert (payload / "START-MAC.command").exists()
        assert (payload / "INSTALL-MAC.command").exists()
        assert (payload / "SEND-REPORTS-MAC.sh").exists()
        assert (out / "START-QWEN36-MAC.command").exists()
        assert not (payload / "RUN-NVIDIA.ps1").exists()
        assert not (payload / "START-NVIDIA.cmd").exists()
        assert not (payload / "INSTALL-NVIDIA.cmd").exists()
        assert not (payload / "SEND-REPORTS-NVIDIA.ps1").exists()
        assert not (payload / "RUN-LINUX-NVIDIA.sh").exists()
        assert not (payload / "START-LINUX-NVIDIA.sh").exists()
        assert not (payload / "INSTALL-LINUX-NVIDIA.sh").exists()
        assert not (payload / "SEND-REPORTS-LINUX-NVIDIA.sh").exists()
        assert not (payload / "RUN-VULKAN.ps1").exists()
        assert not (payload / "START-VULKAN.cmd").exists()
        assert not (payload / "INSTALL-VULKAN.cmd").exists()
        assert not (payload / "SEND-REPORTS-VULKAN.ps1").exists()
        assert not (out / "START-QWEN36-NVIDIA.cmd").exists()
        assert not (out / "START-QWEN36-NVIDIA.ps1").exists()
        assert not (out / "START-QWEN36-LINUX-NVIDIA.sh").exists()
        assert not (out / "START-QWEN36-VULKAN.cmd").exists()
        assert not (out / "START-QWEN36-VULKAN.ps1").exists()


def test_taildrop_dry_run_does_not_require_tailscale():
    result = packet.taildrop(Path("packet.zip"), "node", dry_run=True)

    assert result["dry_run"] is True
    assert result["sent"] is False
    assert result["command"][-1] == "node:"
    assert result["attempt_count"] == 1


def test_taildrop_retries_until_success():
    original_find = packet.find_tailscale
    original_run = packet.run_taildrop_command
    original_sleep = packet.time.sleep
    calls = []
    sleeps = []
    outcomes = [
        subprocess.CompletedProcess(["tailscale"], 1, stdout="", stderr="not replying"),
        subprocess.CompletedProcess(["tailscale"], 0, stdout="sent", stderr=""),
    ]

    try:
        packet.find_tailscale = lambda: "tailscale"

        def fake_run(command, timeout_s=60.0):
            calls.append(command)
            return outcomes.pop(0)

        packet.run_taildrop_command = fake_run
        packet.time.sleep = lambda delay: sleeps.append(delay)

        result = packet.taildrop(
            Path("packet.zip"),
            "anthony",
            dry_run=False,
            retries=3,
            retry_delay_s=0.01,
            attempt_timeout_s=2.0,
        )

        assert result["sent"] is True
        assert result["attempt_count"] == 2
        assert len(calls) == 2
        assert sleeps == [0.01]
        assert result["attempts"][0]["sent"] is False
        assert result["attempts"][0]["stderr"] == "not replying"
        assert result["attempts"][1]["sent"] is True
        assert result["stderr"] == ""
    finally:
        packet.find_tailscale = original_find
        packet.run_taildrop_command = original_run
        packet.time.sleep = original_sleep


def test_taildrop_attempt_timeout_is_reported():
    original_find = packet.find_tailscale
    original_run = packet.run_taildrop_command

    try:
        packet.find_tailscale = lambda: "tailscale"

        def fake_run(command, timeout_s=60.0):
            return subprocess.CompletedProcess(command, 124, stdout="", stderr=f"timed out after {timeout_s:g}s")

        packet.run_taildrop_command = fake_run

        result = packet.taildrop(
            Path("packet.zip"),
            "anthony",
            dry_run=False,
            retries=1,
            attempt_timeout_s=3.0,
        )

        assert result["sent"] is False
        assert result["exit_code"] == 124
        assert result["attempts"][0]["stderr"] == "timed out after 3s"
    finally:
        packet.find_tailscale = original_find
        packet.run_taildrop_command = original_run


def test_taildrop_files_reports_bundle_status():
    original_find = packet.find_tailscale
    original_run = packet.run_taildrop_command
    commands = []

    try:
        packet.find_tailscale = lambda: "tailscale"

        def fake_run(command, timeout_s=60.0):
            commands.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="sent", stderr="")

        packet.run_taildrop_command = fake_run

        result = packet.taildrop_files(
            [Path("packet.zip"), Path("START-QWEN36-MAC.command")],
            "node-macos-a",
            dry_run=False,
            retries=1,
        )

        assert result["sent"] is True
        assert result["file_count"] == 2
        assert len(result["files"]) == 2
        assert commands[0][-2] == "packet.zip"
        assert commands[1][-2] == "START-QWEN36-MAC.command"
    finally:
        packet.find_tailscale = original_find
        packet.run_taildrop_command = original_run


def test_taildrop_files_skips_dependents_after_first_failure():
    original_find = packet.find_tailscale
    original_run = packet.run_taildrop_command
    commands = []

    try:
        packet.find_tailscale = lambda: "tailscale"

        def fake_run(command, timeout_s=60.0):
            commands.append(command)
            return subprocess.CompletedProcess(command, 124, stdout="", stderr="timed out")

        packet.run_taildrop_command = fake_run

        result = packet.taildrop_files(
            [
                Path("packet.zip"),
                Path("START-QWEN36-NVIDIA.cmd"),
                Path("START-QWEN36-NVIDIA.ps1"),
            ],
            "anthony",
            dry_run=False,
            retries=1,
        )

        assert result["sent"] is False
        assert len(commands) == 1
        assert result["files"][0]["exit_code"] == 124
        assert result["files"][1]["skipped"] is True
        assert result["files"][2]["skipped"] is True
        assert "packet.zip failed" in result["files"][1]["reason"]
    finally:
        packet.find_tailscale = original_find
        packet.run_taildrop_command = original_run


def test_taildrop_send_paths_include_bootstrap_by_default():
    manifest = {"bootstrap_files": ["START-QWEN36-MAC.command", "START-QWEN36-NVIDIA.cmd"]}
    paths = packet.taildrop_send_paths(
        Path("packet.zip"),
        Path("out"),
        manifest,
        include_bootstrap=True,
    )

    assert paths == [
        Path("packet.zip"),
        Path("out") / "START-QWEN36-MAC.command",
        Path("out") / "START-QWEN36-NVIDIA.cmd",
    ]


def test_taildrop_send_paths_can_omit_bootstrap():
    manifest = {"bootstrap_files": ["START-QWEN36-MAC.command"]}
    paths = packet.taildrop_send_paths(
        Path("packet.zip"),
        Path("out"),
        manifest,
        include_bootstrap=False,
    )

    assert paths == [Path("packet.zip")]


def test_receive_instructions_cover_mac_and_nvidia():
    instructions = packet.receive_instructions()

    assert instructions["mac"][-2] == "bash RUN-MAC.sh --preflight"
    assert instructions["mac"][-1] == "bash INSTALL-MAC.command"
    assert instructions["linux-nvidia"][-2] == "bash RUN-LINUX-NVIDIA.sh --preflight"
    assert instructions["linux-nvidia"][-1] == "bash INSTALL-LINUX-NVIDIA.sh"
    assert instructions["nvidia"][-2] == ".\\RUN-NVIDIA.ps1 --preflight"
    assert instructions["nvidia"][-1] == ".\\INSTALL-NVIDIA.cmd"
    assert instructions["vulkan"][-2] == ".\\RUN-VULKAN.ps1 --preflight"
    assert instructions["vulkan"][-1] == ".\\INSTALL-VULKAN.cmd"


def test_parse_profiles_supports_vulkan_and_windows_sets():
    assert packet.parse_profiles("dgx") == ["linux-nvidia"]
    assert packet.parse_profiles("linux") == ["linux-nvidia"]
    assert packet.parse_profiles("linux-nvidia") == ["linux-nvidia"]
    assert packet.parse_profiles("vulkan") == ["vulkan"]
    assert packet.parse_profiles("windows") == ["nvidia", "vulkan"]
    assert packet.parse_profiles("all") == ["mac", "linux-nvidia", "nvidia", "vulkan"]


def test_report_target_is_baked_into_start_wrappers():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        manifest = packet.write_packet(
            packet.ROOT,
            out,
            ["mac", "nvidia"],
            packet.DEFAULT_MODEL,
            8131,
            report_target=" node-desktop-b ",
        )

        payload = Path(manifest["payload_dir"])
        data = json.loads((payload / "manifest.json").read_text(encoding="utf-8"))
        mac_start = (payload / "START-MAC.command").read_text(encoding="utf-8")
        nvidia_start = (payload / "START-NVIDIA.cmd").read_text(encoding="utf-8")
        readme = (payload / "README.md").read_text(encoding="utf-8")

        assert data["report_target"] == "node-desktop-b"
        assert "report_target='node-desktop-b'" in mac_start
        assert 'bash SEND-REPORTS-MAC.sh "$report_target" || true' in mac_start
        assert 'set "REPORT_TARGET=node-desktop-b"' in nvidia_start
        assert "SEND-REPORTS-NVIDIA.ps1" in nvidia_start
        assert "driver report target `node-desktop-b`" in readme


def test_report_target_rejects_unsafe_wrapper_text():
    try:
        packet.validate_report_target('driver" & del files')
    except ValueError as exc:
        assert "report target" in str(exc)
    else:
        raise AssertionError("unsafe report target was accepted")


def test_report_target_auto_uses_detected_tailscale_self_name():
    original_detect = packet.detect_report_target
    try:
        packet.detect_report_target = lambda: "node-desktop-b"
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            manifest = packet.write_packet(
                packet.ROOT,
                out,
                ["mac"],
                packet.DEFAULT_MODEL,
                8131,
                report_target=packet.AUTO_REPORT_TARGET,
            )

            payload = Path(manifest["payload_dir"])
            data = json.loads((payload / "manifest.json").read_text(encoding="utf-8"))
            mac_start = (payload / "START-MAC.command").read_text(encoding="utf-8")

        assert data["report_target"] == "node-desktop-b"
        assert "report_target='node-desktop-b'" in mac_start
    finally:
        packet.detect_report_target = original_detect


def test_report_target_auto_fails_when_detection_is_unavailable():
    original_detect = packet.detect_report_target
    try:
        packet.detect_report_target = lambda: ""
        try:
            packet.resolve_report_target(packet.AUTO_REPORT_TARGET)
        except ValueError as exc:
            assert "auto-detect" in str(exc)
        else:
            raise AssertionError("auto report target resolved without a detected node")
    finally:
        packet.detect_report_target = original_detect


if __name__ == "__main__":
    test_packet_contains_launchers_and_manifest()
    test_archive_uses_single_top_level_packet_directory()
    test_linux_nvidia_profile_contains_bash_launchers_and_manifest_rows()
    test_reused_output_directory_drops_stale_profile_files()
    test_taildrop_dry_run_does_not_require_tailscale()
    test_taildrop_retries_until_success()
    test_taildrop_attempt_timeout_is_reported()
    test_taildrop_files_reports_bundle_status()
    test_taildrop_files_skips_dependents_after_first_failure()
    test_taildrop_send_paths_include_bootstrap_by_default()
    test_taildrop_send_paths_can_omit_bootstrap()
    test_receive_instructions_cover_mac_and_nvidia()
    test_parse_profiles_supports_vulkan_and_windows_sets()
    test_report_target_is_baked_into_start_wrappers()
    test_report_target_rejects_unsafe_wrapper_text()
    test_report_target_auto_uses_detected_tailscale_self_name()
    test_report_target_auto_fails_when_detection_is_unavailable()
    print("PASS qwen36_node_packet_test")
