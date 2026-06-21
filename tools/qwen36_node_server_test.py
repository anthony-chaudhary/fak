#!/usr/bin/env python3
"""Smoke tests for qwen36_node_server.py."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen36_node_server as node


def test_profile_detection_from_platform_inputs():
    assert node.detect_profile(system="Darwin", has_nvidia=False, has_amd_gpu=False) == "mac"
    assert node.detect_profile(system="Windows", has_nvidia=True, has_amd_gpu=True) == "nvidia"
    assert node.detect_profile(system="Windows", has_nvidia=False, has_amd_gpu=True) == "vulkan"
    assert node.detect_profile(system="Linux", has_nvidia=False, has_amd_gpu=False) == "cpu"


def test_bind_selection_prefers_tailnet_in_auto_mode():
    assert node.select_bind_host("auto", tailnet_ip="100.64.0.9") == "100.64.0.9"
    assert node.select_bind_host("auto", tailnet_ip="") == "127.0.0.1"
    assert node.select_bind_host("localhost", tailnet_ip="100.64.0.9") == "127.0.0.1"
    assert node.select_bind_host("tailnet", tailnet_ip="100.64.0.9") == "100.64.0.9"


def test_public_bind_is_refused_by_default():
    try:
        node.validate_bind_host("0.0.0.0", allow_public_bind=False)
    except ValueError as exc:
        assert "refusing public bind" in str(exc)
    else:
        raise AssertionError("expected public bind refusal")
    node.validate_bind_host("100.64.0.9", allow_public_bind=False)


def test_profile_defaults_are_applied_to_command():
    cmd = node.build_command(
        llama_server="llama-server",
        model=node.DEFAULT_MODEL,
        host="100.64.0.9",
        port=8131,
        profile=node.PROFILES["nvidia"],
        ctx_size=0,
        n_gpu_layers=-1,
        threads=0,
        extra_args=["--temp", "0"],
    )
    assert cmd == [
        "llama-server",
        "-hf",
        node.DEFAULT_MODEL,
        "--host",
        "100.64.0.9",
        "--port",
        "8131",
        "--ctx-size",
        "8192",
        "--n-gpu-layers",
        "20",
        "--temp",
        "0",
    ]


def test_vulkan_profile_defaults_use_partial_offload_and_fit():
    cmd = node.build_command(
        llama_server="llama-server",
        model=node.DEFAULT_MODEL,
        host="100.64.0.9",
        port=8131,
        profile=node.PROFILES["vulkan"],
        ctx_size=0,
        n_gpu_layers=-1,
        threads=0,
        extra_args=[],
    )

    assert cmd == [
        "llama-server",
        "-hf",
        node.DEFAULT_MODEL,
        "--host",
        "100.64.0.9",
        "--port",
        "8131",
        "--ctx-size",
        "8192",
        "--n-gpu-layers",
        "20",
        "--fit",
        "on",
    ]


def test_default_cpu_threads_leaves_headroom():
    assert node.default_cpu_threads(cpu_count=8) == 6
    assert node.default_cpu_threads(cpu_count=2) == 1  # never below 1
    assert node.default_cpu_threads(cpu_count=1) == 1


def test_cpu_profile_emits_explicit_bounded_threads():
    cmd = node.build_command(
        llama_server="llama-server",
        model=node.DEFAULT_MODEL,
        host="127.0.0.1",
        port=8131,
        profile=node.PROFILES["cpu"],
        ctx_size=0,
        n_gpu_layers=-1,
        threads=0,
        extra_args=[],
    )
    assert "--threads" in cmd
    n = int(cmd[cmd.index("--threads") + 1])
    assert n >= 1


def test_explicit_threads_override_wins_and_gpu_profile_stays_unset():
    cpu_cmd = node.build_command(
        llama_server="llama-server", model="m", host="127.0.0.1", port=1,
        profile=node.PROFILES["cpu"], ctx_size=0, n_gpu_layers=-1, threads=4, extra_args=[],
    )
    assert cpu_cmd[cpu_cmd.index("--threads") + 1] == "4"
    # GPU profile with threads=0 keeps llama's own default (no --threads emitted).
    gpu_cmd = node.build_command(
        llama_server="llama-server", model="m", host="127.0.0.1", port=1,
        profile=node.PROFILES["nvidia"], ctx_size=0, n_gpu_layers=-1, threads=0, extra_args=[],
    )
    assert "--threads" not in gpu_cmd


def test_base_url_brackets_ipv6_hosts():
    assert node.base_url("127.0.0.1", 8131) == "http://127.0.0.1:8131/v1"
    assert node.base_url("fd7a:115c:a1e0::1", 8131) == "http://[fd7a:115c:a1e0::1]:8131/v1"


def test_preflight_plan_reports_missing_llama_server():
    original_tail = node.tailscale_ip
    original_find = node.find_llama_server

    def missing_llama_server(_explicit: str = "") -> str:
        raise FileNotFoundError("missing llama-server")

    try:
        node.tailscale_ip = lambda: "100.64.0.9"
        node.find_llama_server = missing_llama_server
        args = node.build_parser().parse_args(["--profile", "nvidia", "--bind", "tailnet", "--preflight"])

        plan = node.build_runtime_plan(args, require_llama=True)

        assert plan["ok"] is False
        assert "missing llama-server" in plan["failures"]
        assert plan["base_url"] == "http://100.64.0.9:8131/v1"
        llama_check = next(check for check in plan["checks"] if check["name"] == "llama_server")
        assert llama_check["found"] is False
        assert llama_check["required"] is True
    finally:
        node.tailscale_ip = original_tail
        node.find_llama_server = original_find


def test_preflight_plan_accepts_tailnet_launch_requirements():
    original_tail = node.tailscale_ip
    original_find = node.find_llama_server

    try:
        node.tailscale_ip = lambda: "100.64.0.9"
        node.find_llama_server = lambda _explicit="": "/usr/local/bin/llama-server"
        args = node.build_parser().parse_args(["--profile", "mac", "--bind", "tailnet", "--preflight"])

        plan = node.build_runtime_plan(args, require_llama=True)

        assert plan["ok"] is True
        assert plan["failures"] == []
        assert plan["profile"] == "mac"
        assert plan["base_url"] == "http://100.64.0.9:8131/v1"
        assert plan["command"][0] == "/usr/local/bin/llama-server"
        assert "--host" in plan["command"]
    finally:
        node.tailscale_ip = original_tail
        node.find_llama_server = original_find


def test_preflight_plan_requires_nvidia_smi_for_nvidia_profile():
    original_tail = node.tailscale_ip
    original_find = node.find_llama_server
    original_nvidia = node.nvidia_smi_report

    try:
        node.tailscale_ip = lambda: "100.64.0.9"
        node.find_llama_server = lambda _explicit="": r"C:\llama\llama-server.exe"
        node.nvidia_smi_report = lambda: {
            "found": False,
            "ok": False,
            "detail": "nvidia-smi was not found on PATH",
        }
        args = node.build_parser().parse_args(["--profile", "nvidia", "--bind", "tailnet", "--preflight"])

        plan = node.build_runtime_plan(args, require_llama=True)

        assert plan["ok"] is False
        assert "nvidia-smi was not found on PATH" in plan["failures"]
        check = next(check for check in plan["checks"] if check["name"] == "nvidia_smi")
        assert check["required"] is True
        assert check["found"] is False
        assert check["ok"] is False
    finally:
        node.tailscale_ip = original_tail
        node.find_llama_server = original_find
        node.nvidia_smi_report = original_nvidia


def test_preflight_plan_records_nvidia_gpu_identity():
    original_tail = node.tailscale_ip
    original_find = node.find_llama_server
    original_nvidia = node.nvidia_smi_report

    try:
        node.tailscale_ip = lambda: "100.64.0.9"
        node.find_llama_server = lambda _explicit="": r"C:\llama\llama-server.exe"
        node.nvidia_smi_report = lambda: {
            "found": True,
            "ok": True,
            "detail": "NVIDIA GeForce RTX 4070 Laptop GPU, 555.85, 8188, 7000",
            "gpus": [{
                "name": "NVIDIA GeForce RTX 4070 Laptop GPU",
                "driver_version": "555.85",
                "memory_total_mib": "8188",
                "memory_free_mib": "7000",
            }],
        }
        args = node.build_parser().parse_args(["--profile", "nvidia", "--bind", "tailnet", "--preflight"])

        plan = node.build_runtime_plan(args, require_llama=True)

        assert plan["ok"] is True
        check = next(check for check in plan["checks"] if check["name"] == "nvidia_smi")
        assert check["ok"] is True
        assert check["required"] is True
        assert check["gpus"][0]["name"] == "NVIDIA GeForce RTX 4070 Laptop GPU"
    finally:
        node.tailscale_ip = original_tail
        node.find_llama_server = original_find
        node.nvidia_smi_report = original_nvidia


if __name__ == "__main__":
    test_profile_detection_from_platform_inputs()
    test_bind_selection_prefers_tailnet_in_auto_mode()
    test_public_bind_is_refused_by_default()
    test_profile_defaults_are_applied_to_command()
    test_vulkan_profile_defaults_use_partial_offload_and_fit()
    test_default_cpu_threads_leaves_headroom()
    test_cpu_profile_emits_explicit_bounded_threads()
    test_explicit_threads_override_wins_and_gpu_profile_stays_unset()
    test_base_url_brackets_ipv6_hosts()
    test_preflight_plan_reports_missing_llama_server()
    test_preflight_plan_accepts_tailnet_launch_requirements()
    test_preflight_plan_requires_nvidia_smi_for_nvidia_profile()
    test_preflight_plan_records_nvidia_gpu_identity()
    print("PASS qwen36_node_server_test")
