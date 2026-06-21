#!/usr/bin/env python3
"""Smoke tests for qwen36_surface_smoke.py endpoint resolution."""
import argparse
import json
import os
import socket
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen36_surface_smoke as smoke


def write_registry(path, endpoints):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "endpoints": endpoints}, f)


def listening_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    return sock


def closed_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def gateway_args(**overrides):
    defaults = {
        "http_timeout_s": 0.05,
        "model_timeout_s": 10,
        "model": "test-model",
        "gateway_chat": True,
        "perf_decode_baseline_tps": 8.0,
        "min_decode_tps": 0.0,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_disabled_endpoint_does_not_probe():
    with tempfile.TemporaryDirectory() as td:
        reg = os.path.join(td, "endpoints.json")
        write_registry(reg, [{
            "name": "mac",
            "tailscale_ip": "127.0.0.1",
            "enabled": False,
            "serve_port": closed_port(),
            "ssh": {"available": True, "port": closed_port(), "method": "test"},
        }])

        endpoint = smoke.resolve_endpoint("mac", reg, dry_run=False, timeout_s=0.05)

        assert endpoint["state"] == "DISABLED"
        assert endpoint["base_url"].startswith("http://127.0.0.1:")
        assert endpoint["ssh"]["available"] is True


def test_ready_endpoint_uses_open_serve_port():
    with tempfile.TemporaryDirectory() as td, listening_socket() as server:
        port = server.getsockname()[1]
        reg = os.path.join(td, "endpoints.json")
        write_registry(reg, [{
            "name": "gpu",
            "tailscale_ip": "127.0.0.1",
            "enabled": True,
            "serve_port": port,
            "ssh": {"available": False, "port": 22, "method": "test"},
        }])

        endpoint = smoke.resolve_endpoint("gpu", reg, dry_run=False, timeout_s=0.05)

        assert endpoint["state"] == "READY"
        assert endpoint["base_url"] == f"http://127.0.0.1:{port}/v1"
        assert endpoint["serve_tcp"][0]["open"] is True


def test_online_no_serve_when_only_ssh_port_is_open():
    with tempfile.TemporaryDirectory() as td, listening_socket() as sshd:
        reg = os.path.join(td, "endpoints.json")
        write_registry(reg, [{
            "name": "gpu",
            "tailscale_ip": "127.0.0.1",
            "enabled": True,
            "serve_port": closed_port(),
            "ssh": {"available": True, "port": sshd.getsockname()[1], "method": "test"},
        }])

        endpoint = smoke.resolve_endpoint("gpu", reg, dry_run=False, timeout_s=0.05)

        assert endpoint["state"] == "ONLINE_NO_SERVE"
        assert endpoint["ssh"]["tcp_open"] is True
        assert endpoint["serve_tcp"][0]["open"] is False


def test_online_no_serve_when_tailscale_ping_succeeds():
    original = smoke.tailscale_ping
    try:
        smoke.tailscale_ping = lambda host, timeout_s: {
            "host": host,
            "state": "ONLINE",
            "rtt_ms": 12,
            "detail": "pong from node in 12ms",
        }
        with tempfile.TemporaryDirectory() as td:
            reg = os.path.join(td, "endpoints.json")
            write_registry(reg, [{
                "name": "gpu",
                "tailnet_host": "node",
                "tailscale_ip": "127.0.0.1",
                "enabled": True,
                "serve_port": closed_port(),
                "ssh": {"available": False, "port": 22, "method": "test"},
            }])

            endpoint = smoke.resolve_endpoint("gpu", reg, dry_run=False, timeout_s=0.05)

            assert endpoint["state"] == "ONLINE_NO_SERVE"
            assert "tailscale ping" in endpoint["detail"]
            assert endpoint["tailscale_ping"]["state"] == "ONLINE"
    finally:
        smoke.tailscale_ping = original


def test_tailscale_ping_tries_short_magicdns_name():
    calls = []
    original = smoke.tailscale_ping
    try:
        def fake_ping(host, timeout_s):
            calls.append(host)
            return {
                "host": host,
                "state": "ONLINE" if host == "node" else "OFFLINE",
                "detail": "test",
            }

        smoke.tailscale_ping = fake_ping
        with tempfile.TemporaryDirectory() as td:
            reg = os.path.join(td, "endpoints.json")
            write_registry(reg, [{
                "name": "gpu",
                "tailnet_host": "node.tailnet.example",
                "magicdns": "node.tailnet.example",
                "tailscale_ip": "127.0.0.1",
                "enabled": True,
                "serve_port": closed_port(),
                "ssh": {"available": False, "port": 22, "method": "test"},
            }])

            endpoint = smoke.resolve_endpoint("gpu", reg, dry_run=False, timeout_s=0.05)

            assert endpoint["state"] == "ONLINE_NO_SERVE"
            assert endpoint["tailscale_ping"]["host"] == "node"
            assert calls[:2] == ["node.tailnet.example", "node"]
    finally:
        smoke.tailscale_ping = original


def test_tailscale_node_cli_writes_online_no_serve_report():
    original_status = smoke.tailscale_status
    original_ping = smoke.tailscale_ping
    original_tcp_probe = smoke.tcp_probe
    tcp_calls = []
    try:
        smoke.tailscale_status = lambda timeout_s: {
            "Peer": {
                "nodekey:test": {
                    "HostName": "gpu-node",
                    "DNSName": "gpu-node.tailnet.example.",
                    "OS": "windows",
                    "TailscaleIPs": ["127.0.0.1"],
                    "Online": True,
                }
            }
        }
        smoke.tailscale_ping = lambda host, timeout_s: {
            "host": host,
            "state": "ONLINE" if host == "gpu-node" else "OFFLINE",
            "detail": "test",
        }
        def fake_tcp_probe(hosts, port, timeout_s):
            tcp_calls.append((list(hosts), port))
            return [
                {"host": host, "port": port, "open": False} for host in hosts
            ]

        smoke.tcp_probe = fake_tcp_probe
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "report.json")

            rc = smoke.main([
                "--tailscale-node", "gpu-node",
                "--serve-port", "8131",
                "--http-timeout-s", "0.05",
                "--out-dir", td,
                "--out", out,
            ])

            assert rc == 3
            with open(out, encoding="utf-8") as f:
                report = json.load(f)
            assert report["endpoint"]["state"] == "ONLINE_NO_SERVE"
            assert report["endpoint"]["source"] == "tailscale-status"
            assert report["endpoint"]["base_url"] == "http://127.0.0.1:8131/v1"
            assert report["endpoint"]["hosts"] == ["127.0.0.1", "gpu-node.tailnet.example", "gpu-node"]
            assert report["endpoint"]["probe_hosts"] == ["127.0.0.1"]
            assert report["endpoint"]["tailscale_ping"]["host"] == "gpu-node"
            assert tcp_calls == [(["127.0.0.1"], 8131), (["127.0.0.1"], 22)]
    finally:
        smoke.tailscale_status = original_status
        smoke.tailscale_ping = original_ping
        smoke.tcp_probe = original_tcp_probe


def test_tailscale_node_reports_open_ssh_without_claiming_auth():
    original_status = smoke.tailscale_status
    original_ping = smoke.tailscale_ping
    original_tcp_probe = smoke.tcp_probe
    try:
        smoke.tailscale_status = lambda timeout_s: {
            "Peer": {
                "nodekey:test": {
                    "HostName": "gpu-node",
                    "DNSName": "gpu-node.tailnet.example.",
                    "OS": "windows",
                    "TailscaleIPs": ["127.0.0.1"],
                    "Online": True,
                }
            }
        }
        smoke.tailscale_ping = lambda host, timeout_s: {
            "host": host,
            "state": "OFFLINE",
            "detail": "should be skipped when SSH TCP is open",
        }

        def fake_tcp_probe(hosts, port, timeout_s):
            return [
                {"host": host, "port": port, "open": port == 22}
                for host in hosts
            ]

        smoke.tcp_probe = fake_tcp_probe

        endpoint = smoke.resolve_tailscale_node("gpu-node", serve_port=8131, dry_run=False, timeout_s=0.05)

        assert endpoint["state"] == "ONLINE_NO_SERVE"
        assert endpoint["hosts"] == ["127.0.0.1", "gpu-node.tailnet.example", "gpu-node"]
        assert endpoint["probe_hosts"] == ["127.0.0.1"]
        assert "SSH TCP open" in endpoint["detail"]
        assert endpoint["ssh"]["available"] is False
        assert endpoint["ssh"]["probe"] is True
        assert endpoint["ssh"]["auth_verified"] is False
        assert endpoint["ssh"]["tcp_open"] is True
        assert endpoint["tailscale_ping"]["state"] == "SKIPPED"
    finally:
        smoke.tailscale_status = original_status
        smoke.tailscale_ping = original_ping
        smoke.tcp_probe = original_tcp_probe


def test_tailscale_node_missing_is_classified():
    original_status = smoke.tailscale_status
    try:
        smoke.tailscale_status = lambda timeout_s: {"Peer": {}}

        endpoint = smoke.resolve_tailscale_node("missing", serve_port=8131, dry_run=False, timeout_s=0.05)

        assert endpoint["state"] == "MISSING"
        assert endpoint["source"] == "tailscale-status"
    finally:
        smoke.tailscale_status = original_status


def test_missing_endpoint_is_classified():
    with tempfile.TemporaryDirectory() as td:
        reg = os.path.join(td, "endpoints.json")
        write_registry(reg, [])

        endpoint = smoke.resolve_endpoint("missing", reg, dry_run=False, timeout_s=0.05)

        assert endpoint["state"] == "MISSING"
        assert endpoint["registry"] == reg


def test_cli_writes_report_for_missing_endpoint():
    with tempfile.TemporaryDirectory() as td:
        reg = os.path.join(td, "endpoints.json")
        out = os.path.join(td, "report.json")
        write_registry(reg, [])

        rc = smoke.main([
            "--endpoint", "missing",
            "--registry", reg,
            "--out-dir", td,
            "--out", out,
        ])

        assert rc == 3
        with open(out, encoding="utf-8") as f:
            report = json.load(f)
        assert report["endpoint"]["state"] == "MISSING"
        assert report["surfaces"][0]["endpoint_state"] == "MISSING"


def test_gateway_chat_records_response_and_perf_metrics():
    original_get = smoke.json_get
    original_post_timed = smoke.json_post_timed
    try:
        smoke.json_get = lambda url, timeout_s: (
            200,
            {"data": [{"id": "test-model"}]},
            '{"data":[{"id":"test-model"}]}',
        )
        smoke.json_post_timed = lambda url, payload, timeout_s: (
            200,
            {
                "choices": [{"message": {"content": "OK"}}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
                "timings": {
                    "prompt_n": 9,
                    "prompt_ms": 300.0,
                    "prompt_per_second": 30.0,
                    "predicted_n": 4,
                    "predicted_ms": 500.0,
                    "predicted_per_second": 8.0,
                },
            },
            '{"choices":[{"message":{"content":"OK"}}]}',
            2.0,
        )

        detail = smoke.surface_gateway_openai(gateway_args(), "http://gateway")

        assert detail["status"] == "PASS"
        assert detail["chat_response_excerpt"] == "OK"
        assert detail["chat_perf"]["completion_tokens"] == 4
        assert detail["chat_perf"]["decode_tps"] == 8.0
        assert detail["chat_perf"]["decode_vs_baseline"] == 1.0
        assert detail["chat_perf"]["llama_timings"]["predicted_n"] == 4
    finally:
        smoke.json_get = original_get
        smoke.json_post_timed = original_post_timed


def test_gateway_chat_min_decode_tps_can_fail():
    original_get = smoke.json_get
    original_post_timed = smoke.json_post_timed
    try:
        smoke.json_get = lambda url, timeout_s: (
            200,
            {"data": [{"id": "test-model"}]},
            '{"data":[{"id":"test-model"}]}',
        )
        smoke.json_post_timed = lambda url, payload, timeout_s: (
            200,
            {
                "choices": [{"message": {"content": "OK"}}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
                "timings": {"predicted_n": 4, "predicted_per_second": 5.5},
            },
            '{"choices":[{"message":{"content":"OK"}}]}',
            2.0,
        )

        detail = smoke.surface_gateway_openai(gateway_args(min_decode_tps=6.0), "http://gateway")

        assert detail["status"] == "FAIL"
        assert detail["perf_gate"]["status"] == "FAIL"
        assert detail["perf_gate"]["decode_tps"] == 5.5
    finally:
        smoke.json_get = original_get
        smoke.json_post_timed = original_post_timed


if __name__ == "__main__":
    test_disabled_endpoint_does_not_probe()
    test_ready_endpoint_uses_open_serve_port()
    test_online_no_serve_when_only_ssh_port_is_open()
    test_online_no_serve_when_tailscale_ping_succeeds()
    test_tailscale_ping_tries_short_magicdns_name()
    test_tailscale_node_cli_writes_online_no_serve_report()
    test_tailscale_node_reports_open_ssh_without_claiming_auth()
    test_tailscale_node_missing_is_classified()
    test_missing_endpoint_is_classified()
    test_cli_writes_report_for_missing_endpoint()
    test_gateway_chat_records_response_and_perf_metrics()
    test_gateway_chat_min_decode_tps_can_fail()
    print("PASS qwen36_surface_smoke_test")
