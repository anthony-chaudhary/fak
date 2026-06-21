#!/usr/bin/env python3
"""Smoke tests for qwen36_watch_nodes.py."""
from __future__ import annotations

import tempfile
from pathlib import Path

import qwen36_watch_nodes as watch


def opts(tmp: Path, nodes: list[str], mode: str = "all") -> watch.WatchOptions:
    return watch.WatchOptions(
        nodes=nodes,
        model=watch.DEFAULT_MODEL,
        serve_port=8131,
        out_dir=tmp,
        http_timeout_s=0.05,
        agent_max_turns=1,
        agent_timeout_s=10.0,
        gateway_chat=True,
        poll_interval_s=0.1,
        max_wait_s=0.0,
        mode=mode,
    )


def test_ready_node_runs_surface_smoke():
    calls = []

    def resolver(node, serve_port, dry_run, timeout_s):
        return {"state": "READY", "detail": "ready", "base_url": "http://127.0.0.1:8131/v1"}

    def runner(argv):
        calls.append(argv)
        return 0

    with tempfile.TemporaryDirectory() as td:
        rows = watch.poll_once(opts(Path(td), ["gpu-node"]), resolver=resolver, runner=runner)

    assert rows[0]["status"] == "PASS"
    assert rows[0]["smoke_ran"] is True
    assert "--tailscale-node" in calls[0]
    assert "--gateway-chat" in calls[0]


def test_perf_flags_are_passed_to_surface_smoke():
    with tempfile.TemporaryDirectory() as td:
        base = opts(Path(td), ["gpu-node"])
        opts_obj = watch.WatchOptions(
            **{
                **base.__dict__,
                "perf_decode_baseline_tps": 7.29,
                "min_decode_tps": 7.0,
            }
        )
        argv = watch.smoke_argv("gpu-node", opts_obj)

    assert "--perf-decode-baseline-tps" in argv
    assert argv[argv.index("--perf-decode-baseline-tps") + 1] == "7.29"
    assert "--min-decode-tps" in argv
    assert argv[argv.index("--min-decode-tps") + 1] == "7.0"


def test_waiting_node_does_not_run_smoke():
    calls = []

    def resolver(node, serve_port, dry_run, timeout_s):
        return {"state": "ONLINE_NO_SERVE", "detail": "not live"}

    with tempfile.TemporaryDirectory() as td:
        rc, report = watch.watch(
            opts(Path(td), ["mac-node"]),
            resolver=resolver,
            runner=lambda argv: calls.append(argv) or 0,
        )

    assert rc == 1
    assert report["summary"]["waiting"] == 1
    assert report["summary"]["timed_out"] is False
    assert calls == []


def test_any_mode_completes_when_one_node_passes():
    def resolver(node, serve_port, dry_run, timeout_s):
        if node == "ready":
            return {"state": "READY", "detail": "ready", "base_url": "http://127.0.0.1:8131/v1"}
        return {"state": "ONLINE_NO_SERVE", "detail": "not live"}

    with tempfile.TemporaryDirectory() as td:
        rc, report = watch.watch(
            opts(Path(td), ["ready", "waiting"], mode="any"),
            resolver=resolver,
            runner=lambda argv: 0,
        )

    assert rc == 0
    assert report["summary"]["passed"] == 1
    assert report["summary"]["waiting"] == 1


def test_failed_smoke_is_not_retried_on_later_polls():
    calls = []
    ticks = iter([0.0, 0.0, 0.5, 1.1])

    def resolver(node, serve_port, dry_run, timeout_s):
        if node == "ready":
            return {"state": "READY", "detail": "ready", "base_url": "http://127.0.0.1:8131/v1"}
        return {"state": "ONLINE_NO_SERVE", "detail": "not live"}

    def runner(argv):
        calls.append(argv)
        return 1

    with tempfile.TemporaryDirectory() as td:
        opts_obj = opts(Path(td), ["ready", "waiting"])
        opts_obj = watch.WatchOptions(
            **{**opts_obj.__dict__, "max_wait_s": 1.0, "poll_interval_s": 0.1}
        )
        rc, report = watch.watch(
            opts_obj,
            resolver=resolver,
            runner=runner,
            sleeper=lambda seconds: None,
            monotonic=lambda: next(ticks),
        )

    assert rc == 1
    assert report["summary"]["failed"] == 1
    assert report["summary"]["waiting"] == 1
    assert len(calls) == 1


def test_import_reports_attaches_latest_node_report():
    imports = []

    def resolver(node, serve_port, dry_run, timeout_s):
        return {"state": "ONLINE_NO_SERVE", "detail": "not live"}

    def report_importer(opts_obj):
        imports.append(opts_obj.report_inbox)
        return {
            "schema": "fak.qwen36-node-reports.v1",
            "imported": True,
            "status": "PREFLIGHT_FAILED",
            "latest_preflight": {"profile": "mac", "failed_checks": ["llama_server"]},
        }

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        opts_obj = opts(base / "smoke", ["mac-node"])
        opts_obj = watch.WatchOptions(
            **{
                **opts_obj.__dict__,
                "import_reports": True,
                "report_inbox": base / "inbox",
                "report_out_dir": base / "reports",
            }
        )
        rc, report = watch.watch(
            opts_obj,
            resolver=resolver,
            runner=lambda argv: 0,
            report_importer=report_importer,
        )

    assert rc == 1
    assert imports == [base / "inbox"]
    assert report["summary"]["node_report_imported"] is True
    assert report["node_report"]["status"] == "PREFLIGHT_FAILED"
    assert report["node_report"]["latest_preflight"]["failed_checks"] == ["llama_server"]


def test_report_archive_skips_taildrop_receive():
    imports = []

    def resolver(node, serve_port, dry_run, timeout_s):
        return {"state": "ONLINE_NO_SERVE", "detail": "not live"}

    def report_importer(opts_obj):
        imports.append({
            "archive": opts_obj.report_archive,
            "skip": opts_obj.skip_report_taildrop,
        })
        return {
            "schema": "fak.qwen36-node-reports.v1",
            "imported": True,
            "status": "PREFLIGHT_OK",
            "latest_preflight": {"profile": "nvidia", "ok": True},
        }

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        opts_obj = opts(base / "smoke", ["gpu-node"])
        archive = base / "qwen36-node-reports-nvidia.zip"
        opts_obj = watch.WatchOptions(
            **{
                **opts_obj.__dict__,
                "import_reports": True,
                "report_archive": archive,
                "skip_report_taildrop": True,
            }
        )
        rc, report = watch.watch(
            opts_obj,
            resolver=resolver,
            runner=lambda argv: 0,
            report_importer=report_importer,
        )

    assert rc == 1
    assert imports == [{"archive": archive, "skip": True}]
    assert report["summary"]["node_report_imported"] is True
    assert report["node_report"]["status"] == "PREFLIGHT_OK"


def test_packet_profile_uses_endpoint_os():
    assert watch.packet_profile({"os": "macOS"}) == "mac"
    assert watch.packet_profile({"os": "windows"}) == "nvidia"
    assert watch.packet_profile({"os": "windows", "accelerator": "AMD Radeon RX 7600"}) == "vulkan"
    assert watch.packet_profile({"os": "windows", "roles": ["http-model-endpoint", "vulkan"]}) == "vulkan"
    assert watch.packet_profile({"os": "windows", "packet_profile": "vulkan"}) == "vulkan"
    assert watch.packet_profile({"os": "linux"}) == "linux-nvidia"
    assert watch.packet_profile({"os": "linux", "accelerator": "NVIDIA A100"}) == "linux-nvidia"
    assert watch.packet_profile({"os": "linux", "packet_profile": "dgx"}) == "dgx"


def test_send_packet_attaches_dispatch_summary():
    dispatches = []

    def resolver(node, serve_port, dry_run, timeout_s):
        return {"state": "ONLINE_NO_SERVE", "detail": "not live"}

    def packet_dispatcher(opts_obj):
        dispatches.append(opts_obj.packet_report_target)
        return {
            "sent": True,
            "report_target": "node-desktop-b",
            "nodes": [{"node": "mac-node", "profile": "mac"}],
        }

    with tempfile.TemporaryDirectory() as td:
        opts_obj = opts(Path(td), ["mac-node"])
        opts_obj = watch.WatchOptions(
            **{
                **opts_obj.__dict__,
                "send_packet": True,
                "packet_report_target": "auto",
            }
        )
        rc, report = watch.watch(
            opts_obj,
            resolver=resolver,
            runner=lambda argv: 0,
            packet_dispatcher=packet_dispatcher,
        )

    assert rc == 1
    assert dispatches == ["auto"]
    assert report["summary"]["packet_sent"] is True
    assert report["packet_dispatch"]["nodes"][0]["profile"] == "mac"


def test_send_packet_honors_packet_profile_override():
    profiles = []

    def resolver(node, serve_port, dry_run, timeout_s):
        return {"state": "ONLINE_NO_SERVE", "detail": "not live", "os": "windows"}

    def packet_dispatcher(opts_obj):
        profiles.append(opts_obj.packet_profile)
        return {
            "sent": True,
            "report_target": "node-desktop-b",
            "nodes": [{"node": "amd-node", "profile": opts_obj.packet_profile}],
        }

    with tempfile.TemporaryDirectory() as td:
        opts_obj = opts(Path(td), ["amd-node"])
        opts_obj = watch.WatchOptions(
            **{
                **opts_obj.__dict__,
                "send_packet": True,
                "packet_profile": "dgx",
            }
        )
        rc, report = watch.watch(
            opts_obj,
            resolver=resolver,
            runner=lambda argv: 0,
            packet_dispatcher=packet_dispatcher,
        )

    assert rc == 1
    assert profiles == ["dgx"]
    assert report["packet_dispatch"]["nodes"][0]["profile"] == "dgx"


def test_waiting_packeted_node_reports_next_actions():
    def resolver(node, serve_port, dry_run, timeout_s):
        return {
            "state": "ONLINE_NO_SERVE",
            "detail": "serve port closed",
            "os": "windows",
            "ssh": {
                "tcp_open": True,
                "auth_verified": False,
                "port": 22,
                "tcp": [{"host": "100.64.0.10", "port": 22, "open": True}],
            },
        }

    def packet_dispatcher(opts_obj):
        return {
            "sent": True,
            "report_target": "node-desktop-b",
            "nodes": [{
                "node": "gpu-node",
                "profile": "nvidia",
                "packet_dir": r"C:\packets\gpu-node",
                "archive": r"C:\packets\gpu-node\qwen36-node-packet.zip",
                "bootstrap_files": ["START-QWEN36-NVIDIA.cmd", "START-QWEN36-NVIDIA.ps1"],
                "taildrop": {"sent": True},
            }],
        }

    with tempfile.TemporaryDirectory() as td:
        opts_obj = opts(Path(td), ["gpu-node"])
        opts_obj = watch.WatchOptions(
            **{
                **opts_obj.__dict__,
                "send_packet": True,
                "packet_profile": "nvidia",
            }
        )
        rc, report = watch.watch(
            opts_obj,
            resolver=resolver,
            runner=lambda argv: 0,
            packet_dispatcher=packet_dispatcher,
        )

    actions = report["nodes"][0]["next_actions"]
    assert rc == 1
    assert report["summary"]["action_required"] is True
    assert actions[0]["kind"] == "run_node_launcher"
    assert actions[0]["required"] is True
    assert actions[0]["bootstrap_files"] == ["START-QWEN36-NVIDIA.cmd", "START-QWEN36-NVIDIA.ps1"]
    assert actions[0]["report_target"] == "node-desktop-b"
    assert actions[1]["kind"] == "ssh_auth"
    assert actions[1]["required"] is False
    assert actions[1]["host"] == "100.64.0.10"


def test_registry_endpoint_facts_are_preferred_over_tailscale_resolver():
    original = watch.smoke.resolve_endpoint
    fallback_calls = []

    try:
        def fake_registry_endpoint(node, registry, dry_run, timeout_s):
            return {
                "name": node,
                "source": f"registry:{registry}",
                "state": "ONLINE_NO_SERVE",
                "detail": "registered endpoint, serve port closed",
                "os": "windows",
                "packet_profile": "nvidia",
                "ssh": {
                    "tcp_open": True,
                    "auth_verified": True,
                    "port": 22,
                    "tcp": [{"host": "100.64.0.10", "port": 22, "open": True}],
                },
            }

        def fallback(node, serve_port, dry_run, timeout_s):
            fallback_calls.append(node)
            return {"state": "MISSING", "detail": "fallback should not be used"}

        watch.smoke.resolve_endpoint = fake_registry_endpoint
        with tempfile.TemporaryDirectory() as td:
            opts_obj = watch.WatchOptions(
                **{**opts(Path(td), ["gpu-node"]).__dict__, "registry": "local-registry.json"}
            )
            rows = watch.poll_once(opts_obj, resolver=fallback, runner=lambda argv: 0)

        assert fallback_calls == []
        assert rows[0]["state"] == "ONLINE_NO_SERVE"
        assert rows[0]["endpoint"]["source"] == "registry:local-registry.json"
        assert rows[0]["endpoint"]["ssh"]["auth_verified"] is True
    finally:
        watch.smoke.resolve_endpoint = original


def test_verified_ssh_is_reported_as_remote_start_available():
    row = {
        "node": "gpu-node",
        "status": "WAIT",
        "state": "ONLINE_NO_SERVE",
        "endpoint": {
            "ssh": {
                "tcp_open": True,
                "auth_verified": True,
                "port": 22,
                "tcp": [{"host": "100.64.0.10", "port": 22, "open": True}],
            },
        },
    }
    actions = watch.next_actions_for_row(row, opts(Path("."), ["gpu-node"]), None)

    assert actions[0]["kind"] == "send_packet"
    assert actions[1]["kind"] == "ssh_remote_start_available"
    assert actions[1]["required"] is False
    assert actions[1]["host"] == "100.64.0.10"


if __name__ == "__main__":
    test_ready_node_runs_surface_smoke()
    test_perf_flags_are_passed_to_surface_smoke()
    test_waiting_node_does_not_run_smoke()
    test_any_mode_completes_when_one_node_passes()
    test_failed_smoke_is_not_retried_on_later_polls()
    test_import_reports_attaches_latest_node_report()
    test_report_archive_skips_taildrop_receive()
    test_packet_profile_uses_endpoint_os()
    test_send_packet_attaches_dispatch_summary()
    test_send_packet_honors_packet_profile_override()
    test_waiting_packeted_node_reports_next_actions()
    test_registry_endpoint_facts_are_preferred_over_tailscale_resolver()
    test_verified_ssh_is_reported_as_remote_start_available()
    print("PASS qwen36_watch_nodes_test")
