#!/usr/bin/env python3
"""Tests for tools/recent_feature_dogfood.py."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "recent_feature_dogfood.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("recent_feature_dogfood", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def completed(rc: int, payload: dict | None = None) -> subprocess.CompletedProcess[str]:
    stdout = json.dumps(payload or {}) if payload is not None else "ok\n"
    return subprocess.CompletedProcess(["fake"], rc, stdout=stdout, stderr="")


def test_vcache_recall_expected_refutation_passes() -> None:
    mod = load()
    ok, reason = mod.validate_payload("vcache_recall_refuted", {
        "status": "refuted",
        "decision": "cold_prefill",
        "break_even_siblings": 301,
    })
    assert ok, reason


def test_code_slop_action_payload_is_valid_dogfood() -> None:
    mod = load()
    ok, reason = mod.validate_payload("code_slop_scorecard", {
        "schema": "fleet-code-slop-scorecard/1",
        "ok": False,
        "corpus": {"slop_debt": 313},
    })
    assert ok, reason
    assert "313" in reason


def test_build_suite_names_recent_surfaces() -> None:
    mod = load()
    with tempfile.TemporaryDirectory() as d:
        keys = {p.key for p in mod.build_suite(ROOT, Path(d), include_go_tests=True)}
    for key in (
        "loop-append",
        "loop-status",
        "vcache-score",
        "vcache-recall-refutation",
        "benchmarks-list-offline",
        "benchmarks-run-vcache",
        "code-slop-scorecard",
        "dogfood-coverage-scorecard",
        "go-test-benchcatalog",
        "go-test-callavoid",
        "go-test-promptmmu",
    ):
        assert key in keys


def test_benchmarks_payload_is_valid_dogfood() -> None:
    mod = load()
    ok, reason = mod.validate_payload("benchmarks_offline", [
        {"Name": "vcache", "Need": "offline", "Run": "fak vcache bench --json"},
    ])
    assert ok, reason


def test_fak_bin_windows_path_preserves_backslashes() -> None:
    mod = load()
    value = r"C:\work\fak\.fak\tmp\fak-dogfood.exe"
    assert mod.split_configured_command(value) == [value]


def test_run_probe_allows_expected_nonzero_when_payload_valid() -> None:
    mod = load()
    probe = mod.Probe(
        key="vcache-recall-refutation",
        description="expected refutation",
        command=["fake"],
        allowed_exits=(1,),
        json_source="stdout",
        validator="vcache_recall_refuted",
    )
    got = mod.run_probe(
        probe,
        ROOT,
        timeout=1,
        runner=lambda cmd, root, timeout: completed(1, {
            "status": "refuted",
            "decision": "cold_prefill",
            "break_even_siblings": 301,
        }),
    )
    assert got["ok"] is True
    assert got["exit_code"] == 1


def test_run_suite_fails_when_required_probe_fails() -> None:
    mod = load()

    def fake_runner(cmd, root, timeout):
        joined = " ".join(cmd)
        if "loop append" in joined:
            return completed(0, {"schema": "fak.loop-event.v1", "hash": "abc"})
        if "loop status" in joined:
            return completed(0, {"schema": "fak.loop-status.v1", "loops": [{"loop_id": "x"}]})
        if "vcache score" in joined:
            return completed(0, {"schema": "fak.vcache.score.v1", "two_x_better": True, "active_multiplier": 2.5})
        if "prove-recall" in joined:
            return completed(1, {"status": "refuted", "decision": "cold_prefill", "break_even_siblings": 301})
        if "benchmarks list" in joined:
            return subprocess.CompletedProcess(
                ["fake"], 0,
                stdout=json.dumps([{"Name": "vcache", "Need": "offline", "Run": "fak vcache bench --json"}]),
                stderr="",
            )
        if "benchmarks run vcache" in joined:
            return completed(0, {"schema": "fak.vcache.score.v1", "two_x_better": True, "active_multiplier": 2.5})
        if "code_slop_scorecard.py" in joined:
            return completed(1, {"schema": "fleet-code-slop-scorecard/1", "corpus": {"slop_debt": 1}})
        if "dogfood_coverage.py" in joined:
            return completed(0, {"schema": "dogfood-coverage/1", "dogfood_debt": 0})
        if "internal/promptmmu" in joined:
            return completed(1)
        return completed(0)

    with tempfile.TemporaryDirectory() as d:
        report = mod.run_suite(ROOT, Path(d), timeout=1, include_go_tests=True, runner=fake_runner)
    assert report["ok"] is False
    assert report["verdict"] == "ACTION"
    assert "go-test-promptmmu" in report["reason"]


def _run_all() -> int:
    tests = sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f))
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok   {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
