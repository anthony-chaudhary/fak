#!/usr/bin/env python3
"""Smoke tests for qwen36_node_reports.py."""
from __future__ import annotations

import json
import os
import tempfile
import zipfile
from pathlib import Path

import qwen36_node_reports as reports


def write_zip(path: Path, files: dict[str, str | bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, body in files.items():
            zf.writestr(name, body)


def test_import_report_bundle_summarizes_latest_preflight_and_log():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        archive = root / "qwen36-node-reports-mac-20260619-010000.zip"
        out_dir = root / "out"
        write_zip(
            archive,
            {
                "qwen36-reports/preflight-mac-20260619-010000.json": json.dumps({
                    "ok": False,
                    "profile": "mac",
                    "base_url": "http://100.64.0.10:8131/v1",
                    "llama_server_found": False,
                    "failures": ["llama-server was not found"],
                    "checks": [
                        {"name": "llama_server", "ok": False},
                        {
                            "name": "nvidia_smi",
                            "ok": True,
                            "required": False,
                            "gpus": [{
                                "name": "NVIDIA GeForce RTX 4070 Laptop GPU",
                                "driver_version": "555.85",
                            }],
                        },
                    ],
                }),
                "qwen36-reports/server-mac-20260619-010000.log": "line1\nline2\n",
            },
        )

        dest = reports.extract_archive(archive, out_dir, replace=False)
        summary = reports.summarize_dir(dest, log_tail_lines=1)

        assert summary["status"] == "PREFLIGHT_FAILED"
        assert summary["preflight_count"] == 1
        assert summary["server_log_count"] == 1
        assert summary["latest_preflight"]["profile"] == "mac"
        assert summary["latest_preflight"]["failed_checks"] == ["llama_server"]
        assert summary["latest_preflight"]["nvidia_smi"]["gpus"][0]["name"] == "NVIDIA GeForce RTX 4070 Laptop GPU"
        assert summary["latest_server_log_tail"] == "line2"


def test_import_report_bundle_accepts_windows_utf16_preflight():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        archive = root / "qwen36-node-reports-nvidia-20260619-141154.zip"
        out_dir = root / "out"
        preflight = json.dumps({
            "ok": True,
            "profile": "nvidia",
            "base_url": "http://100.64.0.10:8131/v1",
            "llama_server_found": True,
            "failures": [],
            "checks": [{"name": "llama_server", "ok": True}],
        }).encode("utf-16")
        write_zip(
            archive,
            {
                "qwen36-reports/preflight-nvidia-remote-20260619-141154.json": preflight,
            },
        )

        args = reports.build_parser().parse_args([
            "--archive", str(archive),
            "--out-dir", str(out_dir),
            "--skip-taildrop",
            "--replace",
        ])
        summary = reports.import_report_bundle(args)

        assert summary["imported"] is True
        assert summary["status"] == "PREFLIGHT_OK"
        assert summary["latest_preflight"]["profile"] == "nvidia"
        assert summary["latest_preflight"]["llama_server_found"] is True


def test_archive_candidates_prefers_newest_bundle():
    with tempfile.TemporaryDirectory() as td:
        inbox = Path(td)
        old = inbox / "qwen36-node-reports-mac-old.zip"
        new = inbox / "qwen36-node-reports-mac-new.zip"
        write_zip(old, {"qwen36-reports/preflight.json": "{}"})
        write_zip(new, {"qwen36-reports/preflight.json": "{}"})
        os.utime(old, (1000, 1000))
        os.utime(new, (2000, 2000))

        candidates = reports.archive_candidates(inbox)

        assert candidates[0] == new
        assert old in candidates


def test_extract_rejects_unsafe_zip_member_paths():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        archive = root / "qwen36-node-reports-mac.zip"
        write_zip(archive, {"../escape.txt": "bad"})

        try:
            reports.extract_archive(archive, root / "out", replace=False)
        except ValueError as exc:
            assert "unsafe zip member path" in str(exc)
        else:
            raise AssertionError("expected unsafe zip path rejection")


def test_extract_rejects_drive_style_zip_member_paths():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        archive = root / "qwen36-node-reports-nvidia.zip"
        write_zip(archive, {"C:/Users/Public/escape.txt": "bad"})

        try:
            reports.extract_archive(archive, root / "out", replace=False)
        except ValueError as exc:
            assert "unsafe zip member path" in str(exc)
        else:
            raise AssertionError("expected drive-style zip path rejection")


def test_no_bundle_summary_is_import_failure_without_taildrop():
    with tempfile.TemporaryDirectory() as td:
        args = reports.build_parser().parse_args(["--inbox", td, "--skip-taildrop"])

        summary = reports.import_report_bundle(args)

        assert summary["imported"] is False
        assert "no qwen36-node-reports" in summary["error"]


if __name__ == "__main__":
    test_import_report_bundle_summarizes_latest_preflight_and_log()
    test_import_report_bundle_accepts_windows_utf16_preflight()
    test_archive_candidates_prefers_newest_bundle()
    test_extract_rejects_unsafe_zip_member_paths()
    test_extract_rejects_drive_style_zip_member_paths()
    test_no_bundle_summary_is_import_failure_without_taildrop()
    print("PASS qwen36_node_reports_test")
