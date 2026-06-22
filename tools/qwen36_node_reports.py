#!/usr/bin/env python3
"""Receive and summarize Qwen3.6 node packet report bundles."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "fak.qwen36-node-reports.v1"
# The operator superrepo nests the module under <root>/fak/; the standalone
# public checkout has experiments/ at the module root. Prefer whichever exists
# so the default out-dir lands in the real tree (not a phantom fak/experiments).
_BASE = ROOT / "fak" if (ROOT / "fak" / "experiments").is_dir() else ROOT
DEFAULT_INBOX = ROOT / "tools" / "_registry" / "qwen36-report-inbox"
DEFAULT_OUT_DIR = _BASE / "experiments" / "qwen36" / "node-reports"


def find_tailscale() -> str:
    exe = shutil.which("tailscale")
    if exe:
        return exe
    if os.name == "nt":
        candidate = r"C:\Program Files\Tailscale\tailscale.exe"
        if os.path.exists(candidate):
            return candidate
    return ""


def receive_taildrop(inbox: Path, wait: bool) -> dict[str, Any]:
    inbox.mkdir(parents=True, exist_ok=True)
    exe = find_tailscale()
    if not exe:
        return {"ran": False, "reason": "tailscale CLI not found", "inbox": str(inbox)}
    cmd = [exe, "file", "get", "--conflict=rename"]
    if wait:
        cmd.append("--wait")
    cmd.append(str(inbox))
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return {
        "ran": True,
        "command": cmd,
        "exit_code": proc.returncode,
        "stdout": proc.stdout[-2000:],
        "stderr": proc.stderr[-2000:],
        "inbox": str(inbox),
    }


def archive_candidates(inbox: Path) -> list[Path]:
    patterns = ("qwen36-node-reports-*.zip", "*qwen36-node-reports-*.zip")
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(inbox.glob(pattern))
    return sorted(set(paths), key=lambda p: p.stat().st_mtime, reverse=True)


def safe_member_name(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or not path.parts or any(":" in part for part in path.parts):
        raise ValueError(f"unsafe zip member path: {name}")
    return path


def validate_zip(archive: Path) -> list[zipfile.ZipInfo]:
    members: list[zipfile.ZipInfo] = []
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            safe_member_name(info.filename)
            if info.is_dir():
                continue
            members.append(info)
    if not members:
        raise ValueError(f"archive contains no files: {archive}")
    return members


def read_text_auto(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-16")


def extract_archive(archive: Path, out_dir: Path, replace: bool) -> Path:
    archive = archive.resolve()
    dest = out_dir / archive.stem
    if dest.exists() and not replace:
        raise FileExistsError(f"{dest} already exists; rerun with --replace")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        members = validate_zip(archive)
        for info in members:
            name = safe_member_name(info.filename)
            target = (dest / Path(*name.parts)).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise ValueError(f"zip member escapes destination: {info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    return dest


def tail_text(path: Path, max_lines: int) -> str:
    try:
        lines = read_text_auto(path).splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        return f"<failed to read {path}: {exc}>"
    return "\n".join(lines[-max_lines:])


def load_preflight(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(read_text_auto(path))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"path": str(path), "parsed": False, "error": str(exc)}
    if not isinstance(data, dict):
        return {"path": str(path), "parsed": False, "error": "preflight JSON was not an object"}
    checks = data.get("checks") if isinstance(data.get("checks"), list) else []
    failed_checks = [
        check.get("name")
        for check in checks
        if isinstance(check, dict) and check.get("ok") is False
    ]
    nvidia_smi = next(
        (check for check in checks if isinstance(check, dict) and check.get("name") == "nvidia_smi"),
        None,
    )
    return {
        "path": str(path),
        "parsed": True,
        "ok": data.get("ok"),
        "profile": data.get("profile"),
        "base_url": data.get("base_url"),
        "llama_server_found": data.get("llama_server_found"),
        "failures": data.get("failures") if isinstance(data.get("failures"), list) else [],
        "failed_checks": failed_checks,
        "nvidia_smi": nvidia_smi,
    }


def summarize_dir(report_dir: Path, log_tail_lines: int = 60) -> dict[str, Any]:
    preflight_paths = sorted(report_dir.glob("**/preflight-*.json"), key=lambda p: p.stat().st_mtime)
    server_logs = sorted(report_dir.glob("**/server-*.log"), key=lambda p: p.stat().st_mtime)
    preflights = [load_preflight(path) for path in preflight_paths]
    latest_preflight = preflights[-1] if preflights else None
    latest_log = server_logs[-1] if server_logs else None
    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "report_dir": str(report_dir),
        "preflight_count": len(preflight_paths),
        "server_log_count": len(server_logs),
        "latest_preflight": latest_preflight,
        "latest_server_log": str(latest_log) if latest_log else "",
        "latest_server_log_tail": tail_text(latest_log, log_tail_lines) if latest_log else "",
    }
    if latest_preflight is None:
        summary["status"] = "NO_PREFLIGHT"
    elif latest_preflight.get("ok") is True:
        summary["status"] = "PREFLIGHT_OK"
    else:
        summary["status"] = "PREFLIGHT_FAILED"
    return summary


def import_report_bundle(args: argparse.Namespace) -> dict[str, Any]:
    receive = None if args.skip_taildrop else receive_taildrop(args.inbox, args.wait)
    archive = args.archive
    if archive is None:
        candidates = archive_candidates(args.inbox)
        if not candidates:
            return {
                "schema": SCHEMA,
                "imported": False,
                "receive": receive,
                "error": f"no qwen36-node-reports-*.zip bundle found under {args.inbox}",
            }
        archive = candidates[0]
    dest = extract_archive(archive, args.out_dir, args.replace)
    summary = summarize_dir(dest, log_tail_lines=args.log_tail_lines)
    summary.update({"imported": True, "archive": str(archive), "receive": receive})
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inbox", type=Path, default=DEFAULT_INBOX)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--archive", type=Path, help="specific qwen36-node-reports-*.zip to import")
    parser.add_argument("--wait", action="store_true", help="wait for one Taildrop file before importing")
    parser.add_argument("--skip-taildrop", action="store_true", help="do not run tailscale file get first")
    parser.add_argument("--replace", action="store_true", help="replace an existing extracted report directory")
    parser.add_argument("--log-tail-lines", type=int, default=60)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = import_report_bundle(args)
    except Exception as exc:
        print(json.dumps({"schema": SCHEMA, "imported": False, "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("imported") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
