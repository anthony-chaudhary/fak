#!/usr/bin/env python3
"""Release-manifest consumer for fleet.

The manifest is the structured counterpart to a release scope token. A producer
can hand `/release` an explicit set of shipped commits and files; this helper
validates that claim against git and returns the dirty pathspec that release_cut
may stage. It is read-only.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMAS = {"release-manifest-v1", "fleet-release-manifest/1"}
ALLOWED_STATUS = {"shipped", "failed", "partial", "awaiting_commit"}
SHA_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")
ZERO_SHA_RE = re.compile(r"^0+$")

AUTO_DEFER_PREFIXES = (
    "docs/_fanout_runs/",
    "docs/_chained_runs/",
    "docs/_dispatch_loops/",
    "releases/",
    "screenshots/",
)
AUTO_DEFER_SUFFIXES = (".zip", ".exe")


class ManifestError(ValueError):
    """Raised when a release manifest is unreadable or invalid."""


def run(cmd: list[str], *, cwd: Path | None = None, timeout: int = 120) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout or ""
    except subprocess.TimeoutExpired as exc:
        return 124, (exc.output or "") + f"\n(timed out after {timeout}s)"
    except OSError as exc:
        return 127, str(exc)


def repo_root() -> Path:
    code, out = run(["git", "rev-parse", "--show-toplevel"])
    return Path(out.strip()) if code == 0 and out.strip() else Path(__file__).resolve().parent.parent


def git(root: Path, args: list[str], *, timeout: int = 120) -> tuple[int, str]:
    return run(["git", *args], cwd=root, timeout=timeout)


def rel_path(path: str) -> str:
    value = path.replace("\\", "/").strip()
    if not value:
        raise ManifestError("manifest path entries must be non-empty")
    p = Path(value)
    if p.is_absolute() or p.drive or any(part == ".." for part in p.parts):
        raise ManifestError(f"manifest path escapes repo: {path!r}")
    return value[2:] if value.startswith("./") else value


def normalize_file_list(values: object, *, where: str) -> list[str]:
    if not isinstance(values, list) or not values:
        raise ManifestError(f"{where} must be a non-empty list")
    if not all(isinstance(value, str) for value in values):
        raise ManifestError(f"{where} must contain only path strings")
    out = sorted({rel_path(value) for value in values})
    return out


def normalize_pick(raw: object, *, idx: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ManifestError(f"picks[{idx}] must be an object")
    status = str(raw.get("status") or "")
    if status not in ALLOWED_STATUS:
        raise ManifestError(f"picks[{idx}].status must be one of {sorted(ALLOWED_STATUS)}")
    commit = str(raw.get("commit") or "")
    if status == "awaiting_commit":
        if commit and not (SHA_RE.match(commit) or ZERO_SHA_RE.match(commit)):
            raise ManifestError(f"picks[{idx}].commit is not a SHA")
    elif not SHA_RE.match(commit):
        raise ManifestError(f"picks[{idx}].commit is required for shipped picks")
    files = normalize_file_list(raw.get("files"), where=f"picks[{idx}].files")
    pick = {
        "plan_id": str(raw.get("plan_id") or raw.get("plan") or ""),
        "phase_id": str(raw.get("phase_id") or raw.get("phase") or ""),
        "status": status,
        "commit": commit.lower(),
        "files": files,
    }
    for key in ("note", "title"):
        if key in raw:
            pick[key] = str(raw[key])
    return pick


def normalize_manifest(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ManifestError("manifest top level must be an object")
    schema = str(raw.get("schema") or "")
    if schema and schema not in SCHEMAS:
        raise ManifestError(f"unsupported manifest schema {schema!r}")
    producer = raw.get("producer") if isinstance(raw.get("producer"), dict) else {}
    picks_raw = raw.get("picks")
    if picks_raw is None and raw.get("files") is not None:
        files = normalize_file_list(raw.get("files"), where="files")
        commit = str((raw.get("commits") or [""])[0] if isinstance(raw.get("commits"), list) else raw.get("commit") or "")
        picks_raw = [{
            "plan_id": str(raw.get("plan_id") or ""),
            "phase_id": str(raw.get("phase_id") or ""),
            "status": str(raw.get("status") or "shipped"),
            "commit": commit,
            "files": files,
        }]
    if not isinstance(picks_raw, list) or not picks_raw:
        raise ManifestError("manifest must contain non-empty picks[] or files[]")
    picks = [normalize_pick(pick, idx=i) for i, pick in enumerate(picks_raw)]
    return {
        "schema": schema or "fleet-release-manifest/1",
        "producer": {
            "skill": str(producer.get("skill") or ""),
            "run_id": str(producer.get("run_id") or ""),
            "packet_path": str(producer.get("packet_path") or ""),
        },
        "theme": raw.get("theme"),
        "picks": picks,
    }


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ManifestError(f"manifest not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"invalid manifest JSON: {exc}") from exc
    return normalize_manifest(raw)


def dirty_paths(root: Path) -> set[str]:
    code, out = git(root, ["status", "--porcelain=v1", "-z"])
    if code != 0:
        raise ManifestError("could not read git status")
    chunks = [chunk for chunk in out.split("\0") if chunk]
    paths: set[str] = set()
    i = 0
    while i < len(chunks):
        item = chunks[i]
        code_xy = item[:2]
        path = item[3:] if len(item) > 3 else item
        paths.add(rel_path(path))
        if code_xy[0] in {"R", "C"} and i + 1 < len(chunks):
            i += 1
            paths.add(rel_path(chunks[i]))
        i += 1
    return paths


def is_auto_deferred(path: str) -> bool:
    p = path.replace("\\", "/")
    return any(p.startswith(prefix) for prefix in AUTO_DEFER_PREFIXES) or p.endswith(AUTO_DEFER_SUFFIXES)


def reachable_from_head(root: Path, sha: str) -> bool:
    if not sha or ZERO_SHA_RE.match(sha):
        return False
    code, _ = git(root, ["merge-base", "--is-ancestor", sha, "HEAD"])
    return code == 0


def commit_subject(root: Path, sha: str) -> str:
    if not sha or ZERO_SHA_RE.match(sha):
        return "(awaiting commit)"
    code, out = git(root, ["log", "-1", "--format=%s", sha])
    return out.strip() if code == 0 and out.strip() else "(subject unavailable)"


def consume_manifest(root: Path, path: Path, *, dirty: set[str] | None = None) -> dict[str, Any]:
    manifest = load_manifest(path)
    dirty_set = {rel_path(p) for p in dirty} if dirty is not None else dirty_paths(root)
    manifest_files = sorted({f for pick in manifest["picks"] for f in pick["files"]})
    non_shipped = [pick for pick in manifest["picks"] if pick["status"] != "shipped"]
    unreachable = [
        pick["commit"]
        for pick in manifest["picks"]
        if pick["status"] == "shipped" and not reachable_from_head(root, pick["commit"])
    ]
    intersected = sorted(set(manifest_files) & dirty_set)
    auto_deferred = [path for path in intersected if is_auto_deferred(path)]
    staged = [path for path in intersected if not is_auto_deferred(path)]
    commits_section_lines = [
        f"- ({pick.get('plan_id') or '?'} {pick.get('phase_id') or '?'}) "
        f"{(pick.get('commit') or '0000000')[:7]} {commit_subject(root, pick.get('commit') or '')}"
        for pick in manifest["picks"]
    ]
    return {
        "ok": not unreachable and not non_shipped and bool(staged),
        "manifest_path": str(path),
        "producer": manifest.get("producer") or {},
        "theme": manifest.get("theme"),
        "picks": manifest["picks"],
        "manifest_files": manifest_files,
        "dirty_paths": sorted(dirty_set),
        "staged_paths": staged,
        "auto_deferred_paths": auto_deferred,
        "unreachable_commits": unreachable,
        "awaiting_commit_picks": non_shipped,
        "commits_section_lines": commits_section_lines,
    }


def exit_code(envelope: dict[str, Any]) -> int:
    if envelope.get("unreachable_commits"):
        return 3
    if envelope.get("awaiting_commit_picks"):
        return 4
    if not envelope.get("staged_paths"):
        return 5
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate or consume a fleet release manifest.")
    sub = parser.add_subparsers(dest="cmd")
    validate = sub.add_parser("validate")
    validate.add_argument("path", type=Path)
    validate.add_argument("--json", action="store_true", dest="as_json")
    consume = sub.add_parser("consume")
    consume.add_argument("path", type=Path)
    consume.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    if args.cmd is None:
        parser.print_help()
        return 2
    root = repo_root()
    try:
        if args.cmd == "validate":
            payload = load_manifest(args.path)
            code = 0
        else:
            payload = consume_manifest(root, args.path)
            code = exit_code(payload)
    except ManifestError as exc:
        if getattr(args, "as_json", False):
            json.dump({"ok": False, "error": str(exc)}, sys.stdout, indent=2)
            sys.stdout.write("\n")
        else:
            print(f"INVALID: {exc}", file=sys.stderr)
        return 2
    if args.as_json:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        if args.cmd == "validate":
            print(f"OK manifest: {args.path} picks={len(payload['picks'])}")
        else:
            print(f"manifest: {args.path}")
            print(f"staged_paths: {len(payload['staged_paths'])}")
            for path in payload["staged_paths"]:
                print(f"  + {path}")
            for path in payload["auto_deferred_paths"]:
                print(f"  ~ {path} (auto-deferred)")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
