#!/usr/bin/env python3
"""Checked GitHub release-page publisher for fleet releases.

This is the last step after `release_tag.py`: the immutable `vX.Y.Z` tag already
exists, and this helper creates (or confirms) the GitHub release page from the
release note committed at that tag. Default mode is read-only.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
Runner = Callable[[list[str]], tuple[int, str]]


def run(cmd: list[str], *, cwd: Path | None = None, timeout: int = 300) -> tuple[int, str]:
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


def git(root: Path, args: list[str], *, timeout: int = 300) -> tuple[int, str]:
    return run(["git", *args], cwd=root, timeout=timeout)


def normalize_version(version: str | None) -> str:
    return (version or "").strip().lstrip("v")


def release_tag(version: str) -> str:
    return f"v{version}"


def tag_sha(root: Path, tag: str) -> str | None:
    code, out = git(root, ["rev-list", "-n1", tag])
    return out.strip() if code == 0 and out.strip() else None


def show_text(root: Path, ref: str, path: str) -> tuple[bool, str]:
    code, out = git(root, ["show", f"{ref}:{path}"])
    return code == 0, out


def read_version_file(root: Path) -> str | None:
    try:
        return normalize_version((root / "VERSION").read_text(encoding="utf-8").splitlines()[0])
    except (OSError, IndexError):
        return None


def _json_value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def github_release_view(root: Path, tag: str, *, runner: Runner | None = None) -> dict:
    runner = runner or (lambda cmd: run(cmd, cwd=root, timeout=120))
    code, out = runner([
        "gh",
        "release",
        "view",
        tag,
        "--json",
        "tagName,url,publishedAt,isDraft,isPrerelease",
    ])
    if code == 0:
        payload = _json_value(out)
        if isinstance(payload, dict):
            payload["status"] = "present"
            return payload
        return {"status": "unknown", "tag": tag, "reason": "gh emitted non-JSON output"}
    lowered = out.lower()
    if "not found" in lowered or "could not resolve" in lowered or "no release found" in lowered:
        return {"status": "missing", "tag": tag, "reason": out.strip()[-300:]}
    return {"status": "unknown", "tag": tag, "reason": out.strip()[-300:] or "gh release view failed"}


def build_context(
    root: Path,
    *,
    version: str | None = None,
    tag: str | None = None,
    skip_gh: bool = False,
    runner: Runner | None = None,
) -> dict:
    normalized = normalize_version(version) or read_version_file(root) or ""
    tag_name = tag or release_tag(normalized)
    if tag_name.startswith("v") and not normalized:
        normalized = normalize_version(tag_name)
    note_rel = f"docs/releases/{tag_name}.md"
    errors: list[str] = []
    skips: list[str] = []
    warnings: list[str] = []

    result: dict[str, Any] = {
        "ok": False,
        "dry_run": True,
        "version": normalized,
        "tag": tag_name,
        "checks": {},
        "github_release": {},
        "errors": errors,
        "warnings": warnings,
        "idempotent_skips": skips,
    }
    if not SEMVER_RE.match(normalized):
        errors.append("invalid_version")
    if tag_name != release_tag(normalized):
        errors.append("tag_version_mismatch")

    local_sha = tag_sha(root, tag_name)
    result["sha"] = local_sha
    tag_check = {"pass": bool(local_sha), "local_sha": local_sha}
    result["checks"]["tag_exists"] = tag_check
    if not local_sha:
        errors.append("missing_local_tag")

    note_ok, note_text = show_text(root, tag_name, note_rel) if local_sha else (False, "")
    note_check = {"pass": note_ok, "path": note_rel, "bytes": len(note_text.encode("utf-8")) if note_ok else 0}
    result["checks"]["release_note_at_tag"] = note_check
    if not note_ok:
        errors.append(f"missing_release_note_at_tag:{note_rel}")

    if skip_gh:
        result["github_release"] = {"status": "skipped", "tag": tag_name}
        skips.append("github_release_check_skipped")
    else:
        gh_release = github_release_view(root, tag_name, runner=runner)
        result["github_release"] = gh_release
        if gh_release.get("status") == "present":
            if gh_release.get("tagName") and gh_release.get("tagName") != tag_name:
                errors.append("github_release_tag_mismatch")
            else:
                skips.append("github_release_already_exists")
        elif gh_release.get("status") == "missing":
            result["planned_create"] = True
        else:
            errors.append("github_release_status_unknown:" + str(gh_release.get("reason") or "unknown"))

    result["ok"] = not errors
    return result


def publish_release(root: Path, context: dict, *, execute: bool = False,
                    runner: Runner | None = None) -> dict:
    result = dict(context)
    result["dry_run"] = not execute
    result["release_created"] = False
    result["planned_create"] = bool(context.get("planned_create"))
    if not context.get("ok"):
        return result

    tag_name = str(context.get("tag") or "")
    gh_status = (context.get("github_release") or {}).get("status")
    if gh_status == "present":
        result.setdefault("idempotent_skips", []).append("release_create_skipped_existing")
        return result
    if gh_status == "skipped":
        result.setdefault("idempotent_skips", []).append("release_create_skipped_gh_check")
        return result
    if not execute:
        return result

    note_rel = str((context.get("checks") or {}).get("release_note_at_tag", {}).get("path") or "")
    note_ok, note_text = show_text(root, tag_name, note_rel)
    if not note_ok:
        result["ok"] = False
        result.setdefault("errors", []).append(f"missing_release_note_at_tag:{note_rel}")
        return result

    runner = runner or (lambda cmd: run(cmd, cwd=root, timeout=300))
    with tempfile.TemporaryDirectory(prefix="fleet-release-notes-") as tmp:
        notes_path = Path(tmp) / f"{tag_name}.md"
        notes_path.write_text(note_text, encoding="utf-8", newline="")
        code, out = runner([
            "gh",
            "release",
            "create",
            tag_name,
            "--title",
            tag_name,
            "--notes-file",
            str(notes_path),
            "--verify-tag",
        ])
    if code != 0:
        result["ok"] = False
        result.setdefault("errors", []).append("github_release_create_failed:" + out.strip()[-300:])
        return result
    result["release_created"] = True
    result["github_release_create"] = {"exit_code": code, "tail": out.strip()[-300:]}
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify and create a GitHub release page for a fleet tag.")
    parser.add_argument("--version", default=None, help="target X.Y.Z; defaults to VERSION")
    parser.add_argument("--tag", default=None, help="target vX.Y.Z; defaults from --version")
    parser.add_argument("--execute", action="store_true", help="create the GitHub release if it is missing")
    parser.add_argument("--skip-gh", action="store_true", help="skip gh release view/create; intended for tests")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    root = repo_root()
    try:
        context = build_context(root, version=args.version, tag=args.tag, skip_gh=args.skip_gh)
        result = publish_release(root, context, execute=args.execute) if args.execute else context
    except Exception as exc:
        result = {"ok": False, "errors": [str(exc)]}

    if args.as_json:
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        verdict = "OK" if result.get("ok") else "REFUSED"
        mode = "created" if result.get("release_created") else "dry-run"
        print(f"[release-publish] {verdict} {result.get('tag', '')} ({mode})")
        for skip in result.get("idempotent_skips") or []:
            print(f"  skip: {skip}")
        for warning in result.get("warnings") or []:
            print(f"  warning: {warning}")
        for error in result.get("errors") or []:
            print(f"  ERROR: {error}", file=sys.stderr)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
