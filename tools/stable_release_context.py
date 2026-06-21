#!/usr/bin/env python3
"""Gate context for fleet's stable-release skill.

A stable release promotes an existing rolling `vX.Y.Z` tag to a named
`stable/<codename>` tag. It does not bump VERSION or build artifacts. This helper
is read-only: it gathers candidate/tag/idempotency data and gate rows into JSON.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SEMVER_TAG_RE = re.compile(r"^v\d+\.\d+\.\d+$")
CODENAME_RE = re.compile(r"^\d{4}-\d{2}-[a-z][a-z0-9-]{2,20}$")
CI_BILLING_PATTERNS = (
    "payments have failed",
    "spending limit",
    "billing & plans",
)


def run(cmd: list[str], *, cwd: Path | None = None, timeout: int = 600) -> tuple[int, str]:
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


def git(root: Path, args: list[str]) -> str:
    code, out = run(["git", *args], cwd=root)
    return out if code == 0 else ""


def tag_reachable(root: Path, tag: str, ref: str = "HEAD") -> bool:
    code, _ = run(["git", "merge-base", "--is-ancestor", f"{tag}^{{commit}}", ref], cwd=root)
    return code == 0


def latest_semver_tag(root: Path, *, reachable_only: bool = True) -> str | None:
    for line in git(root, ["tag", "--sort=-v:refname"]).splitlines():
        tag = line.strip()
        if not SEMVER_TAG_RE.match(tag):
            continue
        if not reachable_only or tag_reachable(root, tag):
            return tag
    return None


def tag_sha(root: Path, tag: str) -> str | None:
    out = git(root, ["rev-list", "-n1", tag]).strip()
    return out or None


def tag_epoch(root: Path, tag: str) -> int | None:
    raw = git(root, ["log", "-1", "--format=%ct", tag]).strip()
    try:
        return int(raw)
    except ValueError:
        return None


def idempotency(root: Path, codename: str, candidate_sha: str | None) -> dict:
    tag = f"stable/{codename}"
    exists = bool(git(root, ["tag", "-l", tag]).strip())
    existing_sha = tag_sha(root, tag) if exists else None
    evidence = root / "docs" / "stable-releases" / f"{codename}.md"
    return {
        "tag_exists": exists,
        "tag_sha": existing_sha,
        "tag_matches_candidate": existing_sha == candidate_sha if exists and candidate_sha else None,
        "evidence_file_exists": evidence.exists(),
        "evidence_path": f"docs/stable-releases/{codename}.md",
    }


def stable_collision(root: Path, codename: str, candidate_sha: str | None) -> str | None:
    if not candidate_sha:
        return None
    this_tag = f"stable/{codename}"
    for raw in git(root, ["tag", "-l", "stable/*"]).splitlines():
        tag = raw.strip()
        if tag and tag != this_tag and tag_sha(root, tag) == candidate_sha:
            return tag
    return None


def previous_stable(root: Path, codename: str) -> dict | None:
    this_tag = f"stable/{codename}"
    code, out = run([
        "git", "for-each-ref",
        "--sort=-creatordate",
        "--format=%(refname:short)|%(objectname)|%(*objectname)",
        "refs/tags/stable/",
    ], cwd=root)
    if code != 0:
        return None
    for line in out.splitlines():
        parts = line.split("|", 2)
        if len(parts) != 3:
            continue
        tag, raw_sha, peeled_sha = parts
        if tag == this_tag:
            continue
        return {"tag": tag, "sha": peeled_sha or raw_sha}
    return None


def gate_tag_age(root: Path, candidate_tag: str | None, window_days: int, now: int | None = None) -> dict:
    if not candidate_tag:
        return {"name": "tag_age", "pass": False, "reason": "no candidate tag", "window_days": window_days}
    created = tag_epoch(root, candidate_tag)
    if created is None:
        return {"name": "tag_age", "pass": False, "reason": "could not read candidate commit time", "window_days": window_days}
    now = int(time.time()) if now is None else now
    age_days = round((now - created) / 86400.0, 2)
    return {"name": "tag_age", "pass": age_days >= window_days, "age_days": age_days, "window_days": window_days}


def gate_release_topology(root: Path, candidate_tag: str | None) -> dict:
    code, out = run([sys.executable, str(root / "tools" / "release_context.py"), "--no-previews"], cwd=root)
    if code != 0:
        return {"name": "release_topology", "pass": False, "reason": out.strip()[-300:]}
    payload = json.loads(out)
    tag_drift = payload.get("tag_drift") or {}
    blockers: list[str] = []
    if not candidate_tag:
        blockers.append("no candidate tag")
    elif not tag_reachable(root, candidate_tag):
        blockers.append(f"{candidate_tag} is not reachable from HEAD")
    if tag_drift.get("source_behind_reachable_tag"):
        blockers.append("VERSION is behind the reachable release tag")
    workflows = payload.get("workflows_parse_ok") or {}
    if workflows.get("ok") is False:
        blockers.append("workflow YAML is unparseable")
    return {
        "name": "release_topology",
        "pass": not blockers,
        "blockers": blockers,
        "last_tag": payload.get("last_tag"),
        "latest_any_tag": payload.get("latest_any_tag"),
        "unreachable_newer_tags": payload.get("unreachable_newer_tags") or [],
        "tag_drift": tag_drift,
        "workflows_parse_ok": workflows,
    }


def gate_local_release_tests(root: Path, skip: bool) -> dict:
    tests = [
        "tools/release_context_test.py",
        "tools/release_cut_test.py",
        "tools/release_manifest_test.py",
        "tools/release_dry_run_test.py",
        "tools/release_tag_test.py",
        "tools/release_cadence_workflow_test.py",
        "tools/release_status_test.py",
        "tools/release_lock_test.py",
        "tools/safe_ff_sync_test.py",
        "tools/release_decide_test.py",
        "tools/stable_release_context_test.py",
        "tools/stable_release_promote_test.py",
    ]
    if skip:
        return {"name": "local_release_tests", "pass": True, "skipped": True, "tests": tests}
    rows: list[dict] = []
    ok = True
    for test in tests:
        code, out = run([sys.executable, test], cwd=root, timeout=300)
        rows.append({"test": test, "exit_code": code, "tail": "\n".join(out.splitlines()[-6:])})
        ok = ok and code == 0
    return {"name": "local_release_tests", "pass": ok, "tests": rows}


def gate_dos_doctor(root: Path, skip: bool) -> dict:
    if skip:
        return {"name": "dos_doctor", "pass": True, "skipped": True}
    for cmd in (["dos", "doctor", "--workspace", str(root)],
                [sys.executable, "-m", "dos.cli", "doctor", "--workspace", str(root)]):
        code, out = run(cmd, cwd=root, timeout=120)
        if code == 0:
            return {"name": "dos_doctor", "pass": True, "exit_code": code, "tail": "\n".join(out.splitlines()[-6:])}
    return {"name": "dos_doctor", "pass": False, "exit_code": code, "tail": "\n".join(out.splitlines()[-8:])}


def _json_value(text: str) -> Any:
    try:
        return json.loads(text or "null")
    except json.JSONDecodeError:
        return None


def classify_ci_annotations(messages: list[str]) -> dict:
    joined = "\n".join(messages).lower()
    if any(pattern in joined for pattern in CI_BILLING_PATTERNS):
        return {
            "kind": "billing",
            "verdict": "CI_BILLING",
            "action": "fix_ci_billing",
            "detail": (
                "GitHub Actions did not start the ci.yml job because account billing, "
                "payments, or spending limits need operator attention"
            ),
        }
    if messages:
        return {
            "kind": "workflow_start_failure",
            "verdict": "CI_START_FAILURE",
            "action": "inspect_ci_annotation",
            "detail": messages[0][:300],
        }
    return {
        "kind": "unknown",
        "verdict": "RED",
        "action": "inspect_ci",
        "detail": "ci.yml is red, but no job annotation was available",
    }


def diagnose_failed_ci_run(run_id: object) -> dict:
    if not run_id:
        return {**classify_ci_annotations([]), "status": "unavailable", "reason": "missing run id"}
    code, out = run(["gh", "run", "view", str(run_id), "--json", "jobs"], timeout=60)
    payload = _json_value(out)
    if code != 0 or not isinstance(payload, dict):
        return {
            **classify_ci_annotations([]),
            "status": "unavailable",
            "reason": out.strip()[-300:] or "gh run view failed",
        }
    annotations: list[dict[str, Any]] = []
    jobs_brief: list[dict[str, Any]] = []
    for job in payload.get("jobs") or []:
        if not isinstance(job, dict):
            continue
        job_id = job.get("databaseId")
        jobs_brief.append({
            "id": job_id,
            "name": job.get("name"),
            "conclusion": job.get("conclusion"),
            "status": job.get("status"),
            "steps": len(job.get("steps") or []),
        })
        if not job_id:
            continue
        ann_code, ann_out = run([
            "gh", "api", f"repos/:owner/:repo/check-runs/{job_id}/annotations",
        ], timeout=60)
        ann_payload = _json_value(ann_out)
        if ann_code == 0 and isinstance(ann_payload, list):
            annotations.extend(item for item in ann_payload if isinstance(item, dict))
        elif ann_code != 0 and not annotations:
            annotations.append({"message": ann_out.strip()[-300:] or "could not read check-run annotations"})
    messages = [str(item.get("message") or "") for item in annotations if item.get("message")]
    classification = classify_ci_annotations(messages)
    return {
        "status": "diagnosed" if messages else "undifferentiated",
        "jobs": jobs_brief,
        "annotations": [
            {
                "level": item.get("annotation_level"),
                "path": item.get("path"),
                "message": item.get("message"),
            }
            for item in annotations
        ],
        **classification,
    }


def gate_ci_green(candidate_sha: str | None, skip: bool) -> dict:
    if skip:
        return {"name": "ci_green", "pass": True, "skipped": True, "advisory": True}
    if not candidate_sha:
        return {"name": "ci_green", "pass": True, "advisory": True, "verdict": "NO_SIGNAL", "reason": "no candidate SHA"}
    code, out = run([
        "gh", "run", "list", "--workflow", "ci.yml", "--commit", candidate_sha,
        "--json", "databaseId,status,conclusion,url", "--limit", "10",
    ], timeout=60)
    if code != 0:
        return {"name": "ci_green", "pass": True, "advisory": True, "verdict": "NO_SIGNAL", "reason": out.strip()[:300]}
    try:
        runs = json.loads(out or "[]")
    except json.JSONDecodeError:
        return {"name": "ci_green", "pass": True, "advisory": True, "verdict": "NO_SIGNAL", "reason": "gh output was not JSON"}
    if any(r.get("conclusion") == "success" for r in runs):
        return {"name": "ci_green", "pass": True, "advisory": False, "verdict": "GREEN", "runs": runs[:3]}
    failed = [r for r in runs if r.get("status") == "completed" and r.get("conclusion") != "success"]
    if failed:
        diagnosis = diagnose_failed_ci_run(failed[0].get("databaseId"))
        verdict = str(diagnosis.get("verdict") or "RED")
        row = {
            "name": "ci_green",
            "pass": False,
            "advisory": False,
            "verdict": verdict,
            "runs": failed[:3],
            "diagnosis": diagnosis,
        }
        if diagnosis.get("action"):
            row["action"] = diagnosis.get("action")
        if diagnosis.get("detail"):
            row["reason"] = diagnosis.get("detail")
        return row
    return {"name": "ci_green", "pass": True, "advisory": True, "verdict": "PENDING", "runs": runs[:3]}


def build_context(args: argparse.Namespace) -> dict:
    root = repo_root()
    candidate_tag = args.from_tag or latest_semver_tag(root, reachable_only=True)
    candidate_sha = tag_sha(root, candidate_tag) if candidate_tag else None
    idem = idempotency(root, args.codename, candidate_sha)
    collision = stable_collision(root, args.codename, candidate_sha)

    blockers: list[str] = []
    if not CODENAME_RE.match(args.codename):
        blockers.append("codename must match YYYY-MM-<word>")
    if not candidate_tag:
        blockers.append("no reachable vX.Y.Z tag exists to promote")
    if idem["tag_exists"] and idem["tag_matches_candidate"] is False:
        blockers.append("stable codename already points at a different commit")
    if collision:
        blockers.append(f"candidate already promoted as {collision}")

    gate = {
        "release_topology": gate_release_topology(root, candidate_tag),
        "local_release_tests": gate_local_release_tests(root, args.skip_tests),
        "dos_doctor": gate_dos_doctor(root, args.skip_dos),
        "ci_green": gate_ci_green(candidate_sha, args.skip_ci),
        "tag_age": gate_tag_age(root, candidate_tag, args.window_days),
    }
    for row in gate.values():
        if not row.get("pass", False):
            blockers.append(f"gate row {row.get('name')} did not pass")

    all_green = not blockers
    return {
        "candidate_tag": candidate_tag,
        "candidate_sha": candidate_sha,
        "codename": args.codename,
        "window_days": args.window_days,
        "previous_stable": previous_stable(root, args.codename),
        "tag_collision": collision,
        "idempotency": idem,
        "gate": gate,
        "summary": {
            "all_green": all_green or args.force_promote,
            "blockers": blockers,
            "forced": bool(args.force_promote and not all_green),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate context for fleet stable-release.")
    parser.add_argument("--codename", required=True)
    parser.add_argument("--from", dest="from_tag", default=None)
    parser.add_argument("--window-days", type=int, default=3)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-dos", action="store_true")
    parser.add_argument("--skip-ci", action="store_true")
    parser.add_argument("--force-promote", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    payload = build_context(args)
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if payload["summary"]["all_green"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
