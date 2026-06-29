#!/usr/bin/env python3
"""One read-only status fold for fleet releases.

This is the operator front door for the release process: it folds the rolling
release decision, optional cut dry-run, cadence workflow presence, GitHub release
page state, and stable-channel lag into one JSON record. It never edits files,
commits, tags, pushes, or writes scratch files.
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

SCHEMA = "fleet-release-status/1"
SEMVER_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")
STABLE_RE = re.compile(r"^stable/.+")
CI_BILLING_PATTERNS = (
    "payments have failed",
    "spending limit",
    "billing & plans",
)
ACTION_NEXT_ACTIONS = {
    "clean_worktree",
    "cut_release",
    "fix_ci_billing",
    "fix_ci",
    "confirm_ci",
    "fix_workflow",
    "fix_version_topology",
    "repair_stable_evidence",
    "promote_stable",
    "cut_release_hot_tree",
    "hold",
    "consider_stable",
}
RELEASE_RELEVANT_DIRTY_PREFIXES = (
    ".claude/skills/release/",
    ".claude/skills/stable-release/",
    ".github/workflows/",
    "docs/releases/",
    "docs/stable-releases/",
    "tools/release_",
    "tools/stable_release_",
)
RELEASE_RELEVANT_DIRTY_EXACT = {
    "VERSION",
    ".claude/project.yaml",
    "tools/safe_ff_sync.py",
    "tools/safe_ff_sync_test.py",
}


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


def load_json(cmd: list[str], root: Path, *, timeout: int = 300) -> tuple[dict | None, int, str]:
    code, out = run(cmd, cwd=root, timeout=timeout)
    try:
        payload = json.loads(out)
    except json.JSONDecodeError:
        return None, code, out
    return payload if isinstance(payload, dict) else {"value": payload}, code, out


def semver_tuple(value: str | None) -> tuple[int, int, int] | None:
    match = SEMVER_RE.match((value or "").strip())
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def semver_key(value: str) -> tuple[int, int, int]:
    return semver_tuple(value) or (-1, -1, -1)


def tag_sha(root: Path, tag: str) -> str | None:
    code, out = git(root, ["rev-list", "-n1", tag])
    return out.strip() if code == 0 and out.strip() else None


def tag_epoch(root: Path, tag: str) -> int | None:
    code, out = git(root, ["log", "-1", "--format=%ct", tag])
    if code != 0:
        return None
    try:
        return int(out.strip())
    except ValueError:
        return None


def version_at_ref(root: Path, ref: str) -> str | None:
    code, out = git(root, ["show", f"{ref}:VERSION"])
    if code != 0:
        return None
    first = out.splitlines()[0].strip() if out.splitlines() else ""
    return first or None


def semver_tags(root: Path, *, merged: bool = True) -> list[str]:
    args = ["tag", "--sort=v:refname"]
    if merged:
        args.extend(["--merged", "HEAD"])
    code, out = git(root, args)
    if code != 0:
        return []
    return [line.strip() for line in out.splitlines() if semver_tuple(line.strip())]


def stable_tags(root: Path) -> list[dict[str, Any]]:
    code, out = git(root, [
        "for-each-ref",
        "--sort=-creatordate",
        "--format=%(refname:short)|%(objectname)|%(*objectname)|%(creatordate:iso-strict)",
        "refs/tags/stable/",
    ])
    if code != 0:
        return []
    rows: list[dict[str, Any]] = []
    for line in out.splitlines():
        parts = line.split("|", 3)
        if len(parts) != 4 or not STABLE_RE.match(parts[0]):
            continue
        tag, raw_sha, peeled_sha, created = parts
        sha = peeled_sha or raw_sha
        rows.append({
            "tag": tag,
            "sha": sha,
            "short_sha": sha[:12],
            "created_at": created or None,
            "version": version_at_ref(root, sha),
        })
    return rows


def _frontmatter_text(text: str) -> str:
    if not text.startswith("---\n"):
        return ""
    end = text.find("\n---", 4)
    return text[4:end] if end != -1 else ""


def parse_frontmatter_scalars(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in _frontmatter_text(text).splitlines():
        if not line or line.startswith(" ") or ":" not in line:
            continue
        key, raw = line.split(":", 1)
        key = key.strip()
        value = raw.strip()
        if not key:
            continue
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value.strip("'\"")
        if parsed is None:
            values[key] = ""
        else:
            values[key] = str(parsed).strip()
    return values


def sha_matches(a: str | None, b: str | None) -> bool:
    left = (a or "").strip().lower()
    right = (b or "").strip().lower()
    if not left or not right:
        return False
    return left == right or left.startswith(right) or right.startswith(left)


def stable_evidence_status(root: Path, stable: list[dict[str, Any]]) -> dict:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for tag in stable:
        tag_name = str(tag.get("tag") or "")
        codename = tag_name.split("/", 1)[1] if "/" in tag_name else tag_name
        evidence_rel = f"docs/stable-releases/{codename}.md"
        issues: list[str] = []
        fields: dict[str, str] = {}
        code, evidence_text = git(root, ["show", f"HEAD:{evidence_rel}"])
        if code != 0:
            issues.append(f"missing committed evidence file {evidence_rel}")
        else:
            fields = parse_frontmatter_scalars(evidence_text)
            if not fields:
                issues.append(f"{evidence_rel} at HEAD has missing or unparseable frontmatter")

        expected_sha = str(tag.get("sha") or "")
        candidate_sha = fields.get("candidate_sha")
        if fields and not candidate_sha:
            issues.append("frontmatter.candidate_sha missing")
        elif fields and not sha_matches(candidate_sha, expected_sha):
            issues.append(
                f"frontmatter.candidate_sha={candidate_sha!r} does not match {tag_name} commit {expected_sha[:12]}"
            )

        underlying = fields.get("underlying_version") if fields else None
        if fields and not underlying:
            issues.append("frontmatter.underlying_version missing")
        elif fields and semver_tuple(underlying) is None:
            issues.append(f"frontmatter.underlying_version={underlying!r} is not vX.Y.Z")
        elif fields and underlying:
            rolling_sha = tag_sha(root, underlying)
            if not rolling_sha:
                issues.append(f"underlying rolling tag {underlying} does not exist")
            elif not sha_matches(rolling_sha, expected_sha):
                issues.append(
                    f"underlying rolling tag {underlying} points at {rolling_sha[:12]}, not {expected_sha[:12]}"
                )

        if fields and fields.get("codename") and fields["codename"] != codename:
            issues.append(f"frontmatter.codename={fields['codename']!r} does not match {codename!r}")

        row = {
            "tag": tag_name,
            "evidence_path": evidence_rel,
            "ok": not issues,
            "issues": issues,
        }
        rows.append(row)
        for issue in issues:
            failures.append({"tag": tag_name, "evidence_path": evidence_rel, "detail": issue})
    return {
        "ok": not failures,
        "checked": len(stable),
        "failures": failures,
        "rows": rows,
    }


def committed_path_exists(root: Path, rel: str) -> bool:
    code, _ = git(root, ["cat-file", "-e", f"HEAD:{rel}"])
    return code == 0


def stable_tag_exists(root: Path, codename: str) -> bool:
    code, _ = git(root, ["rev-parse", "--verify", "--quiet", f"refs/tags/stable/{codename}"])
    return code == 0


def suggest_stable_codename(root: Path, *, now: int | None = None) -> str:
    now = int(time.time()) if now is None else now
    prefix = time.strftime("%Y-%m-stable", time.gmtime(now))
    for idx in range(1, 100):
        codename = prefix if idx == 1 else f"{prefix}-{idx}"
        evidence_rel = f"docs/stable-releases/{codename}.md"
        if not stable_tag_exists(root, codename) and not committed_path_exists(root, evidence_rel):
            return codename
    return f"{prefix}-{now}"


def stable_candidate_status(
    root: Path,
    rolling_tags: list[str],
    stable: list[dict[str, Any]],
    *,
    window_days: int,
    now: int | None = None,
) -> dict:
    if not rolling_tags:
        return {
            "ready": False,
            "state": "no_candidate",
            "reason": "no reachable rolling vX.Y.Z tag exists",
            "window_days": window_days,
        }
    candidate_tag = rolling_tags[-1]
    candidate_sha = tag_sha(root, candidate_tag)
    row: dict[str, Any] = {
        "candidate_tag": candidate_tag,
        "candidate_sha": candidate_sha,
        "window_days": window_days,
        "ready": False,
    }
    if not candidate_sha:
        row.update({"state": "unresolved", "reason": f"could not resolve {candidate_tag}"})
        return row

    promoted = [
        str(item.get("tag"))
        for item in stable
        if sha_matches(str(item.get("sha") or ""), candidate_sha)
    ]
    if promoted:
        row.update({
            "state": "already_promoted",
            "stable_tag": promoted[0],
            "reason": f"{candidate_tag} is already promoted as {promoted[0]}",
        })
        return row

    created = tag_epoch(root, candidate_tag)
    if created is None:
        row.update({"state": "unknown_age", "reason": f"could not read {candidate_tag} commit time"})
        return row

    now = int(time.time()) if now is None else now
    age_days = round((now - created) / 86400.0, 2)
    row["age_days"] = age_days
    row["suggested_codename"] = suggest_stable_codename(root, now=now)
    row["dry_run_command"] = (
        f"python tools/stable_release_promote.py --codename {row['suggested_codename']} "
        f"--from {candidate_tag} --dry-run --json"
    )
    if age_days < window_days:
        row.update({
            "state": "soaking",
            "remaining_days": round(window_days - age_days, 2),
            "reason": f"{candidate_tag} has soaked {age_days}d of {window_days}d",
        })
        return row

    row.update({
        "ready": True,
        "state": "ready",
        "reason": f"{candidate_tag} has soaked {age_days}d; stable promotion preflight is due",
    })
    return row


def stable_summary(
    root: Path,
    rolling_tags: list[str],
    *,
    window_days: int = 3,
    now: int | None = None,
) -> dict:
    stable = stable_tags(root)
    evidence = stable_evidence_status(root, stable)
    candidate = stable_candidate_status(
        root,
        rolling_tags,
        stable,
        window_days=window_days,
        now=now,
    )
    if not stable:
        return {
            "latest_stable": None,
            "stable_lag": None,
            "evidence": evidence,
            "candidate": candidate,
            "recommendation": stable_recommendation(None, candidate, no_stable=True),
        }
    latest = stable[0]
    stable_version = latest.get("version")
    stable_t = semver_tuple(str(stable_version or ""))
    if stable_t is None:
        return {
            "latest_stable": latest,
            "stable_lag": None,
            "evidence": evidence,
            "candidate": candidate,
            "recommendation": "latest stable tag has no readable VERSION; inspect before promoting another stable",
        }
    newer = [tag for tag in rolling_tags if semver_key(tag) > stable_t]
    lag = len(newer)
    return {
        "latest_stable": latest,
        "stable_lag": lag,
        "newer_rolling_tags": newer,
        "evidence": evidence,
        "candidate": candidate,
        "recommendation": stable_recommendation(lag, candidate, no_stable=False),
    }


def stable_recommendation(lag: int | None, candidate: dict, *, no_stable: bool) -> str:
    state = str(candidate.get("state") or "")
    candidate_tag = candidate.get("candidate_tag")
    if state == "ready":
        return (
            f"promote {candidate_tag} with /stable-release --codename "
            f"{candidate.get('suggested_codename')}"
        )
    if state == "soaking":
        prefix = "no stable tag exists" if no_stable else f"stable lags by {lag} rolling release(s)"
        return f"{prefix}; {candidate.get('reason')}"
    if state == "already_promoted":
        return "stable channel is current"
    if no_stable:
        return "no stable tag exists; promote a soaked rolling tag when one is known good"
    return (
        "stable channel is current"
        if lag == 0 else
        f"stable lags by {lag} rolling release(s); consider /stable-release after soak"
    )


def dirty_summary(root: Path) -> dict:
    code, raw = git(root, ["status", "--porcelain=v1", "-z"])
    entries = [entry for entry in raw.split("\0") if entry] if code == 0 else []
    modified = [entry[3:] for entry in entries if not entry.startswith("?? ")]
    untracked = [entry[3:] for entry in entries if entry.startswith("?? ")]
    release_relevant = [p for p in modified + untracked if is_release_relevant_dirty_path(p)]
    unrelated = [p for p in modified + untracked if p not in release_relevant]
    return {
        "clean": not entries,
        "modified_count": len(modified),
        "untracked_count": len(untracked),
        "modified": modified,
        "untracked": untracked,
        "release_relevant_count": len(release_relevant),
        "release_relevant": release_relevant,
        "unrelated_count": len(unrelated),
        "unrelated": unrelated,
    }


def is_release_relevant_dirty_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    if normalized in RELEASE_RELEVANT_DIRTY_EXACT:
        return True
    return any(normalized.startswith(prefix) for prefix in RELEASE_RELEVANT_DIRTY_PREFIXES)


def dirty_requires_clean_before_status(dirty: dict) -> bool:
    if dirty.get("clean", True):
        return False
    if "release_relevant_count" not in dirty:
        return True
    return int(dirty.get("release_relevant_count") or 0) > 0


def dirty_requires_clean_before_cut(dirty: dict) -> bool:
    return not dirty.get("clean", True)


def github_repo_slug(root: Path) -> str | None:
    code, out = run(["gh", "repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"], cwd=root, timeout=30)
    slug = out.strip()
    return slug if code == 0 and slug and "/" in slug else None


def classify_ci_annotations(messages: list[str]) -> dict:
    joined = "\n".join(messages).lower()
    if any(pattern in joined for pattern in CI_BILLING_PATTERNS):
        return {
            "kind": "billing",
            "action": "fix_ci_billing",
            "detail": (
                "GitHub Actions did not start the ci.yml job because account billing, "
                "payments, or spending limits need operator attention"
            ),
        }
    if messages:
        return {
            "kind": "workflow_start_failure",
            "action": "inspect_ci_annotation",
            "detail": messages[0][:300],
        }
    return {
        "kind": "unknown",
        "action": "inspect_ci",
        "detail": "ci.yml is red, but no job annotation was available",
    }


def first_ci_run(payload: dict | None) -> dict:
    if not payload:
        return {}
    value = payload.get("value")
    runs = value if isinstance(value, list) else payload
    return runs[0] if isinstance(runs, list) and runs and isinstance(runs[0], dict) else {}


def ci_failure_diagnosis(root: Path) -> dict:
    code, head = git(root, ["rev-parse", "HEAD"])
    if code != 0 or not head.strip():
        return {"status": "unavailable", "reason": "could not resolve HEAD"}
    payload, _runs_code, raw = load_json([
        "gh", "run", "list",
        "--workflow", "ci.yml",
        "--commit", head.strip(),
        "--limit", "1",
        "--json", "databaseId,headSha,status,conclusion,displayTitle,url",
    ], root, timeout=30)
    if payload is None:
        return {"status": "unavailable", "reason": raw.strip()[-300:] or "gh run list failed"}
    run_doc = first_ci_run(payload)
    scope = "head"
    if not run_doc:
        payload, _trunk_code, trunk_raw = load_json([
            "gh", "run", "list",
            "--workflow", "ci.yml",
            "--branch", "main",
            "--status", "completed",
            "--limit", "1",
            "--json", "databaseId,headSha,status,conclusion,displayTitle,url",
        ], root, timeout=30)
        if payload is None:
            return {"status": "unavailable", "reason": trunk_raw.strip()[-300:] or "gh trunk run list failed"}
        run_doc = first_ci_run(payload)
        scope = "latest_trunk"
    if not run_doc:
        return {"status": "none", "reason": "no ci.yml run found for HEAD or latest main"}
    if run_doc.get("conclusion") != "failure":
        return {
            "status": "not_failed",
            "scope": scope,
            "run": run_doc,
            "reason": f"{scope} ci.yml run conclusion is {run_doc.get('conclusion') or run_doc.get('status')}",
        }

    run_id = str(run_doc.get("databaseId") or "")
    jobs, jobs_code, jobs_raw = load_json(["gh", "run", "view", run_id, "--json", "jobs"], root, timeout=30)
    if jobs is None:
        return {"status": "unavailable", "run": run_doc, "reason": jobs_raw.strip()[-300:] or "gh run view failed"}
    repo = github_repo_slug(root)
    annotations: list[dict[str, Any]] = []
    jobs_brief: list[dict[str, Any]] = []
    for job in jobs.get("jobs") or []:
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
        if not repo or not job_id:
            continue
        ann, ann_code, ann_raw = load_json([
            "gh", "api", f"repos/{repo}/check-runs/{job_id}/annotations",
        ], root, timeout=30)
        if isinstance(ann, dict) and isinstance(ann.get("value"), list):
            annotations.extend(item for item in ann["value"] if isinstance(item, dict))
        elif ann_code != 0 and not annotations:
            annotations.append({"message": ann_raw.strip()[-300:] or "could not read check-run annotations"})
    messages = [str(item.get("message") or "") for item in annotations if item.get("message")]
    classification = classify_ci_annotations(messages)
    return {
        "status": "diagnosed" if messages else "undifferentiated",
        "scope": scope,
        "run": run_doc,
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


def cadence_status(root: Path) -> dict:
    path = root / ".github" / "workflows" / "release-cadence.yml"
    if not path.exists():
        return {"present": False, "path": ".github/workflows/release-cadence.yml"}
    text = path.read_text(encoding="utf-8", errors="replace")
    manual_dry_run_gate = (
        "inputs.dry_run == false" in text
        or 'inputs.dry_run }}" = "false"' in text
    )
    return {
        "present": True,
        "path": ".github/workflows/release-cadence.yml",
        "schedule": "schedule:" in text,
        "manual_dispatch": "workflow_dispatch:" in text,
        "dry_run_first": "dry_run:" in text and manual_dry_run_gate,
        "single_writer": "group: release-cadence" in text and "cancel-in-progress: false" in text,
        "tag_after_green": "tools/release_tag.py" in text and "--require-ci" in text and "--wait-ci" in text,
        "checked_github_release": "tools/release_publish.py" in text and "--execute" in text,
    }


def github_release_view(root: Path, tag: str | None, *, skip: bool) -> dict:
    if skip:
        return {"status": "skipped", "tag": tag}
    if not tag:
        return {"status": "none", "tag": None, "reason": "no rolling tag"}
    code, out = run([
        "gh", "release", "view", tag,
        "--json", "tagName,url,publishedAt,isDraft,isPrerelease",
    ], cwd=root, timeout=30)
    if code == 0:
        try:
            payload = json.loads(out)
        except json.JSONDecodeError:
            return {"status": "unknown", "tag": tag, "reason": "gh emitted non-JSON output"}
        payload["status"] = "present"
        return payload
    lowered = out.lower()
    if "not found" in lowered or "could not resolve" in lowered:
        return {"status": "missing", "tag": tag, "reason": out.strip()[-300:]}
    return {"status": "unknown", "tag": tag, "reason": out.strip()[-300:] or "gh unavailable"}


def release_decision(root: Path, *, limit_commits: int,
                     require_ci_green: bool, force: bool) -> tuple[dict, dict | None]:
    context, context_code, context_raw = load_json([
        sys.executable,
        str(root / "tools" / "release_context.py"),
        "--no-previews",
        "--limit-commits",
        str(limit_commits),
    ], root, timeout=180)
    if context is None or context_code != 0:
        return {
            "decision": "unknown",
            "reason": "release_context failed",
            "errors": [context_raw.strip()[-500:]],
        }, context
    cmd = [
        sys.executable,
        str(root / "tools" / "release_decide.py"),
        "--json",
        "--limit-commits",
        str(limit_commits),
    ]
    if require_ci_green:
        cmd.append("--require-ci-green")
    if force:
        cmd.append("--force")
    decision, decision_code, decision_raw = load_json(cmd, root, timeout=180)
    if decision is None or decision_code not in (0, 2):
        return {
            "decision": "unknown",
            "reason": "release_decide failed",
            "errors": [decision_raw.strip()[-500:]],
        }, context
    return decision, context


def cut_plan(root: Path, *, limit_commits: int,
             require_ci_green: bool, force: bool, skip: bool) -> dict:
    if skip:
        return {"skipped": True}
    cmd = [
        sys.executable,
        str(root / "tools" / "release_cut.py"),
        "--json",
        "--allow-hold",
        "--limit-commits",
        str(limit_commits),
    ]
    if require_ci_green:
        cmd.append("--require-ci-green")
    if force:
        cmd.append("--force")
    payload, code, raw = load_json(cmd, root, timeout=300)
    if payload is None:
        return {"ok": False, "error": "release_cut emitted no JSON", "detail": raw.strip()[-500:]}
    payload["exit_code"] = code
    return payload


def next_action(
    decision: dict,
    stable: dict,
    dirty: dict | None = None,
    ci_diagnosis: dict | None = None,
) -> dict:
    dirty = dirty or {}
    blockers = [str(b) for b in (decision.get("blockers") or [])]
    if decision.get("decision") == "release":
        if not dirty.get("clean", True):
            modified = int(dirty.get("modified_count") or 0)
            untracked = int(dirty.get("untracked_count") or 0)
            return {
                "kind": "cut_release_hot_tree",
                "detail": (
                    f"cut {decision.get('next_version')} with `fak release ship --execute`; "
                    f"it uses a detached origin/main checkout and leaves this checkout's "
                    f"{modified} modified and {untracked} untracked path(s) untouched"
                ),
            }
        return {
            "kind": "cut_release",
            "detail": f"cut {decision.get('next_version')} with release_cut, push, then tag after green CI",
        }
    if dirty_requires_clean_before_status(dirty):
        modified = int(dirty.get("modified_count") or 0)
        untracked = int(dirty.get("untracked_count") or 0)
        relevant = int(dirty.get("release_relevant_count") or (modified + untracked))
        return {
            "kind": "clean_worktree",
            "detail": (
                f"commit, shelve, or remove {relevant} release-relevant dirty path(s) "
                "before treating release status as trunk evidence"
            ),
        }
    if "CI_BASE_RED" in blockers:
        if (ci_diagnosis or {}).get("action") == "fix_ci_billing":
            return {
                "kind": "fix_ci_billing",
                "detail": str(ci_diagnosis.get("detail")),
            }
        return {"kind": "fix_ci", "detail": "fix current main ci.yml failure before cutting a release"}
    if "CI_RETRY_TO_GREEN" in blockers:
        return {
            "kind": "pause_auto_release",
            "detail": (
                "latest green ci.yml run was a retry; set FAK_AUTO_RELEASE=0 or "
                "confirm a fresh green run before cutting a release"
            ),
        }
    if "CI_BASE_NONE" in blockers or "CI_STATE_UNKNOWN" in blockers:
        return {"kind": "confirm_ci", "detail": "restore or confirm a green main ci.yml signal"}
    if "WORKFLOW_UNPARSEABLE" in blockers:
        return {"kind": "fix_workflow", "detail": "repair GitHub workflow YAML before release"}
    if "VERSION_DRIFT" in blockers or "VERSION_BEHIND_REACHABLE_TAG" in blockers:
        return {"kind": "fix_version_topology", "detail": "reconcile VERSION and semver tag topology"}
    if blockers:
        return {"kind": "hold", "detail": str(decision.get("reason") or blockers[0])}
    stable_evidence = stable.get("evidence") or {}
    if stable_evidence.get("ok") is False:
        failures = stable_evidence.get("failures") or []
        detail = "stable tag evidence is missing or inconsistent"
        if failures and isinstance(failures[0], dict):
            detail = f"repair {failures[0].get('tag')}: {failures[0].get('detail')}"
        return {"kind": "repair_stable_evidence", "detail": detail}
    stable_candidate = stable.get("candidate") or {}
    if stable_candidate.get("ready") is True:
        return {
            "kind": "promote_stable",
            "detail": (
                f"promote {stable_candidate.get('candidate_tag')} with /stable-release "
                f"--codename {stable_candidate.get('suggested_codename')}"
            ),
        }
    if stable.get("stable_lag"):
        return {"kind": "consider_stable", "detail": stable.get("recommendation")}
    return {"kind": "wait", "detail": str(decision.get("reason") or "nothing release-worthy pending")}


def loop_status_fields(action: dict) -> dict:
    kind = str(action.get("kind") or "unknown")
    detail = str(action.get("detail") or kind)
    ok = kind not in ACTION_NEXT_ACTIONS
    return {
        "ok": ok,
        "verdict": "OK" if ok else "ACTION",
        "detail": detail,
    }


def build_status(args: argparse.Namespace) -> dict:
    root = repo_root()
    decision, context = release_decision(
        root,
        limit_commits=args.limit_commits,
        require_ci_green=args.require_ci_green,
        force=args.force,
    )
    rolling_tags = semver_tags(root, merged=True)
    stable = stable_summary(root, rolling_tags, window_days=args.stable_window_days)
    last_tag = (context or {}).get("last_tag") or (rolling_tags[-1] if rolling_tags else None)
    dirty = dirty_summary(root)
    ci_diag = ci_failure_diagnosis(root)
    status = {
        "schema": SCHEMA,
        "root": str(root),
        "head": {
            "sha": ((context or {}).get("head_sha")),
            "branch": ((context or {}).get("current_branch")),
            "dirty": dirty,
        },
        "rolling": {
            "last_tag": last_tag,
            "latest_any_tag": (context or {}).get("latest_any_tag"),
            "commits_since_tag": len((context or {}).get("commits_since_tag") or []),
            "files_touched_since_tag": len((context or {}).get("files_touched_since_tag") or []),
            "tag_drift": (context or {}).get("tag_drift"),
            "ci_on_head": (context or {}).get("ci_on_head"),
            "ci_diagnosis": ci_diag,
            "workflows_parse_ok": (context or {}).get("workflows_parse_ok"),
            "decision": decision,
            "cut_plan": cut_plan(
                root,
                limit_commits=args.limit_commits,
                require_ci_green=args.require_ci_green,
                force=args.force,
                skip=args.skip_cut_plan,
            ),
        },
        "github_release": github_release_view(root, last_tag, skip=args.skip_gh),
        "stable": stable,
        "cadence": cadence_status(root),
    }
    action = next_action(decision, stable, dirty, ci_diag)
    status["next_action"] = action
    status.update(loop_status_fields(action))
    return status


def render_human(status: dict) -> str:
    decision = status.get("rolling", {}).get("decision", {})
    stable = status.get("stable", {})
    action = status.get("next_action", {})
    lines = [
        f"release-status: {decision.get('decision', 'unknown').upper()} - {decision.get('reason', '')}",
        f"  last tag: {status.get('rolling', {}).get('last_tag') or '(none)'}",
        f"  commits since tag: {status.get('rolling', {}).get('commits_since_tag')}",
        f"  next action: {action.get('kind')} - {action.get('detail')}",
    ]
    if stable.get("latest_stable"):
        latest = stable["latest_stable"]
        lines.append(f"  stable: {latest.get('tag')} ({latest.get('version')}); lag={stable.get('stable_lag')}")
    else:
        lines.append("  stable: none")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fold fleet release readiness into one read-only status record.")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--limit-commits", type=int, default=50)
    parser.add_argument("--require-ci-green", action="store_true")
    parser.add_argument("--force", action="store_true", help="bypass only release_decide's substantive floor")
    parser.add_argument("--skip-cut-plan", action="store_true", help="do not run release_cut dry-run")
    parser.add_argument("--skip-gh", action="store_true", help="do not query GitHub release pages")
    parser.add_argument("--stable-window-days", type=int, default=3, help="soak age required before stable candidate is ready")
    parser.add_argument("--check-ready", action="store_true", help="exit 1 unless the rolling release decision is release")
    args = parser.parse_args(argv)

    status = build_status(args)
    if args.as_json:
        json.dump(status, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(render_human(status))
    if args.check_ready and (status.get("rolling", {}).get("decision", {}).get("decision") != "release"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
