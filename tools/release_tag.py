#!/usr/bin/env python3
"""Checked tag-after-green helper for fleet releases.

This is the executable spine for the final `/release` step: inspect a committed
ref, require the release dry-run, fold the GitHub Actions signal for that exact
commit when available, and only then create the local annotated vX.Y.Z tag.

Default mode is read-only. Use `--execute` to create the local tag and `--push`
to push that tag. The helper never creates a release commit and never pushes the
trunk branch.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
Runner = Callable[[list[str]], tuple[int, str]]
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


def git(root: Path, args: list[str], *, timeout: int = 600) -> tuple[int, str]:
    return run(["git", *args], cwd=root, timeout=timeout)


def normalize_version(version: str | None) -> str:
    return (version or "").strip().lstrip("v")


def release_tag(version: str) -> str:
    return f"v{version}"


def ref_sha(root: Path, ref: str) -> str | None:
    code, out = git(root, ["rev-parse", f"{ref}^{{commit}}"])
    return out.strip() if code == 0 and out.strip() else None


def tag_sha(root: Path, tag: str) -> str | None:
    code, out = git(root, ["rev-list", "-n1", tag])
    return out.strip() if code == 0 and out.strip() else None


def show_text(root: Path, sha: str, path: str) -> tuple[bool, str]:
    code, out = git(root, ["show", f"{sha}:{path}"])
    return code == 0, out


def path_exists_at_ref(root: Path, sha: str, path: str) -> bool:
    code, _ = git(root, ["cat-file", "-e", f"{sha}:{path}"])
    return code == 0


def version_at_ref(root: Path, sha: str) -> str | None:
    ok, text = show_text(root, sha, "VERSION")
    if not ok:
        return None
    return normalize_version(text.splitlines()[0] if text.splitlines() else text)


def remote_tag_sha(root: Path, tag: str, remote: str = "origin") -> str | None:
    code, out = git(root, ["ls-remote", "--tags", remote, f"refs/tags/{tag}", f"refs/tags/{tag}^{{}}"], timeout=120)
    if code != 0:
        return None
    direct: str | None = None
    peeled: str | None = None
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        sha, ref = parts[0], parts[1]
        if ref.endswith("^{}"):
            peeled = sha
        elif ref.endswith(f"/{tag}"):
            direct = sha
    return peeled or direct


def is_ancestor(root: Path, ancestor: str, descendant_ref: str) -> bool | None:
    code, _ = git(root, ["rev-parse", "--verify", "--quiet", descendant_ref])
    if code != 0:
        return None
    code, _ = git(root, ["merge-base", "--is-ancestor", ancestor, descendant_ref])
    return code == 0


def load_release_dry_run(root: Path, ref: str) -> dict:
    code, out = run(
        [sys.executable, str(root / "tools" / "release_dry_run.py"), "--json", ref],
        cwd=root,
        timeout=1800,
    )
    try:
        payload = json.loads(out)
    except json.JSONDecodeError:
        payload = {"ok": False, "error": "release_dry_run emitted no JSON", "tail": out[-500:]}
    payload["exit_code"] = code
    return payload


def fold_ci_runs(runs: list[dict[str, Any]], *, require_ci: bool) -> dict:
    if not runs:
        return {
            "pass": not require_ci,
            "verdict": "NO_SIGNAL",
            "advisory": not require_ci,
            "reason": "no GitHub Actions run found for commit",
        }
    run_row = runs[0]
    status = str(run_row.get("status") or "").lower()
    conclusion = str(run_row.get("conclusion") or "").lower()
    if status == "completed":
        if conclusion == "success":
            return {
                "pass": True,
                "verdict": "GREEN",
                "advisory": False,
                "run": run_row,
                "reason": "ci.yml completed successfully for commit",
            }
        return {
            "pass": False,
            "verdict": "RED",
            "advisory": False,
            "run": run_row,
            "reason": f"ci.yml completed with conclusion {conclusion or '(none)'}",
        }
    return {
        "pass": not require_ci,
        "verdict": "PENDING",
        "advisory": not require_ci,
        "run": run_row,
        "reason": f"ci.yml status is {status or '(unknown)'}",
    }


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


def _json_value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def diagnose_failed_ci_run(run_id: object, runner: Runner) -> dict:
    if not run_id:
        return {**classify_ci_annotations([]), "status": "unavailable", "reason": "missing run id"}
    code, out = runner(["gh", "run", "view", str(run_id), "--json", "jobs"])
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
        ann_code, ann_out = runner([
            "gh", "api", f"repos/:owner/:repo/check-runs/{job_id}/annotations",
        ])
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


def ci_signal(root: Path, sha: str, *, workflow: str, require_ci: bool,
              wait: bool = False, runner: Runner | None = None) -> dict:
    runner = runner or (lambda cmd: run(cmd, cwd=root, timeout=900))
    cmd = [
        "gh", "run", "list",
        "--workflow", workflow,
        "--commit", sha,
        "--limit", "10",
        "--json", "databaseId,status,conclusion,url,createdAt,headSha",
    ]
    code, out = runner(cmd)
    if code != 0:
        return {
            "pass": not require_ci,
            "verdict": "NO_SIGNAL",
            "advisory": not require_ci,
            "reason": "gh run list failed or gh is unavailable",
            "detail": out.strip()[-300:],
        }
    try:
        runs = json.loads(out)
    except json.JSONDecodeError:
        return {
            "pass": not require_ci,
            "verdict": "NO_SIGNAL",
            "advisory": not require_ci,
            "reason": "gh run list emitted non-JSON output",
            "detail": out.strip()[-300:],
        }
    if not isinstance(runs, list):
        runs = []
    folded = fold_ci_runs(runs, require_ci=require_ci)
    run_row = folded.get("run") or {}
    if wait and folded.get("verdict") == "PENDING" and run_row.get("databaseId"):
        watch_code, watch_out = runner(["gh", "run", "watch", str(run_row["databaseId"]), "--exit-status"])
        folded["watch_exit_code"] = watch_code
        folded["watch_tail"] = watch_out.strip()[-300:]
        code, out = runner(cmd)
        if code == 0:
            try:
                rerun_rows = json.loads(out)
                if isinstance(rerun_rows, list):
                    folded = fold_ci_runs(rerun_rows, require_ci=require_ci)
                    folded["watched"] = True
            except json.JSONDecodeError:
                pass
    run_row = folded.get("run") or {}
    if folded.get("verdict") == "RED" and run_row.get("databaseId"):
        diagnosis = diagnose_failed_ci_run(run_row.get("databaseId"), runner)
        folded["diagnosis"] = diagnosis
        if diagnosis.get("verdict") != "RED":
            folded["verdict"] = str(diagnosis.get("verdict"))
            folded["reason"] = str(diagnosis.get("detail"))
    return folded


def tag_message(tag: str, sha: str, dry_run_trailer: str) -> str:
    return "\n".join([
        tag,
        "",
        f"commit: {sha}",
        dry_run_trailer,
    ])


def _json_or_tail(out: str) -> dict:
    try:
        payload = json.loads(out)
        return payload if isinstance(payload, dict) else {"raw": payload}
    except json.JSONDecodeError:
        return {"detail": out.strip()[-300:]}


def acquire_release_lock(root: Path, *, tag: str, sha: str, ttl: int) -> dict:
    script = root / "tools" / "release_lock.py"
    if not script.is_file():
        return {"ok": True, "held": False, "skipped": True, "reason": "release_lock.py not present"}
    code, out = run([
        sys.executable,
        str(script),
        "acquire",
        "--ttl",
        str(ttl),
        "--note",
        f"release-tag {tag} {sha[:12]}",
    ], cwd=root)
    payload = _json_or_tail(out)
    payload["exit_code"] = code
    payload["held"] = code == 0 and bool(payload.get("ok"))
    return payload


def release_release_lock(root: Path, *, owner: str | None = None) -> dict:
    script = root / "tools" / "release_lock.py"
    if not script.is_file():
        return {"ok": True, "released": False, "skipped": True}
    cmd = [sys.executable, str(script), "release"]
    if owner:
        cmd += ["--owner", owner]
    code, out = run(cmd, cwd=root)
    payload = _json_or_tail(out)
    payload["exit_code"] = code
    return payload


def verify_release_lock(root: Path) -> dict:
    script = root / "tools" / "release_lock.py"
    if not script.is_file():
        return {"ok": False, "held": False, "skipped": True, "reason": "release_lock.py not present"}
    code, out = run([sys.executable, str(script), "verify"], cwd=root)
    payload = _json_or_tail(out)
    payload["exit_code"] = code
    payload["held"] = code == 0 and bool(payload.get("ok"))
    payload["held_by_parent"] = payload["held"]
    return payload


def build_context(root: Path, *, version: str | None, ref: str,
                  require_ci: bool = False, skip_ci: bool = False,
                  wait_ci: bool = False, skip_dry_run: bool = False,
                  workflow: str = "ci.yml", trunk: str = "main",
                  remote: str = "origin", skip_remote_tag: bool = False) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    skips: list[str] = []
    normalized = normalize_version(version) or normalize_version((root / "VERSION").read_text(encoding="utf-8"))
    tag = release_tag(normalized)
    sha = ref_sha(root, ref)
    result = {
        "ok": False,
        "dry_run": True,
        "version": normalized,
        "tag": tag,
        "ref": ref,
        "sha": sha,
        "checks": {},
        "warnings": warnings,
        "errors": errors,
        "idempotent_skips": skips,
    }
    if not SEMVER_RE.match(normalized):
        errors.append("invalid_version")
    if not sha:
        errors.append("unresolvable_ref")
        return result

    ref_version = version_at_ref(root, sha)
    version_check = {"pass": ref_version == normalized, "at_ref": ref_version}
    result["checks"]["version_at_ref"] = version_check
    if not version_check["pass"]:
        errors.append(f"version_mismatch_at_ref:{ref_version}")

    note_rel = f"docs/releases/{tag}.md"
    note_check = {"pass": path_exists_at_ref(root, sha, note_rel), "path": note_rel}
    result["checks"]["release_note_at_ref"] = note_check
    if not note_check["pass"]:
        errors.append(f"missing_release_note:{note_rel}")

    local_tag_sha = tag_sha(root, tag)
    tag_check = {
        "pass": local_tag_sha in (None, sha),
        "local_sha": local_tag_sha,
        "remote_sha": None,
    }
    if local_tag_sha == sha:
        skips.append("local_tag_already_exists_same_sha")
    elif local_tag_sha:
        errors.append("local_tag_exists_on_different_commit")
    if skip_remote_tag:
        tag_check["remote_skipped"] = True
        skips.append("remote_tag_check_skipped")
    else:
        remote_sha = remote_tag_sha(root, tag, remote=remote)
        tag_check["remote_sha"] = remote_sha
        if remote_sha == sha:
            skips.append("remote_tag_already_exists_same_sha")
        elif remote_sha:
            tag_check["pass"] = False
            errors.append("remote_tag_exists_on_different_commit")
    result["checks"]["tag_collision"] = tag_check

    if trunk:
        branch_ref = f"refs/heads/{trunk}"
        ancestor = is_ancestor(root, sha, branch_ref)
        trunk_check = {"pass": ancestor is not False, "trunk": trunk, "known": ancestor is not None}
        result["checks"]["trunk_reachability"] = trunk_check
        if ancestor is False:
            errors.append(f"ref_not_reachable_from_{trunk}")
        elif ancestor is None:
            warnings.append(f"local trunk {trunk} not found; trunk reachability not checked")

    if skip_dry_run:
        dry = {"pass": True, "skipped": True}
        skips.append("release_dry_run_skipped")
    else:
        payload = load_release_dry_run(root, ref)
        dry = {
            "pass": bool(payload.get("ok")) and payload.get("exit_code") == 0,
            "trailer": payload.get("trailer"),
            "exit_code": payload.get("exit_code"),
            "payload": payload,
        }
        if not dry["pass"]:
            errors.append("release_dry_run_failed")
    result["checks"]["release_dry_run"] = dry

    if skip_ci:
        ci = {"pass": True, "verdict": "SKIPPED", "advisory": True}
        skips.append("ci_check_skipped")
    else:
        ci = ci_signal(root, sha, workflow=workflow, require_ci=require_ci, wait=wait_ci)
        if ci.get("advisory"):
            warnings.append(str(ci.get("reason") or ci.get("verdict")))
        if not ci.get("pass"):
            errors.append(f"ci_not_green:{ci.get('verdict')}")
    result["checks"]["ci"] = ci

    result["ok"] = not errors
    if dry.get("trailer"):
        result["tag_message"] = tag_message(tag, sha, str(dry["trailer"]))
    else:
        result["tag_message"] = tag_message(tag, sha, f"release-dry-run: skipped {sha[:12]}")
    return result


def execute_tag(root: Path, context: dict, *, push: bool = False, remote: str = "origin",
                ttl: int = 1800, use_lock: bool = True,
                lock_already_held: bool = False) -> dict:
    result = dict(context)
    result["dry_run"] = False
    result["tag_created"] = False
    result["tag_pushed"] = False
    if not context.get("ok"):
        return result

    tag = str(context["tag"])
    sha = str(context["sha"])
    lock_held = False
    lock_owner: str | None = None
    try:
        if lock_already_held:
            lock = verify_release_lock(root)
            result["release_lock"] = lock
            if not lock.get("held"):
                result["ok"] = False
                result.setdefault("errors", []).append("release_lock_verify_failed")
                return result
        elif use_lock:
            lock = acquire_release_lock(root, tag=tag, sha=sha, ttl=ttl)
            result["release_lock"] = lock
            if not lock.get("held") and not lock.get("skipped"):
                result["ok"] = False
                result.setdefault("errors", []).append("release_lock_acquire_failed")
                return result
            lock_held = bool(lock.get("held"))
            lock_data = lock.get("lock")
            if isinstance(lock_data, dict):
                lock_owner = str(lock_data.get("owner") or "") or None
            if lock.get("skipped"):
                result.setdefault("warnings", []).append(str(lock.get("reason") or "release lock skipped"))
        else:
            result["release_lock"] = {"ok": True, "held": False, "skipped": True, "reason": "disabled by caller"}
            result.setdefault("idempotent_skips", []).append("release_lock_skipped")

        existing = tag_sha(root, tag)
        if existing:
            if existing != sha:
                result["ok"] = False
                result.setdefault("errors", []).append("local_tag_exists_on_different_commit")
                return result
            result.setdefault("idempotent_skips", []).append("tag_create_skipped_existing_same_sha")
        else:
            code, out = git(root, ["tag", "-a", tag, sha, "-m", str(context.get("tag_message") or tag)])
            if code != 0:
                result["ok"] = False
                result.setdefault("errors", []).append("tag_create_failed:" + out.strip()[-300:])
                return result
            result["tag_created"] = True

        if push:
            code, out = git(root, ["push", remote, tag], timeout=300)
            if code != 0:
                result["ok"] = False
                result.setdefault("errors", []).append("tag_push_failed:" + out.strip()[-300:])
                return result
            result["tag_pushed"] = True
    finally:
        if lock_held:
            result["release_lock_release"] = release_release_lock(root, owner=lock_owner)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify and create a fleet vX.Y.Z release tag.")
    parser.add_argument("--version", default=None, help="target X.Y.Z; defaults to VERSION")
    parser.add_argument("--ref", default="HEAD", help="committed ref to tag")
    parser.add_argument("--workflow", default="ci.yml")
    parser.add_argument("--trunk", default="main", help="local trunk branch ref required to contain --ref")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--require-ci", action="store_true",
                        help="block if gh cannot confirm a green ci.yml run for --ref")
    parser.add_argument("--wait-ci", action="store_true", help="watch a pending GitHub Actions run before folding CI")
    parser.add_argument("--skip-ci", action="store_true", help="skip CI folding; intended for tests only")
    parser.add_argument("--skip-dry-run", action="store_true", help="skip release_dry_run; intended for tests only")
    parser.add_argument("--skip-remote-tag", action="store_true",
                        help="skip remote tag collision check; intended for tests only")
    parser.add_argument("--execute", action="store_true", help="create the local annotated tag")
    parser.add_argument("--push", action="store_true", help="push the tag after creating or confirming it locally")
    parser.add_argument("--ttl", type=int, default=1800, help="release lock TTL while tagging")
    parser.add_argument("--no-lock", action="store_true", help="skip release lock during --execute")
    parser.add_argument("--lock-already-held", action="store_true",
                        help="execute under a parent-held release lock; verifies ownership and does not release it")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    root = repo_root()
    try:
        context = build_context(
            root,
            version=args.version,
            ref=args.ref,
            require_ci=args.require_ci,
            skip_ci=args.skip_ci,
            wait_ci=args.wait_ci,
            skip_dry_run=args.skip_dry_run,
            workflow=args.workflow,
            trunk=args.trunk,
            remote=args.remote,
            skip_remote_tag=args.skip_remote_tag,
        )
        result = execute_tag(
            root,
            context,
            push=args.push,
            remote=args.remote,
            ttl=args.ttl,
            use_lock=not args.no_lock,
            lock_already_held=args.lock_already_held,
        ) if args.execute else context
    except Exception as exc:
        result = {"ok": False, "errors": [str(exc)]}

    if args.as_json:
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        verdict = "OK" if result.get("ok") else "REFUSED"
        mode = "tagged" if result.get("tag_created") else "dry-run"
        print(f"[release-tag] {verdict} {result.get('tag', '')} ({mode})")
        for warning in result.get("warnings") or []:
            print(f"  warning: {warning}")
        for skip in result.get("idempotent_skips") or []:
            print(f"  skip: {skip}")
        for error in result.get("errors") or []:
            print(f"  ERROR: {error}", file=sys.stderr)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
