#!/usr/bin/env python3
"""Checked tag-after-green helper for fleet releases.

This is the executable spine for the final `/release` step: inspect a committed
ref, require the release dry-run, fold the GitHub Actions signal for that exact
commit when available, and only then create the local annotated vX.Y.Z tag.

The root module zip budget is measured against the ref before any tag is cut:
a tree the Go proxy would refuse must not become a public release tag.

Default mode is read-only. Use `--execute` to create the local tag and `--push`
to push that tag. The helper never creates a release commit and never pushes the
trunk branch. `--module-zip-only` runs just the budget measurement, so the
release workflow can refuse before it pushes trunk.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
# First-parent window searched for the newest green cut base below the cut commit.
RELEASE_CANDIDATE_MAX_DEPTH = 25
Runner = Callable[[list[str]], tuple[int, str]]
# A bot-pushed cut commit (GITHUB_TOKEN) starts no push-triggered run, so the
# cadence dispatches ci-fast for that exact sha with intent=release. Such a run
# is admitted only when its aggregate provenance binds this sha and the run
# executed the same ci-fast workflow bytes as the tested commit.
CI_FAST_WORKFLOW_PATH = ".github/workflows/ci-fast.yml"
# Must equal the matrix size ci-fast.yml seals into its provenance; release_tag_test
# checks the two agree so a new split bucket cannot silently refuse every release run.
CI_FAST_GO_TEST_JOBS = 17
CI_FAST_REQUIRED_JOBS = ["build-vet-fast"] + [f"go-test-shard-{index}" for index in range(CI_FAST_GO_TEST_JOBS)]
RELEASE_CI_CLASS = "release"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
# `gh run watch` is bounded by the runner timeout; re-watch rather than fail a
# run that outlives one bound.
DEFAULT_WATCH_ROUNDS = 5
# A dispatched run is listed a few seconds after `gh workflow run` returns.
DEFAULT_APPEAR_POLLS = 12
APPEAR_POLL_SECONDS = 10
CI_BILLING_PATTERNS = (
    "payments have failed",
    "spending limit",
    "billing & plans",
)

MODULE_ZIP_BUDGET_FILE = "internal/architest/module_zip_budget_test.go"
MODULE_ZIP_BUDGET_NAMES = (
    "moduleZipProxyCapBytes",
    "moduleZipBudgetBytes",
    "moduleZipNonGoBudget",
    "moduleZipNonGoFileLimit",
)
MODULE_ZIP_CONST_RE = re.compile(
    r"^\s*(moduleZip\w+)\s*=\s*(\d+)(?:\s*<<\s*(\d+))?\s*$", re.MULTILINE,
)
# The Go proxy's external limit. Kept here, not read from the ref, so a commit
# cannot raise its own budget past it.
MODULE_ZIP_HARD_CAP_BYTES = 500 << 20
# Caps the per-file non-Go limit the same way the total cap bounds the ref.
MODULE_ZIP_FILE_HARD_CAP_BYTES = 5 << 20
MODULE_ZIP_ARCHIVE_TIMEOUT_S = 600
MODULE_ZIP_REMEDY = (
    "move evidence, receipts and vendored bulk out of the root module (its own "
    "go.mod or export-ignore), or commit a pointer (path, sha256, size) instead of the raw artifact"
)


class ModuleZipUnmeasurable(Exception):
    pass


def _mib(n: int) -> str:
    return f"{n / (1 << 20):.1f} MiB"


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


def fold_ci_runs(runs: Any, *, require_ci: bool) -> dict:
    if not isinstance(runs, list):
        return {
            "pass": False,
            "verdict": "INVALID_RESPONSE",
            "advisory": False,
            "reason": "GitHub Actions run response was not a JSON array",
        }
    # Manual diagnostics may run only a narrow command or test a different
    # checkout. Validate every event before filtering, so unreadable evidence
    # cannot hide a failed push behind an otherwise qualified green run.
    for row in runs:
        event = row.get("event") if isinstance(row, dict) else None
        if not isinstance(event, str) or re.fullmatch(r"[a-z][a-z0-9_]*", event) is None:
            return {
                "pass": False,
                "verdict": "INVALID_EVENT",
                "advisory": False,
                "reason": "GitHub Actions run has a missing or malformed event",
            }
    runs = [row for row in runs if row["event"] == "push"]
    if not runs:
        return {
            "pass": not require_ci,
            "verdict": "NO_SIGNAL",
            "advisory": not require_ci,
            "reason": "no push-triggered GitHub Actions run found for commit",
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


def verify_release_dispatch(sha: str, run_row: Any, provenance: Any, *, trunk: str,
                            workflow_blob_at_run: str | None,
                            workflow_blob_at_sha: str | None) -> str | None:
    """Return why a dispatched ci-fast run is not release evidence for sha, or None.

    The aggregate provenance is the ci-fast verdict job's own check that every
    required job tested one commit and tree with the release command set; this
    re-binds it to the GitHub-reported run so a branch-modified workflow or a
    run of another commit cannot stand in for the release commit.
    """
    if not isinstance(run_row, dict):
        return "run row is not an object"
    run_id = str(run_row.get("databaseId") or "")
    attempt = str(run_row.get("attempt") or "")
    head_sha = str(run_row.get("headSha") or "")
    if run_row.get("event") != "workflow_dispatch":
        return "run is not a workflow_dispatch run"
    if run_row.get("headBranch") != trunk:
        return f"run did not execute on trunk {trunk}"
    if (str(run_row.get("status") or "").lower() != "completed"
            or str(run_row.get("conclusion") or "").lower() != "success"):
        return "run did not complete successfully"
    if (not re.fullmatch(r"[1-9][0-9]*", run_id) or not re.fullmatch(r"[1-9][0-9]*", attempt)
            or not SHA_RE.match(head_sha)):
        return "run identity (id, attempt, head) is malformed"
    if not isinstance(provenance, dict):
        return "aggregate provenance missing or unreadable"
    expected = {
        "schema": "fak.ci-provenance.v1",
        "workflow": "ci-fast",
        "verified": True,
        "ci_class": RELEASE_CI_CLASS,
        "requested_intent": "release",
        "event": "workflow_dispatch",
        "ref": f"refs/heads/{trunk}",
        "tested_sha": sha,
        "represented_sha": head_sha,
        "run_id": run_id,
        "run_attempt": attempt,
        "required_jobs": CI_FAST_REQUIRED_JOBS,
    }
    for key, want in expected.items():
        if provenance.get(key) != want:
            return f"provenance {key}={provenance.get(key)!r}, want {want!r}"
    if not SHA_RE.match(str(provenance.get("tested_tree") or "")):
        return "provenance tested_tree is malformed"
    if not workflow_blob_at_run or not workflow_blob_at_sha:
        return f"cannot read {CI_FAST_WORKFLOW_PATH} at the run head and the tested commit"
    if workflow_blob_at_run != workflow_blob_at_sha:
        return f"{CI_FAST_WORKFLOW_PATH} at run head {head_sha[:12]} differs from the tested commit"
    return None


def blob_id(root: Path, rev: str, path: str, *, remote: str = "origin") -> str | None:
    for attempt in range(2):
        code, out = git(root, ["rev-parse", "--verify", "--quiet", f"{rev}:{path}"])
        if code == 0 and out.strip():
            return out.strip()
        if attempt == 0:
            git(root, ["fetch", "--quiet", remote, rev], timeout=300)
    return None


def download_provenance(run_row: dict, runner: Runner) -> Any:
    run_id, attempt = run_row.get("databaseId"), run_row.get("attempt")
    with tempfile.TemporaryDirectory(prefix="release-tag-provenance-") as tmp:
        code, _ = runner(["gh", "run", "download", str(run_id),
                          "-n", f"ci-fast-provenance-{run_id}-{attempt}", "-D", tmp])
        path = Path(tmp) / "ci-fast-provenance.json"
        if code != 0 or not path.is_file():
            return None
        return _json_value(path.read_text(encoding="utf-8", errors="replace"))


def release_dispatch_signal(root: Path, sha: str, *, workflow: str, trunk: str, require_ci: bool,
                            wait: bool = False, runner: Runner | None = None,
                            fetch_provenance: Callable[[dict], Any] | None = None,
                            blob_at: Callable[[str], str | None] | None = None,
                            sleep: Callable[[float], None] = time.sleep,
                            appear_polls: int = DEFAULT_APPEAR_POLLS,
                            watch_rounds: int = DEFAULT_WATCH_ROUNDS) -> dict:
    runner = runner or (lambda cmd: run(cmd, cwd=root, timeout=900))
    fetch_provenance = fetch_provenance or (lambda row: download_provenance(row, runner))
    blob_at = blob_at or (lambda rev: blob_id(root, rev, CI_FAST_WORKFLOW_PATH))
    # The run title only locates the candidate; admission rests on provenance.
    title = f"ci-fast release · {sha}"
    cmd = [
        "gh", "run", "list",
        "--workflow", workflow,
        "--event", "workflow_dispatch",
        "--branch", trunk,
        "--limit", "30",
        "--json", "databaseId,status,conclusion,url,createdAt,headSha,headBranch,event,attempt,displayTitle",
    ]

    def find() -> dict | None:
        code, out = runner(cmd)
        rows = _json_value(out) if code == 0 else None
        if not isinstance(rows, list):
            return None
        for row in rows:
            if isinstance(row, dict) and row.get("displayTitle") == title:
                return row
        return None

    run_row = find()
    polls = 0
    while run_row is None and wait and polls < appear_polls:
        sleep(APPEAR_POLL_SECONDS)
        polls += 1
        run_row = find()
    if run_row is None:
        return {
            "pass": not require_ci,
            "verdict": "NO_SIGNAL",
            "advisory": not require_ci,
            "evidence": "release-dispatch",
            "reason": f"no trunk {workflow} release dispatch found for commit {sha[:12]}",
        }
    rounds = 0
    while wait and str(run_row.get("status") or "").lower() != "completed" and rounds < watch_rounds:
        runner(["gh", "run", "watch", str(run_row.get("databaseId")), "--exit-status"])
        rounds += 1
        run_row = find() or run_row
    status = str(run_row.get("status") or "").lower()
    conclusion = str(run_row.get("conclusion") or "").lower()
    base = {"evidence": "release-dispatch", "run": run_row}
    if status != "completed":
        return {**base, "pass": not require_ci, "verdict": "PENDING", "advisory": not require_ci,
                "reason": f"{workflow} release dispatch status is {status or '(unknown)'}"}
    if conclusion != "success":
        return {**base, "pass": False, "verdict": "RED", "advisory": False,
                "reason": f"{workflow} release dispatch completed with conclusion {conclusion or '(none)'}"}
    provenance = fetch_provenance(run_row)
    head_sha = str(run_row.get("headSha") or "")
    refusal = verify_release_dispatch(
        sha, run_row, provenance, trunk=trunk,
        workflow_blob_at_run=blob_at(head_sha) if SHA_RE.match(head_sha) else None,
        workflow_blob_at_sha=blob_at(sha),
    )
    if refusal:
        return {**base, "pass": False, "verdict": "UNVERIFIED", "advisory": False,
                "provenance": provenance,
                "reason": f"release dispatch is not evidence for {sha[:12]}: {refusal}"}
    return {**base, "pass": True, "verdict": "GREEN", "advisory": False, "provenance": provenance,
            "reason": (f"{workflow} release dispatch tested exact commit {sha[:12]} "
                       f"(tree {provenance['tested_tree'][:12]})")}


def ci_signal(root: Path, sha: str, *, workflow: str, require_ci: bool,
              wait: bool = False, runner: Runner | None = None,
              release_dispatch: bool = False, trunk: str = "main",
              dispatch_signal: Callable[..., dict] | None = None) -> dict:
    runner = runner or (lambda cmd: run(cmd, cwd=root, timeout=900))
    cmd = [
        "gh", "run", "list",
        "--workflow", workflow,
        "--commit", sha,
        "--event", "push",
        "--limit", "10",
        "--json", "databaseId,status,conclusion,url,createdAt,headSha,event",
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
    folded = fold_ci_runs(runs, require_ci=require_ci)
    if release_dispatch and folded.get("verdict") == "NO_SIGNAL":
        # A push run is the normal signal; only a sha with none falls back to its
        # own release-intent dispatch, which must itself be green for this sha.
        signal = dispatch_signal or release_dispatch_signal
        dispatched = signal(root, sha, workflow=workflow, trunk=trunk, require_ci=require_ci,
                            wait=wait, runner=runner)
        dispatched["push_signal"] = folded
        folded = dispatched
    run_row = folded.get("run") or {}
    if (wait and folded.get("evidence") != "release-dispatch" and folded.get("verdict") == "PENDING"
            and run_row.get("databaseId")):
        watch_code, watch_out = runner(["gh", "run", "watch", str(run_row["databaseId"]), "--exit-status"])
        folded["watch_exit_code"] = watch_code
        folded["watch_tail"] = watch_out.strip()[-300:]
        code, out = runner(cmd)
        if code == 0:
            try:
                rerun_rows = json.loads(out)
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


def first_parent_chain(root: Path, head: str, depth: int) -> list[str]:
    code, out = git(root, ["rev-list", "--first-parent", f"--max-count={depth}", head])
    if code != 0:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


# Verdicts that let the candidate search look past a commit to an older one.
# Anything else (RED, billing, invalid evidence) blocks the search, so a
# tag never lands on an ancestor older than a known-bad newer run.
CANDIDATE_TOLERATED_VERDICTS = frozenset({"GREEN", "NO_SIGNAL", "PENDING"})


def select_green_candidate(root: Path, head: str, *, workflow: str, wait: bool = False,
                           depth: int = RELEASE_CANDIDATE_MAX_DEPTH) -> dict:
    """Pick the newest first-parent commit of head that has its own green push run.

    Coalesced trunk CI leaves superseded commits without a run, so the tag
    candidate is the first ancestor whose exact-sha push run concluded success.
    Newer commits that are still pending or have no run are tolerated; a
    newer red or unreadable run blocks the search.
    """
    scanned: list[dict[str, Any]] = []
    head_ci: dict | None = None
    for index, candidate in enumerate(first_parent_chain(root, head, depth)):
        ci = ci_signal(root, candidate, workflow=workflow, require_ci=True,
                       wait=wait and index == 0)
        if index == 0:
            head_ci = ci
        verdict = str(ci.get("verdict"))
        scanned.append({"sha": candidate, "verdict": verdict})
        if verdict == "GREEN":
            return {"sha": candidate, "ci": ci, "scanned": scanned,
                    "pass": True, "reason": "newest first-parent commit with a green push run"}
        if verdict not in CANDIDATE_TOLERATED_VERDICTS:
            return {"sha": None, "ci": ci, "scanned": scanned, "pass": False,
                    "reason": f"{verdict} at {candidate[:12]} blocks the candidate search"}
    # Nothing green in the window: report HEAD's own signal so the refusal
    # names the same verdict a non-gated check would.
    return {
        "sha": None,
        "ci": head_ci or {"pass": False, "verdict": "NO_SIGNAL", "advisory": False,
                           "reason": "HEAD has no first-parent history to search"},
        "scanned": scanned,
        "pass": False,
        "reason": f"no green push run within {depth} first-parent commits of HEAD",
    }


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


def module_zip_budgets(root: Path, sha: str) -> dict[str, int]:
    ok, text = show_text(root, sha, MODULE_ZIP_BUDGET_FILE)
    if not ok:
        raise ModuleZipUnmeasurable(f"{MODULE_ZIP_BUDGET_FILE} is missing at the ref")
    found = {name: int(base) << int(shift or 0)
             for name, base, shift in MODULE_ZIP_CONST_RE.findall(text)}
    missing = [name for name in MODULE_ZIP_BUDGET_NAMES if name not in found]
    if missing:
        raise ModuleZipUnmeasurable(
            "budget constants not found at the ref (each must be a bare `name = value` "
            "line with no trailing comment): " + ", ".join(missing)
        )
    return {name: found[name] for name in MODULE_ZIP_BUDGET_NAMES}


def module_zip_entries(root: Path, sha: str) -> list[tuple[str, int]]:
    # git archive applies export-ignore exactly as the module proxy's zip does,
    # so the measured set is the installable tree, not the raw tracked tree.
    # The watchdog bounds a hung git; stderr goes to a file so it cannot fill a pipe.
    entries: list[tuple[str, int]] = []
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(
            ["git", "archive", "--format=tar", sha],
            cwd=str(root), stdout=subprocess.PIPE, stderr=err,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        watchdog = threading.Timer(MODULE_ZIP_ARCHIVE_TIMEOUT_S, proc.kill)
        watchdog.start()
        try:
            try:
                with tarfile.open(fileobj=proc.stdout, mode="r|") as archive:
                    for member in archive:
                        if member.isfile():
                            entries.append((member.name, member.size))
                # Drain the record padding after the tar end marker, or git sees EPIPE.
                proc.stdout.read()
            except Exception as exc:
                proc.kill()
                proc.wait()
                raise ModuleZipUnmeasurable(f"git archive stream unreadable: {exc}") from exc
            returncode = proc.wait()
        finally:
            watchdog.cancel()
            proc.stdout.close()
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        err.seek(0)
        stderr = err.read().decode("utf-8", errors="replace")
    if returncode != 0:
        raise ModuleZipUnmeasurable(f"git archive failed: {stderr.strip()[-200:]}")
    return entries


def _inside_nested_module(path: str, nested: set[str]) -> bool:
    parts = path.split("/")[:-1]
    return any("/".join(parts[:i]) in nested for i in range(1, len(parts) + 1))


def module_zip_check(root: Path, sha: str) -> dict:
    try:
        budgets = module_zip_budgets(root, sha)
        entries = module_zip_entries(root, sha)
    except ModuleZipUnmeasurable as exc:
        return {
            "pass": False,
            "verdict": "UNMEASURABLE",
            "errors": [f"module_zip_unmeasurable: {exc}; refusing to tag without a measured module zip"],
        }

    paths = [name for name, _ in entries]
    if "go.mod" not in paths:
        return {
            "pass": False,
            "verdict": "UNMEASURABLE",
            "errors": ["module_zip_unmeasurable: no root go.mod in the archive; wrong ref or tree"],
        }
    # Mirrors the architest budget: a directory with its own go.mod is a nested
    # module and is not shipped in the root module zip.
    nested = {p[: -len("/go.mod")] for p in paths if p.endswith("/go.mod")}
    file_limit = min(budgets["moduleZipNonGoFileLimit"], MODULE_ZIP_FILE_HARD_CAP_BYTES)
    total = 0
    non_go = 0
    oversized: list[tuple[str, int]] = []
    for name, size in entries:
        if _inside_nested_module(name, nested):
            continue
        total += size
        if not name.endswith(".go"):
            non_go += size
            if size > file_limit:
                oversized.append((name, size))
    budget_total = min(budgets["moduleZipBudgetBytes"], MODULE_ZIP_HARD_CAP_BYTES)
    budget_non_go = min(budgets["moduleZipNonGoBudget"], MODULE_ZIP_HARD_CAP_BYTES)
    errors: list[str] = []
    if total > budget_total:
        errors.append(
            f"module_zip_over_budget: root module source tree is {_mib(total)}, over the "
            f"{_mib(budget_total)} budget (Go proxy cap "
            f"{_mib(MODULE_ZIP_HARD_CAP_BYTES)}); remedy: {MODULE_ZIP_REMEDY}"
        )
    if non_go > budget_non_go:
        errors.append(
            f"module_zip_non_go_over_budget: non-Go content is {_mib(non_go)}, over the "
            f"{_mib(budget_non_go)} budget; remedy: {MODULE_ZIP_REMEDY}"
        )
    for name, size in sorted(oversized, key=lambda item: -item[1])[:5]:
        errors.append(
            f"module_zip_file_over_limit: {name} is {_mib(size)}, over the "
            f"{_mib(file_limit)} per-file limit; remedy: {MODULE_ZIP_REMEDY}"
        )
    return {
        "pass": not errors,
        "verdict": "WITHIN_BUDGET" if not errors else "OVER_BUDGET",
        "total_bytes": total,
        "non_go_bytes": non_go,
        "budgets": {**budgets, "hard_cap_bytes": MODULE_ZIP_HARD_CAP_BYTES},
        "errors": errors,
    }


def module_zip_only(root: Path, ref: str) -> dict:
    sha = ref_sha(root, ref)
    if not sha:
        return {"ok": False, "ref": ref, "sha": None, "errors": ["unresolvable_ref"]}
    check = module_zip_check(root, sha)
    return {
        "ok": check["pass"],
        "ref": ref,
        "sha": sha,
        "checks": {"module_zip": check},
        "errors": check["errors"],
    }


def build_context(root: Path, *, version: str | None, ref: str,
                  require_ci: bool = False, skip_ci: bool = False,
                  wait_ci: bool = False, skip_dry_run: bool = False,
                  workflow: str = "ci.yml", trunk: str = "main",
                  remote: str = "origin", skip_remote_tag: bool = False,
                  release_dispatch: bool = False) -> dict:
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

    module_zip = module_zip_check(root, sha)
    result["checks"]["module_zip"] = module_zip
    errors.extend(module_zip["errors"])

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
        # The tagged sha is always the cut commit itself: it must carry its own
        # green run (push, or its exact-sha release dispatch) before tagging.
        ci = ci_signal(root, sha, workflow=workflow, require_ci=require_ci, wait=wait_ci,
                       release_dispatch=release_dispatch, trunk=trunk or "main")
        if ci.get("advisory"):
            warnings.append(str(ci.get("reason") or ci.get("verdict")))
        if not ci.get("pass"):
            errors.append(f"ci_not_green:{ci.get('verdict')}")
            if require_ci:
                # The newest green ancestor is reported as the cut base only; it
                # never becomes the tagged sha.
                base = select_green_candidate(root, f"{sha}^", workflow=workflow)
                result["cut_base"] = {"sha": base["sha"], "reason": base["reason"],
                                      "scanned": base["scanned"]}
                if base["sha"]:
                    warnings.append(f"cut base {base['sha'][:12]} is green; cut {sha[:12]} is not covered")
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
    parser.add_argument("--release-dispatch", action="store_true",
                        help="with no push run for --ref, accept a trunk release-intent dispatch of --workflow "
                             "whose verified provenance tested exactly --ref")
    parser.add_argument("--skip-remote-tag", action="store_true",
                        help="skip remote tag collision check; intended for tests only")
    parser.add_argument("--execute", action="store_true", help="create the local annotated tag")
    parser.add_argument("--push", action="store_true", help="push the tag after creating or confirming it locally")
    parser.add_argument("--ttl", type=int, default=1800, help="release lock TTL while tagging")
    parser.add_argument("--no-lock", action="store_true", help="skip release lock during --execute")
    parser.add_argument("--lock-already-held", action="store_true",
                        help="execute under a parent-held release lock; verifies ownership and does not release it")
    parser.add_argument("--module-zip-only", action="store_true",
                        help="measure the root module zip budget for --ref and stop; no tag, no CI")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    root = repo_root()
    try:
        if args.module_zip_only:
            result = module_zip_only(root, args.ref)
            if args.as_json:
                json.dump(result, sys.stdout, indent=2)
                sys.stdout.write("\n")
            else:
                print(f"[release-tag] {'OK' if result['ok'] else 'REFUSED'} module zip ({args.ref})")
                for error in result["errors"]:
                    print(f"  ERROR: {error}", file=sys.stderr)
            return 0 if result["ok"] else 1
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
            release_dispatch=args.release_dispatch,
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
