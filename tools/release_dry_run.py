#!/usr/bin/env python3
"""Tag-last release dry-run for fleet.

Adjudicates a committed ref, not the hot working tree. The script parses
workflow YAML at the ref and runs the release-substrate tests in an isolated
local clone, so sibling-agent edits in the main checkout and worktree-pruner
cadences cannot affect the verdict.

Exit 0 = safe to spend the tag name. Exit 1 = fix forward, commit, re-run, then
tag. This script never pushes, tags, or edits the main worktree.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RELEASE_SUBSTRATE_TESTS = (
    "tools/release_context_test.py",
    "tools/release_cut_test.py",
    "tools/release_manifest_test.py",
    "tools/release_dry_run_test.py",
    "tools/release_tag_test.py",
    "tools/release_publish_test.py",
    "tools/release_cadence_workflow_test.py",
    "tools/release_status_test.py",
    "tools/release_lock_test.py",
    "tools/safe_ff_sync_test.py",
    "tools/release_decide_test.py",
    "tools/stable_release_context_test.py",
    "tools/stable_release_promote_test.py",
)

FAST_RELEASE_SUBSTRATE_TESTS = (
    "tools/release_context_test.py",
    "tools/release_decide_test.py",
    "tools/release_cut_test.py",
    "tools/release_manifest_test.py",
    "tools/release_tag_test.py",
    "tools/release_publish_test.py",
    "tools/release_cadence_workflow_test.py",
    "tools/release_status_test.py",
    "tools/release_lock_test.py",
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


def parse_workflows_at_ref(root: Path, ref: str) -> dict:
    out: dict = {"ok": True, "files": {}, "note": None}
    code, listing = run(["git", "ls-tree", "-r", "--name-only", ref, "--", ".github/workflows"], cwd=root)
    if code != 0:
        return {"ok": False, "files": {}, "note": f"git ls-tree failed for {ref}"}
    paths = [p for p in listing.splitlines() if p.endswith((".yml", ".yaml"))]
    if not paths:
        out["note"] = "no workflow files at ref"
        return out
    try:
        import yaml  # type: ignore
    except Exception as exc:
        return {
            "ok": None,
            "files": {p: None for p in paths},
            "note": f"PyYAML unavailable; parse check skipped ({type(exc).__name__})",
        }
    for path in paths:
        code, text = run(["git", "show", f"{ref}:{path}"], cwd=root)
        if code != 0:
            out["files"][path] = "unreadable at ref"
            out["ok"] = False
            continue
        try:
            yaml.safe_load(text)
            out["files"][path] = None
        except Exception as exc:
            out["files"][path] = " ".join(str(exc).split())[:300] or type(exc).__name__
            out["ok"] = False
    return out


def run_release_tests_at_ref(root: Path, ref: str, *, keep_worktree: bool = False,
                             fast: bool = False) -> dict:
    tmp_parent = Path(tempfile.mkdtemp(prefix="fleet-release-dryrun-"))
    wt = tmp_parent / "wt"
    requested_tests = FAST_RELEASE_SUBSTRATE_TESTS if fast else RELEASE_SUBSTRATE_TESTS
    suite: dict = {
        "ran": False,
        "mode": "fast" if fast else "full",
        "worktree": str(wt),
        "tests": [],
        "missing": [],
        "failures": [],
    }
    try:
        code, sha = run(["git", "rev-parse", f"{ref}^{{commit}}"], cwd=root)
        if code != 0:
            suite["error"] = "could not resolve ref: " + sha.strip()[-300:]
            return suite
        sha = sha.strip()
        code, out = run(["git", "clone", "--quiet", "--no-hardlinks", "--no-checkout", str(root), str(wt)],
                        cwd=root, timeout=300)
        if code != 0:
            suite["error"] = "isolated clone failed: " + out.strip()[-300:]
            return suite
        code, out = run(["git", "checkout", "--detach", sha], cwd=wt, timeout=120)
        if code != 0:
            suite["error"] = "checkout failed: " + out.strip()[-300:]
            return suite
        present = [test for test in requested_tests if (wt / test).is_file()]
        missing = [test for test in requested_tests if not (wt / test).is_file()]
        suite["missing"] = missing
        if not present:
            suite["error"] = f"no {suite['mode']} release-substrate tests exist at ref"
            return suite
        suite["ran"] = True
        ok = True
        rows: list[dict] = []
        for test in present:
            code, out = run([sys.executable, test], cwd=wt, timeout=300)
            row = {
                "test": test,
                "exit_code": code,
                "tail": "\n".join(out.splitlines()[-8:]),
            }
            rows.append(row)
            if code != 0:
                ok = False
                suite["failures"].append(row)
                break
        suite["tests"] = rows
        suite["ok"] = ok
        return suite
    finally:
        if not keep_worktree:
            shutil.rmtree(tmp_parent, ignore_errors=True)


def dry_run(root: Path, ref: str, *, skip_tests: bool = False, fast: bool = False,
            keep_worktree: bool = False) -> dict:
    code, sha = run(["git", "rev-parse", f"{ref}^{{commit}}"], cwd=root)
    if code != 0:
        return {"ok": False, "ref": ref, "error": "unresolvable ref"}
    sha = sha.strip()
    workflows = parse_workflows_at_ref(root, ref)
    suite = {"ran": False, "ok": True, "skipped": True, "mode": "fast" if fast else "full"} if skip_tests else run_release_tests_at_ref(
        root, ref, keep_worktree=keep_worktree, fast=fast
    )
    ok = workflows.get("ok") is not False and bool(suite.get("ok"))
    return {
        "ok": ok,
        "ref": ref,
        "sha": sha,
        "mode": "fast" if fast else "full",
        "workflows": workflows,
        "suite": suite,
        "trailer": f"release-dry-run: {'pass' if ok else 'FAIL'} {sha[:12]} mode={'fast' if fast else 'full'}",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run fleet release dry-run against a committed ref.")
    parser.add_argument("ref", nargs="?", default="HEAD")
    parser.add_argument("--fast", action="store_true",
                        help="run the release-perturbation subset instead of the full release substrate")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--keep-worktree", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    verdict = dry_run(repo_root(), args.ref, skip_tests=args.skip_tests, fast=args.fast,
                      keep_worktree=args.keep_worktree)
    if args.as_json:
        json.dump(verdict, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(verdict.get("trailer", "release-dry-run: FAIL"))
        for path, error in (verdict.get("workflows") or {}).get("files", {}).items():
            if error:
                print(f"  workflow UNPARSEABLE: {path}: {error}")
        for failure in (verdict.get("suite") or {}).get("failures", []):
            print(f"  {failure.get('test')}: exit {failure.get('exit_code')}")
    return 0 if verdict.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
