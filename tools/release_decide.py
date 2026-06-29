#!/usr/bin/env python3
"""Read-only release decision for fleet.

This is the mechanical judgment layer in front of `/release`: it reads
`tools/release_context.py`, answers whether a release should be cut, picks the
semver level, and surfaces blockers/warnings as JSON. It never edits files,
commits, tags, or pushes.

Exit codes: 0 = release, 2 = hold, 1 = context/usage failure.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

SEMVER_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")
CC_RE = re.compile(r"^(?P<type>[a-zA-Z]+)(?:\((?P<scope>[^)]*)\))?(?P<bang>!)?:")
VERSION_SUBJECT_RE = re.compile(r"^v\d+\.\d+\.\d+:")
BREAKING_RE = re.compile(r"\bBREAKING[ -]CHANGE\b")

LEVELS = ("patch", "minor", "major")
TYPE_LEVEL = {
    "feat": "minor",
    "fix": "patch",
    "perf": "patch",
    "revert": "patch",
    "docs": "patch",
    "build": "patch",
    "chore": "patch",
    "ci": "patch",
    "refactor": "patch",
    "style": "patch",
    "test": "patch",
}
SUBSTANTIVE_TYPES = {"feat", "fix", "perf", "revert"}


def semver_tuple(s: str | None) -> tuple[int, int, int] | None:
    m = SEMVER_RE.match((s or "").strip())
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def classify_subject(subject: str, body: str = "") -> str:
    m = CC_RE.match(subject.strip())
    if (m and m.group("bang")) or (body and BREAKING_RE.search(body)):
        return "major"
    if m:
        return TYPE_LEVEL.get(m.group("type").lower(), "patch")
    return "patch"


def _max_level(levels: list[str]) -> str:
    best = 0
    for level in levels:
        if level in LEVELS:
            best = max(best, LEVELS.index(level))
    return LEVELS[best]


def decide_level(commits: list[dict]) -> tuple[str, list[str]]:
    levels: list[str] = []
    themes: list[str] = []
    seen: set[str] = set()
    for commit in commits:
        subject = str(commit.get("subject") or "").strip()
        if VERSION_SUBJECT_RE.match(subject):
            continue
        levels.append(classify_subject(subject, str(commit.get("body") or "")))
        m = CC_RE.match(subject)
        scope = (m.group("scope") or "").strip() if m else ""
        if scope and scope not in seen:
            seen.add(scope)
            themes.append(scope)
    return (_max_level(levels) if levels else "patch"), themes


def significance(commits: list[dict]) -> dict:
    substantive = 0
    total = 0
    kinds: dict[str, int] = {}
    for commit in commits:
        subject = str(commit.get("subject") or "").strip()
        if VERSION_SUBJECT_RE.match(subject):
            continue
        total += 1
        body = str(commit.get("body") or "")
        m = CC_RE.match(subject)
        kind = (m.group("type").lower() if m else "") or "(none)"
        kinds[kind] = kinds.get(kind, 0) + 1
        breaking = bool(m and m.group("bang")) or bool(BREAKING_RE.search(body))
        if breaking or kind in SUBSTANTIVE_TYPES:
            substantive += 1
    return {"substantive": substantive, "total": total, "kinds": kinds}


def next_version_from_base(base: tuple[int, int, int], level: str) -> str:
    major, minor, patch = base
    if level == "major":
        return f"{major + 1}.0.0"
    if level == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def highest_base(*versions: str | None) -> tuple[int, int, int]:
    parsed = [v for v in (semver_tuple(x) for x in versions) if v is not None]
    return max(parsed) if parsed else (0, 0, 0)


def _ci_attempt(ci_on_head: dict) -> int | None:
    latest = ci_on_head.get("latest_trunk_ci") or {}
    try:
        return int(latest.get("attempt"))
    except (TypeError, ValueError):
        return None


def decide(payload: dict, *, min_substantive: int = 1, force: bool = False,
           require_ci_green: bool = False) -> dict:
    commits = payload.get("commits_since_tag") or []
    level, themes = decide_level(commits)
    sig = significance(commits)
    blockers: list[str] = []
    warnings: list[str] = []

    tag_drift = payload.get("tag_drift") or {}
    version_files = payload.get("version_files") or {}
    last_tag = payload.get("last_tag")
    latest_any_tag = payload.get("latest_any_tag")
    source_version = version_files.get("version")

    if not commits and not tag_drift.get("files_ahead_of_tag"):
        blockers.append("NOTHING_TO_SHIP")
    elif not force and sig["substantive"] < min_substantive and not tag_drift.get("files_ahead_of_tag"):
        blockers.append("BELOW_SIGNIFICANCE")

    if version_files.get("drift") is True:
        blockers.append("VERSION_DRIFT")
    if tag_drift.get("source_behind_reachable_tag"):
        blockers.append("VERSION_BEHIND_REACHABLE_TAG")

    ci_on_head = payload.get("ci_on_head") or {}
    ci_status = ci_on_head.get("status")
    if ci_status == "red":
        blockers.append("CI_BASE_RED")
    elif ci_status == "none":
        blockers.append("CI_BASE_NONE")
    elif ci_status == "unknown" and require_ci_green:
        blockers.append("CI_STATE_UNKNOWN")
    elif ci_status == "green":
        attempt = _ci_attempt(ci_on_head)
        if attempt is not None and attempt > 1:
            blockers.append("CI_RETRY_TO_GREEN")

    workflows = payload.get("workflows_parse_ok") or {}
    if workflows.get("ok") is False:
        blockers.append("WORKFLOW_UNPARSEABLE")
    elif workflows.get("ok") is None:
        warnings.append(str(workflows.get("note") or "workflow parse check unavailable"))

    for tag in payload.get("unreachable_newer_tags") or []:
        warnings.append(f"newer semver tag {tag} is not reachable from HEAD")

    recover = bool(tag_drift.get("files_ahead_of_tag"))
    if recover:
        next_version = source_version
        reason = tag_drift.get("reason") or f"source version {source_version} has no matching tag"
    else:
        base = highest_base(last_tag, latest_any_tag, source_version)
        next_version = next_version_from_base(base, level)
        reason = (
            f"{len(commits)} commit(s) since {last_tag or '(no reachable tag)'} "
            f"({sig['substantive']} substantive); highest signal {level} -> {next_version}"
        )

    if blockers:
        reason = _blocker_reason(blockers[0], last_tag, sig, min_substantive, tag_drift)

    return {
        "decision": "hold" if blockers else "release",
        "level": None if blockers else level,
        "next_version": None if blockers else next_version,
        "last_tag": last_tag,
        "latest_any_tag": latest_any_tag,
        "n_commits": len(commits),
        "substantive": sig["substantive"],
        "themes": themes,
        "blockers": blockers,
        "warnings": warnings,
        "recover": recover and not blockers,
        "reason": reason,
    }


def _blocker_reason(blocker: str, last_tag: str | None, sig: dict,
                    min_substantive: int, tag_drift: dict) -> str:
    reasons = {
        "NOTHING_TO_SHIP": f"no commits since {last_tag or '(no reachable tag)'} and no tag recovery pending",
        "BELOW_SIGNIFICANCE": (
            f"{sig['substantive']} substantive commit(s) of {sig['total']} since "
            f"{last_tag or '(no reachable tag)'}; floor is {min_substantive}"
        ),
        "VERSION_DRIFT": "version markers disagree",
        "VERSION_BEHIND_REACHABLE_TAG": tag_drift.get("reason") or "VERSION is behind the reachable release tag",
        "CI_BASE_RED": "latest decisive main ci.yml run is red",
        "CI_BASE_NONE": "no decisive completed main ci.yml run is available",
        "CI_STATE_UNKNOWN": "CI state is unknown and --require-ci-green was set",
        "CI_RETRY_TO_GREEN": (
            "latest decisive main ci.yml run is green only after a retry; pause auto-cut "
            "with FAK_AUTO_RELEASE=0 or confirm a fresh green run before releasing"
        ),
        "WORKFLOW_UNPARSEABLE": "a GitHub workflow file is not parseable",
    }
    return reasons.get(blocker, blocker)


def repo_root() -> Path:
    try:
        top = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        top = ""
    return Path(top) if top else Path(__file__).resolve().parent.parent


def context_payload(root: Path, limit_commits: int) -> dict:
    proc = subprocess.run(
        [sys.executable, str(root / "tools" / "release_context.py"),
         "--no-previews", "--limit-commits", str(limit_commits)],
        cwd=str(root),
        text=True,
        encoding="utf-8",
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "").strip()[:500])
    return json.loads(proc.stdout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Decide whether fleet should cut a release now.")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--limit-commits", type=int, default=300)
    parser.add_argument("--min-substantive", type=int, default=1)
    parser.add_argument("--force", action="store_true",
                        help="bypass only the substantive-commit floor")
    parser.add_argument("--require-ci-green", action="store_true",
                        help="block when gh cannot confirm a green main ci.yml base")
    args = parser.parse_args(argv)

    try:
        verdict = decide(
            context_payload(repo_root(), args.limit_commits),
            min_substantive=args.min_substantive,
            force=args.force,
            require_ci_green=args.require_ci_green,
        )
    except Exception as exc:
        print(f"release-decide: could not read context: {exc}", file=sys.stderr)
        return 1

    if args.as_json:
        json.dump(verdict, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(f"release-decide: {verdict['decision'].upper()} - {verdict['reason']}")
        if verdict["warnings"]:
            print("  warnings: " + "; ".join(verdict["warnings"]))
    return 0 if verdict["decision"] == "release" else 2


if __name__ == "__main__":
    raise SystemExit(main())
