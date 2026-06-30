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
import os
import re
import subprocess
import sys
from pathlib import Path

SEMVER_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")
CC_RE = re.compile(r"^(?P<type>[a-zA-Z]+)(?:\((?P<scope>[^)]*)\))?(?P<bang>!)?:")
VERSION_SUBJECT_RE = re.compile(r"^v\d+\.\d+\.\d+:")
BREAKING_RE = re.compile(r"\bBREAKING[ -]CHANGE\b")

# The fast release-critical CI subset (issue #1374). release_decide normally
# gates CI_BASE_RED on the WHOLE ci.yml run, which only concludes once the slow
# `-race` job finishes — a real release-latency tax. ci-fast.yml runs the
# correctness subset (build + vet + `go test ./...`, no `-race`) and produces a
# fast decisive conclusion. When the payload carries a decisive `ci_fast`
# signal, we gate on THAT instead of the `-race`-inclusive `ci.yml`, so a green
# fast subset clears the cut even while `-race` is still in flight.
#
# FAIL-SAFE: if `ci_fast` is absent, unknown, or unreadable, we fall back to the
# old whole-`ci.yml` `ci_on_head` behavior, so a missing fast signal never cuts
# on a blind base — it can only ever speed up a cut we would already allow, never
# bypass a real red. The gated workflow name is configurable so an operator can
# point the signal at a renamed/forked fast workflow without editing this file.
FAST_CI_WORKFLOW = (os.environ.get("FAK_RELEASE_FAST_CI_WORKFLOW") or "ci-fast.yml").strip()
_DECISIVE_CI_STATES = {"green", "red"}

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

# The significance floor (issue #1389). With auto-cut default-on at a 2h cadence,
# the floor is the only thing between a docs-typo commit and a public release
# every 2h. A window whose substantive commits are ALL trivial (docs/chore/test/
# style/ci/build) is BELOW the floor and must not mint a release.
#
# TRIVIAL_TYPES are the Conventional-Commits types that, on their own, never
# justify a release. Any other type — feat/fix/perf/refactor/revert, OR a type
# we do not recognize — is treated as significant. The fail-safe is deliberate:
# when in doubt about a commit's significance we count it as significant, so the
# floor can only ever SUPPRESS a window we are confident is trivial, never a real
# release. (refactor is significant here: it touches shippable code.)
TRIVIAL_TYPES = {"docs", "chore", "test", "style", "ci", "build"}


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_flag(name: str, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    return default


def is_significant(subject: str, body: str = "") -> bool:
    """Fail-safe significance test for the auto-cut floor.

    True (significant, count it) unless the commit's Conventional-Commits type is
    a recognized TRIVIAL type. A breaking change is always significant; an
    unrecognized type is counted as significant (in-doubt -> significant) so the
    floor never suppresses a real release.
    """
    m = CC_RE.match(subject.strip())
    if (m and m.group("bang")) or (body and BREAKING_RE.search(body)):
        return True
    if not m:
        # No Conventional-Commits prefix at all: cannot prove it is trivial.
        return True
    return m.group("type").lower() not in TRIVIAL_TYPES


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
    significant = 0
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
        if is_significant(subject, body):
            significant += 1
    return {
        "substantive": substantive,
        # `significant` is the fail-safe floor count: every non-trivial commit,
        # plus any commit whose triviality we cannot prove (issue #1389).
        "significant": significant,
        "total": total,
        "kinds": kinds,
    }


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


def effective_ci(payload: dict) -> tuple[dict, str]:
    """Pick the CI signal release_decide gates on (issue #1374).

    Returns ``(ci, source)`` where ``ci`` is the dict whose ``status`` drives the
    CI_BASE_* blockers and ``source`` is ``"fast"`` or ``"whole"`` for
    diagnostics.

    The fast subset (ci-fast.yml, build + vet + non-race tests) is preferred ONLY
    when it carries a DECISIVE status (``green`` or ``red``) — that is the whole
    point: a green fast subset clears the cut without waiting on the slow `-race`
    tail in ci.yml. Fail-safe: if the fast signal is absent, ``unknown``,
    ``none``, or otherwise not decisive, fall back to the whole-``ci.yml``
    ``ci_on_head`` signal, so a missing/in-progress fast subset can never cut on a
    blind base.
    """
    whole = payload.get("ci_on_head") or {}
    fast = payload.get("ci_fast")
    if isinstance(fast, dict) and str(fast.get("status")) in _DECISIVE_CI_STATES:
        return fast, "fast"
    return whole, "whole"


def decide(payload: dict, *, min_substantive: int = 1, force: bool = False,
           require_ci_green: bool = False, significance_floor: bool = True) -> dict:
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

    files_ahead = bool(tag_drift.get("files_ahead_of_tag"))
    if not commits and not files_ahead:
        blockers.append("NOTHING_TO_SHIP")
    elif not force and sig["substantive"] < min_substantive and not files_ahead:
        blockers.append("BELOW_SIGNIFICANCE")
    # Significance floor (issue #1389): a window with commits but ZERO significant
    # ones (all docs/chore/test/style/ci/build, none breaking, none unrecognized)
    # is below the floor and must not mint a release on the 2h auto-cut cadence.
    # Fail-safe: any commit whose triviality we cannot prove counts as significant,
    # so this gate only fires when the whole window is provably trivial.
    elif (
        significance_floor
        and not force
        and commits
        and sig["significant"] == 0
        and not files_ahead
    ):
        blockers.append("BELOW_FLOOR")

    if version_files.get("drift") is True:
        blockers.append("VERSION_DRIFT")
    if tag_drift.get("source_behind_reachable_tag"):
        blockers.append("VERSION_BEHIND_REACHABLE_TAG")

    # Gate on the fast release-critical subset when it is decisive, else fall
    # back to the whole `-race`-inclusive ci.yml run (issue #1374). The
    # CI_RETRY_TO_GREEN guard only applies to the whole-ci signal: a fast-subset
    # retry does not carry the same "flaky -race / heavy suite" concern, and the
    # fast subset has no per-run attempt contract.
    ci_signal, ci_source = effective_ci(payload)
    ci_status = ci_signal.get("status")
    if ci_status == "red":
        blockers.append("CI_BASE_RED")
    elif ci_status == "none":
        blockers.append("CI_BASE_NONE")
    elif ci_status == "unknown" and require_ci_green:
        blockers.append("CI_STATE_UNKNOWN")
    elif ci_status == "green" and ci_source == "whole":
        attempt = _ci_attempt(ci_signal)
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
        reason = _blocker_reason(blockers[0], last_tag, sig, min_substantive,
                                 tag_drift, ci_source)

    return {
        "decision": "hold" if blockers else "release",
        "level": None if blockers else level,
        "next_version": None if blockers else next_version,
        "last_tag": last_tag,
        "latest_any_tag": latest_any_tag,
        "n_commits": len(commits),
        "substantive": sig["substantive"],
        "significant": sig["significant"],
        "themes": themes,
        "blockers": blockers,
        "warnings": warnings,
        "recover": recover and not blockers,
        # Which CI signal answered the CI_BASE_* gate: "fast" (ci-fast.yml subset,
        # decisive) or "whole" (the -race-inclusive ci.yml, the fail-safe). #1374.
        "ci_source": ci_source,
        "reason": reason,
    }


def _blocker_reason(blocker: str, last_tag: str | None, sig: dict,
                    min_substantive: int, tag_drift: dict,
                    ci_source: str = "whole") -> str:
    # Name the workflow whose conclusion actually decided the CI gate (#1374):
    # the fast subset (ci-fast.yml) when it was decisive, else the whole ci.yml.
    ci_label = f"{FAST_CI_WORKFLOW} (fast subset)" if ci_source == "fast" else "ci.yml"
    reasons = {
        "NOTHING_TO_SHIP": f"no commits since {last_tag or '(no reachable tag)'} and no tag recovery pending",
        "BELOW_SIGNIFICANCE": (
            f"{sig['substantive']} substantive commit(s) of {sig['total']} since "
            f"{last_tag or '(no reachable tag)'}; floor is {min_substantive}"
        ),
        "BELOW_FLOOR": (
            f"all {sig['total']} commit(s) since {last_tag or '(no reachable tag)'} are "
            f"trivial (docs/chore/test/style/ci/build); no significant change to ship. "
            f"kinds={sig['kinds']}. Force with --force or set "
            f"FAK_RELEASE_SIGNIFICANCE_FLOOR=0 to disable the floor."
        ),
        "VERSION_DRIFT": "version markers disagree",
        "VERSION_BEHIND_REACHABLE_TAG": tag_drift.get("reason") or "VERSION is behind the reachable release tag",
        "CI_BASE_RED": f"latest decisive main {ci_label} run is red",
        "CI_BASE_NONE": f"no decisive completed main {ci_label} run is available",
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
    # Defaults read from env so the CI auto-cut path can tune the floor without
    # editing the workflow: FAK_RELEASE_MIN_SUBSTANTIVE (int, default 1) and
    # FAK_RELEASE_SIGNIFICANCE_FLOOR (0/1, default on) — issue #1389.
    parser.add_argument("--min-substantive", type=int,
                        default=_env_int("FAK_RELEASE_MIN_SUBSTANTIVE", 1),
                        help="minimum substantive commits to clear the floor "
                             "(env FAK_RELEASE_MIN_SUBSTANTIVE)")
    parser.add_argument("--no-significance-floor", dest="significance_floor",
                        action="store_false",
                        default=_env_flag("FAK_RELEASE_SIGNIFICANCE_FLOOR", True),
                        help="disable the all-trivial significance floor that "
                             "holds docs/chore-only windows "
                             "(env FAK_RELEASE_SIGNIFICANCE_FLOOR=0)")
    parser.add_argument("--force", action="store_true",
                        help="bypass the substantive-commit floor and the "
                             "significance floor")
    parser.add_argument("--require-ci-green", action="store_true",
                        help="block when gh cannot confirm a green main ci.yml base")
    args = parser.parse_args(argv)

    try:
        verdict = decide(
            context_payload(repo_root(), args.limit_commits),
            min_substantive=args.min_substantive,
            force=args.force,
            require_ci_green=args.require_ci_green,
            significance_floor=args.significance_floor,
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
