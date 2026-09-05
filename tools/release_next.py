"""Track and synchronize the "next" version draft for fleet.

This helper powers the proactive "next" version concept:
between releases, all work in progress landed on trunk is tracked
in a living draft (`docs/releases/NEXT.md`). When an operator or agent
is ready to cut the next release, the release notes and impact summary
are already curated and validated, making release cuts fast and error-free.

Usage:
  python tools/release_next.py              # print human summary of next version status
  python tools/release_next.py --json       # emit machine-readable JSON (schema: fak-release-next/1)
  python tools/release_next.py --sync       # update docs/releases/NEXT.md with in-flight commits
  python tools/release_next.py --check      # exit 0 if NEXT.md is up-to-date, 1 if missing/drifted
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
from dispatch_worker import install_no_window_subprocess_defaults

install_no_window_subprocess_defaults(subprocess)

SCHEMA = "fak-release-next/1"
DEFAULT_DRAFT_PATH = "docs/releases/NEXT.md"
SEMVER_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")
VERSION_SUBJECT_RE = re.compile(r"^v\d+\.\d+\.\d+:")
NEXT_SYNC_SUBJECT_RE = re.compile(r"^(?:docs\(release\):\s*)?sync\s+NEXT\.md", re.IGNORECASE)
CC_RE = re.compile(r"^(?P<type>[a-zA-Z]+)(?:\((?P<scope>[^)]*)\))?(?P<bang>!)?:")
BREAKING_RE = re.compile(r"\bBREAKING[ -]CHANGE\b")
MERGE_COMMIT_RE = re.compile(r"^(?:sync trunk with origin/main|merge (?:branch|remote-tracking|origin/main))", re.IGNORECASE)


def extract_commit_scope(subject: str) -> str:
    cleaned_prefix = re.sub(r"^@\s*", "", subject.strip())
    m = CC_RE.match(cleaned_prefix)
    if m and m.group("scope"):
        return m.group("scope").strip()
    return ""


def is_pure_merge_commit(subject: str, scope: str = "", clean: str = "") -> bool:
    if scope:
        return False
    raw = re.sub(r"^@?\s*[a-zA-Z]+(?:\([^)]*\))?!?: *", "", subject.strip())
    return bool(
        MERGE_COMMIT_RE.search(subject.strip())
        or MERGE_COMMIT_RE.search(raw)
        or (clean and MERGE_COMMIT_RE.search(clean.strip()))
    )


def format_consolidated_bullet(clean: str, count: int, scopes: list[str]) -> str:
    if count == 1:
        return f"- {clean}."
    if scopes:
        if len(scopes) <= 6:
            return f"- {clean} ({', '.join(scopes)})."
        else:
            return f"- {clean} across {len(scopes)} packages ({', '.join(scopes[:6])}, ...)."
    else:
        return f"- {clean} ({count} occurrences)."


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
    except Exception as exc:
        return 1, str(exc)


def repo_root() -> Path:
    code, out = run(["git", "rev-parse", "--show-toplevel"])
    if code == 0 and out.strip():
        return Path(out.strip())
    return TOOLS_DIR.parent


def semver_tuple(v: str) -> tuple[int, int, int] | None:
    m = SEMVER_RE.match(v.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def next_version_from_base(base: tuple[int, int, int], level: str) -> str:
    major, minor, patch = base
    if level == "major":
        return f"{major + 1}.0.0"
    if level == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def last_reachable_tag(root: Path) -> str | None:
    code, out = run(["git", "describe", "--tags", "--abbrev=0", "--match=v[0-9]*"], cwd=root)
    tag = out.strip()
    if code == 0 and tag and SEMVER_RE.match(tag):
        return tag
    code, out = run(["git", "tag", "--list", "v[0-9]*", "--sort=-v:refname"], cwd=root)
    for line in out.splitlines():
        t = line.strip()
        if SEMVER_RE.match(t):
            return t
    return None


def commits_since(root: Path, tag: str | None, limit: int = 300) -> list[dict]:
    rng = f"{tag}..HEAD" if tag else "HEAD"
    cmd = [
        "git",
        "log",
        f"-n{limit}",
        "--no-merges",
        "--format=%H%x1f%s%x1f%b%x1e",
        rng,
    ]
    code, out = run(cmd, cwd=root)
    if code != 0 or not out:
        return []
    commits = []
    for record in out.split("\x1e"):
        record = record.strip()
        if not record:
            continue
        parts = record.split("\x1f")
        sha = parts[0].strip() if len(parts) > 0 else ""
        subject = parts[1].strip() if len(parts) > 1 else ""
        body = parts[2].strip() if len(parts) > 2 else ""
        if not sha or not subject or VERSION_SUBJECT_RE.match(subject) or NEXT_SYNC_SUBJECT_RE.match(subject):
            continue
        commits.append({
            "sha": sha,
            "short_sha": sha[:9],
            "subject": subject,
            "body": body,
        })
    return commits


def classify_commit(subject: str, body: str = "") -> str:
    """Return 'feat', 'fix', or 'other' for a commit subject."""
    if "BREAKING CHANGE" in body or BREAKING_RE.search(subject):
        return "feat"
    m = CC_RE.match(subject)
    if not m:
        return "other"
    if m.group("bang"):
        return "feat"
    kind = m.group("type").lower()
    if kind == "feat":
        return "feat"
    if kind == "fix":
        return "fix"
    return "other"


def compute_projected_level(commits: list[dict]) -> str:
    """Determine semver bump level: major, minor, or patch."""
    has_breaking = False
    has_feat = False
    has_fix = False
    for c in commits:
        subj = c.get("subject", "")
        body = c.get("body", "")
        if BREAKING_RE.search(subj) or BREAKING_RE.search(body):
            has_breaking = True
        m = CC_RE.match(subj)
        if not m:
            continue
        if m.group("bang"):
            has_breaking = True
        k = m.group("type").lower()
        if k == "feat":
            has_feat = True
        elif k == "fix":
            has_fix = True
    if has_breaking:
        return "major"
    if has_feat:
        return "minor"
    return "patch"


def clean_public_subject(subject: str) -> str:
    text = re.sub(r"^@\s+", "", subject.strip())
    text = re.sub(r"\s+\(fak [^)]+\)\s*$", "", text)
    text = re.sub(r"\s+#[0-9]+(?=\s*$)", "", text)
    text = re.sub(r"^[a-z]+(?:\([^)]+\))?!?: *", "", text)
    text = text.strip().rstrip(".")
    if not text:
        return ""
    return text[0].upper() + text[1:] + "."


def inspect_next_state(root: Path, draft_rel: str = DEFAULT_DRAFT_PATH, limit: int = 300) -> dict:
    tag = last_reachable_tag(root)
    tag_base = semver_tuple(tag or "v0.0.0") or (0, 0, 0)
    commits = commits_since(root, tag, limit=limit)
    level = compute_projected_level(commits) if commits else "minor"
    projected_version = next_version_from_base(tag_base, level)

    draft_file = root / draft_rel
    draft_exists = draft_file.is_file()
    draft_text = draft_file.read_text(encoding="utf-8", errors="replace") if draft_exists else ""

    untracked_commits = []
    tracked_commits = []

    for c in commits:
        subj = c.get("subject", "")
        clean = clean_public_subject(subj)
        short_sha = c.get("short_sha", "")
        scope = extract_commit_scope(subj)

        if is_pure_merge_commit(subj, scope, clean.rstrip(".")):
            tracked_commits.append(c)
            continue

        if draft_exists:
            if (clean and clean.rstrip(".") in draft_text) or (subj in draft_text):
                tracked_commits.append(c)
                continue
            if short_sha and short_sha in draft_text:
                tracked_commits.append(c)
                continue

        untracked_commits.append(c)

    in_sync = draft_exists and (len(untracked_commits) == 0)

    return {
        "schema": SCHEMA,
        "root": str(root),
        "base_tag": tag,
        "base_version": f"{tag_base[0]}.{tag_base[1]}.{tag_base[2]}",
        "projected_level": level,
        "projected_version": projected_version,
        "projected_tag": f"v{projected_version}",
        "commits_count": len(commits),
        "commits": commits,
        "draft_file": draft_rel,
        "draft_exists": draft_exists,
        "in_sync": in_sync,
        "tracked_count": len(tracked_commits),
        "untracked_count": len(untracked_commits),
        "untracked_commits": untracked_commits,
    }


def parse_existing_draft(text: str) -> dict[str, list[str]]:
    """Extract existing sections from NEXT.md to preserve human annotations."""
    sections: dict[str, list[str]] = {
        "What changed": [],
        "Reliability and correctness": [],
        "Engineering quality and evidence": [],
        "Upgrade and breaking changes": [],
    }
    current_sec = None
    for line in text.splitlines():
        trimmed = line.strip()
        if trimmed.startswith("## "):
            sec_name = trimmed[3:].strip()
            current_sec = sec_name
            if current_sec not in sections:
                sections[current_sec] = []
            continue
        if current_sec and trimmed.startswith("- "):
            item = trimmed[2:].strip()
            if item and not item.startswith("*(No ") and item not in sections[current_sec]:
                sections[current_sec].append(item)
    return sections


def _process_section_commits(section_commits: list[dict]) -> tuple[list[str], set[str]]:
    groups: dict[str, dict] = {}
    for c in section_commits:
        subj = str(c.get("subject", ""))
        scope = extract_commit_scope(subj)
        clean = clean_public_subject(subj).rstrip(".")
        if not clean:
            continue
        if is_pure_merge_commit(subj, scope, clean):
            continue
        if clean not in groups:
            groups[clean] = {"count": 0, "scopes": []}
        groups[clean]["count"] += 1
        if scope and scope not in groups[clean]["scopes"]:
            groups[clean]["scopes"].append(scope)

    bullets: list[str] = []
    seen: set[str] = set()
    for clean, data in groups.items():
        bullet = format_consolidated_bullet(clean, data["count"], data["scopes"])
        if bullet not in seen:
            seen.add(bullet)
            bullets.append(bullet)
    return bullets, set(groups.keys())


def render_next_draft(state: dict, existing_sections: dict[str, list[str]] | None = None) -> str:
    version = state["projected_version"]
    level = state["projected_level"]
    base_tag = state.get("base_tag") or "initial"
    commits = state.get("commits") or []

    feat_commits = []
    fix_commits = []
    other_commits = []

    for c in commits:
        subj = c.get("subject", "")
        k = classify_commit(subj, c.get("body", ""))
        if k == "feat":
            feat_commits.append(c)
        elif k == "fix":
            fix_commits.append(c)
        else:
            other_commits.append(c)

    feat_items, feat_cleans = _process_section_commits(feat_commits)
    fix_items, fix_cleans = _process_section_commits(fix_commits)
    other_items, other_cleans = _process_section_commits(other_commits)

    def merge_existing(bullets: list[str], cleans: set[str], existing: list[str]) -> list[str]:
        out = list(bullets)
        seen = {b.lstrip("- ").strip() for b in bullets}
        for raw_item in existing:
            item = raw_item.strip()
            if not item or item.startswith("*(No "):
                continue
            item_text = item[2:].strip() if item.startswith("- ") else item
            base = re.sub(r"\s+\([^)]*\)\.?$", "", item_text)
            base = re.sub(r"\s+across \d+ packages \([^)]*\)\.?$", "", base).strip()
            base_clean = base.rstrip(".")
            if base_clean in cleans or item_text.rstrip(".") in cleans:
                continue
            if is_pure_merge_commit(item_text, "", base_clean):
                continue
            if item_text not in seen:
                seen.add(item_text)
                out.append(f"- {item_text}")
        return out

    if existing_sections:
        feat_items = merge_existing(feat_items, feat_cleans, existing_sections.get("What changed", []))
        fix_items = merge_existing(fix_items, fix_cleans, existing_sections.get("Reliability and correctness", []))
        other_items = merge_existing(other_items, other_cleans, existing_sections.get("Engineering quality and evidence", []))

    lines = [
        f"# fak vNext (targeting v{version}): Work in Progress",
        "",
        f"This document tracks in-flight work on `main` targeting the upcoming `v{version}` release.",
        "It is updated as commits land so that release notes are maintained proactively rather than scrambled at cut time.",
        "",
        f"- **Projected version:** `{version}` (`{level}` bump)",
        f"- **Base release tag:** `{base_tag}`",
        f"- **Commits in flight:** {len(commits)}",
        "",
        "## What changed",
        "",
    ]
    if feat_items:
        for item in feat_items:
            lines.append(item if item.startswith("- ") else f"- {item}")
    else:
        lines.append("- *(No new user-visible features landed yet)*")
    lines.append("")

    lines.extend(["## Reliability and correctness", ""])
    if fix_items:
        for item in fix_items:
            lines.append(item if item.startswith("- ") else f"- {item}")
    else:
        lines.append("- *(No bug fixes landed yet)*")
    lines.append("")

    lines.extend(["## Engineering quality and evidence", ""])
    if other_items:
        for item in other_items:
            lines.append(item if item.startswith("- ") else f"- {item}")
    else:
        lines.append("- *(No maintenance/quality commits landed yet)*")
    lines.append("")

    upgrade_items = existing_sections.get("Upgrade and breaking changes", []) if existing_sections else []
    lines.extend(["## Upgrade and breaking changes", ""])
    if upgrade_items:
        for item in upgrade_items:
            lines.append(item if item.startswith("- ") else f"- {item}")
    else:
        lines.append("- No manual migration required unless specified above.")
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def sync_next(root: Path, draft_rel: str = DEFAULT_DRAFT_PATH, limit: int = 300) -> tuple[dict, bool]:
    state = inspect_next_state(root, draft_rel=draft_rel, limit=limit)
    draft_file = root / draft_rel
    existing_sections = None
    if draft_file.is_file():
        existing_text = draft_file.read_text(encoding="utf-8", errors="replace")
        existing_sections = parse_existing_draft(existing_text)
    rendered = render_next_draft(state, existing_sections=existing_sections)
    draft_file.parent.mkdir(parents=True, exist_ok=True)
    changed = False
    if not draft_file.is_file() or draft_file.read_text(encoding="utf-8", errors="replace") != rendered:
        draft_file.write_text(rendered, encoding="utf-8", newline="\n")
        changed = True
    new_state = inspect_next_state(root, draft_rel=draft_rel, limit=limit)
    new_state["written"] = True
    new_state["changed"] = changed
    return new_state, changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Track and synchronize the 'next' version draft (docs/releases/NEXT.md)"
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--sync", action="store_true", help="sync docs/releases/NEXT.md with in-flight commits")
    parser.add_argument("--check", action="store_true", help="check if NEXT.md exists and is up to date (exit 1 if not)")
    parser.add_argument("--file", default=DEFAULT_DRAFT_PATH, help=f"path to NEXT draft (default {DEFAULT_DRAFT_PATH})")
    parser.add_argument("--limit-commits", type=int, default=300, help="maximum commits to inspect")
    args = parser.parse_args(argv)

    root = repo_root()
    if args.sync:
        state, changed = sync_next(root, draft_rel=args.file, limit=args.limit_commits)
        if args.json:
            print(json.dumps(state, indent=2))
            return 0
        action = "Updated" if changed else "Already in sync:"
        print(f"{action} {state['draft_file']} (projected: v{state['projected_version']}, {state['commits_count']} commits since {state['base_tag']})")
        return 0

    state = inspect_next_state(root, draft_rel=args.file, limit=args.limit_commits)
    if args.json:
        print(json.dumps(state, indent=2))
    else:
        status_word = "in-sync" if state["in_sync"] else ("missing" if not state["draft_exists"] else "out-of-sync")
        print(f"Next version draft: {state['draft_file']} [{status_word}]")
        print(f"  Base tag:          {state['base_tag']}")
        print(f"  Projected version: v{state['projected_version']} ({state['projected_level']} bump)")
        print(f"  Commits in flight: {state['commits_count']}")
        print(f"  Tracked in draft:  {state['tracked_count']}/{state['commits_count']}")
        if state["untracked_count"] > 0:
            print(f"  Untracked commits ({state['untracked_count']}):")
            for c in state["untracked_commits"][:5]:
                print(f"    - {c['short_sha']}: {c['subject']}")
            if state["untracked_count"] > 5:
                print(f"    ... and {state['untracked_count'] - 5} more (run `python tools/release_next.py --sync` to update)")

    if args.check:
        if not state["in_sync"]:
            if not state["draft_exists"]:
                sys.stderr.write(f"release-next check: draft {state['draft_file']} does not exist (run with --sync)\n")
            else:
                sys.stderr.write(f"release-next check: {state['untracked_count']} commits since {state['base_tag']} not in {state['draft_file']} (run with --sync)\n")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
