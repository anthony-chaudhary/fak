#!/usr/bin/env python3
"""Pre-digest git + version state for the /release skill — fleet's lean helper.

Emits the JSON contract documented in `.claude/skills/release/SKILL.md`
§ "Expected JSON shape", so the universal release skill can run without
re-issuing `git log` / `git status` / `git diff` / per-file Reads.

This is a deliberately lean fleet-native rewrite of the 880-line job helper
(`C:\\work\\job\\scripts\\release_context.py`): no dos-pin reconcile, no
dispatch-loop lease join, no fanout buckets, no phantom-YAML guard — fleet is a
research-doc cluster + a Go kernel under `fak/`, with ONE version marker
(`VERSION`) and git tags as the shipped history. Pure stdlib; read-only.

Usage:
  python tools/release_context.py
  python tools/release_context.py --limit-commits 40
  python tools/release_context.py --no-previews

Top-level JSON keys:
  head_sha, current_branch, last_tag, latest_any_tag, unreachable_newer_tags,
  last_tag_age_seconds: whole seconds since the last reachable tag landed, or null
                      (feeds release_decide's auto-cut min-interval debounce, #1389)
  commits_since_tag:  [{sha, subject, files: [{path, additions, deletions}]}, ...]
  files_touched_since_tag: [path, ...]
  clean_tree:         true iff `git status --porcelain` is empty
  modified:           tracked modified/staged paths
  untracked:          {scratch, release_drafts, tracked_docs, other}
  version_files:      {version, drift}   (drift=false — fleet has one marker)
  tag_drift:          source/tag topology, including source-ahead recovery and
                      side-branch newer tags
  workflows_parse_ok: workflow YAML parse result when PyYAML is available
  ci_on_head:         latest decisive ci.yml verdict on main, if gh can read it
  modified_diff_previews:  {path: "~30-line `git diff HEAD -- path` preview"}
  untracked_doc_previews:  {path: "docstring / first ~10 non-blank lines"}
  prior_release_style:     parsed front-matter + headings of the newest release note
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from dispatch_worker import install_no_window_subprocess_defaults
import sys
from pathlib import Path

install_no_window_subprocess_defaults(subprocess)

SEMVER_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")
RELEASE_FILE_RE = re.compile(r"^docs/releases/v(\d+\.\d+\.\d+)\.md$")
VERSION_CUT_DUE_COMMIT_THRESHOLD = 20

# Scratch that must never survive a release. `.dos/` and `*.exe` are already
# gitignored (so they don't appear as untracked); these catch the rest fleet
# tends to drop at the repo root.
SCRATCH_PATTERNS = [
    re.compile(r"\.exe$"),
    re.compile(r"\.zip$"),
    re.compile(r"^.*\.err$"),
    re.compile(r"^_?scratch[-_/].*"),
    re.compile(r"^pp_run\d+/"),
]
# Cluster docs (durable): any markdown, plus the data/tooling subtrees the
# INDEX tracks. Everything that isn't scratch / a release draft falls through
# to `other`, which the release skill also treats as in-scope.
TRACKED_DOC_RE = re.compile(r"\.md$|^experiments/|^visuals/|^tools/")
DEFAULT_BRANCH = "main"
_GH_TIMEOUT_SECONDS = 8
_DECISIVE_CI_CONCLUSIONS = {"success", "failure", "timed_out", "startup_failure"}
GENERATION_LABELS = {"gen/now", "gen/next", "gen/second-next", "gen/future"}


def run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    except subprocess.CalledProcessError as exc:
        return exc.output or ""


def run_status(cmd: list[str]) -> int:
    try:
        return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    except OSError:
        return 127


def generation_from_text(text: str) -> str | None:
    for raw in (text or "").splitlines():
        key, sep, value = raw.strip().partition(":")
        if sep != ":" or key.strip().lower() != "generation":
            continue
        label = value.strip().lower()
        if label in GENERATION_LABELS:
            return label
        if label in {"now", "next", "second-next", "future"}:
            return f"gen/{label}"
        return None
    return None


def commit_generation(sha: str) -> str | None:
    if not sha:
        return None
    return generation_from_text(run(["git", "show", "-s", "--format=%B", sha]))


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def semver_tuple(s: str) -> tuple[int, int, int] | None:
    m = SEMVER_RE.match(s)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def semver_tags() -> list[str]:
    return [line.strip() for line in run(["git", "tag", "--sort=-v:refname"]).splitlines()
            if SEMVER_RE.match(line.strip())]


def tag_reachable(tag: str, ref: str = "HEAD") -> bool:
    return run_status(["git", "merge-base", "--is-ancestor", f"{tag}^{{commit}}", ref]) == 0


def latest_tag(*, reachable_only: bool = True) -> str | None:
    for tag in semver_tags():
        if not reachable_only or tag_reachable(tag):
            return tag
    return None


def tag_age_seconds(tag: str | None) -> int | None:
    """Whole seconds since `tag` came to exist, or None when it cannot be dated.

    Feeds release_decide's auto-cut debounce (#1389): on the scheduled cadence we
    refuse to mint a fresh tag less than N hours after the previous one. We read
    `%(creatordate:unix)` — the tagger date for an annotated tag, the commit's
    committer date for a lightweight one — i.e. "when the last release tag landed".
    Falls back to the target commit's committer date, then None. None means "no
    debounce evidence", and the debounce fails OPEN (does not block) on None.
    """
    if not tag:
        return None
    created: int | None = None
    raw = run(["git", "for-each-ref", "--format=%(creatordate:unix)", f"refs/tags/{tag}"]).strip()
    for line in raw.splitlines():
        token = line.strip()
        if token.isdigit():
            created = int(token)
            break
    if created is None:
        raw = run(["git", "log", "-1", "--format=%ct", f"{tag}^{{commit}}"]).strip()
        for line in raw.splitlines():
            token = line.strip()
            if token.isdigit():
                created = int(token)
                break
    if created is None:
        return None
    age = int(time.time()) - created
    return age if age >= 0 else 0


def unreachable_newer_tags(reachable_tag: str | None) -> list[str]:
    base = semver_tuple(reachable_tag or "")
    out: list[str] = []
    for tag in semver_tags():
        t = semver_tuple(tag)
        if not t:
            continue
        if base is not None and t <= base:
            continue
        if not tag_reachable(tag):
            out.append(tag)
    return out


def commits_since(tag: str | None, limit: int) -> list[dict]:
    fmt = "COMMIT\x1f%h\x1f%s"
    rng = [f"{tag}..HEAD"] if tag else []
    raw = run(["git", "log", *rng, f"--pretty=format:{fmt}", "--numstat", f"-n{limit}"])
    out: list[dict] = []
    cur: dict | None = None
    for raw_line in raw.splitlines():
        line = raw_line.rstrip("\r")
        if not line:
            continue
        if line.startswith("COMMIT\x1f"):
            parts = line.split("\x1f", 2)
            if len(parts) >= 3:
                if cur is not None:
                    out.append(cur)
                cur = {"sha": parts[1], "subject": parts[2], "files": []}
            continue
        if cur is None:
            continue
        cols = line.split("\t")
        if len(cols) < 3:
            continue
        def _to_int(x: str) -> int | None:
            try:
                return int(x)
            except ValueError:
                return None
        cur["files"].append({"path": "\t".join(cols[2:]), "additions": _to_int(cols[0]), "deletions": _to_int(cols[1])})
    if cur is not None:
        out.append(cur)
    for commit in out:
        gen = commit_generation(str(commit.get("sha") or ""))
        if gen:
            commit["generation"] = gen
    return out


_DIFF_LINE_CAP, _DIFF_BYTE_CAP = 30, 2048
_DOC_BYTE_CAP, _DOC_LINE_CAP = 500, 10


def diff_preview(path: str) -> str | None:
    raw = run(["git", "diff", "HEAD", "--", path])
    if not raw or raw.startswith("Binary files"):
        return None
    text = "\n".join(raw.splitlines()[:_DIFF_LINE_CAP])
    return text[:_DIFF_BYTE_CAP] + "\n... (truncated)" if len(text) > _DIFF_BYTE_CAP else text


def doc_preview(root: Path, rel_path: str) -> str | None:
    p = root / rel_path
    try:
        if not p.exists() or p.is_dir():
            return None
        text = p.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError):
        return None
    if not text.strip():
        return None
    if rel_path.endswith(".py"):
        m = re.search(r'^\s*[ru]?"""(.+?)"""', text, re.DOTALL)
        if m:
            doc = m.group(1).strip()
            return doc[:_DOC_BYTE_CAP] + "..." if len(doc) > _DOC_BYTE_CAP else doc
    non_blank: list[str] = []
    for line in text.splitlines():
        if line.strip():
            non_blank.append(line.rstrip())
        if len(non_blank) >= _DOC_LINE_CAP:
            break
    out = "\n".join(non_blank)
    return (out[:_DOC_BYTE_CAP] + "...") if len(out) > _DOC_BYTE_CAP else (out or None)


def prior_release_style(root: Path) -> dict | None:
    rel_dir = root / "docs" / "releases"
    if not rel_dir.exists():
        return None
    latest: tuple[tuple[int, int, int], Path] | None = None
    for p in rel_dir.glob("v*.md"):
        t = semver_tuple(p.stem)
        if t and (latest is None or t > latest[0]):
            latest = (t, p)
    if latest is None:
        return None
    try:
        text = latest[1].read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    out: dict = {"file": latest[1].name}
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            for raw_line in text[3:end].strip().splitlines():
                line = raw_line.strip()
                for key in ("version", "date", "headline"):
                    if line.startswith(key + ":"):
                        out[key] = line.split(":", 1)[1].strip().strip('"').strip("'")
                if line.startswith("themes:"):
                    rest = line.split(":", 1)[1].strip()
                    if rest.startswith("[") and rest.endswith("]"):
                        out["themes"] = [t.strip().strip('"').strip("'") for t in rest[1:-1].split(",") if t.strip()]
    out["section_headings"] = [line[3:].strip() for line in text.splitlines() if line.startswith("## ")][:8]
    return out


def files_touched(tag: str | None) -> list[str]:
    raw = run(["git", "diff", "--name-only", f"{tag}..HEAD"]) if tag else run(["git", "ls-files"])
    return [line for line in raw.splitlines() if line.strip()]


def porcelain_status() -> list[tuple[str, str]]:
    raw = run(["git", "status", "--porcelain=v1", "-z"])
    if not raw:
        return []
    items = raw.split("\0")
    out: list[tuple[str, str]] = []
    i = 0
    while i < len(items):
        entry = items[i]
        if not entry:
            i += 1
            continue
        code, path = entry[:2], entry[3:]
        i += 2 if code[0] in ("R", "C") else 1
        out.append((code, path))
    return out


def classify_untracked(paths: list[str]) -> dict:
    buckets = {"scratch": [], "release_drafts": [], "tracked_docs": [], "other": []}
    for p in paths:
        pn = p.replace("\\", "/")
        if RELEASE_FILE_RE.match(pn):
            buckets["release_drafts"].append(pn)
        elif any(pat.search(pn) for pat in SCRATCH_PATTERNS):
            buckets["scratch"].append(pn)
        elif TRACKED_DOC_RE.search(pn):
            buckets["tracked_docs"].append(pn)
        else:
            buckets["other"].append(pn)
    return buckets


def read_version_files(root: Path) -> dict:
    p = root / "VERSION"
    version = p.read_text(encoding="utf-8").strip() if p.exists() else None
    # Single load-bearing marker → markers cannot disagree → drift is always false.
    return {"version": version, "drift": False}


def tag_drift(last_reachable_tag: str | None, latest_any: str | None,
              version_files: dict, commits: list[dict]) -> dict:
    source_version = version_files.get("version")
    source_t = semver_tuple(source_version or "")
    reachable_t = semver_tuple(last_reachable_tag or "")
    any_t = semver_tuple(latest_any or "")

    files_ahead = bool(source_t and (any_t is None or source_t > any_t))
    source_behind_latest = bool(source_t and any_t and source_t < any_t)
    source_behind_reachable = bool(source_t and reachable_t and source_t < reachable_t)
    source_matches_reachable = bool(source_t and reachable_t and source_t == reachable_t)
    commit_due = len(commits) >= VERSION_CUT_DUE_COMMIT_THRESHOLD
    cut_due = files_ahead or commit_due

    if files_ahead:
        reason = f"VERSION is {source_version}, ahead of latest tag {latest_any or '(none)'}; recover by tagging v{source_version}"
    elif source_behind_reachable:
        reason = f"VERSION is {source_version}, behind reachable tag {last_reachable_tag}; reconcile before releasing"
    elif commit_due:
        reason = f"{len(commits)} commits since {last_reachable_tag or '(no reachable tag)'}; version cut is due"
    elif source_behind_latest:
        reason = f"newer semver tag {latest_any} exists outside HEAD; next release must choose a non-colliding version"
    else:
        reason = "no cut due"

    return {
        "source_version": source_version,
        "last_reachable_tag": last_reachable_tag,
        "latest_any_tag": latest_any,
        "tag_version": (latest_any or "").lstrip("v") or None,
        "reachable_tag_version": (last_reachable_tag or "").lstrip("v") or None,
        "files_ahead_of_tag": files_ahead,
        "source_behind_latest_tag": source_behind_latest,
        "source_behind_reachable_tag": source_behind_reachable,
        "source_matches_reachable_tag": source_matches_reachable,
        "tag_missing_for_files": f"v{source_version}" if files_ahead else None,
        "cut_due": cut_due,
        "commit_threshold": VERSION_CUT_DUE_COMMIT_THRESHOLD,
        "commits_since_reachable_tag": len(commits),
        "reason": reason,
    }


def workflows_parse_ok(root: Path) -> dict:
    wf_dir = root / ".github" / "workflows"
    paths = sorted([*wf_dir.glob("*.yml"), *wf_dir.glob("*.yaml")])
    if not paths:
        return {"ok": True, "files": {}, "note": "no workflow files"}
    try:
        import yaml  # type: ignore
    except Exception as exc:
        return {
            "ok": None,
            "files": {str(p.relative_to(root)).replace("\\", "/"): None for p in paths},
            "note": f"PyYAML unavailable; parse check skipped ({type(exc).__name__})",
        }
    files: dict[str, str | None] = {}
    ok = True
    for path in paths:
        rel = str(path.relative_to(root)).replace("\\", "/")
        try:
            yaml.safe_load(path.read_text(encoding="utf-8"))
            files[rel] = None
        except Exception as exc:
            files[rel] = " ".join(str(exc).split())[:300] or type(exc).__name__
            ok = False
    return {"ok": ok, "files": files}


def _run_gh_json(args: list[str]) -> object | None:
    try:
        raw = subprocess.check_output(
            ["gh", *args],
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_GH_TIMEOUT_SECONDS,
        )
        return json.loads(raw)
    except Exception:
        return None


def fold_latest_trunk_ci(latest_trunk: object) -> tuple[str, dict | None, str | None]:
    if latest_trunk is None:
        return "unknown", None, "gh unavailable/offline - CI state not read"
    if not isinstance(latest_trunk, list):
        return "unknown", None, "gh returned an unexpected CI payload"

    indecisive = 0
    for row in latest_trunk:
        if not isinstance(row, dict):
            continue
        conclusion = str(row.get("conclusion") or "")
        if conclusion not in _DECISIVE_CI_CONCLUSIONS:
            indecisive += 1
            continue
        trunk_ci = {
            "conclusion": conclusion,
            "head_sha": (str(row.get("headSha") or "")[:7] or None),
            "updated_at": row.get("updatedAt"),
            "attempt": row.get("attempt"),
            "database_id": row.get("databaseId"),
            "url": row.get("url"),
            "indecisive_runs_since": indecisive,
        }
        if conclusion == "success":
            return "green", trunk_ci, None
        return "red", trunk_ci, (
            "latest decisive main ci.yml run is not green; a release cut on "
            "this base inherits that failure"
        )
    return "none", None, "no decisive completed ci.yml run on main in the last 30"


def ci_on_head(default_branch: str = DEFAULT_BRANCH) -> dict:
    head = run(["git", "rev-parse", "HEAD"]).strip()
    runs_on_head: list[dict] = []
    if head:
        got = _run_gh_json([
            "run", "list", "--workflow", "ci.yml", "--commit", head,
            "--limit", "10", "--json", "workflowName,status,conclusion,attempt,databaseId,url",
        ])
        if isinstance(got, list):
            runs_on_head = [
                {
                    "workflow": row.get("workflowName"),
                    "status": row.get("status"),
                    "conclusion": row.get("conclusion") or None,
                    "attempt": row.get("attempt"),
                    "database_id": row.get("databaseId"),
                    "url": row.get("url"),
                }
                for row in got if isinstance(row, dict)
            ]

    latest_trunk = _run_gh_json([
        "run", "list", "--workflow", "ci.yml", "--branch", default_branch,
        "--status", "completed", "--limit", "30",
        "--json", "conclusion,headSha,updatedAt,attempt,databaseId,url",
    ])
    status, latest, note = fold_latest_trunk_ci(latest_trunk)
    return {
        "status": status,
        "runs_on_head": runs_on_head,
        "latest_trunk_ci": latest,
        "note": note,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Pre-digest git+version state for /release (fleet).")
    ap.add_argument("--limit-commits", type=int, default=50)
    ap.add_argument("--no-diff-stat", action="store_true")
    ap.add_argument("--no-previews", action="store_true")
    args = ap.parse_args()

    root = repo_root()
    head_sha = run(["git", "rev-parse", "--short", "HEAD"]).strip() or None
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip()
    if branch == "HEAD":
        branch = ""

    tag = latest_tag(reachable_only=True)
    any_tag = latest_tag(reachable_only=False)
    commits = commits_since(tag, args.limit_commits)
    status = porcelain_status()
    modified = [p for code, p in status if code != "??"]
    untracked = classify_untracked([p for code, p in status if code == "??"])
    version_files = read_version_files(root)

    payload: dict = {
        "head_sha": head_sha,
        "current_branch": branch,
        "last_tag": tag,
        "last_tag_age_seconds": tag_age_seconds(tag),
        "latest_any_tag": any_tag,
        "unreachable_newer_tags": unreachable_newer_tags(tag),
        "commits_since_tag": commits,
        "files_touched_since_tag": [] if args.no_diff_stat else files_touched(tag),
        "clean_tree": len(status) == 0,
        "modified": modified,
        "untracked": untracked,
        "version_files": version_files,
        "tag_drift": tag_drift(tag, any_tag, version_files, commits),
        "workflows_parse_ok": workflows_parse_ok(root),
        "ci_on_head": ci_on_head(DEFAULT_BRANCH),
    }

    if not args.no_previews:
        payload["modified_diff_previews"] = {p: dp for p in modified if (dp := diff_preview(p))}
        doc_paths = list(untracked["tracked_docs"]) + [p for p in untracked["other"] if p.endswith(".py")]
        payload["untracked_doc_previews"] = {p: dp for p in doc_paths if (dp := doc_preview(root, p))}
        payload["prior_release_style"] = prior_release_style(root)

    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
