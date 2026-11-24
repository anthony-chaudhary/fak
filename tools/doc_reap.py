#!/usr/bin/env python3
"""Rank stale, orphaned, low-quality docs and archive them without breaking links.

  rank     score every tracked doc under docs/ (top level by default) and list
           the reap candidates worst-first
  archive  git-mv a doc into docs/archive/, stamp the archive banner that
           docs/archive/README.md requires (status+date, replacement, scope,
           reason) and repoint every inbound link to the new location
  check    every docs/archive/*.md (except README.md) carries that banner

The rank score is a triage queue, not a verdict: a human or agent reads the
top rows and decides archive, refresh (regenerate) or keep. Index files
(INDEX.md, llms.txt, ...) are excluded from inbound counts because they list
everything and would hide orphans.

Exit: 0 ok, 1 check/archive refusal, 2 could not run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

INDEX_FILES = (
    "INDEX.md", "llms.txt", "llms-full.txt",
    "docs/index.md", "docs/extended-index.md",
)
NON_REFERENCE_PREFIXES = (
    "docs/archive/", "docs/_witnesses/", "docs/research/inventory/",
    "docs/nightrun/", "docs/generated/",
)
# Front doors are kept current by their own gates; never queue them for reaping.
PROTECTED = {
    "docs/index.md", "docs/FAQ.md", "docs/extended-index.md",
    "docs/glossary.md", "docs/architecture.md",
}
CODE_SUFFIXES = (".go", ".py", ".yml", ".yaml", ".ps1", ".sh", ".json", ".toml")
STAMP_RE = re.compile(r"<!--\s*[a-z0-9-]+:\s*(\d{4}-\d{2}-\d{2})\s*·\s*process:\s*(\S+)")
ARCHIVE_BANNER_RE = re.compile(
    r"^> \*\*Archived (\d{4}-\d{2}-\d{2})\*\* · reason: ([\w-]+) · replacement: (.+?) · scope: (.+)$",
    re.M,
)
REASONS = ("superseded", "retired", "completed", "rejected", "orphaned", "stale-snapshot")
LINK_RE = re.compile(r"(\]\()([^)\s#]+)((?:#[^)\s]*)?\))")


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    ).stdout


def repo_root() -> Path:
    return Path(git(Path.cwd(), "rev-parse", "--show-toplevel").strip())


def tracked_docs(root: Path, recursive: bool) -> list[str]:
    out = []
    for p in git(root, "ls-files", "docs").splitlines():
        if not p.endswith(".md") or p.startswith("docs/archive/"):
            continue
        if not recursive and p.count("/") != 1:
            continue
        out.append(p)
    return out


def last_commit_dates(root: Path) -> dict[str, str]:
    dates: dict[str, str] = {}
    cur = ""
    for line in git(root, "log", "--format=@%ad", "--date=short", "--name-only", "--", "docs").splitlines():
        if line.startswith("@"):
            cur = line[1:]
        elif line and line not in dates:
            dates[line] = cur
    return dates


def inbound_refs(root: Path) -> dict[str, dict[str, set[str]]]:
    """basename -> {"doc": referencing docs, "code": referencing code/config}."""
    excludes = [f":!{p}" for p in INDEX_FILES] + [f":!{p}" for p in NON_REFERENCE_PREFIXES] + [":!*.jsonl"]
    try:
        raw = git(root, "grep", "-I", "-o", "-E", r"[A-Za-z0-9_.-]+\.md", "--", ".", *excludes)
    except subprocess.CalledProcessError as e:
        raw = e.stdout or ""
    refs: dict[str, dict[str, set[str]]] = {}
    for line in raw.splitlines():
        src, _, name = line.rpartition(":")
        if not src or src.rsplit("/", 1)[-1] == name:
            continue
        kind = "code" if src.endswith(CODE_SUFFIXES) else "doc"
        refs.setdefault(name, {"doc": set(), "code": set()})[kind].add(src)
    return refs


def appeal_score(root: Path, rel: str, text: str) -> float | None:
    if str(root / "tools") not in sys.path:
        sys.path.insert(0, str(root / "tools"))
    try:
        import doc_appeal_scorecard as appeal  # noqa: PLC0415
        return float(appeal.score_doc(text, rel)["appeal_score"])
    except Exception:  # appeal scorer is optional signal, not a dependency
        return None


def rank(root: Path, recursive: bool, today: dt.date) -> list[dict]:
    dates = last_commit_dates(root)
    refs = inbound_refs(root)
    rows = []
    for rel in tracked_docs(root, recursive):
        if rel in PROTECTED:
            continue
        path = root / rel
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        name = rel.rsplit("/", 1)[-1]
        r = refs.get(name, {"doc": set(), "code": set()})
        doc_refs, code_refs = len(r["doc"]), len(r["code"])
        last = dates.get(rel, "")
        age = (today - dt.date.fromisoformat(last)).days if last else 0
        stamp = STAMP_RE.search(text)
        stamp_age = (today - dt.date.fromisoformat(stamp.group(1))).days if stamp else None
        appeal = appeal_score(root, rel, text)

        score = min(age / 180.0, 1.0) * 35
        score += {0: 25, 1: 15, 2: 8}.get(doc_refs + code_refs, 0)
        if appeal is not None:
            score += (100.0 - appeal) / 100.0 * 25
        if stamp_age is not None and stamp_age > 60:
            score += 15
        reasons = []
        if age >= 90:
            reasons.append(f"untouched {age}d")
        if doc_refs + code_refs == 0:
            reasons.append("orphan")
        if stamp_age is not None and stamp_age > 60:
            reasons.append(f"snapshot {stamp_age}d old (regen: {stamp.group(2)})")
        if appeal is not None and appeal < 70:
            reasons.append(f"appeal {appeal:.0f}")
        rows.append({
            "path": rel, "score": round(score, 1), "last_commit": last,
            "age_days": age, "doc_refs": doc_refs, "code_refs": code_refs,
            "appeal": None if appeal is None else round(appeal, 1),
            "generated_by": stamp.group(2) if stamp else None,
            "action": "regenerate" if stamp else ("repoint+archive" if code_refs else "archive"),
            "reasons": reasons,
        })
    rows.sort(key=lambda x: (-x["score"], x["path"]))
    return rows


def _rel(target: Path, start: Path) -> str:
    return os.path.relpath(target, start).replace(os.sep, "/")


def rewrite_links(text: str, file_dir: Path, root: Path, old: Path, new: Path) -> str:
    def sub(m: re.Match) -> str:
        target = m.group(2)
        if re.match(r"^[a-z]+:", target) or target.startswith("/"):
            return m.group(0)
        resolved = (file_dir / target).resolve()
        if resolved == old.resolve():
            return m.group(1) + _rel(new, file_dir) + m.group(3)
        return m.group(0)
    return LINK_RE.sub(sub, text)


def rebase_own_links(text: str, old_dir: Path, new_dir: Path) -> str:
    def sub(m: re.Match) -> str:
        target = m.group(2)
        if re.match(r"^[a-z]+:", target) or target.startswith("/"):
            return m.group(0)
        return m.group(1) + _rel((old_dir / target).resolve(), new_dir) + m.group(3)
    return LINK_RE.sub(sub, text)


def insert_banner(text: str, banner: str) -> str:
    nl = "\r\n" if text.startswith("---\r\n") else "\n"
    if text.startswith("---" + nl):
        fence = nl + "---" + nl
        end = text.find(fence, 3)
        if end != -1:
            cut = end + len(fence)
            return text[:cut] + nl + banner + nl + text[cut:]
    return banner + nl + nl + text


def archive(root: Path, rel: str, reason: str, successor: str, scope: str,
            today: dt.date, dry_run: bool) -> int:
    old = root / rel
    if not old.is_file():
        print(f"refuse: {rel} does not exist", file=sys.stderr)
        return 1
    if reason not in REASONS:
        print(f"refuse: reason must be one of {REASONS}", file=sys.stderr)
        return 1
    new = root / "docs" / "archive" / old.name
    if new.exists():
        print(f"refuse: {new.relative_to(root)} already exists", file=sys.stderr)
        return 1
    if successor != "none" and not (root / successor).exists():
        print(f"refuse: successor {successor} does not exist", file=sys.stderr)
        return 1
    repl = "none" if successor == "none" else f"[{successor}]({_rel(root / successor, new.parent)})"
    banner = (f"> **Archived {today.isoformat()}** · reason: {reason} · replacement: {repl}"
              f" · scope: {scope}")

    old_posix = rel
    new_posix = new.relative_to(root).as_posix()
    touched = []
    # Evidence records and generated files keep their original paths; generated
    # files are refreshed by their own generators (llms-full.txt: gen_llms_full.py).
    # Archived docs are repointed too so links between archived docs survive.
    frozen = [f":!{p}" for p in NON_REFERENCE_PREFIXES if p != "docs/archive/"] + [":!*.jsonl", ":!llms-full.txt"]
    try:
        hits = git(root, "grep", "-I", "-l", "-F", old.name, "--", ".", *frozen).splitlines()
    except subprocess.CalledProcessError:
        hits = []
    for src in hits:
        if src == rel:
            continue
        p = root / src
        text = p.read_text(encoding="utf-8", errors="replace")
        updated = rewrite_links(text, p.parent, root, old, new)
        updated = re.sub(r"(?<![\w./-])" + re.escape(old_posix) + r"(?![\w-])", new_posix, updated)
        if updated != text:
            touched.append(src)
            if not dry_run:
                p.write_text(updated, encoding="utf-8", newline="\n")

    body = rebase_own_links(old.read_text(encoding="utf-8", errors="replace"), old.parent, new.parent)
    body = insert_banner(body, banner)
    print(f"archive {rel} -> {new_posix} ({reason}); repointed {len(touched)} file(s)")
    for t in touched:
        print(f"  repoint {t}")
    if dry_run:
        return 0
    git(root, "mv", rel, new_posix)
    new.write_text(body, encoding="utf-8", newline="\n")
    return 0


def check(root: Path) -> int:
    bad = []
    for rel in git(root, "ls-files", "docs/archive").splitlines():
        if not rel.endswith(".md") or rel.endswith("/README.md"):
            continue
        text = (root / rel).read_text(encoding="utf-8", errors="replace")
        m = ARCHIVE_BANNER_RE.search(text)
        if not m or m.group(2) not in REASONS:
            bad.append(rel)
    for rel in bad:
        print(f"ARCHIVE_BANNER_MISSING {rel}: needs '> **Archived DATE** · reason: R · replacement: X · scope: S'")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rank", help="list reap candidates worst-first")
    r.add_argument("--top", type=int, default=30)
    r.add_argument("--recursive", action="store_true", help="include docs/ subdirectories")
    r.add_argument("--json", action="store_true")
    a = sub.add_parser("archive", help="move a doc into docs/archive/ and repoint links")
    a.add_argument("doc")
    a.add_argument("--reason", required=True, choices=REASONS)
    a.add_argument("--successor", required=True, help="repo-relative replacement path, or 'none'")
    a.add_argument("--scope", default="", help="what the historical record describes")
    a.add_argument("--dry-run", action="store_true")
    sub.add_parser("check", help="every archived doc carries the archive banner")
    for p in (r, a):
        p.add_argument("--today", default="", help="override today's date (YYYY-MM-DD)")
    args = ap.parse_args(argv)
    try:
        root = repo_root()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"could not run: {e}", file=sys.stderr)
        return 2
    today = dt.date.fromisoformat(args.today) if getattr(args, "today", "") else dt.date.today()

    if args.cmd == "check":
        return check(root)
    if args.cmd == "archive":
        dates = last_commit_dates(root)
        scope = args.scope or f"state of the repository as of {dates.get(args.doc, 'its last edit')}"
        return archive(root, args.doc, args.reason, args.successor, scope, today, args.dry_run)

    rows = rank(root, args.recursive, today)[: args.top]
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    print(f"{'score':>5}  {'last':10}  {'refs':>4}  {'appeal':>6}  {'action':15}  path — why")
    for x in rows:
        ap_s = "-" if x["appeal"] is None else f"{x['appeal']:.0f}"
        print(f"{x['score']:5.1f}  {x['last_commit']:10}  {x['doc_refs'] + x['code_refs']:>4}  {ap_s:>6}  "
              f"{x['action']:15}  {x['path']} — {', '.join(x['reasons']) or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
