#!/usr/bin/env python3
"""dos_sync — back up the DURABLE work-product in `.dos/` to a safe archive.

`.dos/` is the DOS kernel's per-clone runtime dir. It is gitignored everywhere
(`.dos/` and `**/.dos/`) and rightly so: most of it is regenerable machine-local
state — `metrics/observations.jsonl` (hook telemetry) and `streams/*.jsonl`
(per-session tool-call digests) are rewritten on every tool call by the live
`dos-mcp` processes. Committing that into git would bloat history with
regenerable noise and churn on every turn.

But a few files under `.dos/` are genuine WORK-PRODUCT, not runtime: top-level
markdown like `awesome-list-prs-DRAFT.md` (pending-approval PRs to third-party
repos) or `release-notes-*.md`. Those are authored once and LOST if `.dos/` is
wiped — there is no backup today. This tool mirrors exactly that durable subset
to an archive directory, keeping the runtime local while making the work-product
survivable.

Design (deliberately conservative — see the audit that produced it):

  * SOURCE is the `.dos/` dir ONLY, at DEPTH 1, and ONLY `*.md` files. The
    `metrics/` and `streams/` subtrees are never read. The tool REFUSES to run if
    pointed at a repo root (it must see a dir literally named `.dos`), so it can
    never sweep the repo's tracked top-level docs.
  * DESTINATION defaults to the private fleet repo's gitignored `.dos-archive/`
    (`../fleet/.dos-archive/fak/` relative to a `fak` clone). It is a BACKUP, not
    a promotion into version control — the default destination is itself
    gitignored, so this can never leak a `.dos` work-product into a tracked
    public commit. To instead track the archive (private repo only), the operator
    points `--to` at a tracked dir and commits it themselves; this tool NEVER runs
    git and NEVER commits.
  * CONCURRENCY-SAFE: every source file is re-stat'd after it is read; if its
    size or mtime changed (a live writer touched it), the copy is SKIPPED rather
    than written torn. Writes are atomic (temp file in the destination, then
    `os.replace`).
  * `pull` (archive -> `.dos/`) refuses to overwrite a newer local file unless
    `--force`, so a restore never clobbers fresher local work.

Verbs:
  status   show what WOULD sync each direction (no writes).
  push     copy durable `.dos/*.md` -> archive (skips unchanged + live-churning).
  pull     copy archive/*.md -> `.dos/` (refuses to overwrite newer-local; --force).

Exit: 0 = ok, 1 = a real error (bad source, refused root, write failure),
2 = nothing to do / could-not-resolve a path (fail-soft for callers).

This tool is intentionally git-free and side-effect-scoped to the archive +
`.dos/`; it prints the suggested `git commit` line for a tracked-archive setup
but never executes it.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

# Subtrees under .dos/ that are RUNTIME, never work-product. Never read/copied.
RUNTIME_SUBDIRS = ("metrics", "streams")

# Default archive location, relative to the .dos/ parent (the repo root). A `fak`
# clone sits beside its private `fleet` source; the archive lands in fleet's
# gitignored scratch so a backup can never become a public commit by accident.
DEFAULT_ARCHIVE_REL = os.path.join("..", "fleet", ".dos-archive", "fak")


def _eprint(*a):
    print(*a, file=sys.stderr)


def resolve_dos_dir(arg: str) -> str | None:
    """Resolve and VALIDATE the .dos source dir. Returns abspath or None.

    Refuses anything not literally named `.dos` — the single guard that stops the
    tool from ever walking a repo root and sweeping tracked top-level docs.
    """
    p = os.path.abspath(arg)
    if os.path.basename(p.rstrip(os.sep)) != ".dos":
        _eprint(f"refused: --dos-dir must be a directory named '.dos', got {p!r}")
        return None
    if not os.path.isdir(p):
        _eprint(f"error: .dos dir not found: {p}")
        return None
    return p


def durable_md(dos_dir: str) -> list[str]:
    """Basenames of the durable work-product: *.md directly under .dos/ (depth 1).

    Excludes everything in the RUNTIME_SUBDIRS subtrees (they hold no top-level
    .md anyway, but the depth-1 rule makes that structural, not incidental).
    """
    out = []
    for name in sorted(os.listdir(dos_dir)):
        full = os.path.join(dos_dir, name)
        if not os.path.isfile(full):
            continue  # depth-1 files only; subdirs (metrics/, streams/) skipped
        if name.lower().endswith(".md"):
            out.append(name)
    return out


def _safe_copy(src: str, dst: str) -> bool:
    """Copy src->dst atomically, SKIPPING if src changed under us mid-read.

    Returns True if written, False if skipped (live writer touched the source).
    """
    try:
        st0 = os.stat(src)
        with open(src, "rb") as f:
            data = f.read()
        st1 = os.stat(src)
    except OSError as e:
        _eprint(f"  skip {os.path.basename(src)}: read error ({e})")
        return False
    # Re-stat guard: a concurrent dos-mcp / authoring session may be writing.
    if (st0.st_size, st0.st_mtime_ns) != (st1.st_size, st1.st_mtime_ns):
        _eprint(f"  skip {os.path.basename(src)}: changed during read (live writer)")
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dst), prefix=".dossync.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, dst)  # atomic on the same filesystem
    except OSError as e:
        _eprint(f"  error writing {dst}: {e}")
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return False
    return True


def _newer(a: str, b: str) -> bool:
    """True if file a is strictly newer (mtime) than file b (b may be absent)."""
    if not os.path.exists(b):
        return True
    return os.stat(a).st_mtime_ns > os.stat(b).st_mtime_ns


def cmd_status(dos_dir: str, archive: str, as_json: bool) -> int:
    files = durable_md(dos_dir)
    rows = []
    for name in files:
        src = os.path.join(dos_dir, name)
        dst = os.path.join(archive, name)
        push_state = "new" if not os.path.exists(dst) else ("update" if _newer(src, dst) else "same")
        pull_state = "n/a" if not os.path.exists(dst) else ("would-overwrite-newer-local" if _newer(src, dst) else ("restore" if _newer(dst, src) else "same"))
        rows.append({"file": name, "push": push_state, "pull": pull_state})
    if as_json:
        print(json.dumps({"dos_dir": dos_dir, "archive": archive, "files": rows}, indent=2))
    else:
        print(f".dos work-product: {dos_dir}")
        print(f"archive:           {archive}")
        if not rows:
            print("  (no durable *.md under .dos/ — nothing to sync)")
        for r in rows:
            print(f"  {r['file']:42}  push:{r['push']:8}  pull:{r['pull']}")
    return 0 if rows else 2


def cmd_push(dos_dir: str, archive: str, dry: bool) -> int:
    files = durable_md(dos_dir)
    if not files:
        print("nothing to push (no durable *.md under .dos/)")
        return 2
    wrote = 0
    for name in files:
        src = os.path.join(dos_dir, name)
        dst = os.path.join(archive, name)
        if not _newer(src, dst):
            continue  # identical-or-older source; skip
        if dry:
            print(f"  would push {name} -> {dst}")
            wrote += 1
            continue
        if _safe_copy(src, dst):
            print(f"  pushed {name}")
            wrote += 1
    print(f"push: {wrote} file(s) {'would be ' if dry else ''}written to {archive}")
    if wrote and not dry:
        # Git-free by design: only SUGGEST the commit if the archive is tracked.
        print("  (if the archive is a tracked dir in the PRIVATE repo, commit it there:")
        print(f"     git -C {os.path.dirname(archive)} commit -s -- {archive} -m 'chore: archive .dos work-product')")
    return 0


def cmd_pull(dos_dir: str, archive: str, force: bool, dry: bool) -> int:
    if not os.path.isdir(archive):
        _eprint(f"nothing to pull: archive not found: {archive}")
        return 2
    names = [n for n in sorted(os.listdir(archive)) if n.lower().endswith(".md") and os.path.isfile(os.path.join(archive, n))]
    if not names:
        print("nothing to pull (no *.md in archive)")
        return 2
    wrote = 0
    for name in names:
        src = os.path.join(archive, name)
        dst = os.path.join(dos_dir, name)
        if os.path.exists(dst) and _newer(dst, src) and not force:
            _eprint(f"  refuse {name}: local is newer than archive (use --force to overwrite)")
            continue
        if dry:
            print(f"  would pull {name} -> {dst}")
            wrote += 1
            continue
        if _safe_copy(src, dst):
            print(f"  pulled {name}")
            wrote += 1
    print(f"pull: {wrote} file(s) {'would be ' if dry else ''}restored to {dos_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Back up durable .dos/ work-product to a safe archive.")
    ap.add_argument("verb", choices=["status", "push", "pull"])
    ap.add_argument("--root", default=".", help="repo root containing .dos/ (default: cwd)")
    ap.add_argument("--dos-dir", default=None, help="explicit .dos dir (default: <root>/.dos)")
    ap.add_argument("--to", "--archive", dest="archive", default=None,
                    help=f"archive dir (default: <root>/{DEFAULT_ARCHIVE_REL})")
    ap.add_argument("--force", action="store_true", help="pull: overwrite a newer-local file")
    ap.add_argument("--dry-run", action="store_true", help="show actions without writing")
    ap.add_argument("--json", action="store_true", help="status: machine-readable output")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.root)
    dos_arg = args.dos_dir if args.dos_dir else os.path.join(root, ".dos")
    dos_dir = resolve_dos_dir(dos_arg)
    if dos_dir is None:
        return 1
    archive = os.path.abspath(args.archive) if args.archive else os.path.abspath(os.path.join(root, DEFAULT_ARCHIVE_REL))

    if args.verb == "status":
        return cmd_status(dos_dir, archive, args.json)
    if args.verb == "push":
        return cmd_push(dos_dir, archive, args.dry_run)
    if args.verb == "pull":
        return cmd_pull(dos_dir, archive, args.force, args.dry_run)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
