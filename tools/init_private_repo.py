#!/usr/bin/env python3
"""Scaffold a NEW minimal private repo for the hard cut (private = private-only + ref).

    python tools/init_private_repo.py                              # dry-run plan
    python tools/init_private_repo.py --apply --dest ../fleet-private   # create it
    python tools/init_private_repo.py --apply --dest ../fleet-private --git   # ...and git init

The HARD CUT (PLAN-hard-cut-private-public-2026-06-20.md) leaves a choice for the
private side: surgically trim the existing private repo, or start a CLEAN new
private repo that holds ONLY the private material + a pointer to public. This tool
does the latter -- usually the easier, lower-risk path (no big history rewrite).

What goes in the new private repo is NOT hand-listed -- it is exactly what the
scrubber already removes from the public copy, read from the SAME source of truth
(``DELETE_PATHS`` + ``DELETE_GLOBS`` in the private ``scrub_public_copy.py``). So
the new private repo and the public scrub can never drift: whatever the scrub
strips from public is what the private repo keeps. Plus two generated files:

  * ``scrub_needles.json`` -- the canonical, script-independent scan-instructions
    artifact that ``tools/pull_scan_needles.py`` pulls (so this minimal repo does
    not need to carry the whole scrub script);
  * ``README.md`` -- a pointer to the public repo explaining the split.

Dry-run by default: prints the plan (paths, counts, bytes) and writes nothing.
``--apply`` copies into ``--dest``. It NEVER pushes -- the operator names and
pushes the remote. Pure stdlib. Exit 0 ok, 2 precondition.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys

_CREATE_NO_WINDOW = 0x08000000


def _win_creationflags() -> int:
    return _CREATE_NO_WINDOW if os.name == "nt" else 0


HERE = os.path.dirname(os.path.abspath(__file__))
PUBLIC_ROOT = os.path.dirname(HERE)
PUBLIC_REMOTE_HINT = "https://github.com/anthony-chaudhary/fak"


def _default_source() -> str:
    return os.path.join(os.path.dirname(PUBLIC_ROOT), "fleet")


def _load_scrub(source_root: str):
    src = os.path.join(source_root, "tools", "scrub_public_copy.py")
    if not os.path.isfile(src):
        return None, f"private scrubber not found: {src}"
    spec = importlib.util.spec_from_file_location("src_scrub", src)
    if spec is None or spec.loader is None:
        return None, f"cannot load module spec from {src}"
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        return None, f"failed importing {src}: {exc}"
    return module, None


def _dir_bytes(path: str) -> int:
    total = 0
    for dp, _d, fns in os.walk(path):
        for fn in fns:
            try:
                total += os.path.getsize(os.path.join(dp, fn))
            except OSError:
                pass
    return total


def plan_private_paths(module, source_root: str):
    """Resolve the private-only paths (DELETE_PATHS + DELETE_GLOBS) that EXIST."""
    rels: list[str] = []
    for rel in getattr(module, "DELETE_PATHS", []) or []:
        if os.path.exists(os.path.join(source_root, rel)):
            rels.append(rel.replace("\\", "/"))
    expand = getattr(module, "expand_glob", None)
    for pat in getattr(module, "DELETE_GLOBS", []) or []:
        if not callable(expand):
            break
        for full in expand(source_root, pat):
            rel = os.path.relpath(full, source_root).replace(os.sep, "/")
            if rel not in rels:
                rels.append(rel)
    return sorted(set(rels))


def _readme(source_root: str) -> str:
    return (
        "# fleet (private)\n\n"
        "Private companion to the **public** canonical repo "
        f"([fleet-public]({PUBLIC_REMOTE_HINT})).\n\n"
        "As of the 2026-06-20 hard cut, the public repo is canonical for all public\n"
        "content (edited directly, gardened first-class). This repo holds ONLY the\n"
        "operator-private material the public scrub keeps out of public -- see the\n"
        "PRIVATE-ONLY list in the public repo's `PUBLIC-SCRUB-POLICY.md` -- plus\n"
        "`scrub_needles.json`, the scan-instructions artifact the public repo's\n"
        "`tools/pull_scan_needles.py` pulls to run its leak scan in full mode.\n\n"
        "Do not copy content from public into here, or vice-versa, except through\n"
        "the documented scan-needle pull. This repo's contents were scaffolded from\n"
        f"`{source_root}` by `tools/init_private_repo.py`.\n"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--source", default=None,
                    help="existing private repo to scaffold FROM (default: sibling ../fleet)")
    ap.add_argument("--dest", default=None,
                    help="new private repo dir (required with --apply)")
    ap.add_argument("--apply", action="store_true", help="actually create the repo (default: dry-run)")
    ap.add_argument("--git", action="store_true", help="git init + initial commit in --dest")
    args = ap.parse_args(argv)

    source_root = os.path.abspath(args.source or _default_source())
    if not os.path.isdir(source_root):
        print(f"ERROR: source private repo not found: {source_root}", file=sys.stderr)
        return 2
    module, err = _load_scrub(source_root)
    if err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2

    rels = plan_private_paths(module, source_root)
    audit = list(getattr(module, "AUDIT_NEEDLES", []) or [])
    export = list(getattr(module, "EXPORT_AUDIT_NEEDLES", []) or [])

    total_bytes = 0
    rows = []
    for rel in rels:
        full = os.path.join(source_root, rel)
        if os.path.isdir(full):
            b = _dir_bytes(full)
            kind = "dir"
        else:
            b = os.path.getsize(full) if os.path.isfile(full) else 0
            kind = "file"
        total_bytes += b
        rows.append((rel, kind, b))

    print("=" * 72)
    print(f"NEW PRIVATE REPO PLAN  source={source_root}")
    print("=" * 72)
    print(f"private-only paths: {len(rows)}   (~{total_bytes/1e6:.1f} MB)")
    for rel, kind, b in rows:
        print(f"  [{kind:4}] {rel}  ({b/1e3:.0f} KB)")
    print(f"generated: scrub_needles.json ({len(audit)} audit / {len(export)} export needles), "
          f"README.md, .gitignore")

    if not args.apply:
        print("\nDRY-RUN. Re-run with --apply --dest <path> to create the repo.")
        return 0

    dest = os.path.abspath(args.dest or "")
    if not args.dest:
        print("ERROR: --apply requires --dest <path>", file=sys.stderr)
        return 2
    if os.path.abspath(dest) in (os.path.abspath(source_root), os.path.abspath(PUBLIC_ROOT)):
        print("ERROR: --dest must differ from the source and the public repo", file=sys.stderr)
        return 2
    os.makedirs(dest, exist_ok=True)

    copied = 0
    for rel in rels:
        src = os.path.join(source_root, rel)
        dst = os.path.join(dest, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif os.path.isfile(src):
            shutil.copy2(src, dst)
        copied += 1

    with open(os.path.join(dest, "scrub_needles.json"), "w", encoding="utf-8") as f:
        json.dump({"schema": "fleet-scrub-needles/1",
                   "audit_needles": audit, "export_audit_needles": export}, f, indent=2)
    with open(os.path.join(dest, "README.md"), "w", encoding="utf-8") as f:
        f.write(_readme(source_root))
    with open(os.path.join(dest, ".gitignore"), "w", encoding="utf-8") as f:
        f.write("tools/_registry/\n.dos/\n__pycache__/\n*.pyc\n")

    print(f"\nAPPLIED -> {dest}  ({copied} private-only path(s) copied)")
    if args.git:
        subprocess.run(["git", "-C", dest, "init", "-q"], check=False,
                       creationflags=_win_creationflags())
        subprocess.run(["git", "-C", dest, "add", "-A"], check=False,
                       creationflags=_win_creationflags())
        subprocess.run(["git", "-C", dest, "commit", "-q", "-m",
                        "Initial private-only commit (hard cut)"], check=False,
                       creationflags=_win_creationflags())
        print("  git: initialized + initial commit (no remote; push manually)")
    else:
        print("  (no --git; inspect, then `git init` + push to a PRIVATE remote yourself)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
