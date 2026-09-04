#!/usr/bin/env python3
"""
find_cama_patches.py -- verify patch_manifest.json is COMPLETE.

Clones the pinned upstream base (sglang-with-cama-connector/UPSTREAM.txt) and
diffs the in-tree pre-packaged tree against it, line-ending-normalized. Every
SGLang file the in-tree tree modifies should be listed in patch_manifest.json.
This is the safety net that would have caught schedule_policy.py,
scheduler_metrics_mixin.py, hicache_storage.py and hiradix_cache.py back when the
old hardcoded PATCH_MAP silently omitted them.

Reports four buckets:
  TRACKED    differs from upstream AND in manifest        -> good
  UNTRACKED  differs from upstream but NOT in manifest    -> ERROR (add it)
  STALE      in manifest but does NOT differ from upstream-> WARN (upstream caught up?)
  ADDED      only in in-tree, not upstream (informational; the CAMA module etc.)

Exit code is non-zero if any UNTRACKED file is found.

    python scripts/find_cama_patches.py                 # clone base into a temp dir
    python scripts/find_cama_patches.py --base v0.5.7   # override the pinned ref
    python scripts/find_cama_patches.py --clone-dir DIR # reuse an existing upstream checkout
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cama_patchlib as cpl  # noqa: E402

# in-tree subtree we compare; ignore generated/added noise
SUBTREE = "python/sglang"
IGNORE_DIR_NAMES = {"__pycache__"}
# the CAMA module is an ADDED tree (deployed separately), never a "patch"
ADDED_PREFIXES = ("python/sglang/srt/mem_cache/storage/cama/",)


def clone_base(repo: str, ref: str, dest: Path) -> None:
    """Shallow + sparse clone of <repo>@<ref>, python/sglang only, forced LF.

    core.autocrlf=false + core.eol=lf are mandatory: a CRLF checkout makes every
    line look changed (see cama_patchlib docstring).
    """
    subprocess.run(
        ["git", "-c", "core.autocrlf=false", "-c", "core.eol=lf",
         "clone", "--depth", "1", "--branch", ref,
         "--filter=blob:none", "--sparse", repo, str(dest)],
        check=True,
    )
    subprocess.run(["git", "sparse-checkout", "set", SUBTREE],
                   cwd=str(dest), check=True)


def iter_intree_files() -> list[str]:
    """Relative (to sglang tree root) paths of every file under python/sglang."""
    root = cpl.INTREE_SGLANG / SUBTREE
    out = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in IGNORE_DIR_NAMES for part in p.parts):
            continue
        out.append(p.relative_to(cpl.INTREE_SGLANG).as_posix())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=None, help="upstream ref (default: from UPSTREAM.txt)")
    ap.add_argument("--clone-dir", default=None, type=Path,
                    help="reuse an existing upstream checkout instead of cloning")
    args = ap.parse_args()

    up = cpl.read_upstream()
    repo = up.get("repo", "https://github.com/sgl-project/sglang.git")
    ref = args.base or up.get("ref")
    if not ref:
        print("ERROR: no base ref (pass --base or populate UPSTREAM.txt)", file=sys.stderr)
        return 2

    manifest_paths = {e["path"] for e in cpl.patch_entries()}

    tmp = None
    if args.clone_dir:
        base_dir = args.clone_dir
    else:
        tmp = tempfile.TemporaryDirectory(prefix="cama-upstream-")
        base_dir = Path(tmp.name) / "sglang"
        print(f"Cloning {repo}@{ref} (sparse, LF) ...")
        clone_base(repo, ref, base_dir)

    try:
        tracked, untracked, added = [], [], []
        for rel in sorted(iter_intree_files()):
            if rel.startswith(ADDED_PREFIXES):
                continue  # CAMA module: added tree, not a patch
            upstream_file = base_dir / rel
            intree_file = cpl.INTREE_SGLANG / rel
            if not upstream_file.is_file():
                added.append(rel)
                continue
            if cpl.files_differ(upstream_file, intree_file):
                (tracked if rel in manifest_paths else untracked).append(rel)

        # manifest entries that no longer differ from upstream
        differ_set = set(tracked) | set(untracked)
        stale = sorted(p for p in manifest_paths if p not in differ_set)

        print(f"\nBase: {ref}   manifest: {len(manifest_paths)} patches\n")
        print(f"TRACKED   ({len(tracked)}): differ & in manifest")
        for p in tracked:
            print(f"  ok  {p}")
        if untracked:
            print(f"\nUNTRACKED ({len(untracked)}): differ but NOT in manifest  <-- ADD THESE")
            for p in untracked:
                print(f"  !!  {p}")
        if stale:
            print(f"\nSTALE     ({len(stale)}): in manifest but identical to upstream")
            for p in stale:
                print(f"  ?   {p}  (upstream caught up, or file reverted -- review)")
        if added:
            print(f"\nADDED     ({len(added)}): only in in-tree (informational)")
            for p in added:
                print(f"  +   {p}")

        if untracked:
            print(f"\nFAIL: {len(untracked)} untracked patch(es). Add them to patch_manifest.json.")
            return 1
        print("\nOK: manifest is complete -- every modified file is tracked.")
        return 0
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
