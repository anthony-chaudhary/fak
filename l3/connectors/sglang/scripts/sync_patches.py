#!/usr/bin/env python3
"""
sync_patches.py -- regenerate cama-connector/patches/ from the in-tree tree.

The in-tree pre-packaged tree (sglang-with-cama-connector/) is the source of
truth for patch CONTENT. cama-connector/patches/ is a generated mirror that
ships in the standalone connector zip and is what deploy.py copies onto a fresh
SGLang checkout.

Run this whenever you edit a patched file in the in-tree tree, then commit both
the in-tree change and the regenerated patches/ together. The CI drift gate
(scripts/check_drift.py) fails the build if they ever diverge.

    python scripts/sync_patches.py            # write patches/ from in-tree
    python scripts/sync_patches.py --check    # report drift, write nothing (exit 1 if drift)
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cama_patchlib as cpl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="report drift without writing (exit 1 if any drift)")
    args = ap.parse_args()

    pmap = cpl.patch_map()
    cpl.PATCHES_DIR.mkdir(parents=True, exist_ok=True)

    drift, copied, missing = [], [], []
    for leaf, rel in sorted(pmap.items()):
        src = cpl.INTREE_SGLANG / rel          # source of truth
        dst = cpl.PATCHES_DIR / leaf            # generated mirror
        if not src.is_file():
            missing.append(rel)
            continue
        if cpl.files_differ(src, dst):
            drift.append(leaf)
            if not args.check:
                shutil.copy2(str(src), str(dst))
                copied.append(leaf)

    if missing:
        print("ERROR: manifest files absent from in-tree tree:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        return 2

    if args.check:
        if drift:
            print(f"DRIFT: {len(drift)} patch file(s) differ from in-tree:")
            for d in drift:
                print(f"  ~ {d}")
            print("\nRun: python scripts/sync_patches.py")
            return 1
        print(f"OK: all {len(pmap)} patch file(s) match the in-tree source of truth.")
        return 0

    if copied:
        print(f"Regenerated {len(copied)} patch file(s) from in-tree:")
        for c in copied:
            print(f"  + {c}")
    else:
        print(f"Up to date: all {len(pmap)} patch file(s) already match in-tree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
