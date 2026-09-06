#!/usr/bin/env python3
"""
upgrade-sglang.py -- rebase the in-tree CAMA SGLang tree onto a new upstream ref.

The in-tree tree `sglang-with-cama-connector/` is full upstream SGLang with CAMA
patches applied on top. Upgrading means: take a newer upstream, re-apply CAMA's
patches via 3-way merge, surface conflicts for a human, and (only on explicit
--apply) swap the new tree in.

Two phases, run as two separate commands:

  1. ANALYZE (default, safe, no repo changes):
         python upgrade-sglang.py v0.5.9
     - clones base (from UPSTREAM.txt) and target (v0.5.9), python/sglang only
     - 3-way merges every file in patch_manifest.json
         BASE   = upstream @ pinned base
         OURS   = in-tree patched file
         THEIRS = upstream @ target
     - writes merged results + MERGE_REPORT.md to a staging dir
     - prints a clean/conflict summary; CONFLICT files keep <<<<<<< markers
     - flags any patched file that was deleted/moved upstream (critical)

  2. APPLY (destructive, gated):
     - first, a human edits the CONFLICT files in the staging dir to resolve markers
     - then:
         python upgrade-sglang.py v0.5.9 --apply
     - refuses if any conflict markers remain in staging
     - builds a full new tree (full target checkout + merged patches + CAMA-added
       files preserved), atomically swaps it in (old tree -> *.old), rewrites
       UPSTREAM.txt, and regenerates cama-connector/patches/

HARD PREREQUISITE before committing an --apply result: run sglang.launch_server
against a cama-server on Linux. v0.5.7-era patch logic may call SGLang internals
that changed in the target; only a runtime smoke test proves the merge is sound.

All git operations force LF (core.autocrlf=false) -- a CRLF checkout makes the
merge see every line as conflicting.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cama_patchlib as cpl  # noqa: E402

SUBTREE = "python/sglang"
ADDED_PREFIXES = ("python/sglang/srt/mem_cache/storage/cama/",)
IGNORE_DIR_NAMES = {"__pycache__"}


# ── filesystem helpers ─────────────────────────────────────────────────
def rmtree(path: Path) -> None:
    """rmtree that survives read-only files and transient Windows filesystem locks."""
    if not path.exists():
        return

    def _onexc(func, p, exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except Exception:
            pass

    for attempt in range(5):
        try:
            shutil.rmtree(path, onexc=_onexc) if sys.version_info >= (3, 12) \
                else shutil.rmtree(path, onerror=lambda f, p, e: _onexc(f, p, e))
            return
        except (PermissionError, OSError):
            if attempt == 4:
                raise
            time.sleep(0.5)


def no_window_creationflags() -> int:
    """CREATE_NO_WINDOW on Windows to prevent console flashing in background tools."""
    return 0x08000000 if sys.platform == "win32" else 0


# ── git helpers ────────────────────────────────────────────────────────
def _git(*args, cwd=None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, check=True,
                          capture_output=True, text=True,
                          creationflags=no_window_creationflags())


def clone_sparse(repo: str, ref: str, dest: Path) -> None:
    """Shallow + sparse (python/sglang) clone, forced LF."""
    _git("-c", "core.autocrlf=false", "-c", "core.eol=lf",
         "clone", "--depth", "1", "--branch", ref,
         "--filter=blob:none", "--sparse", repo, str(dest))
    _git("sparse-checkout", "set", SUBTREE, cwd=str(dest))


def clone_full(repo: str, ref: str, dest: Path) -> None:
    """Shallow full-tree clone, forced LF (for --apply)."""
    _git("-c", "core.autocrlf=false", "-c", "core.eol=lf",
         "clone", "--depth", "1", "--branch", ref, repo, str(dest))


# ── 3-way merge ────────────────────────────────────────────────────────
def merge_one(base: Path, ours: Path, theirs: Path, out: Path, target_ref: str) -> int:
    """Write a 3-way merge of (base, ours, theirs) to `out`.

    Returns conflict count (0 = clean), or -1 if `theirs` is missing
    (file deleted/renamed upstream -- needs manual attention).
    """
    if not theirs.is_file():
        return -1
    # stage normalized LF copies so merge inputs can't disagree on EOL
    with tempfile.TemporaryDirectory() as td:
        tb, to, tt = Path(td) / "base", Path(td) / "ours", Path(td) / "theirs"
        tb.write_text(cpl.read_lf(base) if base.is_file() else "", encoding="utf-8")
        to.write_text(cpl.read_lf(ours), encoding="utf-8")
        tt.write_text(cpl.read_lf(theirs), encoding="utf-8")
        res = subprocess.run(
            ["git", "merge-file", "-p", "--diff3",
             "-L", "ours (CAMA in-tree)",
             "-L", "base (current upstream)",
             "-L", f"theirs (upstream {target_ref})",
             str(to), str(tb), str(tt)],
            capture_output=True, text=True,
            creationflags=no_window_creationflags(),
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(res.stdout, encoding="utf-8")
    # git merge-file exit code = number of conflicts (>=0), or <0 on error
    return res.returncode if res.returncode >= 0 else 0


def has_conflict_markers(path: Path) -> bool:
    return "<<<<<<<" in cpl.read_lf(path)


# ── analyze ────────────────────────────────────────────────────────────
def analyze(target_ref: str, staging: Path, clone_dir: Path | None) -> int:
    up = cpl.read_upstream()
    repo = up.get("repo", "https://github.com/sgl-project/sglang.git")
    base_ref = up.get("ref")
    if not base_ref:
        print("ERROR: UPSTREAM.txt has no base ref", file=sys.stderr)
        return 2

    staging.mkdir(parents=True, exist_ok=True)
    merged_dir = staging / "merged"
    rmtree(merged_dir)

    tmp = None
    if clone_dir:
        base_clone = clone_dir / "base"
        target_clone = clone_dir / "target"
    else:
        tmp = tempfile.TemporaryDirectory(prefix="cama-upgrade-")
        base_clone = Path(tmp.name) / "base"
        target_clone = Path(tmp.name) / "target"

    try:
        if not (base_clone / SUBTREE).is_dir():
            print(f"Cloning base {repo}@{base_ref} ...")
            clone_sparse(repo, base_ref, base_clone)
        if not (target_clone / SUBTREE).is_dir():
            print(f"Cloning target {repo}@{target_ref} ...")
            clone_sparse(repo, target_ref, target_clone)

        rows = []  # (rel, status, detail)
        for entry in cpl.patch_entries():
            rel = entry["path"]
            base_f = base_clone / rel
            ours_f = cpl.INTREE_SGLANG / rel
            theirs_f = target_clone / rel
            out_f = merged_dir / rel

            n = merge_one(base_f, ours_f, theirs_f, out_f, target_ref)
            if n < 0:
                rows.append((rel, "GONE", "deleted/renamed upstream -- patch manually"))
            elif n == 0:
                # did upstream even touch this file between base and target?
                churned = theirs_f.is_file() and cpl.files_differ(base_f, theirs_f)
                rows.append((rel, "CLEAN", "upstream changed it" if churned
                             else "upstream unchanged"))
            else:
                rows.append((rel, "CONFLICT", f"{n} conflict hunk(s)"))

        write_report(staging, base_ref, target_ref, repo, rows)
        print_summary(base_ref, target_ref, staging, rows)
        # exit nonzero if anything needs hands
        return 1 if any(s in ("CONFLICT", "GONE") for _, s, _ in rows) else 0
    finally:
        if tmp is not None:
            tmp.cleanup()


def write_report(staging: Path, base_ref, target_ref, repo, rows) -> None:
    lines = [
        f"# SGLang upgrade merge report: {base_ref} -> {target_ref}",
        "",
        f"- repo: {repo}",
        f"- generated: {_dt.date.today().isoformat()}",
        f"- staging: `{staging}`",
        "",
        "Merged files are under `merged/<path>`. CONFLICT files keep `<<<<<<<`",
        "markers -- resolve them in place, then run with `--apply`.",
        "",
        "| status | file | detail |",
        "|---|---|---|",
    ]
    for rel, status, detail in rows:
        lines.append(f"| {status} | `{rel}` | {detail} |")
    lines += [
        "",
        "## Next steps",
        "1. Open each CONFLICT file under `merged/` and resolve the markers.",
        "2. For any GONE file, find where upstream moved the logic and re-apply by hand.",
        "3. `python upgrade-sglang.py " + target_ref + " --apply` (atomic swap).",
        "4. **Linux smoke test** `sglang.launch_server ... --hicache-storage-backend cama`",
        "   against a running cama-server before committing.",
        "",
    ]
    (staging / "MERGE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def print_summary(base_ref, target_ref, staging, rows) -> None:
    counts: dict[str, int] = {}
    for _, s, _ in rows:
        counts[s] = counts.get(s, 0) + 1
    print(f"\n=== {base_ref} -> {target_ref} ===")
    for rel, status, detail in rows:
        print(f"  {status:9} {rel}   ({detail})")
    print("\n  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"\nReport:  {staging / 'MERGE_REPORT.md'}")
    print(f"Merged:  {staging / 'merged'}")
    if counts.get("CONFLICT") or counts.get("GONE"):
        print("\nResolve CONFLICT/GONE files in the staging dir, then re-run with --apply.")
    else:
        print("\nAll clean. Review merged/, then re-run with --apply.")


# ── apply ──────────────────────────────────────────────────────────────
def apply(target_ref: str, staging: Path, repo: str, force: bool) -> int:
    merged_dir = staging / "merged"
    if not merged_dir.is_dir():
        print(f"ERROR: no staging at {merged_dir}. Run the analyze phase first.",
              file=sys.stderr)
        return 2

    # 1. refuse on unresolved conflicts
    unresolved = [e["path"] for e in cpl.patch_entries()
                  if (merged_dir / e["path"]).is_file()
                  and has_conflict_markers(merged_dir / e["path"])]
    if unresolved and not force:
        print("ERROR: unresolved conflict markers remain in staging:", file=sys.stderr)
        for u in unresolved:
            print(f"  {u}", file=sys.stderr)
        print("Resolve them, or pass --force to override.", file=sys.stderr)
        return 2

    # 2. full target checkout (the new base tree)
    new_tree = staging / "new-tree"
    rmtree(new_tree)
    print(f"Cloning full target {repo}@{target_ref} ...")
    clone_full(repo, target_ref, new_tree)
    rmtree(new_tree / ".git")

    # 3. overlay merged patches
    for entry in cpl.patch_entries():
        rel = entry["path"]
        src = merged_dir / rel
        dst = new_tree / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))

    # 4. preserve CAMA-added files (module, tests) that aren't upstream
    base_clone = staging / "base"  # left behind if analyze ran with --clone-dir; else re-clone
    if not (base_clone / SUBTREE).is_dir():
        base_clone = staging / "_base_for_added"
        if not (base_clone / SUBTREE).is_dir():
            clone_sparse(repo, cpl.read_upstream()["ref"], base_clone)
    added = added_files(base_clone)
    for rel in added:
        src = cpl.INTREE_SGLANG / rel
        dst = new_tree / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
    # carry forward UPSTREAM.txt position; rewritten below
    print(f"Overlaid {len(list(cpl.patch_entries()))} patches + {len(added)} added files.")

    # 5. atomic swap
    old = cpl.INTREE_SGLANG.with_name(cpl.INTREE_SGLANG.name + ".old")
    rmtree(old)
    cpl.INTREE_SGLANG.rename(old)
    shutil.move(str(new_tree), str(cpl.INTREE_SGLANG))
    print(f"Swapped: old tree -> {old.name}")

    # 6. rewrite UPSTREAM.txt
    write_upstream(target_ref, repo)

    # 7. regenerate patches/
    subprocess.run([sys.executable,
                    str(cpl.CONNECTOR_DIR / "scripts" / "sync_patches.py")],
                   check=True,
                   creationflags=no_window_creationflags())

    print("\nDONE. Now: Linux smoke test before committing, then update")
    print("DIFF_STANDALONE_VS_SGLANG.md + CHANGELOG. The old tree is kept as")
    print(f"  {old}\nDelete it once the upgrade is verified.")
    return 0


def added_files(base_clone: Path) -> list[str]:
    """Files present in current in-tree but not upstream@base (CAMA additions)."""
    out = []
    root = cpl.INTREE_SGLANG / SUBTREE
    for p in root.rglob("*"):
        if not p.is_file() or any(part in IGNORE_DIR_NAMES for part in p.parts):
            continue
        rel = p.relative_to(cpl.INTREE_SGLANG).as_posix()
        if not (base_clone / rel).is_file():
            out.append(rel)
    return out


def write_upstream(target_ref: str, repo: str) -> None:
    # resolve target commit via ls-remote; prefer the dereferenced (^{}) commit
    # for annotated tags so we record the commit, not the tag object.
    try:
        ls = _git("ls-remote", repo, f"{target_ref}^{{}}").stdout.strip()
        if not ls:
            ls = _git("ls-remote", repo, target_ref).stdout.strip()
        commit = ls.split()[0] if ls else "unknown"
    except Exception:
        commit = "unknown"
    up = cpl.read_upstream()
    content = (
        "# Upstream SGLang base for this pre-packaged CAMA tree.\n"
        "# Maintained by upgrade-sglang.py; do not hand-edit.\n\n"
        f"ref={target_ref}\n"
        f"commit={commit}\n"
        f"repo={repo}\n"
        f"last_upgrade_by={up.get('last_upgrade_by', 'unknown')}\n"
        f"last_upgrade_at={_dt.date.today().isoformat()}\n"
    )
    cpl.UPSTREAM_TXT.write_text(content, encoding="utf-8")


# ── main ───────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target_ref", help="upstream ref to upgrade to (e.g. v0.5.9)")
    ap.add_argument("--apply", action="store_true",
                    help="destructive: build new tree and atomic-swap it in")
    ap.add_argument("--force", action="store_true",
                    help="with --apply: proceed despite unresolved conflict markers")
    ap.add_argument("--staging", type=Path, default=None,
                    help="staging dir (default: scratch/sglang-upgrade-<ref>/)")
    ap.add_argument("--clone-dir", type=Path, default=None,
                    help="reuse base/ and target/ clones under this dir")
    args = ap.parse_args()

    staging = args.staging or (cpl.REPO_ROOT / "scratch" / f"sglang-upgrade-{args.target_ref}")
    repo = cpl.read_upstream().get("repo", "https://github.com/sgl-project/sglang.git")

    if args.apply:
        return apply(args.target_ref, staging, repo, args.force)
    return analyze(args.target_ref, staging, args.clone_dir)


if __name__ == "__main__":
    raise SystemExit(main())
