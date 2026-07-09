#!/usr/bin/env python3
"""refactor_verify — prove a god-split / code-motion refactor dropped NO definition.

The ``/modularize`` skill splits a monolith by MOVING top-level declarations between
files in the SAME package — a semantic no-op in Go, because package (not file) is the
scope. ``tools/godsplit_plan.py`` PLANS the cut (the doc-comment-aware ranges + the
hazards); this tool VERIFIES the result. It answers the one question ``go build`` cannot:

    did any top-level declaration silently disappear?

``go build`` / ``go vet`` / ``go test`` catch a *referenced* symbol that went missing
(``undefined: X``) and a *duplicated* one (``X redeclared``). But a top-level decl that was
DROPPED and happens to be unreferenced in-module — dead-but-real, or an exported symbol
whose only consumers are out-of-tree or a JSON wire contract — compiles clean and vanishes
without a trace. THAT is the "god module thing then missing a definition" failure: an
*incomplete* split that still builds green. This tool closes that gap mechanically.

How: it folds every touched package's top-level declaration multiset BEFORE (a git ref,
default ``HEAD``) and AFTER (the working tree), and diffs them. A pure code-motion split is
declaration-set-preserving BY CONSTRUCTION, so the correct diff is EMPTY. Anything in the
diff is one of:

  * RELOCATED — a decl that left package ``P`` and reappeared in package ``Q``. A legit
                cross-package consolidation (the "single source of truth" refactor);
                informational, not a failure, unless ``--expect-motion`` asserts a pure
                in-package split.
  * DROPPED   — a decl that left a package and reappeared NOWHERE in the change. THIS is the
                missing definition. For a split it is always a bug; for a cleanup it must be
                an *intended* deletion the author confirms — never a silent casualty of a cut.

It also flags OVER-SPLIT: a new file carrying a single top-level decl is the file-per-function
anti-pattern the skill's anti-gaming laws forbid ("never create a file-per-function").

Decl extraction reuses ``godsplit_plan.plan`` (raw-string/comment-aware), so this tool
inherits the same tested, hazard-aware fold rather than standing up a second fragile parser.

Scope (honest boundaries — being complete about what "complete" means):
  * It is DECL-level: it proves no whole top-level declaration was dropped. It does NOT diff
    STRUCT FIELDS across a type-alias consolidation — a dropped field inside a struct that was
    replaced by ``type Local = pkg.Remote`` is out of scope for v1 (that needs ``go/types`` to
    resolve the aliased type cross-package). The alias itself keeps the local NAME, so the
    decl-level check stays quiet on it; field-drop is the documented follow-on.
  * ``var (`` / ``const (`` grouped blocks are counted as one ``(group)`` decl each (godsplit
    does not name their members), so a drop *inside* a group is not seen — only a whole group
    going missing is. Named ``var``/``const`` decls are covered exactly.

Read-only. Pure core (``verify``, driven by in-memory strings) + a git-I/O ``main``, mirroring
``godsplit_plan.py`` so the core is testable without a repo. It never edits, moves, or commits.

    python tools/refactor_verify.py                       # working tree vs HEAD, whole change
    python tools/refactor_verify.py --ref HEAD~1          # against an earlier ref
    python tools/refactor_verify.py cmd/fak/guard.go      # scope to touched paths
    python tools/refactor_verify.py --expect-motion       # strict: a pure in-package split
    python tools/refactor_verify.py --json                # machine payload
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
_HERE = Path(__file__).resolve().parent


def _load_godsplit() -> Any:
    """Load ``tools/godsplit_plan.py`` as a module (tools/ is not a package) so we reuse its
    tested, raw-string-aware decl fold instead of re-implementing a Go line parser."""
    spec = importlib.util.spec_from_file_location("godsplit_plan", _HERE / "godsplit_plan.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_GS = _load_godsplit()


def decls_of(text: str) -> list[tuple[str, str]]:
    """The ``(kind, name)`` identity of every top-level decl in one Go source, via
    ``godsplit_plan.plan``. Methods report ``("method", <name>)`` (receiver-unqualified —
    a same-named method pair is disambiguated by multiplicity in the Counter, which is
    enough to catch a drop); grouped ``var``/``const`` blocks report ``(kind, "(group)")``."""
    return [(d["kind"], d["name"]) for d in _GS.plan(text)["decls"]]


def package_decls(files: dict[str, str]) -> Counter:
    """The declaration MULTISET of a package: ``(kind, name) -> count`` folded across all its
    ``.go`` files. Package = directory in Go, so this is the scope that a pure code-motion
    split must preserve exactly."""
    c: Counter = Counter()
    for text in files.values():
        c.update(decls_of(text))
    return c


def verify(before: dict[str, dict[str, str]], after: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Pure fold. ``before``/``after`` map ``package-dir -> {filename: source-text}``. Returns
    the completeness report: ``dropped`` (missing definitions), ``relocated`` (moved across
    packages), ``oversplit`` (file-per-function). No I/O — the test drives it with strings."""
    pkgs = sorted(set(before) | set(after))
    bdec = {p: package_decls(before.get(p, {})) for p in pkgs}
    adec = {p: package_decls(after.get(p, {})) for p in pkgs}
    removed = {p: bdec[p] - adec[p] for p in pkgs}   # in before, fewer/none in after
    added = {p: adec[p] - bdec[p] for p in pkgs}     # in after, not in before

    # A grouped var(/const) block collapses to the identity (kind, "(group)"), which is not
    # unique — so a group removed from P and a group added in Q are NOT the same declaration,
    # and matching them would fabricate a bogus cross-package "relocation" to every package
    # that touched a group. This tool only makes claims about NAMED decls; groups are counted
    # and footnoted, never classified. (Naming group members is the documented follow-on.)
    grouped_skipped = sum(c for p in pkgs for (k, n), c in removed[p].items() if n == "(group)")

    dropped: list[dict[str, Any]] = []
    relocated: list[dict[str, Any]] = []
    for p in pkgs:
        for (kind, name), cnt in sorted(removed[p].items()):
            if name == "(group)":
                continue
            targets = [q for q in pkgs if q != p and added[q].get((kind, name), 0) > 0]
            rec = {"pkg": p, "kind": kind, "name": name, "count": cnt}
            if targets:
                rec["to"] = targets
                relocated.append(rec)
            else:
                dropped.append(rec)

    oversplit: list[dict[str, Any]] = []
    for p in pkgs:
        existing = set(before.get(p, {}))
        for fname, text in sorted(after.get(p, {}).items()):
            if fname not in existing and len(decls_of(text)) == 1:
                oversplit.append({"pkg": p, "file": f"{p}/{fname}"})

    return {
        "packages": pkgs,
        "dropped": dropped,
        "relocated": relocated,
        "oversplit": oversplit,
        "grouped_skipped": grouped_skipped,
    }


# --------------------------------------------------------------------------- git I/O


def _git(*args: str) -> subprocess.CompletedProcess:
    # Go source is UTF-8 by spec; decode git output as UTF-8 explicitly rather than the
    # platform default (cp1252 on Windows), which would choke on non-ASCII bytes, empty the
    # blob, and manufacture a FALSE "dropped" for that file.
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )


def _pkg_of(path: str) -> str:
    return str(Path(path).parent).replace("\\", "/")


def _touched_pkgs(ref: str, paths: list[str]) -> list[str]:
    """The set of package dirs a change touched: from explicit ``.go`` ``paths`` if given,
    else every ``.go`` file that differs from ``ref`` PLUS untracked ``.go`` files (a split's
    new concern files are untracked until committed, so ``git diff`` alone would miss them)."""
    if paths:
        files = [p for p in paths if p.endswith(".go")]
    else:
        changed = _git("diff", "--name-only", ref, "--", "*.go").stdout.splitlines()
        untracked = _git("ls-files", "--others", "--exclude-standard", "--", "*.go").stdout.splitlines()
        files = changed + untracked
    return sorted({_pkg_of(f) for f in files if f.strip()})


def _after_files(pkgdir: str) -> dict[str, str]:
    """Every top-level ``.go`` file of ``pkgdir`` in the WORKING TREE (non-recursive — a
    package is one directory)."""
    d = ROOT / pkgdir
    files: dict[str, str] = {}
    if d.is_dir():
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix == ".go":
                files[f.name] = f.read_text(encoding="utf-8", errors="replace")
    return files


def _before_files(ref: str, pkgdir: str) -> dict[str, str]:
    """Every top-level ``.go`` file of ``pkgdir`` at ``ref`` (``git ls-tree`` is non-recursive,
    so it lists exactly the package's own files, not sub-packages')."""
    listing = _git("ls-tree", "--name-only", ref, f"{pkgdir}/").stdout.splitlines()
    files: dict[str, str] = {}
    for path in listing:
        path = path.strip()
        if not path.endswith(".go"):
            continue
        blob = _git("show", f"{ref}:{path}")
        if blob.returncode == 0:
            files[path.rsplit("/", 1)[-1]] = blob.stdout
    return files


def _render(rep: dict[str, Any], expect_motion: bool) -> str:
    out: list[str] = []
    n = len(rep["packages"])
    out.append(f"refactor_verify: {n} package(s) touched")
    if rep["dropped"]:
        out.append("")
        out.append("DROPPED - a definition left a package and reappeared NOWHERE (missing definition):")
        for d in rep["dropped"]:
            mult = f" x{d['count']}" if d["count"] > 1 else ""
            out.append(f"  !! {d['pkg']}: {d['kind']} {d['name']}{mult}")
        out.append("  -> a pure code-motion split must preserve every decl. If this deletion is")
        out.append("    intentional, say so in the commit body; otherwise the split is INCOMPLETE.")
    if rep["relocated"]:
        out.append("")
        verb = "!! (expect-motion) RELOCATED" if expect_motion else "relocated (cross-package consolidation)"
        out.append(f"{verb} - a decl moved between packages:")
        for r in rep["relocated"]:
            out.append(f"  {r['pkg']}: {r['kind']} {r['name']}  ->  {', '.join(r['to'])}")
        if expect_motion:
            out.append("  -> --expect-motion asserts a PURE in-package split; a cross-package move fails it.")
    if rep["oversplit"]:
        out.append("")
        out.append("OVER-SPLIT - a new file carries a single decl (file-per-function anti-pattern):")
        for o in rep["oversplit"]:
            out.append(f"  ~ {o['file']}")
        out.append("  -> group related decls into one cohesive concern file; don't split per-function.")
    fail = bool(rep["dropped"]) or (expect_motion and bool(rep["relocated"]))
    if not fail and not rep["oversplit"]:
        out.append("clean - declaration set preserved; no definition dropped.")
    if rep.get("grouped_skipped"):
        out.append("")
        out.append(f"note: {rep['grouped_skipped']} grouped var/const block(s) not tracked "
                   "(members unnamed at this resolution).")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Prove a code-motion/split refactor dropped no top-level declaration (read-only)."
    )
    ap.add_argument("paths", nargs="*", help="scope to these .go paths (default: the whole change vs --ref)")
    ap.add_argument("--ref", default="HEAD", help="the BEFORE git ref to diff against (default: HEAD)")
    ap.add_argument("--expect-motion", action="store_true",
                    help="strict: assert a PURE in-package split — any cross-package RELOCATION also fails")
    ap.add_argument("--json", action="store_true", help="emit the machine-readable report")
    args = ap.parse_args(argv)

    if _git("rev-parse", "--verify", "--quiet", f"{args.ref}^{{commit}}").returncode != 0:
        print(f"refactor_verify: not a commit: {args.ref}", file=sys.stderr)
        return 2

    pkgs = _touched_pkgs(args.ref, args.paths)
    before = {p: _before_files(args.ref, p) for p in pkgs}
    after = {p: _after_files(p) for p in pkgs}
    rep = verify(before, after)

    if args.json:
        print(json.dumps(rep, indent=2))
    else:
        print(_render(rep, args.expect_motion))

    if rep["dropped"] or (args.expect_motion and rep["relocated"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
