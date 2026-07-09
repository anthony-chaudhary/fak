#!/usr/bin/env python3
"""Diagnose *why* the release gate's CI subset is red: is it a forgotten-`git add`?

`tools/release_decide.py` gates the auto-cut on the `ci-fast` subset, whose first
step is `go build ./...` (see `.github/workflows/ci-fast.yml`). Under always-on-dev
the single commonest way trunk goes — and *stays* — red is a **coherence break**: a
commit lands a caller (or a modified file) that references a symbol whose definition
lives only in an UNCOMMITTED sibling file the author forgot to `git add`. The whole
tree builds on the author's disk, so nothing looks wrong locally; committed HEAD does
not build, and the gate correctly holds. But the gate's reason — "ci-fast is red" —
is opaque, so the freeze can sit for days (the observed v0.37.0 → HEAD lag) while
everyone assumes CI flakiness rather than a one-file omission.

This probe makes that break legible and fast to fix. It is READ-ONLY and never edits,
stages, commits, or pushes. It:

  1. builds a clean archive of *committed* HEAD — exactly what CI checks out, with no
     uncommitted files — and runs `go build ./...`;
  2. if the build fails, parses the compiler errors into (failing package, missing
     symbol) pairs;
  3. searches the working tree's UNCOMMITTED files (modified + untracked) for a
     definition of each missing symbol; a hit means "committed code references a
     symbol defined only in an uncommitted file" — i.e. a forgotten `git add`.

The verdict distinguishes a fixable coherence break (BUILD_BROKEN_COHERENCE, with the
exact forgotten files to `git add`) from a genuine compile error with no uncommitted
source to explain it (BUILD_BROKEN_OTHER). The error-parsing and forgotten-file
heuristics are pure functions so they unit-test without a Go toolchain; the build
itself is a thin wrapper and can be replaced by `--build-log FILE` to diagnose a
captured CI log offline.

Exit codes mirror release_decide: 0 = builds, 2 = broken, 1 = usage/probe failure.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:  # repo convention: keep Windows subprocesses window-less (best-effort)
    from dispatch_worker import install_no_window_subprocess_defaults

    install_no_window_subprocess_defaults(subprocess)
except Exception:  # pragma: no cover - the probe must run without the helper
    pass

# A Go build error block looks like:
#   # github.com/anthony-chaudhary/fak/cmd/fak
#   cmd\fak\main.go:52:16: undefined: parseVerbArgv
#   cmd\fak\guard.go:1094:21: lv.MetricsText undefined (type *logvault.Vault has no ...)
#   cmd\fak\slack_outbox.go:305:73: unknown field RetainDead in struct literal of ...
#   cmd\fak\guard.go:181:51: undefined: gateway.DefaultVCacheAnchor
_PKG_RE = re.compile(r"^#\s+(\S+)")
# Every shape the compiler uses to say "this name has no definition". Each capture
# group is the bare identifier we then hunt for in the uncommitted files.
_SYMBOL_PATTERNS = (
    re.compile(r"\bundefined:\s+(?:[\w./]+\.)?(\w+)\b"),
    re.compile(r"\bhas no field or method\s+(\w+)\b"),
    re.compile(r"\bunknown field\s+(\w+)\b"),
)
# The `path:line:col:` prefix that opens a diagnostic line (Windows `\` or POSIX `/`).
_DIAG_RE = re.compile(r"^(?P<file>\S+\.go):(?P<line>\d+):(?:\d+:)?\s")

# Uncommitted-file definition probes for a bare identifier `S` (substituted in).
# Deliberately permissive on the DEFINITION shapes below (a false "defined here" is
# cheap — the user eyeballs the named file — while a miss hides the forgotten add),
# but every template must match a DEFINITION, never a mere use: a struct-literal key
# `S:` or a field assignment is a use and is intentionally NOT probed, else any file
# that merely references the symbol would be mis-reported as its definer.
_DEF_TEMPLATES = (
    r"func\s+{s}\s*[\(\[]",          # func S(...)  / func S[T any](...)
    r"func\s+\([^)]*\)\s+{s}\s*[\(\[]",  # func (r T) S(...)  (method)
    r"type\s+{s}\b",                 # type S ...
    r"(?:const|var)\s+\(?\s*{s}\b",  # const S = ... / var S T = ...  (keyword-prefixed)
    r"^\s*{s}\s+[\w\*\[\]\.]+",      # struct field / typed block member:  S SomeType
    r"^\s*{s}\s*=",                  # untyped const/var block member:  S = ...
)


def parse_build_errors(stderr: str) -> dict:
    """Parse `go build ./...` stderr into failing packages and missing symbols.

    Returns ``{"failing_packages": [...], "missing_symbols": [ {symbol, referenced_in,
    at} ]}``. ``referenced_in`` is the package (last ``# pkg`` header seen);
    ``at`` is the ``file:line`` the compiler flagged. Pure — no I/O.
    """
    packages: list[str] = []
    symbols: list[dict] = []
    seen_sym: set[tuple[str, str]] = set()
    current_pkg = ""
    for raw in stderr.splitlines():
        line = raw.rstrip()
        mpkg = _PKG_RE.match(line)
        if mpkg:
            current_pkg = mpkg.group(1)
            if current_pkg not in packages:
                packages.append(current_pkg)
            continue
        mdiag = _DIAG_RE.match(line.strip())
        at = ""
        if mdiag:
            at = f"{mdiag.group('file')}:{mdiag.group('line')}"
        for pat in _SYMBOL_PATTERNS:
            for sym in pat.findall(line):
                key = (current_pkg, sym)
                if key in seen_sym:
                    continue
                seen_sym.add(key)
                symbols.append({"symbol": sym, "referenced_in": current_pkg, "at": at})
    return {"failing_packages": packages, "missing_symbols": symbols}


def defines_symbol(content: str, symbol: str) -> bool:
    """True if ``content`` looks like it DEFINES the bare identifier ``symbol``.

    Over-matches on purpose (see module note): the caller only uses this to point a
    human at a file to `git add`, so a spurious hit costs a glance, a miss costs the
    whole diagnosis.
    """
    if not symbol:
        return False
    esc = re.escape(symbol)
    for tmpl in _DEF_TEMPLATES:
        if re.search(tmpl.format(s=esc), content, re.MULTILINE):
            return True
    return False


def find_forgotten_files(missing_symbols: list[dict],
                         uncommitted: dict[str, str]) -> list[dict]:
    """Map each missing symbol to the uncommitted file(s) that define it.

    ``uncommitted`` maps a repo-relative path to its WORKING-TREE content (the new
    definition lives there, whether the file is modified or brand-new). Returns one
    entry per implicated file: ``{path, defines: [symbols...]}``, sorted by path.
    A symbol with no uncommitted definer is simply absent — that is the signal the
    break is NOT a forgotten add.
    """
    by_path: dict[str, set[str]] = {}
    for entry in missing_symbols:
        sym = entry.get("symbol") or ""
        for path, content in uncommitted.items():
            # `go build ./...` never compiles _test.go, so a build-time reference can
            # never resolve to a test file — excluding them keeps the diagnosis to
            # the non-test source that actually greens the build.
            if not path.endswith(".go") or path.endswith("_test.go"):
                continue
            if defines_symbol(content, sym):
                by_path.setdefault(path, set()).add(sym)
    return [
        {"path": path, "defines": sorted(defs)}
        for path, defs in sorted(by_path.items())
    ]


def classify(builds: bool, forgotten: list[dict], missing: list[dict]) -> str:
    if builds:
        return "BUILD_OK"
    if forgotten:
        return "BUILD_BROKEN_COHERENCE"
    return "BUILD_BROKEN_OTHER"


def diagnose(builds: bool, stderr: str, uncommitted: dict[str, str],
             head: str = "") -> dict:
    """Assemble the full verdict from a build result + the uncommitted file contents.

    Pure over its inputs so the whole decision is unit-testable without Go or git.
    """
    parsed = parse_build_errors("" if builds else stderr)
    forgotten = [] if builds else find_forgotten_files(parsed["missing_symbols"], uncommitted)
    verdict = classify(builds, forgotten, parsed["missing_symbols"])
    if verdict == "BUILD_OK":
        summary = "committed HEAD builds; the ci-fast red is not a build break"
    elif verdict == "BUILD_BROKEN_COHERENCE":
        files = ", ".join(f["path"] for f in forgotten)
        summary = (
            f"committed HEAD does not build; {len(parsed['missing_symbols'])} missing "
            f"symbol(s) are defined only in uncommitted file(s): {files}. "
            f"This is a forgotten `git add` — stage and commit those files to green the gate."
        )
    else:
        summary = (
            f"committed HEAD does not build ({len(parsed['failing_packages'])} package(s)) "
            f"but no uncommitted file defines the missing symbol(s); this is a genuine "
            f"compile error, not a forgotten add — inspect the flagged sites."
        )
    return {
        "head": head,
        "builds": builds,
        "verdict": verdict,
        "failing_packages": parsed["failing_packages"],
        "missing_symbols": parsed["missing_symbols"],
        "forgotten_files": forgotten,
        "summary": summary,
    }


# --------------------------------------------------------------------------- I/O


def repo_root() -> Path:
    try:
        top = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        top = ""
    return Path(top) if top else Path(__file__).resolve().parent.parent


def head_sha(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(root),
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def is_go_buildable_path(path: str) -> bool:
    """True if `go build ./...` would compile this path's package.

    Go's package discovery skips any directory component beginning with ``.`` or
    ``_`` and the ``testdata``/``vendor`` trees. A definition living under such a
    path can never satisfy a committed reference, so it must never be reported as a
    forgotten file — otherwise a stray untracked build-artifact dir (``.head_build_check/``)
    or a vendored copy pollutes the diagnosis. Mirrors the compiler's own rule.
    """
    if not path.endswith(".go"):
        return False
    parts = path.replace("\\", "/").split("/")
    for comp in parts[:-1]:
        if comp.startswith(".") or comp.startswith("_") or comp in ("testdata", "vendor"):
            return False
    return True


def uncommitted_files(root: Path) -> dict[str, str]:
    """Repo-relative path -> working-tree content for every uncommitted .go file.

    Covers modified/added/renamed (index or worktree) and untracked-but-not-ignored
    files, restricted to paths `go build ./...` actually compiles. Reads the
    working-tree bytes (where a forgotten definition lives).
    """
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all", "-z"],
            cwd=str(root), text=True, encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError):
        return {}
    result: dict[str, str] = {}
    for record in out.split("\0"):
        if len(record) < 4:
            continue
        path = record[3:]
        # Rename records carry `old\0new`; the split already isolated them, and the
        # `-z` `->`-free form gives us the new path directly for R entries.
        if not is_go_buildable_path(path):
            continue
        fp = root / path
        try:
            result[path] = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    return result


def build_committed_head(root: Path) -> tuple[bool, str]:
    """Archive committed HEAD to a temp tree and run `go build ./...` there.

    The archive contains ONLY committed content — the same blind base CI checks out —
    so the result reflects trunk, not the author's dirty working tree. Returns
    ``(ok, combined_output)``.
    """
    go = shutil.which("go")
    if not go:
        raise RuntimeError("go toolchain not found on PATH")
    tmp = Path(tempfile.mkdtemp(prefix="trunk_build_probe_"))
    try:
        archive = subprocess.run(
            ["git", "archive", "HEAD"], cwd=str(root),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if archive.returncode != 0:
            raise RuntimeError((archive.stderr or b"").decode("utf-8", "replace")[:400])
        untar = subprocess.run(
            ["tar", "-x", "-C", str(tmp)], input=archive.stdout,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if untar.returncode != 0:
            raise RuntimeError((untar.stderr or b"").decode("utf-8", "replace")[:400])
        proc = subprocess.run(
            [go, "build", "./..."], cwd=str(tmp),
            text=True, encoding="utf-8", capture_output=True,
        )
        return proc.returncode == 0, (proc.stderr or "") + (proc.stdout or "")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose whether the release gate's red trunk is a forgotten `git add`.")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument(
        "--build-log", type=str, default="",
        help="diagnose a captured `go build` error log instead of building HEAD "
             "(for offline/CI-log use; implies the build failed)")
    args = parser.parse_args(argv)

    root = repo_root()
    head = head_sha(root)
    try:
        if args.build_log:
            log = Path(args.build_log).read_text(encoding="utf-8", errors="replace")
            builds, stderr = False, log
        else:
            builds, stderr = build_committed_head(root)
    except Exception as exc:  # probe failure is distinct from a red trunk
        print(f"trunk-build-probe: could not probe HEAD: {exc}", file=sys.stderr)
        return 1

    verdict = diagnose(builds, stderr, uncommitted_files(root), head=head)

    if args.as_json:
        json.dump(verdict, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(f"trunk-build-probe: {verdict['verdict']} — {verdict['summary']}")
        for f in verdict["forgotten_files"]:
            print(f"  forgotten: {f['path']}  (defines {', '.join(f['defines'])})")
    return 0 if verdict["builds"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
