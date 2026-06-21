#!/usr/bin/env python3
"""Public-readiness auditor for the fleet-public repo.

Verifies the deterministic criteria in docs/PUBLIC-READINESS-DOD.md so that
"is this repo ready to be flipped PUBLIC?" is a re-runnable check instead of a
one-time manual sweep. The judgment-heavy criteria (A5 demo runs, prose
staleness) are out of scope here and handled by review; everything mechanical
lives in this file.

Each check yields zero or more findings. A finding is FAIL (blocks the go/no-go)
or WARN (surface, don't block). Exit code is 1 if any FAIL, else 0. `--json`
emits machine-readable output; `--check NAME[,NAME]` runs a subset.

Design notes:
  * The set of "tracked files" comes from `git ls-files` — never the working
    tree — so untracked scratch never counts and deleted-but-staged does.
  * The Go module path `github.com/anthony-chaudhary/fak` is an INTENTIONAL
    exception (DoD B3): the module is not renamed to fleet-public. Only
    human-facing URLs (install/releases/clone/blob/tree/issues) must carry the
    public repo slug.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PUBLIC_SLUG = "fleet-public"
OWNER = "anthony-chaudhary"

# Front-door docs: what a brand-new public visitor reads first. Stricter scope —
# findings here are FAIL, not WARN.
FRONT_DOOR = [
    "README.md",
    "START-HERE.md",
    "INSTALL.md",
    "INDEX.md",
    "CONTRIBUTING.md",
    "fak/GETTING-STARTED.md",
    "fak/README.md",
]

REQUIRED_FRONT_DOOR = [
    "README.md",
    "START-HERE.md",
    "INSTALL.md",
    "fak/GETTING-STARTED.md",
    "CONTRIBUTING.md",
    "CLA.md",
    "LICENSE",
]

# Stray artifacts that must never be tracked in a public repo.
STRAY_BASENAMES = {"testfile_collision.txt"}
STRAY_PATTERNS = [re.compile(r".*_synth\.txt$"), re.compile(r".*\.orig$"), re.compile(r".*\.rej$")]

# Regenerable caches that must stay untracked.
CACHE_DIR_MARKERS = (".pytest_cache/", ".ruff_cache/", "__pycache__/", ".mypy_cache/")

# Human-facing URL shapes that must use the public slug. Each matches the WRONG
# (bare `fleet`) form; the Go import path `fleet/fak` is deliberately excluded.
HUMAN_URL_PATTERNS = [
    re.compile(r"raw\.githubusercontent\.com/" + OWNER + r"/fleet/"),
    re.compile(r"github\.com/" + OWNER + r"/fleet/(releases|blob|tree|pull|wiki|archive)"),
    re.compile(r"github\.com/" + OWNER + r"/fleet\.git"),
]
ISSUE_URL_PATTERN = re.compile(r"github\.com/" + OWNER + r"/fleet/issues/\d+")

# Markdown link: [text](target). Capture target, ignore images handled the same.
MD_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")

VERSION_MARKER = re.compile(r"readme-verified:[^\n]*VERSION\s+([0-9]+\.[0-9]+\.[0-9]+)")


def repo_root(start: Path) -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=start, capture_output=True, text=True, check=True,
    )
    return Path(out.stdout.strip())


def tracked_files(root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True,
    )
    return [line for line in out.stdout.splitlines() if line]


def read(root: Path, rel: str) -> str:
    p = root / rel
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="replace")


class Finding:
    def __init__(self, check: str, level: str, msg: str, where: str = ""):
        self.check = check
        self.level = level  # FAIL | WARN
        self.msg = msg
        self.where = where

    def as_dict(self):
        return {"check": self.check, "level": self.level, "msg": self.msg, "where": self.where}


# --- individual checks ------------------------------------------------------

def check_frontdoor_present(root, tracked):
    out = []
    for rel in REQUIRED_FRONT_DOOR:
        if not (root / rel).exists():
            out.append(Finding("frontdoor-present", "FAIL", f"required front-door file missing", rel))
    return out


def check_security_policy(root, tracked):
    candidates = ["SECURITY.md", ".github/SECURITY.md", "docs/SECURITY.md"]
    if any((root / c).exists() for c in candidates):
        return []
    return [Finding("security-policy", "FAIL",
                    "no security-disclosure policy found (SECURITY.md at root or .github/)", "")]


def check_version_drift(root, tracked):
    out = []
    version = read(root, "VERSION").strip()
    if not version:
        return [Finding("version-drift", "WARN", "VERSION file missing/empty", "VERSION")]
    readme = read(root, "README.md")
    m = VERSION_MARKER.search(readme)
    if not m:
        out.append(Finding("version-drift", "WARN",
                           "README has no parseable readme-verified VERSION marker", "README.md"))
    elif m.group(1) != version:
        out.append(Finding("version-drift", "FAIL",
                           f"README readme-verified marks VERSION {m.group(1)} but VERSION file is {version}",
                           "README.md"))
    return out


def check_retired_pipeline(root, tracked):
    # refresh_public_copy.py was retired by the 2026-06-20 hard cut. Front-door /
    # index prose must not describe the repo as a *live* regenerated mirror. A line
    # that explicitly says the pipeline is retired/superseded is correct and passes.
    out = []
    retired_markers = ("retire", "supersede", "no longer", "deprecat", "was the", "old model")
    scope = ["README.md", "INDEX.md"]
    for rel in scope:
        txt = read(root, rel)
        for i, line in enumerate(txt.splitlines(), 1):
            if "refresh_public_copy" not in line:
                continue
            low = line.lower()
            if any(m in low for m in retired_markers):
                continue  # mentions it only to record that it's gone
            out.append(Finding("retired-pipeline", "FAIL",
                               "describes the repo as regenerated via refresh_public_copy (hard-cut: repo is canonical)",
                               f"{rel}:{i}"))
    return out


def check_internal_links(root, tracked):
    # Repo-wide: every relative markdown link in every tracked .md must resolve.
    # Front-door breakage is FAIL (a public visitor hits it first); elsewhere WARN.
    # Template/agent dirs carry intentional placeholder links, so they're skipped.
    out = []
    tracked_set = set(tracked)
    front = set(FRONT_DOOR + ["START-HERE.md"])
    skip_prefixes = (".claude/", ".opencode/")
    md_files = [f for f in tracked if f.endswith(".md") and not f.startswith(skip_prefixes)]
    for rel in md_files:
        if not (root / rel).exists():
            continue
        base = (root / rel).parent
        txt = read(root, rel)
        for i, line in enumerate(txt.splitlines(), 1):
            for m in MD_LINK.finditer(line):
                target = m.group(1).strip()
                if not target or target.startswith(("http://", "https://", "#", "mailto:", "tel:")):
                    continue
                # strip anchor and any title
                target = target.split()[0]
                target = target.split("#", 1)[0]
                if not target:
                    continue
                resolved = (base / target).resolve()
                try:
                    relpath = resolved.relative_to(root.resolve()).as_posix()
                except ValueError:
                    continue  # link escapes the repo root; not our concern here
                if not resolved.exists() and relpath not in tracked_set:
                    level = "FAIL" if rel in front else "WARN"
                    out.append(Finding("internal-links", level,
                                       f"broken internal link -> {target}", f"{rel}:{i}"))
    return out


def check_human_urls(root, tracked):
    out = []
    docs = [f for f in tracked if f.endswith((".md", ".sh", ".html", ".txt"))]
    for rel in docs:
        txt = read(root, rel)
        for i, line in enumerate(txt.splitlines(), 1):
            for pat in HUMAN_URL_PATTERNS:
                if pat.search(line):
                    out.append(Finding("human-urls", "FAIL",
                                       f"human-facing URL uses bare `fleet` (should be {PUBLIC_SLUG})",
                                       f"{rel}:{i}"))
                    break
    return out


def check_issue_refs(root, tracked):
    out = []
    for rel in FRONT_DOOR:
        txt = read(root, rel)
        for i, line in enumerate(txt.splitlines(), 1):
            if ISSUE_URL_PATTERN.search(line):
                out.append(Finding("issue-refs", "FAIL",
                                   "front-door doc links a bare `fleet` (private) issue number",
                                   f"{rel}:{i}"))
    # deep docs: WARN only
    deep = [f for f in tracked if f.endswith(".md") and f not in FRONT_DOOR]
    n = 0
    for rel in deep:
        txt = read(root, rel)
        if ISSUE_URL_PATTERN.search(txt):
            n += 1
    if n:
        out.append(Finding("issue-refs", "WARN",
                           f"{n} non-front-door doc(s) carry bare fleet/issues/N provenance links", ""))
    return out


def check_stray_artifacts(root, tracked):
    out = []
    for f in tracked:
        base = Path(f).name
        if base in STRAY_BASENAMES or any(p.match(f) for p in STRAY_PATTERNS):
            out.append(Finding("stray-artifacts", "FAIL", "stray/scratch artifact tracked", f))
    return out


def check_cache_tracked(root, tracked):
    out = []
    for f in tracked:
        if any(marker in (f + "/") for marker in CACHE_DIR_MARKERS):
            out.append(Finding("cache-tracked", "FAIL", "regenerable cache is tracked", f))
    return out


def check_hooks_armed(root, tracked):
    # Two distinct facts: (a) the gate is ARMED in this clone (core.hooksPath), and
    # (b) the gate actually EXISTS in the tree. (b) is FAIL — the leak gate was once
    # removed wholesale (the #78 regression); this catches that class directly.
    out = []
    val = subprocess.run(["git", "config", "--get", "core.hooksPath"],
                         cwd=root, capture_output=True, text=True).stdout.strip()
    if val != "tools/githooks":
        out.append(Finding("hooks-armed", "WARN",
                           f"core.hooksPath is '{val or '(unset)'}' — run tools/install_trunk_guard.py", ""))
    if not (root / "tools/githooks/pre-commit").exists():
        out.append(Finding("hooks-armed", "FAIL",
                           "pre-commit leak hook is missing — the public secret-leak gate is gone",
                           "tools/githooks/pre-commit"))
    if not (root / "tools/scrub_public_copy.py").exists():
        out.append(Finding("hooks-armed", "FAIL",
                           "scrub_public_copy.py (the secret-leak scanner) is missing",
                           "tools/scrub_public_copy.py"))
    return out


def check_ci_leakgate(root, tracked):
    ci = read(root, ".github/workflows/ci.yml")
    if not ci:
        return [Finding("ci-leakgate", "FAIL", "no .github/workflows/ci.yml", "")]
    if "scrub_public_copy" not in ci and "leak" not in ci.lower():
        return [Finding("ci-leakgate", "WARN", "CI workflow does not appear to run the leak gate",
                        ".github/workflows/ci.yml")]
    return []


def check_leakscan_loop(root, tracked):
    loops = read(root, "tools/control_pane.loops.json")
    if "public-leak-scan" not in loops:
        return [Finding("leakscan-loop", "WARN",
                        "no standing public-leak-scan loop in control_pane.loops.json", "")]
    return []


def check_index_sync(root, tracked):
    out = []
    index = read(root, "INDEX.md")
    # dated notes now live under docs/notes/ (moved out of the root, see
    # check_doc_placement); each should be reachable from INDEX.md.
    dated = sorted(f for f in tracked
                   if f.startswith("docs/notes/") and f.endswith(".md")
                   and f.rsplit("/", 1)[-1] != "README.md"
                   and (re.search(r"20\d\d-\d\d-\d\d", f) or f.rsplit("/", 1)[-1].startswith("PLAN-")))
    for f in dated:
        base = f.rsplit("/", 1)[-1]
        if base not in index:
            out.append(Finding("index-sync", "WARN", "dated note not listed in INDEX.md", f))
    return out


def check_doc_placement(root, tracked):
    # Dated research/strategy docs belong in docs/notes/, not the repo root — the
    # public front door must stay uncluttered. Shares the allowlist with the
    # pre-commit gate (tools/check_doc_placement.py) so the two cannot drift.
    import sys as _sys
    allow = None
    try:
        _sys.path.insert(0, str(root / "tools"))
        from check_doc_placement import ALLOWED_ROOT_MD as _allow
        allow = set(_allow)
    except Exception:
        allow = {"README.md", "START-HERE.md", "INSTALL.md", "INDEX.md", "CONTRIBUTING.md",
                 "CLA.md", "SECURITY.md", "PUBLIC-SCRUB-POLICY.md", "AGENTS.md", "CLAUDE.md",
                 "CODE_OF_CONDUCT.md", "CHANGELOG.md", "GOVERNANCE.md", "MAINTAINERS.md",
                 "ROADMAP.md", "AUTHORS.md", "NOTICE.md", "SUPPORT.md", "HISTORY.md"}
    out = []
    for f in sorted(tracked):
        if "/" not in f and f.endswith(".md") and f not in allow:
            out.append(Finding("doc-placement", "FAIL",
                               "dated/research doc at repo root — move to docs/notes/", f))
    return out


def check_example_files(root, tracked):
    out = []
    readme = read(root, "README.md")
    # example policy json paths referenced in README
    for m in re.finditer(r"(fak/examples/[\w./-]+\.json)", readme):
        rel = m.group(1)
        if not (root / rel).exists():
            out.append(Finding("example-files", "FAIL", "README references missing example file", rel))
    return out


def check_committed_files(root, tracked):
    # No build artifacts / junk / oversized blobs in the tree (shares the gate's
    # rules: tools/check_committed_files.py).
    import sys as _sys
    try:
        _sys.path.insert(0, str(root / "tools"))
        import check_committed_files as _ccf
    except Exception:
        return []
    out = []
    for f in sorted(tracked):
        reason = _ccf._classify(f, str(root), _ccf.DEFAULT_MAX_BYTES)
        if reason:
            out.append(Finding("committed-files", "FAIL", reason, f))
    return out


def check_secret_shapes(root, tracked):
    # Operator-leak SHAPES the literal needle list misses (shares the gate's
    # patterns: tools/check_secret_shapes.py).
    import sys as _sys
    try:
        _sys.path.insert(0, str(root / "tools"))
        import check_secret_shapes as _css
    except Exception:
        return []
    out = []
    for f in sorted(tracked):
        if f in _css.SELF_REF or not f.endswith(_css.TEXT_EXT):
            continue
        try:
            text = (root / f).read_text(encoding="utf-8")
        except Exception:
            continue
        seen = set()
        for shape, hit in _css._scan_text(text):
            if hit in seen:
                continue
            seen.add(hit)
            out.append(Finding("secret-shapes", "FAIL", f"operator-leak shape [{shape}]: {hit}", f))
    return out


CHECKS = {
    "frontdoor-present": check_frontdoor_present,
    "security-policy": check_security_policy,
    "version-drift": check_version_drift,
    "retired-pipeline": check_retired_pipeline,
    "internal-links": check_internal_links,
    "human-urls": check_human_urls,
    "issue-refs": check_issue_refs,
    "stray-artifacts": check_stray_artifacts,
    "cache-tracked": check_cache_tracked,
    "hooks-armed": check_hooks_armed,
    "ci-leakgate": check_ci_leakgate,
    "leakscan-loop": check_leakscan_loop,
    "index-sync": check_index_sync,
    "example-files": check_example_files,
    "doc-placement": check_doc_placement,
    "committed-files": check_committed_files,
    "secret-shapes": check_secret_shapes,
}


def run(root: Path, names: list[str]) -> list[Finding]:
    tracked = tracked_files(root)
    findings: list[Finding] = []
    for name in names:
        findings.extend(CHECKS[name](root, tracked))
    return findings


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".", help="repo root (default: cwd, resolved via git)")
    ap.add_argument("--check", default="", help="comma-separated subset of checks")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    root = repo_root(Path(args.root))
    names = [c.strip() for c in args.check.split(",") if c.strip()] if args.check else list(CHECKS)
    bad = [c for c in names if c not in CHECKS]
    if bad:
        print(f"unknown check(s): {', '.join(bad)}", file=sys.stderr)
        return 2

    findings = run(root, names)
    fails = [f for f in findings if f.level == "FAIL"]
    warns = [f for f in findings if f.level == "WARN"]

    if args.json:
        print(json.dumps({
            "root": str(root),
            "checks": names,
            "fail": len(fails),
            "warn": len(warns),
            "findings": [f.as_dict() for f in findings],
        }, indent=2))
    else:
        if not findings:
            print(f"public-readiness: CLEAN across {len(names)} checks")
        else:
            for f in findings:
                loc = f" [{f.where}]" if f.where else ""
                print(f"{f.level:4} {f.check}: {f.msg}{loc}")
            print(f"\n{len(fails)} FAIL, {len(warns)} WARN across {len(names)} checks")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
