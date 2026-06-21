#!/usr/bin/env python3
"""Tests for tools/security_audit.py.

Plants a file tree in a temp dir that trips each deterministic check, asserts the
auditor catches it, then a clean tree and asserts silence. Also unit-tests the
go.mod require parser directly. Pure stdlib, no git needed.

Run: `python tools/security_audit_test.py`  (exit 0 = all pass).
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
MOD_PATH = os.path.join(HERE, "security_audit.py")

spec = importlib.util.spec_from_file_location("security_audit", MOD_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

failures: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    status = "ok  " if cond else "FAIL"
    print(f"  [{status}] {label}")
    if not cond:
        failures.append(label + (f" :: {detail}" if detail else ""))


def _write(root: Path, rel: str, content: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _make_tree(tmp: str, files: dict[str, str]) -> Path:
    root = Path(tmp)
    for rel, content in files.items():
        _write(root, rel, content)
    return root


# A dirty tree: a workflow that names NEITHER govulncheck nor the leak scanner nor
# `go vet`, an external dep with no go.sum, and no SECURITY.md.
DIRTY = {
    ".github/workflows/ci.yml": "name: ci\njobs:\n  x:\n    steps:\n      - run: go build ./...\n",
    "fak/go.mod": (
        "module github.com/anthony-chaudhary/fak\n\n"
        "go 1.26\n\n"
        "require github.com/some/external v1.2.3\n"
    ),
    # no SECURITY.md, no tools/scrub_public_copy.py, no fak/go.sum, no pre-commit hook
}

# A clean tree: govulncheck + leak scanner + go vet all wired, zero external deps,
# SECURITY.md present, scanner + hook present.
CLEAN = {
    ".github/workflows/security-audit.yml": (
        "name: security-audit\njobs:\n  v:\n    steps:\n"
        "      - run: govulncheck ./...\n"
    ),
    ".github/workflows/ci.yml": (
        "name: ci\njobs:\n  x:\n    steps:\n"
        "      - run: go vet ./...\n"
        "      - run: python tools/scrub_public_copy.py --audit-range HEAD~1..HEAD\n"
    ),
    "fak/go.mod": "module github.com/anthony-chaudhary/fak\n\ngo 1.26\n",
    "SECURITY.md": "# Security Policy\nReport privately via GitHub advisories.\n",
    "tools/scrub_public_copy.py": "# secret-leak scanner stub\n",
    "tools/githooks/pre-commit": "#!/bin/sh\nexit 0\n",
}


def test_dirty():
    print("dirty tree:")
    with tempfile.TemporaryDirectory() as tmp:
        root = _make_tree(tmp, DIRTY)
        f = mod.run(root, list(mod.CHECKS))
        names = {x.check for x in f if x.level == "FAIL"}
        check("govulncheck-ci fires (no govulncheck in CI)", "govulncheck-ci" in names)
        check("secret-leak-gate fires (scanner missing)", "secret-leak-gate" in names)
        check("dependency-surface fires (external dep, no go.sum)",
              "dependency-surface" in names)
        check("security-policy fires (no SECURITY.md)", "security-policy" in names)
        check("vet-lint warns (no go vet)",
              any(x.check == "vet-lint" and x.level == "WARN" for x in f))


def test_clean():
    print("clean tree:")
    with tempfile.TemporaryDirectory() as tmp:
        root = _make_tree(tmp, CLEAN)
        f = mod.run(root, list(mod.CHECKS))
        fails = [x for x in f if x.level == "FAIL"]
        check("no FAILs on clean tree", not fails,
              "; ".join(f"{x.check}:{x.msg}" for x in fails))
        warns = {x.check for x in f if x.level == "WARN"}
        check("no vet-lint WARN on clean tree", "vet-lint" not in warns)
        check("no dependency-surface finding on zero-dep go.mod",
              not any(x.check == "dependency-surface" for x in f))


def test_require_parser():
    print("require parser:")
    mp = "github.com/anthony-chaudhary/fak"
    # the real (current) go.mod shape: own module + commented-out require block -> zero.
    real = (
        "module github.com/anthony-chaudhary/fak\n"
        "go 1.26\n"
        "// require (\n"
        "//   github.com/microsoft/agent-governance-toolkit/x vX\n"
        "// )\n"
    )
    check("commented requires => zero external deps",
          mod.parse_external_requires(real, mp) == [])
    block = (
        "module m\nrequire (\n"
        "  github.com/a/b v1.0.0\n"
        "  golang.org/x/c v0.1.0 // indirect\n"
        ")\n"
    )
    check("block requires parsed",
          mod.parse_external_requires(block, "m") == ["github.com/a/b", "golang.org/x/c"])
    single = "module m\nrequire github.com/a/b v1.0.0\n"
    check("single-line require parsed",
          mod.parse_external_requires(single, "m") == ["github.com/a/b"])
    # stdlib-shaped tokens (no dotted domain) and the module's own path are excluded.
    check("stdlib-shaped + self excluded",
          mod.parse_external_requires("module m\nrequire m v1\nrequire fmt v0\n", "m") == [])


def test_real_repo():
    """Against the actual repo this lives in: the posture must be INTACT (zero FAIL)."""
    print("real repo (this checkout):")
    repo_root = Path(HERE).parent  # tools/ -> repo root
    if not (repo_root / ".github" / "workflows").is_dir():
        check("real-repo check skipped (no .github/workflows)", True)
        return
    f = mod.run(repo_root, list(mod.CHECKS))
    fails = [x for x in f if x.level == "FAIL"]
    check("real repo has zero security-posture FAILs", not fails,
          "; ".join(f"{x.check}:{x.msg}" for x in fails))


def main():
    test_dirty()
    test_clean()
    test_require_parser()
    test_real_repo()
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("security_audit_test: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
