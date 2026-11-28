#!/usr/bin/env python3
"""Tests for tools/public_readiness_audit.py.

Builds a throwaway git repo in a temp dir, plants a file tree that trips each
deterministic check, and asserts the auditor catches it -- then a clean tree and
asserts silence. Pure stdlib.

Run: `python tools/public_readiness_audit_test.py`  (exit 0 = all pass).
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
MOD_PATH = os.path.join(HERE, "public_readiness_audit.py")

spec = importlib.util.spec_from_file_location("public_readiness_audit", MOD_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

failures: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    status = "ok  " if cond else "FAIL"
    print(f"  [{status}] {label}")
    if not cond:
        failures.append(label + (f" :: {detail}" if detail else ""))


def _git(repo: str, *args: str) -> str:
    env = dict(os.environ)
    env.update(
        GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
        GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t",
        GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull,
    )
    out = subprocess.run(["git", "-C", repo, *args],
                         capture_output=True, text=True, encoding="utf-8",
                         errors="replace", env=env)
    return out.stdout


def _write(root: Path, rel: str, content: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _levels(findings, name):
    return [f for f in findings if f.check == name]


def _make_repo(tmp: str, files: dict[str, str]) -> Path:
    root = Path(tmp)
    _git(str(root), "init", "-q")
    for rel, content in files.items():
        _write(root, rel, content)
    _git(str(root), "add", "-A")
    _git(str(root), "commit", "-q", "-m", "fixture")
    return root


# A dirty tree that should trip many checks.
DIRTY = {
    "VERSION": "0.30.0\n",
    "README.md": (
        "# fak\n"
        "<!-- readme-verified: 2026-06-20 vs VERSION 0.26.0 -->\n"
        "curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fleet/main/install.sh | sh\n"
        "regenerated via tools/refresh_public_copy.py\n"
        "see [#81](https://github.com/anthony-chaudhary/fleet/issues/81)\n"
        "[missing](does-not-exist.md)\n"
        "[ok](START-HERE.md)\n"
    ),
    "START-HERE.md": "# start\n",
    "INSTALL.md": "# install\n",
    "INDEX.md": "# index\nregenerated via tools/refresh_public_copy.py\n",
    "CONTRIBUTING.md": "# contributing\n",
    "fak/GETTING-STARTED.md": "# gs\n",
    "fak/README.md": "# fak readme\n",
    "testfile_collision.txt": "scratch\n",
    "tools/control_pane.loops.json": "{}\n",
    ".github/workflows/ci.yml": "name: ci\n",
}

# A clean tree that should be silent on the deterministic FAIL checks.
CLEAN = {
    "VERSION": "0.30.0\n",
    "README.md": (
        "# fak\n"
        "<!-- readme-verified: 2026-06-21 vs VERSION 0.30.0 -->\n"
        "curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh\n"
        "This repo is canonical and edited directly.\n"
        "[ok](START-HERE.md)\n"
    ),
    "START-HERE.md": "# start\n",
    "INSTALL.md": "# install\n",
    "INDEX.md": "# index\nedited directly (hard cut)\n",
    "CONTRIBUTING.md": "# contributing\n",
    "CLA.md": "# cla\n",
    "LICENSE": "Apache-2.0\n",
    "SECURITY.md": "# security\nreport to security@example.com\n",
    "fak/GETTING-STARTED.md": "# gs\n",
    "fak/README.md": "# fak readme\n",
    "tools/control_pane.loops.json": '{"public-leak-scan": {}}\n',
    "tools/githooks/pre-commit": "#!/bin/sh\nexit 0\n",
    "tools/scrub_public_copy.py": "# secret-leak scanner stub\n",
    ".github/workflows/ci.yml": "name: ci\nrun: python tools/scrub_public_copy.py --audit-range\n",
}


def test_dirty():
    print("dirty tree:")
    with tempfile.TemporaryDirectory() as tmp:
        root = _make_repo(tmp, DIRTY)
        f = mod.run(root, list(mod.CHECKS))
        names = {x.check for x in f if x.level == "FAIL"}
        check("version-drift fires", any(x.check == "version-drift" and x.level == "FAIL" for x in f))
        check("human-urls fires", "human-urls" in names)
        check("retired-pipeline fires (README+INDEX)",
              len([x for x in f if x.check == "retired-pipeline"]) >= 2)
        check("issue-refs front-door fires", "issue-refs" in names)
        check("stray-artifacts fires", "stray-artifacts" in names)
        check("internal-links broken fires", "internal-links" in names)
        check("security-policy fires (no SECURITY.md)", "security-policy" in names)
        check("frontdoor-present fires (no CLA/LICENSE)", "frontdoor-present" in names)
        check("hooks-armed fires (leak gate missing)",
              any(x.check == "hooks-armed" and x.level == "FAIL" for x in f))


def test_clean():
    print("clean tree:")
    with tempfile.TemporaryDirectory() as tmp:
        root = _make_repo(tmp, CLEAN)
        f = mod.run(root, list(mod.CHECKS))
        fails = [x for x in f if x.level == "FAIL"]
        # hooks-armed is a WARN in a fresh temp repo (no core.hooksPath) -- ignore.
        deterministic = [x for x in fails
                         if x.check in {"version-drift", "human-urls", "retired-pipeline",
                                        "issue-refs", "stray-artifacts", "internal-links",
                                        "security-policy", "frontdoor-present", "example-files",
                                        "cache-tracked", "hooks-armed"}]
        check("no deterministic FAILs on clean tree", not deterministic,
              "; ".join(f"{x.check}:{x.msg}" for x in deterministic))


def test_doc_placement_no_goal_fallback():
    print("doc-placement GOAL-*.md fallback:")
    with tempfile.TemporaryDirectory() as tmp:
        files = dict(CLEAN)
        files["GOAL-internal.md"] = "# internal goal\n"
        root = _make_repo(tmp, files)
        f = mod.run(root, ["doc-placement"])
        fails = [x for x in f if x.level == "FAIL" and x.where == "GOAL-internal.md"]
        check("doc-placement rejects GOAL-*.md without check_doc_placement.py", bool(fails))
        assert fails, "expected GOAL-internal.md to fail doc-placement"


def test_hooks_armed_githooks_variants():
    print("hooks-armed .githooks and tools/githooks:")
    for hooks_dir in (".githooks", "tools/githooks"):
        with tempfile.TemporaryDirectory() as tmp:
            files = {
                f"{hooks_dir}/pre-commit": "#!/bin/sh\nexit 0\n",
                "tools/scrub_public_copy.py": "# scanner\n",
            }
            root = _make_repo(tmp, files)
            _git(str(root), "config", "core.hooksPath", hooks_dir)
            f = mod.run(root, ["hooks-armed"])
            check(f"hooks-armed clean for {hooks_dir}", not f, str([x.as_dict() for x in f]))
            assert not f, f"expected no findings for {hooks_dir}, got {[x.as_dict() for x in f]}"


def test_frontdoor_leaks():
    print("frontdoor-leaks:")
    with tempfile.TemporaryDirectory() as tmp:
        files = dict(CLEAN)
        files["GETTING-STARTED.md"] = (
            "./fak run /var/lib/fak/models/m.gguf\n"
            "see AVOID-TESTING-ON-THIS-MACHINE\n"
            "see docs/_witnesses/issue-1/\n"
        )
        files["CONTRIBUTING.md"] = "sync with private companion repo\n"
        files["llms.txt"] = (
            "Version: **0.26.0**\n"
            "source: internal/marketing/aeo.go\n"
        )
        root = _make_repo(tmp, files)
        f = mod.run(root, ["frontdoor-leaks"])
        msgs = "\n".join(x.msg for x in f if x.level == "FAIL")
        expected = [
            "/var/lib/fak/",
            "AVOID-TESTING-ON-THIS-MACHINE",
            "docs/_witnesses/",
            "private companion",
            "internal/marketing/aeo.go",
            "llms.txt Version: 0.26.0 lags VERSION 0.30.0",
        ]
        for needle in expected:
            check(f"frontdoor-leaks catches {needle}", needle in msgs, msgs)
            assert needle in msgs, f"missing finding for {needle}: {msgs}"

    with tempfile.TemporaryDirectory() as tmp:
        files = dict(CLEAN)
        files["GETTING-STARTED.md"] = "# Getting Started\n"
        files["STATUS.md"] = "# Status\n"
        files["llms.txt"] = "Version: **0.30.0** ([VERSION](VERSION))\n"
        root = _make_repo(tmp, files)
        f = mod.run(root, ["frontdoor-leaks"])
        check("frontdoor-leaks clean on sanitized frontdoor", not f, str([x.as_dict() for x in f]))
        assert not f, f"expected clean frontdoor-leaks, got {[x.as_dict() for x in f]}"


def main():
    test_dirty()
    test_clean()
    test_doc_placement_no_goal_fallback()
    test_hooks_armed_githooks_variants()
    test_frontdoor_leaks()
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("public_readiness_audit_test: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
