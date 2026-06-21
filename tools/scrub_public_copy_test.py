#!/usr/bin/env python3
"""Tests for the scrubber's CI leak-gate (`--audit-range`) and its self-ref exemption.

The pre-commit gate (`audit_staged`) only fires on an armed clone; `audit_range`
is the machine backstop that runs in CI over the pushed range. This proves it:
(1) catches a planted secret-tier needle in an added line, (2) passes a clean
range, (3) is line-oriented (only ADDED lines, not removed/context), and (4)
honors SELF_REFERENTIAL so editing the denylist itself does not cry wolf -- the
same exemption the post-export audit applies.

Run: `python tools/scrub_public_copy_test.py`  (exit 0 = all pass).
Pure stdlib; builds a throwaway git repo in a temp dir.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRUB = os.path.join(HERE, "scrub_public_copy.py")
# A real secret-tier needle the gate must catch (the IP this change added).
# Assembled from parts so the literal never appears in THIS file -- this test is
# not SELF_REFERENTIAL, so a bare literal would (correctly) trip the leak gate on
# its own commit. We want the gate live on test code, not exempted.
NEEDLE = ".".join(["100", "95", "5", "23"])


def _git(repo: str, *args: str) -> str:
    env = dict(os.environ)
    env.update(
        GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
        GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t",
        # Never let the repo-under-test inherit fleet's hooksPath.
        GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull,
    )
    out = subprocess.run(
        ["git", "-C", repo, *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
    )
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {out.stderr}")
    return out.stdout.strip()


def _audit_range(repo: str, rng: str):
    """Return (exit_code, combined_output) of the scrubber's --audit-range."""
    out = subprocess.run(
        [sys.executable, SCRUB, "--audit-range", rng, "--root", repo],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return out.returncode, out.stdout + out.stderr


def _audit_tree(repo: str, as_json: bool = False):
    """Return (exit_code, combined_output) of the scrubber's --audit-tree."""
    cmd = [sys.executable, SCRUB, "--audit-tree", "--root", repo]
    if as_json:
        cmd.append("--json")
    out = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return out.returncode, out.stdout + out.stderr


def _write(repo: str, rel: str, text: str) -> None:
    path = os.path.join(repo, rel)
    os.makedirs(os.path.dirname(path) or repo, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main() -> int:
    failures = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        status = "ok" if cond else "FAIL"
        print(f"  [{status}] {name}" + (f"  -- {detail}" if not cond and detail else ""))
        if not cond:
            failures.append(name)

    with tempfile.TemporaryDirectory() as repo:
        _git(repo, "init", "-q")
        _git(repo, "config", "commit.gpgsign", "false")
        # Pull the private scan instructions into this throwaway repo. The public
        # AUDIT_NEEDLES are de-fanged, so under the hard cut the gate folds in the
        # REAL needle from a gitignored sidecar. Ignore the sidecar dir so it is
        # never tracked or scanned, exactly as in the real repos.
        _write(repo, ".gitignore", "tools/_registry/\n")
        _write(repo, "tools/_registry/scrub_needles.private.json",
               '{"schema":"fleet-scrub-needles/1","audit_needles":["%s"],'
               '"export_audit_needles":["%s"]}\n' % (NEEDLE, NEEDLE))
        # base commit: a clean file.
        _write(repo, "docs/notes.md", "hello world\nno secrets here\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "base")
        base = _git(repo, "rev-parse", "HEAD")

        # 1) clean range -> exit 0
        _write(repo, "docs/notes.md", "hello world\nstill clean\nanother line\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "clean change")
        rc, out = _audit_range(repo, f"{base}..HEAD")
        check("clean range exits 0", rc == 0, out)
        check("clean range reports clean", "clean" in out, out)

        clean_head = _git(repo, "rev-parse", "HEAD")

        # 2) added needle -> exit 1, names the file+needle
        _write(repo, "node.json", '{"ip": "%s"}\n' % NEEDLE)
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "leak")
        rc, out = _audit_range(repo, f"{clean_head}..HEAD")
        check("planted needle exits 1", rc == 1, out)
        check("planted needle is named", NEEDLE in out and "node.json" in out, out)

        leak_head = _git(repo, "rev-parse", "HEAD")

        # 3) REMOVING the needle is not a hit (only ADDED lines are scanned)
        _write(repo, "node.json", '{"ip": "0.0.0.0"}\n')
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "remove leak")
        rc, out = _audit_range(repo, f"{leak_head}..HEAD")
        check("removing a needle is not a hit", rc == 0, out)

        # 4) SELF_REFERENTIAL exemption: the same needle inside the denylist file
        #    itself must NOT trip the gate (it has to name needles to enforce them).
        safe_head = _git(repo, "rev-parse", "HEAD")
        _write(repo, "PUBLIC-SCRUB-POLICY.md", "| `%s` | redact | x |\n" % NEEDLE)
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "edit denylist")
        rc, out = _audit_range(repo, f"{safe_head}..HEAD")
        check("self-referential file is exempt", rc == 0, out)

        # 5) unreadable range -> exit 2 (mirrors dos review's grammar)
        rc, out = _audit_range(repo, "nope-no-such-ref..HEAD")
        check("unreadable range exits 2", rc == 2, out)

    # ---- audit-tree: the HARD-CUT full-tree backstop (its own repo) --------
    # Separate repo so the sidecar's presence/absence is controlled here and does
    # not interact with the commit-gate block above.
    with tempfile.TemporaryDirectory() as repo:
        _git(repo, "init", "-q")
        _git(repo, "config", "commit.gpgsign", "false")
        # Ignore the pulled-needle sidecar dir exactly like the real repos do, so
        # `git add -A` never tracks it (and audit-tree never self-hits on it).
        _write(repo, ".gitignore", "tools/_registry/\n")
        _write(repo, "docs/notes.md", "hello world\nno secrets here\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "base")

        # 6) clean tree, no sidecar pulled -> exit 0, honest shape-only mode.
        rc, out = _audit_tree(repo)
        check("audit-tree clean (no sidecar) exits 0", rc == 0, out)
        check("audit-tree reports shape-only without sidecar", "shape-only" in out, out)

        # 7) a live-token SHAPE is caught even in shape-only mode (always sound).
        #    Assembled from parts so the literal token never appears in this file.
        token = "xoxb-" + "12345678" + "-" + "87654321" + "-" + "abcdefGHIJ0123456789"
        _write(repo, "cfg.txt", "slack_token=%s\n" % token)
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "add token")
        rc, out = _audit_tree(repo)
        check("audit-tree catches live-token shape (shape-only)", rc == 1, out)
        check("audit-tree names the offending file", "cfg.txt" in out, out)
        os.remove(os.path.join(repo, "cfg.txt"))
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "remove token")

        # 8) FULL mode: a pulled private needle in a tracked file is caught; the
        #    sidecar itself (ignored, untracked) is NOT scanned, so no self-hit.
        real = ".".join(["10", "11", "12", "13"])
        _write(repo, "tools/_registry/scrub_needles.private.json",
               '{"schema":"fleet-scrub-needles/1","audit_needles":[],'
               '"export_audit_needles":["%s"]}\n' % real)
        _write(repo, "node.json", '{"ip":"%s"}\n' % real)
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "leak with sidecar pulled")
        rc, out = _audit_tree(repo)
        check("audit-tree full mode catches pulled needle", rc == 1, out)
        check("audit-tree full mode names file + reports full",
              "node.json" in out and "full" in out, out)

        # 9) --json emits the ok + mode contract the control-pane loop reads.
        rc, jout = _audit_tree(repo, as_json=True)
        check("audit-tree --json emits ok+mode keys", '"ok"' in jout and '"mode"' in jout, jout)

    # ---- §6 commit-gate EXPORT/identity-tier fold (its own repo) ------------
    # Under the hard cut the PUBLIC clone has no export step left to rewrite the
    # identity tier, so a provisioned clone (sidecar present) must catch an
    # EXPORT-tier needle AT COMMIT -- not just the secret tier. The private repo
    # (no sidecar) keeps its secret-tier-only gate, unchanged, so a module-org
    # identity string never cries wolf there. _audit_range routes through the same
    # _effective_audit_needles the pre-commit gate uses, so it is a faithful proxy.
    with tempfile.TemporaryDirectory() as repo:
        _git(repo, "init", "-q")
        _git(repo, "config", "commit.gpgsign", "false")
        _write(repo, ".gitignore", "tools/_registry/\n")
        _write(repo, "docs/notes.md", "hello world\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "base")
        base = _git(repo, "rev-parse", "HEAD")

        # An EXPORT/identity-tier needle: present ONLY in export_audit_needles, not
        # in the committed secret-tier AUDIT_NEEDLES. Assembled from parts so the
        # literal never appears in this (non-self-referential) file.
        export_needle = "example" + "corp" + "-private"

        # 10) WITHOUT a sidecar, an identity-tier string is NOT gated -- the private
        #     repo's secret-tier-only gate is unchanged (no crying wolf on imports).
        _write(repo, "go.mod", "module %s/x\n" % export_needle)
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "import")
        rc, out = _audit_range(repo, f"{base}..HEAD")
        check("export-tier NOT gated without sidecar (private unchanged)", rc == 0, out)
        import_head = _git(repo, "rev-parse", "HEAD")

        # 11) WITH the sidecar (a provisioned public clone), the SAME identity-tier
        #     needle IS caught at commit -- the §6 hardening. The sidecar lists it
        #     only in export_audit_needles, so this proves the EXPORT tier folds in.
        _write(repo, "tools/_registry/scrub_needles.private.json",
               '{"schema":"fleet-scrub-needles/1","audit_needles":[],'
               '"export_audit_needles":["%s"]}\n' % export_needle)
        _write(repo, "leak.json", '{"org":"%s"}\n' % export_needle)
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "leak with sidecar")
        rc, out = _audit_range(repo, f"{import_head}..HEAD")
        check("export-tier gated WITH sidecar (PLAN §6)", rc == 1, out)
        check("export-tier hit names the file", "leak.json" in out, out)

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("scrub_public_copy_test: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
