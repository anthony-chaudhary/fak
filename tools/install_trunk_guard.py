#!/usr/bin/env python3
"""Activate fleet's trunk-discipline git guard for THIS clone.

Why a git hook (and not just a CLAUDE.md line): git runs the hook below the agent
layer, so the *same* refusal binds Claude Code, Codex, and a human -- the
cross-agent enforcement the docs alone cannot give. The hook is the named "floor"
under ``dos.toml [reasons.OFF_TRUNK]``.

Effect: sets ``core.hooksPath -> tools/githooks`` and marks the hook executable.
Idempotent -- safe to re-run. Default mode blocks off-trunk branch creation; export
``FLEET_TRUNK_GUARD=warn`` to make it advisory, or ``FLEET_TRUNK_GUARD=off`` to mute.

Run once per clone/node:  ``python tools/install_trunk_guard.py``
"""
import os
import stat
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOOKS_REL = "tools/githooks"
HOOK = os.path.join(ROOT, "tools", "githooks", "reference-transaction")


def main() -> int:
    if not os.path.isfile(HOOK):
        print(f"error: hook not found: {HOOK}", file=sys.stderr)
        return 1
    # Executable bit is a no-op on Windows (git runs it via the bundled sh) but
    # matters on the Linux/WSL fleet nodes.
    mode = os.stat(HOOK).st_mode
    os.chmod(HOOK, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    # chmod every other tracked hook in tools/githooks/ too (e.g. pre-commit
    # leak-scan guard) so installing the trunk guard also arms its siblings.
    armed = [os.path.basename(HOOK)]
    for name in sorted(os.listdir(os.path.dirname(HOOK))):
        full = os.path.join(os.path.dirname(HOOK), name)
        if not os.path.isfile(full) or name in armed:
            continue
        m = os.stat(full).st_mode
        os.chmod(full, m | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        armed.append(name)
    # Relative path: git resolves core.hooksPath from the worktree root.
    subprocess.run(["git", "-C", ROOT, "config", "core.hooksPath", HOOKS_REL], check=True)
    got = subprocess.run(
        ["git", "-C", ROOT, "config", "--get", "core.hooksPath"],
        capture_output=True, text=True,
    ).stdout.strip()
    print(f"core.hooksPath = {got}  (armed: {', '.join(armed)})")
    print("  trunk guard active, blocking off-trunk branches")
    print("  leak-scan guard active, blocking redact-needles in staged content")
    print("  claim-honesty guard active (pre-push, advisory), surfacing CLAIM_UNWITNESSED via `dos review`")
    print("  override once:  FLEET_ALLOW_BRANCH=1 <git cmd>   |   FLEET_ALLOW_LEAK=1 <git cmd>   |   FLEET_ALLOW_RESIDUAL=1 git push")
    print("  advisory mode:  export FLEET_TRUNK_GUARD=warn     |   export FLEET_SCRUB_GUARD=warn   |   hard-enforce: FLEET_REVIEW_GUARD=block")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
