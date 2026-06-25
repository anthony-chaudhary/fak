#!/usr/bin/env python3
"""Unit tests for tools/claude_config_home_guard.py (issue #606).

Hermetic + pure-stdlib: the detection (`scan_text`) runs on synthetic source
snippets AND on the real content of the three tools the issue names as the correct
pattern (sync_memory.py / ctxcost.py / ctxwin.py). No subprocess, no network, no
tmpdir needed for the snippet cases. Run directly: `python tools/
claude_config_home_guard_test.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import claude_config_home_guard as G  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

_fail = 0


def check(name: str, cond: bool) -> None:
    global _fail
    print(("ok   " if cond else "FAIL ") + name)
    if not cond:
        _fail += 1


# ---- the bug: bare ~/.claude/<store> with NO CLAUDE_CONFIG_DIR -> flagged ----------
BAD_EXPANDUSER = '''
import os
def store(slug):
    home = os.path.expanduser("~")
    return os.path.join(home, ".claude", "projects", slug, "memory")
'''
BAD_PATHLIB = '''
from pathlib import Path
def store(slug):
    return Path.home() / ".claude" / "projects" / slug / "memory"
'''
BAD_TILDE_LITERAL = '''
import os
TRANSCRIPTS = os.path.expanduser("~/.claude/projects")
'''

check("flags os.path.join(home, '.claude', 'projects')", bool(G.scan_text(BAD_EXPANDUSER)))
check("flags Path.home() / '.claude' / 'projects'", bool(G.scan_text(BAD_PATHLIB)))
check("flags '~/.claude/projects' literal", bool(G.scan_text(BAD_TILDE_LITERAL)))

# ---- the antidote present anywhere -> NOT flagged ---------------------------------
GOOD_WITH_ENV = '''
import os
from pathlib import Path
def store(slug):
    cfg = os.environ.get("CLAUDE_CONFIG_DIR")
    base = Path(cfg) if cfg else Path.home() / ".claude"
    return base / "projects" / slug / "memory"
'''
check("passes when CLAUDE_CONFIG_DIR is consulted (precedence pattern)",
      not G.scan_text(GOOD_WITH_ENV))

# ---- legitimate patterns that must NOT be flagged --------------------------------
GLOB_ALL_ACCOUNTS = '''
import glob, os
home = os.path.expanduser("~")
for p in glob.glob(os.path.join(home, ".claude*", "projects", "*", "*.jsonl")):
    pass
'''
check("passes the all-accounts glob '.claude*' (relocation-aware)",
      not G.scan_text(GLOB_ALL_ACCOUNTS))

REPO_ROOT_MIRROR = '''
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
mirror = ROOT / ".claude" / "memory"   # repo-root committed mirror, not the home
'''
check("passes the repo-root mirror '<root>/.claude/memory' (not home-anchored)",
      not G.scan_text(REPO_ROOT_MIRROR))

PER_ACCOUNT_VARIANT = '''
import os
home = os.path.expanduser("~")
p = os.path.join(home, ".claude-q-netra", "projects")
'''
check("passes a per-account variant '.claude-q-netra' (not bare .claude)",
      not G.scan_text(PER_ACCOUNT_VARIANT))

DOCSTRING_ONLY = '''
"""Backs up ~/.claude/projects/<slug>/memory into the archive."""
import os
def go():
    return os.environ.get("ARCHIVE")
'''
check("passes a docstring that merely MENTIONS ~/.claude/projects",
      not G.scan_text(DOCSTRING_ONLY))

COMMENT_ONLY = '''
import os
# resolves ~/.claude/projects/<slug>/memory when relocated
x = os.environ.get("ARCHIVE")
'''
check("passes a comment that merely MENTIONS ~/.claude/projects",
      not G.scan_text(COMMENT_ONLY))

# ---- the three exemplars the issue names: must be CLEAN on real content -----------
for name in ("sync_memory.py", "ctxcost.py", "ctxwin.py"):
    src = (REPO / "tools" / name).read_text(encoding="utf-8")
    check(f"real {name} passes (named correct-pattern exemplar)", not G.scan_text(src))

# ---- the whole production tree is clean today (so --check is green in CI) ---------
tree_hits = {
    f.relative_to(REPO).as_posix(): G.scan_text(f.read_text(encoding="utf-8"))
    for f in G.production_tools()
}
dirty = {k: v for k, v in tree_hits.items() if v}
check(f"production tools tree is clean today ({len(tree_hits)} files; dirty={list(dirty)})",
      not dirty)

# ---- the masker never crashes on a syntactically-broken file ----------------------
BROKEN = "def (:\n    ~/.claude/projects\n"
try:
    G.scan_text(BROKEN)
    check("scan_text tolerates unparseable source", True)
except Exception as e:  # noqa: BLE001
    check(f"scan_text tolerates unparseable source (raised {e!r})", False)

if _fail:
    print(f"\nFAIL: {_fail} assertion(s)")
    raise SystemExit(1)
print("\nok: claude_config_home_guard self-tests pass")
