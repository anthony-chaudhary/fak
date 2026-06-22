#!/usr/bin/env python3
"""repo_guard.py -- refuse a DESTRUCTIVE or out-of-tree write before it escapes the repo.

Root cause this exists for (witnessed 2026-06-21): a build script resolved its
output path one level ABOVE the repo root and wrote ``fak.exe`` into a *sibling
git repo* (``work/tools``, the real ``anthony-chaudhary/tools`` project); that
sibling was then ``rm -rf``'d while mistaken for build scratch, destroying it.
Two failure modes, one structural cause -- a tool operating on a path that
resolves OUTSIDE the workspace, into another project's tree.

This is the structural counterpart to the regex-based FAK floor
(``examples/repo-guard-policy.json``): a regex over a command string catches the
``../x`` *relative* escape family but cannot resolve an *absolute* path like
``/c/Users/.../work/tools`` and decide it is a sibling repo. This guard resolves
every destructive/write target against the workspace root and flags the ones that
land outside it (and outside an allow-listed scratch root like the OS temp dir or
``~/.cache``). It is the named floor of ``dos.toml [reasons.OUT_OF_TREE_WRITE]``.

Two surfaces, both backed by the same pure core:
  * ``--check "<cmd>"`` / ``--selftest`` / ``--json``  -- a control-pane / CI lens.
  * ``--hook``  -- a Claude Code **PreToolUse** hook: reads the tool call as JSON on
    stdin and, on a violation, emits a ``permissionDecision: deny`` so the call
    never runs. Fail-OPEN on any internal error (a guard bug must never wedge a
    live multi-session fleet -- it is defense-in-depth behind the FAK policy and
    human review, not the sole gate). Soften with ``FAK_REPO_GUARD=warn`` (log,
    allow) or disable with ``FAK_REPO_GUARD=off``; default is enforce.

Exit code (non-hook): 0 == clean ; 1 == at least one out-of-tree violation.
No third-party deps; cross-platform (handles git-bash ``/c/..`` and ``C:\\..``).
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import PurePosixPath

SCHEMA = "fak-repo-guard/1"
REASON = "OUT_OF_TREE_WRITE"

# Verbs whose path operands are DELETE targets (every non-flag operand).
DELETE_VERBS = frozenset({"rm", "rmdir", "unlink", "shred", "trash", "trash-put"})
# Verbs whose LAST non-flag operand is a WRITE destination.
DEST_VERBS = frozenset({"cp", "mv", "install", "rsync", "ln"})
# Verbs whose every non-flag operand is a WRITE target.
WRITE_VERBS = frozenset({"tee", "truncate"})
# ``--output``/``-out`` almost always name a real output file (tar, objcopy, ...).
# ``-o`` is overloaded (go/gcc = output file, but grep -o = only-matching, sort -o,
# ...), so it counts as an output flag ONLY for a build/compile verb.
OUTPUT_FLAGS = frozenset({"--output", "-out"})
BUILD_VERBS = frozenset({"go", "gcc", "g++", "cc", "clang", "clang++", "ld", "rustc", "gccgo", "tcc", "zig"})
# Segment separators that start a fresh simple-command.
_SEPARATORS = frozenset({"|", "||", "&&", ";", "&", "\n"})


# --------------------------------------------------------------------------- #
# Path normalization -- the load-bearing primitive (git-bash + Windows aware)
# --------------------------------------------------------------------------- #
def normalize(path: str) -> str:
    """Normalize a path string to forward-slash form with an upper-case drive.

    Handles the three forms an agent command mixes on Windows: POSIX
    (``/c/Users/x`` from git-bash/MSYS), Windows (``C:\\Users\\x``), and plain
    relative (``../tools``). Returns a normalized *string* (not resolved against
    the filesystem -- this is pure and testable). MSYS ``/c/`` becomes ``C:/``.
    """
    p = path.strip().strip('"').strip("'").replace("\\", "/")
    # MSYS drive form: /c/Users/... -> C:/Users/...
    if len(p) >= 3 and p[0] == "/" and p[2] == "/" and p[1].isalpha():
        p = p[1].upper() + ":" + p[2:]
    # Upper-case a leading drive letter: c:/x -> C:/x
    if len(p) >= 2 and p[1] == ":" and p[0].isalpha():
        p = p[0].upper() + p[1:]
    return p


def _is_absolute(p: str) -> bool:
    return p.startswith("/") or (len(p) >= 2 and p[1] == ":")


def to_abs(raw: str, base: str) -> str | None:
    """Resolve ``raw`` (a path operand) to a normalized absolute path string,
    relative to normalized ``base``. Returns None for an UNRESOLVABLE target
    (shell variable / glob / command substitution) so the caller can fall back
    to a conservative textual check instead of guessing."""
    if not raw or any(ch in raw for ch in ("$", "*", "?", "`", "~")):
        # ~ could be home, but mixing it with destructive ops is rare; treat as
        # unresolvable and let the textual backstop decide. $/glob are genuinely
        # unknowable here.
        return None
    n = normalize(raw)
    base = normalize(base)
    joined = n if _is_absolute(n) else f"{base.rstrip('/')}/{n}"
    # Collapse . and .. without touching the filesystem (PurePosixPath keeps
    # leading .. but our inputs resolve to absolute, so os.path.normpath is safe).
    parts: list[str] = []
    for seg in joined.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if parts and parts[-1] != "..":
                parts.pop()
            continue
        parts.append(seg)
    # Preserve drive (C:) or leading slash.
    if joined[1:2] == ":":
        return parts[0] + "/" + "/".join(parts[1:]) if len(parts) > 1 else parts[0] + "/"
    return "/" + "/".join(parts)


def is_under(child: str, parent: str) -> bool:
    """True iff normalized-absolute ``child`` is ``parent`` or below it."""
    if not child or not parent:
        return False
    c = PurePosixPath(child)
    p = PurePosixPath(parent)
    try:
        return c == p or p in c.parents
    except (ValueError, TypeError):
        return False


# --------------------------------------------------------------------------- #
# Pure core: extract write/delete targets from a command, classify each
# --------------------------------------------------------------------------- #
def _split_segments(command: str) -> list[str]:
    """Split a compound command into simple-command segments on ; | && || & and
    newlines, without a full parser. Best-effort: good enough to find the verb +
    operands of each piece for a destructive-op scan."""
    out, cur = [], []
    i, n = 0, len(command)
    while i < n:
        two = command[i : i + 2]
        if two in ("||", "&&"):
            out.append("".join(cur))
            cur = []
            i += 2
            continue
        ch = command[i]
        if ch in (";", "|", "&", "\n"):
            out.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(ch)
        i += 1
    out.append("".join(cur))
    return [s for s in out if s.strip()]


def extract_targets(command: str) -> list[tuple[str, str]]:
    """Return ``[(op, raw_path), ...]`` -- the write/delete targets a command
    acts on. ``op`` is the verb (or 'redirect'/'output-flag'). Redirections
    (``> f``, ``>> f``) are scanned on the raw text; everything else via a
    lenient shlex tokenization of each segment."""
    targets: list[tuple[str, str]] = []
    for seg in _split_segments(command):
        # Redirections: > file / >> file (skip >&2, >/dev/null is handled by
        # the scratch allow-list downstream).
        toks_raw = seg.split()
        for j, t in enumerate(toks_raw):
            if t in (">", ">>") and j + 1 < len(toks_raw):
                targets.append(("redirect", toks_raw[j + 1]))
            elif t.startswith(">") and len(t) > 1 and not t.startswith(">&"):
                targets.append(("redirect", t.lstrip(">")))
        try:
            toks = shlex.split(seg, posix=True)
        except ValueError:
            toks = toks_raw
        if not toks:
            continue
        verb = os.path.basename(toks[0])
        operands = toks[1:]
        # -o / --output PATH  (and --output=PATH). `-o` only for build verbs.
        for k, t in enumerate(operands):
            if t in OUTPUT_FLAGS and k + 1 < len(operands):
                targets.append(("output-flag", operands[k + 1]))
            elif t == "-o" and verb in BUILD_VERBS and k + 1 < len(operands):
                targets.append(("output-flag", operands[k + 1]))
            elif t.startswith("--output="):
                targets.append(("output-flag", t.split("=", 1)[1]))
            elif t.startswith("of=") and verb == "dd":
                targets.append(("dd", t.split("=", 1)[1]))
        non_flags = [t for t in operands if not t.startswith("-")]
        if verb in DELETE_VERBS:
            targets += [(verb, t) for t in non_flags]
        elif verb in WRITE_VERBS:
            targets += [(verb, t) for t in non_flags]
        elif verb in DEST_VERBS and non_flags:
            targets.append((verb, non_flags[-1]))  # destination is the last operand
    return targets


def _textual_escape(raw: str) -> bool:
    """A conservative escape signal for an UNRESOLVABLE target: it literally
    starts a parent traversal or names an absolute path. (Used only when to_abs
    cannot resolve, so a ``$VAR/../x`` still trips but an in-repo glob does not.)"""
    n = normalize(raw)
    return n.startswith("../") or "/../" in n or _is_absolute(n)


def classify_command(
    command: str,
    *,
    workspace_root: str,
    safe_roots: tuple[str, ...] = (),
) -> list[dict]:
    """Return a list of violation dicts for destructive/write targets that escape
    the workspace into a non-scratch location. Pure: no filesystem access."""
    ws = normalize(workspace_root)
    safe = tuple(normalize(s) for s in safe_roots)
    violations: list[dict] = []
    for op, raw in extract_targets(command):
        abs_target = to_abs(raw, ws)
        if abs_target is None:
            if _textual_escape(raw):
                violations.append(_violation(op, raw, "<unresolved>", "parent/absolute escape"))
            continue
        if is_under(abs_target, ws):
            continue  # in-repo: fine
        if any(is_under(abs_target, s) for s in safe):
            continue  # scratch (tmp / ~/.cache / ...): fine
        why = "sibling of workspace" if _is_sibling(abs_target, ws) else "outside workspace"
        violations.append(_violation(op, raw, abs_target, why))
    return violations


def classify_write_path(
    file_path: str,
    *,
    workspace_root: str,
    safe_roots: tuple[str, ...] = (),
) -> list[dict]:
    """Same idea for a Write/Edit/NotebookEdit ``file_path``."""
    ws = normalize(workspace_root)
    abs_target = to_abs(file_path, ws)
    if abs_target is None:
        return (
            [_violation("write", file_path, "<unresolved>", "parent/absolute escape")]
            if _textual_escape(file_path)
            else []
        )
    if is_under(abs_target, ws):
        return []
    if any(is_under(abs_target, normalize(s)) for s in safe_roots):
        return []
    why = "sibling of workspace" if _is_sibling(abs_target, ws) else "outside workspace"
    return [_violation("write", file_path, abs_target, why)]


def _is_sibling(abs_target: str, ws: str) -> bool:
    parent = str(PurePosixPath(ws).parent)
    return is_under(abs_target, parent) and not is_under(abs_target, ws)


def _violation(op: str, raw: str, resolved: str, why: str) -> dict:
    return {"reason": REASON, "op": op, "target": raw, "resolved": resolved, "why": why}


# --------------------------------------------------------------------------- #
# Context (workspace root + scratch allow-list) -- minimal, no spawns
# --------------------------------------------------------------------------- #
def find_repo_root(start: str) -> str:
    """Walk up from ``start`` to the nearest dir containing ``.git``; fall back to
    ``start`` itself. No subprocess (the perf-sensitive hook path stays a stat
    walk, not a ``git`` spawn)."""
    cur = PurePosixPath(normalize(start))
    for cand in [cur, *cur.parents]:
        try:
            if os.path.exists(os.path.join(str(cand), ".git")):
                return str(cand)
        except OSError:
            break
    return str(cur)


def _is_agent_state_dir(name: str) -> bool:
    """True iff a home-level entry is the agent's own Claude Code state/memory
    tree: the canonical ``.claude`` or a per-account variant ``.claude-<acct>`` /
    ``.claude.<x>`` (e.g. ``.claude-gem8-netra``, ``.claude.json``). Deliberately
    a STRUCTURED match, not a loose prefix: ``.claudex`` is some other directory,
    not the agent's tree, and must NOT be admitted."""
    return name == ".claude" or name.startswith(".claude-") or name.startswith(".claude.")


def agent_state_roots(home: str, entries: list[str] | None = None) -> list[str]:
    """The agent's own state/memory trees under ``home`` — canonical ``~/.claude``
    plus any per-account variant present (``~/.claude-<acct>``). Writing there is
    the agent persisting its own memory/plans/settings, never a cross-project leak
    (the failure mode this guard exists to catch); without it the hook blocks the
    agent from writing its own memory. ``entries`` is injectable for tests; the
    default is a cheap stat-walk listing of ``home`` (no subprocess). The canonical
    ``~/.claude`` is always included even if absent on disk."""
    roots = [f"{home}/.claude"]
    if entries is None:
        try:
            entries = os.listdir(home)
        except OSError:
            entries = []
    for name in sorted(entries):
        if _is_agent_state_dir(name):
            roots.append(f"{home}/{name}")
    return list(dict.fromkeys(roots))


def private_companion_roots(workspace_root: str) -> tuple[str, ...]:
    """The workspace's OWN private companion repo — the same-named ``<ws>-private``
    sibling (``fak`` -> ``fak-private``): the agent's durable private memory/notes
    store. Bounded ON PURPOSE — ONLY the same-named ``-private`` sibling is admitted,
    never an arbitrary sibling project (the ``OUT_OF_TREE_WRITE`` incident: a write
    into the unrelated ``work/tools`` repo). A look-alike like ``fak-private-evil``
    is a different path component and is NOT admitted."""
    ws = normalize(workspace_root).rstrip("/")
    if not ws:
        return ()
    return (ws + "-private",)


def default_safe_roots() -> tuple[str, ...]:
    home = normalize(os.path.expanduser("~"))
    roots = [
        "/tmp",
        "/var/tmp",
        f"{home}/.cache",
        f"{home}/Downloads",
    ]
    # The agent's own state/memory tree(s): ~/.claude AND per-account variants like
    # ~/.claude-gem8-netra (see agent_state_roots). The workspace's <ws>-private
    # companion is added at the call sites that know the workspace root (run_hook /
    # --check), since it is workspace- not home-relative.
    roots.extend(agent_state_roots(home))
    for var in ("TMPDIR", "TEMP", "TMP"):
        v = os.environ.get(var)
        if v:
            roots.append(normalize(v))
    return tuple(dict.fromkeys(roots))  # de-dup, preserve order


# --------------------------------------------------------------------------- #
# Evaluate one tool call (used by both --check and --hook)
# --------------------------------------------------------------------------- #
def evaluate(tool_name: str, tool_input: dict, *, workspace_root: str, safe_roots: tuple[str, ...]) -> list[dict]:
    if tool_name == "Bash":
        cmd = str(tool_input.get("command") or "")
        return classify_command(cmd, workspace_root=workspace_root, safe_roots=safe_roots)
    if tool_name in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
        fp = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
        return classify_write_path(str(fp), workspace_root=workspace_root, safe_roots=safe_roots)
    return []


def render_reason(violations: list[dict]) -> str:
    parts = [f"{v['op']} -> {v['target']} ({v['why']}: {v['resolved']})" for v in violations]
    return (
        f"{REASON}: a destructive/write op targets a path OUTSIDE this repo. "
        + "; ".join(parts)
        + ". Operate inside the workspace, or write scratch to a temp dir. "
        "If this is intentional, re-run with FAK_REPO_GUARD=warn (advisory) or off."
    )


# --------------------------------------------------------------------------- #
# Hook mode (Claude Code PreToolUse)
# --------------------------------------------------------------------------- #
def run_hook(stdin_text: str) -> int:
    """Parse a PreToolUse payload, emit a deny decision on a violation. Fail-open
    on any error (defense-in-depth must never wedge the fleet)."""
    mode = (os.environ.get("FAK_REPO_GUARD") or "enforce").strip().lower()
    if mode == "off":
        return 0
    try:
        payload = json.loads(stdin_text or "{}")
        tool_name = payload.get("tool_name") or ""
        tool_input = payload.get("tool_input") or {}
        cwd = payload.get("cwd") or os.getcwd()
        workspace_root = find_repo_root(cwd)
        safe_roots = default_safe_roots() + private_companion_roots(workspace_root)
        violations = evaluate(
            tool_name, tool_input, workspace_root=workspace_root, safe_roots=safe_roots
        )
    except Exception as exc:  # noqa: BLE001 -- fail-open is deliberate here
        print(f"repo_guard: internal error, allowing ({exc})", file=sys.stderr)
        return 0
    if not violations:
        return 0
    reason = render_reason(violations)
    if mode == "warn":
        print(f"repo_guard (advisory): {reason}", file=sys.stderr)
        return 0
    # enforce: deny via the PreToolUse decision protocol.
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }))
    print(f"repo_guard: DENY {reason}", file=sys.stderr)
    return 0


# --------------------------------------------------------------------------- #
# Self-test (runnable without pytest) + CLI
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    WS = "C:/Users/u/work/fak"
    HOME = "C:/Users/u"
    SAFE = (
        "/tmp", "/var/tmp", "C:/Users/u/.cache", "C:/Users/u/Downloads",
        # agent state trees (incl. the per-account variant) + the <ws>-private
        # companion — the exact roots the production hook composes.
        *agent_state_roots(HOME, entries=[".claude", ".claude-gem8-netra", ".claudex", "Documents"]),
        *private_companion_roots(WS),
    )
    deny = [  # (tool, input) that MUST produce >=1 violation
        ("Bash", {"command": "go build -o ../tools/.bin/fak.exe ./cmd/fak"}),
        ("Bash", {"command": "rm -rf ../tools"}),
        ("Bash", {"command": "rm -rf /c/Users/u/work/tools"}),
        ("Bash", {"command": "echo x > ../tools/y"}),
        ("Bash", {"command": "cp a.txt ../tools/b.txt"}),
        ("Bash", {"command": "mv internal/x ../sibling/x"}),
        ("Bash", {"command": "rm -rf /"}),
        ("Bash", {"command": "cd src && rm -rf ../../other"}),
        ("Write", {"file_path": "../tools/poison.txt"}),
        ("Write", {"file_path": "C:/Users/u/work/tools/poison.txt"}),
        # the broadened allow-list must NOT leak: a private-companion look-alike,
        # an unrelated sibling, and a .claude look-alike all still DENY.
        ("Write", {"file_path": "C:/Users/u/work/fak-private-evil/x.md"}),
        ("Write", {"file_path": "C:/Users/u/work/fak-ci/x.md"}),
        ("Write", {"file_path": "C:/Users/u/.claudex/leak.md"}),
    ]
    allow = [  # (tool, input) that MUST produce ZERO violations
        ("Bash", {"command": "go build -o fak.exe ./cmd/fak"}),
        ("Bash", {"command": "go build -o tools/.bin/fak.exe ./cmd/fak"}),
        ("Bash", {"command": "rm -rf ./build"}),
        ("Bash", {"command": "rm -rf internal/model/.cache"}),
        ("Bash", {"command": "echo x > /tmp/log.txt"}),
        ("Bash", {"command": "cp a.txt /var/tmp/b.txt"}),
        ("Bash", {"command": "cp a.txt ~/.cache/b.txt"}),
        ("Bash", {"command": "grep -o ../foo internal/policy/x.go"}),  # read, no write verb
        ("Bash", {"command": "cat ../README.md"}),
        ("Bash", {"command": "mv internal/a internal/b"}),
        ("Write", {"file_path": "internal/policy/x.go"}),
        ("Write", {"file_path": "examples/repo-guard-policy.json"}),
        # the agent's own state/memory tree (per-account variant) + private companion.
        ("Write", {"file_path": "C:/Users/u/.claude-gem8-netra/projects/C--Users-u-work-fak/memory/note.md"}),
        ("Write", {"file_path": "C:/Users/u/work/fak-private/MEMORY-glm52-2026-06-21.md"}),
    ]
    fails = 0
    for tool, ti in deny:
        v = evaluate(tool, ti, workspace_root=WS, safe_roots=SAFE)
        if not v:
            print(f"  FAIL (expected DENY, got allow): {tool} {ti}")
            fails += 1
    for tool, ti in allow:
        v = evaluate(tool, ti, workspace_root=WS, safe_roots=SAFE)
        if v:
            print(f"  FAIL (expected ALLOW, got {v}): {tool} {ti}")
            fails += 1
    total = len(deny) + len(allow)
    print(f"repo_guard selftest: {total - fails}/{total} passed ({len(deny)} deny, {len(allow)} allow)")
    return 1 if fails else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Refuse destructive/out-of-tree writes before they escape the repo.")
    ap.add_argument("--hook", action="store_true", help="Claude Code PreToolUse hook mode (reads JSON on stdin)")
    ap.add_argument("--selftest", action="store_true", help="run the built-in case table and exit")
    ap.add_argument("--check", metavar="CMD", default="", help="classify a single Bash command and report")
    ap.add_argument("--workspace", default="", help="workspace root (default: nearest .git above cwd)")
    ap.add_argument("--json", action="store_true", help="machine-readable output for --check")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()
    if args.hook:
        return run_hook(sys.stdin.read())

    ws = find_repo_root(args.workspace or os.getcwd())
    if args.check:
        safe_roots = default_safe_roots() + private_companion_roots(ws)
        violations = classify_command(args.check, workspace_root=ws, safe_roots=safe_roots)
        payload = {"schema": SCHEMA, "ok": not violations, "workspace": ws, "violations": violations}
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            if violations:
                print(f"DENY  {render_reason(violations)}")
            else:
                print(f"ALLOW  no out-of-tree write in: {args.check}")
        return 1 if violations else 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
