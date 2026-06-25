#!/usr/bin/env python3
"""Canonical task identity for an agent session — the kernel-owned primitive (#618).

The fleet keys session dedup, cover-detection, and `dos_verify` on a *task
identity*: a stable signature derived from the session's real first directive
(`hash(project, cwd, directive)`). The hazard this module exists to remove is
that the directive used to be re-derived from raw transcript text at each call
site, where a harness wrapper the derivation did not anticipate would silently
re-collapse every session to one identity.

On 2026-06-24 exactly that happened: the harness opens a slash-command session
with a `<local-command-caveat>Caveat: ...` block and an `/effort ultracode`
preamble that are BYTE-IDENTICAL across every such session, with the real
`/goal` directive two records later. 15 distinct `/goal` workers collapsed to
ONE signature, and the dedup pre-pass DEFERed 13 genuinely-crashed workers as
phantom duplicates.

This module is the single source of truth for that derivation. It is
deliberately standalone (no project imports) so every consumer can route
through it instead of re-deriving the directive from transcript text — and so a
future wrapper added by the harness is fixed in ONE place, not N. It is robust,
by construction, to harness boilerplate:

  * the whole ``<local-command-*`` wrapper family (caveat, stdout, ...), plus
    `<system-reminder>`, `<command-*>` envelopes, memory blocks, and the bare
    "Caveat:" line, are skipped (see ``WRAPPER_RE``);
  * config-command preambles whose args are fleet-identical (`/effort
    ultracode`, `/model ...`) do NOT contribute to the identity — only a
    *task-defining* command's argument does (see ``TASK_CMD_RE``);
  * the synthetic resume prompt a re-home injects is skipped (see
    ``RESUME_PROMPT_PREFIX``).

Public API
----------
``canonical_directive(head_records)`` — the real first directive string.
``signature(project, cwd, directive)`` — the 16-hex sig of a directive.
``task_identity(project, cwd, head_records)`` — directive + sig in one call.

The derivation is a faithful, independently-tested extraction of the logic that
shipped in ``fleet_sessions.py`` (commit 6875beb, "skip <local-command-caveat>
wrapper"); ``task_identity_test.py`` asserts byte-for-byte parity with it, so
routing ``fleet_sessions`` (and any other consumer) through this primitive is a
provably behavior-preserving change.
"""
from __future__ import annotations

import hashlib
import re

# Wrapper text that is NOT the session's real task instruction. The harness injects
# these ahead of (or around) the operator's actual first message; the first head
# record whose text matches none of them, once stripped, is the task identity.
# ``<local-command-`` (not just ``<local-command-stdout>``) matches the WHOLE
# local-command-* family, so the byte-identical ``<local-command-caveat>Caveat: ...``
# block the harness opens every slash-command session with can never be mistaken for
# the real directive. A future ``<local-command-foo>`` wrapper is skipped for free.
WRAPPER_RE = re.compile(
    r"^\s*(?:Caveat:|<system-reminder|<command-name>|<command-message>|"
    r"<command-args>|<local-command-|<user-memory|Codebase and user instructions)",
    re.I)

# Slash commands whose ARGUMENT defines the session's task. /effort, /model, etc.
# only CONFIGURE the session and carry fleet-identical args ("ultracode"), so
# capturing their command-args would re-collapse distinct workers; only these
# task-defining commands contribute their payload to the identity.
TASK_CMD_RE = re.compile(r"<command-name>\s*/(goal|loop|dispatch|fanout|next-up)\b", re.I)

# A re-homed transcript opens with this exact synthetic resume prompt. It is
# identical across every re-home, so it must NOT be treated as a task instruction --
# otherwise every re-homed session in a project collapses to one signature.
RESUME_PROMPT_PREFIX = "Resume where you left off"

# Cap the directive contribution to the hash so a runaway prompt can't dominate it;
# kept identical to the historical fleet_sessions._task_sig bound for parity.
_DIRECTIVE_HASH_CAP = 400


def text_of(content) -> str:
    """Flatten a transcript message ``content`` field to its plain text.

    ``content`` is either a bare string or a list of content blocks (text /
    tool_result / ...); only the human-readable text is kept. Matches the
    ``fleet_sessions.text_of`` extraction so the two see identical directive text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for b in content:
            if isinstance(b, dict):
                if b.get("type") == "text":
                    out.append(b.get("text", ""))
                elif b.get("type") == "tool_result":
                    c = b.get("content")
                    out.append(c if isinstance(c, str) else text_of(c))
        return " ".join(x for x in out if x)
    return ""


def canonical_directive(head_records) -> str:
    """The session's real first task directive, robust to harness boilerplate.

    Walk the parsed head records in order; return the first user/system text that
    is a genuine instruction -- skipping harness wrappers (caveat / system-reminder
    / command-* / local-command-* / memory blocks) and the fixed resume prompt a
    re-home injects. A ``/goal``/``/loop`` directive's ARGUMENT is the truest task
    identity, so when a TASK-DEFINING command (/goal,/loop,/dispatch,/fanout,/next-up)
    carries a command-args payload it is prepended; a config command's args (e.g.
    ``/effort ultracode``) are ignored, since they are identical fleet-wide and would
    re-collapse distinct workers. Returns a normalized, whitespace-collapsed string
    (``""`` when the head is nothing but boilerplate)."""
    cmd_args = None
    for ho in head_records:
        if ho.get("type") not in ("user", "system"):
            continue
        mc = (ho.get("message") or {}).get("content", ho.get("content", ""))
        txt = mc if isinstance(mc, str) else text_of(mc)
        if not txt or not txt.strip():
            continue
        # capture a /goal|/loop|... argument payload (the truest task identity); a config
        # command like /effort carries fleet-identical args, so only task-defining commands
        # contribute -- otherwise every /effort+/goal worker re-collapses to one signature.
        if TASK_CMD_RE.search(txt):
            m = re.search(r"<command-args>(.*?)</command-args>", txt, re.S | re.I)
            if m and m.group(1).strip():
                cmd_args = " ".join(m.group(1).split())
        stripped = txt.strip()
        if WRAPPER_RE.match(stripped):
            continue
        if stripped.startswith(RESUME_PROMPT_PREFIX):
            continue
        return " ".join((cmd_args + " " + stripped).split()) if cmd_args else " ".join(stripped.split())
    return cmd_args or ""


def signature(project: str, cwd: str, directive: str) -> str:
    """Stable 16-hex signature of a task identity. Same (project, cwd, directive)
    across different session ids => the same recurring task => dedup candidates.
    Returns ``""`` for an empty directive (an interactive / no-instruction row,
    which must never dedup against anything)."""
    if not directive:
        return ""
    raw = f"{project}\0{cwd}\0{directive[:_DIRECTIVE_HASH_CAP]}".encode("utf-8", "replace")
    return hashlib.sha256(raw).hexdigest()[:16]


def task_identity(project: str, cwd: str, head_records):
    """The canonical (directive, signature) pair for a session, in one call.

    ``project`` is typically the session-dir basename and ``cwd`` the session's
    working directory; together with the directive they form the dedup key the
    fleet's cover-detection and ``dos_verify`` should all share. Returns a tuple
    ``(directive, sig)`` where an empty directive yields an empty sig."""
    directive = canonical_directive(head_records)
    return directive, signature(project, cwd, directive)
