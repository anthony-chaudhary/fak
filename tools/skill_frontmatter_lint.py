#!/usr/bin/env python3
"""Lint: which Claude skill frontmatter is load-bearing yet lost under opencode? (#422)

Background
----------
opencode's skill loader scans ``**/SKILL.md`` and honors only a small set of
frontmatter fields — ``name, description, license, compatibility, metadata``.
Every other field is silently routed to inert ``options``. So a Claude skill
that uses ``allowed-tools`` for a read-only boundary, or ``disable-model-
invocation`` / ``user-invocable`` for invocation gating, KEEPS LOADING under
opencode but LOSES those semantics: the read-only scope widens to the invoking
agent's full permission, and an operator-only skill becomes model-invocable.

This lint reads every ``SKILL.md``, separates the Claude-only fields from the
opencode-honored ones, and flags the skills where a Claude-only field is
*load-bearing* — i.e. dropping it changes the access/invocation posture:

  * ``read_only_boundary``     — ``allowed-tools`` withholds the mutating tools
                                 (Write/Edit/NotebookEdit); cross-loading widens it.
  * ``model_invocation_gated`` — ``disable-model-invocation: true``; cross-loading
                                 lets the model invoke an operator-only skill.
  * ``not_user_invocable``     — ``user-invocable: false``; an auto-load-only skill
                                 becomes directly invocable.

A skill clears the flag by acknowledging the gap via the opencode-HONORED
``metadata`` field (which DOES survive the cross-load), pointing at the
mitigation it took (issue #422 acceptance (a)/(b)):

  metadata:
    opencode: claude-only       # (b) excluded from the opencode skills.paths scan
    # or
    opencode: agent-permission  # (a) boundary re-expressed on the invoking agent

Read-only; takes no lease; writes nothing.

Run:  python tools/skill_frontmatter_lint.py            # human report
      python tools/skill_frontmatter_lint.py --json     # machine-readable
      python tools/skill_frontmatter_lint.py --check     # exit 1 if any
                                                          # UNacknowledged load-bearing skill
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCHEMA = "fak/skill-frontmatter-lint@1"

# The fields opencode's skill loader actually honors; everything else is dropped.
OPENCODE_HONORED = ("name", "description", "license", "compatibility", "metadata")
# Claude-only frontmatter — silently inert under opencode.
CLAUDE_ONLY = ("allowed-tools", "user-invocable", "disable-model-invocation",
               "argument-hint", "output_root")
# Tools that mutate the tree; an allowlist that omits ALL of them is a read-only scope.
MUTATING_TOOLS = ("Write", "Edit", "NotebookEdit")
# Recognized acknowledgements under metadata.opencode (the field that survives cross-load).
ACK_VALUES = ("claude-only", "agent-permission")


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def _unquote(val: str) -> str:
    val = val.strip()
    if val and val[0] in "\"'":  # quoted scalar — return the inner string verbatim
        end = val.find(val[0], 1)
        return val[1:end] if end != -1 else val[1:]
    cut = val.find(" #")  # unquoted — drop a trailing inline YAML comment
    if cut != -1:
        val = val[:cut].rstrip()
    return "" if val.startswith("#") else val


def parse_frontmatter(text: str) -> dict:
    """Parse the leading ``---`` YAML block — flat keys plus a one-level ``metadata`` map.

    Dependency-free on purpose: SKILL.md frontmatter is shallow, and the lint
    must run anywhere the rest of the python tools do, with or without pyyaml.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    fm: dict = {}
    parent: str | None = None
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0:
            key, sep, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            if not sep:
                continue
            if val == "":
                fm[key] = {}
                parent = key
            else:
                fm[key] = _unquote(val)
                parent = None
        elif parent is not None and isinstance(fm.get(parent), dict):
            key, sep, val = line.strip().partition(":")
            if sep:
                fm[parent][key.strip()] = _unquote(val.strip())
    return fm


def parse_allowed_tools(value) -> list[str]:
    if isinstance(value, list):
        return [str(t).strip() for t in value if str(t).strip()]
    if not isinstance(value, str):
        return []
    raw = value.strip().strip("[]")
    return [t.strip() for t in raw.split(",") if t.strip()]


def acknowledgement(fm: dict) -> str | None:
    """The metadata.opencode mitigation marker, if it names a recognized path."""
    md = fm.get("metadata")
    if isinstance(md, dict):
        val = str(md.get("opencode", "")).strip()
        if val in ACK_VALUES:
            return val
    return None


def classify(fm: dict) -> dict:
    """Decide whether this skill's Claude-only frontmatter is load-bearing."""
    present = [f for f in CLAUDE_ONLY if f in fm]
    reasons: list[str] = []

    if "allowed-tools" in fm:
        tools = parse_allowed_tools(fm["allowed-tools"])
        if tools and not (set(MUTATING_TOOLS) & set(tools)):
            reasons.append("read_only_boundary")
    if str(fm.get("disable-model-invocation", "")).strip().lower() == "true":
        reasons.append("model_invocation_gated")
    if str(fm.get("user-invocable", "")).strip().lower() == "false":
        reasons.append("not_user_invocable")

    ack = acknowledgement(fm)
    load_bearing = bool(reasons)
    flagged = load_bearing and ack is None
    return {
        "name": fm.get("name", ""),
        "claude_only_fields": present,
        "load_bearing": load_bearing,
        "reasons": reasons,
        "acknowledged": ack,
        "flagged": flagged,
    }


def lint_skill(path: Path, root: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:  # noqa: BLE001
        return {"path": str(path), "error": str(exc), "flagged": False,
                "load_bearing": False, "reasons": [], "claude_only_fields": []}
    fm = parse_frontmatter(text)
    rec = classify(fm)
    try:
        rel = str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        rel = str(path)
    rec["path"] = rel
    return rec


def find_skills(root: Path) -> list[Path]:
    base = root / ".claude" / "skills"
    if not base.is_dir():
        return []
    return sorted(base.glob("*/SKILL.md"))


def fold(records: list[dict], root: Path) -> dict:
    flagged = [r for r in records if r.get("flagged")]
    load_bearing = [r for r in records if r.get("load_bearing")]
    acked = [r for r in records if r.get("load_bearing") and r.get("acknowledged")]
    ok = not flagged
    return {
        "schema": SCHEMA,
        "ok": ok,
        "verdict": "OK" if ok else "LOAD_BEARING_UNACKNOWLEDGED",
        "workspace": str(root),
        "taxonomy": {
            "opencode_honored": list(OPENCODE_HONORED),
            "claude_only": list(CLAUDE_ONLY),
            "ack_values": list(ACK_VALUES),
        },
        "n_skills": len(records),
        "n_load_bearing": len(load_bearing),
        "n_acknowledged": len(acked),
        "n_flagged": len(flagged),
        "flagged": flagged,
        "skills": records,
        "next_action": (
            "none — no skill's Claude-only frontmatter is load-bearing-and-unacknowledged"
            if ok else
            "for each flagged skill: port the boundary to the invoking agent's "
            "permission and mark `metadata.opencode: agent-permission`, OR exclude it "
            "from the opencode skills.paths scan and mark `metadata.opencode: claude-only`"
        ),
    }


def render(payload: dict) -> str:
    lines = [
        f"skill-frontmatter-lint: {payload['verdict']}  "
        f"({payload['n_skills']} skills, {payload['n_load_bearing']} load-bearing, "
        f"{payload['n_acknowledged']} acknowledged, {payload['n_flagged']} flagged)",
        "",
        f"opencode honors : {', '.join(payload['taxonomy']['opencode_honored'])}",
        f"claude-only     : {', '.join(payload['taxonomy']['claude_only'])}",
        "",
    ]
    if payload["flagged"]:
        lines.append("FLAGGED — load-bearing Claude-only frontmatter, not acknowledged:")
        for r in payload["flagged"]:
            lines.append(f"  ✗ {r['path']}  [{', '.join(r['reasons'])}]")
        lines.append("")
    acked = [r for r in payload["skills"] if r.get("load_bearing") and r.get("acknowledged")]
    if acked:
        lines.append("acknowledged (mitigated):")
        for r in acked:
            lines.append(f"  ✓ {r['path']}  [{', '.join(r['reasons'])}] -> "
                         f"metadata.opencode: {r['acknowledged']}")
        lines.append("")
    lines.append(payload["next_action"])
    return "\n".join(lines)


def check_gate(payload: dict) -> tuple[int, str]:
    if payload["ok"]:
        return 0, (f"OK: no load-bearing Claude-only frontmatter is unacknowledged "
                   f"({payload['n_load_bearing']} load-bearing, all acknowledged)")
    names = ", ".join(r.get("path") or r.get("name") or "?" for r in payload["flagged"])
    return 1, (f"FAIL: {payload['n_flagged']} skill(s) carry load-bearing Claude-only "
               f"frontmatter that opencode silently drops, unacknowledged: {names}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Flag Claude skills whose Claude-only frontmatter is load-bearing "
                    "and lost when cross-loaded into opencode (#422).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--check", action="store_true",
                    help="gate: exit 1 if any skill is load-bearing-and-unacknowledged")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    records = [lint_skill(p, root) for p in find_skills(root)]
    payload = fold(records, root)

    if args.check:
        code, message = check_gate(payload)
        print(json.dumps({**payload, "gate_exit": code, "gate_message": message}, indent=2)
              if args.json else message)
        return code

    print(json.dumps(payload, indent=2) if args.json else render(payload))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
