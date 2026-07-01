#!/usr/bin/env python3
"""issue-smallness lint — flag an issue body that bundles more than one
unrelated deliverable, so a "30-minute scope" leaf issue stays a leaf.

fleet-400iph filing convention (tools/issue_worker_prompt.py, tools/gate_signal.py,
and every `fleet-400iph[NN]` issue in the backlog) uses a fixed body template:

    ## Goal
    <one sentence>
    ## Scope for one GPT-5.5 session
    - Keep this to one focused edit, report row, fixture, or doc update.
    ...
    ## Done condition
    <one sentence>
    ## Witness
    <one sentence>

The contract each issue makes is "exactly one primary deliverable and one
witness" — a single worker session should be able to finish it. The failure
mode this catches: a filer (human or LLM) pastes THREE unrelated asks into one
issue (e.g. "fix the login bug" + "add a metrics dashboard" + "rewrite the
onboarding docs") — each independently small, but bundled they blow the
30-minute budget and defeat the one-lane-per-worker dispatch model.

Detection is deliberately narrow to stay low-false-positive: it does not try
to parse arbitrary prose for "how many ideas are in here" (that is a modeling
problem, not a lint). It looks at the ``## Goal`` section only — ``## Done
condition`` describes how to tell the SAME goal is finished, not a second
ask, so it is used only as a fallback when an issue has no ``## Goal``
heading at all, and the whole body is the last-resort fallback when neither
heading exists. Within the chosen section it counts DISTINCT top-level
bullet/numbered deliverable lines, plus clauses in a single paragraph joined
by a hard task-separator ("; " or " and then ") that each open with a
distinct action verb. Three or more distinct items is a FAIL (mirrors the
witness fixture below); exactly two is a WARN (small issues sometimes
legitimately pair a fix with its test); zero or one is PASS.

Modes:
  --body-file PATH   lint a single issue body read from a file ('-' = stdin).
  --issue N          fetch issue #N via `gh issue view --json body` and lint it.
  --open             fetch every OPEN issue (gh) and lint each — the dry-run
                     backlog report this lint is wired into.
  --json             machine-readable {schema, verdict, count, items, ...}.

Exit codes: 0 = PASS/WARN (never blocks by itself — advisory), 1 = FAIL present
in a summed run, 2 = infra error (gh missing / bad file / bad issue number).
Read-only: never edits, labels, or closes an issue.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA = "fak-issue-smallness-lint/1"

PASS, WARN, FAIL = "pass", "warn", "fail"

_CREATE_NO_WINDOW = 0x08000000


def _win_creationflags() -> int:
    import os
    return _CREATE_NO_WINDOW if os.name == "nt" else 0


# ---------------------------------------------------------------------------
# Section extraction.
# ---------------------------------------------------------------------------
_HEADING_RE = re.compile(r"^#{1,6}\s*(.+?)\s*$", re.MULTILINE)


def extract_section(body: str, heading_substrings: tuple[str, ...]) -> str | None:
    """Return the text under the first heading whose text contains (case-
    insensitively) any of ``heading_substrings``, up to the next heading (of any
    level) or end of body. ``None`` if no such heading exists."""
    matches = list(_HEADING_RE.finditer(body))
    for i, m in enumerate(matches):
        title = m.group(1).lower()
        if any(sub in title for sub in heading_substrings):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            return body[start:end].strip()
    return None


# A deliverable bullet: a markdown list item (-, *, or N.) at the start of a line.
_BULLET_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+(.+)$", re.MULTILINE)

# Splits a single paragraph into candidate task clauses on a hard separator: a
# semicolon, or " and then "/" then " conjunction chaining a second imperative.
_CLAUSE_SPLIT_RE = re.compile(r";\s*|(?:,\s*)?\band then\b\s*")

# A leading verb token signals a fresh imperative ask, e.g. "Add a...", "Fix
# the...", "Rewrite...". Used only to decide whether a split clause is really a
# distinct deliverable (vs. a subordinate clause of the first).
_LEADING_VERB_RE = re.compile(
    r"^(add|fix|remove|rewrite|refactor|create|build|update|write|implement|"
    r"replace|delete|migrate|design|extend|wire|document|test|lint|register|"
    r"surface|expose|ship)\b",
    re.IGNORECASE,
)


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        key = it.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(it.strip())
    return out


def find_deliverables(section_text: str) -> list[str]:
    """Return the distinct candidate deliverable items in a section's text.

    Two shapes are recognized:
      1. Explicit list items (each bullet/numbered line is one candidate).
      2. A single-paragraph body with clauses hard-separated by ';' or
         ' and then '; a clause counts only if it opens with a recognizable
         imperative verb (so a subordinate clause like "so X still works"
         is not miscounted as a second deliverable).
    """
    bullets = [m.group(1).strip() for m in _BULLET_RE.finditer(section_text)]
    if bullets:
        return _dedupe_keep_order(bullets)

    # No bullets: treat as prose. Split on hard separators and keep clauses
    # that read as their own imperative ask.
    flat = " ".join(line.strip() for line in section_text.splitlines() if line.strip())
    if not flat:
        return []
    clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(flat) if c.strip()]
    if len(clauses) <= 1:
        return [flat] if flat else []
    imperative = [c for c in clauses if _LEADING_VERB_RE.match(c)]
    # Only count the split if at least two clauses independently read as their
    # own imperative — otherwise it's one sentence with a dependent clause.
    if len(imperative) >= 2:
        return _dedupe_keep_order(imperative)
    return [flat]


def lint_body(body: str) -> dict[str, Any]:
    """Lint one issue body. Returns a verdict record (see module docstring).

    Only the ``## Goal`` section is scanned for bundled deliverables: ``## Done
    condition`` describes how to tell the Goal is finished (an acceptance
    check on the SAME deliverable), not an independent ask, so folding it in
    would double-count a single-deliverable issue that phrases its Goal and
    Done condition as two sentences about the same thing. Falls back to
    ``## Done condition`` alone (a filer that skipped ``## Goal``), then the
    whole body when neither heading is present.
    """
    body = body or ""
    goal = extract_section(body, ("goal",))
    done = extract_section(body, ("done condition", "done-condition", "acceptance"))

    if goal is not None:
        items = find_deliverables(goal)
        source = "goal"
    elif done is not None:
        items = find_deliverables(done)
        source = "done"
    else:
        items = find_deliverables(body)
        source = "body"

    count = len(items)
    if count >= 3:
        verdict = FAIL
        reason = f"{count} distinct deliverables found in {source} — split before dispatch"
    elif count == 2:
        verdict = WARN
        reason = f"2 distinct deliverables found in {source} — confirm they are one unit of work"
    else:
        verdict = PASS
        reason = f"{count} deliverable(s) found in {source}"

    return {
        "verdict": verdict,
        "count": count,
        "items": items,
        "section_source": source,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# gh I/O boundary.
# ---------------------------------------------------------------------------
def gh_json(args: list[str], timeout: int = 60) -> Any:
    proc = subprocess.run(["gh", *args], capture_output=True, text=True,
                           encoding="utf-8", timeout=timeout,
                           creationflags=_win_creationflags())
    if proc.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} -> {proc.returncode}: "
                            f"{proc.stderr.strip()[:300]}")
    out = proc.stdout.strip()
    return json.loads(out) if out else None


def fetch_issue_body(number: int) -> str:
    data = gh_json(["issue", "view", str(number), "--json", "body"])
    if not isinstance(data, dict):
        raise RuntimeError(f"issue #{number}: unexpected gh response shape")
    return data.get("body") or ""


def fetch_open_issues(limit: int = 500) -> list[dict[str, Any]]:
    data = gh_json(["issue", "list", "--state", "open", "--limit", str(limit),
                     "--json", "number,title,body"])
    return data or []


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--body-file", help="lint a single body read from a file ('-' = stdin).")
    g.add_argument("--issue", type=int, help="fetch and lint issue #N via gh.")
    g.add_argument("--open", action="store_true",
                    help="fetch and lint every OPEN issue via gh (dry-run backlog report).")
    ap.add_argument("--limit", type=int, default=500,
                     help="max open issues to scan with --open (default 500).")
    ap.add_argument("--json", action="store_true", help="machine-readable output.")
    args = ap.parse_args(argv)

    try:
        if args.body_file is not None:
            text = (sys.stdin.read() if args.body_file == "-"
                     else Path(args.body_file).read_text(encoding="utf-8"))
            result = lint_body(text)
            report = {"schema": SCHEMA, "mode": "body-file", **result}
            fail = result["verdict"] == FAIL
        elif args.issue is not None:
            body = fetch_issue_body(args.issue)
            result = lint_body(body)
            report = {"schema": SCHEMA, "mode": "issue", "issue": args.issue, **result}
            fail = result["verdict"] == FAIL
        else:
            issues = fetch_open_issues(args.limit)
            rows = []
            fail = False
            for it in issues:
                res = lint_body(it.get("body") or "")
                if res["verdict"] == FAIL:
                    fail = True
                rows.append({"number": it.get("number"), "title": it.get("title"),
                             **res})
            counts = {PASS: 0, WARN: 0, FAIL: 0}
            for r in rows:
                counts[r["verdict"]] += 1
            report = {"schema": SCHEMA, "mode": "open", "scanned": len(rows),
                       "counts": counts,
                       "flagged": [r for r in rows if r["verdict"] != PASS]}
    except (OSError, RuntimeError, ValueError) as e:
        print(f"refuse: {e}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1 if fail else 0

    if report["mode"] == "open":
        print(f"issue-smallness-lint — scanned {report['scanned']} open issue(s): "
              f"{report['counts'][PASS]} pass, {report['counts'][WARN]} warn, "
              f"{report['counts'][FAIL]} fail")
        for r in report["flagged"]:
            print(f"  [{r['verdict'].upper():4s}] #{r['number']} {r['title']!r} "
                  f"-> {r['reason']}")
    else:
        label = report.get("issue", report["mode"])
        print(f"issue-smallness-lint [{label}]: {report['verdict'].upper()} — "
              f"{report['reason']}")
        for item in report["items"]:
            print(f"  - {item}")

    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
