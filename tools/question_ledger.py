#!/usr/bin/env python3
r"""Question-loop ledger discipline — the deterministic guardrail for docs/questions/asked.jsonl.

The /question-loop skill's whole value rests on the ledger being TRUSTWORTHY:
unambiguous ids, a closed category vocabulary, a closed status lifecycle, and no
leaked host/path. Prose in a SKILL.md cannot enforce any of that; this tool can.
It is the labeling authority the ASK loop and the NEXT-STEP loop both defer to —
the shell/core split the repo insists on, with the *rules* here in code and the
*question generation* left to the agent.

Subcommands (all take --ledger, default docs/questions/asked.jsonl):

    lint                     validate every row; exit 1 on any violation.
    next-id [--date D]       print the next Q-YYYYMMDD-NNN id for date D (default
                             today UTC), continuing that day's sequence. The ASK
                             loop calls this instead of guessing an id.
    dedupe-check --question  exit 3 if a near-duplicate question already exists.
                 "<text>" [--target "<text>"]
    stats                    JSON counts by category and status.
    ensure-label [--apply]   print (or, with --apply, run) the `gh label create`
                             for the `question-loop` provenance label.

Exit codes: 0 = OK, 1 = lint violation(s), 2 = harness error (unreadable ledger),
3 = dedupe-check found a near-duplicate.

Row schema (exactly these 8 keys — kept in lockstep with docs/questions/README.md):
    id       "Q-YYYYMMDD-NNN"  unique
    ts       ISO-8601 UTC      e.g. 2026-07-08T00:00:00Z
    category UNASKED|AFRAID|CONTRARIAN|STEELMAN
    target   non-empty string  (the claim/area interrogated)
    question non-empty string, ends with '?'
    why      non-empty string
    status   open|answered|ticketed|dismissed
    ticket   null unless status==ticketed, then a positive int
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LEDGER = "docs/questions/asked.jsonl"

CATEGORIES = {"UNASKED", "AFRAID", "CONTRARIAN", "STEELMAN"}
STATUSES = {"open", "answered", "ticketed", "dismissed"}
KEYS = {"id", "ts", "category", "target", "question", "why", "status", "ticket"}

ID_RE = re.compile(r"^Q-(\d{8})-(\d{3})$")
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")

# Public-safe: a question or its rationale must never carry a machine-absolute
# path, a home dir, or an email — the same PUBLIC_LEAK class the fleet guards.
LEAK_RES = [
    re.compile(r"[A-Za-z]:\\"),                 # C:\ style Windows path
    re.compile(r"(?:^|[\s(])/(?:home|Users|root|mnt|tmp)/"),  # POSIX absolute
    re.compile(r"\\\\[A-Za-z0-9._-]+\\"),       # UNC \\host\share
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),  # email
]

LABEL = "question-loop"
LABEL_COLOR = "8a63d2"
LABEL_DESC = "Filed by /question-loop from a witnessed provocative question"


def _norm(s: str) -> str:
    """Coarse normalization for dedupe: lowercase, alnum+space only, collapsed."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", (s or "").lower())).strip()


def read_rows(path: Path):
    """Return [(lineno, obj_or_None, raw)] for every non-blank line."""
    out = []
    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            out.append((i, json.loads(raw), raw))
        except json.JSONDecodeError:
            out.append((i, None, raw))
    return out


def _leak_hit(text: str) -> str | None:
    for rx in LEAK_RES:
        m = rx.search(text or "")
        if m:
            return m.group(0)
    return None


def lint_rows(rows) -> list[str]:
    """Return a list of human-readable violation strings (empty == clean)."""
    problems: list[str] = []
    seen_ids: dict[str, int] = {}
    for lineno, obj, raw in rows:
        tag = f"line {lineno}"
        if obj is None:
            problems.append(f"{tag}: not valid JSON")
            continue
        if not isinstance(obj, dict):
            problems.append(f"{tag}: row is not a JSON object")
            continue
        keys = set(obj.keys())
        missing = KEYS - keys
        extra = keys - KEYS
        if missing:
            problems.append(f"{tag}: missing key(s) {sorted(missing)}")
        if extra:
            problems.append(f"{tag}: unexpected key(s) {sorted(extra)}")
        rid = obj.get("id")
        if isinstance(rid, str):
            tag = f"{rid} (line {lineno})"
            if not ID_RE.match(rid):
                problems.append(f"{tag}: id must match Q-YYYYMMDD-NNN")
            if rid in seen_ids:
                problems.append(f"{tag}: duplicate id (also line {seen_ids[rid]})")
            seen_ids[rid] = lineno
        else:
            problems.append(f"{tag}: id missing or not a string")
        if not TS_RE.match(str(obj.get("ts", ""))):
            problems.append(f"{tag}: ts must be ISO-8601 UTC (…Z)")
        if obj.get("category") not in CATEGORIES:
            problems.append(f"{tag}: category {obj.get('category')!r} not in {sorted(CATEGORIES)}")
        status = obj.get("status")
        if status not in STATUSES:
            problems.append(f"{tag}: status {status!r} not in {sorted(STATUSES)}")
        for field in ("target", "why"):
            if not isinstance(obj.get(field), str) or not obj.get(field, "").strip():
                problems.append(f"{tag}: {field} must be a non-empty string")
        q = obj.get("question")
        if not isinstance(q, str) or not q.strip():
            problems.append(f"{tag}: question must be a non-empty string")
        elif not q.rstrip().endswith("?"):
            problems.append(f"{tag}: question must end with '?'")
        # status/ticket coherence
        ticket = obj.get("ticket")
        if status == "ticketed":
            if not (isinstance(ticket, int) and not isinstance(ticket, bool) and ticket > 0):
                problems.append(f"{tag}: status 'ticketed' requires a positive int ticket, got {ticket!r}")
        else:
            if ticket is not None:
                problems.append(f"{tag}: ticket must be null unless status is 'ticketed' (got {ticket!r})")
        # public-safe
        for field in ("question", "why", "target"):
            hit = _leak_hit(str(obj.get(field, "")))
            if hit:
                problems.append(f"{tag}: {field} leaks {hit!r} (absolute path/host/email)")
    return problems


def next_id(rows, date_str: str) -> str:
    if not re.match(r"^\d{8}$", date_str):
        raise ValueError(f"--date must be YYYYMMDD, got {date_str!r}")
    hi = 0
    for _, obj, _ in rows:
        if not isinstance(obj, dict):
            continue
        m = ID_RE.match(str(obj.get("id", "")))
        if m and m.group(1) == date_str:
            hi = max(hi, int(m.group(2)))
    return f"Q-{date_str}-{hi + 1:03d}"


def dedupe_match(rows, question: str, target: str = "") -> dict | None:
    nq = _norm(question)[:70]
    nt = _norm(target)[:40]
    for _, obj, _ in rows:
        if not isinstance(obj, dict):
            continue
        oq = _norm(obj.get("question", ""))[:70]
        ot = _norm(obj.get("target", ""))[:40]
        if nq and nq == oq:
            return obj
        if nt and nt == ot and nq and oq and (nq in oq or oq in nq):
            return obj
    return None


def stats(rows) -> dict:
    by_cat: dict[str, int] = {}
    by_status: dict[str, int] = {}
    total = 0
    for _, obj, _ in rows:
        if not isinstance(obj, dict):
            continue
        total += 1
        by_cat[obj.get("category", "?")] = by_cat.get(obj.get("category", "?"), 0) + 1
        by_status[obj.get("status", "?")] = by_status.get(obj.get("status", "?"), 0) + 1
    return {"total": total, "by_category": by_cat, "by_status": by_status}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Question-loop ledger discipline.")
    ap.add_argument("cmd", choices=["lint", "next-id", "dedupe-check", "stats", "ensure-label"])
    ap.add_argument("--ledger", default=DEFAULT_LEDGER)
    ap.add_argument("--date", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    ap.add_argument("--question", default="")
    ap.add_argument("--target", default="")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "ensure-label":
        cmd = ["gh", "label", "create", LABEL, "--color", LABEL_COLOR, "--description", LABEL_DESC]
        if args.apply:
            r = subprocess.run(cmd + ["--force"], capture_output=True, text=True)
            sys.stdout.write(r.stdout)
            sys.stderr.write(r.stderr)
            return 0
        print(" ".join(cmd))
        return 0

    path = Path(args.ledger)
    if not path.exists():
        # An absent ledger is empty, not an error — the loop may run before its first append.
        rows: list = []
    else:
        try:
            rows = read_rows(path)
        except OSError as e:
            print(f"question-ledger: cannot read {path}: {e}", file=sys.stderr)
            return 2

    if args.cmd == "lint":
        problems = lint_rows(rows)
        if problems:
            for p in problems:
                print(f"FAIL {p}")
            print(f"\nquestion-ledger: {len(problems)} violation(s) in {path}")
            return 1
        print(f"question-ledger: clean - {stats(rows)['total']} row(s) in {path}")
        return 0

    if args.cmd == "next-id":
        print(next_id(rows, args.date))
        return 0

    if args.cmd == "dedupe-check":
        if not args.question:
            print("dedupe-check requires --question", file=sys.stderr)
            return 2
        hit = dedupe_match(rows, args.question, args.target)
        if hit:
            print(f"DUPLICATE of {hit.get('id')}: {hit.get('question')}")
            return 3
        print("unique")
        return 0

    if args.cmd == "stats":
        print(json.dumps(stats(rows), indent=2))
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
