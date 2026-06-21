#!/usr/bin/env python3
"""Plan-completion audit for fleet — the /plan-audit helper.

Emits the reconciled JSON contract documented in
`.claude/skills/plan-audit/SKILL.md` § "Expected JSON shape". Fleet keeps a
SINGLE plan-state surface — the plan documents themselves (no typed registry, no
live-percent state file) — so there is nothing to reconcile against and `drift`
is honestly always empty. The audit's job here is the completion snapshot, not
drift detection; if fleet ever adds a second surface, extend this to join them.

Completion signal (honest + coarse): fleet's plan docs are freeform — the
`PLAN-…-100-units` doc is a 100-row unit TABLE with no per-row done-checkbox, and
newer public operational plans may use DOS-style numbered markdown headings
(`## 1. ...`). So per plan we report:
  * total_units    — count of `| N | …` unit-table rows plus numbered headings
  * signal         — "shipped-marker" if a shipped/built/done marker is in the
                     header region, else "none" (NOT a per-unit census)
  * percent_complete — 100 on a shipped-marker, else 0 (unknown floored to 0)
The skill surfaces these as a floor, never as a verified per-unit completion.

Usage:
  python tools/plan_audit.py --json
  python tools/plan_audit.py --markdown --out docs/_audits/plan-completion-YYYY-MM-DD.md
  python tools/plan_audit.py --check          # exit 1 on drift (always 0 today)

Exit codes: 0 = clean / no drift · 1 = --check + drift · 2 = infra error.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GLOB = "{PLAN,BUILD}-*.md"

UNIT_ROW_RE = re.compile(r"^\|\s*\d+\s*\|")
UNIT_HEADING_RE = re.compile(r"^#{2,6}\s+\d+(?:\.\d+)*[.)]?(?:\s|[—–-]|$)")
HEADING_RE = re.compile(r"^#\s+(.*)")
# Shipped/built/done markers, scanned only in the header region so a mid-doc
# "shipped" in prose (e.g. a risk table) doesn't false-positive a whole plan.
SHIPPED_RE = re.compile(r"shipped|built|✅|\bdone\b|complete[d]?", re.IGNORECASE)
HEADER_LINES = 60


def _expand_glob(pattern: str) -> list[Path]:
    """Expand a brace glob like {PLAN,BUILD}-*.md (stdlib glob has no brace support)."""
    m = re.search(r"\{([^}]*)\}", pattern)
    patterns = (
        [pattern.replace(m.group(0), alt) for alt in m.group(1).split(",")]
        if m else [pattern]
    )
    seen: dict[str, Path] = {}
    for pat in patterns:
        for hit in glob.glob(str(REPO_ROOT / pat)):
            seen[hit] = Path(hit)
    return sorted(seen.values())


def count_units(lines: list[str]) -> int:
    return sum(1 for l in lines if UNIT_ROW_RE.match(l) or UNIT_HEADING_RE.match(l))


def audit_plan(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    name = next((HEADING_RE.match(l).group(1).strip() for l in lines if HEADING_RE.match(l)), path.stem)
    total_units = count_units(lines)
    header = "\n".join(lines[:HEADER_LINES])
    shipped = bool(SHIPPED_RE.search(header))
    percent = 100 if shipped else 0
    status = "complete" if shipped else "not_started"
    return {
        "id": path.stem,
        "name": name,
        "file": path.name,
        "total_units": total_units,
        "signal": "shipped-marker" if shipped else "none",
        "percent_complete": percent,
        "status": status,
    }


def build_report(plans: list[dict]) -> dict:
    counts = {
        "total_plans": len(plans),
        "complete": sum(1 for p in plans if p["status"] == "complete"),
        "in_progress": sum(1 for p in plans if p["status"] == "in_progress"),
        "not_started": sum(1 for p in plans if p["status"] == "not_started"),
    }
    # plan-weighted: each plan = 1 unit, scaled by reconciled percent.
    pw_pct = round(sum(p["percent_complete"] for p in plans) / max(len(plans), 1), 1)
    # task-weighted: sum of per-plan unit counts (a floor — only plans whose unit
    # rows parse contribute; prose plans report 0 units and don't inflate it).
    with_units = [p for p in plans if p["total_units"] > 0]
    total_units = sum(p["total_units"] for p in with_units)
    done_units = sum(p["total_units"] for p in with_units if p["status"] == "complete")
    tw_pct = round(100 * done_units / total_units, 1) if total_units else 0.0
    return {
        "counts": counts,
        "plans": plans,
        "drift": [],  # single surface — nothing to reconcile against
        "work_units": {
            "plan_weighted": {"pct_complete": pw_pct, "n_plans": len(plans)},
            "task_weighted": {
                "pct_complete": tw_pct,
                "total_units": total_units,
                "done_units": done_units,
                "coverage_plans": len(with_units),
                "coverage_total": len(plans),
            },
        },
    }


def render_markdown(report: dict, as_of: str) -> str:
    c, wu = report["counts"], report["work_units"]
    L = [
        f"# Plan-completion audit — {as_of}",
        "",
        f"**Plans:** {c['total_plans']}  ·  complete {c['complete']}  ·  "
        f"in-progress {c['in_progress']}  ·  not-started {c['not_started']}",
        "",
        "> Fleet keeps a single plan-state surface (the plan docs). Drift is "
        "structurally empty; `percent_complete` is a coarse header-marker signal, "
        "**not** a verified per-unit census.",
        "",
        "## Work units",
        f"- **Plan-weighted:** {wu['plan_weighted']['pct_complete']}% "
        f"over {wu['plan_weighted']['n_plans']} plans",
        f"- **Task-weighted (floor):** {wu['task_weighted']['pct_complete']}% — "
        f"{wu['task_weighted']['done_units']}/{wu['task_weighted']['total_units']} units, "
        f"coverage {wu['task_weighted']['coverage_plans']}/{wu['task_weighted']['coverage_total']} plans",
        "",
        "## Plans",
        "",
        "| Plan | Units | Signal | % | Status |",
        "|---|---:|---|---:|---|",
    ]
    for p in report["plans"]:
        L.append(f"| `{p['file']}` | {p['total_units']} | {p['signal']} | "
                 f"{p['percent_complete']} | {p['status']} |")
    L.append("")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Plan-completion audit for fleet.")
    ap.add_argument("--glob", default=DEFAULT_GLOB, help="plan-doc glob (brace ok)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--out", default=None, help="write rendered output to this path")
    ap.add_argument("--check", action="store_true", help="exit 1 on drift (silent otherwise)")
    ap.add_argument("--as-of", default=None, help="date stamp for the report (default: today UTC)")
    a = ap.parse_args(argv)

    try:
        paths = _expand_glob(a.glob)
    except Exception as exc:  # infra error
        print(f"ERROR: glob failed: {exc}", file=sys.stderr)
        return 2

    if not paths:
        if a.json:
            print(json.dumps({"counts": {"total_plans": 0}, "plans": [], "drift": [], "work_units": {}}, indent=2))
        else:
            print("no plan docs found — nothing to audit")
        return 0

    try:
        plans = [audit_plan(p) for p in paths]
    except Exception as exc:
        print(f"ERROR: audit failed: {exc}", file=sys.stderr)
        return 2
    report = build_report(plans)

    as_of = a.as_of or dt.datetime.now(dt.timezone.utc).date().isoformat()

    if a.check:
        return 1 if report["drift"] else 0

    rendered = render_markdown(report, as_of) if a.markdown else json.dumps(report, indent=2)
    if a.out:
        out_path = Path(a.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote {out_path}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
