#!/usr/bin/env python3
"""Issue-contract repair-assist helper — turns a contract-gate HOLD into a
reviewable repair manifest, one row per issue.

`fak issue contract` / `tools/issue_resolve_dispatch.py` (floor
DEFAULT_ISSUE_CONTRACT_MIN_SCORE) refuse to spawn a worker on an issue whose
17-field structured contract (parent_ref, done_condition, acceptance_gate,
work_unit, expected_steps, ...) is incomplete. As of 2026-06-30 that gate holds
the ENTIRE open backlog (0/342 routed via `fak dispatch route --json`) because
the schema was introduced the same commit as the gate itself — the backlog
predates it, not because the issues are badly chosen.

This helper does NOT bypass the gate and does NOT edit GitHub. It classifies
each candidate issue's contract-review reasons into a repair KIND (mirroring
`issueContractRepairKinds` in cmd/fak/issue_contract.go), and for each kind
either:
  - computes a deterministic, zero-authorship-risk fix (`template`: the
    already-built normalized-header repair for a corrupted generation
    marker), or
  - proposes a lane suggestion from the heuristic router (`route`), or
  - lists exactly the missing fields with a one-line human question per
    field (`scope`/`noise`/`split`/`private`/`other`) -- it NEVER invents the
    field content.

Every action's `cmd` is always `None`: applying a proposal (via `gh issue
edit`) stays a manual, operator-approved step, same discipline as
`tools/issue_triage.py`'s review-only actions.

Usage:
    python tools/issue_contract_repair.py --lane docs --limit 10 --json
    python tools/issue_contract_repair.py --limit 50 --markdown --out docs/_audits/issue-contract-repairs-2026-07-01.md
    python tools/issue_contract_repair.py --actions --out docs/_audits/issue-contract-repairs-2026-07-01.json

Exit codes: 0 = ran clean (including "no candidates") · 2 = infra error.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dispatch_worker import install_no_window_subprocess_defaults  # noqa: E402
install_no_window_subprocess_defaults(subprocess)

import issue_resolve_dispatch as ird  # noqa: E402
import issue_lane_router as ilr  # noqa: E402

FETCH_CAP = 500  # generous over the ~342-issue current backlog; not a CLI knob

# ---------------------------------------------------------------------------
# Repair-kind classification -- mirrors cmd/fak/issue_contract.go's
# issueContractRepairKinds()/issueContractRepairAction() exactly (same reason
# constants, same order-preserving dedup, same default->"other" fallback) so
# a Python-side manifest and a `fak issue contract` JSON dump agree on kind.
# ---------------------------------------------------------------------------
_REASON_KIND = {
    "ISSUE_NOT_DISPATCH_LEAF": "split",
    "ISSUE_OVERSIZED_EXPECTED_STEPS": "split",
    "ISSUE_SCOPE_INCOMPLETE": "scope",
    "ISSUE_UNROUTED": "route",
    "ISSUE_LIVE_UNARMORED": "noise",
    "ISSUE_NOISE_CONTROL_INCOMPLETE": "noise",
    "ISSUE_AGENT_CONTEXT_INCOMPLETE": "noise",
    "ISSUE_PRIVATE_BOUNDARY": "private",
    "ISSUE_UNEXPANDED_TEMPLATE": "template",
}
_KIND_RANK = {"split": 1, "scope": 2, "route": 3, "noise": 4, "private": 5,
              "template": 6, "other": 9}
_KIND_ACTION = {
    "split": "decompose each non-leaf or oversized row into child issues within the dispatch step budget",
    "scope": "add the missing parent/current-state/scope/done/witness/closure fields before dispatch",
    "route": "add a lane or path hints section so the issue maps to one dispatch lane",
    "noise": "add trigger, batch policy, agent context, and live dedupe/cap evidence before automated sync",
    "private": "remove private/operator-only evidence or move the work to the private companion repo",
    "template": "dry-run a normalized generated-header repair, review it, then apply explicitly if accepted",
    "other": "inspect the review reasons and repair the row before dispatch",
}
# One-line human question per missing field. Never invents the answer -- this
# is the question a human (or a follow-up agent under human review) answers.
_FIELD_QUESTION = {
    "parent_ref": "What epic/parent issue (if any) does this belong under?",
    "current_state": "What is the current state of the code/system before this change?",
    "why_now": "Why does this need to happen now, not later?",
    "working_spine": "What is the smallest end-to-end path (the working spine) this change moves?",
    "in_scope": "What is explicitly in scope for this issue?",
    "out_of_scope": "What is explicitly out of scope (so a worker doesn't over-reach)?",
    "done_condition": "What observable state means this issue is done?",
    "acceptance_gate": "What check/command proves the done condition is met?",
    "closure_binding": "What commit-message convention binds the closing commit to this issue (e.g. `#N` in the subject)?",
    "work_unit": "What is the single unit of work here (one leaf, one commit)?",
    "expected_steps": "Roughly how many discrete steps should this take (a size estimate)?",
    "assumptions": "What is being assumed that, if wrong, would change the approach?",
    "confusion_risks": "What could a worker misunderstand or conflate here?",
    "coordination": "Does this touch a lane/file another worker might also be editing?",
    "trigger": "What event or condition should cause this to be picked up?",
    "batch_policy": "Should this run standalone or as part of a batch with related issues?",
    "witness": "What evidence (log, diff, test) proves this was actually done?",
}


def repair_kinds(reasons: list[str]) -> list[str]:
    """Order-preserving, deduped reason->kind fold. Empty/unmapped -> ["other"]."""
    kinds: list[str] = []
    for reason in reasons:
        kind = _REASON_KIND.get(reason, "other")
        if kind not in kinds:
            kinds.append(kind)
    return kinds or ["other"]


def primary_kind(kinds: list[str]) -> str:
    return min(kinds, key=lambda k: _KIND_RANK.get(k, 9))


def repair_action(kind: str) -> str:
    return _KIND_ACTION.get(kind, _KIND_ACTION["other"])


def field_scaffold(missing_fields: list[str]) -> list[dict[str, str]]:
    return [{"field": f, "question": _FIELD_QUESTION.get(f, f"What is the missing '{f}'?")}
            for f in missing_fields]


# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------

def fetch_open_issues(workspace: Path, *, fetch_cap: int = FETCH_CAP) -> list[dict[str, Any]]:
    """All open issues (number/title/body/labels), oldest (lowest number) first.
    `gh issue list` returns newest-first; this reverses it on purpose -- same
    ordering decision `issue_resolve_dispatch.lane_issue_numbers` makes."""
    proc = subprocess.run(
        ["gh", "issue", "list", "--state", "open", "--limit", str(fetch_cap),
         "--json", "number,title,body,labels"],
        cwd=str(workspace), capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"gh issue list failed (rc={proc.returncode}): "
                           f"{(proc.stderr or '').strip() or 'no stderr'}")
    issues = json.loads(proc.stdout or "[]")
    return sorted(issues, key=lambda i: int(i.get("number") or 0))


def template_repair_plan(workspace: Path, issue: dict[str, Any],
                         number: int) -> dict[str, Any] | None:
    """Targeted live call for one issue's already-computed template repair
    (`internal/issuecontract.BuildTemplateRepairPlan`, dry-run only). Returns
    the matching `template_repair_plans[]` entry, or None if the Go side found
    nothing to propose."""
    import tempfile
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json",
                                         delete=False) as f:
            tmp = Path(f.name)
            json.dump([ird._issue_record_for_contract(issue, number)], f, ensure_ascii=False)
            f.write("\n")
        cmd = [*ird._fak_command_prefix(workspace), "issue", "contract",
               "--from-issues", str(tmp), "--live", "--json"]
        proc = subprocess.run(
            cmd, cwd=str(workspace), capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=60, creationflags=ird.no_window_creationflags())
    except (OSError, subprocess.SubprocessError):
        return None
    finally:
        if tmp is not None:
            try:
                tmp.unlink()
            except OSError:
                pass
    try:
        doc = json.loads(proc.stdout or "{}")
    except (TypeError, ValueError):
        return None
    for plan in (doc.get("template_repair_plans") or []):
        if isinstance(plan, dict) and int(plan.get("issue_number") or plan.get("number") or -1) == number:
            return plan
    return None


def build_repair_row(workspace: Path, issue: dict[str, Any],
                     concurrent: list[str], trees: dict[str, list[str]]) -> dict[str, Any] | None:
    """One issue -> one manifest row, or None if the contract already passes
    (nothing to repair)."""
    number = int(issue.get("number") or 0)
    contract = ird.issue_contract_review(workspace, issue, number)
    score = int(contract.get("score") or 0)
    if contract.get("ok") and score >= ird.DEFAULT_ISSUE_CONTRACT_MIN_SCORE:
        return None

    review = contract.get("review") if isinstance(contract.get("review"), dict) else {}
    reasons = [str(r) for r in (review.get("reasons") or []) if r]
    missing_fields = [str(m) for m in (review.get("missing_fields") or []) if m]
    kinds = repair_kinds(reasons)
    kind = primary_kind(kinds)

    row: dict[str, Any] = {
        "number": number,
        "title": str(issue.get("title") or "")[:120],
        "score": score,
        "reasons": reasons,
        "kinds": kinds,
        "kind": kind,
        "next_action": repair_action(kind),
        "ready": False,
        "proposed_lane": None,
        "route_confidence": None,
        "proposed_header": None,
        "missing_fields": field_scaffold(missing_fields) if kind in
            ("scope", "noise", "split", "private", "other") else [],
    }

    if "route" in kinds:
        route = ilr.route_issue(issue, concurrent, trees)
        if route.get("lane"):
            row["proposed_lane"] = route["lane"]
            row["route_confidence"] = route.get("confidence")

    if "template" in kinds:
        plan = template_repair_plan(workspace, issue, number)
        header = str((plan or {}).get("proposed_normalized_header") or "").strip()
        if header:
            row["ready"] = True
            row["proposed_header"] = header

    return row


# ---------------------------------------------------------------------------
# Manifest + rendering
# ---------------------------------------------------------------------------

def build_manifest(workspace: Path, *, lane: str | None, limit: int,
                   as_of: str) -> dict[str, Any]:
    concurrent, trees = ilr.lane_taxonomy(workspace)
    issues = fetch_open_issues(workspace)

    if lane:
        filtered = []
        for issue in issues:
            route = ilr.route_issue(issue, concurrent, trees)
            if route.get("lane") == lane or route.get("blocked_lane") == lane:
                filtered.append(issue)
        issues = filtered

    issues = issues[:limit]

    rows: list[dict[str, Any]] = []
    for issue in issues:
        row = build_repair_row(workspace, issue, concurrent, trees)
        if row is not None:
            rows.append(row)

    by_kind: dict[str, int] = {}
    for r in rows:
        by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1

    return {
        "schema": "fak.issue-contract-repair.v1",
        "as_of": as_of,
        "workspace": str(workspace),
        "lane": lane,
        "limit": limit,
        "counts": {
            "candidates_examined": len(issues),
            "needs_repair": len(rows),
            "ready": sum(1 for r in rows if r["ready"]),
            "by_kind": by_kind,
        },
        "issues": rows,
    }


def build_actions(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Lean, review-only action list. `cmd` is ALWAYS None -- this tool never
    emits a `gh issue edit/comment/close/assign` command; applying a proposal
    stays a manual, operator-approved step."""
    return [{
        "number": r["number"],
        "kind": r["kind"],
        "ready": r["ready"],
        "reason": ", ".join(r["reasons"]) or "issue contract below spawn floor",
        "next_action": r["next_action"],
        "cmd": None,
    } for r in manifest["issues"]]


def render_markdown(manifest: dict[str, Any]) -> str:
    c = manifest["counts"]
    lane = manifest["lane"] or "(all lanes)"
    L = [
        f"# Issue-contract repairs — {manifest['as_of']}",
        "",
        f"**Lane:** {lane}  ·  **examined:** {c['candidates_examined']}  ·  "
        f"**needs repair:** {c['needs_repair']}  ·  **template-ready:** {c['ready']}",
        "",
        "> Read-only pass. Never edits, labels, comments on, or closes an issue. "
        "`template`-kind rows carry a dry-run-computed header fix; every other "
        "kind lists the missing fields as questions for a human/agent to answer "
        "-- content is never invented here.",
        "",
        "## Counts by kind",
        "",
        "| kind | count |",
        "|---|---:|",
    ]
    for kind, n in sorted(c["by_kind"].items(), key=lambda kv: _KIND_RANK.get(kv[0], 9)):
        L.append(f"| {kind} | {n} |")
    L.append("")
    L.append("## Rows")
    L.append("")
    L.append("| # | kind | ready | score | title |")
    L.append("|---|---|---|---:|---|")
    for r in manifest["issues"]:
        L.append(f"| #{r['number']} | {r['kind']} | {r['ready']} | {r['score']} | {r['title']} |")
    L.append("")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Issue-contract repair-assist manifest (read-only).")
    ap.add_argument("--workspace", default=None, help="repo root (default: this tool's repo)")
    ap.add_argument("--lane", default=None, help="restrict to one dispatch lane")
    ap.add_argument("--limit", type=int, default=50,
                    help="max issues to examine, oldest issue number first (default: 50)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--actions", action="store_true")
    ap.add_argument("--out", default=None, help="write output to this path")
    ap.add_argument("--as-of", default=None, help="date stamp (default: today UTC)")
    a = ap.parse_args(argv)

    workspace = Path(a.workspace).resolve() if a.workspace else ird.repo_root()
    as_of = a.as_of or dt.datetime.now(dt.timezone.utc).date().isoformat()

    try:
        manifest = build_manifest(workspace, lane=a.lane, limit=a.limit, as_of=as_of)
    except (RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if a.actions:
        rendered = json.dumps({"as_of": as_of, "actions": build_actions(manifest)}, indent=2)
    elif a.markdown:
        rendered = render_markdown(manifest)
    else:
        rendered = json.dumps(manifest, indent=2)

    if a.out:
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote {p}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
