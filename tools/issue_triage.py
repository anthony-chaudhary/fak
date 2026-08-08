#!/usr/bin/env python3
"""GitHub issue triage + rank + garden helper for fleet — the /issue-triage helper.

READ-ONLY. Fetches OPEN issues via `gh`, classifies each into triage buckets
(missing priority/kind/area/class/milestone, orphaned high-prio, stale, dormant question,
bare), ranks them into a deterministic attention order, and emits:

  --json            the full triage model (stdout)
  --md OUT          a dated human report
  --actions OUT     a proposed-action manifest (gh commands, NOT executed)

It NEVER edits, labels, comments on, or closes an issue. Applying the proposed
actions is the skill's job, gated on operator approval. This mirrors plan-audit's
read-only discipline (docs/_audits) and the release skill's dry-run-first rule:
the helper decides what is *true* and what *could* be done; the operator decides
what *is* done.

Duplicate detection is NOT here. The title-only Jaccard dup-cluster pass that used
to live in this file was ported to Go and now lives in `fak issue dedup` (the
body-aware backlog duplicate census, internal/issuededup) — it clusters
title-divergent/body-similar twins this title-token pass was blind to, and emits
ranked merge proposals with per-pair evidence. Run `fak issue dedup [--json]`.

Label taxonomy is fleet's, baked in as defaults:
  priority  priority/P0 | priority/P1 | priority/P2
  kind      bug | enhancement | documentation | question | performance
              (also tolerated as kind: build, research)
  area      agentic-serving | trust-floor | model-arch | compute | gpu |
              model | substrate | loader | security | dispatch | rsi | licensing
  class     class:frontdoor | class:infra | class:dev  (work-class axis: the
              public release path vs fleet infra vs product dev — SEPARATE from
              area; derived from the lane an issue routes to)
  workflow  in-progress | duplicate | wontfix | invalid | help wanted |
              good first issue
  dependency blocked-by | blocks  (directional pair naming a prerequisite edge
              between two issues; see DEPENDENCY below)
Override thresholds via flags; override label sets via --config (JSON file).

Ranking (the "do next" order) is a transparent integer score, not a model:
  base       priority weight (P0 1000, P1 400, P2 150, none 60)
  +300       P0/P1 with no in-progress AND no assignee  (orphan, claim-needed)
  +40        bug                                          (bugs edge enhancements)
  -20        documentation                                (docs spam sinks)
  +min(idle_days, 90)                                     (stale rises)
  -200       question with idle < Q_IDLE                  (fresh Qs wait for reply)
Ties broken by number, newest first. The score is surfaced in the report so the
ranking is auditable, not a black box.

Usage:
  python tools/issue_triage.py --json
  python tools/issue_triage.py --md docs/_audits/issue-triage-YYYY-MM-DD.md
  python tools/issue_triage.py --actions docs/_audits/issue-actions-YYYY-MM-DD.json
  python tools/issue_triage.py --since-days 30      # only issues updated in last 30d
  python tools/issue_triage.py --scope priority     # filter: priority|kind|area|
                                                    #         orphans|stale|question

Exit codes: 0 = ran clean · 2 = infra error (gh missing / not authed / not a repo).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from dispatch_worker import install_no_window_subprocess_defaults
import sys
from pathlib import Path
install_no_window_subprocess_defaults(subprocess)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# ---- Label taxonomy (fleet defaults; override via --config) -----------------
PRIORITY = {"priority/P0": 1000, "priority/P1": 400, "priority/P2": 150}
KIND = {"bug", "enhancement", "documentation", "question", "performance",
        "build", "research"}
AREA = {"agentic-serving", "trust-floor", "model-arch", "compute", "gpu",
        "model", "substrate", "loader", "security", "dispatch", "rsi",
        "licensing"}
# CLASS is a SEPARATE axis from AREA: area = which subsystem, class = which KIND of
# work (fleet infra vs product dev vs the public front-door/release path). It lets
# an operator hide the CI/dispatch/observability plumbing and dispatch only product
# leaves, or fence the release path. The label is `class:<name>`; the value is
# DERIVED from the lane an issue routes to (issue_lane_router.derive_class) rather
# than hand-picked, so a freshly-triaged issue is born with the right class.
CLASS = {"class:frontdoor", "class:infra", "class:dev"}
WORKFLOW = {"in-progress", "duplicate", "wontfix", "invalid", "help wanted",
            "good first issue"}
# DEPENDENCY is the directional blocker pair (#3563): `blocked-by` points at a
# PREREQUISITE ("this issue waits on #N"), `blocks` at a DEPENDENT ("#N waits on
# this"). Deliberately a directional PAIR and not a bare `blocked` state label —
# a state label records that an issue is stuck without recording what it is stuck
# behind, which is the exact gap that left the backlog's dependency structure
# invisible to a human reader.
#
# Each name is a label FAMILY, not a whole label: the counterpart issue rides in
# the scoped form `blocked-by:#N` / `blocks:#N` (also accepted with `/` or a bare
# number), the same family:value shape the repo already uses for `class:infra`,
# `priority/P0`, and `tier/T0-required`. A bare family label with no `#N` is a
# human marker only and carries no edge.
#
# Scope note: this module DECLARES the taxonomy and recognises the scoped form; no
# reader consumes it yet. `fak blockers` today is post/feed/selfcheck only — it folds
# a `gh issue list` payload for ONE label into a roll-up and has no graph subcommand,
# so nothing currently unions these label edges with #3224's `blocked-by: #N`
# issue-BODY token. Declaring the label side first is what lets the edges accumulate
# on real issues before a reader exists to walk them.
DEPENDENCY = {"blocked-by", "blocks"}
# Bare kind fallback for the "documentation" docs(fak) spam pattern — the
# docs(fak):* issues carry only `documentation`+`enhancement`, no priority, no
# area. They are the canonical "needs-priority + garden" candidate.

# ---- Thresholds (override via flags) ----------------------------------------
STALE_DAYS = 60        # open + idle this long + not in-progress => stale
Q_IDLE_DAYS = 30       # question idle this long => dormant close candidate
LIST_LIMIT = 500       # gh issue list cap


def _load_config(path: str | None) -> None:
    """Override label sets from a JSON file. Schema (all optional):
       {"priority": {"label": weight}, "kind": [...], "area": [...],
        "workflow": [...], "dependency": [...], "stale_days": N,
        "q_idle_days": N}"""
    if not path:
        return
    global PRIORITY, KIND, AREA, CLASS, WORKFLOW, DEPENDENCY
    global STALE_DAYS, Q_IDLE_DAYS
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if "priority" in cfg:
        PRIORITY = {k: int(v) for k, v in cfg["priority"].items()}
    if "kind" in cfg:
        KIND = set(cfg["kind"])
    if "area" in cfg:
        AREA = set(cfg["area"])
    if "class" in cfg:
        CLASS = set(cfg["class"])
    if "workflow" in cfg:
        WORKFLOW = set(cfg["workflow"])
    if "dependency" in cfg:
        DEPENDENCY = set(cfg["dependency"])
    STALE_DAYS = int(cfg.get("stale_days", STALE_DAYS))
    Q_IDLE_DAYS = int(cfg.get("q_idle_days", Q_IDLE_DAYS))


def fetch_issues() -> list[dict]:
    """All open issues via gh. Raises on infra failure (caller maps to exit 2)."""
    fields = "number,title,url,state,labels,createdAt,updatedAt,author,assignees,milestone,comments"
    proc = subprocess.run(
        ["gh", "issue", "list", "--state", "open", "--limit", str(LIST_LIMIT),
         "--json", fields],
        capture_output=True, text=True, encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"gh failed (rc={proc.returncode}): {proc.stderr.strip()}"
                           or "gh returned nonzero with no stderr (not authed? not a repo?)")
    return json.loads(proc.stdout or "[]")


def load_injected_issues(source: str) -> list[dict]:
    """Read issues from a ``gh issue list --json`` array (a file path, or ``-`` for
    stdin) instead of the built-in gh fetch — so a NAMED VIEW can drive triage:
    ``issue_views.py show --view needs-triage --json --fields <triage fields> |
    issue_triage.py --issues -``. issue-views is the default selection surface;
    this lets that same slice feed the ranker instead of always re-fetching the
    whole open backlog.

    Field-tolerant: triage reads each field with a default, so a view that omits
    one degrades gracefully — but ``createdAt`` drives the age/stale score, so pass
    enough ``--fields`` upstream (``createdAt``/``updatedAt``/``assignees`` …) when
    those rungs matter. Raises ValueError on non-array / invalid JSON.
    """
    raw = sys.stdin.read() if source == "-" else Path(source).read_text(encoding="utf-8")
    text = raw.strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"--issues input is not valid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError("--issues input must be a JSON array of gh issue objects")
    return data


def _label_names(issue: dict) -> set[str]:
    return {lab["name"] for lab in issue.get("labels", [])}


def dependency_labels(labels: set[str]) -> list[str]:
    """The DEPENDENCY-family labels an issue wears, sorted (deterministic).

    Matches on the label FAMILY — the part before the `:`/`/` scope separator —
    so both the bare marker (`blocked-by`) and the edge-carrying scoped form
    (`blocked-by:#3224`) are recognised, while a merely similar name that is a
    different label (`blocked-by-human`, the human-block dispatch hold) is NOT.
    Read-only surfacing: this never re-weights the ranking, it just makes the
    blocker edges a human is scanning for visible in the triage model.
    """
    out = []
    for name in labels:
        family = name.split(":", 1)[0].split("/", 1)[0].strip()
        if family in DEPENDENCY:
            out.append(name)
    return sorted(out)


def _days(iso: str, now: dt.datetime) -> int:
    try:
        then = dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return 0
    return max(0, int((now - then).total_seconds() // 86400))


def classify(issue: dict, now: dt.datetime) -> dict:
    """Attach triage tags + a rank score to one issue. Pure + deterministic."""
    labels = _label_names(issue)
    age_days = _days(issue.get("createdAt", ""), now)
    idle_days = _days(issue.get("updatedAt", ""), now)
    assigned = bool(issue.get("assignees"))
    in_prog = "in-progress" in labels
    prio = next((p for p in PRIORITY if p in labels), None)
    prio_weight = PRIORITY.get(prio, 60)
    is_p0p1 = prio in ("priority/P0", "priority/P1")

    tags: list[str] = []
    if not prio:
        tags.append("needs-priority")
    if not (labels & KIND):
        tags.append("needs-kind")
    if not (labels & AREA):
        tags.append("needs-area")
    if not (labels & CLASS):
        tags.append("needs-class")
    if not issue.get("milestone"):
        tags.append("needs-milestone")
    if not labels:
        tags.append("bare")
    if is_p0p1 and not in_prog and not assigned:
        tags.append("orphan")
    if idle_days >= STALE_DAYS and not in_prog:
        tags.append("stale")
    if "question" in labels and idle_days >= Q_IDLE_DAYS:
        tags.append("dormant-question")

    score = prio_weight
    if is_p0p1 and not in_prog and not assigned:
        score += 300
    if "bug" in labels:
        score += 40
    if "documentation" in labels:
        score -= 20
    score += min(idle_days, 90)
    if "question" in labels and idle_days < Q_IDLE_DAYS:
        score -= 200

    return {
        "number": issue["number"],
        "title": issue["title"],
        "url": issue["url"],
        "labels": sorted(labels),
        "author": (issue.get("author") or {}).get("login", "?"),
        "assignees": [a.get("login") for a in issue.get("assignees", [])],
        "milestone": (issue.get("milestone") or {}).get("title"),
        "comments": issue.get("comments", 0),
        "age_days": age_days,
        "idle_days": idle_days,
        "priority": prio,
        "in_progress": in_prog,
        "dependency": dependency_labels(labels),
        "tags": tags,
        "score": int(score),
    }


# ---- Proposed actions (mechanical + defensible only) ------------------------
STALE_MSG = ("Marking stale: no activity for {idle} days and not in-progress. "
             "Reply to keep open, else this will be closed in the next triage pass.")
Q_CLOSE_MSG = ("Closing as dormant: this is a `question` with no activity for "
               "{idle} days. Reopen with new info if it is still live.")
# needs-class is the ONE `needs-*` gap that is NOT an operator judgment call: the
# work-class is DERIVED from the lane an issue routes to (issue_lane_router.
# derive_class), so the lane router can backfill every missing class:* label
# idempotently in one batch. Preview the diff, then commit it:
CLASS_BACKFILL_CMD = (
    "python tools/issue_lane_router.py --apply-labels                     "
    "# DRY-RUN: preview the class:* diff\n"
    "python tools/issue_lane_router.py --apply-labels --apply-labels-write "
    "# commit the class:* labels"
)


def build_actions(rows: list[dict]) -> list[dict]:
    """Only MECHANICAL, defensible actions get a `cmd`. Genuine judgment calls
    (needs-priority, needs-kind, needs-area, …) are surfaced as REVIEW with no
    cmd — the operator decides per-issue. The helper never executes these.

    needs-class is deliberately NOT treated as a judgment call: because the
    work-class is derived from the routed lane, the whole needs-class cohort is
    closed by a single mechanical `backfill-class` action (the lane router's
    idempotent label reconcile), and the tag is stripped from each issue's REVIEW
    reason so a class-only gap never masquerades as needing human judgment."""
    actions = []
    needs_class: list[int] = []
    for r in rows:
        if "needs-class" in r["tags"]:
            needs_class.append(r["number"])
        if "dormant-question" in r["tags"]:
            actions.append({
                "number": r["number"], "kind": "close-dormant-question",
                "reason": f"question, idle {r['idle_days']}d",
                "cmd": (f'gh issue close {r["number"]} --reason "not planned" '
                        f'--comment "{Q_CLOSE_MSG.format(idle=r["idle_days"])}"'),
            })
        elif "stale" in r["tags"] and r["priority"] not in ("priority/P0", "priority/P1"):
            actions.append({
                "number": r["number"], "kind": "mark-stale",
                "reason": f"idle {r['idle_days']}d, not in-progress, not P0/P1",
                "cmd": (f'gh issue edit {r["number"]} --add-label "stale" '
                        f'--add-comment \'{STALE_MSG.format(idle=r["idle_days"])}\''),
            })
        else:
            # Genuine judgment calls only — class is mechanical, so drop it here.
            judgment = [t for t in r["tags"] if t != "needs-class"]
            if judgment:
                actions.append({
                    "number": r["number"], "kind": "review",
                    "reason": ", ".join(judgment),
                    "cmd": None,
                })
    if needs_class:
        # One batch move for the whole cohort — the lane router re-derives the
        # class from the live backlog, so this list is the informational scope.
        actions.append({
            "number": None, "kind": "backfill-class",
            "reason": (f"{len(needs_class)} issue(s) missing class:* — derive + "
                       "backfill via the lane router (mechanical, idempotent)"),
            "issues": sorted(needs_class),
            "cmd": CLASS_BACKFILL_CMD,
        })
    return actions


# ---- Rendering --------------------------------------------------------------
def render_md(report: dict, as_of: str) -> str:
    c = report["counts"]
    L = [
        f"# Issue triage — {as_of}",
        "",
        f"**Open issues:** {c['open']}  ·  needs-priority {c['needs_priority']}  "
        f"·  needs-kind {c['needs_kind']}  ·  needs-area {c['needs_area']}  ·  "
        f"needs-class {c['needs_class']}  ·  "
        f"needs-milestone {c['needs_milestone']}  ·  orphan {c['orphan']}  ·  "
        f"stale {c['stale']}  ·  dormant-Q {c['dormant_question']}  "
        f"·  bare {c['bare']}",
        "",
        "> Read-only pass. The `--actions` manifest proposes only mechanical moves "
        "(mark stale, close dormant questions); priority assignment is a judgment "
        "call surfaced as REVIEW — operator decides. Duplicate clusters live in "
        "`fak issue dedup` (body-aware census), not here.",
        "",
        "## Triage queue (top 25 by score)",
        "",
        "| # | Score | Prio | Tags | Title |",
        "|---|---:|---|---|---|",
    ]
    for r in report["rows"][:25]:
        tags = ", ".join(r["tags"]) or "—"
        L.append(f"| [{r['number']}]({r['url']}) | {r['score']} | "
                 f"{r['priority'] or '—'} | {tags} | {r['title']} |")
    L.append("")

    def bucket(title: str, tag: str, hint: str):
        hits = [r for r in report["rows"] if tag in r["tags"]]
        if not hits:
            return
        L.append(f"## {title} ({len(hits)})")
        L.append("")
        L.append(f"*{hint}*")
        L.append("")
        for r in hits:
            L.append(f"- [#{r['number']}]({r['url']}) — idle {r['idle_days']}d · "
                     f"{r['title']}")
        L.append("")

    bucket("Needs a priority label", "needs-priority",
           "Review and set priority/P0|P1|P2. The single largest triage gap.")
    bucket("Needs a milestone (roadmap placement)", "needs-milestone",
           "Assign a milestone so the issue rolls up to a generation/theme and shows "
           "on the roadmap early — the planning gap that keeps the backlog off the map.")
    bucket("Orphan P0/P1 (no in-progress, no assignee)", "orphan",
           "Highest-leverage: claim or explicitly defer.")
    bucket("Stale (open + idle, not in-progress)", "stale",
           "In the --actions manifest as mark-stale (unless P0/P1).")
    bucket("Dormant questions (close candidates)", "dormant-question",
           "In the --actions manifest as close-dormant-question.")
    bucket("Needs a kind label", "needs-kind",
           "bug / enhancement / documentation / question / performance.")
    bucket("Needs an area label", "needs-area",
           "One of the domain labels (agentic-serving, gpu, model, …).")

    acts = report.get("actions", [])
    if acts:
        mechanical = [a for a in acts if a["cmd"]]
        L.append(f"## Proposed actions ({len(mechanical)} mechanical, "
                 f"{len(acts) - len(mechanical)} review-only)")
        L.append("")
        for a in mechanical:
            if a["number"] is None:
                # Batch action (e.g. backfill-class) — no single issue number;
                # show the reason and the one command to run.
                L.append(f"- **{a['kind']}** — {a['reason']}")
                L.append("  ```")
                for line in a["cmd"].splitlines():
                    L.append(f"  {line}")
                L.append("  ```")
            else:
                L.append(f"- `#{a['number']}` **{a['kind']}** — {a['reason']}")
        L.append("")
        L.append("Apply via the skill's approve-then-run step; commands live in the "
                 "`--actions` JSON, never run by this helper.")
        L.append("")

    return "\n".join(L)


def build_report(issues: list[dict], now: dt.datetime) -> dict:
    rows = sorted(
        (classify(i, now) for i in issues),
        key=lambda r: (-r["score"], -r["number"]),
    )
    actions = build_actions(rows)

    def count(tag: str) -> int:
        return sum(1 for r in rows if tag in r["tags"])

    return {
        "as_of": now.date().isoformat(),
        "counts": {
            "open": len(rows),
            "needs_priority": count("needs-priority"),
            "needs_kind": count("needs-kind"),
            "needs_area": count("needs-area"),
            "needs_class": count("needs-class"),
            "needs_milestone": count("needs-milestone"),
            "orphan": count("orphan"),
            "stale": count("stale"),
            "dormant_question": count("dormant-question"),
            "bare": count("bare"),
        },
        "rows": rows,
        "actions": actions,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GitHub issue triage + rank helper.")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--actions", action="store_true")
    ap.add_argument("--out", default=None, help="write output to this path")
    ap.add_argument("--since-days", type=int, default=None,
                    help="only issues updated in the last N days")
    ap.add_argument("--scope", default=None,
                    help="filter rows by a tag: priority|kind|area|orphans|stale|question")
    ap.add_argument("--config", default=None, help="JSON file overriding label sets/thresholds")
    ap.add_argument("--as-of", default=None, help="date stamp (default: today UTC)")
    ap.add_argument("--issues", default=None, metavar="PATH|-",
                    help="triage a gh-issue-list JSON array from a file or '-' (stdin) "
                         "instead of fetching via gh — lets a named view drive triage; "
                         "pass enough --fields upstream (createdAt for the stale score)")
    a = ap.parse_args(argv)

    _load_config(a.config)

    try:
        issues = load_injected_issues(a.issues) if a.issues else fetch_issues()
    except FileNotFoundError:
        msg = f"--issues file not found: {a.issues}" if a.issues else "`gh` not found on PATH."
        print(f"ERROR: {msg}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    now = dt.datetime.now(dt.timezone.utc)
    if a.as_of:
        now = dt.datetime.fromisoformat(a.as_of + "T00:00:00+00:00")
    if a.since_days is not None:
        issues = [i for i in issues if _days(i.get("updatedAt", ""), now) <= a.since_days]

    report = build_report(issues, now)

    scope_map = {
        "priority": "needs-priority", "kind": "needs-kind", "area": "needs-area",
        "orphans": "orphan", "stale": "stale",
        "question": "dormant-question",
    }
    if a.scope:
        tag = scope_map.get(a.scope)
        if not tag:
            print(f"ERROR: unknown scope '{a.scope}'. One of: {list(scope_map)}",
                  file=sys.stderr)
            return 2
        report["rows"] = [r for r in report["rows"] if tag in r["tags"]]

    as_of = (a.as_of or now.date().isoformat())

    if a.actions:
        out = {"as_of": as_of, "actions": report["actions"]}
        rendered = json.dumps(out, indent=2)
    elif a.markdown:
        rendered = render_md(report, as_of)
    else:
        rendered = json.dumps(report, indent=2)

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
