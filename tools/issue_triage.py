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
  requires   requires:none | requires:gpu | requires:hardware | requires:metal |
              requires:quota (static hardware/resource capability requirement axis;
              see REQUIRES below)
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
import hashlib
import json
import subprocess
import sys
from pathlib import Path

_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
from dispatch_worker import install_no_window_subprocess_defaults
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
# REQUIRES is the static hardware/resource capability requirement axis (#10965):
# `requires:gpu`, `requires:hardware`, `requires:metal`, `requires:quota`,
# defaulting to unconstrained `requires:none`. Distinct from the transient
# `blocked` status or causal `blocked-by` dependencies — an issue carrying
# `requires:gpu` is not stuck, it simply requires an execution host with that
# capability to run.
REQUIRES = {"requires"}
# Bare kind fallback for the "documentation" docs(fak) spam pattern — the
# docs(fak):* issues carry only `documentation`+`enhancement`, no priority, no
# area. They are the canonical "needs-priority + garden" candidate.

# ---- Thresholds (override via flags) ----------------------------------------
STALE_DAYS = 60        # open + idle this long + not in-progress => stale
Q_IDLE_DAYS = 30       # question idle this long => dormant close candidate
LIST_LIMIT = 100_000   # explicit safety ceiling; ranking refuses below the live total
SNAPSHOT_MAX_AGE_SECONDS = 15 * 60
DEFAULT_REPAIR_BATCH_SIZE = 50
LIVE_SNAPSHOT_ATTEMPTS = 3


class IncompleteRankingError(RuntimeError):
    """The issue snapshot cannot support a complete, current ranking."""


def _load_config(path: str | None) -> None:
    """Override label sets from a JSON file. Schema (all optional):
       {"priority": {"label": weight}, "kind": [...], "area": [...],
        "workflow": [...], "dependency": [...], "requires": [...], "stale_days": N,
        "q_idle_days": N}"""
    if not path:
        return
    global PRIORITY, KIND, AREA, CLASS, WORKFLOW, DEPENDENCY, REQUIRES
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
    if "requires" in cfg:
        REQUIRES = set(cfg["requires"])
    STALE_DAYS = int(cfg.get("stale_days", STALE_DAYS))
    Q_IDLE_DAYS = int(cfg.get("q_idle_days", Q_IDLE_DAYS))


def _run_gh(args: list[str]) -> str:
    proc = subprocess.run(
        ["gh", *args], capture_output=True, text=True, encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"gh failed (rc={proc.returncode}): {proc.stderr.strip()}"
                           or "gh returned nonzero with no stderr (not authed? not a repo?)")
    return proc.stdout or ""


def _resolve_repo(repo: str | None = None) -> str:
    if not repo:
        raw = _run_gh(["repo", "view", "--json", "nameWithOwner"])
        repo = str(json.loads(raw).get("nameWithOwner") or "")
    parts = repo.strip().split("/")
    if len(parts) != 2 or not all(parts):
        raise RuntimeError(f"repository must be owner/name, got {repo!r}")
    return repo


def _fetch_issue_total(repo: str, state: str) -> int:
    owner, name = repo.split("/", 1)
    state_upper = state.upper()
    if state_upper not in {"OPEN", "CLOSED", "ALL"}:
        raise ValueError(f"issue state must be open, closed, or all, got {state!r}")
    issue_args = "" if state_upper == "ALL" else f"(states:{state_upper})"
    query = (
        "query($owner:String!,$name:String!){"
        f"repository(owner:$owner,name:$name){{issues{issue_args}{{totalCount}}}}"
        "}"
    )
    raw = _run_gh([
        "api", "graphql", "-f", f"query={query}",
        "-F", f"owner={owner}", "-F", f"name={name}",
    ])
    try:
        return int(json.loads(raw)["data"]["repository"]["issues"]["totalCount"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"gh issue total response has no integer totalCount: {exc}") from exc


def reconcile_census(*, scope: str, state: str, fetched_count: int,
                     total_count: int, snapshot_age_seconds: int,
                     includes_pull_requests: bool,
                     snapshot_at: str | None = None) -> dict:
    """Type the relationship between one fetched snapshot and its declared scope."""
    scope = scope.strip()
    state = state.strip().lower()
    page_complete = fetched_count == total_count
    reconciliation = "complete"
    if includes_pull_requests:
        reconciliation = "pull_requests_included"
    elif fetched_count < 0 or total_count < 0:
        reconciliation = "count_mismatch"
    elif (scope not in {"repository_issues", "provided_issues"}
          and not scope.startswith("repository_issues:")
          and not scope.startswith("provided_issues:")):
        reconciliation = "scope_mismatch"
    elif snapshot_age_seconds > SNAPSHOT_MAX_AGE_SECONDS:
        reconciliation = "snapshot_stale"
    elif fetched_count < total_count:
        reconciliation = "pagination_truncated"
    elif fetched_count > total_count:
        reconciliation = "count_mismatch"
    out = {
        "scope": scope,
        "state": state,
        "fetched_count": int(fetched_count),
        "total_count": int(total_count),
        "page_complete": page_complete,
        "snapshot_age_seconds": max(0, int(snapshot_age_seconds)),
        "includes_pull_requests": bool(includes_pull_requests),
        "reconciliation": reconciliation,
    }
    if snapshot_at:
        out["snapshot_at"] = snapshot_at
    return out


def _snapshot_id(issues: list[dict]) -> str:
    """Return a stable identity for the exact fetched issue set."""
    identity = sorted(
        (int(issue.get("number", 0)), str(issue.get("updatedAt", "")),
         str(issue.get("state", "")))
        for issue in issues
    )
    encoded = json.dumps(identity, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def fetch_issues(*, repo: str | None = None, state: str = "open",
                 limit: int = LIST_LIMIT,
                 max_attempts: int = LIVE_SNAPSHOT_ATTEMPTS) -> tuple[list[dict], dict]:
    """Fetch one complete issue snapshot, retrying bounded live count races.

    A live attempt is accepted only when the independently queried totals bracketing
    the list fetch are equal and match the decoded issue count. Every attempt keeps
    a digest of the exact issue set, so success and terminal refusal are auditable.
    """
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    repo = _resolve_repo(repo)
    fields = "number,title,url,state,labels,createdAt,updatedAt,author,assignees,milestone,comments"
    attempts = []
    issues: list[dict] = []
    census: dict = {}
    for attempt_number in range(1, max_attempts + 1):
        total_before = _fetch_issue_total(repo, state)
        fetch_limit = max(1, min(limit, total_before))
        raw = _run_gh([
            "issue", "list", "--state", state, "--limit", str(fetch_limit),
            "--json", fields, "--repo", repo,
        ])
        issues = json.loads(raw or "[]")
        if not isinstance(issues, list):
            raise RuntimeError("gh issue list returned a non-array payload")
        total_after = _fetch_issue_total(repo, state)
        snapshot_at = (dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
                       .isoformat().replace("+00:00", "Z"))
        snapshot_id = _snapshot_id(issues)
        census = reconcile_census(
            scope="repository_issues", state=state, fetched_count=len(issues),
            total_count=total_after, snapshot_age_seconds=0,
            includes_pull_requests=False, snapshot_at=snapshot_at,
        )
        stable_counts = total_before == total_after
        if not stable_counts:
            census["page_complete"] = False
            census["reconciliation"] = "count_mismatch"
        attempt = {
            "attempt": attempt_number,
            "total_before": total_before,
            "fetched_count": len(issues),
            "total_after": total_after,
            "fetch_limit": fetch_limit,
            "reconciliation": census["reconciliation"],
            "snapshot_at": snapshot_at,
            "snapshot_id": snapshot_id,
        }
        attempts.append(attempt)
        census.update({
            "attempt_count": attempt_number,
            "max_attempts": max_attempts,
            "snapshot_id": snapshot_id,
            "attempts": list(attempts),
        })
        if stable_counts and census["reconciliation"] == "complete":
            return issues, census
    return issues, census


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


def load_injected_snapshot(source: str, *, state: str,
                           now: dt.datetime) -> tuple[list[dict], dict]:
    """Load a raw provided scope or an explicit ``{issues,census}`` envelope."""
    raw = sys.stdin.read() if source == "-" else Path(source).read_text(encoding="utf-8")
    text = raw.strip()
    if not text:
        issues: list[dict] = []
        return issues, reconcile_census(
            scope="provided_issues", state=state, fetched_count=0, total_count=0,
            snapshot_age_seconds=0, includes_pull_requests=False,
            snapshot_at=now.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        )
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"--issues input is not valid JSON: {exc}") from exc
    if isinstance(data, list):
        return data, reconcile_census(
            scope="provided_issues", state=state, fetched_count=len(data),
            total_count=len(data), snapshot_age_seconds=0,
            includes_pull_requests=False,
            snapshot_at=now.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        )
    if not isinstance(data, dict) or not isinstance(data.get("issues"), list):
        raise ValueError("--issues input must be a JSON array or {issues,census} object")
    raw_census = data.get("census")
    if not isinstance(raw_census, dict):
        raise ValueError("--issues envelope must carry census metadata")
    snapshot_at = str(raw_census.get("snapshot_at") or "")
    age_seconds = int(raw_census.get("snapshot_age_seconds") or 0)
    if snapshot_at:
        try:
            then = dt.datetime.fromisoformat(snapshot_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"census snapshot_at is invalid: {exc}") from exc
        age_seconds = max(0, int((now - then).total_seconds()))
    census = reconcile_census(
        scope=str(raw_census.get("scope") or ""),
        state=str(raw_census.get("state") or state),
        fetched_count=int(raw_census.get("fetched_count", len(data["issues"]))),
        total_count=int(raw_census.get("total_count", len(data["issues"]))),
        snapshot_age_seconds=age_seconds,
        includes_pull_requests=bool(raw_census.get("includes_pull_requests", False)),
        snapshot_at=snapshot_at or None,
    )
    return data["issues"], census


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
        "state": issue.get("state", "OPEN"),
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


def build_repair_batches(rows: list[dict], batch_size: int) -> list[dict]:
    """Bound review-only taxonomy queues without inventing labeling commands."""
    if batch_size <= 0:
        raise ValueError("repair batch size must be positive")
    batches: list[dict] = []
    for axis, tag in (("priority", "needs-priority"),
                      ("kind", "needs-kind"),
                      ("area", "needs-area")):
        numbers = [r["number"] for r in rows if tag in r["tags"]]
        for start in range(0, len(numbers), batch_size):
            batches.append({
                "axis": axis,
                "batch": start // batch_size + 1,
                "issues": numbers[start:start + batch_size],
                "review_only": True,
            })
    return batches


# ---- Rendering --------------------------------------------------------------
def render_md(report: dict, as_of: str) -> str:
    c = report["counts"]
    census = report["census"]
    L = [
        f"# Issue triage — {as_of}",
        "",
        (f"**Census:** scope `{census['scope']}:{census['state']}` · fetched "
         f"{census['fetched_count']} / total {census['total_count']} · page-complete "
         f"{str(census['page_complete']).lower()} · snapshot-age "
         f"{census['snapshot_age_seconds']}s · reconciliation "
         f"`{census['reconciliation']}` · attempts "
         f"{census.get('attempt_count', 1)} · snapshot "
         f"`{census.get('snapshot_id', 'not-recorded')}`"),
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

    repairs = report.get("repair_batches", [])
    if repairs:
        L.append(f"## Taxonomy repair batches ({len(repairs)} review-only)")
        L.append("")
        for batch in repairs:
            L.append(f"- **{batch['axis']} batch {batch['batch']}** — "
                     f"{len(batch['issues'])} issue(s), review only")
        L.append("")

    return "\n".join(L)


def build_report(issues: list[dict], now: dt.datetime, *, census: dict | None = None,
                 repair_batch_size: int = DEFAULT_REPAIR_BATCH_SIZE) -> dict:
    if census is None:
        census = reconcile_census(
            scope="provided_issues", state="all", fetched_count=len(issues),
            total_count=len(issues), snapshot_age_seconds=0,
            includes_pull_requests=False,
        )
    if census.get("reconciliation") != "complete" or not census.get("page_complete"):
        raise IncompleteRankingError(
            "ranking refused: "
            f"{census.get('reconciliation', 'scope_mismatch')} "
            f"(scope={census.get('scope', '?')} fetched={census.get('fetched_count', '?')} "
            f"total={census.get('total_count', '?')} "
            f"age={census.get('snapshot_age_seconds', '?')}s)"
        )
    if int(census.get("fetched_count", -1)) != len(issues):
        raise IncompleteRankingError(
            f"ranking refused: count_mismatch (decoded={len(issues)} "
            f"fetched={census.get('fetched_count')} total={census.get('total_count')})"
        )
    rows = sorted(
        (classify(i, now) for i in issues),
        key=lambda r: (-r["score"], -r["number"]),
    )
    actions = build_actions(rows)

    def count(tag: str) -> int:
        return sum(1 for r in rows if tag in r["tags"])

    return {
        "as_of": now.date().isoformat(),
        "census": census,
        "counts": {
            "open": sum(1 for r in rows if str(r.get("state", "OPEN")).upper() == "OPEN"),
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
        "repair_batches": build_repair_batches(rows, repair_batch_size),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GitHub issue triage + rank helper.")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--actions", action="store_true")
    ap.add_argument("--out", default=None, help="write output to this path")
    ap.add_argument("--repo", default=None, help="owner/repo for gh; default current repository")
    ap.add_argument("--state", default="open", choices=("open", "closed", "all"),
                    help="repository issue state scope (default: open)")
    ap.add_argument("--limit", type=int, default=LIST_LIMIT,
                    help="safety ceiling; ranking refuses if the issue-only total exceeds it")
    ap.add_argument("--repair-batch-size", type=int, default=DEFAULT_REPAIR_BATCH_SIZE,
                    help="maximum review-only issue numbers per taxonomy repair batch")
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

    if a.limit <= 0 or a.repair_batch_size <= 0:
        print("ERROR: --limit and --repair-batch-size must be positive", file=sys.stderr)
        return 2

    _load_config(a.config)

    now = dt.datetime.now(dt.timezone.utc)
    if a.as_of:
        now = dt.datetime.fromisoformat(a.as_of + "T00:00:00+00:00")

    try:
        if a.issues:
            issues, census = load_injected_snapshot(a.issues, state=a.state, now=now)
        else:
            if a.repo is None and a.state == "open" and a.limit == LIST_LIMIT:
                fetched = fetch_issues()
            else:
                fetched = fetch_issues(repo=a.repo, state=a.state, limit=a.limit)
            # Compatibility with focused tests and older callers that monkeypatch
            # fetch_issues to return only the legacy list shape.
            if isinstance(fetched, tuple):
                issues, census = fetched
            else:
                issues = fetched
                census = reconcile_census(
                    scope="provided_issues", state=a.state,
                    fetched_count=len(issues), total_count=len(issues),
                    snapshot_age_seconds=0, includes_pull_requests=False,
                )
    except FileNotFoundError:
        msg = f"--issues file not found: {a.issues}" if a.issues else "`gh` not found on PATH."
        print(f"ERROR: {msg}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if a.since_days is not None:
        issues = [i for i in issues if _days(i.get("updatedAt", ""), now) <= a.since_days]
        census = reconcile_census(
            scope=f"{census['scope']}:updated_within_{a.since_days}d",
            state=census["state"], fetched_count=len(issues), total_count=len(issues),
            snapshot_age_seconds=int(census.get("snapshot_age_seconds", 0)),
            includes_pull_requests=bool(census.get("includes_pull_requests", False)),
            snapshot_at=census.get("snapshot_at"),
        )

    try:
        report = build_report(
            issues, now, census=census, repair_batch_size=a.repair_batch_size,
        )
    except IncompleteRankingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

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
        out = {
            "as_of": as_of,
            "census": report["census"],
            "actions": report["actions"],
            "repair_batches": report["repair_batches"],
        }
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
