#!/usr/bin/env python3
r"""pm_review.py — the PM "what should we START next, under our WIP limit?" rollup.

The repo already has rich per-surface scorecards (industry, concept-disambiguation,
intent-literal, dispatch closure-honesty, bench-plan frontier) and fleet tools
(``fleet_bottleneck``=what limits us now; ``issue_closure_audit``=is a close honest;
``dispatch_throughput``=close rate; ``plan_audit``=how much is done). Each answers a
DIFFERENT question. None folds **market-standing importance + operational urgency +
capacity** into one operator do-next view — the synthesis the 2026-06-27 PM review
produced *by hand*. This makes that durable and repeatable.

It is read-only and **composes existing ``--json`` payloads** — it never edits,
labels, or closes an issue, and never mutates an upstream tool. Its only inputs are
read-back surfaces:

  * ``gh issue list --state open --json`` — the open backlog (the snapshot it scores).
  * (optional) ``fleet_bottleneck.py json`` payload via ``--bottleneck-json`` — folded
    into the "one blocker to fix" line when the limiter is a fleet/system constraint.
  * (optional) ``industry_scorecard.py --json`` via ``--industry-json`` — its headline
    standing (grade / parity-debt) is folded into ``context`` as competitive pressure
    on the SCHEDULE quadrant. Per-issue importance stays label-derived (below).

The PM layer it adds on top:

  1. **Importance × urgency 2×2 grid** over every open issue. Importance is the repo's
     own per-issue standing distillation — the ``priority/Pn`` label triage sets where
     fak is behind the field — lifted by a **trust-floor** boost (an issue touching the
     safety / correctness / honesty floor — security, guard, scrub, witness, … — is not
     a feature gap, it is a breach of the floor) and an epic-scope boost. Urgency is
     ``age`` + **orphaned-P0/P1** (a fire nobody owns) + **blocker** state. Each issue
     lands in DO_NOW / SCHEDULE / DELEGATE / BACKLOG.
  2. **Epic-spawn-graph** — one epic → its child issues, read from the GitHub task-list
     convention an epic body uses (``- [ ] #N`` open child, ``- [x] #N`` done child, e.g.
     ``#897`` → ``#898``/``#899``/``#925``). This makes true WIP visible in *units* (open
     children), not just the one-line issue count. An epic with no parsed child checklist
     is reported as such — never fabricated to a number.
  3. **WIP-cap analysis** — flags when more than ``cap`` live (open) epics run at once.
     The cap is config-driven (``.claude/project.yaml`` → ``pm_review.wip_cap`` or
     ``--wip-cap``), default **2**, per the review's "don't open a new epic until a live
     one closes." Never hard-coded as the only source.
  4. A bounded **do-next under capacity** recommendation — the top few units to start,
     the single blocker to fix, and (when WIP is over cap) the live epic closest to done
     to push to completion first. It is HONEST about truncation: every top-N cut is
     counted in ``coverage`` and announced via ``log()``; nothing is silently dropped.

Determinism: every grader is pure and the wall clock is INJECTED (``now_ts`` /
``--as-of``), so two runs over the same snapshot at one commit are byte-identical
(stable ``(-score, number)`` sorts, no dict-order leak). The live ``gh`` fetch is the
only nondeterminism, isolated behind one I/O seam the hermetic ``pm_review_test.py``
replaces with synthetic payloads.

    python tools/pm_review.py                       # the operator card (text)
    python tools/pm_review.py --json                # control-pane payload
    python tools/pm_review.py --markdown --out docs/_audits/pm-review-2026-06-28.md
    python tools/pm_review.py --wip-cap 3 --bottleneck-json bl.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
from dispatch_worker import install_no_window_subprocess_defaults
install_no_window_subprocess_defaults(subprocess)
import sys
from pathlib import Path
from typing import Any, Callable

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

SCHEMA = "fleet-pm-review/1"
DEFAULT_WIP_CAP = 2
DEFAULT_IMPORTANCE_THRESHOLD = 50.0
DEFAULT_URGENCY_THRESHOLD = 50.0
DEFAULT_TOP_STARTS = 3
DEFAULT_FETCH_LIMIT = 400

# Importance: the repo's per-issue standing distillation lives in the priority label
# (triage sets Pn by where fak is behind the field), so it is the importance base.
PRIORITY_BASE = {"P0": 70.0, "P1": 55.0, "P2": 40.0, "P3": 25.0}
UNLABELED_PRIORITY_BASE = 35.0
TRUST_FLOOR_BOOST = 20.0
EPIC_SCOPE_BOOST = 10.0

# Urgency components.
URGENCY_AGE_CAP_DAYS = 40.0   # 1 pt/day, saturating at 40 — an old issue is overdue
ORPHANED_P01_BOOST = 35.0     # a P0/P1 with no assignee — a fire nobody owns
BLOCKER_BOOST = 25.0

# The trust/safety/correctness/honesty FLOOR — a defect here is a breach, not a gap.
TRUST_FLOOR_TERMS = (
    "security", "secret", "leak", "guard", "trust", "safety", "correctness",
    "data-loss", "dataloss", "honesty", "honest", "witness", "provenance",
    "scrub", "redact", "auth", "vuln", "cve", "quarantine", "refuse",
)
BLOCKER_TERMS = (
    "blocker", "blocked", "blocking", "blocks", "wedged", "broken",
    "regression", "outage", "stuck", "deadlock",
)

QUADRANTS = ("DO_NOW", "SCHEDULE", "DELEGATE", "BACKLOG")


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


# ---------------------------------------------------------------------------
# I/O seams (real implementations; the hermetic test injects fakes / data)
# ---------------------------------------------------------------------------

IssueFetcher = Callable[[Path, int], list[dict[str, Any]]]


def fetch_open_issues(root: Path, limit: int = DEFAULT_FETCH_LIMIT) -> list[dict[str, Any]]:
    """Open issues with the fields the grid + spawn-graph need — the snapshot.

    One ``gh`` call so the backlog is read as ONE atomic snapshot (a per-issue
    body fetch would race the live backlog). Bodies are included to read the epic
    child checklist and blocker cues without an N+1 fan-out.
    """
    try:
        proc = subprocess.run(
            ["gh", "issue", "list", "--state", "open", "--limit", str(limit),
             "--json", "number,title,labels,assignees,createdAt,updatedAt,body"],
            cwd=str(root), capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [{"_error": str(exc)}]
    text = (proc.stdout or "").strip()
    if not text:
        return [] if proc.returncode == 0 else [{"_error": (proc.stderr or "gh failed").strip()}]
    try:
        data = json.loads(text)
    except ValueError as exc:
        return [{"_error": f"bad gh json: {exc}"}]
    return data if isinstance(data, list) else []


def load_json_file(path: str | None) -> dict[str, Any] | None:
    """Read an upstream tool's ``--json`` payload from a file (optional compose hook)."""
    if not path:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_wip_cap(root: Path, override: int | None) -> tuple[int, str]:
    """Resolve the WIP cap: flag > .claude/project.yaml > default. Returns (cap, source)."""
    if override is not None:
        return override, "flag"
    cfg = root / ".claude" / "project.yaml"
    if cfg.exists():
        try:
            import yaml  # type: ignore
            data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
            block = data.get("pm_review")
            if isinstance(block, dict) and "wip_cap" in block:
                return int(block["wip_cap"]), "project.yaml"
        except Exception:
            pass  # PyYAML missing / malformed block — fall through to the default
    return DEFAULT_WIP_CAP, "default"


# ---------------------------------------------------------------------------
# Pure helpers (no I/O, no wall clock)
# ---------------------------------------------------------------------------

def _parse_iso(ts: str | None) -> dt.datetime | None:
    if not ts:
        return None
    try:
        t = dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)
    except (ValueError, AttributeError):
        return None


def _age_days(created: str | None, now: dt.datetime) -> float | None:
    t = _parse_iso(created)
    if t is None:
        return None
    return max(0.0, (now - t).total_seconds() / 86400.0)


def _label_names(issue: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for lab in issue.get("labels") or []:
        name = lab.get("name") if isinstance(lab, dict) else lab
        if name:
            out.append(str(name))
    return out


def _priority_of(labels: list[str]) -> str | None:
    """Parse priority/Pn -> 'Pn' (the repo's standing label)."""
    for name in labels:
        m = re.search(r"\bP([0-3])\b", name)
        if m and ("priority" in name.lower() or name.strip().upper() == f"P{m.group(1)}"):
            return f"P{m.group(1)}"
    return None


def _is_epic(issue: dict[str, Any], labels: list[str]) -> bool:
    if any(label.strip().lower() == "epic" for label in labels):
        return True
    title = str(issue.get("title") or "").strip().lower()
    return title.startswith("epic(") or title.startswith("epic ") or title == "epic"


def _hit_terms(text: str, terms: tuple[str, ...]) -> bool:
    low = text.lower()
    return any(t in low for t in terms)


_CHILD_RE = re.compile(r"^\s*[-*]\s*\[(?P<box>[ xX])\]\s*#(?P<num>\d+)\b")


def parse_children(body: str | None) -> list[dict[str, Any]]:
    """Children from the GitHub task-list convention: '- [ ] #N' / '- [x] #N'.

    The checkbox is the open/done signal. Only this explicit convention is parsed —
    an epic that lists related issues in prose reports NO children here (honest:
    we never fabricate a unit count from an ambiguous mention).
    """
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for line in (body or "").splitlines():
        m = _CHILD_RE.match(line)
        if not m:
            continue
        num = int(m.group("num"))
        if num in seen:
            continue
        seen.add(num)
        out.append({"number": num, "done": m.group("box").lower() == "x"})
    return out


# ---------------------------------------------------------------------------
# Scorers (pure; clock injected via `now`)
# ---------------------------------------------------------------------------

def importance_score(issue: dict[str, Any], labels: list[str], *, is_epic: bool) -> dict[str, Any]:
    priority = _priority_of(labels)
    base = PRIORITY_BASE.get(priority, UNLABELED_PRIORITY_BASE)
    text = f"{issue.get('title') or ''} {' '.join(labels)}"
    floor = _hit_terms(text, TRUST_FLOOR_TERMS)
    score = base + (TRUST_FLOOR_BOOST if floor else 0.0) + (EPIC_SCOPE_BOOST if is_epic else 0.0)
    return {
        "score": round(min(100.0, score), 1),
        "priority": priority,
        "trust_floor": floor,
        "epic_scope": is_epic,
    }


def urgency_score(issue: dict[str, Any], labels: list[str], *, priority: str | None,
                  now: dt.datetime) -> dict[str, Any]:
    age = _age_days(issue.get("createdAt"), now)
    age_pts = min(URGENCY_AGE_CAP_DAYS, age) if age is not None else 0.0
    assignees = issue.get("assignees") or []
    orphaned = priority in ("P0", "P1") and not assignees
    text = f"{issue.get('title') or ''} {' '.join(labels)} {issue.get('body') or ''}"
    blocker = _hit_terms(text, BLOCKER_TERMS)
    score = age_pts + (ORPHANED_P01_BOOST if orphaned else 0.0) + (BLOCKER_BOOST if blocker else 0.0)
    return {
        "score": round(min(100.0, score), 1),
        "age_days": round(age, 1) if age is not None else None,
        "orphaned_p01": orphaned,
        "blocker": blocker,
    }


def quadrant(importance: float, urgency: float, *, it: float, ut: float) -> str:
    hi_i, hi_u = importance >= it, urgency >= ut
    if hi_i and hi_u:
        return "DO_NOW"
    if hi_i and not hi_u:
        return "SCHEDULE"
    if not hi_i and hi_u:
        return "DELEGATE"
    return "BACKLOG"


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def grade_issues(issues: list[dict[str, Any]], *, now: dt.datetime,
                 it: float, ut: float) -> tuple[list[dict[str, Any]], int]:
    """Enrich every issue with importance/urgency/quadrant. Returns (graded, undated)."""
    graded: list[dict[str, Any]] = []
    undated = 0
    for issue in issues:
        labels = _label_names(issue)
        is_epic = _is_epic(issue, labels)
        imp = importance_score(issue, labels, is_epic=is_epic)
        urg = urgency_score(issue, labels, priority=imp["priority"], now=now)
        if urg["age_days"] is None:
            undated += 1
        q = quadrant(imp["score"], urg["score"], it=it, ut=ut)
        graded.append({
            "number": issue.get("number"),
            "title": issue.get("title"),
            "is_epic": is_epic,
            "priority": imp["priority"],
            "importance": imp["score"],
            "urgency": urg["score"],
            "quadrant": q,
            "do_next_score": round(imp["score"] * urg["score"] / 100.0, 1),
            "factors": {
                "trust_floor": imp["trust_floor"],
                "epic_scope": imp["epic_scope"],
                "age_days": urg["age_days"],
                "orphaned_p01": urg["orphaned_p01"],
                "blocker": urg["blocker"],
            },
            "children": parse_children(issue.get("body")) if is_epic else [],
        })
    # Stable, deterministic order: do-next score desc, then number asc.
    graded.sort(key=lambda g: (-g["do_next_score"], -(g["importance"]), g["number"] or 0))
    return graded, undated


def epic_spawn_graph(graded: list[dict[str, Any]]) -> dict[str, Any]:
    """One epic -> child issues, with true WIP measured in OPEN child units."""
    epics = [g for g in graded if g["is_epic"]]
    rows: list[dict[str, Any]] = []
    units_open = 0
    epics_without_checklist = 0
    for e in epics:
        children = e["children"]
        open_n = sum(1 for c in children if not c["done"])
        done_n = sum(1 for c in children if c["done"])
        if not children:
            epics_without_checklist += 1
        units_open += open_n
        rows.append({
            "epic": e["number"], "title": e["title"],
            "open_children": open_n, "done_children": done_n,
            "child_numbers": [c["number"] for c in children],
            "has_checklist": bool(children),
            "done_ratio": round(done_n / len(children), 2) if children else None,
        })
    rows.sort(key=lambda r: (-(r["open_children"]), -(r["epic"] or 0)))
    return {
        "live_epics": len(epics),
        "true_wip_open_child_units": units_open,
        "epics_without_child_checklist": epics_without_checklist,
        "epics": rows,
    }


def wip_analysis(graph: dict[str, Any], *, cap: int, cap_source: str) -> dict[str, Any]:
    live = graph["live_epics"]
    over = live > cap
    # The live epic closest to done (push it to completion first to free a WIP slot):
    # highest done-ratio, then fewest open children, then number — only among epics
    # that actually carry a parsed checklist (a ratio we can stand behind).
    measurable = [e for e in graph["epics"] if e["has_checklist"]]
    closest = None
    if measurable:
        closest = sorted(
            measurable,
            key=lambda e: (-(e["done_ratio"] or 0.0), e["open_children"], -(e["epic"] or 0)),
        )[0]
    return {
        "cap": cap, "cap_source": cap_source, "live_epics": live,
        "over_cap": over,
        "headroom": cap - live,
        "verdict": "WIP_OVER_CAP" if over else "WIP_OK",
        "closest_to_done_epic": closest,
    }


def do_next(graded: list[dict[str, Any]], graph: dict[str, Any], wip: dict[str, Any], *,
            top_starts: int, bottleneck: dict[str, Any] | None) -> dict[str, Any]:
    """Bounded do-next-under-capacity: top units to start + the one blocker + the
    epic to finish first. Honest about every truncation (counts in `coverage`)."""
    notes: list[str] = []

    # Units to START: top non-epic DO_NOW issues (genuine high importance AND urgency).
    do_now_units = [g for g in graded if g["quadrant"] == "DO_NOW" and not g["is_epic"]]
    starts = do_now_units[:top_starts]
    dropped_starts = len(do_now_units) - len(starts)
    if dropped_starts > 0:
        notes.append(f"start list shows top {len(starts)} of {len(do_now_units)} DO_NOW units "
                     f"(dropped {dropped_starts})")

    # The ONE blocker to fix. A fleet/system limiter (from fleet_bottleneck's payload)
    # outranks an issue-level blocker — it gates the whole fleet, not one ticket.
    one_blocker: dict[str, Any] | None = None
    if bottleneck:
        head = bottleneck.get("headline") or (bottleneck.get("snapshot") or {}).get("headline")
        if head:
            one_blocker = {"kind": "fleet_bottleneck", "title": head.get("title"),
                           "severity": head.get("severity"), "fix": head.get("fix")}
    if one_blocker is None:
        blockers = [g for g in graded if g["factors"]["blocker"]]
        blockers.sort(key=lambda g: (-g["urgency"], -g["do_next_score"], g["number"] or 0))
        if blockers:
            b = blockers[0]
            one_blocker = {"kind": "issue", "number": b["number"], "title": b["title"],
                           "urgency": b["urgency"]}
            if len(blockers) > 1:
                notes.append(f"one blocker shown of {len(blockers)} flagged")

    # Capacity recommendation.
    if wip["over_cap"]:
        rec = (f"WIP over cap: {wip['live_epics']} live epics > cap {wip['cap']}. "
               "Finish a live epic before opening another.")
        ctd = wip.get("closest_to_done_epic")
        if ctd:
            rec += (f" Closest to done: epic #{ctd['epic']} "
                    f"({ctd['done_children']}/{ctd['open_children'] + ctd['done_children']} children done).")
        # If a new-epic candidate is sitting in DO_NOW, say plainly: don't open it.
        epic_candidates = [g for g in graded if g["is_epic"] and g["quadrant"] == "DO_NOW"]
        if epic_candidates:
            notes.append(f"{len(epic_candidates)} epic(s) in DO_NOW but WIP is full — "
                         "do NOT open a new epic; drain first")
    else:
        rec = (f"WIP has {wip['headroom']} epic slot(s) free (cap {wip['cap']}). "
               "Starting the top DO_NOW units below is within capacity.")

    return {
        "recommendation": rec,
        "starts": [{"number": g["number"], "title": g["title"], "priority": g["priority"],
                    "importance": g["importance"], "urgency": g["urgency"],
                    "do_next_score": g["do_next_score"]} for g in starts],
        "one_blocker": one_blocker,
        "finish_first_epic": wip.get("closest_to_done_epic"),
        "notes": notes,
    }


def build_payload(*, root: Path, issues: list[dict[str, Any]], now_ts: float,
                  cap: int, cap_source: str, it: float, ut: float, top_starts: int,
                  bottleneck: dict[str, Any] | None = None,
                  industry: dict[str, Any] | None = None) -> dict[str, Any]:
    now = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(seconds=now_ts)

    fetch_error = None
    if issues and isinstance(issues[0], dict) and issues[0].get("_error"):
        fetch_error = issues[0]["_error"]
        issues = []

    graded, undated = grade_issues(issues, now=now, it=it, ut=ut)
    graph = epic_spawn_graph(graded)
    wip = wip_analysis(graph, cap=cap, cap_source=cap_source)
    rec = do_next(graded, graph, wip, top_starts=top_starts, bottleneck=bottleneck)

    counts = {q: sum(1 for g in graded if g["quadrant"] == q) for q in QUADRANTS}

    context: dict[str, Any] = {}
    if industry:
        # Fold the headline industry standing as competitive pressure (read-only).
        context["industry"] = {
            "grade": industry.get("grade") or (industry.get("composite") or {}).get("grade"),
            "parity_debt": industry.get("parity_debt")
            or (industry.get("corpus") or {}).get("parity_debt"),
            "note": "weak standing -> schedule competitive SCHEDULE-quadrant items sooner",
        }

    verdict = "WIP_OVER_CAP" if wip["over_cap"] else "OK"
    return {
        "schema": SCHEMA,
        "ok": not wip["over_cap"] and fetch_error is None,
        "verdict": "FETCH_ERROR" if fetch_error else verdict,
        "generated_utc": now.isoformat(timespec="seconds"),
        "workspace": str(root),
        "grid": {"importance_threshold": it, "urgency_threshold": ut,
                 "quadrant_counts": counts, "issues": graded},
        "epic_spawn_graph": graph,
        "wip": wip,
        "do_next": rec,
        "context": context,
        "coverage": {
            "open_issues_scanned": len(graded),
            "undated_skipped": undated,
            "quadrant_counts": counts,
            "do_now_units": counts["DO_NOW"],
        },
        "fetch_error": fetch_error,
    }


def collect(root: Path, *, now_ts: float | None = None, wip_cap: int | None = None,
            it: float = DEFAULT_IMPORTANCE_THRESHOLD, ut: float = DEFAULT_URGENCY_THRESHOLD,
            top_starts: int = DEFAULT_TOP_STARTS, fetch_limit: int = DEFAULT_FETCH_LIMIT,
            fetcher: IssueFetcher | None = None, bottleneck_json: str | None = None,
            industry_json: str | None = None) -> dict[str, Any]:
    import time
    now_ts = time.time() if now_ts is None else now_ts
    cap, cap_source = load_wip_cap(root, wip_cap)
    issues = (fetcher or fetch_open_issues)(root, fetch_limit)
    return build_payload(
        root=root, issues=issues, now_ts=now_ts, cap=cap, cap_source=cap_source,
        it=it, ut=ut, top_starts=top_starts,
        bottleneck=load_json_file(bottleneck_json), industry=load_json_file(industry_json))


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def render(p: dict[str, Any]) -> str:
    L: list[str] = []
    L.append(f"PM review: {p.get('verdict')} ({'ok' if p.get('ok') else 'ACTION'})  "
             f"@ {p.get('generated_utc')}")
    if p.get("fetch_error"):
        L.append(f"  ! could not read the backlog: {p['fetch_error']}")
        return "\n".join(L)
    cov, wip = p["coverage"], p["wip"]
    qc = cov["quadrant_counts"]
    L.append(f"  backlog: {cov['open_issues_scanned']} open  ·  "
             f"DO_NOW={qc['DO_NOW']} SCHEDULE={qc['SCHEDULE']} "
             f"DELEGATE={qc['DELEGATE']} BACKLOG={qc['BACKLOG']}")
    L.append(f"  WIP: {wip['verdict']}  ({wip['live_epics']} live epics, cap {wip['cap']} "
             f"from {wip['cap_source']}, headroom {wip['headroom']})  ·  "
             f"true WIP {p['epic_spawn_graph']['true_wip_open_child_units']} open child units")
    L.append("")
    L.append(f"  >>> {p['do_next']['recommendation']}")
    L.append("")
    L.append("  Start next (top DO_NOW units):")
    if p["do_next"]["starts"]:
        for s in p["do_next"]["starts"]:
            L.append(f"    #{s['number']:<5} [{s['priority'] or '--'}] "
                     f"imp {s['importance']:>5} · urg {s['urgency']:>5} · "
                     f"do-next {s['do_next_score']:>5}  {str(s['title'])[:72]}")
    else:
        L.append("    (none in DO_NOW — backlog is not on fire)")
    b = p["do_next"]["one_blocker"]
    if b:
        if b["kind"] == "fleet_bottleneck":
            L.append(f"  One blocker to fix: [fleet] {b['title']} ({b.get('severity')})")
        else:
            L.append(f"  One blocker to fix: #{b['number']} {str(b['title'])[:64]}")
    for note in p["do_next"]["notes"]:
        L.append(f"  · {note}")
    g = p["epic_spawn_graph"]
    L.append("")
    L.append(f"  Epic spawn-graph ({g['live_epics']} live, "
             f"{g['epics_without_child_checklist']} without a parsed checklist):")
    for e in g["epics"][:8]:
        tag = "" if e["has_checklist"] else "  (no checklist)"
        L.append(f"    #{e['epic']:<5} open {e['open_children']:>2} / done {e['done_children']:>2}"
                 f"{tag}  {str(e['title'])[:60]}")
    return "\n".join(L)


def render_md(p: dict[str, Any], as_of: str) -> str:
    if p.get("fetch_error"):
        return f"# PM review — {as_of}\n\n> Could not read the backlog: {p['fetch_error']}\n"
    cov, wip, g, dn = p["coverage"], p["wip"], p["epic_spawn_graph"], p["do_next"]
    qc = cov["quadrant_counts"]
    L = [
        f"# PM review — do-next under capacity — {as_of}",
        "",
        f"**Verdict:** `{p['verdict']}`  ·  backlog **{cov['open_issues_scanned']}** open  ·  "
        f"WIP **{wip['live_epics']}** live epics (cap **{wip['cap']}** from `{wip['cap_source']}`, "
        f"headroom {wip['headroom']})  ·  true WIP **{g['true_wip_open_child_units']}** open child units",
        "",
        "> Importance = `priority/Pn` label (the repo's standing distillation) + trust-floor "
        "boost + epic scope. Urgency = age + orphaned-P0/P1 + blocker state. "
        "Read-only over the live backlog; deterministic at one commit.",
        "",
        "## Do next",
        "",
        f"**{dn['recommendation']}**",
        "",
        "| # | Pri | Importance | Urgency | Do-next | Title |",
        "|---|---|---:|---:|---:|---|",
    ]
    for s in dn["starts"]:
        L.append(f"| {s['number']} | {s['priority'] or '—'} | {s['importance']} | "
                 f"{s['urgency']} | {s['do_next_score']} | {str(s['title'])[:80]} |")
    if not dn["starts"]:
        L.append("| — | — | — | — | — | _none in DO_NOW_ |")
    b = dn["one_blocker"]
    if b:
        blk = (f"[fleet] {b['title']} ({b.get('severity')})" if b["kind"] == "fleet_bottleneck"
               else f"#{b['number']} — {b['title']}")
        L += ["", f"**One blocker to fix:** {blk}"]
    for note in dn["notes"]:
        L.append(f"- _{note}_")
    L += [
        "",
        "## Importance × urgency grid",
        "",
        f"- **DO_NOW** (important & urgent): {qc['DO_NOW']}",
        f"- **SCHEDULE** (important, not urgent): {qc['SCHEDULE']}",
        f"- **DELEGATE** (urgent, not important): {qc['DELEGATE']}",
        f"- **BACKLOG** (neither): {qc['BACKLOG']}",
        "",
        "## Epic spawn-graph (true WIP in units)",
        "",
        f"{g['live_epics']} live epics · {g['epics_without_child_checklist']} without a parsed "
        f"`- [ ] #N` checklist (reported, never fabricated).",
        "",
        "| Epic | Open children | Done children | Has checklist |",
        "|---|---:|---:|---|",
    ]
    for e in g["epics"]:
        L.append(f"| #{e['epic']} | {e['open_children']} | {e['done_children']} | "
                 f"{'yes' if e['has_checklist'] else 'no'} |")
    L.append("")
    return "\n".join(L)


def log(msg: str) -> None:
    """Announce a truncation / dropped-coverage note to stderr (stdout stays clean)."""
    print(f"[pm_review] {msg}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="PM do-next-under-capacity rollup over the open backlog (read-only).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--wip-cap", type=int, default=None,
                    help="max concurrent live epics (default: .claude/project.yaml pm_review.wip_cap, else 2)")
    ap.add_argument("--importance-threshold", type=float, default=DEFAULT_IMPORTANCE_THRESHOLD)
    ap.add_argument("--urgency-threshold", type=float, default=DEFAULT_URGENCY_THRESHOLD)
    ap.add_argument("--top-starts", type=int, default=DEFAULT_TOP_STARTS,
                    help=f"how many DO_NOW units to recommend starting (default: {DEFAULT_TOP_STARTS})")
    ap.add_argument("--bottleneck-json", default=None,
                    help="fleet_bottleneck.py json payload to fold into the one-blocker line")
    ap.add_argument("--industry-json", default=None,
                    help="industry_scorecard.py --json to fold standing into context")
    ap.add_argument("--limit", type=int, default=DEFAULT_FETCH_LIMIT, help="max open issues to fetch")
    ap.add_argument("--json", action="store_true", help="emit the control-pane JSON payload")
    ap.add_argument("--markdown", action="store_true", help="emit the dated markdown snapshot")
    ap.add_argument("--out", default=None, help="write rendered output to this path")
    ap.add_argument("--as-of", default=None, help="date stamp for --markdown (default: today UTC)")
    a = ap.parse_args(argv)

    root = Path(a.workspace).resolve() if a.workspace else repo_root()
    payload = collect(
        root, wip_cap=a.wip_cap, it=a.importance_threshold, ut=a.urgency_threshold,
        top_starts=a.top_starts, fetch_limit=a.limit,
        bottleneck_json=a.bottleneck_json, industry_json=a.industry_json)

    # No silent truncation: surface every dropped-coverage note on stderr.
    for note in payload.get("do_next", {}).get("notes", []):
        log(note)
    if payload.get("coverage", {}).get("undated_skipped"):
        log(f"{payload['coverage']['undated_skipped']} issue(s) had no parseable createdAt "
            "(urgency age component treated as 0)")

    if a.json:
        out = json.dumps(payload, indent=2)
    elif a.markdown:
        as_of = a.as_of or payload.get("generated_utc", "")[:10]
        out = render_md(payload, as_of)
    else:
        out = render(payload)

    if a.out:
        Path(a.out).write_text(out + ("" if out.endswith("\n") else "\n"), encoding="utf-8")
        print(f"wrote {a.out}", file=sys.stderr)
    else:
        print(out)
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
