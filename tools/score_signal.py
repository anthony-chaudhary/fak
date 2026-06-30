#!/usr/bin/env python3
r"""score-signal — turn a CI scorecard REGRESSION into a specific, deduped GitHub
issue the dispatch loop can resolve. The INTERNAL feeder of the fleet loop.

The issue-dispatch loop (docs/dispatch-loop.md) RESOLVES the open backlog; two
tools FEED it. tools/idea_scout.py is the EXTERNAL feeder (arXiv + GitHub -> new
ideas). This is the INTERNAL feeder: it watches fak's own scorecard control pane
(tools/scorecard_control_pane.py) and, when a metric's debt rises above its pinned
baseline, turns that score CHANGE into one actionable, scoped, lane-routed issue —
the "intelligence" rung that closes CI signal -> tracked work -> a worker.

  CI ratchet sees a regression  ->  score-signal files a scoped issue
  ->  the dispatch loop routes it to a lane + spawns a worker
  ->  the worker runs the owning scorecard's RSI skill, ships a `#N` commit
  ->  the witness closes the issue per-SHA.

The signal SOURCE is the control pane's per-metric trend. `compute_trend` already
emits `trend.early_warning = [{key,label,delta,from,to}, ...]` for EVERY scorecard
whose debt rose vs the pinned baseline (tools/scorecard_baseline.json), independent
of where the portfolio total landed. So a single metric's regression hiding under
another's improvement (the #712 early-warning case) still surfaces here. score-signal
reads that list, NOT the raw scores, so it acts only on a real measured rise.

The hard part of an UNATTENDED issue filer is NOT detecting — it is NOT spamming.
Dedup is intentionally different from idea_scout's permanent seen-cache: a score
regression is RECURRING work. Once a worker retires it and the issue is CLOSED, a
LATER regression of the SAME scorecard SHOULD file a fresh issue. So score-signal
dedups against the OPEN score-signal-labelled issues only, by a stable per-scorecard
marker stamped in the body (`<!-- fak-score-signal: <key> -->`): at most one OPEN
issue per scorecard at a time, re-fileable after a close. Scoping the dedup fetch to
the tool's own label makes that bound EXACT regardless of total backlog size. A hard
CAP (--max-issues, worst-regression-first) keeps a multi-metric regression from
storming the tracker.

"if relevant" — the relevance filter, in order:
  * baseline must be PINNED (an unpinned pane has no delta to act on -> nothing).
  * the metric's debt must have RISEN by at least --min-delta (default 1).
  * an ERRORED/unmeasured scorecard is skipped (that's a measurement failure the
    cadence/ratchet own, not an actionable debt rise).

SAFE BY DEFAULT: dry-run. The tool prints exactly the issues it WOULD file and
mutates nothing. `--live` is the explicit opt-in that actually creates them via
`gh issue create` — the same dry-run-first contract as idea_scout and the dispatch
tools.

    python tools/score_signal.py                       # dry-run: plan, file nothing
    python tools/score_signal.py --json                # machine-readable plan
    python tools/score_signal.py --from pane.json      # read a pre-folded pane
    python tools/scorecard_control_pane.py --json | python tools/score_signal.py --from -
    python tools/score_signal.py --max-issues 5 --live # file at most 5, worst-first

Exit codes: 0 = ran clean (including "nothing to signal") · 2 = infra error (gh
missing / not authed / not a repo / the control pane could not be folded).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

_CREATE_NO_WINDOW = 0x08000000


def _win_creationflags() -> int:
    return _CREATE_NO_WINDOW if os.name == "nt" else 0


# The control pane is a sibling tool; importing it gives the AUTHORITATIVE
# key -> scorecard metadata (script/cmd/label) so the issue names the right tool to
# re-run, and never drifts from the family the pane actually folds. Import-safe:
# scorecard_control_pane defines only functions/constants at module load.
try:
    import scorecard_control_pane as _cp  # noqa: E402
except ImportError:  # pragma: no cover - only when run outside tools/ on sys.path
    _cp = None  # type: ignore[assignment]

SCHEMA = "fak-score-signal/1"
SIGNAL_LABEL = "score-signal"
DEBT_LABEL = "tech-debt"
MARKER_RE = re.compile(r"fak-score-signal:\s*([A-Za-z0-9_\-]+)")
# The NOTED-DELTA marker (#981): a second stable comment recording the delta the
# issue currently reports, so a worsening regression on an already-open ticket can
# be detected and REFRESHED — while a flat metric is never re-commented. Kept
# separate from MARKER_RE so the dedup anchor regex stays byte-for-byte unchanged.
DELTA_MARKER_RE = re.compile(r"fak-score-signal-delta:\s*([A-Za-z0-9_\-]+)=(-?\d+)")

DEFAULTS = {
    "min_delta": 1,        # a metric must rise by at least this to be actionable
    "max_issues": 5,       # hard cap on issues filed per run (anti-storm)
    "issue_scan_limit": 800,  # open score-signal issues fetched for the dedup index
    "worsen_delta": 2,     # an already-open metric must worsen by >= this to REFRESH
}

# The owning RSI skill per scorecard key — the "do this next" hint baked into each
# issue so the worker (or a human) knows which repeatable pass retires the debt. A
# HINT, not a hard claim: a mapped skill is offered ONLY when it resolves on disk
# (.claude/skills/<name>/SKILL.md, checked at render time via available_skills);
# otherwise the issue degrades to the generic /score-2x conductor, so a removed or
# renamed skill can never become a false lead. Keys whose retire path is a native
# `fak` subcommand (no slash skill) are absent here and carry a SOURCE_BY_KEY
# Go-source pointer instead.
SKILL_BY_KEY: dict[str, str] = {
    "readme": "/refresh-readme",
    "code": "/quality-score",
    "appeal": "/appeal-score",
    "parity": "/industry-score",
    "agent": "/agent-readiness",
    "persona": "/persona-score",
    "stability": "/stability-score",
    "slop": "/slop-score",
    "steer": "/steerability-score",
    "conflation": "/conflation-score",
    "disambiguation": "/disambiguation-score",
    "tokendefaults": "/token-defaults-score",
    "guard_rsi": "/guard-rsi-score",
    "doc": "/curate-cluster",
}

# Native-cmd scorecards: their debt is retired in Go source, not a tools/ script.
# The body names the owning Go file in the `fak/cmd/...` doc-link form the dispatch
# router (issue_lane_router.path_matches_lane) normalizes to the repo-relative
# `cmd/**` tree — so these tickets route to the `cmd` lane, not the `tools` lane the
# incidental tools/ references would otherwise pull them to.
SOURCE_BY_KEY: dict[str, str] = {
    "tokendefaults": "fak/cmd/fak/token_defaults.go",
    "guard_rsi": "fak/cmd/fak/guardrsi.go",
    "dogfood": "fak/cmd/fak/dogfoodscore.go",
}


# ============================================================================
# Pure helpers (no I/O) — unit-tested directly.
# ============================================================================
def tool_by_key() -> dict[str, str]:
    """key -> the command that re-measures that scorecard, from the control pane's
    own SCORECARDS table (a `tools/<script>.py` path, or a `go run`/`fak` cmd).
    Empty when the pane can't be imported (the issue then degrades gracefully)."""
    out: dict[str, str] = {}
    if _cp is None:
        return out
    for card in getattr(_cp, "SCORECARDS", []):
        key = card.get("key")
        if not key:
            continue
        if card.get("cmd"):
            out[key] = card["cmd"]
        elif card.get("script"):
            out[key] = f"tools/{card['script']}"
    return out


def available_skills(root: Path) -> set[str]:
    """The slash-skills that actually resolve on disk (.claude/skills/<name>/SKILL.md).
    A SKILL_BY_KEY hint is offered only when its name is in this set; otherwise the
    issue degrades to the generic /score-2x fallback, so a removed/renamed skill is
    never asserted as a false lead. Empty (degrade-all) when the dir is absent."""
    out: set[str] = set()
    try:
        for child in (root / ".claude" / "skills").iterdir():
            if (child / "SKILL.md").is_file():
                out.add(child.name)
    except OSError:
        return out
    return out


def marker(key: str) -> str:
    """The load-bearing dedup anchor: one stable HTML-comment per scorecard key."""
    return f"<!-- fak-score-signal: {key} -->"


def delta_marker(key: str, delta: int) -> str:
    """The noted-delta anchor (#981): records the delta the issue currently reports
    so a later worsening can be detected and a flat metric never re-commented."""
    return f"<!-- fak-score-signal-delta: {key}={delta} -->"


def open_issue_keys(issues: list[dict[str, Any]]) -> set[str]:
    """The set of scorecard keys already tracked by an OPEN issue (its body carries
    the marker). These are skipped — at most one open issue per scorecard."""
    keys: set[str] = set()
    for iss in issues:
        for m in MARKER_RE.findall(iss.get("body") or ""):
            keys.add(m.strip())
    return keys


def open_issue_index(issues: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """key -> {number, noted_delta} for every OPEN score-signal issue (#981). The
    noted delta is read from the delta-marker; an older issue filed before the marker
    existed has no recorded delta, so its noted value falls back to the delta parsed
    from the title (`+N`), and finally to 0 — a conservative floor that lets a real
    worsening still trip the threshold rather than being hidden by a missing marker.
    The LAST marker/title wins, so a refreshed issue's bumped value is the one read."""
    idx: dict[str, dict[str, Any]] = {}
    for iss in issues:
        body = iss.get("body") or ""
        keys = [m.strip() for m in MARKER_RE.findall(body)]
        if not keys:
            continue
        key = keys[-1]
        noted: int | None = None
        dm = DELTA_MARKER_RE.findall(body)
        for k, v in dm:
            if k.strip() == key:
                noted = int(v)  # last wins
        if noted is None:
            tm = re.search(r"\+(\d+)", str(iss.get("title") or ""))
            noted = int(tm.group(1)) if tm else 0
        idx[key] = {"number": iss.get("number"), "noted_delta": noted}
    return idx


def regressions(payload: dict[str, Any], min_delta: int) -> list[dict[str, Any]]:
    """Every scorecard whose debt rose by >= min_delta vs the pinned baseline, worst
    first. Pure over a folded control-pane payload — reads `trend.early_warning`
    (the per-metric risen set), enriched with each metric's grade. An unpinned or
    baseline-less pane yields [] (there is no delta to act on)."""
    trend = payload.get("trend") or {}
    if not isinstance(trend, dict) or trend.get("direction") == "unpinned":
        return []
    risen = trend.get("early_warning") or []
    grade_by_key = {
        m.get("key"): m.get("grade")
        for m in (payload.get("metrics") or [])
        if isinstance(m, dict)
    }
    portfolio_regressed = trend.get("direction") == "regressed"
    base_commit = str(trend.get("baseline_commit") or "")
    out: list[dict[str, Any]] = []
    for e in risen:
        if not isinstance(e, dict):
            continue
        delta = e.get("delta")
        if not isinstance(delta, int) or isinstance(delta, bool) or delta < min_delta:
            continue
        key = str(e.get("key") or "")
        if not key:
            continue
        out.append({
            "key": key,
            "label": str(e.get("label") or key),
            "delta": int(delta),
            "from": int(e.get("from", 0)),
            "to": int(e.get("to", 0)),
            "grade": grade_by_key.get(key),
            "portfolio_regressed": bool(portfolio_regressed),
            "baseline_commit": base_commit,
        })
    # Worst regression first (then by key for a stable, deterministic order).
    out.sort(key=lambda r: (-r["delta"], r["key"]))
    return out


def render_issue(cand: dict[str, Any], commit: str, today: str,
                 tools: dict[str, str], available: set[str]) -> dict[str, Any]:
    """Build the {title, body, labels} an issue is created from. The per-scorecard
    marker is the load-bearing dedup anchor; the body names the owning skill (only
    when it resolves on disk) + the exact re-measure command so the resolving worker
    has a closed contract. `available` is the set of on-disk skill names."""
    label = cand["label"]
    key = cand["key"]
    delta, frm, to = cand["delta"], cand["from"], cand["to"]
    grade = cand.get("grade")
    grade_note = f" (grade {grade})" if grade else ""

    title = f"score-signal: {label} debt rose +{delta} ({frm}->{to})"

    if cand.get("portfolio_regressed"):
        severity = ("**BLOCKING** — the portfolio total regressed too, so the "
                    "scorecard ratchet (`scorecard_control_pane.py --check`, wired "
                    "into `.github/workflows/ci.yml`) is RED until this is retired.")
    else:
        severity = ("**Advisory (early-warning)** — the portfolio total still holds "
                    "at-or-below baseline, so the ratchet is green, but this metric "
                    "rose under it (#712). Retire it BEFORE the next `--pin` re-floors "
                    "the rise as the new baseline.")

    # The skill hint is offered ONLY when it resolves on disk; a stale/absent mapping
    # degrades to the generic /score-2x conductor rather than asserting a false lead.
    skill = SKILL_BY_KEY.get(key)
    skill_ok = bool(skill) and skill.lstrip("/") in available
    remeasure = tools.get(key, "")
    source = SOURCE_BY_KEY.get(key, "")
    do_lines = []
    if skill_ok:
        do_lines.append(f"1. Run the owning RSI pass: `{skill}` "
                        f"(or the generic `/score-2x {label}` conductor).")
    else:
        do_lines.append(f"1. Run the generic `/score-2x {label}` conductor "
                        f"(no dedicated skill resolves on disk for `{key}`).")
    do_lines.append("2. Retire the regressed debt worst-first; change no claim, "
                    "number, or link the metric does not cover.")
    if remeasure:
        do_lines.append(f"3. Re-measure to PROVE the drop: `{remeasure}` for the "
                        f"per-metric debt, then `python tools/scorecard_control_pane.py` "
                        f"for the portfolio.")
    else:
        do_lines.append("3. Re-measure with `python tools/scorecard_control_pane.py` "
                        "to PROVE the per-metric debt dropped.")
    do_lines.append("4. Ship a commit citing this issue's `#N` in the subject + a "
                    "`(fak <leaf>)` trailer, so the witness can bind and close it.")

    # Native-cmd scorecards: the fix lives in Go, not tools/. Naming the owning Go
    # source routes the ticket to the `cmd` lane (the router strips the `fak/` and
    # matches `cmd/**`), correcting the otherwise-tools-lane misroute.
    fix_loc = ""
    if source:
        fix_loc = (f"**Fix location:** this scorecard is a native `fak` subcommand — "
                   f"its debt is retired in Go source (`{source}`), NOT under "
                   f"`tools/`; route to the `cmd` lane.\n\n")
    lane = "cmd" if source.startswith("fak/cmd/") or source.startswith("cmd/") else "tools"
    path_hints = []
    if source:
        path_hints.append(source)
    elif remeasure and remeasure.startswith("tools/"):
        path_hints.append(remeasure)
    path_hints.extend(["tools/score_signal.py", "tools/scorecard_control_pane.py"])
    seen_paths = set()
    path_hint_lines = []
    for path in path_hints:
        if path and path not in seen_paths:
            seen_paths.add(path)
            path_hint_lines.append(f"- `{path}`")

    body = (
        f"> Auto-filed by **score-signal** (`tools/score_signal.py`, {today}) from "
        f"the CI scorecard control pane @{commit or 'unknown'}. A measured debt "
        f"regression — **needs a worker**; close as `wontfix` if the rise is "
        f"intentional and re-pin the baseline.\n\n"
        f"**Metric:** `{label}` ({key}){grade_note}\n"
        f"**Change:** debt rose **+{delta}** ({frm} -> {to}) vs the pinned baseline"
        + (f" @{cand['baseline_commit']}" if cand.get("baseline_commit") else "")
        + ".\n\n"
        f"**Severity:** {severity}\n\n"
        + fix_loc
        + "**What to do**\n"
        + "\n".join(do_lines)
        + "\n\n### Parent context\n"
        "score-signal scorecard regression from the CI control pane.\n\n"
        "### Current state\n"
        + f"`{label}` ({key}) debt rose +{delta} ({frm} -> {to}){grade_note} "
        + "against the pinned baseline"
        + (f" @{cand['baseline_commit']}" if cand.get("baseline_commit") else "")
        + ".\n\n"
        "### Why this is next\n"
        "The scorecard debt rose from the pinned baseline, so the working path is "
        "less trustworthy until the regression is retired or explicitly accepted.\n\n"
        "### Working spine\n"
        + f"Retire the `{label}` scorecard regression before unrelated quality work.\n\n"
        "### Priority context\n"
        "Working path: control pane -> score-signal issue -> owning RSI/gate -> debt "
        "back to baseline. Current blocker: the metric debt rose. Unblocks: the "
        "scorecard ratchet remains trustworthy for this lane. Not polish: this fixes "
        "a measured regression before cosmetic cleanup.\n\n"
        "### In scope\n"
        + f"Run the owning pass for `{label}`, fix the smallest responsible cause, "
        "and re-measure the scorecard/control pane.\n\n"
        "### Out of scope\n"
        "Do not re-pin the baseline, rewrite unrelated scorecards, or change claims "
        "outside this metric unless the regression is explicitly accepted.\n\n"
        "### Done condition\n"
        "The metric debt is back at or below the pinned baseline, or the baseline is "
        "intentionally re-pinned with the reason recorded.\n\n"
        "### Witness\n"
        + (f"`{remeasure}` and " if remeasure else "")
        + "`python tools/scorecard_control_pane.py --json` show the regression retired "
        "or accepted.\n\n"
        "### Acceptance gate\n"
        + (f"`{remeasure}` followed by " if remeasure else "")
        + "`python tools/scorecard_control_pane.py --check` passes or the accepted "
        "trade-off is re-pinned.\n\n"
        "### Lane\n"
        + lane + "\n\n"
        "### Path hints\n"
        + "\n".join(path_hint_lines)
        + "\n\n"
        "### Boundary notes\n"
        "Public scorecard regression only; do not include private operator telemetry.\n\n"
        "### Closure binding\n"
        + f"Resolving commit cites this issue's #N in the subject and carries `(fak {lane})`.\n"
        + "\n\n---\n"
        "_The scorecard family is listed in `tools/scorecard_control_pane.py`. After "
        "this issue is closed, a future regression of the same metric files a FRESH "
        "issue — only on an armed `--live` run; the scheduled CI run is dry-run. While "
        "this issue stays open, a materially worse delta REFRESHES it in place rather "
        "than filing a duplicate._\n"
        + marker(key)
        + "\n"
        + delta_marker(key, delta)
    )

    return {
        "title": title,
        "body": body,
        "labels": [SIGNAL_LABEL, DEBT_LABEL],
        "key": key,
        "delta": delta,
    }


def render_refresh(cand: dict[str, Any], number: int, noted: int, today: str,
                   ) -> dict[str, Any]:
    """Build the in-place REFRESH of an already-open issue whose tracked regression
    worsened (#981): a dated comment with the new before->after numbers, the bumped
    title (so the backlog reflects current severity), and the new noted-delta marker
    appended so a later stable run never re-comments. Strictly bounded — the caller
    emits at most one refresh per metric per run, only past the --worsen-delta
    threshold; this function just renders the chosen refresh."""
    label = cand["label"]
    key = cand["key"]
    delta, frm, to = cand["delta"], cand["from"], cand["to"]
    base = cand.get("baseline_commit") or ""
    base_s = f" @{base}" if base else ""
    comment = (
        f"> **score-signal refresh** ({today}) — the tracked regression worsened.\n\n"
        f"`{label}` ({key}) debt is now **+{delta}** ({frm} -> {to}) vs the pinned "
        f"baseline{base_s} — up from the **+{noted}** this issue last recorded. The "
        f"underlying debt grew while this ticket stayed open; the title is bumped to "
        f"the current severity.\n\n"
        f"_Bounded: refreshed at most once per run and only when the delta grew "
        f"materially. Re-measure + retire per the contract above._\n"
        + delta_marker(key, delta)
    )
    title = f"score-signal: {label} debt rose +{delta} ({frm}->{to})"
    return {
        "action": "refresh",
        "number": number,
        "title": title,
        "comment": comment,
        "key": key,
        "delta": delta,
        "noted_delta": noted,
    }


def plan_issues(payload: dict[str, Any], open_keys: set[str], *,
                min_delta: int, max_issues: int, today: str,
                available: set[str] | None = None,
                open_index: dict[str, dict[str, Any]] | None = None,
                worsen_delta: int = DEFAULTS["worsen_delta"],
                ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """relevance filter -> dedup vs OPEN issues -> CAP, with a REFRESH branch (#981).
    Returns (new_issues, refreshes, skip_stats). Deterministic: regressions are
    worst-first, deduped by key, then capped.

    An already-open metric is NOT silently dropped: if its current delta exceeds the
    delta the open issue last recorded by >= worsen_delta, it is planned as a REFRESH
    (at most one per metric per run — there is at most one open issue per key) instead
    of being skipped. `open_index` carries the per-key {number, noted_delta} needed to
    decide; when it is omitted the behavior is the original skip-only dedup, so callers
    that pass only `open_keys` keep the same semantics with an empty refresh list."""
    available = available if available is not None else set()
    open_index = open_index or {}
    stats = {"already-open": 0, "below-min-delta": 0, "within-run-dup": 0,
             "over-cap": 0, "refresh": 0, "open-but-flat": 0}
    commit = str(payload.get("commit") or "")
    tools = tool_by_key()
    out: list[dict[str, Any]] = []
    refreshes: list[dict[str, Any]] = []
    seen_run: set[str] = set()
    # All risen metrics (delta >= 1); the min-delta filter is accounted as a skip so
    # the stats honestly report what was dropped and why, not silently swallowed.
    for cand in regressions(payload, min_delta=1):
        key = cand["key"]
        if key in seen_run:
            stats["within-run-dup"] += 1
            continue
        seen_run.add(key)
        if cand["delta"] < min_delta:
            stats["below-min-delta"] += 1
            continue
        if key in open_keys:
            stats["already-open"] += 1
            # #981: an already-open metric that worsened materially is refreshed in
            # place rather than dropped. Without an index entry (older index that
            # only knew keys) we keep the original skip — no number to comment on.
            entry = open_index.get(key)
            if entry is not None and entry.get("number") is not None:
                noted = int(entry.get("noted_delta", 0))
                if cand["delta"] - noted >= worsen_delta:
                    refreshes.append(render_refresh(
                        cand, int(entry["number"]), noted, today))
                    stats["refresh"] += 1
                else:
                    stats["open-but-flat"] += 1
            continue
        out.append(render_issue(cand, commit, today, tools, available))

    if len(out) > max_issues:
        stats["over-cap"] = len(out) - max_issues
        out = out[:max_issues]
    return out, refreshes, stats


# ============================================================================
# I/O boundary — gh + the control pane. Thin wrappers so the logic above stays
# testable.
# ============================================================================
def gh_json(args: list[str], timeout: int = 60) -> Any:
    """Run a `gh` subcommand that emits JSON; return the parsed value. Raises
    RuntimeError on non-zero exit / TimeoutExpired on a hang — both caught by the
    caller so a stuck CLI can't wedge the run."""
    proc = subprocess.run(["gh", *args], capture_output=True, text=True,
                          encoding="utf-8", timeout=timeout,
                          creationflags=_win_creationflags())
    if proc.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} -> {proc.returncode}: "
                           f"{proc.stderr.strip()[:300]}")
    out = proc.stdout.strip()
    return json.loads(out) if out else []


def fetch_open_issues(limit: int) -> list[dict[str, Any]]:
    """Open issues carrying the score-signal label — the dedup index. Scoping to the
    tool's OWN label bounds the index to the ~one-per-scorecard issues it could have
    filed, so dedup stays EXACT no matter how large the total open backlog grows
    (an unlabelled repo-wide fetch could drop an older still-open marker past the
    limit and re-file a duplicate). `gh issue list --label X` returns [] cleanly when
    the label does not exist yet, so this is safe on a fresh repo."""
    return gh_json([
        "issue", "list", "--state", "open", "--label", SIGNAL_LABEL,
        "--limit", str(limit), "--json", "number,title,body",
    ])


def ensure_labels() -> None:
    """Idempotently create the two marker labels so `gh issue create` never fails on
    a missing label. Best-effort but NOT silent (an auth/permission failure here
    would resurface as a confusing per-issue 'label not found')."""
    wanted = [
        (SIGNAL_LABEL, "1d76db",
         "Auto-filed by score-signal (tools/score_signal.py) from a CI scorecard "
         "regression; needs a worker"),
        (DEBT_LABEL, "fbca04a", "Tracked quality/scorecard debt to retire"),
    ]
    for name, color, desc in wanted:
        try:
            proc = subprocess.run(
                ["gh", "label", "create", name, "--color", color,
                 "--description", desc, "--force"],
                capture_output=True, text=True, encoding="utf-8", timeout=30,
                creationflags=_win_creationflags())
        except (OSError, subprocess.TimeoutExpired) as e:
            print(f"warning: could not run `gh label create {name}`: {e}",
                  file=sys.stderr)
            continue
        if proc.returncode != 0:
            print(f"warning: could not ensure '{name}' label "
                  f"(issue creation may fail): {proc.stderr.strip()[:200]}",
                  file=sys.stderr)


def create_issue(issue: dict[str, Any]) -> str:
    """`gh issue create` -> the new issue URL."""
    args = ["issue", "create", "--title", issue["title"], "--body", issue["body"]]
    for lab in issue["labels"]:
        args += ["--label", lab]
    proc = subprocess.run(["gh", *args], capture_output=True, text=True,
                          encoding="utf-8", creationflags=_win_creationflags())
    if proc.returncode != 0:
        raise RuntimeError(f"gh issue create -> {proc.returncode}: "
                           f"{proc.stderr.strip()[:300]}")
    return proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""


def issue_contract_draft(issue: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": issue["title"],
        "body": issue["body"],
        "labels": [{"name": lab} for lab in issue.get("labels", [])],
    }


def check_issue_contract(issue: dict[str, Any], *, dedupe_cap: int,
                         runner: Any = subprocess.run,
                         fak_bin: str | None = None) -> None:
    """Fail unless the exact issue body passes `fak issue contract` for live sync."""
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json",
                                         delete=False) as f:
            tmp = Path(f.name)
            json.dump([issue_contract_draft(issue)], f, ensure_ascii=False)
            f.write("\n")
        cmd = [
            fak_bin or os.environ.get("FAK_BIN", "fak"),
            "issue", "contract",
            "--from-issues", str(tmp),
            "--live", "--dedupe-checked", "--dedupe-cap", str(dedupe_cap),
            "--json",
        ]
        proc = runner(cmd, capture_output=True, text=True, encoding="utf-8",
                      timeout=60)
    finally:
        if tmp is not None:
            try:
                tmp.unlink()
            except OSError:
                pass
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = detail[-1] if detail else "issue contract refused"
        raise RuntimeError(f"fak issue contract -> {proc.returncode}: {tail[:300]}")


def refresh_issue(refresh: dict[str, Any]) -> None:
    """Apply a planned REFRESH (#981): post the dated comment, then bump the title to
    the current severity. The comment carries the new noted-delta marker, so a later
    flat run reads the bumped value and does not re-comment. A failure on either step
    raises RuntimeError, surfaced per-refresh by the caller's error list."""
    num = str(refresh["number"])
    c = subprocess.run(
        ["gh", "issue", "comment", num, "--body", refresh["comment"]],
        capture_output=True, text=True, encoding="utf-8",
        creationflags=_win_creationflags())
    if c.returncode != 0:
        raise RuntimeError(f"gh issue comment {num} -> {c.returncode}: "
                           f"{c.stderr.strip()[:300]}")
    e = subprocess.run(
        ["gh", "issue", "edit", num, "--title", refresh["title"]],
        capture_output=True, text=True, encoding="utf-8",
        creationflags=_win_creationflags())
    if e.returncode != 0:
        raise RuntimeError(f"gh issue edit {num} -> {e.returncode}: "
                           f"{e.stderr.strip()[:300]}")


def load_payload(from_arg: str, root: Path, timeout: int) -> dict[str, Any]:
    """The folded control-pane payload: read it from --from (a file or '-' for
    stdin), else fold it live by invoking tools/scorecard_control_pane.py --json.
    Raises RuntimeError on any failure so the caller exits 2 cleanly."""
    if from_arg == "-":
        raw = sys.stdin.read()
    elif from_arg:
        raw = Path(from_arg).read_text(encoding="utf-8")
    else:
        script = root / "tools" / "scorecard_control_pane.py"
        if not script.exists():
            raise RuntimeError(f"missing {script}")
        proc = subprocess.run(
            [sys.executable, str(script), "--json"],
            cwd=str(root), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout)
        raw = proc.stdout
        if not raw.strip():
            tail = (proc.stderr or "").strip().splitlines()[-1:] or [""]
            raise RuntimeError(f"control pane emitted no JSON (exit "
                               f"{proc.returncode}): {tail[0][:200]}")
    try:
        payload = json.loads(raw)
    except ValueError as e:
        raise RuntimeError(f"control-pane payload is not JSON: {e}") from e
    if not isinstance(payload, dict):
        raise RuntimeError("control-pane payload is not an object")
    return payload


# ============================================================================
# Driver.
# ============================================================================
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="score-signal: a CI scorecard regression -> a deduped GitHub issue.")
    ap.add_argument("--workspace", default=".",
                    help="repo root. Default: cwd.")
    ap.add_argument("--from", dest="from_arg", default="",
                    help="read a pre-folded control-pane JSON from this file "
                         "('-' = stdin). Default: fold it live. NOTE: a STALE pane "
                         "under --live can re-file an already-resolved regression; "
                         "prefer a live fold when arming --live.")
    ap.add_argument("--min-delta", type=int,
                    help=f"a metric must rise by at least this to file "
                         f"(default {DEFAULTS['min_delta']}).")
    ap.add_argument("--worsen-delta", type=int,
                    help=f"an already-open metric must worsen by at least this past "
                         f"the delta its ticket last recorded to REFRESH it in place "
                         f"(default {DEFAULTS['worsen_delta']}); below it the open "
                         f"ticket is left untouched (#981).")
    ap.add_argument("--max-issues", type=int,
                    help=f"hard cap on issues filed, worst-first "
                         f"(default {DEFAULTS['max_issues']}).")
    ap.add_argument("--live", action="store_true",
                    help="actually create the issues (default: dry-run).")
    ap.add_argument("--json", action="store_true", help="machine-readable output.")
    ap.add_argument("--timeout", type=int, default=300,
                    help="control-pane fold timeout seconds (when folding live).")
    args = ap.parse_args(argv)

    root = Path(args.workspace).resolve()
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    min_delta = args.min_delta if args.min_delta is not None else DEFAULTS["min_delta"]
    max_issues = args.max_issues if args.max_issues is not None else DEFAULTS["max_issues"]
    worsen_delta = (args.worsen_delta if args.worsen_delta is not None
                    else DEFAULTS["worsen_delta"])

    # A supplied pane is a point-in-time snapshot; under --live a stale one could
    # re-file a regression already retired+closed (dedup is open-only by design).
    # Warn loudly — the automated workflow always folds a fresh pane, so this only
    # fires on a manual stale-snapshot arm.
    if args.live and args.from_arg:
        print("warning: --from with --live trusts a possibly STALE pane; a regression "
              "already retired and closed could be re-filed. Prefer a live fold when "
              "arming --live.", file=sys.stderr)

    try:
        payload = load_payload(args.from_arg, root, args.timeout)
    except (OSError, RuntimeError, subprocess.SubprocessError) as e:
        print(f"refuse: could not load the control-pane payload: {e}",
              file=sys.stderr)
        return 2

    # The OPEN-issue dedup index is mandatory: with no way to know what is already
    # tracked, --live could re-file an open regression every tick. Refuse rather
    # than risk a storm (same contract as idea_scout).
    errors: list[str] = []
    try:
        issues = fetch_open_issues(DEFAULTS["issue_scan_limit"])
        open_keys = open_issue_keys(issues)
        open_idx = open_issue_index(issues)
    except Exception as e:  # noqa: BLE001
        print(f"refuse: cannot fetch open issues for the dedup index ({e})",
              file=sys.stderr)
        return 2

    available = available_skills(root)
    to_file, refreshes, skip_stats = plan_issues(
        payload, open_keys, min_delta=min_delta, max_issues=max_issues, today=today,
        available=available, open_index=open_idx, worsen_delta=worsen_delta)

    # The RED-ratchet/empty-plan reconciliation: the portfolio can regress with no
    # per-metric rise attributable (a new/unpinned scorecard adds debt but has no
    # baseline prior, so it is excluded from early_warning). Surface that gap rather
    # than leaving an empty plan next to a RED ratchet unexplained.
    trend = payload.get("trend") or {}
    note = ""
    if trend.get("direction") == "regressed" and not regressions(payload, min_delta=1):
        td = trend.get("total_delta")
        td_s = f"+{td}" if isinstance(td, int) and td > 0 else "?"
        note = (f"portfolio regressed ({td_s}) but no per-metric rise is attributable "
                f"— likely a new/unpinned scorecard; re-pin the baseline "
                f"(`python tools/scorecard_control_pane.py --pin`) or investigate")

    filed: list[dict[str, Any]] = []
    refreshed: list[dict[str, Any]] = []
    if args.live and (to_file or refreshes):
        checked_to_file = []
        for issue in to_file:
            try:
                check_issue_contract(issue, dedupe_cap=DEFAULTS["issue_scan_limit"])
            except Exception as e:  # noqa: BLE001
                errors.append(f"contract[{issue['key']}]: {e}")
                continue
            checked_to_file.append(issue)
        if checked_to_file or refreshes:
            ensure_labels()
        for issue in checked_to_file:
            try:
                url = create_issue(issue)
            except Exception as e:  # noqa: BLE001
                errors.append(f"create[{issue['key']}]: {e}")
                continue
            filed.append({**issue, "issue_url": url})
        for r in refreshes:
            try:
                refresh_issue(r)
            except Exception as e:  # noqa: BLE001
                errors.append(f"refresh[{r['key']}#{r['number']}]: {e}")
                continue
            refreshed.append(r)

    result = {
        "schema": SCHEMA,
        "date": today,
        "mode": "live" if args.live else "dry-run",
        "commit": payload.get("commit", ""),
        "trend_direction": trend.get("direction", ""),
        "open_signal_issues": sorted(open_keys),
        "skipped": skip_stats,
        "note": note,
        "planned": [
            {"title": i["title"], "key": i["key"], "delta": i["delta"],
             "labels": i["labels"]}
            for i in to_file],
        "refreshes": [
            {"number": r["number"], "key": r["key"], "delta": r["delta"],
             "noted_delta": r["noted_delta"], "title": r["title"]}
            for r in refreshes],
        "filed": [{"title": f["title"], "key": f["key"],
                   "issue_url": f.get("issue_url", "")} for f in filed],
        "refreshed": [{"number": r["number"], "key": r["key"],
                       "delta": r["delta"]} for r in refreshed],
        "errors": errors,
    }

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    # Human report.
    print(f"score-signal {today} — {result['mode']}  @{result['commit']}  "
          f"(portfolio trend: {result['trend_direction'] or 'unknown'})")
    if trend.get("direction") == "unpinned":
        print("  → baseline unpinned; run `python tools/scorecard_control_pane.py "
              "--pin` first. Nothing to signal.")
        return 0
    sk = ", ".join(f"{k}={v}" for k, v in skip_stats.items() if v) or "none"
    print(f"  deduped/dropped: {sk}")
    if result["open_signal_issues"]:
        print(f"  already tracked (open): {', '.join(result['open_signal_issues'])}")
    if not to_file:
        print("  → no NEW actionable score regression. Nothing to file.")
    else:
        verb = "FILED" if args.live else "would file"
        print(f"  → {verb} {len(to_file)} issue(s) (cap {max_issues}, "
              f"min-delta {min_delta}):")
        for i in to_file:
            f = next((x for x in filed if x["key"] == i["key"]), None)
            mark = ""
            if args.live:
                mark = f"  {f['issue_url']}" if f else "  (create failed)"
            print(f"     [+{i['delta']:>2}] {i['title']}{mark}")
    if refreshes:
        verb = "REFRESHED" if args.live else "would refresh"
        print(f"  → {verb} {len(refreshes)} open ticket(s) that worsened "
              f"(worsen-delta {worsen_delta}):")
        for r in refreshes:
            done = any(x["number"] == r["number"] for x in refreshed)
            mark = ""
            if args.live:
                mark = "  ok" if done else "  (refresh failed)"
            print(f"     refresh #{r['number']} {r['key']} "
                  f"(+{r['noted_delta']} → +{r['delta']}){mark}")
    if note:
        print(f"  note: {note}")
    if errors:
        print("  errors:")
        for e in errors:
            print(f"     ! {e}")
    if not args.live and to_file:
        print("\n  dry-run — file these for real with:  "
              "python tools/score_signal.py --live")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
