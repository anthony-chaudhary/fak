#!/usr/bin/env python3
r"""trajctl-signal — turn a TRAJECTORY-HEALTH regression (a trajctl calibration
collapse, a stall-rate spike, a nudge-outcome regression) into a specific, deduped
GitHub issue the dispatch loop can resolve. The trajectory-health sibling of
tools/score_signal.py (#2574, epic #2533).

  score_signal.py   : scorecard control-pane debt rose        -> a deduped issue
  gate_signal.py    : a non-scorecard CI finding (red)         -> a deduped issue
  bench_signal.py   : a benchmark tok/s dropped                -> a deduped issue
  trajctl_signal.py : a trajctl health fold regressed (this)   -> a deduped issue

Same "turn a signal into an actionable item, IF RELEVANT" discipline as the
siblings: relevance filter -> label-scoped per-regression marker dedup ->
worst-drop-first hard CAP (--max-issues) -> DRY-RUN by default, `--live` the
explicit arm. Only the input differs: trajectory-control's own health metrics.

THE INPUT — a trajctl HEALTH FOLD envelope (schema `fak-trajctl-health/1`), read
from `--from` (a file or '-' for stdin). There is no live fold to default to: the
producer that folds the `fak-trajctl/1` ledger into this envelope is the epic's
source work (#2566 the comparison metric, #2561 the second source); this feeder is
the consumer seam that turns the fold into filed work. Two folds are read — the
CALIBRATION fold and the METRICS fold — nothing else (the issue's out-of-scope):

  {
    "schema": "fak-trajctl-health/1",
    "commit": "<short sha the fold was taken at>",
    "window": "7d",
    "calibration": {"methods": [
      {"method": "judge-verdict", "version": "2",
       "agreement": 0.31, "baseline_agreement": 0.82, "samples": 40}]},
    "metrics": {
      "stall_rate": {"scope": "fleet", "current": 0.42, "baseline": 0.08,
                     "sessions": 25},
      "nudge_outcomes": [
        {"nudge": "re-anchor", "success_rate": 0.35,
         "baseline_success_rate": 0.75, "samples": 12}]}
  }

THE THREE REGRESSION KINDS (all rates in [0,1], each vs its pinned baseline):
  * calibration collapse — a scorer method's agreement-with-evidence DROPPED
    (`baseline_agreement - agreement >= --min-drop`). A collapsed method means the
    curve's witness rungs are lying; steering on them is harm.
  * stall-rate spike — the fraction of sampled curves signalling STALL ROSE
    (`current - baseline >= --min-drop`). A fleet-wide spike is a real regression
    in forward progress, not one session's bad day.
  * nudge-outcome regression — a steering nudge's success rate DROPPED
    (`baseline_success_rate - success_rate >= --min-drop`). A nudge that stopped
    working must be re-tuned or retired, not kept firing.

THE NOISE FLOORS — a row fires only when BOTH hold: the drop/rise >= --min-drop
(default 0.10, i.e. ten points of a 0..1 rate) AND the row carries at least
--min-samples observations (default 5) — a two-sample calibration "collapse" is a
measurement gap, not a regression. A missing/invalid current or baseline value is
SKIPPED (the fold owns its measurement gaps).

DEDUP — a trajectory-health regression is RECURRING work (re-file after a close).
Dedups against OPEN trajctl-signal-labelled issues only, by a stable per-regression
marker (`<!-- fak-trajctl-signal: <kind>:<subject> -->`): at most one OPEN issue
per regression key, re-fileable after a close. The label-scoped fetch bounds the
index EXACTLY. A hard CAP (--max-issues, worst-drop-first) keeps a multi-fold
regression from storming the tracker.

A RERUN DEDUPES TO AN UPDATE (the #2574 done condition, score-signal's #981
pattern): while an issue stays open, a rerun never files a duplicate — the open
marker dedups it. When the tracked regression WORSENED by >= --worsen-drop past
the drop the issue last recorded (the noted-drop marker), the rerun plans an
in-place UPDATE (a dated comment + a title bump) instead; a flat rerun mutates
nothing, so a scheduled rerun over an unchanged-but-open backlog is idempotent.

SAFE BY DEFAULT: dry-run. `--live` is the explicit opt-in that creates/updates.

    python tools/trajctl_signal.py --from health.json          # dry-run plan
    python tools/trajctl_signal.py --from health.json --json   # machine-readable
    python tools/trajctl_signal.py --from tools/trajctl_signal.data/regressed-health.json
    python tools/trajctl_signal.py --from health.json --max-issues 3 --live

Exit codes: 0 = ran clean (including "nothing to signal") · 2 = infra error (gh
missing / not authed / the envelope could not be read or parsed).
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


SCHEMA = "fak-trajctl-signal/1"
HEALTH_SCHEMA = "fak-trajctl-health/1"
SIGNAL_LABEL = "trajctl-signal"
OBS_LABEL = "observability"
MARKER_RE = re.compile(r"fak-trajctl-signal:\s*([A-Za-z0-9_:.@/\-]+)")
# The noted-drop marker (score-signal's #981 pattern): a second stable comment
# recording the drop (in basis points of the 0..1 rate) the issue currently
# reports, so a worsening regression on an already-open ticket can be detected
# and UPDATED — while a flat rerun never re-comments.
DROP_MARKER_RE = re.compile(r"fak-trajctl-signal-drop:\s*([A-Za-z0-9_:.@/\-]+)=(\d+)")

DEFAULTS = {
    "min_drop": 0.10,      # a rate must move by at least this (0..1) to be actionable
    "min_samples": 5,      # a row needs at least this many observations to fire
    "max_issues": 5,       # hard cap on issues filed per run (anti-storm)
    "issue_scan_limit": 800,  # open trajctl-signal issues fetched for the dedup index
    "worsen_drop": 0.05,   # an open ticket must worsen by >= this to UPDATE in place
}

# The owning substrate file per regression kind, in the `fak/internal/...`
# doc-link form the dispatch router (issue_lane_router._PATH_RE +
# path_matches_lane, which strips the `fak/` prefix) matches against the
# `trajctl` lane tree (dos.toml: trajctl = ["internal/trajctl/**"]) — the same
# convention score_signal's SOURCE_BY_KEY uses to route native-cmd tickets. This
# is the body's ONLY router-greppable path — the filer is named by basename — so
# the ticket path-confirms the trajctl lane cleanly (cf. gate_signal's note).
FIX_HINT_BY_KIND: dict[str, str] = {
    "calibration": "fak/internal/trajctl/audit.go",
    "stall-rate": "fak/internal/trajctl/curve.go",
    "nudge-outcome": "fak/internal/trajctl/steer.go",
}


# ============================================================================
# Pure helpers (no I/O) — unit-tested directly.
# ============================================================================
def _slug(text: str) -> str:
    """A stable, marker-safe slug: lowercase, non-[a-z0-9._-] -> '-', collapsed."""
    s = re.sub(r"[^a-z0-9._-]+", "-", str(text).lower()).strip("-")
    return re.sub(r"-{2,}", "-", s) or "x"


def _rate(v: Any) -> float | None:
    """A real 0..1 rate, else None — a missing/invalid value is a measurement gap
    the fold owns, never read as a 0 (which would fabricate a collapse)."""
    if isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= float(v) <= 1.0:
        return float(v)
    return None


def _sample_count(row: dict[str, Any]) -> int:
    """The row's observation count (`samples`, or `sessions` for the stall fold)."""
    for k in ("samples", "sessions"):
        v = row.get(k)
        if isinstance(v, int) and not isinstance(v, bool) and v >= 0:
            return v
    return 0


def drop_bp(drop: float) -> int:
    """A drop as integer basis points of the 0..1 rate (0.51 -> 5100) — the stable
    integer the noted-drop marker records so a worsening compare never bit-rots on
    float formatting."""
    return int(round(drop * 10000))


def regressions(payload: dict[str, Any], *, min_drop: float,
                min_samples: int) -> list[dict[str, Any]]:
    """Both health folds -> a uniform list of actionable regressions, worst-drop
    first. Pure over the envelope; the relevance filter lives here: a row fires only
    past BOTH the --min-drop floor and the --min-samples evidence floor, and a row
    with a missing current/baseline rate is skipped. Nothing else in the envelope is
    read (new signal kinds are the issue's out-of-scope)."""
    out: list[dict[str, Any]] = []
    commit = str(payload.get("commit") or "")
    window = str(payload.get("window") or "")

    def add(kind: str, subject: str, label: str, current: float, baseline: float,
            drop: float, samples: int, direction: str) -> None:
        if drop < min_drop or samples < min_samples:
            return
        out.append({
            "kind": kind,
            "key": f"{kind}:{subject}",
            "label": label,
            "current": current,
            "baseline": baseline,
            "drop": drop,
            "direction": direction,
            "samples": samples,
            "commit": commit,
            "window": window,
        })

    # Fold 1: CALIBRATION — a scorer method's agreement-with-evidence collapsed.
    cal = payload.get("calibration") or {}
    methods = cal.get("methods") if isinstance(cal, dict) else None
    for row in methods if isinstance(methods, list) else []:
        if not isinstance(row, dict):
            continue
        cur = _rate(row.get("agreement"))
        base = _rate(row.get("baseline_agreement"))
        if cur is None or base is None:
            continue
        method = str(row.get("method") or "method")
        version = str(row.get("version") or "0")
        add("calibration", f"{_slug(method)}@{_slug(version)}",
            f"calibration of {method}@v{version}",
            cur, base, base - cur, _sample_count(row), "collapsed")

    # Fold 2: METRICS — the stall-rate spike + the nudge-outcome regressions.
    metrics = payload.get("metrics") or {}
    if isinstance(metrics, dict):
        sr = metrics.get("stall_rate")
        if isinstance(sr, dict):
            cur = _rate(sr.get("current"))
            base = _rate(sr.get("baseline"))
            if cur is not None and base is not None:
                scope = str(sr.get("scope") or "fleet")
                add("stall-rate", _slug(scope), f"stall rate ({scope})",
                    cur, base, cur - base, _sample_count(sr), "spiked")
        nudges = metrics.get("nudge_outcomes")
        for row in nudges if isinstance(nudges, list) else []:
            if not isinstance(row, dict):
                continue
            cur = _rate(row.get("success_rate"))
            base = _rate(row.get("baseline_success_rate"))
            if cur is None or base is None:
                continue
            nudge = str(row.get("nudge") or "nudge")
            add("nudge-outcome", _slug(nudge), f"{nudge} nudge outcome",
                cur, base, base - cur, _sample_count(row), "regressed")

    # Worst drop first (then by key for a stable, deterministic order).
    out.sort(key=lambda r: (-r["drop"], r["key"]))
    return out


def marker(key: str) -> str:
    """The load-bearing dedup anchor: one stable HTML-comment per regression key."""
    return f"<!-- fak-trajctl-signal: {key} -->"


def drop_marker(key: str, drop: float) -> str:
    """The noted-drop anchor: records the drop (basis points) the issue currently
    reports so a later worsening can be detected and a flat rerun never re-comments."""
    return f"<!-- fak-trajctl-signal-drop: {key}={drop_bp(drop)} -->"


def open_issue_keys(issues: list[dict[str, Any]]) -> set[str]:
    """The set of regression keys already tracked by an OPEN trajctl-signal issue."""
    keys: set[str] = set()
    for iss in issues:
        for m in MARKER_RE.findall(iss.get("body") or ""):
            keys.add(m.strip())
    return keys


_TITLE_RATES_RE = re.compile(r"\((\d(?:\.\d+)?) -> (\d(?:\.\d+)?)\)")


def open_issue_index(issues: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """key -> {number, noted_drop_bp} for every OPEN trajctl-signal issue. The noted
    drop is the MAX of the body's drop-marker (last wins) and the drop parsed from
    the title's `(base -> current)` numbers. The title arm matters after a LIVE
    update: the new drop marker lands in a COMMENT (which the dedup fetch does not
    return), but the title IS bumped — reading it back stops an already-updated
    issue from being re-updated every run. An issue with neither floors at 0 —
    conservative, so a real worsening still trips the threshold instead of hiding."""
    idx: dict[str, dict[str, Any]] = {}
    for iss in issues:
        body = iss.get("body") or ""
        keys = [m.strip() for m in MARKER_RE.findall(body)]
        if not keys:
            continue
        key = keys[-1]
        noted = 0
        for k, v in DROP_MARKER_RE.findall(body):
            if k.strip() == key:
                noted = int(v)  # last wins
        tm = _TITLE_RATES_RE.search(str(iss.get("title") or ""))
        if tm:
            noted = max(noted, drop_bp(abs(float(tm.group(1)) - float(tm.group(2)))))
        idx[key] = {"number": iss.get("number"), "noted_drop_bp": noted}
    return idx


def render_issue(cand: dict[str, Any], today: str) -> dict[str, Any]:
    """Build the {title, body, labels} an issue is created from. The per-regression
    marker is the load-bearing dedup anchor; the body is CONTRACT-READY (the same
    stable sections the sibling feeders render, so `fak issue contract` passes) and
    names the owning internal/trajctl substrate file as its ONLY router-greppable
    path, so the dispatch router path-confirms the trajctl lane."""
    kind = cand["kind"]
    key = cand["key"]
    label = cand["label"]
    cur, base, drop = cand["current"], cand["baseline"], cand["drop"]
    direction = cand["direction"]
    samples = cand["samples"]
    commit = cand.get("commit") or ""
    window = cand.get("window") or ""
    fix_hint = FIX_HINT_BY_KIND.get(kind, "fak/internal/trajctl/trajctl.go")
    pts = drop * 100

    title = (f"trajctl-signal: {label} {direction} "
             f"({base:.2f} -> {cur:.2f})")

    kind_note = {
        "calibration": ("A collapsed calibration means this scorer method's verdicts "
                        "no longer agree with the evidence they cite — every curve it "
                        "feeds is untrustworthy, and steering on it is harm."),
        "stall-rate": ("A fleet-wide stall-rate spike means materially more sampled "
                       "curves signal STALL than the pinned baseline — forward "
                       "progress regressed across sessions, not in one bad run."),
        "nudge-outcome": ("A nudge whose success rate dropped is an actuator that "
                          "stopped working; keeping it firing burns the regime gate's "
                          "trust for zero corrective effect."),
    }.get(kind, "")

    do_lines = [
        f"1. Reproduce the fold: re-run the trajctl health fold over the same "
        f"window (`{window or 'the fold window'}`) and confirm `{key}` still reads "
        f"{cur:.2f} vs its {base:.2f} baseline.",
        f"2. Find the smallest responsible cause in the owning substrate "
        f"(`{fix_hint}`) — a scorer/method change, a curve-fold change, a steering "
        f"policy change — and fix or explicitly re-baseline it; change nothing the "
        f"metric does not cover.",
        "3. Re-fold to PROVE the regression retired (or record the accepted "
        "trade-off with the re-pinned baseline).",
        "4. Ship a commit citing this issue's `#N` in the subject + a "
        "`(fak trajctl)` trailer, so the witness can bind and close it.",
    ]

    body = (
        f"> Auto-filed by **trajctl-signal** (the `trajctl_signal.py` feeder, "
        f"{today}) from a trajectory-health regression in the trajctl health fold"
        + (f" @{commit}" if commit else "")
        + ". A measured regression vs the pinned baseline — **needs a worker**; "
        "close as `wontfix` and re-pin the baseline if the change is intentional.\n\n"
        f"**Regression:** `{label}` ({key})\n"
        f"**Change:** {direction} **{pts:.0f}pt** ({base:.2f} -> {cur:.2f}) vs the "
        f"pinned baseline over {samples} observation(s)"
        + (f", window `{window}`" if window else "")
        + ".\n\n"
        f"**Why this fired:** the move cleared both the --min-drop floor and the "
        f"--min-samples evidence floor — a real measured regression, not wobble. "
        f"{kind_note}\n\n"
        "**What to do**\n"
        + "\n".join(do_lines)
        + "\n\n### Parent context\n"
        "trajctl-signal trajectory-health regression from the trajctl health fold "
        "(epic #2533).\n\n"
        "### Current state\n"
        + f"`{label}` ({key}) {direction} from {base:.2f} to {cur:.2f} against the "
        "pinned baseline"
        + (f" @{commit}" if commit else "")
        + f" over {samples} observation(s).\n\n"
        "### Why this is next\n"
        "A regressed trajectory-health metric means the fleet's own steering "
        "substrate is less trustworthy; every curve and nudge downstream of it "
        "inherits the doubt until this is retired or explicitly accepted.\n\n"
        "### Working spine\n"
        + f"Retire the `{key}` regression before steering on the affected curves.\n\n"
        "### Priority context\n"
        "Working path: trajctl health fold -> trajctl-signal issue -> scoped fix -> "
        "re-folded metric back at baseline. Current blocker: the health metric "
        "regressed. Unblocks: trajectory control stays trustworthy as a steering "
        "input. Not polish: this fixes a measured regression before cosmetic work.\n\n"
        "### In scope\n"
        + f"Investigate the `{key}` regression, fix the smallest responsible cause "
        "in the owning substrate, and re-fold the health metrics to prove it.\n\n"
        "### Out of scope\n"
        "Do not retune unrelated scorers/nudges, rewrite the health fold, or re-pin "
        "the baseline without recording why the regression is accepted.\n\n"
        "### Done condition\n"
        + f"The `{key}` metric is back at or within --min-drop of its pinned "
        "baseline, or the baseline is intentionally re-pinned with the reason "
        "recorded.\n\n"
        "### Witness\n"
        "The re-run trajctl health fold shows the regression retired or accepted.\n\n"
        "### Acceptance gate\n"
        "Re-run the trajctl health fold for this window; the regressed metric no "
        "longer clears the --min-drop floor.\n\n"
        "### Lane\n"
        "trajctl\n\n"
        "### Path hints\n"
        + f"- `{fix_hint}`\n\n"
        "### Boundary notes\n"
        "Public trajectory-health regression only; do not include private operator "
        "telemetry or session transcripts.\n\n"
        "### Closure binding\n"
        "Resolving commit cites this issue's #N in the subject and carries "
        "`(fak trajctl)`.\n"
        "\n\n---\n"
        "_trajctl-signal is the trajectory-health sibling of score-signal (the "
        "`score_signal.py` feeder). After this issue is closed, a future regression "
        "of the same metric files a FRESH issue — only on an armed `--live` run. "
        "While it stays open, a materially worse drop UPDATES it in place rather "
        "than filing a duplicate._\n"
        + marker(key)
        + "\n"
        + drop_marker(key, drop)
    )

    return {
        "title": title, "body": body, "labels": [SIGNAL_LABEL, OBS_LABEL],
        "key": key, "kind": kind, "drop": drop,
    }


def render_update(cand: dict[str, Any], number: int, noted_bp: int, today: str,
                  ) -> dict[str, Any]:
    """Build the in-place UPDATE of an already-open issue whose tracked regression
    worsened: a dated comment with the old->new numbers and the new noted-drop
    marker (so a later flat rerun never re-comments), plus the bumped title.
    Strictly bounded — the planner emits at most one update per key per run."""
    key = cand["key"]
    label = cand["label"]
    cur, base, drop = cand["current"], cand["baseline"], cand["drop"]
    comment = (
        f"> **trajctl-signal update** ({today}) — the tracked regression worsened.\n\n"
        f"`{label}` ({key}) now reads **{cur:.2f}** vs its {base:.2f} baseline — a "
        f"**{drop * 100:.0f}pt** {cand['direction']} move, up from the "
        f"**{noted_bp / 100:.0f}pt** this issue last recorded. The underlying health "
        f"metric degraded while this ticket stayed open; the title is bumped to the "
        f"current severity.\n\n"
        f"_Bounded: updated at most once per run and only when the drop grew "
        f"materially. Re-fold + retire per the contract above._\n"
        + drop_marker(key, drop)
    )
    title = (f"trajctl-signal: {label} {cand['direction']} "
             f"({base:.2f} -> {cur:.2f})")
    return {
        "action": "update", "number": number, "title": title, "comment": comment,
        "key": key, "drop": drop, "noted_drop_bp": noted_bp,
    }


def plan_issues(regs: list[dict[str, Any]], open_keys: set[str], *,
                max_issues: int, today: str,
                open_index: dict[str, dict[str, Any]] | None = None,
                worsen_drop: float = DEFAULTS["worsen_drop"],
                ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """relevance (already in regressions()) -> dedup vs OPEN issues -> CAP, with the
    dedupe-to-UPDATE branch. Returns (new_issues, updates, skip_stats).
    Deterministic: regressions are worst-drop-first, deduped by key, then capped.

    An already-open key is NEVER re-filed (the rerun dedupes); when its current drop
    exceeds the drop the open issue last recorded by >= worsen_drop it is planned as
    an in-place UPDATE, else it is left untouched (idempotent flat rerun). Without
    an `open_index` the behavior is skip-only dedup (nothing to comment on)."""
    open_index = open_index or {}
    stats = {"already-open": 0, "within-run-dup": 0, "over-cap": 0,
             "update": 0, "open-but-flat": 0}
    out: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []
    seen_run: set[str] = set()
    for cand in regs:
        key = cand["key"]
        if key in seen_run:
            stats["within-run-dup"] += 1
            continue
        seen_run.add(key)
        if key in open_keys:
            stats["already-open"] += 1
            entry = open_index.get(key)
            if entry is not None and entry.get("number") is not None:
                noted = int(entry.get("noted_drop_bp", 0))
                if drop_bp(cand["drop"]) - noted >= drop_bp(worsen_drop):
                    updates.append(render_update(
                        cand, int(entry["number"]), noted, today))
                    stats["update"] += 1
                else:
                    stats["open-but-flat"] += 1
            continue
        out.append(render_issue(cand, today))
    if len(out) > max_issues:
        stats["over-cap"] = len(out) - max_issues
        out = out[:max_issues]
    return out, updates, stats


# ============================================================================
# I/O boundary — gh + the envelope. Thin wrappers so the logic above stays
# testable.
# ============================================================================
def gh_json(args: list[str], timeout: int = 60) -> Any:
    """Run a `gh` subcommand that emits JSON; return the parsed value."""
    proc = subprocess.run(["gh", *args], capture_output=True, text=True,
                          encoding="utf-8", timeout=timeout,
                          creationflags=_win_creationflags())
    if proc.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} -> {proc.returncode}: "
                           f"{proc.stderr.strip()[:300]}")
    out = proc.stdout.strip()
    return json.loads(out) if out else []


def fetch_open_issues(limit: int) -> list[dict[str, Any]]:
    """Open issues carrying the trajctl-signal label — the dedup index. Scoping to
    the tool's OWN label bounds the index EXACTLY regardless of total backlog size
    (cf. score_signal.fetch_open_issues)."""
    return gh_json([
        "issue", "list", "--state", "open", "--label", SIGNAL_LABEL,
        "--limit", str(limit), "--json", "number,title,body",
    ])


def ensure_labels() -> None:
    """Idempotently create the marker labels so `gh issue create` never fails."""
    wanted = [
        (SIGNAL_LABEL, "5319e7",
         "Auto-filed by trajctl-signal (tools/trajctl_signal.py) from a trajectory-"
         "health regression; needs a worker"),
        (OBS_LABEL, "c5def5", "Fleet observability surface"),
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


def update_issue(update: dict[str, Any]) -> None:
    """Apply a planned UPDATE: post the dated comment (carrying the new noted-drop
    marker), then bump the title to the current severity. A failure on either step
    raises RuntimeError, surfaced per-update by the caller's error list."""
    num = str(update["number"])
    c = subprocess.run(
        ["gh", "issue", "comment", num, "--body", update["comment"]],
        capture_output=True, text=True, encoding="utf-8",
        creationflags=_win_creationflags())
    if c.returncode != 0:
        raise RuntimeError(f"gh issue comment {num} -> {c.returncode}: "
                           f"{c.stderr.strip()[:300]}")
    e = subprocess.run(
        ["gh", "issue", "edit", num, "--title", update["title"]],
        capture_output=True, text=True, encoding="utf-8",
        creationflags=_win_creationflags())
    if e.returncode != 0:
        raise RuntimeError(f"gh issue edit {num} -> {e.returncode}: "
                           f"{e.stderr.strip()[:300]}")


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


def load_envelope(from_arg: str) -> dict[str, Any]:
    """Read the trajctl health-fold envelope from --from (a file or '-' for stdin).
    Raises RuntimeError on any failure so the caller exits 2 cleanly."""
    if not from_arg:
        raise RuntimeError("--from is required (a fak-trajctl-health/1 envelope "
                           "file or '-' for stdin); there is no live fold to "
                           "default to — the fold producer is #2566/#2561's seam")
    raw = sys.stdin.read() if from_arg == "-" else Path(from_arg).read_text(
        encoding="utf-8")
    try:
        payload = json.loads(raw)
    except ValueError as e:
        raise RuntimeError(f"envelope is not JSON: {e}") from e
    if not isinstance(payload, dict):
        raise RuntimeError("envelope is not a JSON object")
    return payload


# ============================================================================
# Driver.
# ============================================================================
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="trajctl-signal: a trajectory-health regression -> a deduped "
                    "GitHub issue.")
    ap.add_argument("--from", dest="from_arg", default="",
                    help="read the fak-trajctl-health/1 envelope from this file "
                         "('-' = stdin). Required: there is no live fold yet.")
    ap.add_argument("--min-drop", type=float,
                    help=f"a rate must move by at least this (0..1) to file "
                         f"(default {DEFAULTS['min_drop']}).")
    ap.add_argument("--min-samples", type=int,
                    help=f"a row needs at least this many observations to fire "
                         f"(default {DEFAULTS['min_samples']}).")
    ap.add_argument("--worsen-drop", type=float,
                    help=f"an already-open ticket is UPDATED in place only when its "
                         f"drop grew by at least this past the drop it last "
                         f"recorded (default {DEFAULTS['worsen_drop']}).")
    ap.add_argument("--max-issues", type=int,
                    help=f"hard cap on issues filed, worst-drop-first "
                         f"(default {DEFAULTS['max_issues']}).")
    ap.add_argument("--live", action="store_true",
                    help="actually create/update the issues (default: dry-run).")
    ap.add_argument("--json", action="store_true", help="machine-readable output.")
    args = ap.parse_args(argv)

    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    min_drop = args.min_drop if args.min_drop is not None else DEFAULTS["min_drop"]
    min_samples = (args.min_samples if args.min_samples is not None
                   else DEFAULTS["min_samples"])
    worsen_drop = (args.worsen_drop if args.worsen_drop is not None
                   else DEFAULTS["worsen_drop"])
    max_issues = (args.max_issues if args.max_issues is not None
                  else DEFAULTS["max_issues"])

    try:
        payload = load_envelope(args.from_arg)
    except (OSError, RuntimeError) as e:
        print(f"refuse: could not load the trajctl health envelope: {e}",
              file=sys.stderr)
        return 2

    regs = regressions(payload, min_drop=min_drop, min_samples=min_samples)

    # The OPEN-issue dedup index is mandatory: with no way to know what is already
    # tracked, --live could re-file an open regression every tick. Refuse rather
    # than risk a storm (same contract as the sibling feeders).
    errors: list[str] = []
    try:
        issues = fetch_open_issues(DEFAULTS["issue_scan_limit"])
        open_keys = open_issue_keys(issues)
        open_idx = open_issue_index(issues)
    except Exception as e:  # noqa: BLE001
        print(f"refuse: cannot fetch open issues for the dedup index ({e})",
              file=sys.stderr)
        return 2

    to_file, updates, skip_stats = plan_issues(
        regs, open_keys, max_issues=max_issues, today=today,
        open_index=open_idx, worsen_drop=worsen_drop)

    filed: list[dict[str, Any]] = []
    updated: list[dict[str, Any]] = []
    if args.live and (to_file or updates):
        checked: list[dict[str, Any]] = []
        for issue in to_file:
            try:
                check_issue_contract(issue, dedupe_cap=DEFAULTS["issue_scan_limit"])
            except Exception as e:  # noqa: BLE001
                errors.append(f"contract[{issue['key']}]: {e}")
                continue
            checked.append(issue)
        if checked or updates:
            ensure_labels()
        for issue in checked:
            try:
                url = create_issue(issue)
            except Exception as e:  # noqa: BLE001
                errors.append(f"create[{issue['key']}]: {e}")
                continue
            filed.append({**issue, "issue_url": url})
        for u in updates:
            try:
                update_issue(u)
            except Exception as e:  # noqa: BLE001
                errors.append(f"update[{u['key']}#{u['number']}]: {e}")
                continue
            updated.append(u)

    result = {
        "schema": SCHEMA,
        "date": today,
        "mode": "live" if args.live else "dry-run",
        "commit": payload.get("commit", ""),
        "window": payload.get("window", ""),
        "regressions_total": len(regs),
        "open_signal_issues": sorted(open_keys),
        "skipped": skip_stats,
        "planned": [
            {"title": i["title"], "key": i["key"], "kind": i["kind"],
             "drop": round(i["drop"], 4), "labels": i["labels"]}
            for i in to_file],
        "updates": [
            {"number": u["number"], "key": u["key"], "drop": round(u["drop"], 4),
             "noted_drop_bp": u["noted_drop_bp"], "title": u["title"]}
            for u in updates],
        "filed": [{"title": f["title"], "key": f["key"],
                   "issue_url": f.get("issue_url", "")} for f in filed],
        "updated": [{"number": u["number"], "key": u["key"],
                     "drop": round(u["drop"], 4)} for u in updated],
        "errors": errors,
    }

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    # Human report.
    print(f"trajctl-signal {today} — {result['mode']}  "
          f"@{result['commit'] or '?'}  ({len(regs)} actionable regression(s) in "
          f"the health fold)")
    sk = ", ".join(f"{k}={v}" for k, v in skip_stats.items() if v) or "none"
    print(f"  deduped/dropped: {sk}")
    if result["open_signal_issues"]:
        print(f"  already tracked (open): {', '.join(result['open_signal_issues'])}")
    if not to_file:
        print(f"  → no NEW actionable trajectory-health regression past "
              f"{min_drop:.2f} drop / {min_samples} samples. Nothing to file.")
    else:
        verb = "FILED" if args.live else "would file"
        print(f"  → {verb} {len(to_file)} issue(s) (cap {max_issues}, "
              f"min-drop {min_drop:.2f}):")
        for i in to_file:
            f = next((x for x in filed if x["key"] == i["key"]), None)
            mark = ""
            if args.live:
                mark = f"  {f['issue_url']}" if f else "  (create failed)"
            print(f"     [-{i['drop'] * 100:>3.0f}pt] {i['title']}{mark}")
    if updates:
        verb = "UPDATED" if args.live else "would update"
        print(f"  → {verb} {len(updates)} open ticket(s) that worsened "
              f"(worsen-drop {worsen_drop:.2f}):")
        for u in updates:
            done = any(x["number"] == u["number"] for x in updated)
            mark = ""
            if args.live:
                mark = "  ok" if done else "  (update failed)"
            print(f"     update #{u['number']} {u['key']} "
                  f"({u['noted_drop_bp'] / 100:.0f}pt → {u['drop'] * 100:.0f}pt){mark}")
    if errors:
        print("  errors:")
        for e in errors:
            print(f"     ! {e}")
    if not args.live and (to_file or updates):
        print("\n  dry-run — file/update these for real with:  "
              "python tools/trajctl_signal.py --from <envelope> --live")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
