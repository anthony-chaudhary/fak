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
dedups against OPEN issues only, by a stable per-scorecard marker stamped in the
body (`<!-- fak-score-signal: <key> -->`): at most one OPEN issue per scorecard at a
time, re-fileable after a close. A hard CAP (--max-issues, worst-regression-first)
keeps even a multi-metric regression from storming the tracker.

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
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

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

DEFAULTS = {
    "min_delta": 1,        # a metric must rise by at least this to be actionable
    "max_issues": 5,       # hard cap on issues filed per run (anti-storm)
    "issue_scan_limit": 800,  # open issues fetched for the dedup index
}

# The owning RSI skill per scorecard key — the "do this next" hint baked into each
# issue so the worker (or a human) knows which repeatable pass retires the debt. A
# HINT, not a hard claim: the authoritative re-run command is the scorecard's own
# tool path (pulled from the control pane), and the generic /score-2x conductor is
# always offered as a fallback, so an unmapped/renamed key never files a false lead.
SKILL_BY_KEY: dict[str, str] = {
    "readme": "/refresh-readme",
    "code": "/quality-score",
    "appeal": "/appeal-score",
    "hygiene": "/repo-hygiene",
    "parity": "/industry-score",
    "agent": "/agent-readiness",
    "product": "/product-score",
    "persona": "/persona-score",
    "stability": "/stability-score",
    "slop": "/slop-score",
    "steer": "/steerability-score",
    "conflation": "/conflation-score",
    "disambiguation": "/disambiguation-score",
    "tokendefaults": "/token-defaults-score",
    "guard_rsi": "/guard-rsi-score",
    "dogfood": "/dogfood-score",
    "doc": "/curate-cluster",
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


def marker(key: str) -> str:
    """The load-bearing dedup anchor: one stable HTML-comment per scorecard key."""
    return f"<!-- fak-score-signal: {key} -->"


def open_issue_keys(issues: list[dict[str, Any]]) -> set[str]:
    """The set of scorecard keys already tracked by an OPEN issue (its body carries
    the marker). These are skipped — at most one open issue per scorecard."""
    keys: set[str] = set()
    for iss in issues:
        for m in MARKER_RE.findall(iss.get("body") or ""):
            keys.add(m.strip())
    return keys


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
                 tools: dict[str, str]) -> dict[str, Any]:
    """Build the {title, body, labels} an issue is created from. The per-scorecard
    marker is the load-bearing dedup anchor; the body names the owning skill + the
    exact re-measure command so the resolving worker has a closed contract."""
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

    skill = SKILL_BY_KEY.get(key)
    remeasure = tools.get(key, "")
    do_lines = []
    if skill:
        do_lines.append(f"1. Run the owning RSI pass: `{skill}` "
                        f"(or the generic `/score-2x {label}` conductor).")
    else:
        do_lines.append(f"1. Run the generic `/score-2x {label}` conductor (no "
                        f"dedicated skill is mapped for `{key}`).")
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
        "**What to do**\n"
        + "\n".join(do_lines)
        + "\n\n---\n"
        "_The scorecard family + its skills are listed in "
        "`tools/scorecard_control_pane.py`. This issue re-opens automatically on a "
        "future regression of the same metric only after it is closed._\n"
        + marker(key)
    )

    return {
        "title": title,
        "body": body,
        "labels": [SIGNAL_LABEL, DEBT_LABEL],
        "key": key,
        "delta": delta,
    }


def plan_issues(payload: dict[str, Any], open_keys: set[str], *,
                min_delta: int, max_issues: int, today: str,
                ) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """relevance filter -> dedup vs OPEN issues -> CAP. Returns (issues, skip_stats).
    Deterministic: regressions are worst-first, deduped by key, then capped."""
    stats = {"already-open": 0, "below-min-delta": 0, "within-run-dup": 0,
             "over-cap": 0}
    commit = str(payload.get("commit") or "")
    tools = tool_by_key()
    out: list[dict[str, Any]] = []
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
            continue
        out.append(render_issue(cand, commit, today, tools))

    if len(out) > max_issues:
        stats["over-cap"] = len(out) - max_issues
        out = out[:max_issues]
    return out, stats


# ============================================================================
# I/O boundary — gh + the control pane. Thin wrappers so the logic above stays
# testable.
# ============================================================================
def gh_json(args: list[str], timeout: int = 60) -> Any:
    """Run a `gh` subcommand that emits JSON; return the parsed value. Raises
    RuntimeError on non-zero exit / TimeoutExpired on a hang — both caught by the
    caller so a stuck CLI can't wedge the run."""
    proc = subprocess.run(["gh", *args], capture_output=True, text=True,
                          encoding="utf-8", timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} -> {proc.returncode}: "
                           f"{proc.stderr.strip()[:300]}")
    out = proc.stdout.strip()
    return json.loads(out) if out else []


def fetch_open_issues(limit: int) -> list[dict[str, Any]]:
    return gh_json([
        "issue", "list", "--state", "open", "--limit", str(limit),
        "--json", "number,title,body",
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
                capture_output=True, text=True, encoding="utf-8", timeout=30)
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
                          encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"gh issue create -> {proc.returncode}: "
                           f"{proc.stderr.strip()[:300]}")
    return proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""


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
                         "('-' = stdin). Default: fold it live.")
    ap.add_argument("--min-delta", type=int,
                    help=f"a metric must rise by at least this to file "
                         f"(default {DEFAULTS['min_delta']}).")
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
    except Exception as e:  # noqa: BLE001
        print(f"refuse: cannot fetch open issues for the dedup index ({e})",
              file=sys.stderr)
        return 2

    to_file, skip_stats = plan_issues(
        payload, open_keys, min_delta=min_delta, max_issues=max_issues, today=today)

    filed: list[dict[str, Any]] = []
    if args.live and to_file:
        ensure_labels()
        for issue in to_file:
            try:
                url = create_issue(issue)
            except Exception as e:  # noqa: BLE001
                errors.append(f"create[{issue['key']}]: {e}")
                continue
            filed.append({**issue, "issue_url": url})

    trend = payload.get("trend") or {}
    result = {
        "schema": SCHEMA,
        "date": today,
        "mode": "live" if args.live else "dry-run",
        "commit": payload.get("commit", ""),
        "trend_direction": trend.get("direction", ""),
        "open_signal_issues": sorted(open_keys),
        "skipped": skip_stats,
        "planned": [
            {"title": i["title"], "key": i["key"], "delta": i["delta"],
             "labels": i["labels"]}
            for i in to_file],
        "filed": [{"title": f["title"], "key": f["key"],
                   "issue_url": f.get("issue_url", "")} for f in filed],
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
