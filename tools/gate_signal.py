#!/usr/bin/env python3
r"""gate-signal — turn a NON-scorecard CI finding (a security-audit FAIL, a garden
dimension gone red, an advisory gate surfaced on a schedule) into a specific,
deduped, lane-routed GitHub issue the dispatch loop can resolve. The sibling of
tools/score_signal.py, which does the same for a SCORECARD regression.

  score_signal.py  : scorecard control-pane regression  -> a deduped issue
  gate_signal.py   : a non-scorecard CI finding (this)   -> a deduped issue
  idea_scout.py    : an EXTERNAL idea (arXiv + GitHub)    -> a deduped issue

score-signal proved the shape; gate-signal generalizes the SOURCE while keeping the
exact same "turn a CI signal into an actionable item, IF RELEVANT" discipline:
relevance filter -> label-scoped per-finding marker dedup -> worst-first hard CAP ->
DRY-RUN by default, `--live` the explicit arm. Only the input differs.

THE INPUT — a structured CI finding envelope, read from `--from` (a file or '-' for
stdin). Two envelope shapes are accepted, normalized to ONE internal finding:

  * the security-audit / posture shape (tools/security_audit.py --json):
      {root, checks, fail, warn, findings: [{check, level, msg, where}, ...]}
    a finding is ACTIONABLE when level == "FAIL" (a WARN is advisory, skipped).

  * the generic garden/control-pane verdict shape
    ({schema, ok, verdict, finding, reason, next_action}), optionally a LIST of such
    members under `members`/`findings`/`results`. A member is ACTIONABLE when it is
    NOT ok (ok is False, or a red verdict token: FAIL/RED/REGRESSED/BLOCKING/ERROR).
    An advisory pass merely SURFACING a condition (ok stays True) is the pass
    working — not a finding, skipped.

ROUTABILITY — the issue body NAMES the owning tool path (tools/security_audit.py,
the failing gate's source) in the `tools/...` / `.github/...` doc-link convention,
so the dispatch router's path-grep rung (issue_lane_router.route_issue) confirms a
lane (`tools` / `ci`) the same way score-signal's native-cmd tickets route to `cmd`.

DEDUP — different from idea_scout's permanent seen-cache: a CI finding is RECURRING
work (re-file after a close). Dedups against OPEN gate-signal-labelled issues only,
by a stable per-finding marker (`<!-- fak-gate-signal: <key> -->`): at most one OPEN
issue per finding key, re-fileable after a close. The label-scoped fetch bounds the
dedup index EXACTLY regardless of total backlog size. A hard CAP (--max-issues,
worst-first) keeps a multi-finding red from storming the tracker.

SAFE BY DEFAULT: dry-run. `--live` is the explicit opt-in that creates the issues.

    python tools/security_audit.py --json | python tools/gate_signal.py --from -
    python tools/gate_signal.py --from audit.json --json     # machine-readable plan
    python tools/gate_signal.py --from audit.json --max-issues 5 --live

Exit codes: 0 = ran clean (including "nothing to signal") · 2 = infra error (gh
missing / not authed / the envelope could not be read or parsed).
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

SCHEMA = "fak-gate-signal/1"
SIGNAL_LABEL = "gate-signal"
CI_LABEL = "ci"
MARKER_RE = re.compile(r"fak-gate-signal:\s*([A-Za-z0-9_:./\-]+)")

DEFAULTS = {
    "max_issues": 5,       # hard cap on issues filed per run (anti-storm)
    "issue_scan_limit": 800,  # open gate-signal issues fetched for the dedup index
}

# Red verdict tokens for the generic envelope: a member carrying any of these (or an
# explicit ok=False) is actionable. An advisory pass that SURFACES a condition keeps
# ok=True and is not a finding — so a surfaced-but-green member is correctly skipped.
RED_VERDICTS = {"FAIL", "RED", "REGRESSED", "BLOCKING", "ERROR", "DOWN", "BROKEN"}

# The owning tool path per known source slug, in the `tools/...` doc-link form the
# dispatch router (issue_lane_router.path_matches_lane) normalizes to the repo-
# relative `tools/**` tree — so the ticket path-confirms the `tools` lane. A finding
# that carries its own `where`/`source` path uses that instead; this is the fallback
# when the envelope names only a check slug.
PATH_BY_SOURCE: dict[str, str] = {
    "security-audit": "tools/security_audit.py",
    "security_audit": "tools/security_audit.py",
    "posture": "tools/security_audit.py",
    "garden": "tools/garden_bundle.py",
    "garden-bundle": "tools/garden_bundle.py",
    "fresh-status": "tools/fresh_status.py",
    "loop-audit": "tools/fleet_loop_audit.py",
}

# The doc/workflow that schedules each source, named in the body so a scheduled-gate
# finding also path-confirms the `ci` lane when the source path is a workflow.
WORKFLOW_BY_SOURCE: dict[str, str] = {
    "security-audit": ".github/workflows/security-audit.yml",
    "security_audit": ".github/workflows/security-audit.yml",
    "posture": ".github/workflows/security-audit.yml",
    "garden": ".github/workflows/garden.yml",
    "garden-bundle": ".github/workflows/garden.yml",
}


# ============================================================================
# Pure helpers (no I/O) — unit-tested directly.
# ============================================================================
def _slug(text: str) -> str:
    """A stable, marker-safe slug: lowercase, non-[a-z0-9._/-] -> '-', collapsed."""
    s = re.sub(r"[^a-z0-9._/-]+", "-", str(text).lower()).strip("-")
    return re.sub(r"-{2,}", "-", s) or "finding"


def _is_red(member: dict[str, Any]) -> bool:
    """True when a generic-envelope member is actionable: ok explicitly False, or a
    red verdict token. A member with ok=True (a pass surfacing a benign condition) is
    NOT red. Absent ok + absent verdict defaults to NOT red (no false finding)."""
    if member.get("ok") is False:
        return True
    v = str(member.get("verdict") or "").strip().upper()
    return v in RED_VERDICTS


def normalize_findings(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Both envelope shapes -> a uniform list of actionable findings, worst-first.

    Each finding: {key, label, source, detail, next_action, owning_path, workflow,
    severity}. `key` is a stable per-finding slug (`<source>:<check>`) that anchors
    the dedup marker. Returns [] when nothing is actionable (all WARN / all green) —
    the relevance filter lives here, so a green envelope yields no issues."""
    out: list[dict[str, Any]] = []

    # Shape A: the security-audit posture envelope (findings[] of check/level/msg).
    raw = payload.get("findings")
    looks_posture = (
        isinstance(raw, list)
        and ("checks" in payload or "fail" in payload or "warn" in payload)
        and all(isinstance(f, dict) and "level" in f for f in raw)
    )
    if looks_posture:
        source = _envelope_source(payload, default="security-audit")
        for f in raw:
            if str(f.get("level") or "").strip().upper() != "FAIL":
                continue  # WARN is advisory — the cadence/ratchet owns it, not this.
            check = str(f.get("check") or "finding")
            where = str(f.get("where") or "")
            out.append(_finding(
                source=source, check=check,
                detail=str(f.get("msg") or ""),
                next_action="", where=where, severity=3))
        out.sort(key=lambda r: (-r["severity"], r["key"]))
        return out

    # Shape B: the generic verdict envelope — a single member, or a list of members
    # under members/findings/results. Each red (not-ok) member is a finding.
    members = _generic_members(payload)
    source = _envelope_source(payload, default="gate")
    for m in members:
        if not isinstance(m, dict) or not _is_red(m):
            continue
        msrc = str(m.get("source") or m.get("name") or m.get("member") or source)
        check = str(m.get("finding") or m.get("check") or m.get("verdict") or "red")
        out.append(_finding(
            source=msrc, check=check,
            detail=str(m.get("reason") or m.get("finding") or ""),
            next_action=str(m.get("next_action") or ""),
            where=str(m.get("where") or m.get("path") or ""),
            severity=4 if str(m.get("verdict") or "").upper() == "BLOCKING" else 3))
    out.sort(key=lambda r: (-r["severity"], r["key"]))
    return out


def _envelope_source(payload: dict[str, Any], *, default: str) -> str:
    """The source slug naming which gate produced the envelope, from common keys."""
    for k in ("source", "name", "tool", "workflow", "schema"):
        v = payload.get(k)
        if v:
            s = str(v).split("/")[0].split(".")[0]
            if s:
                return _slug(s)
    return default


def _generic_members(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """The member list of a generic envelope: an explicit members/findings/results
    list, else the payload itself as a single member."""
    for k in ("members", "findings", "results", "panes"):
        v = payload.get(k)
        if isinstance(v, list) and v:
            return [m for m in v if isinstance(m, dict)]
    return [payload]


def _finding(*, source: str, check: str, detail: str, next_action: str,
             where: str, severity: int) -> dict[str, Any]:
    """Assemble one normalized finding + resolve its owning path for routability.
    A finding's own `where` path (if it names a real tree path) wins; else the
    PATH_BY_SOURCE map; else the generic fallback (no path -> degrades, not asserts)."""
    src_slug = _slug(source)
    key = f"{src_slug}:{_slug(check)}"
    owning = ""
    if where and _PATH_LIKE.search(where):
        w = where.replace("\\", "/")
        # Strip a leading "./" PREFIX only — never lstrip a char SET, which would
        # eat the leading "." of ".github/..." and mangle the path so the router's
        # path-grep (which requires the literal ".github") no longer matches it.
        owning = w[2:] if w.startswith("./") else w
    elif src_slug in PATH_BY_SOURCE:
        owning = PATH_BY_SOURCE[src_slug]
    workflow = WORKFLOW_BY_SOURCE.get(src_slug, "")
    label = f"{src_slug} · {check}"
    return {
        "key": key, "label": label, "source": src_slug, "check": check,
        "detail": detail, "next_action": next_action, "owning_path": owning,
        "workflow": workflow, "severity": int(severity),
    }


_PATH_LIKE = re.compile(r"(?:tools|cmd|internal|docs|\.github|\.claude)/[\w./-]+")


def marker(key: str) -> str:
    """The load-bearing dedup anchor: one stable HTML-comment per finding key."""
    return f"<!-- fak-gate-signal: {key} -->"


def open_issue_keys(issues: list[dict[str, Any]]) -> set[str]:
    """The set of finding keys already tracked by an OPEN gate-signal issue."""
    keys: set[str] = set()
    for iss in issues:
        for m in MARKER_RE.findall(iss.get("body") or ""):
            keys.add(m.strip())
    return keys


def render_issue(finding: dict[str, Any], today: str) -> dict[str, Any]:
    """Build the {title, body, labels} an issue is created from. The per-finding
    marker is the load-bearing dedup anchor; the body names the owning tool path
    (for lane routing) + the `#N`-stamp worker contract so the resolver has a closed
    loop."""
    key = finding["key"]
    label = finding["label"]
    source = finding["source"]
    detail = finding["detail"] or "(no detail provided in the envelope)"
    owning = finding["owning_path"]
    workflow = finding["workflow"]
    next_action = finding["next_action"]
    severity = finding["severity"]

    sev_word = "BLOCKING" if severity >= 4 else "FAIL"
    title = f"gate-signal: {label} is {sev_word}"

    # Route line: name the owning path(s) in the doc-link convention so the dispatch
    # router's path-grep rung confirms a lane. Without a resolvable path the ticket
    # still files (with the source named) but routes by label/scope, not path.
    route_lines = []
    if owning:
        route_lines.append(f"**Owning tool:** `{owning}` — route to its lane "
                           f"(the dispatch router path-confirms it).")
    if workflow:
        route_lines.append(f"**Scheduled by:** `{workflow}`.")
    if not route_lines:
        route_lines.append(f"**Source:** `{source}` (no owning path resolved; "
                           f"routes by label/scope).")

    do_lines = [
        f"1. Reproduce the finding from its source (`{owning or source}`).",
        "2. Fix the underlying gate failure — restore the red gate to green; change "
        "no behavior the finding does not cover.",
        "3. Re-run the owning gate to PROVE it is green again.",
        "4. Ship a commit citing this issue's `#N` in the subject + a `(fak <leaf>)` "
        "trailer, so the witness can bind and close it.",
    ]
    next_line = f"\n**Suggested next action (from the envelope):** {next_action}\n" \
        if next_action else ""

    body = (
        f"> Auto-filed by **gate-signal** (`tools/gate_signal.py`, {today}) from a "
        f"non-scorecard CI finding. A red gate — **needs a worker**; close as "
        f"`wontfix` if the finding is intentional/accepted.\n\n"
        f"**Finding:** `{label}` ({key})\n"
        f"**Detail:** {detail}\n"
        + "\n".join(route_lines) + "\n"
        + next_line
        + "\n**What to do**\n"
        + "\n".join(do_lines)
        + "\n\n---\n"
        "_gate-signal is the non-scorecard sibling of score-signal "
        "(`tools/score_signal.py`). After this issue is closed, a future recurrence "
        "of the same finding files a FRESH issue — only on an armed `--live` run._\n"
        + marker(key)
    )

    labels = [SIGNAL_LABEL, CI_LABEL]
    return {
        "title": title, "body": body, "labels": labels,
        "key": key, "severity": severity, "owning_path": owning,
    }


def plan_issues(findings: list[dict[str, Any]], open_keys: set[str], *,
                max_issues: int, today: str,
                ) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """relevance (already in normalize_findings) -> dedup vs OPEN issues -> CAP.
    Returns (issues, skip_stats). Deterministic: findings are worst-first, deduped by
    key, then capped."""
    stats = {"already-open": 0, "within-run-dup": 0, "over-cap": 0}
    out: list[dict[str, Any]] = []
    seen_run: set[str] = set()
    for f in findings:
        key = f["key"]
        if key in seen_run:
            stats["within-run-dup"] += 1
            continue
        seen_run.add(key)
        if key in open_keys:
            stats["already-open"] += 1
            continue
        out.append(render_issue(f, today))
    if len(out) > max_issues:
        stats["over-cap"] = len(out) - max_issues
        out = out[:max_issues]
    return out, stats


# ============================================================================
# I/O boundary — gh + the envelope. Thin wrappers so the logic above stays testable.
# ============================================================================
def gh_json(args: list[str], timeout: int = 60) -> Any:
    """Run a `gh` subcommand that emits JSON; return the parsed value."""
    proc = subprocess.run(["gh", *args], capture_output=True, text=True,
                          encoding="utf-8", timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} -> {proc.returncode}: "
                           f"{proc.stderr.strip()[:300]}")
    out = proc.stdout.strip()
    return json.loads(out) if out else []


def fetch_open_issues(limit: int) -> list[dict[str, Any]]:
    """Open issues carrying the gate-signal label — the dedup index. Scoping to the
    tool's OWN label bounds the index EXACTLY (cf. score_signal.fetch_open_issues)."""
    return gh_json([
        "issue", "list", "--state", "open", "--label", SIGNAL_LABEL,
        "--limit", str(limit), "--json", "number,title,body",
    ])


def ensure_labels() -> None:
    """Idempotently create the marker labels so `gh issue create` never fails."""
    wanted = [
        (SIGNAL_LABEL, "b60205",
         "Auto-filed by gate-signal (tools/gate_signal.py) from a non-scorecard CI "
         "finding; needs a worker"),
        (CI_LABEL, "ededed", "CI / automation surface"),
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


def load_envelope(from_arg: str) -> dict[str, Any]:
    """Read the CI-finding envelope from --from (a file or '-' for stdin). Raises
    RuntimeError on any failure so the caller exits 2 cleanly."""
    if not from_arg:
        raise RuntimeError("--from is required (a CI-finding envelope file or '-' "
                           "for stdin); there is no live fold to default to")
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
        description="gate-signal: a non-scorecard CI finding -> a deduped GitHub issue.")
    ap.add_argument("--from", dest="from_arg", default="",
                    help="read the CI-finding envelope from this file ('-' = stdin). "
                         "Accepts the security_audit --json shape or a generic "
                         "{schema,ok,verdict,finding,reason,next_action} envelope.")
    ap.add_argument("--max-issues", type=int,
                    help=f"hard cap on issues filed, worst-first "
                         f"(default {DEFAULTS['max_issues']}).")
    ap.add_argument("--live", action="store_true",
                    help="actually create the issues (default: dry-run).")
    ap.add_argument("--json", action="store_true", help="machine-readable output.")
    args = ap.parse_args(argv)

    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    max_issues = args.max_issues if args.max_issues is not None else DEFAULTS["max_issues"]

    try:
        payload = load_envelope(args.from_arg)
    except (OSError, RuntimeError) as e:
        print(f"refuse: could not load the CI-finding envelope: {e}", file=sys.stderr)
        return 2

    findings = normalize_findings(payload)

    # The OPEN-issue dedup index is mandatory: with no way to know what is already
    # tracked, --live could re-file an open finding every tick. Refuse rather than
    # risk a storm (same contract as score-signal / idea_scout).
    errors: list[str] = []
    try:
        issues = fetch_open_issues(DEFAULTS["issue_scan_limit"])
        open_keys = open_issue_keys(issues)
    except Exception as e:  # noqa: BLE001
        print(f"refuse: cannot fetch open issues for the dedup index ({e})",
              file=sys.stderr)
        return 2

    to_file, skip_stats = plan_issues(
        findings, open_keys, max_issues=max_issues, today=today)

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

    result = {
        "schema": SCHEMA,
        "date": today,
        "mode": "live" if args.live else "dry-run",
        "findings_total": len(findings),
        "open_signal_issues": sorted(open_keys),
        "skipped": skip_stats,
        "planned": [
            {"title": i["title"], "key": i["key"], "severity": i["severity"],
             "owning_path": i["owning_path"], "labels": i["labels"]}
            for i in to_file],
        "filed": [{"title": f["title"], "key": f["key"],
                   "issue_url": f.get("issue_url", "")} for f in filed],
        "errors": errors,
    }

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    # Human report.
    print(f"gate-signal {today} — {result['mode']}  "
          f"({len(findings)} actionable finding(s) in the envelope)")
    sk = ", ".join(f"{k}={v}" for k, v in skip_stats.items() if v) or "none"
    print(f"  deduped/dropped: {sk}")
    if result["open_signal_issues"]:
        print(f"  already tracked (open): {', '.join(result['open_signal_issues'])}")
    if not to_file:
        print("  → no NEW actionable CI finding. Nothing to file.")
    else:
        verb = "FILED" if args.live else "would file"
        print(f"  → {verb} {len(to_file)} issue(s) (cap {max_issues}):")
        for i in to_file:
            f = next((x for x in filed if x["key"] == i["key"]), None)
            mark = ""
            if args.live:
                mark = f"  {f['issue_url']}" if f else "  (create failed)"
            print(f"     [{i['key']}] {i['title']}{mark}")
    if errors:
        print("  errors:")
        for e in errors:
            print(f"     ! {e}")
    if not args.live and to_file:
        print("\n  dry-run — file these for real with:  "
              "python tools/gate_signal.py --from <envelope> --live")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
