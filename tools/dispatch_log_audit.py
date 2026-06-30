#!/usr/bin/env python3
r"""dispatch-log-audit — scan ``.dispatch-runs/`` worker logs, classify failure
signatures, dedup against open issues, and file capped issue candidates. The
*failure → ticket* FEEDER, the complement to the *dead-backend / no-op* status
card #1276 made self-auditing inside ``docs/dispatch-status.md`` (that card SHOWS
a problem; it does not TRACK it). See #1300.

A manual sweep of ``.dispatch-runs/*.log`` on 2026-06-29 surfaced three real
defects in minutes (#1275, #1276, #1277). That sweep — *scan recent worker logs →
classify failure signatures → dedup against open issues → file a ticket with
evidence* — is mechanical, and this tool runs it the same way ``tools/idea_scout.py``
turns arXiv/GitHub into deduped issue candidates on a daily cron.

PURE-LOCAL: no model, no network beyond ``gh`` for the dedup index + the optional
file step. The hard part of an UNATTENDED issue filer is not detecting — it is NOT
spamming — so the same three-rung dedup discipline as ``idea_scout`` gates every
candidate before it can become an issue:

  1. open-issue   the candidate's signature-key (detector + backend + normalized
                  message) stamped in / matched by any existing OPEN issue.
  2. seen-ledger  ``.dispatch-runs/log-audit-seen.json`` — a persistent
                  {signature_key: record} of every signature ever FILED.
  3. per-run CAP  at most --max-issues per run (default 3), highest-severity first,
                  so even a pathological day cannot storm the tracker.

Initial detector pack (each tells the worker exactly what to look at):

  * banner-only no-op  a ``resolve-*.log`` at/under the real-turn byte floor over a
                       dead pid — reuses ``dispatch_status.silent_workers`` (#1276).
  * hook-failure storm ≥N ``hook: <name> Failed`` lines in one session (#1277).
  * panic / traceback  a ``panic:`` or ``Traceback (most recent call last):`` at the
                       start of a worker-log line (quoted grep output is skipped).
  * OFF_TRUNK storm    repeated OFF_TRUNK guard refusals across sessions.
  * auth wall          a ``Not logged in`` / credit-balance / usage-limit banner streak.

SAFE BY DEFAULT: dry-run. The tool prints exactly the issues it WOULD file and
mutates nothing (not even the seen-ledger). ``--enact`` (alias ``--live``) is the
explicit opt-in that creates issues via ``gh issue create`` and records them.

    python tools/dispatch_log_audit.py                 # dry-run: plan, file nothing
    python tools/dispatch_log_audit.py --json          # machine-readable plan
    python tools/dispatch_log_audit.py --max-issues 3 --enact   # file at most 3

Exit codes: 0 = ran clean · 2 = infra error (gh missing / not authed with no
seen-ledger to fall back on / not a runs dir).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

_CREATE_NO_WINDOW = 0x08000000


def _win_creationflags() -> int:
    return _CREATE_NO_WINDOW if os.name == "nt" else 0


SCHEMA = "fleet-dispatch-log-audit/1"
RUNS_DIRNAME = ".dispatch-runs"
LEDGER_FILENAME = "log-audit-seen.json"
# The candidates are filed under the existing `dispatch` label (the issue this tool
# implements carries it); a non-existent label would make `gh issue create` fail.
AUDIT_LABEL = "dispatch"
TRIAGE_LABELS = ["needs-triage", "triage-only"]
# Stamped in each filed issue body as the load-bearing dedup anchor (rung 1).
_SIG_STAMP_RE = re.compile(r"dispatch-log-audit-sig:\s*([^\s>]+)")

DEFAULTS: dict[str, Any] = {
    "max_issues": 3,        # hard cap on issues filed per run (anti-storm)
    "hook_min": 3,          # ≥N `hook: … Failed` lines in one session → a storm
    "lookback_hours": 0,    # 0 = scan every resolve log; >0 filters by mtime
    "issue_scan_limit": 400,  # open issues fetched for the dedup index
    "max_read_bytes": 2_000_000,  # per-log read bound (worker logs can be MB)
}

# How many trailing chars of a per-detector message survive into the signature key.
_MSG_KEEP = 160


# ============================================================================
# Pure helpers (no I/O) — unit-tested directly.
# ============================================================================
_NUM_RE = re.compile(r"0x[0-9a-fA-F]+|\d+")
# A ripgrep-style `path:line:` prefix — worker logs echo grep output that quotes
# repo files mentioning `panic:` / `OFF_TRUNK`; those are NOT real failures here.
_QUOTE_RE = re.compile(r"^[.\\/]?[\w./\\-]+:\d+:")


def normalize_message(s: str) -> str:
    """Collapse a raw message to a stable signature token: lowercase, every run of
    digits / hex address → ``#``, whitespace squeezed. So ``index out of range
    [7]`` and ``[12]`` map to one signature, but two different panics stay distinct."""
    s = _NUM_RE.sub("#", (s or "").lower().strip())
    return re.sub(r"\s+", " ", s)[:_MSG_KEEP]


def signature_key(detector: str, backend: str, message: str) -> str:
    """The dedup identity of a finding: detector + backend + normalized message."""
    return f"{detector}::{backend}::{normalize_message(message)}"


def _looks_like_quote(line: str) -> bool:
    """A `file:line:` ripgrep echo (the worker reading the repo), not a real emit."""
    return bool(_QUOTE_RE.match(line))


def backend_of_log(log: Path) -> str:
    """The backend that produced a worker log, from its ``.backend`` sidecar; legacy
    logs without one are the original ``claude`` backend. Mirrors
    ``issue_resolve_dispatch._backend_of_log`` (kept dependency-light: a 3-line read
    rather than importing the whole dispatcher)."""
    try:
        return log.with_suffix(".backend").read_text(encoding="utf-8").strip() or "claude"
    except OSError:
        return "claude"


# ---- Detector matchers: pure (text) -> list[{message, count, sample}] ---------
_HOOK_RE = re.compile(r"^\s*hook:\s*(\S+)\s+Failed\b")
_AUTH_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("not logged in", re.compile(r"not logged in", re.IGNORECASE)),
    ("credit balance too low", re.compile(r"credit balance", re.IGNORECASE)),
    ("usage limit reached", re.compile(r"usage limit", re.IGNORECASE)),
    ("login required", re.compile(r"please run\s*/login|run `?/login`?", re.IGNORECASE)),
]
_OFF_TRUNK_REFUSE = re.compile(r"refus|blocked|denied|reason|guard", re.IGNORECASE)


def _match_hook_failures(text: str) -> list[dict[str, Any]]:
    hits = [ln.strip() for ln in text.splitlines() if _HOOK_RE.match(ln)]
    if not hits:
        return []
    return [{"message": "hook handler failures", "count": len(hits), "sample": hits[:5]}]


def _match_panic(text: str) -> list[dict[str, Any]]:
    lines = text.splitlines()
    out: list[dict[str, Any]] = []
    for i, raw in enumerate(lines):
        line = raw.strip()
        if line.startswith("panic:"):
            out.append({"message": line, "count": 1, "sample": _stack_sample(lines, i)})
        elif line.startswith("Traceback (most recent call last):"):
            msg = "python traceback" + (f": {_py_exc(lines, i)}" if _py_exc(lines, i) else "")
            out.append({"message": msg, "count": 1, "sample": _stack_sample(lines, i)})
    return _collapse(out)


def _stack_sample(lines: list[str], i: int, span: int = 4) -> list[str]:
    return [ln.rstrip() for ln in lines[i:i + span] if ln.strip()]


def _py_exc(lines: list[str], i: int) -> str:
    """The exception headline a Python traceback ends on (``KeyError: 'x'``), scanned
    forward from the ``Traceback`` line so identical exceptions share a signature."""
    rx = re.compile(r"^\s*([A-Za-z_][\w.]*(?:Error|Exception|Warning|Exit)\b.*)$")
    for ln in lines[i + 1:i + 40]:
        m = rx.match(ln)
        if m and not ln.lstrip().startswith("File "):
            return m.group(1).strip()[:120]
    return ""


def _collapse(found: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge findings that normalize to the same message within one log: sum counts,
    keep the first sample. Keeps a log with 12 identical panics to one finding."""
    by: dict[str, dict[str, Any]] = {}
    for f in found:
        k = normalize_message(f["message"])
        cur = by.get(k)
        if cur is None:
            by[k] = dict(f)
        else:
            cur["count"] += f["count"]
    return list(by.values())


def _match_off_trunk(text: str) -> list[dict[str, Any]]:
    hits: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if "OFF_TRUNK" not in line.upper():
            continue
        if _looks_like_quote(line) or not _OFF_TRUNK_REFUSE.search(line):
            continue  # a quoted repo line / a bare mention is not a guard refusal
        hits.append(line[:200])
    if not hits:
        return []
    return [{"message": "OFF_TRUNK guard refusal", "count": len(hits), "sample": hits[:5]}]


def _match_auth_wall(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for label, rx in _AUTH_PATTERNS:
        hits = [ln.strip() for ln in text.splitlines()
                if rx.search(ln) and not _looks_like_quote(ln.strip())]
        if hits:
            out.append({"message": f"auth wall: {label}", "count": len(hits),
                        "sample": hits[:3]})
    return out


# detector metadata: severity orders the per-run cap; per_log_min is the
# "storm in one session" floor; min_total is the aggregated-across-logs floor.
TEXT_DETECTORS: list[dict[str, Any]] = [
    {"key": "panic-traceback", "title": "worker panic / traceback", "severity": 100,
     "per_log_min": 1, "min_total": 1, "match": _match_panic},
    {"key": "hook-failure-storm", "title": "hook-handler failure storm", "severity": 80,
     "per_log_min": DEFAULTS["hook_min"], "min_total": DEFAULTS["hook_min"],
     "match": _match_hook_failures},
    {"key": "off-trunk-storm", "title": "OFF_TRUNK guard-refusal storm", "severity": 60,
     "per_log_min": 1, "min_total": 2, "match": _match_off_trunk},
    {"key": "auth-wall", "title": "auth / credit wall streak", "severity": 50,
     "per_log_min": 1, "min_total": 3, "match": _match_auth_wall},
]
_DETECTOR_BY_KEY = {d["key"]: d for d in TEXT_DETECTORS}
BANNER_DETECTOR = {"key": "banner-only-noop", "title": "banner-only / empty no-op worker",
                   "area": "dispatch", "severity": 40, "min_total": 2}


def scan_text(name: str, text: str, backend: str,
              detectors: list[dict[str, Any]] | None = None,
              *, hook_min: int = DEFAULTS["hook_min"]) -> list[dict[str, Any]]:
    """Run every text detector over one worker log → flat findings (one per
    detector-hit-group). The hook storm floor is overridable so a noisier fleet can
    raise it without editing the table."""
    detectors = detectors if detectors is not None else TEXT_DETECTORS
    out: list[dict[str, Any]] = []
    for det in detectors:
        per_log_min = hook_min if det["key"] == "hook-failure-storm" else det["per_log_min"]
        min_total = hook_min if det["key"] == "hook-failure-storm" else det["min_total"]
        for f in det["match"](text):
            if f["count"] < per_log_min:
                continue
            out.append({"detector": det["key"], "severity": det["severity"],
                        "min_total": min_total, "backend": backend, "log": name,
                        "message": f["message"], "count": f["count"],
                        "sample": f["sample"]})
    return out


def aggregate_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse per-log findings into one candidate per signature-key: sum counts,
    union the source logs and a bounded sample. Deterministic on insertion order
    (callers feed sorted logs)."""
    by_key: dict[str, dict[str, Any]] = {}
    for f in findings:
        key = signature_key(f["detector"], f["backend"], f["message"])
        c = by_key.get(key)
        if c is None:
            c = {"signature_key": key, "detector": f["detector"],
                 "severity": f["severity"], "min_total": f["min_total"],
                 "backend": f["backend"], "message": f["message"],
                 "count": 0, "logs": [], "sample": []}
            by_key[key] = c
        c["count"] += f["count"]
        if f["log"] not in c["logs"]:
            c["logs"].append(f["log"])
        for s in f["sample"]:
            if s not in c["sample"] and len(c["sample"]) < 6:
                c["sample"].append(s)
    return list(by_key.values())


# ---- Dedup ------------------------------------------------------------------
def existing_issue_index(issues: list[dict[str, Any]]) -> tuple[set[str], str]:
    """From the OPEN issues: every signature-key stamped in a body, plus the joined
    lowercased title+body blob for a substring fallback (survives a lost stamp)."""
    stamped: set[str] = set()
    blob: list[str] = []
    for iss in issues:
        body = iss.get("body") or ""
        blob.append(((iss.get("title") or "") + "\n" + body).lower())
        for m in _SIG_STAMP_RE.findall(body):
            stamped.add(m.strip())
    return stamped, "\n".join(blob)


def is_duplicate(cand: dict[str, Any], seen: dict[str, Any], stamped: set[str],
                 bodies_joined: str) -> str | None:
    """The dedup rung that fires ('seen-ledger' / 'open-issue'), or None if novel."""
    key = cand["signature_key"]
    if key in seen:
        return "seen-ledger"
    if key in stamped or key.lower() in bodies_joined:
        return "open-issue"
    return None


def render_issue(cand: dict[str, Any], today: str) -> dict[str, Any]:
    """Build the {title, body, labels, signature_key} an issue is created from. The
    signature-key stamp in an HTML comment is the load-bearing dedup anchor (rung 1)."""
    det = _DETECTOR_BY_KEY.get(cand["detector"], BANNER_DETECTOR)
    title = (f"dispatch-log-audit: {det['title']} on {cand['backend']} "
             f"({cand['count']}×)")
    if len(title) > 120:
        title = title[:117].rstrip() + "…"

    logs = cand["logs"][:6]
    sample = cand["sample"][:6]
    ev_logs = "\n".join(f"- `.dispatch-runs/{lg}`" for lg in logs) or "- (none)"
    body = (
        f"> Auto-filed by **dispatch-log-audit** (`tools/dispatch_log_audit.py`, "
        f"{today}). A novel worker-log failure signature; **needs triage** — close "
        f"as `wontfix`/`duplicate` if it is not real or already tracked.\n\n"
        f"**Detector:** `{cand['detector']}`  ·  **Backend:** `{cand['backend']}`  ·  "
        f"**Occurrences:** {cand['count']} across {len(cand['logs'])} session(s)\n\n"
        f"**Where to look** ({len(cand['logs'])} log(s)):\n{ev_logs}\n\n"
        f"**Sample evidence:**\n```\n" + "\n".join(sample[:6]) + "\n```\n\n"
        f"**Dispatchability:** `triage_only`\n\n"
        f"This is a diagnostic finding. A human must scope the lane, witness, and "
        f"acceptance gate before removing the triage labels.\n\n"
        f"---\n"
        f"_Triage hint: is this a real worker-side defect to fix, a flaky "
        f"environment to harden, or noise to suppress? If noise, close it._\n"
        f"<!-- dispatch-log-audit-sig: {cand['signature_key']} -->"
    )
    return {"title": title, "body": body, "labels": [AUDIT_LABEL, *TRIAGE_LABELS],
            "signature_key": cand["signature_key"], "detector": cand["detector"],
            "backend": cand["backend"], "count": cand["count"],
            "severity": cand["severity"], "logs": logs}


def plan_issues(candidates: list[dict[str, Any]], seen: dict[str, Any],
                stamped: set[str], bodies_joined: str, cfg: dict[str, Any],
                today: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """min-total → dedup → sort by (severity, count) → CAP. Returns (issues, stats).
    Deterministic: ties break on the signature-key so the plan is stable."""
    stats = {"seen-ledger": 0, "open-issue": 0, "below-min": 0}
    kept: list[dict[str, Any]] = []
    for c in candidates:
        if c["count"] < c["min_total"]:
            stats["below-min"] += 1
            continue
        rung = is_duplicate(c, seen, stamped, bodies_joined)
        if rung:
            stats[rung] += 1
            continue
        kept.append(c)
    kept.sort(key=lambda c: (-c["severity"], -c["count"], c["signature_key"]))
    return [render_issue(c, today) for c in kept[: cfg["max_issues"]]], stats


# ============================================================================
# I/O boundary — files + gh. Thin wrappers so the logic above stays testable.
# ============================================================================
def _read_text(log: Path, max_bytes: int = DEFAULTS["max_read_bytes"]) -> str | None:
    try:
        data = log.read_bytes()[:max_bytes]
    except OSError:
        return None
    return data.decode("utf-8", "replace")


def banner_findings(runs_dir: Path) -> list[dict[str, Any]]:
    """The banner-only / empty no-op workers, reusing ``dispatch_status.silent_workers``
    (#1276) so this detector shares the dispatcher's real-turn byte floor + dead-pid
    guard. Best-effort: psutil is optional, so its absence yields no false silents."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import dispatch_status  # type: ignore
        rows = dispatch_status.silent_workers(runs_dir)
    except Exception:  # noqa: BLE001 — a missing/blown classifier just means no banners
        return []
    out: list[dict[str, Any]] = []
    for r in rows:
        backend = backend_of_log(runs_dir / str(r.get("log", "")))
        out.append({
            "detector": BANNER_DETECTOR["key"], "severity": BANNER_DETECTOR["severity"],
            "min_total": BANNER_DETECTOR["min_total"], "backend": backend,
            "log": r.get("log", ""), "count": 1,
            "message": "banner-only / empty resolve log over a dead pid",
            "sample": [f"{r.get('log')} ({r.get('size')} B, kind={r.get('kind')}, "
                       f"pid {r.get('pid')} dead)"],
        })
    return out


def scan_runs(runs_dir: Path, cfg: dict[str, Any], *, now_ts: float | None = None,
              reader: Callable[[Path], str | None] | None = None) -> list[dict[str, Any]]:
    """Scan every ``resolve-*.log`` (newest filtered by --lookback-hours, if set)
    plus the banner-only no-op detector. A failing read is skipped, never fatal."""
    if not runs_dir.is_dir():
        return []
    reader = reader or _read_text
    lookback = cfg.get("lookback_hours") or 0
    horizon = (now_ts - lookback * 3600) if (lookback and now_ts is not None) else None
    findings: list[dict[str, Any]] = []
    for log in sorted(runs_dir.glob("resolve-*.log")):
        if horizon is not None:
            try:
                if log.stat().st_mtime < horizon:
                    continue
            except OSError:
                continue
        text = reader(log)
        if text is None:
            continue
        findings += scan_text(log.name, text, backend_of_log(log),
                              hook_min=cfg.get("hook_min", DEFAULTS["hook_min"]))
    findings += banner_findings(runs_dir)
    return findings


def gh_json(args: list[str], timeout: int = 60) -> Any:
    proc = subprocess.run(["gh", *args], capture_output=True, text=True,
                          encoding="utf-8", timeout=timeout,
                          creationflags=_win_creationflags())
    if proc.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} -> {proc.returncode}: "
                           f"{proc.stderr.strip()[:300]}")
    out = proc.stdout.strip()
    return json.loads(out) if out else []


def fetch_open_issues(limit: int) -> list[dict[str, Any]]:
    return gh_json(["issue", "list", "--state", "open", "--limit", str(limit),
                    "--json", "number,title,body"])


def ensure_label() -> None:
    """Ensure issue labels exist so `gh issue create` never fails on a missing
    label. ``dispatch`` is a SHARED, curated repo label (it predates this tool),
    and the triage labels may also be curated. This NEVER ``--force``-overwrites
    an existing label's color/description: it only creates missing labels."""
    ensure_one_label(AUDIT_LABEL, "b60205", "Worker-dispatch defect / log-audit finding")
    ensure_one_label("needs-triage", "d4c5f9",
                     "Needs human scoping before an agent dispatch can take it")
    ensure_one_label("triage-only", "d4c5f9",
                     "Useful issue, but not a worker-ready dispatch leaf")


def ensure_one_label(name: str, color: str, desc: str) -> None:
    try:
        existing = gh_json(["label", "list", "--search", name, "--json", "name"])
        if any(str(lab.get("name", "")).lower() == name for lab in existing):
            return  # already present -> leave its curated color/description untouched
    except Exception:  # noqa: BLE001 — fall through to a best-effort create
        pass
    try:
        proc = subprocess.run(
            ["gh", "label", "create", name, "--color", color, "--description", desc],
            capture_output=True, text=True, encoding="utf-8", timeout=30,
            creationflags=_win_creationflags())
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"warning: could not run `gh label create {name}`: {e}",
              file=sys.stderr)
        return
    if proc.returncode != 0:
        print(f"warning: could not ensure '{name}' label "
              f"(issue creation may fail): {proc.stderr.strip()[:200]}", file=sys.stderr)


def create_issue(issue: dict[str, Any]) -> str:
    args = ["issue", "create", "--title", issue["title"], "--body", issue["body"]]
    for lab in issue["labels"]:
        args += ["--label", lab]
    proc = subprocess.run(["gh", *args], capture_output=True, text=True, encoding="utf-8",
                          creationflags=_win_creationflags())
    if proc.returncode != 0:
        raise RuntimeError(f"gh issue create -> {proc.returncode}: "
                           f"{proc.stderr.strip()[:300]}")
    return proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""


# ---- Ledger I/O -------------------------------------------------------------
def ledger_path(runs_dir: Path) -> Path:
    return runs_dir / LEDGER_FILENAME


def load_seen(runs_dir: Path) -> dict[str, Any]:
    p = ledger_path(runs_dir)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data.get("seen", data) if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_seen(runs_dir: Path, seen: dict[str, Any]) -> None:
    p = ledger_path(runs_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"schema": SCHEMA, "seen": seen}, indent=2,
                            ensure_ascii=False), encoding="utf-8")


# ============================================================================
# Driver.
# ============================================================================
def main(argv: list[str] | None = None) -> int:
    import time

    ap = argparse.ArgumentParser(
        description="dispatch-log-audit: scan .dispatch-runs/ → deduped, capped "
                    "GitHub issue candidates for novel worker-log failure modes.")
    ap.add_argument("--workspace", default=".",
                    help="repo root (holds .dispatch-runs/). Default: cwd.")
    ap.add_argument("--runs-dir", default=None,
                    help="override the runs dir (default: <workspace>/.dispatch-runs).")
    ap.add_argument("--max-issues", type=int,
                    help=f"hard cap on issues filed (default {DEFAULTS['max_issues']}).")
    ap.add_argument("--hook-min", type=int,
                    help=f"`hook: … Failed` lines/session to call a storm "
                         f"(default {DEFAULTS['hook_min']}).")
    ap.add_argument("--lookback-hours", type=int,
                    help="only scan logs modified within this many hours (0 = all).")
    ap.add_argument("--enact", "--live", dest="enact", action="store_true",
                    help="actually create issues + record them (default: dry-run).")
    ap.add_argument("--json", action="store_true", help="machine-readable output.")
    args = ap.parse_args(argv)

    workspace = Path(args.workspace).resolve()
    runs_dir = Path(args.runs_dir).resolve() if args.runs_dir else workspace / RUNS_DIRNAME
    today = time.strftime("%Y-%m-%d")

    cfg = dict(DEFAULTS)
    for k, v in (("max_issues", args.max_issues), ("hook_min", args.hook_min),
                 ("lookback_hours", args.lookback_hours)):
        if v is not None:
            cfg[k] = v

    if not runs_dir.is_dir():
        print(f"refuse: no runs dir at {runs_dir}", file=sys.stderr)
        return 2

    findings = scan_runs(runs_dir, cfg, now_ts=time.time())
    candidates = aggregate_findings(findings)

    # The open-issue dedup index is mandatory: with no way to know what is already
    # filed, --enact could re-storm the tracker. If gh fails AND there is no
    # seen-ledger, refuse rather than risk a spam run (idea_scout's contract).
    seen = load_seen(runs_dir)
    errors: list[str] = []
    try:
        issues = fetch_open_issues(cfg["issue_scan_limit"])
        stamped, bodies_joined = existing_issue_index(issues)
    except Exception as e:  # noqa: BLE001
        errors.append(f"issues: {e}")
        if not seen:
            print(f"refuse: cannot fetch open issues and no seen-ledger to fall back "
                  f"on ({e})", file=sys.stderr)
            return 2
        stamped, bodies_joined = set(), ""

    to_file, skip_stats = plan_issues(candidates, seen, stamped, bodies_joined, cfg, today)

    filed: list[dict[str, Any]] = []
    if args.enact and to_file:
        ensure_label()
        for issue in to_file:
            try:
                url = create_issue(issue)
            except Exception as e:  # noqa: BLE001
                errors.append(f"create[{issue['signature_key']}]: {e}")
                continue
            seen[issue["signature_key"]] = {
                "filed_at": today, "issue_url": url, "detector": issue["detector"],
                "backend": issue["backend"], "count": issue["count"]}
            filed.append({**issue, "issue_url": url})
        if filed:
            save_seen(runs_dir, seen)

    result = {
        "schema": SCHEMA, "date": today, "mode": "enact" if args.enact else "dry-run",
        "candidates_found": len(candidates),
        "skipped": skip_stats,
        "planned": [
            {"title": i["title"], "signature_key": i["signature_key"],
             "detector": i["detector"], "backend": i["backend"], "count": i["count"],
             "logs": i["logs"]}
            for i in to_file],
        "filed": [{"title": f["title"], "issue_url": f.get("issue_url", "")}
                  for f in filed],
        "errors": errors,
    }

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    # Human report.
    print(f"dispatch-log-audit {today} — {result['mode']}")
    print(f"  scanned {runs_dir} → {len(candidates)} candidate signature(s)")
    sk = ", ".join(f"{k}={v}" for k, v in skip_stats.items() if v) or "none"
    print(f"  deduped/dropped: {sk}")
    if not to_file:
        print("  → nothing new worth filing.")
    else:
        verb = "FILED" if args.enact else "would file"
        print(f"  → {verb} {len(to_file)} issue(s) (cap {cfg['max_issues']}):")
        for i in to_file:
            mark = ""
            if args.enact:
                f = next((x for x in filed if x["signature_key"] == i["signature_key"]), None)
                mark = f"  {f['issue_url']}" if f else "  (create failed)"
            print(f"     [{i['severity']:>3}] {i['title']}{mark}")
            print(f"           logs={', '.join(i['logs'][:4])}")
    if errors:
        print("  errors:")
        for e in errors:
            print(f"     ! {e}")
    if not args.enact and to_file:
        print("\n  dry-run — file these for real with:  "
              "python tools/dispatch_log_audit.py --enact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
