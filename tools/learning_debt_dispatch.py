#!/usr/bin/env python3
r"""Staleness -> backlog — turn the learning scorecard's HARD defects into
cap-bounded GitHub triage issues the dispatch fleet can fix. The backlog-FEEDER
arm of the docs-freshness loop (epic #1278, child #1283).

`tools/learning_scorecard.py` is a deterministic THERMOMETER: it re-derives
learning-debt from disk (orphan lessons, dead links, stale install pins, missing
courses, a stale stamp) and prints a number nobody can hand-edit. But a number
read by hand is exactly what let the teaching docs drift 34 defects deep before a
person noticed. This tool closes that gap on the intake side: it reads the
scorecard's machine-readable JSON and files ONE triage issue per HARD defect, so
each rot becomes tracked, fixable work instead of a line in a report.

It reuses the `tools/idea_scout.py` intake-loop contract verbatim — the hard part
of an unattended issue filer is not finding work, it is NOT spamming. Three dedup
rungs gate every defect before it can become an issue:

  1. seen-cache   .learning-debt-dispatch/seen.json — a persistent
                  {source_id: record} of every defect ever FILED. A defect filed
                  once is never filed again until it is fixed AND re-detected with
                  a new fingerprint. The durable rung; gitignored, node-local.
  2. issue-body   the defect's source_id stamped in an existing issue body
                  (an HTML-comment marker) ⇒ already filed (survives a lost cache).
  3. title-near   token-overlap (Jaccard) with an existing issue title ⇒ a
                  near-dup a human already opened by hand.

And a hard CAP: at most --cap issues per run (default 5), highest-priority first,
so even a corpus that rots 30 defects deep in one night cannot storm the tracker.
Priority is a TRANSPARENT integer (defect-class weight + how broken the doc is),
surfaced on every planned issue so the ranking is auditable, never a black box.

Each filed issue cites the EXACT doc + defect class + the scorecard's verbatim
detail string (no prose drift): the issue is a faithful pointer back to the JSON
the scorecard emitted, not a re-description of it.

SAFE BY DEFAULT: dry-run. The tool prints exactly the issues it WOULD file and
mutates nothing (not even the seen-cache). `--live` is the explicit opt-in that
actually creates issues via `gh issue create` and records them — the same
dry-run-first contract as the dispatch tools.

    python tools/learning_debt_dispatch.py                 # dry-run: plan, file nothing
    python tools/learning_debt_dispatch.py --json          # machine-readable plan
    python tools/learning_debt_dispatch.py --cap 5 --live  # file at most 5, record them
    python tools/learning_debt_dispatch.py --scorecard report.json  # use a saved JSON

Exit codes: 0 = ran clean · 2 = infra error (scorecard failed / gh missing / not
authed / not a repo / cannot fetch existing issues with no cache to fall back on).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
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

SCHEMA = "fleet-learning-debt-dispatch/1"
CACHE_DIRNAME = ".learning-debt-dispatch"
CACHE_FILENAME = "seen.json"
DEBT_LABEL = "learning-debt"
# The load-bearing dedup anchor (rung 2): a defect's source_id stamped in an
# HTML comment in the issue body. Matches `tools/idea_scout.py`'s pattern so the
# dispatch fleet's tooling treats both feeders the same way.
SOURCE_MARKER = "learning-debt-source"
# The doc the stale-stamp defect is attributed to when the scorecard does not
# name one — the committed scorecard snapshot is what goes stale.
STAMP_DOC = "docs/LEARNING-SCORECARD.md"
SCORECARD_REL = "tools/learning_scorecard.py"

DEFAULTS = {
    "cap": 5,                # hard cap on issues filed per run (anti-storm)
    "dup_jaccard": 0.6,      # title token-overlap to call a near-duplicate
    "issue_scan_limit": 800,  # existing issues fetched for the dedup index
}

# Transparent per-class priority weight. Structural rot (an orphaned lesson, a
# missing course, a stale stamp) outranks an in-doc pedagogy gap, and the most
# mechanical hygiene defects (a dead link, a stale pin) outrank softer ones. The
# weight dominates ordering; how broken the doc is (100 - score) breaks ties
# within a class so the most-rotted doc of a class is filed first.
CLASS_WEIGHT = {
    "orphan-lesson": 50,
    "missing-course": 45,
    "stale-stamp": 40,
    "freshness": 35,    # in-doc: stale install pin / unresolved placeholder
    "links": 35,        # in-doc: dead link
    "honesty": 30,
    "runnable": 25,
    "worked": 20,
    "orientation": 15,
    "structure": 10,
}
DEFAULT_WEIGHT = 20


# ============================================================================
# Pure helpers (no I/O) — unit-tested directly.
# ============================================================================
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> set[str]:
    """Lowercase alnum tokens of length >= 3 (drops 'the', 'a', punctuation)."""
    return {t for t in _TOKEN_RE.findall((text or "").lower()) if len(t) >= 3}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _fingerprint(detail: str) -> str:
    """A short stable hash of the scorecard's verbatim detail string. Lets two
    distinct defects of the SAME class in the SAME doc (two stale pins, two dead
    links) carry distinct source_ids, while a re-detected identical defect keeps
    its id so the seen-cache rung still catches it run-to-run."""
    return hashlib.sha1(detail.encode("utf-8")).hexdigest()[:8]


def make_defect(*, bucket: str, doc: str, defect_class: str, detail: str,
                score: float | None) -> dict[str, Any]:
    """Build one defect record. `source_id` is the dedup anchor: it binds the
    exact doc + defect class + a fingerprint of the verbatim detail."""
    doc = (doc or "").strip()
    defect_class = (defect_class or "doc").strip()
    detail = detail.strip()
    source_id = f"learning-debt:{doc}:{defect_class}:{_fingerprint(detail)}"
    return {
        "bucket": bucket, "doc": doc, "defect_class": defect_class,
        "detail": detail, "score": score, "source_id": source_id,
    }


def _classify_coverage(s: str) -> tuple[str, str]:
    """Map a coverage-defect string to (defect_class, cited_doc_or_topic).
    The strings are emitted verbatim by the scorecard's `coverage()`:
        'orphan lesson (unreachable from any front door): <path>'
        'uncovered learning topic: <topic>'
    """
    if s.startswith("orphan lesson"):
        return "orphan-lesson", s.rsplit(": ", 1)[-1].strip()
    if s.startswith("uncovered learning topic"):
        return "missing-course", s.split(":", 1)[-1].strip()
    # Generic fallback: '<class>: <doc>' — keep the tool robust to new coverage
    # defect kinds the scorecard may add later.
    cls, _, doc = s.partition(": ")
    return (cls.strip() or "coverage"), doc.strip()


def extract_defects(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull every HARD defect out of a learning-scorecard JSON payload, from all
    three buckets it carries: per-doc pedagogy/hygiene defects, corpus coverage
    defects (orphans + missing courses), and the stamp-freshness defect.

    In-doc defect strings are `'<kpi>: <detail>'` (the kpi is the defect class:
    freshness, links, orientation, runnable, worked, honesty, structure)."""
    out: list[dict[str, Any]] = []

    for d in payload.get("docs", []) or []:
        doc = d.get("path", "")
        score = d.get("score")
        for s in d.get("defects", []) or []:
            cls, sep, _rest = str(s).partition(": ")
            defect_class = cls.strip() if sep else "doc"
            out.append(make_defect(bucket="doc", doc=doc, defect_class=defect_class,
                                   detail=str(s), score=score))

    cov = payload.get("coverage") or {}
    for s in cov.get("defects", []) or []:
        defect_class, cited = _classify_coverage(str(s))
        out.append(make_defect(bucket="coverage", doc=cited,
                               defect_class=defect_class, detail=str(s), score=None))

    stamp = payload.get("stamp_freshness") or {}
    if stamp.get("stale_stamp"):
        detail = (stamp.get("flag") or stamp.get("detail")
                  or "stale learning-scorecard stamp")
        out.append(make_defect(bucket="stamp", doc=stamp.get("path", STAMP_DOC),
                               defect_class="stale-stamp", detail=str(detail),
                               score=None))
    return out


def priority(defect: dict[str, Any]) -> int:
    """Transparent integer priority: defect-class weight (×100, so class dominates
    ordering) plus how broken the doc is (100 - score, 0-99, breaking ties within
    a class). Higher = filed first."""
    weight = CLASS_WEIGHT.get(defect["defect_class"], DEFAULT_WEIGHT)
    score = defect.get("score")
    brokenness = 0
    if isinstance(score, (int, float)):
        brokenness = max(0, min(99, int(round(100 - score))))
    return weight * 100 + brokenness


def existing_issue_index(issues: list[dict[str, Any]],
                         ) -> tuple[set[str], list[set[str]]]:
    """Build the dedup index from existing issues: every source_id already stamped
    in a body, and the title token-sets for near-dup detection."""
    stamped: set[str] = set()
    title_sets: list[set[str]] = []
    for iss in issues:
        body = iss.get("body") or ""
        title_sets.append(tokenize(iss.get("title") or ""))
        for m in re.findall(rf"{SOURCE_MARKER}:\s*([^\s>]+)", body):
            stamped.add(m.strip())
    return stamped, title_sets


def is_duplicate(defect: dict[str, Any], seen: dict[str, Any], stamped: set[str],
                 title_sets: list[set[str]], dup_jaccard: float) -> str | None:
    """Return the dedup rung that fires ('seen-cache' / 'issue-body' /
    'title-near'), or None if the defect is genuinely new."""
    sid = defect["source_id"]
    if sid in seen:
        return "seen-cache"
    if sid in stamped:
        return "issue-body"
    ttoks = tokenize(issue_title(defect))
    for tset in title_sets:
        if jaccard(ttoks, tset) >= dup_jaccard:
            return "title-near"
    return None


def issue_title(defect: dict[str, Any]) -> str:
    """The issue title: cites the defect class + the doc/topic it lives in.
    Kept short (the doc tail is enough to disambiguate)."""
    doc = defect["doc"] or "(corpus)"
    tail = doc.split("/")[-1] if "/" in doc else doc
    title = f"learning-debt: {defect['defect_class']} in {tail}"
    if len(title) > 110:
        title = title[:107].rstrip() + "…"
    return title


def render_issue(defect: dict[str, Any], today: str) -> dict[str, Any]:
    """Build the {title, body, labels} an issue is created from. The body cites
    the EXACT doc + defect class + the scorecard's verbatim detail (no prose
    drift), and stamps the source_id marker that powers the issue-body dedup rung."""
    doc = defect["doc"] or "(corpus)"
    body = (
        f"> Auto-filed by **learning-debt dispatch** "
        f"(`tools/learning_debt_dispatch.py`, {today}) from a HARD defect the "
        f"learning scorecard (`{SCORECARD_REL}`) found on disk. Fix the doc, then "
        f"re-run the scorecard to clear it; close as `wontfix` if it is not worth "
        f"fixing.\n\n"
        f"**Doc:** `{doc}`\n\n"
        f"**Defect class:** `{defect['defect_class']}`  (bucket: {defect['bucket']})\n\n"
        f"**Scorecard detail (verbatim):**\n\n```\n{defect['detail']}\n```\n\n"
        f"**Priority** (class weight + how broken the doc is): {priority(defect)}\n\n"
        f"---\n"
        f"_Reproduce: `python {SCORECARD_REL} --json` and look for this defect "
        f"under `{doc}`._\n"
        f"<!-- {SOURCE_MARKER}: {defect['source_id']} -->"
    )
    return {
        "title": issue_title(defect), "body": body,
        "labels": [DEBT_LABEL, "rsi"],
        "source_id": defect["source_id"], "doc": doc,
        "defect_class": defect["defect_class"], "priority": priority(defect),
    }


def plan_issues(defects: list[dict[str, Any]], seen: dict[str, Any],
                stamped: set[str], title_sets: list[set[str]],
                cfg: dict[str, Any], today: str,
                ) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Dedup → priority-sort → CAP. Returns (issues_to_file, skip_stats).
    Deterministic: defects are de-duplicated by source_id within the run and
    sorted by (priority desc, source_id) before the cap so the plan is stable."""
    stats = {"seen-cache": 0, "issue-body": 0, "title-near": 0, "within-run-dup": 0}
    planned: list[dict[str, Any]] = []
    run_seen: set[str] = set()
    for defect in defects:
        sid = defect["source_id"]
        if sid in run_seen:
            stats["within-run-dup"] += 1
            continue
        run_seen.add(sid)
        rung = is_duplicate(defect, seen, stamped, title_sets, cfg["dup_jaccard"])
        if rung:
            stats[rung] += 1
            continue
        planned.append(render_issue(defect, today))

    planned.sort(key=lambda r: (-r["priority"], r["source_id"]))
    return planned[: cfg["cap"]], stats


# ============================================================================
# I/O boundary — scorecard + gh. Thin wrappers so the logic above stays testable.
# ============================================================================
def run_scorecard(workspace: Path, timeout: int = 180) -> dict[str, Any]:
    """Run `learning_scorecard.py --json` and parse its payload. Raises
    RuntimeError on a non-zero exit so the caller can refuse (exit 2)."""
    script = workspace / SCORECARD_REL
    proc = subprocess.run(
        [sys.executable, str(script), "--json", "--workspace", str(workspace)],
        capture_output=True, text=True, encoding="utf-8", timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"learning_scorecard --json -> {proc.returncode}: "
                           f"{proc.stderr.strip()[:300]}")
    return json.loads(proc.stdout)


def load_scorecard(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def gh_json(args: list[str], timeout: int = 60) -> Any:
    proc = subprocess.run(["gh", *args], capture_output=True, text=True,
                          encoding="utf-8", timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} -> {proc.returncode}: "
                           f"{proc.stderr.strip()[:300]}")
    out = proc.stdout.strip()
    return json.loads(out) if out else []


def fetch_existing_issues(limit: int) -> list[dict[str, Any]]:
    return gh_json(["issue", "list", "--state", "all", "--limit", str(limit),
                    "--json", "number,title,body"])


def ensure_debt_label() -> None:
    """Idempotently create the marker label so `gh issue create` never fails on a
    missing label. Best-effort, but NOT silent — a real auth/permission failure
    would otherwise resurface as a confusing per-issue 'label not found'."""
    try:
        proc = subprocess.run(
            ["gh", "label", "create", DEBT_LABEL, "--color", "b60205",
             "--description",
             "Auto-filed by learning-debt dispatch (tools/learning_debt_dispatch.py); "
             "a HARD learning-scorecard defect to fix",
             "--force"],
            capture_output=True, text=True, encoding="utf-8", timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"warning: could not run `gh label create {DEBT_LABEL}`: {e}",
              file=sys.stderr)
        return
    if proc.returncode != 0:
        print(f"warning: could not ensure '{DEBT_LABEL}' label "
              f"(issue creation may fail): {proc.stderr.strip()[:200]}",
              file=sys.stderr)


def create_issue(issue: dict[str, Any]) -> str:
    args = ["issue", "create", "--title", issue["title"], "--body", issue["body"]]
    for lab in issue["labels"]:
        args += ["--label", lab]
    proc = subprocess.run(["gh", *args], capture_output=True, text=True,
                          encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"gh issue create -> {proc.returncode}: "
                           f"{proc.stderr.strip()[:300]}")
    return proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""


# ============================================================================
# Cache I/O.
# ============================================================================
def cache_path(workspace: Path) -> Path:
    return workspace / CACHE_DIRNAME / CACHE_FILENAME


def load_seen(workspace: Path) -> dict[str, Any]:
    p = cache_path(workspace)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data.get("seen", data) if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_seen(workspace: Path, seen: dict[str, Any]) -> None:
    p = cache_path(workspace)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"schema": SCHEMA, "seen": seen}, indent=2,
                            ensure_ascii=False), encoding="utf-8")


# ============================================================================
# Driver.
# ============================================================================
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Staleness -> backlog: learning-scorecard HARD defects -> "
                    "deduped, capped GitHub triage issues.")
    ap.add_argument("--workspace", default=".",
                    help="repo root (holds the cache + the scorecard). Default: cwd.")
    ap.add_argument("--scorecard",
                    help="path to a saved `learning_scorecard.py --json` payload "
                         "(default: run the scorecard live).")
    ap.add_argument("--cap", type=int,
                    help=f"hard cap on issues filed per run (default {DEFAULTS['cap']}).")
    ap.add_argument("--dup-jaccard", type=float,
                    help=f"title token-overlap to call a near-dup "
                         f"(default {DEFAULTS['dup_jaccard']}).")
    ap.add_argument("--live", action="store_true",
                    help="actually create issues + record them (default: dry-run).")
    ap.add_argument("--json", action="store_true", help="machine-readable output.")
    args = ap.parse_args(argv)

    workspace = Path(args.workspace).resolve()
    today = _now_utc().strftime("%Y-%m-%d")
    cfg = dict(DEFAULTS)
    if args.cap is not None:
        cfg["cap"] = args.cap
    if args.dup_jaccard is not None:
        cfg["dup_jaccard"] = args.dup_jaccard

    try:
        payload = (load_scorecard(args.scorecard) if args.scorecard
                   else run_scorecard(workspace))
    except (OSError, json.JSONDecodeError, RuntimeError,
            subprocess.SubprocessError) as e:
        print(f"scorecard error: {e}", file=sys.stderr)
        return 2

    defects = extract_defects(payload)

    # The existing-issue dedup index is mandatory: with no way to know what is
    # already filed, --live could re-storm the tracker. If gh fails AND there is
    # no cache, refuse rather than risk a spam run.
    seen = load_seen(workspace)
    errors: list[str] = []
    try:
        issues = fetch_existing_issues(cfg["issue_scan_limit"])
        stamped, title_sets = existing_issue_index(issues)
    except Exception as e:  # noqa: BLE001
        errors.append(f"issues: {e}")
        if not seen:
            print(f"refuse: cannot fetch existing issues and no seen-cache to fall "
                  f"back on ({e})", file=sys.stderr)
            return 2
        stamped, title_sets = set(), []

    to_file, skip_stats = plan_issues(defects, seen, stamped, title_sets, cfg, today)

    filed: list[dict[str, Any]] = []
    if args.live and to_file:
        ensure_debt_label()
        for issue in to_file:
            try:
                url = create_issue(issue)
            except Exception as e:  # noqa: BLE001
                errors.append(f"create[{issue['source_id']}]: {e}")
                continue
            seen[issue["source_id"]] = {
                "filed_at": today, "issue_url": url, "doc": issue["doc"],
                "defect_class": issue["defect_class"]}
            filed.append({**issue, "issue_url": url})
        if filed:
            save_seen(workspace, seen)

    result = {
        "schema": SCHEMA, "date": today, "mode": "live" if args.live else "dry-run",
        "defects_found": len(defects),
        "skipped": skip_stats,
        "planned": [
            {"title": i["title"], "labels": i["labels"], "doc": i["doc"],
             "defect_class": i["defect_class"], "source_id": i["source_id"],
             "priority": i["priority"]}
            for i in to_file],
        "filed": [{"title": f["title"], "issue_url": f.get("issue_url", "")}
                  for f in filed],
        "errors": errors,
    }

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    # Human report.
    print(f"learning-debt dispatch {today} — {result['mode']}")
    print(f"  found {len(defects)} HARD defect(s) in the learning scorecard")
    sk = ", ".join(f"{k}={v}" for k, v in skip_stats.items() if v) or "none"
    print(f"  deduped/dropped: {sk}")
    if not to_file:
        print("  → nothing new to file (clean, or every defect already tracked).")
    else:
        verb = "FILED" if args.live else "would file"
        print(f"  → {verb} {len(to_file)} issue(s) (cap {cfg['cap']}):")
        for i in to_file:
            mark = ""
            if args.live:
                f = next((x for x in filed if x["source_id"] == i["source_id"]), None)
                mark = f"  {f['issue_url']}" if f else "  (create failed)"
            print(f"     [{i['priority']:>5}] {i['title']}{mark}")
            print(f"             doc={i['doc']}  class={i['defect_class']}")
    if errors:
        print("  errors:")
        for e in errors:
            print(f"     ! {e}")
    if not args.live and to_file:
        print("\n  dry-run — file these for real with:  "
              "python tools/learning_debt_dispatch.py --live")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
