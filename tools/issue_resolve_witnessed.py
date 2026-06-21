#!/usr/bin/env python3
r"""The dispatch loop's *close-the-resolved* arm: drive OPEN_WITNESSED issues to
CLOSED, each gated on a witness the loop did not author.

``issue_closure_audit.py`` surfaces issues bucketed ``OPEN_WITNESSED`` — still
open on GitHub, yet already carrying a *diff-witnessed* resolving commit in git
ancestry. Those are exactly the tickets a correct dispatcher should close: the
work shipped, only the bookkeeping lags, and every extra OPEN_WITNESSED row drags
``closure_rate`` down. This is the deterministic half of "do N issues" — no model
worker, no code edit, no DoS — and it is the safest possible live proof that the
loop can move real issues, because the keep-bit is a git fact:

  for each candidate:
    re-run `dos commit-audit <sha> --json`     # env-authored, re-verified HERE
    iff verdict==OK and witness==diff-witnessed:
       gh issue close <n> --comment "<sha> (<subject>) resolves this; ..."

The re-verification is the whole point (.claude/rsi-loop-dod.md: "no keep on a
self-authored claim"): the closer does NOT trust the audit's bucket, it re-asks
the oracle per-SHA at close time. A close cites its witnessing SHA + subject in
the comment, so it is auditable and trivially reversible (``gh issue reopen``).

DRY-RUN BY DEFAULT — prints the exact `gh` commands and the per-issue witness.
``--live`` executes. ``--limit N`` bounds the batch (default 10).

    python tools/issue_resolve_witnessed.py                  # plan 10 closes (dry-run)
    python tools/issue_resolve_witnessed.py --limit 10 --live  # execute
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

SCHEMA = "fleet-issue-resolve-witnessed/1"
WITNESS_OK = "diff-witnessed"


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def _py() -> str:
    return sys.executable or "python"


def run_capture(cmd: list[str], cwd: Path, timeout: int) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s"
    except OSError as exc:
        return 127, "", str(exc)
    return proc.returncode, proc.stdout, proc.stderr


def load_audit(root: Path, audit_json: str | None, max_commits: int) -> dict[str, Any]:
    """Get the closure audit: from a provided JSON file, else run it fresh."""
    if audit_json:
        try:
            return json.loads(Path(audit_json).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return {"_error": f"could not read --audit-json: {exc}"}
    rc, out, err = run_capture(
        [_py(), str(root / "tools" / "issue_closure_audit.py"), "--json",
         "--max-commits", str(max_commits)], root, timeout=300)
    try:
        return json.loads(out)
    except ValueError:
        return {"_error": (err or out or "closure audit produced no JSON").strip()[-400:]}


def open_witnessed(audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for i in audit.get("issues") or []:
        if i.get("bucket") != "OPEN_WITNESSED":
            continue
        wc = i.get("witnessed_commits") or i.get("resolving_commits") or []
        first = wc[0] if wc else None
        sha = (first.get("sha") if isinstance(first, dict) else first) or ""
        subject = (first.get("subject") if isinstance(first, dict) else "") or ""
        rows.append({"number": i.get("number"), "title": i.get("title") or "",
                     "sha": str(sha), "subject": str(subject)})
    rows.sort(key=lambda r: -(r["number"] or 0))
    return rows


def reverify(root: Path, sha: str) -> dict[str, Any]:
    """Re-ask the oracle at close time — do NOT trust the audit's bucket."""
    if not sha:
        return {"witness_ok": False, "reason": "no witnessing sha"}
    rc, out, err = run_capture(
        ["dos", "commit-audit", sha, "--workspace", str(root), "--json"],
        root, timeout=60)
    # `dos commit-audit --json` emits a JSON ARRAY (one row per audited sha).
    doc: dict[str, Any] = {}
    try:
        parsed = json.loads(out.strip()) if out.strip() else []
    except ValueError:
        parsed = []
    if isinstance(parsed, list):
        # prefer the row whose sha matches; else the first row.
        doc = next((r for r in parsed if isinstance(r, dict)
                    and str(r.get("sha")) and str(sha).startswith(str(r.get("sha")))),
                   parsed[0] if parsed and isinstance(parsed[0], dict) else {})
    elif isinstance(parsed, dict):
        doc = parsed
    verdict = str(doc.get("verdict") or "")
    witness = str(doc.get("witness") or "")
    ok = verdict.upper() == "OK" and witness == WITNESS_OK
    return {"witness_ok": ok, "verdict": verdict or None, "witness": witness or None,
            "reason": None if ok else f"commit-audit verdict={verdict or '?'} witness={witness or '?'}"}


def close_comment(row: dict[str, Any]) -> str:
    subj = row.get("subject") or "resolving commit"
    return (f"Resolved by `{row['sha'][:10]}` ({subj}). Closed by the DOS dispatch "
            f"loop's close-resolved arm, witnessed via `dos commit-audit` "
            f"(verdict OK / diff-witnessed). Reopen if this does not fully resolve it.")


def close_cmd(row: dict[str, Any]) -> list[str]:
    return ["gh", "issue", "close", str(row["number"]), "--comment", close_comment(row)]


def evaluate(root: Path, *, limit: int, live: bool, audit_json: str | None,
             max_commits: int) -> dict[str, Any]:
    audit = load_audit(root, audit_json, max_commits)
    if audit.get("_error"):
        return {"schema": SCHEMA, "ok": False, "verdict": "ERROR",
                "reason": audit["_error"], "planned": [], "results": []}
    candidates = open_witnessed(audit)[:limit]
    planned, results = [], []
    closed = skipped = failed = 0
    for row in candidates:
        rv = reverify(root, row["sha"])
        item = {**row, **rv, "command": close_cmd(row)}
        planned.append(item)
        if not rv["witness_ok"]:
            item["action"] = "skip_unwitnessed"
            skipped += 1
            results.append(item)
            continue
        if not live:
            item["action"] = "would_close"
            results.append(item)
            continue
        rc, out, err = run_capture(close_cmd(row), root, timeout=60)
        item["action"] = "closed" if rc == 0 else "close_failed"
        item["returncode"] = rc
        if rc == 0:
            closed += 1
        else:
            item["error"] = (err or out).strip()[-300:]
            failed += 1
        results.append(item)
    ok = failed == 0 and (live or bool(candidates))
    return {
        "schema": SCHEMA, "ok": ok,
        "verdict": ("CLOSED" if live and closed else
                    "PLANNED" if not live else "NO_CLOSES"),
        "live": live, "limit": limit,
        "candidates_total": len(open_witnessed(audit)),
        "planned_count": len(planned),
        "counts": {"closed": closed, "would_close": sum(
            1 for r in results if r.get("action") == "would_close"),
            "skipped_unwitnessed": skipped, "failed": failed},
        "closure_rate_before": audit.get("closure_rate"),
        "results": results,
    }


def render(p: dict[str, Any]) -> str:
    c = p.get("counts") or {}
    lines = [f"resolve-witnessed: {p.get('verdict')} ({'ok' if p.get('ok') else 'action'})  "
             f"live={p.get('live')}  candidates={p.get('candidates_total')} "
             f"planned={p.get('planned_count')}"]
    for r in p.get("results") or []:
        mark = {"closed": "[CLOSED]", "would_close": "[would-close]",
                "skip_unwitnessed": "[SKIP no-witness]", "close_failed": "[FAILED]"}.get(
                    r.get("action"), "[?]")
        lines.append(f"  {mark} #{r.get('number')} {r.get('sha','')[:10]}  "
                     f"{(r.get('title') or '')[:50]}")
    lines.append(f"  -> closed={c.get('closed')} would_close={c.get('would_close')} "
                 f"skipped={c.get('skipped_unwitnessed')} failed={c.get('failed')}  "
                 f"(closure_rate before={p.get('closure_rate_before')})")
    if not p.get("live"):
        lines.append("  DRY-RUN — re-run with --live to execute the gh closes")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Close OPEN_WITNESSED issues, each re-verified via dos commit-audit.")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--limit", type=int, default=10, help="max issues to close (default: 10)")
    ap.add_argument("--live", action="store_true", help="execute the gh closes (default: dry-run)")
    ap.add_argument("--audit-json", default=None,
                    help="path to a saved issue_closure_audit --json (else run it fresh)")
    ap.add_argument("--max-commits", type=int, default=600,
                    help="git history budget when running the audit fresh (default: 600)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = evaluate(root, limit=args.limit, live=args.live,
                       audit_json=args.audit_json, max_commits=args.max_commits)
    print(json.dumps(payload, indent=2) if args.json else render(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
