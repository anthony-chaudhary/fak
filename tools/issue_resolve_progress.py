#!/usr/bin/env python3
r"""Track the dispatch loop's progress toward an issue-resolution target (e.g. 50)
and (optionally) run the deterministic close arm — the proof instrument.

The operator's question is "is the auto-dispatcher actually moving the backlog,
and how far to 50?". This tick answers it from evidence, not narration:

  * SNAPSHOT — the current open / closed-by-the-loop / witnessed-not-yet-closed
    counts, plus the delta since a recorded baseline, appended to a durable JSONL
    (``.dispatch-runs/progress.jsonl``) so the trajectory is a curve, not a guess.
  * CLOSE (``--close``) — drive every OPEN_WITNESSED issue to CLOSED via
    ``issue_resolve_witnessed.py`` (each close re-verified per-SHA by
    ``dos commit-audit``). This is the bookkeeping arm: a shipped ``#N`` commit
    becomes a closed ticket. DRY-RUN unless ``--live``.

"Closed by the loop" is measured as issues whose closing comment carries the
close-arm's witness signature (so a human-closed or unrelated close is NOT
counted as the loop's work — the proof stays honest). The baseline is the first
snapshot's open-count, recorded once; ``resolved_toward_target`` is
``baseline_open - open_now`` clamped at 0, and ``target_remaining`` is
``max(0, target - resolved)``.

    python tools/issue_resolve_progress.py                 # snapshot only (dry)
    python tools/issue_resolve_progress.py --close --live  # snapshot + close witnessed
    python tools/issue_resolve_progress.py --target 50 --json

The CLI path appends this tick to fak's durable loop ledger by default
(``fak loop append``): ``fire``, admitted/refused ``admit``, ``end``, and a
``witness`` row when the snapshot has independent read-back. Disable with
``--no-loop-ledger`` for hermetic/manual probes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue_resolve_dispatch as loop_writer  # noqa: E402  (canonical fak loop append wrapper)

SCHEMA = "fleet-issue-resolve-progress/1"
RUNS_DIRNAME = ".dispatch-runs"
PROGRESS_LOG = "progress.jsonl"
BASELINE_FILE = "progress-baseline.json"
LOOP_ID = "issue-resolve-progress"
# The close-arm stamps this phrase into every close comment; we count only closes
# carrying it as "the loop's work" so the proof never inflates with foreign closes.
LOOP_CLOSE_SIGNATURE = "DOS dispatch loop's close-resolved arm"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _py() -> str:
    return sys.executable or "python"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_capture(cmd: list[str], cwd: Path, timeout: int) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s"
    except OSError as exc:
        return 127, "", str(exc)
    return proc.returncode, proc.stdout, proc.stderr


def open_issue_count(root: Path) -> int | None:
    rc, out, _ = run_capture(
        ["gh", "api", "repos/{owner}/{repo}", "--jq", ".open_issues_count"],
        root, timeout=60)
    if rc != 0:
        return None
    try:
        return int(out.strip())
    except ValueError:
        return None


def loop_closed_count(root: Path, *, limit: int = 200) -> int:
    """Issues closed carrying the close-arm's witness signature — the loop's own
    work, not a foreign/human close. Best effort: 0 if gh is unavailable."""
    rc, out, _ = run_capture(
        ["gh", "issue", "list", "--state", "closed", "--limit", str(limit),
         "--json", "number"],
        root, timeout=90)
    if rc != 0:
        return 0
    try:
        closed = json.loads(out)
    except ValueError:
        return 0
    # Counting the signature on every closed issue's comments is N gh calls — too
    # slow for a tick. Instead read the close-arm's own run records: each live close
    # is recorded in progress.jsonl's `closed_now`. The durable count is the sum of
    # closed_now across history (see fold_closed below); this function is the cheap
    # upper-bound fallback (total closed) used only when no history exists.
    return len(closed) if isinstance(closed, list) else 0


def closure_audit(root: Path, *, max_commits: int) -> dict[str, Any]:
    rc, out, err = run_capture(
        [_py(), str(root / "tools" / "issue_closure_audit.py"), "--json",
         "--max-commits", str(max_commits)], root, timeout=300)
    try:
        return json.loads(out)
    except ValueError:
        return {"_error": (err or out or "no JSON").strip()[-300:]}


def witnessed_open(audit: dict[str, Any]) -> list[int]:
    return [i.get("number") for i in (audit.get("issues") or [])
            if i.get("bucket") == "OPEN_WITNESSED" and i.get("number") is not None]


def load_baseline(runs_dir: Path) -> dict[str, Any] | None:
    f = runs_dir / BASELINE_FILE
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    return None


def save_baseline(runs_dir: Path, open_now: int) -> dict[str, Any]:
    runs_dir.mkdir(parents=True, exist_ok=True)
    base = {"baseline_open": open_now, "recorded_utc": _now()}
    (runs_dir / BASELINE_FILE).write_text(json.dumps(base, indent=2), encoding="utf-8")
    return base


def fold_closed_history(runs_dir: Path) -> int:
    """Sum of ``closed_now`` across every prior progress record — the durable
    count of issues THIS loop has driven to CLOSED (the honest proof metric)."""
    log = runs_dir / PROGRESS_LOG
    if not log.exists():
        return 0
    total = 0
    try:
        for line in log.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            total += int(rec.get("closed_now") or 0)
    except OSError:
        return 0
    return total


def run_close(root: Path, *, live: bool, audit_path: Path | None,
              limit: int) -> dict[str, Any]:
    cmd = [_py(), str(root / "tools" / "issue_resolve_witnessed.py"),
           "--limit", str(limit), "--json"]
    if audit_path:
        cmd += ["--audit-json", str(audit_path)]
    if live:
        cmd += ["--live"]
    rc, out, err = run_capture(cmd, root, timeout=300)
    try:
        doc = json.loads(out)
    except ValueError:
        return {"_error": (err or out or "no JSON").strip()[-300:], "closed": 0}
    counts = doc.get("counts") or {}
    return {"verdict": doc.get("verdict"), "closed": int(counts.get("closed") or 0),
            "would_close": int(counts.get("would_close") or 0),
            "skipped": int(counts.get("skipped_unwitnessed") or 0),
            "skipped_unpushed": int(counts.get("skipped_unpushed") or 0),
            "pushed_gate": doc.get("pushed_gate"),
            "failed": int(counts.get("failed") or 0)}


def append_progress(runs_dir: Path, rec: dict[str, Any]) -> None:
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        with open(runs_dir / PROGRESS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    except OSError:
        pass


def _metric_ints(rec: dict[str, Any]) -> dict[str, int]:
    metrics: dict[str, int] = {}
    for key in (
        "target", "open_now", "baseline_open", "resolved_toward_target",
        "target_remaining", "witnessed_open", "closed_now",
        "closed_by_loop_total",
    ):
        if rec.get(key) is not None:
            metrics[key] = int(rec[key])
    if rec.get("close_live") is not None:
        metrics["close_live"] = 1 if rec.get("close_live") else 0
    return metrics


def progress_run_id(rec: dict[str, Any]) -> str:
    stamp = str(rec.get("utc") or _now()).replace("-", "").replace(":", "")
    stamp = stamp.replace("Z", "").replace("T", "T")
    return f"progress-{stamp}"


def record_loop_tick(root: Path, rec: dict[str, Any],
                     *,
                     ledger: Path | None = None,
                     append: Any | None = None,
                     mint: Any | None = None) -> dict[str, Any]:
    """Lower one progress/close tick into fak loop-ledger events.

    The witness row is about the progress instrument's independent read-back
    (GitHub open-count + closure audit), not a worker's self-report.
    """
    ledger = ledger or loop_writer.default_loop_ledger(root)
    append = append or (lambda record, loop_ledger, event: loop_writer.append_loop_event(
        record, loop_ledger, event, source="issue_resolve_progress"))
    existing = str(rec.get("run_id") or "")
    minted = None
    if not existing:
        minted = (mint or loop_writer.mint_dos_run_id)(root, "issue-resolve-progress")
    if loop_writer.is_dos_run_id(existing):
        run_id = existing
    elif loop_writer.is_dos_run_id(minted):
        run_id = str(minted)
    else:
        run_id = existing or progress_run_id(rec)
    rec["run_id"] = run_id
    metrics = _metric_ints(rec)
    evidence = [("progress_log", str(root / RUNS_DIRNAME / PROGRESS_LOG))]
    for n in rec.get("witnessed_numbers") or []:
        evidence.append(("open_witnessed_issue", str(n)))
    status_reason = "OK" if rec.get("ok") else "OPEN_COUNT_UNAVAILABLE"
    if rec.get("audit_error"):
        status_reason = "AUDIT_UNAVAILABLE"

    events: list[dict[str, Any]] = [{
        "loop_id": LOOP_ID, "run_id": run_id, "kind": "fire",
        "summary": f"issue progress tick target={rec.get('target')}",
        "metrics": metrics, "evidence": evidence,
    }, {
        "loop_id": LOOP_ID, "run_id": run_id, "kind": "admit",
        "status": "admitted" if rec.get("ok") else "refused",
        "reason": status_reason,
        "summary": f"open={rec.get('open_now')} target_remaining={rec.get('target_remaining')}",
        "metrics": metrics, "evidence": evidence,
    }, {
        "loop_id": LOOP_ID, "run_id": run_id, "kind": "end",
        "status": "claimed_done" if rec.get("ok") else "failed",
        "reason": status_reason,
        "summary": f"closed_now={rec.get('closed_now')} witnessed_open={rec.get('witnessed_open')}",
        "metrics": metrics, "evidence": evidence,
    }]
    if rec.get("ok"):
        witness_status = "witness_unavailable" if rec.get("audit_error") else "witnessed_done"
        verified_state = "verified_unavailable" if rec.get("audit_error") else "verified_done"
        events.append({
            "loop_id": LOOP_ID, "run_id": run_id, "kind": "witness",
            "status": witness_status,
            "verified_state": verified_state,
            "reason": status_reason,
            "summary": f"open_count={rec.get('open_now')} audit_error={rec.get('audit_error') or ''}",
            "metrics": metrics, "evidence": evidence,
        })

    rows = [append(root, ledger, ev) for ev in events]
    return {
        "ledger": str(ledger),
        "loop_id": LOOP_ID,
        "run_id": run_id,
        "events": rows,
        "ok": all(r.get("ok") for r in rows) if rows else True,
    }


def evaluate(root: Path, *, target: int, do_close: bool, live: bool,
             max_commits: int,
             record_loop: bool = False,
             loop_ledger: Path | None = None) -> dict[str, Any]:
    runs_dir = root / RUNS_DIRNAME
    open_now = open_issue_count(root)
    audit = closure_audit(root, max_commits=max_commits)
    witnessed = witnessed_open(audit) if "_error" not in audit else []

    baseline = load_baseline(runs_dir)
    if baseline is None and open_now is not None:
        baseline = save_baseline(runs_dir, open_now)
    baseline_open = (baseline or {}).get("baseline_open")

    closed_now = 0
    close_result: dict[str, Any] | None = None
    if do_close and witnessed:
        # Re-run the audit to a file the close-arm can consume (avoid a 2nd scan).
        audit_path = runs_dir / "progress-audit.json"
        try:
            runs_dir.mkdir(parents=True, exist_ok=True)
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
        except OSError:
            audit_path = None  # close-arm will scan fresh
        close_result = run_close(root, live=live, audit_path=audit_path,
                                 limit=len(witnessed))
        closed_now = close_result.get("closed", 0) if live else 0

    # Durable proof metric: total closed by the loop across all ticks (+ this one).
    closed_total = fold_closed_history(runs_dir) + closed_now
    resolved = None
    if baseline_open is not None and open_now is not None:
        resolved = max(0, baseline_open - open_now)
    target_remaining = (max(0, target - resolved) if resolved is not None
                        else None)

    # A snapshot is OK as long as we got a live open-count — that is the proof
    # metric. A closure-audit hiccup (e.g. `dos` momentarily unreachable under a
    # hidden-window scheduled task) only blanks the witnessed count for this tick;
    # it must NOT fail the whole tick, or the always-on curve develops gaps and the
    # task's LastResult flaps to 1 on a transient. Surface the audit error in the
    # record, but key `ok` (and the exit code) on the open-count alone.
    ok = open_now is not None
    rec = {
        "schema": SCHEMA, "utc": _now(), "target": target, "ok": ok,
        "open_now": open_now, "baseline_open": baseline_open,
        "resolved_toward_target": resolved, "target_remaining": target_remaining,
        "witnessed_open": len(witnessed), "witnessed_numbers": witnessed[:50],
        "closed_now": closed_now, "closed_by_loop_total": closed_total,
        "close_live": live if do_close else None,
        "close_result": close_result,
        "audit_error": audit.get("_error"),
    }
    append_progress(runs_dir, rec)   # rec already carries `ok` — the log is honest
    if record_loop:
        rec["loop_ledger"] = record_loop_tick(root, rec, ledger=loop_ledger)
    return rec


def render(p: dict[str, Any]) -> str:
    tgt = p.get("target")
    res = p.get("resolved_toward_target")
    rem = p.get("target_remaining")
    bar = ""
    if isinstance(res, int) and isinstance(tgt, int) and tgt > 0:
        filled = min(tgt, res)
        width = 30
        n = int(width * filled / tgt)
        bar = "[" + "#" * n + "-" * (width - n) + f"] {filled}/{tgt}"
    lines = [
        f"issue-resolve-progress: open={p.get('open_now')} "
        f"(baseline {p.get('baseline_open')})  toward {tgt}: {bar or f'{res}/{tgt}'}",
        f"  witnessed-open (closeable now): {p.get('witnessed_open')}  "
        f"{p.get('witnessed_numbers') or ''}",
        f"  closed this tick: {p.get('closed_now')}  "
        f"closed-by-loop total: {p.get('closed_by_loop_total')}  "
        f"remaining to {tgt}: {rem}",
    ]
    cr = p.get("close_result")
    if cr:
        lines.append(f"  close arm: verdict={cr.get('verdict')} closed={cr.get('closed')} "
                     f"would_close={cr.get('would_close')} failed={cr.get('failed')}")
    if p.get("audit_error"):
        lines.append(f"  ! audit error: {p['audit_error']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Track dispatch progress toward an issue-resolution target; "
                    "optionally close witnessed issues.")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--target", type=int, default=50, help="resolution target (default: 50)")
    ap.add_argument("--close", action="store_true",
                    help="also run the close arm on OPEN_WITNESSED issues")
    ap.add_argument("--live", action="store_true",
                    help="with --close, execute the gh closes (default: dry-run)")
    ap.add_argument("--max-commits", type=int, default=2000,
                    help="git history budget for the closure audit; must stay "
                         "above the repo's commit count or resolving commits "
                         "older than the window can't bind a witnessed close "
                         "(default: 2000, matching issue_closure_audit.py)")
    ap.add_argument("--loop-ledger", default="",
                    help="append this tick to a fak loop ledger (default: FAK_LOOP_LEDGER or .fak/loops.jsonl)")
    ap.add_argument("--no-loop-ledger", action="store_true",
                    help="disable loop-ledger append for this tick")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    p = evaluate(root, target=args.target, do_close=args.close, live=args.live,
                 max_commits=args.max_commits,
                 record_loop=not args.no_loop_ledger,
                 loop_ledger=(Path(args.loop_ledger).resolve()
                              if args.loop_ledger else None))
    print(json.dumps(p, indent=2) if args.json else render(p))
    return 0 if p.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
