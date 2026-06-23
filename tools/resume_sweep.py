#!/usr/bin/env python3
r"""resume_sweep.py -- discover EVERY recently-crashed session across ALL local
``~/.claude*`` accounts and bucket each by the action it actually needs.

Why this exists (the gap it closes):
  ``resume_watch.py`` only classifies the sessions already listed in
  ``tools/_registry/resume_watch_manifest.json``. A crash wave that the manifest
  never recorded -- e.g. a whole account's workers capping at once -- is therefore
  INVISIBLE to it: the operator sees "all watched sessions healthy" while a dozen
  real sessions sit stranded under a different account. This sweep is the
  manifest-free discovery half: it walks the transcripts on disk, not a registry.

What it does (read-only):
  * scan ``<home>/.claude*/projects/*/<sid>.jsonl`` across every account,
  * keep sessions whose NEWEST copy was touched within ``--window`` minutes (the
    current crash wave, not ancient history),
  * read each terminal turn and classify with the shared ``fleet_session_signals``:
      LIMIT_RESET_PASSED  usage cap whose reset window has elapsed   -> resumable now
      LIMIT_RESET_FUTURE  usage cap still in its window              -> wait for reset
      API_ERR             transient 529/overload/transport error     -> resumable now
      AUTH                login/credit/access wall                    -> needs /login, NOT resume
      LIVE                a claude.exe is already driving the sid     -> leave alone
  * for each, resolve the SUPERSET copy (uuid-set + last-ts, NOT file mtime -- a
    re-capped resume rewrites only the banner and bumps mtime on a stale PREFIX) and
    the per-session cwd (recovered from the project slug, so sessions in other repos
    resume in their own tree), and report which account holds it + whether that seat
    is healthy.

It RESUMES NOTHING. Like ``resume_watch``, launching is a separate, gated step:
this tool's job is to make the full actionable set visible and correctly bucketed.
``--json`` emits the machine record for a launcher to consume.

Usage:
  python tools/resume_sweep.py                 # human summary (default 600m window)
  python tools/resume_sweep.py --window 1440   # last 24h
  python tools/resume_sweep.py --json          # machine record
  python tools/resume_sweep.py --probe         # also probe seat health (slower)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import fleet_session_signals as sig  # noqa: E402

HOME = os.path.expanduser("~")
LEDGER = os.path.join(HERE, "_registry", "resume_ledger.jsonl")


def recently_resumed_sids(window_min: float, now_utc) -> set:
    """sids the resume ledger shows we (re)launched within the window. During an
    active resume pass a session's OLD copies still terminate on their pre-resume
    error, so without this the sweep re-flags work already in flight. Reading the
    ledger -- the record of what was actually launched -- is the honest dedup."""
    from datetime import datetime, timezone
    cutoff = now_utc.timestamp() - window_min * 60
    out = set()
    try:
        with open(LEDGER, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rec = json.loads(ln)
                except ValueError:
                    continue
                ts = rec.get("ts")
                sid = rec.get("session")
                if not ts or not sid:
                    continue
                try:
                    t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except ValueError:
                    continue
                if t.timestamp() >= cutoff:
                    out.add(sid)
    except OSError:
        pass
    return out


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _text(rec: dict) -> str:
    msg = rec.get("message") or rec
    content = msg.get("content") if isinstance(msg, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content
                         if isinstance(b, dict) and b.get("type") == "text")
    return ""


def _role(rec: dict):
    m = rec.get("message")
    if isinstance(m, dict):
        return m.get("role")
    return rec.get("role")


def _clip(text: str, width: int = 90) -> str:
    """One-line, length-bounded evidence snippet for the observability fields."""
    s = " ".join((text or "").split())
    return s if len(s) <= width else s[: width - 1] + "…"


def _load(path: str) -> list:
    out = []
    try:
        with open(path, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    out.append(json.loads(ln))
                except ValueError:
                    continue
    except OSError:
        pass
    return out


def _uuids(recs) -> set:
    return {r.get("uuid") for r in recs if r.get("uuid")}


def _last_ts(recs) -> str:
    for r in reversed(recs):
        if r.get("timestamp"):
            return r["timestamp"]
    return ""


def _slugify(path: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "-", path)


def cwd_for_slug(proj: str, fallback: str) -> str:
    """Recover the real cwd for a project slug. The slug is lossy (``-`` collapses
    both path separators and real hyphens), so enumerate plausible roots and match
    by slug rather than string-reversing."""
    roots = (glob.glob("C:\\work\\*") + glob.glob("C:\\Users\\*") + glob.glob("C:\\*"))
    for d in roots:
        if os.path.isdir(d) and _slugify(d) == proj:
            return d
    return fallback


def live_resume_sids() -> set:
    """sids a running ``claude.exe`` is currently driving (--resume <sid>)."""
    try:
        ps = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='claude.exe'\" | "
             "ForEach-Object { $_.CommandLine }"],
            capture_output=True, text=True, timeout=40).stdout
    except Exception:
        return set()
    return set(re.findall(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", ps))


def probe_seat_status() -> dict:
    """{normalized config dir: status} from account_probe --all. Empty on failure."""
    try:
        out = subprocess.run(
            [sys.executable, os.path.join(HERE, "account_probe.py"),
             "--all", "--json", "--no-ledger", "--timeout", "35"],
            capture_output=True, text=True, timeout=120).stdout
        data = json.loads(out)
        rows = data if isinstance(data, list) else next(
            (v for v in data.values() if isinstance(v, list)), [])
    except Exception:
        return {}
    return {os.path.normcase(os.path.abspath(r["dir"])): r.get("status")
            for r in rows if r.get("dir")}


def classify(sid: str, paths: list, live: set, now_utc) -> dict:
    """Bucket one session from its newest copy's terminal turn + resolve superset."""
    data = {p: _load(p) for p in paths}
    # superset = latest last_ts, then most records (NOT file mtime).
    best = max(paths, key=lambda p: (_last_ts(data[p]), len(data[p])))
    best_recs = data[best]
    is_superset = all(_uuids(data[p]) <= _uuids(best_recs) for p in paths)
    acct = os.path.basename(best.split(os.sep + "projects" + os.sep)[0])
    proj = os.path.basename(os.path.dirname(best))

    last_a = last_e = None
    for r in best_recs:
        if _role(r) == "assistant" and _text(r).strip():
            last_a = r
        if r.get("type") == "error" or r.get("isApiErrorMessage"):
            last_e = r
    # Adjudicate the FAILURE MODE off the error record ONLY -- never the assistant prose.
    # The old code blended last_assistant + last_error into one `blob`, so a worker that
    # merely *narrated* an auth wall / 529 / usage limit in its final turn was mis-bucketed
    # into that failure (2026-06-23: gem7 732edb34 narrated "the gem7 auth wall ... logged
    # back in" while its real error record was a transient 529 -> wrongly AUTH). The
    # taxonomy now lives in one shared place (sig.terminal_failure), the same error-channel
    # discipline resume_watch.terminal_auth_failure uses. For observability we also compute
    # what the prose ALONE would have said and flag the divergence, so a prose-only
    # false-positive the fix averted is visible instead of silent.
    err_text = _text(last_e) if last_e else ""
    prose_text = _text(last_a) if last_a else ""
    kind, detail = sig.terminal_failure(err_text)
    prose_kind, _ = sig.terminal_failure(prose_text)
    prose_diverged = bool(prose_kind) and prose_kind != kind

    reset = ""
    if sid in live:
        bucket = "LIVE"
    elif kind == "AUTH":
        bucket = "AUTH"
    elif kind == "LIMIT":
        reset = detail
        anchor = _last_ts(best_recs)
        anchor_dt = None
        if anchor:
            try:
                anchor_dt = datetime.fromisoformat(anchor.replace("Z", "+00:00"))
            except ValueError:
                anchor_dt = None
        passed = sig.reset_passed(detail, now_utc=now_utc, anchor_utc=anchor_dt)
        bucket = "LIMIT_RESET_PASSED" if passed else "LIMIT_RESET_FUTURE"
    elif kind == "API_ERR":
        bucket = "API_ERR"
    else:
        bucket = "OTHER"

    return {
        "sid": sid, "bucket": bucket, "superset_account": acct, "project": proj,
        "is_superset": is_superset, "n_records": len(best_recs), "copies": len(paths),
        "reset": reset,
        # observability: the non-forgeable error text that drove the bucket, and whether
        # the assistant prose alone would have disagreed (a prose false-positive averted).
        "evidence": _clip(err_text) if bucket not in ("LIVE", "OTHER") else "",
        "prose_diverged": prose_diverged,
        "cwd": cwd_for_slug(proj, os.getcwd()),
        "superset_path": best,
    }


def sweep(window_min: float, *, probe: bool = False,
          exclude_resumed: bool = True) -> dict:
    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc.timestamp() - window_min * 60
    by_sid: dict = {}
    for p in glob.glob(os.path.join(HOME, ".claude*", "projects", "*", "*.jsonl")):
        sid = os.path.basename(p)[:-6]
        if len(sid) != 36:
            continue
        by_sid.setdefault(sid, []).append(p)

    live = live_resume_sids()
    seat = probe_seat_status() if probe else {}
    resumed = recently_resumed_sids(window_min, now_utc) if exclude_resumed else set()

    rows = []
    n_excluded = 0
    for sid, paths in by_sid.items():
        newest = max(paths, key=os.path.getmtime)
        if os.path.getmtime(newest) < cutoff:
            continue
        if sid in resumed:
            n_excluded += 1
            continue
        r = classify(sid, paths, live, now_utc)
        if r["bucket"] == "OTHER":
            continue
        if probe:
            cfg = os.path.join(HOME, r["superset_account"])
            r["seat_ok"] = seat.get(os.path.normcase(os.path.abspath(cfg))) == "OK"
        rows.append(r)

    order = {"LIMIT_RESET_PASSED": 0, "API_ERR": 1, "LIMIT_RESET_FUTURE": 2,
             "AUTH": 3, "LIVE": 4}
    rows.sort(key=lambda r: (order.get(r["bucket"], 9), r["superset_account"], -r["n_records"]))
    return {"ts": now_iso(), "window_min": window_min, "count": len(rows),
            "excluded_recently_resumed": n_excluded, "rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="resume_sweep", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", type=float, default=600,
                    help="only sessions whose newest copy changed within N minutes (default 600)")
    ap.add_argument("--min-records", type=int, default=0,
                    help="only sessions with >= N transcript records -- filters out the "
                         "batch-spawned micro-stubs (n=10-11) that cap before doing real work")
    ap.add_argument("--probe", action="store_true",
                    help="also probe seat health so each row carries seat_ok (slower)")
    ap.add_argument("--json", action="store_true", help="emit the machine record")
    ap.add_argument("--bucket", default=None,
                    help="filter to one bucket (e.g. LIMIT_RESET_PASSED)")
    ap.add_argument("--include-resumed", action="store_true",
                    help="don't exclude sessions the ledger shows were already resumed "
                         "in-window (default: exclude them so an active pass isn't re-flagged)")
    args = ap.parse_args(argv)

    res = sweep(args.window, probe=args.probe, exclude_resumed=not args.include_resumed)
    rows = res["rows"]
    dropped_stub = 0
    if args.min_records > 0:
        before = len(rows)
        rows = [r for r in rows if r["n_records"] >= args.min_records]
        dropped_stub = before - len(rows)
        res = {**res, "rows": rows, "count": len(rows), "dropped_below_min_records": dropped_stub}
    if args.bucket:
        rows = [r for r in rows if r["bucket"] == args.bucket]
        res = {**res, "rows": rows, "count": len(rows)}

    if args.json:
        print(json.dumps(res, indent=1))
        return 0

    from collections import Counter
    c = Counter(r["bucket"] for r in rows)
    hdr = f"resume_sweep {res['ts']}  window={int(args.window)}m"
    if args.min_records:
        hdr += f"  min_records={args.min_records}"
    print(hdr + "  " + "  ".join(f"{k}={v}" for k, v in sorted(c.items())))
    if res.get("excluded_recently_resumed") and not args.include_resumed:
        print(f"  (excluded {res['excluded_recently_resumed']} session(s) already resumed "
              f"in-window per the ledger -- pass --include-resumed to show them)")
    if dropped_stub:
        print(f"  (dropped {dropped_stub} micro-stub session(s) below {args.min_records} records "
              f"-- raise/lower --min-records to see them)")
    diverged = sum(1 for r in rows if r.get("prose_diverged"))
    if diverged:
        print(f"  (averted {diverged} prose-only false-positive(s): the final assistant prose "
              f"alone would have mis-bucketed these, but the error channel overruled it "
              f"-- shown as [prose≠err])")
    # grouped view: project x bucket, so a batch-spawned cohort reads as ONE line.
    grp = Counter((r["project"], r["bucket"], r["superset_account"]) for r in rows)
    if len([g for g in grp.values() if g >= 5]):
        print("  groups (project / bucket / account >=5):")
        for (proj, bkt, acct), n in sorted(grp.items(), key=lambda x: -x[1]):
            if n >= 5:
                print(f"     {n:3}  {proj:22} {bkt:18} {acct}")
    for r in rows:
        seat = ""
        if "seat_ok" in r:
            seat = f" seat_ok={r['seat_ok']}"
        sup = "" if r["is_superset"] else " NON-SUPERSET!"
        div = " [prose≠err]" if r.get("prose_diverged") else ""
        print(f"  {r['bucket']:18} {r['sid'][:8]} {r['superset_account']:22} "
              f"proj={r['project']:20} n={r['n_records']:<4} "
              f"reset={r['reset'] or '-':28}{seat}{sup}{div}")
    if not rows:
        print("  (no recently-crashed sessions in window)")
    # action hints
    np = sum(1 for r in rows if r["bucket"] in ("LIMIT_RESET_PASSED", "API_ERR"))
    if np:
        print(f"\n{np} resumable NOW (LIMIT_RESET_PASSED + API_ERR) -- pin to the "
              f"superset_account if healthy, else re-home; launch sequential.")
    nf = sum(1 for r in rows if r["bucket"] == "LIMIT_RESET_FUTURE")
    if nf:
        print(f"{nf} waiting on a usage reset -- resume in place after the named window.")
    na = sum(1 for r in rows if r["bucket"] == "AUTH")
    if na:
        print(f"{na} AUTH-walled -- need `claude /login` (a re-resume on the same account can't fix it).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
