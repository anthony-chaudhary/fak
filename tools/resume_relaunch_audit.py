#!/usr/bin/env python3
"""Audit the actual RELAUNCH OUTCOME of every session in the resume ledger.

``resume_sweep`` answers "what failure is this crashed session in?" (crash state).
This answers the downstream question the sweep cannot: of the sessions a relaunch was
*attempted* on (the ledger records the ATTEMPT), which actually TOOK -- advanced past
their error -- and which are still STRANDED on it? The ledger is a self-report of the
attempt; it does not prove the outcome. This verifies the outcome from the transcript,
never the ledger's word -- the same distrust discipline as ``dos_verify``.

A session is ``RELAUNCHED_OK`` iff its newest copy's last real (non-banner) assistant
turn is NEWER than its last error record. Else it is ``STRANDED`` on that error,
classified by the shared ``fleet_session_signals.terminal_failure`` taxonomy
(AUTH / LIMIT / API_ERR). ``NEVER_WORKED`` if it produced no real turn at all;
``NO_TRANSCRIPT`` if no copy is on disk.

Liveness is a SECONDARY signal: a headless ``claude --resume -p`` runs one turn then
EXITS, so ``RELAUNCHED_OK`` with no live process is the normal healthy state. ``--live``
cross-references currently-running ``claude --resume <sid>`` workers so a session that is
mid-relaunch right now is shown as ``live=Y`` rather than mistaken for finished.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import fleet_session_signals as sig  # noqa: E402  (shared failure taxonomy)
import resume_sweep as RS            # noqa: E402  (reuse _text/_role/_load/_last_ts/_clip/HOME)

LEDGER = os.path.join(HERE, "_registry", "resume_ledger.jsonl")
BANNER_MARK = "hit your session limit"
VERDICTS = ("RELAUNCHED_OK", "STRANDED", "NEVER_WORKED", "NO_TRANSCRIPT")


def _ts(rec: dict) -> str:
    return rec.get("timestamp") or ""


def _is_err(rec: dict) -> bool:
    return bool(rec.get("isApiErrorMessage") or rec.get("type") == "error")


def relaunch_verdict(recs: list) -> dict:
    """Pure core: did this transcript advance PAST its last error? (see module docstring).

    Keyed off record ORDER + timestamps: the last error/banner record vs the last real
    (non-error, non-banner) assistant turn. Returns {verdict, kind, last_real_ts,
    last_err_ts, evidence}.
    """
    last_err = None
    last_err_ts = ""
    last_real_ts = ""
    for r in recs:
        tx = RS._text(r)
        t = _ts(r)
        if _is_err(r) or BANNER_MARK in (tx or "").lower():
            last_err, last_err_ts = tx, t
        elif RS._role(r) == "assistant" and tx.strip():
            last_real_ts = t
    if not last_real_ts:
        return {"verdict": "NEVER_WORKED", "kind": "", "last_real_ts": "",
                "last_err_ts": last_err_ts, "evidence": RS._clip(last_err or "")}
    if not last_err_ts or last_real_ts > last_err_ts:
        return {"verdict": "RELAUNCHED_OK", "kind": "", "last_real_ts": last_real_ts,
                "last_err_ts": last_err_ts, "evidence": ""}
    kind, _ = sig.terminal_failure(last_err or "")
    return {"verdict": "STRANDED", "kind": kind or "OTHER", "last_real_ts": last_real_ts,
            "last_err_ts": last_err_ts, "evidence": RS._clip(last_err or "")}


def ledger_actions(path: str) -> dict:
    """sid -> sorted list of the distinct ledger actions recorded for it."""
    acts: dict = {}
    if not os.path.isfile(path):
        return acts
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = r.get("session")
            if sid:
                acts.setdefault(sid, set()).add(r.get("action", "?"))
    return {sid: sorted(a) for sid, a in acts.items()}


def _superset(sid: str, home: str):
    """The richest on-disk copy: latest last_ts, then most records (mirrors resume_sweep)."""
    best = best_recs = None
    best_key = None
    for p in glob.glob(os.path.join(home, ".claude*", "projects", "*", sid + ".jsonl")):
        recs = RS._load(p)
        key = (RS._last_ts(recs), len(recs))
        if best_key is None or key > best_key:
            best_key, best, best_recs = key, p, recs
    return best, best_recs


def live_resume_sids() -> set:
    """8-char prefixes of sids that a `claude --resume <sid>` process is running RIGHT NOW."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='claude.exe'\" | "
             "ForEach-Object { $_.CommandLine }"],
            capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return set()
    return {m.group(1)[:8] for m in re.finditer(r"--resume\s+([0-9a-f-]{36})", out or "")}


def audit(home: str = None, ledger: str = LEDGER, live: bool = False) -> list:
    home = home or RS.HOME
    live_sids = live_resume_sids() if live else set()
    rows = []
    for sid, acts in ledger_actions(ledger).items():
        path, recs = _superset(sid, home)
        if not path:
            rows.append({"sid": sid, "actions": acts, "account": "", "n": 0,
                         "verdict": "NO_TRANSCRIPT", "kind": "", "evidence": "",
                         "last_real_ts": "", "last_err_ts": "", "live": False})
            continue
        v = relaunch_verdict(recs)
        acct = os.path.basename(path.split(os.sep + "projects" + os.sep)[0])
        rows.append({"sid": sid, "actions": acts, "account": acct, "n": len(recs),
                     "live": sid[:8] in live_sids, "superset_path": path, **v})
    order = {"STRANDED": 0, "NEVER_WORKED": 1, "NO_TRANSCRIPT": 2, "RELAUNCHED_OK": 3}
    rows.sort(key=lambda r: (order.get(r["verdict"], 9), r["account"], -r["n"]))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="resume_relaunch_audit", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="emit the machine record")
    ap.add_argument("--not-ok", action="store_true",
                    help="show only sessions that did NOT properly relaunch")
    ap.add_argument("--live", action="store_true",
                    help="cross-reference currently-running `claude --resume` workers")
    ap.add_argument("--ledger", default=LEDGER)
    args = ap.parse_args(argv)

    rows = audit(ledger=args.ledger, live=args.live)
    if args.not_ok:
        rows = [r for r in rows if r["verdict"] != "RELAUNCHED_OK"]

    if args.json:
        print(json.dumps({"count": len(rows), "rows": rows}, indent=1))
        return 1 if any(r["verdict"] != "RELAUNCHED_OK" for r in rows) else 0

    from collections import Counter
    def tag(r):
        return r["verdict"] + (":" + r["kind"] if r["kind"] else "")
    counts = Counter(tag(r) for r in audit(ledger=args.ledger, live=args.live))
    print("resume_relaunch_audit  ledger=" + os.path.basename(args.ledger)
          + "  sessions=" + str(sum(counts.values())))
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    not_ok = [r for r in rows if r["verdict"] != "RELAUNCHED_OK"]
    if not_ok:
        print("\n  NOT properly relaunched:")
        for r in not_ok:
            live = " live=Y" if r.get("live") else ""
            print(f"    {r['sid'][:8]} {r['account']:<26} n={r['n']:<4} "
                  f"{tag(r):<16}{live}  [{','.join(r['actions'])[:42]}]")
            if r["evidence"]:
                print(f"        evidence: {r['evidence']}")
    else:
        print("\n  all ledger sessions advanced past their error (RELAUNCHED_OK).")
    return 1 if not_ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
