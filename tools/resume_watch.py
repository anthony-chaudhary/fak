#!/usr/bin/env python3
"""resume_watch.py -- periodic health check for the sessions recovered by the
resume pass (manifest: tools/_registry/resume_watch_manifest.json).

For each watched session it reports one of:
  WORKING   PID alive AND the session's newest transcript grew since last tick.
  IDLE      PID alive but transcript quiet for >IDLE_MIN minutes (maybe stalled).
  STALLED   last transcript record is an assistant turn that asks a question /
            requests input, with no following user turn -> needs an answer.
  DONE      PID gone AND last transcript record is a clean assistant completion.
  DEAD      PID gone AND last record is an error / mid-tool / abrupt cut.
  AUTH_FAIL the resume's terminal turn is a login/credit/access wall ("Not logged
            in / Please run /login", OAuth expired, 401, credit too low, ...). A
            re-resume on the SAME account can't fix it: the account needs `claude
            /login` AND a re-home to a healthy account. This is the silent-strand
            the resume-once ledger would otherwise hide, so it is surfaced loudest.

Sessions classified AUTH_FAIL / STALLED / RETRY / DEAD are written to
tools/_registry/resume_watch_attention.json -- the operator's "look at these"
list (AUTH_FAIL also gets its own key there). WORKING/DONE need no action.

The watcher is READ-ONLY over the sessions: it never resumes or re-prompts (that
would need an account+ledger decision). It exists to keep an eye on the pass and
surface the few that need a human, exactly as the operator asked.

Usage:
  python tools/resume_watch.py            # human summary + write attention list
  python tools/resume_watch.py --json     # machine record
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
from dispatch_worker import install_no_window_subprocess_defaults
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import fleet_session_signals  # noqa: E402  -- shared auth/limit/api signal patterns
install_no_window_subprocess_defaults(subprocess)

REG = os.path.join(HERE, "_registry")
MANIFEST = os.path.join(REG, "resume_watch_manifest.json")
ATTENTION = os.path.join(REG, "resume_watch_attention.json")
STATE = os.path.join(REG, ".resume_watch_state.json")   # last-seen mtime/size per sid
HOME = os.path.expanduser("~")
IDLE_MIN = float(os.environ.get("RESUME_IDLE_MIN", "20"))

# question/stall signatures in a trailing assistant turn
ASK = ("?", "could you", "should i", "which ", "do you want", "please confirm",
       "let me know", "waiting for", "need you to", "clarify", "your call",
       "would you like", "permission")


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def pid_alive(pid: int) -> bool:
    if not pid:
        return False
    try:
        out = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command",
             f"if (Get-Process -Id {int(pid)} -ErrorAction SilentlyContinue) {{'1'}} else {{'0'}}"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
        return out.endswith("1")
    except Exception:
        return False


def newest_transcript(sid: str) -> str | None:
    """The most-recently-modified transcript for this session across ALL account
    dirs (the resume wrote a fresh one under the target account's config dir)."""
    pats = glob.glob(os.path.join(HOME, ".claude*", "projects", "*", sid + ".jsonl"))
    pats = [p for p in pats if os.path.isfile(p)]
    if not pats:
        return None
    return max(pats, key=lambda p: os.path.getmtime(p))


def _role(rec: dict) -> str | None:
    m = rec.get("message") if isinstance(rec, dict) else None
    if isinstance(m, dict):
        return m.get("role")
    return rec.get("role") if isinstance(rec, dict) else None


def scan_tail(path: str) -> dict:
    """Read the transcript once and extract what classification needs: the last
    real assistant text turn, whether a trailing API/error record is present, and
    the total record count. A headless `claude --resume -p` writes ONE turn and
    its process exits -- so PID-gone is normal; the transcript is ground truth."""
    last_asst, last_err, n = None, None, 0
    try:
        with open(path, encoding="utf-8") as fh:
            for ln in fh:
                if not ln.strip():
                    continue
                try:
                    rec = json.loads(ln)
                except ValueError:
                    continue
                n += 1
                if _role(rec) == "assistant" and trailing_text(rec).strip():
                    last_asst = rec
                    last_err = None  # a later good turn supersedes an earlier error
                if rec.get("type") == "error" or rec.get("isApiErrorMessage"):
                    last_err = rec
                txt = trailing_text(rec)
                if "API Error" in txt or "Overloaded" in txt:
                    last_err = rec
    except Exception:
        pass
    return {"last_asst": last_asst, "last_err": last_err, "records": n}


def trailing_text(rec: dict) -> str:
    """Best-effort extract of an assistant turn's text for stall detection."""
    if not rec:
        return ""
    msg = rec.get("message") or rec
    content = msg.get("content") if isinstance(msg, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                out.append(b.get("text", ""))
        return "\n".join(out)
    return ""


def auth_failure(text: str) -> tuple[bool, str]:
    """Did this terminal turn fail on a login/credit/access wall the resume can't
    self-heal? This is the failure mode a re-resume on the SAME account can never
    fix (the fleet hit it 2026-06-23: gem7-netra resumes wrote 'Not logged in /
    Please run /login' while the account's interactive sessions kept working off a
    cached token). Reuses the shared fleet signals so detection matches the
    planner exactly. Returns (is_auth_fail, human_reason) -- reason names the
    specific remediation (auth/login required vs credit vs subscription access)."""
    text = text or ""
    if fleet_session_signals.is_auth_error(text) or fleet_session_signals.needs_login_prompt(text):
        return True, fleet_session_signals.auth_block_reason(text)
    return False, ""


def terminal_auth_failure(scan: dict) -> tuple[bool, str]:
    """AUTH_FAIL iff the transcript's TERMINAL error record is a login/credit/access
    wall. Reads the ERROR channel only (an injected ``isApiErrorMessage`` / error
    record), NOT plain assistant prose -- a session that merely *discusses* '/login'
    or auth in its final turn (e.g. a worker editing the resume tooling itself) is
    DONE, not auth-failed. This mirrors how fleet_sessions keys auth off the actual
    failure record, and is the guard that stops the AUTH_FAIL signal crying wolf."""
    err = scan.get("last_err")
    if not err:
        return False, ""
    return auth_failure(trailing_text(err))


GPU_GATE = ("on a cuda node", "run_48", "_on_gpu", "gpu node",
            "hardware-gated", "cuda node", "gated by #47", "on a gpu")
# explicit "I'm finished, nothing outstanding" -- overrides a GPU_GATE phrase that
# appears only as backstory (e.g. "benchmarked on a GPU" in a session that then
# says "green and shipped, no further action").
SHIPPED = ("no further action", "green and shipped", "nothing left to commit",
           "nothing to do", "fully shipped", "it's green and shipped",
           "no action needed", "nothing outstanding")


def classify(sid: str, pid: int, prev: dict) -> dict:
    """Status of one resumed session. A headless `claude --resume -p` turn writes
    its work then EXITS -- so PID liveness is NOT the signal. The transcript's
    terminal assistant turn is. Order matters: a transient API error outranks a
    clean finish (it's the real interruption); a question outranks a plain finish
    (it needs an answer); a hardware-gated residual is DONE-but-flagged (operator,
    not me, owns the GPU)."""
    alive = pid_alive(pid)
    tpath = newest_transcript(sid)
    size = os.path.getsize(tpath) if tpath else 0
    mtime = os.path.getmtime(tpath) if tpath else 0
    grew = size > int(prev.get("size", 0)) or mtime > float(prev.get("mtime", 0))
    quiet_min = (time.time() - mtime) / 60 if mtime else 9999

    if not tpath:
        return {"sid": sid, "pid": pid, "alive": alive, "status": "DEAD",
                "quiet_min": 9999, "transcript": None, "size": 0, "mtime": 0,
                "tail": "no transcript found"}

    t = scan_tail(tpath)
    asst = t["last_asst"]
    text = trailing_text(asst) if asst else ""
    low = text.lower()
    err = t["last_err"]
    err_text = trailing_text(err) if err else ""
    transient = bool(err and ("overloaded" in err_text.lower()
                              or "529" in err_text
                              or "rate" in err_text.lower()
                              or "api error" in err_text.lower()))
    # A usage-limit wall ("You've hit your session limit . resets 6am") is NOT a
    # transient 529 and NOT an auth wall: the account is simply capped until the
    # printed reset, then the session is resumable again. Detect it off the ERROR
    # record (where the limit banner lands as an isApiErrorMessage turn) and carry
    # the machine-readable reset window so it can be DEFERRED-until-reset rather than
    # buried in the terminal DEAD bucket. Distinct from `transient` because the
    # remediation is "wait for the named reset", not "re-resume now".
    limit_reset = fleet_session_signals.limit_reset(err_text) if err else None
    asks = bool(asst) and any(k in low for k in ASK) and low.rstrip().endswith("?")
    # Auth/login/credit/access wall: the resume did NOT take and a re-resume on the
    # SAME account can never fix it (it needs `claude /login` + a re-home to a healthy
    # account). Read off the ERROR channel only (not plain prose) so a "Not logged in /
    # Please run /login" failure record is surfaced distinctly instead of being buried
    # in the generic DEAD/RETRY bucket -- without flagging a session that merely
    # discusses auth in its final message.
    auth_is_fail, auth_reason = terminal_auth_failure(t)
    # a GPU residual only counts when it's the live outcome, not when an explicit
    # "shipped, no further action" verdict closes the turn (then the GPU phrase is
    # just backstory).
    shipped = any(k in low for k in SHIPPED)
    # A GPU residual must be an OUTSTANDING action this session owns -- a run_48* script
    # to execute, an explicit "residual ... on a cuda node", or the #47/#474 numeric gate.
    # A weak token like "hardware-gated" / "on a gpu" alone is NOT enough: a session that
    # *reports* it "correctly left GPU work untouched as hardware-gated" is DONE, not gated.
    # GPU-specific: a CUDA/GPU acceptance script or an explicit GPU/CUDA-node residual.
    # Deliberately NOT triggered by an arm64/m3 residual (run_*_acceptance_on_arm64.sh) --
    # that's a different hardware gate (Apple-silicon node), not the DGX/GCP GPU path.
    gpu_action = ("_on_gpu" in low or "gated by #47" in low
                  or "run_485" in low or "run_486" in low or "run_484" in low
                  or "run_483" in low or "run_482" in low or "run_479" in low
                  or ("residual" in low and ("cuda node" in low or "gpu node" in low
                                             or "on a gpu" in low)))
    gpu_deferred_report = ("left untouched" in low or "correctly left" in low
                           or "left alone" in low or "not run here" in low
                           or "left for" in low)
    gpu_gated = gpu_action and not shipped and not gpu_deferred_report

    if alive and grew:
        status = "WORKING"                 # still actively producing
    elif auth_is_fail:
        status = "AUTH_FAIL"               # login/credit/access wall -> needs /login + re-home
    elif limit_reset:
        status = "LIMIT"                   # usage cap -> resumable after the named reset, not dead
    elif transient:
        status = "RETRY"                   # hit a transient API error -> safe to re-resume
    elif err:
        status = "DEAD"                    # hard error terminal
    elif asks:
        status = "STALLED"                 # ended on a question -> needs an answer
    elif gpu_gated:
        status = "GPU_GATED"               # finished its share; residual needs a GPU node
    elif asst:
        status = "DONE"                    # clean completion
    elif alive:
        status = "WORKING"
    else:
        status = "UNKNOWN"

    return {"sid": sid, "pid": pid, "alive": alive, "status": status,
            "quiet_min": round(quiet_min, 1), "transcript": tpath,
            "size": size, "mtime": mtime, "records": t["records"],
            "auth_reason": auth_reason, "limit_reset": limit_reset,
            "tail": text[-300:].replace("\n", " ").strip()}


def main() -> int:
    man = json.load(open(MANIFEST, encoding="utf-8"))
    prev = {}
    if os.path.exists(STATE):
        try:
            prev = json.load(open(STATE, encoding="utf-8"))
        except Exception:
            prev = {}

    rows, new_state = [], {}
    for s in man.get("sessions", []):
        sid = s["sid"]
        r = classify(sid, int(s.get("pid") or 0), prev.get(sid, {}))
        r["disp"] = s.get("disp")
        r["owner"] = s.get("owner")
        r["target"] = s.get("target")
        r["note"] = s.get("note", "")
        rows.append(r)
        new_state[sid] = {"size": r["size"], "mtime": r["mtime"]}

    json.dump(new_state, open(STATE, "w", encoding="utf-8"))

    # RETRY/STALLED/DEAD need an action; AUTH_FAIL is the loudest -- the resume hit a
    # login/credit/access wall, so a plain re-resume on the SAME account can't fix it
    # (it needs `claude /login` AND a re-home to a healthy account); GPU_GATED is
    # surfaced for the operator but is not a failure (residual is hardware-gated).
    ACT = ("AUTH_FAIL", "RETRY", "STALLED", "DEAD", "UNKNOWN")
    attention = [r for r in rows if r["status"] in ACT]
    gpu = [r for r in rows if r["status"] == "GPU_GATED"]
    auth_fail = [r for r in rows if r["status"] == "AUTH_FAIL"]
    # LIMIT is NOT a failure: the account is capped until the named reset, after which
    # the session is resumable. Surfaced in its own band (with the reset window) so the
    # operator/watchdog can defer-then-retry instead of treating it as a terminal DEAD.
    limited = [r for r in rows if r["status"] == "LIMIT"]
    json.dump({"ts": now_iso(), "attention": attention, "gpu_gated": gpu,
               "auth_fail": auth_fail, "limit_deferred": limited},
              open(ATTENTION, "w", encoding="utf-8"), indent=1)

    if "--json" in sys.argv:
        print(json.dumps({"ts": now_iso(), "rows": rows}, indent=1))
        return 0

    from collections import Counter
    c = Counter(r["status"] for r in rows)
    print(f"resume_watch {now_iso()}  " + "  ".join(f"{k}={v}" for k, v in sorted(c.items())))
    for r in sorted(rows, key=lambda x: x["status"]):
        if r["status"] == "AUTH_FAIL":
            flag = "  <-- AUTH FAIL: needs `claude /login` + re-home (NOT a re-resume)"
        elif r["status"] in ACT:
            flag = "  <-- ACTION"
        elif r["status"] == "GPU_GATED":
            flag = "  (gpu)"
        else:
            flag = ""
        print(f"  {r['status']:9} {r['sid'][:8]} pid={r['pid']:<6} {r['disp']:14} "
              f"->{r['target']:14} quiet={r['quiet_min']}m{flag}")
        if r["status"] in ("AUTH_FAIL", "STALLED", "RETRY", "GPU_GATED", "LIMIT") and r["tail"]:
            print(f"           tail: {r['tail'][:170]}")
    if limited:
        print(f"\n{len(limited)} session(s) DEFERRED on a usage limit -- resumable after the "
              f"named reset (NOT dead; do not re-home, just retry past the reset):")
        for r in limited:
            print(f"  {r['sid'][:8]} on {r.get('target')}: resets {r.get('limit_reset')}")
    if auth_fail:
        print(f"\n{len(auth_fail)} session(s) FAILED RESUME ON AUTH -- a re-resume on the same "
              f"account WON'T help; the account needs `claude /login`, then re-home to a healthy one:")
        for r in auth_fail:
            print(f"  {r['sid'][:8]} on {r.get('target')}: {r.get('auth_reason') or 'auth/login required'}")
    if attention:
        print(f"\n{len(attention)} session(s) need an action (AUTH_FAIL=login+re-home, "
              f"RETRY=re-resume, STALLED=answer) -> {ATTENTION}")
    if gpu:
        print(f"{len(gpu)} session(s) finished with a GPU-gated residual (operator/hardware) -> see attention file")
    if not attention and not gpu and not limited:
        print("\nall watched sessions healthy (WORKING/DONE); no operator action needed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
