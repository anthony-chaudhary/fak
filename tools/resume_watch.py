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

Sessions classified STALLED or DEAD are written to
tools/_registry/resume_watch_attention.json -- the operator's "look at these"
list. WORKING/DONE need no action.

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
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
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
    transient = bool(err and ("overloaded" in trailing_text(err).lower()
                              or "529" in trailing_text(err)
                              or "rate" in trailing_text(err).lower()
                              or "api error" in trailing_text(err).lower()))
    asks = bool(asst) and any(k in low for k in ASK) and low.rstrip().endswith("?")
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

    # RETRY/STALLED/DEAD need an action; GPU_GATED is surfaced for the operator
    # but is not a failure (the residual is hardware-gated, not stuck).
    ACT = ("RETRY", "STALLED", "DEAD", "UNKNOWN")
    attention = [r for r in rows if r["status"] in ACT]
    gpu = [r for r in rows if r["status"] == "GPU_GATED"]
    json.dump({"ts": now_iso(), "attention": attention, "gpu_gated": gpu},
              open(ATTENTION, "w", encoding="utf-8"), indent=1)

    if "--json" in sys.argv:
        print(json.dumps({"ts": now_iso(), "rows": rows}, indent=1))
        return 0

    from collections import Counter
    c = Counter(r["status"] for r in rows)
    print(f"resume_watch {now_iso()}  " + "  ".join(f"{k}={v}" for k, v in sorted(c.items())))
    for r in sorted(rows, key=lambda x: x["status"]):
        flag = "  <-- ACTION" if r["status"] in ACT else ("  (gpu)" if r["status"] == "GPU_GATED" else "")
        print(f"  {r['status']:9} {r['sid'][:8]} pid={r['pid']:<6} {r['disp']:14} "
              f"->{r['target']:14} quiet={r['quiet_min']}m{flag}")
        if r["status"] in ("STALLED", "RETRY", "GPU_GATED") and r["tail"]:
            print(f"           tail: {r['tail'][:170]}")
    if attention:
        print(f"\n{len(attention)} session(s) need an action (RETRY=re-resume, STALLED=answer) -> {ATTENTION}")
    if gpu:
        print(f"{len(gpu)} session(s) finished with a GPU-gated residual (operator/hardware) -> see attention file")
    if not attention and not gpu:
        print("\nall watched sessions healthy (WORKING/DONE); no operator action needed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
