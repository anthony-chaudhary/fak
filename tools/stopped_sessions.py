#!/usr/bin/env python3
"""Find recently-stopped top-level Claude Code sessions across all accounts and
decide which are safe to resume headlessly.

Authoritative signals (learned from the on-disk transcript format, v2.1.x):
  * throttle  -> a `<synthetic>` assistant message "... limit . resets <time>"
                 means the OWNING ACCOUNT is rate-limited until <time>.
  * mid-tool  -> last MEANINGFUL record is an assistant tool_use with no
                 following tool_result  => the process died mid-work.
  * interrupt -> last meaningful text contains "Login interrupted" /
                 "[Request interrupted by user".
  * waiting   -> last assistant text says it is awaiting a background
                 task/workflow notification (don't resume; it's parked).
  * done      -> last assistant text reads as a wrap-up.

Liveness is mtime-based (a live agent appends within LIVE_MIN minutes).

Usage:  python stopped_sessions.py [window_hours=10]
Output: JSON {accounts:{throttle...}, rows:[...], decisions:{resume,defer,skip}}
"""
import os
import sys
import json
import glob
import re
import datetime as dt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # import sibling helper
import fleet_accounts  # noqa: E402  -- account-policy layer (worker/excluded/non-account)
import fleet_session_signals  # noqa: E402

USER = os.environ.get("FLEET_USER_HOME", os.path.expanduser("~"))
NOW = dt.datetime.now(dt.timezone.utc)
WINDOW_H = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
LIVE_MIN = 4.0
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
# trailing non-conversational record types to skip when finding the last real turn
META_TYPES = {"mode", "permission-mode", "ai-title", "last-prompt", "summary",
              "queue-operation", "file-history-snapshot", "system"}

def read_lines(path, tail_bytes=512 * 1024):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - tail_bytes))
            return f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return []

def text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for b in content:
            if isinstance(b, dict):
                if b.get("type") == "text":
                    out.append(b.get("text", ""))
                elif b.get("type") == "tool_result":
                    c = b.get("content")
                    out.append(c if isinstance(c, str) else text_of(c))
        return " ".join(x for x in out if x)
    return ""

def last_tooluse_name(content):
    if isinstance(content, list):
        for b in reversed(content):
            if isinstance(b, dict) and b.get("type") == "tool_use":
                return b.get("name")
    return None

def parse(path):
    lines = read_lines(path)
    objs = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            objs.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return objs

def classify(path):
    st = os.stat(path)
    mtime = dt.datetime.fromtimestamp(st.st_mtime, dt.timezone.utc)
    age_min = (NOW - mtime).total_seconds() / 60.0
    objs = parse(path)
    cwd = git = ver = sid = None
    throttle_seen = None
    saw_tool_use = None          # name of an unmatched tool_use awaiting a result
    last_meaning = None          # last user/assistant record
    for o in objs:
        cwd = o.get("cwd", cwd)
        git = o.get("gitBranch", git)
        ver = o.get("version", ver)
        sid = o.get("sessionId", sid)
        t = o.get("type")
        if t not in ("user", "assistant"):
            continue
        msg = o.get("message") or {}
        content = msg.get("content")
        txt = text_of(content)
        # synthetic limit banner
        if msg.get("model") == "<synthetic>":
            reset = fleet_session_signals.limit_reset(txt)
            if reset:
                throttle_seen = reset
        last_meaning = {"role": msg.get("role", t), "txt": txt,
                        "synthetic": msg.get("model") == "<synthetic>",
                        "ts": o.get("timestamp")}
        # track unmatched tool_use -> tool_result pairing
        if t == "assistant":
            n = last_tooluse_name(content)
            if n:
                saw_tool_use = n
        elif t == "user":
            # a user turn carrying tool_result clears the pending tool_use
            if isinstance(content, list) and any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
                saw_tool_use = None

    lt = (last_meaning or {}).get("txt", "") or ""
    lt1 = lt[:300].replace("\n", " ")
    throttle_current = bool(throttle_seen and last_meaning and last_meaning.get("synthetic"))
    # disposition
    if throttle_current:
        disp = "STOPPED_LIMIT"
    elif fleet_session_signals.is_auth_error(lt):
        disp = "STOPPED_AUTH"
    elif age_min <= LIVE_MIN:
        disp = "LIVE"
    elif re.search(r"Login interrupted|\[Request interrupted by user", lt):
        disp = "STOPPED_INTERRUPT"
    elif saw_tool_use:
        disp = "STOPPED_MIDTOOL"
    elif re.search(r"still running|awaiting|wait for|will notify me|harness will|"
                   r"notify me when it completes|background", lt, re.I):
        disp = "PARKED_WAIT"
    elif re.search(r"^\s*(Done|Shipped|Complete|Summary|All set|✅)\b|delivered\b|"
                   r"committed and pushed|pushed .* to origin", lt, re.I):
        disp = "DONE"
    else:
        disp = "STOPPED_QUIET"
    return {
        "disp": disp, "age_min": round(age_min, 1), "size_kb": round(st.st_size/1024),
        "seen_utc": mtime.isoformat(),
        "session": sid or os.path.splitext(os.path.basename(path))[0],
        "cwd": cwd, "git": git, "version": ver,
        "throttle_reset": throttle_seen if throttle_current else None,
        "throttle_seen": throttle_seen,
        "throttle_current": throttle_current,
        "pending_tool": saw_tool_use,
        "last_role": (last_meaning or {}).get("role"), "last": lt1, "path": path,
    }

def main():
    rows = []
    acct_throttle = {}
    policy = fleet_accounts.load_policy()
    for acct_dir in glob.glob(os.path.join(USER, ".claude*")):
        acct = os.path.basename(acct_dir)
        proj = os.path.join(acct_dir, "projects")
        if not os.path.isdir(proj):
            continue
        # account policy: skip tombstoned/excluded accounts (e.g. the backup account)
        if not fleet_accounts.is_worker(acct, USER, policy):
            continue
        for path in glob.glob(os.path.join(proj, "*", "*.jsonl")):
            base = os.path.splitext(os.path.basename(path))[0]
            if not UUID_RE.match(base):
                continue
            if os.path.basename(os.path.dirname(path)).startswith("wf_"):
                continue
            try:
                st = os.stat(path)
            except OSError:
                continue
            mt = dt.datetime.fromtimestamp(st.st_mtime, dt.timezone.utc)
            if (NOW - mt).total_seconds()/60.0 > WINDOW_H*60:
                continue
            r = classify(path)
            r["account"] = acct
            r["project"] = os.path.basename(os.path.dirname(path))
            rows.append(r)
            if r["throttle_reset"] and r["disp"] == "STOPPED_LIMIT" and fleet_accounts.throttle_is_active(r["throttle_reset"]):
                # most-recent (smallest age) throttle wins
                cur = acct_throttle.get(acct)
                if not cur or r["age_min"] < cur["age_min"]:
                    acct_throttle[acct] = {"reset": r["throttle_reset"], "age_min": r["age_min"]}
    rows.sort(key=lambda r: r["age_min"])

    resume, defer, skip = [], [], []
    for r in rows:
        thr = acct_throttle.get(r["account"])
        if r["disp"] in ("STOPPED_MIDTOOL", "STOPPED_INTERRUPT", "STOPPED_QUIET"):
            if thr:
                defer.append({**r, "blocked_by": f"account throttled, resets {thr['reset']}"})
            else:
                resume.append(r)
        elif r["disp"] == "STOPPED_LIMIT":
            defer.append({**r, "blocked_by": f"session limit, resets {r['throttle_reset']}"})
        elif r["disp"] == "STOPPED_AUTH":
            defer.append({**r, "blocked_by": "account auth/subscription disabled"})
        else:
            skip.append(r)  # LIVE / PARKED_WAIT / DONE

    counts = {}
    for r in rows:
        counts[r["disp"]] = counts.get(r["disp"], 0) + 1
    print(json.dumps({
        "now_utc": NOW.isoformat(), "window_h": WINDOW_H,
        "account_throttle": acct_throttle, "counts": counts,
        "n_resume": len(resume), "n_defer": len(defer), "n_skip": len(skip),
        "resume": resume, "defer": defer, "rows": rows,
    }, indent=1))

if __name__ == "__main__":
    main()
