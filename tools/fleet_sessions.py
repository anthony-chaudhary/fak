#!/usr/bin/env python3
r"""fleet_sessions — the cross-account "what stopped, why, and how to resume" index.

The problem this kills: with N Claude Code accounts under ``<home>/.claude*``,
when a headless worker stops you otherwise have to GUESS which account owns it,
whether it really stopped or is just idle, why it stopped, and whether its
account is rate-limited. Resuming under the wrong account fails with
"No conversation found with session ID". This tool answers all of that
deterministically from the on-disk transcripts — no guessing.

Signals (transcript format v2.1.x):
  throttle  : a `<synthetic>` assistant message ".. limit . resets <when>"
              => the OWNING ACCOUNT is rate-limited until <when>.
  mid-tool  : last meaningful record is an assistant tool_use with no
              following tool_result  => the process died mid-work.
  interrupt : last text has "Login interrupted" / "[Request interrupted by user".
  parked    : last assistant text says it is awaiting a background task.
  done      : last assistant text reads as a wrap-up.
  live      : transcript appended within LIVE_MIN minutes.

Modes:
  summary  (default)  compact operator table, grouped by disposition
  json                full machine payload
  resume              ready-to-run, account-correct resume commands for
                      genuinely-stopped sessions on NON-throttled accounts

Usage:  python fleet_sessions.py [summary|json|resume] [--window H] [--max-age MIN]
"""
import os
import sys
import json
import glob
import re
import datetime as dt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # import sibling helper
import fleet_version  # noqa: E402
import fleet_accounts  # noqa: E402  -- account-policy layer (worker/excluded/non-account)
import fleet_session_signals  # noqa: E402

USER = os.environ.get("FLEET_USER_HOME", os.path.expanduser("~"))
NOW = dt.datetime.now(dt.timezone.utc)
LIVE_MIN = 4.0
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
DONE_RE = re.compile(r"\b(complete|completed|shipped|pushed|committed|delivered|"
                     r"lease released|all checks|all set|terminated cleanly|"
                     r"goal is met|witness (?:holds|is met)|this completes)\b", re.I)
PARK_RE = re.compile(r"still running|awaiting|wait for the|will notify me|harness will|"
                     r"notify me when|waiting for the (?:exit|completion|result)|holding", re.I)
# autonomy: a session that WANTS to keep going on its own (vs. an interactive Q&A).
AUTON_RE = re.compile(r"<command-name>/(goal|loop|dispatch|next-up|fanout)\b|"
                      r"Stop hook is now active|/loop\b|ScheduleWakeup|"
                      r"autonomous-loop|keep working until", re.I)
# supervised: owned by the job-fleet supervisor -> leave it to run_supervise_loop, don't --resume.
SUPERVISED_RE = re.compile(r"JOB_SUPERVISED_WORKER|supervisor-spawned|/dispatch-loop\b", re.I)
GOALCLEAR_RE = re.compile(r"goal (?:condition )?(?:met|satisfied|cleared)|hook (?:auto-)?clear", re.I)
# disposition -> (category, cause). category buckets: INFRA / AGENT / USER / HANGING / LIVE.
CATEGORY = {
    "LIVE":          ("LIVE",    "live"),
    "DONE":          ("AGENT",   "completed"),
    "DEAD_MIDTOOL":  ("AGENT",   "crash_mid_tool"),
    "DEAD_KILLED":   ("AGENT",   "killed_mid_turn"),
    "USER_CLOSED":   ("USER",    "user_stopped"),
    "STOPPED_LIMIT": ("INFRA",   "rate_limit"),
    "STOPPED_APIERR":("INFRA",   "api_error"),
    "INFRA_AUTH":    ("INFRA",   "auth"),
    "PARKED_WAIT":   ("HANGING", "parked_on_task"),
    "STOPPED_QUIET": ("HANGING", "ambiguous_quiet"),
}

def args_get(flag, default):
    if flag in sys.argv:
        return sys.argv[sys.argv.index(flag) + 1]
    return default

MODE = next((a for a in sys.argv[1:] if not a.startswith("-")), "summary")
WINDOW_H = float(args_get("--window", "10"))
MAX_AGE = float(args_get("--max-age", "1e9"))
# Active probing: --probe[=blocked|stale|all|none]. Default none keeps the fast passive
# path untouched. A bare --probe means "blocked". Probe rows are appended to the scanned
# rows BEFORE the merge pipeline, so a fresh OK probe clears a stale carry-forward blocker
# and a fresh AUTH/LIMIT probe sets one -- without anyone running a real session.
def _probe_selector():
    for a in sys.argv:
        if a.startswith("--probe="):
            val = a.split("=", 1)[1].strip().lower()
            return val if val in ("blocked", "stale", "all", "none") else "blocked"
    if "--probe" in sys.argv:
        # bare "--probe" means blocked; "--probe <selector>" reads the next token only
        # when it names a known selector (otherwise the next token is some other arg).
        i = sys.argv.index("--probe")
        nxt = sys.argv[i + 1].strip().lower() if i + 1 < len(sys.argv) else ""
        return nxt if nxt in ("blocked", "stale", "all", "none") else "blocked"
    return "none"

PROBE_SELECTOR = _probe_selector()
# Anti-spam floor: skip an account probed within the last N minutes (0 = no floor).
PROBE_MIN_INTERVAL = float(args_get("--min-interval-min", "0"))

def read_tail(path, tail_bytes=512 * 1024):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - tail_bytes))
            return f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return []

def read_head(path, n=40):
    out = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for _ in range(n):
                ln = f.readline()
                if not ln:
                    break
                out.append(ln)
    except OSError:
        pass
    return out

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

def last_tooluse(content):
    if isinstance(content, list):
        for b in reversed(content):
            if isinstance(b, dict) and b.get("type") == "tool_use":
                return b.get("name")
    return None

def classify(path):
    st = os.stat(path)
    mtime = dt.datetime.fromtimestamp(st.st_mtime, dt.timezone.utc)
    age = (NOW - mtime).total_seconds() / 60.0
    cwd = git = sid = None
    throttle = None
    throttle_weekly = None  # weekly reset window, when the banner carries one
    pending = None          # an assistant tool_use still awaiting its tool_result
    last = None             # summary of the last meaningful user/assistant record
    last_kind = None        # nature of that final record (drives DONE vs DEAD vs USER_CLOSED)
    for ln in read_tail(path):
        ln = ln.strip()
        if not ln:
            continue
        try:
            o = json.loads(ln)
        except json.JSONDecodeError:
            continue
        cwd = o.get("cwd", cwd)
        git = o.get("gitBranch", git)
        sid = o.get("sessionId", sid)
        if o.get("type") not in ("user", "assistant"):
            continue
        m = o.get("message") or {}
        c = m.get("content")
        txt = text_of(c)
        if m.get("model") == "<synthetic>":
            windows = fleet_session_signals.limit_resets(txt)
            primary = windows.get("daily") or windows.get("weekly")
            if primary:
                throttle = primary
                throttle_weekly = windows.get("weekly")
        last = {"role": m.get("role", o.get("type")), "txt": txt,
                "syn": m.get("model") == "<synthetic>", "stop": m.get("stop_reason")}
        if o.get("type") == "assistant":
            n = last_tooluse(c)
            if n:
                pending = n
                last_kind = "assistant_tooluse"
            elif m.get("stop_reason") == "end_turn":
                last_kind = "assistant_end"      # model voluntarily ended its turn => finished
            else:
                last_kind = "assistant_text"
        else:  # user
            is_tr = isinstance(c, list) and any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in c)
            if is_tr:
                pending = None
            if re.search(r"\[Request interrupted by user", txt):
                last_kind = "user_interrupt"     # user pressed Esc/Ctrl-C
            elif re.search(r"Login interrupted", txt) or \
                    re.search(r"<command-name>/(quit|exit|clear|logout|login)\b", txt):
                last_kind = "user_close"          # user issued a close/login command
            elif is_tr:
                last_kind = "user_toolresult"     # a tool answer with no following assistant
            else:
                last_kind = "user_text"
    lt = (last or {}).get("txt", "") or ""
    # disposition. Infra failures (rate-limit/API/auth) are separated from agent
    # failures (crash vs finished), from user stops, from hanging sessions.
    #   DONE        = finished cleanly (model ended its turn)        -> never resume   [AGENT]
    #   USER_CLOSED = user intentionally interrupted/closed it       -> never resume   [USER]
    #   DEAD_*      = crashed/killed mid-work                        -> resume if auton [AGENT]
    #   STOPPED_LIMIT / STOPPED_APIERR / INFRA_AUTH                  -> infra, retry    [INFRA]
    #   PARKED_WAIT / STOPPED_QUIET                                  -> hanging/orphan  [HANGING]
    throttle_current = bool(throttle and last and last.get("syn"))
    if throttle_current:
        disp, reason = "STOPPED_LIMIT", "hit account session limit; resets %s" % throttle
    elif fleet_session_signals.is_auth_error(lt):
        kind = fleet_session_signals.auth_block_kind(lt)
        if kind == "access":
            reason = "stopped on a Claude subscription/access wall (not solved by /login)"
        elif kind == "credit":
            reason = "stopped on account credit/billing state"
        else:
            reason = "stopped on an auth/login requirement (needs re-login)"
        disp = "INFRA_AUTH"
    elif fleet_session_signals.is_api_error(lt) and last_kind != "assistant_end":
        disp, reason = "STOPPED_APIERR", "stopped on an API/transport error (transient infra)"
    elif age <= LIVE_MIN:
        disp, reason = "LIVE", "appended within %g min" % LIVE_MIN
    elif last_kind in ("user_interrupt", "user_close"):
        disp, reason = "USER_CLOSED", "ended on %s (user intentionally stopped it)" % last_kind
    elif pending:
        disp, reason = "DEAD_MIDTOOL", "died mid tool_use (%s) with no tool_result" % pending
    elif PARK_RE.search(lt):
        disp, reason = "PARKED_WAIT", "parked awaiting a background task"
    elif last_kind == "assistant_end" or (last and last.get("role") == "assistant" and DONE_RE.search(lt)):
        disp, reason = "DONE", "last assistant turn ended cleanly (stop_reason=end_turn / wrap-up)"
    elif last_kind == "user_toolresult":
        disp, reason = "DEAD_KILLED", "killed after a tool_result before the next assistant turn"
    elif last and last.get("role") == "assistant":
        disp, reason = "DONE", "ended on an assistant message"
    else:
        disp, reason = "STOPPED_QUIET", "quiet; no completion/crash/close signal"
    category, cause = CATEGORY.get(disp, ("HANGING", "unknown"))
    # autonomy / ownership — parse the HEAD RECORDS (the session's own directive),
    # not a content blob. A session is autonomous only if it was actually launched
    # with /goal|/loop|/dispatch|/fanout|/next-up, or a Stop-hook goal was installed,
    # or it carries the supervised-worker marker. This avoids flagging interactive
    # sessions that merely DISCUSS goals/loops.
    autonomous = supervised = False
    for hl in read_head(path, 30):
        hl = hl.strip()
        if not hl:
            continue
        try:
            ho = json.loads(hl)
        except json.JSONDecodeError:
            continue
        htxt = ""
        if ho.get("type") in ("user", "system"):
            mc = (ho.get("message") or {}).get("content", ho.get("content", ""))
            htxt = mc if isinstance(mc, str) else text_of(mc)
        if AUTON_RE.search(htxt):
            autonomous = True
        if SUPERVISED_RE.search(htxt):
            supervised = True
            autonomous = True
    autonomous = autonomous or supervised
    return {"disp": disp, "category": category, "cause": cause, "reason": reason,
            "last_kind": last_kind, "age_min": round(age, 1),
            "seen_utc": mtime.isoformat(),
            "throttle_reset": throttle if throttle_current else None,
            "throttle_weekly": throttle_weekly if throttle_current else None,
            "throttle_seen": throttle,
            "throttle_current": throttle_current,
            "pending_tool": pending,
            "session": sid or os.path.splitext(os.path.basename(path))[0],
            "cwd": cwd, "git": git, "last": lt[:200].replace("\n", " "), "path": path,
            "autonomous": autonomous, "supervised": supervised}

ACCT_POLICY = fleet_accounts.load_policy()

def scan():
    rows, throttle = [], {}
    for acct_dir in glob.glob(os.path.join(USER, ".claude*")):
        acct = os.path.basename(acct_dir)
        proj = os.path.join(acct_dir, "projects")
        if not os.path.isdir(proj):
            continue
        # account policy: skip tombstoned/excluded accounts (e.g. the backup
        # account) so they never produce rows, resume commands, or plan entries.
        if not fleet_accounts.is_worker(acct, USER, ACCT_POLICY):
            continue
        for path in glob.glob(os.path.join(proj, "*", "*.jsonl")):
            base = os.path.splitext(os.path.basename(path))[0]
            if not UUID_RE.match(base) or os.path.basename(os.path.dirname(path)).startswith("wf_"):
                continue
            try:
                st = os.stat(path)
            except OSError:
                continue
            age = (NOW - dt.datetime.fromtimestamp(st.st_mtime, dt.timezone.utc)).total_seconds() / 60.0
            if age > WINDOW_H * 60 or age > MAX_AGE:
                continue
            r = classify(path)
            r["account"] = acct
            r["project"] = os.path.basename(os.path.dirname(path))
            rows.append(r)
            if r["throttle_reset"] and r["disp"] == "STOPPED_LIMIT" and fleet_accounts.throttle_is_active(r["throttle_reset"]):
                cur = throttle.get(acct)
                if not cur or r["age_min"] < cur["age_min"]:
                    entry = {"reset": r["throttle_reset"], "age_min": r["age_min"]}
                    if r.get("throttle_weekly"):
                        entry["weekly"] = r["throttle_weekly"]
                    throttle[acct] = entry
    rows.sort(key=lambda r: r["age_min"])
    return rows, throttle

def merge_known_throttle(throttle, rows):
    """Carry forward cached account limits whose reset has not expired yet."""
    newest = {}
    for r in sorted(rows, key=lambda x: x["age_min"]):
        newest.setdefault(r["account"], r)
    cleared = {
        acct for acct, r in newest.items()
        if r.get("disp") == "LIVE" and not r.get("throttle_current")
    }
    prev = fleet_accounts.load_registry().get("throttle", {}) or {}
    merged = {}
    for source in (prev, throttle):
        for acct, info in source.items():
            if acct in cleared:
                continue
            if not fleet_accounts.throttle_is_active(info):
                continue
            merged[acct] = info if isinstance(info, dict) else {"reset": info}
    return merged

def _parse_utc(raw):
    if not raw:
        return None
    try:
        ts = dt.datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return ts.astimezone(dt.timezone.utc)

def _row_seen_utc(row):
    seen = _parse_utc(row.get("seen_utc"))
    if seen is not None:
        return seen
    age = fleet_accounts._age_min(row)
    if age is None:
        return None
    return NOW - dt.timedelta(minutes=age)

def _auth_info_seen_utc(info, generated_utc=None):
    if not isinstance(info, dict):
        return None
    seen = _parse_utc(info.get("seen_utc"))
    if seen is not None:
        return seen
    age = fleet_accounts._age_min(info)
    generated = _parse_utc(generated_utc)
    if age is None or generated is None:
        return None
    return generated - dt.timedelta(minutes=age)

def _auth_info_from_row(row):
    last = str(row.get("last") or row.get("reason") or "")
    seen = _row_seen_utc(row) or NOW
    return {
        "block_kind": fleet_session_signals.auth_block_kind(last),
        "block_reason": fleet_session_signals.auth_block_reason(last),
        "seen_utc": seen.isoformat(),
        "age_min": row.get("age_min"),
        "session": row.get("session"),
        "project": row.get("project"),
        "last": last[:200],
    }

def _normalize_auth_info(info):
    row = dict(info) if isinstance(info, dict) else {
        "block_kind": "auth",
        "block_reason": str(info) if info else "auth/login required",
    }
    reason_text = " ".join(
        str(row.get(k) or "") for k in ("last", "reason", "block_reason")
    )
    if reason_text.strip():
        row["block_kind"] = fleet_session_signals.auth_block_kind(reason_text)
        row["block_reason"] = fleet_session_signals.auth_block_reason(reason_text)
    else:
        row.setdefault("block_kind", "auth")
        row.setdefault("block_reason", "auth/login required")
    return row

def merge_known_auth(rows):
    """Carry forward account auth blockers until a newer successful turn clears them."""
    latest_success = {}
    current_auth = {}
    for r in rows:
        acct = r.get("account")
        seen = _row_seen_utc(r)
        if not acct or seen is None:
            continue
        if r.get("disp") in ("LIVE", "DONE"):
            if acct not in latest_success or seen > latest_success[acct]:
                latest_success[acct] = seen
        if r.get("disp") == "INFRA_AUTH":
            cur = current_auth.get(acct)
            if cur is None or seen > (_auth_info_seen_utc(cur) or dt.datetime.min.replace(tzinfo=dt.timezone.utc)):
                current_auth[acct] = _auth_info_from_row(r)

    prev_reg = fleet_accounts.load_registry()
    prev = prev_reg.get("auth", {}) or {}
    prev_generated = prev_reg.get("generated_utc")
    merged = {}
    for source in (prev, current_auth):
        for acct, info in source.items():
            row = _normalize_auth_info(info)
            seen = _auth_info_seen_utc(row, prev_generated)
            success_seen = latest_success.get(acct)
            if success_seen is not None and seen is not None and success_seen > seen:
                continue
            merged[acct] = row
    return merged

DEAD = {"DEAD_MIDTOOL", "DEAD_KILLED"}              # crashed/killed mid-work -> resumable
STOPLIKE = DEAD | {"STOPPED_QUIET", "STOPPED_LIMIT", "STOPPED_APIERR", "INFRA_AUTH",
                   "USER_CLOSED", "PARKED_WAIT"}
REG_DIR = os.environ.get("FLEET_REG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "_registry"))
RESUME_PROMPT = ("Resume where you left off; re-establish any /goal or /loop "
                 "and continue toward it.")

def config_dir(acct):
    return os.path.join(USER, acct)

def _verdict_freshness(account, rows, auth):
    """How fresh is THIS account's verdict, and where did it come from?

    Returns (source, age_min):
      "probe"   a synthetic probe row fed this refresh (ground truth, age~0)
      "passive" a real transcript row inside the window is the evidence
      "carried" no fresh row -- the verdict is a carry-forward from the auth/throttle
                map (the stale-latch case the observability exists to surface)
    The age is minutes since the newest evidence backing the verdict.
    """
    acct_rows = [r for r in rows if r.get("account") == account]
    probe_rows = [r for r in acct_rows if r.get("project") == "_probe"]
    if probe_rows:
        age = min((float(r.get("age_min") or 0.0) for r in probe_rows), default=0.0)
        return "probe", round(age, 1)
    session_rows = [r for r in acct_rows if r.get("project") != "_probe"]
    if session_rows:
        age = min((float(r.get("age_min") or 0.0) for r in session_rows
                   if r.get("age_min") is not None), default=None)
        if age is not None:
            return "passive", round(age, 1)
    # no fresh row -> the verdict rides a carried auth/throttle entry; age it from seen_utc
    info = (auth or {}).get(account)
    seen = fleet_accounts._parse_utc(info.get("seen_utc")) if isinstance(info, dict) else None
    if seen is not None:
        return "carried", round((NOW - seen).total_seconds() / 60.0, 1)
    return "carried", None


def account_availability(throttle, rows, auth=None):
    """Per worker account: is it safe for the switcher to offer right now?"""
    registry = {"generated_utc": NOW.isoformat(), "auth": auth or {}}
    annotated = fleet_accounts.annotated_roster(
        USER, ACCT_POLICY, registry=registry, throttle=throttle, sessions=rows)
    out = []
    for a in annotated:
        if a["kind"] != "worker":
            continue
        source, age = _verdict_freshness(a["account"], rows, auth)
        out.append({
            "account": a["account"], "tag": a["tag"],
            "config_dir": config_dir(a["account"]),
            "available": a["available"],
            "blocked": a["blocked"],
            "block_kind": a["block_kind"],
            "block_reason": a["block_reason"],
            "throttled": a["throttled"],
            "reset": a["reset"],
            "weekly": a.get("weekly"),
            "active_sessions": a["active_sessions"],
            "live_sessions": a["live_sessions"],
            "auth_blocked_sessions": a["auth_blocked_sessions"],
            # freshness: makes a stale carried-forward verdict visibly stale, the single
            # field whose absence let seven accounts silently latch as blocked.
            "verdict_source": source,
            "verdict_age_min": age,
        })
    out.sort(key=lambda a: (not a["available"], a["tag"]))
    return out

def resume_cmd(r):
    """Operator-runnable resume command, account-correct.

    For a re-homed session the command first copies the transcript out of the
    throttled owner's config dir into the healthy target account's config dir
    (claude --resume is CLAUDE_CONFIG_DIR + cwd scoped, so the conversation must
    physically live under the account it resumes on), then resumes there."""
    cfg = r.get("resume_config_dir") or config_dir(r["account"])
    prefix = ""
    if r.get("rehomed"):
        src = r.get("source_config_dir") or config_dir(r["account"])
        proj, sid = r.get("project", ""), r["session"]
        src_file = os.path.join(src, "projects", proj, sid + ".jsonl")
        dst_dir = os.path.join(cfg, "projects", proj)
        prefix = (f"New-Item -ItemType Directory -Force -Path '{dst_dir}' | Out-Null; "
                  f"Copy-Item '{src_file}' '{os.path.join(dst_dir, sid + '.jsonl')}' -Force; ")
    return (f"{prefix}$env:CLAUDE_CONFIG_DIR='{cfg}'; "
            f"claude --resume {r['session']} -p '{RESUME_PROMPT}' --dangerously-skip-permissions")

# How many sessions one account may be assigned IN A SINGLE re-home pass before
# it is considered "full" and dropped from the candidate pool. A re-home adds a
# fresh autonomous `claude --resume` to the target, and an account that is already
# running near its session ceiling will itself hit the usage limit the moment the
# burst lands -- which is exactly the 32->1 stampede that wedged every resume onto
# one account. The cap is the in-pass admission ceiling: assigned + already-live
# must stay under it. Override with FAK_REHOME_CAP for hosts with fatter accounts.
REHOME_CAP = int(os.environ.get("FAK_REHOME_CAP", "4"))


def _rehome_targets(availability, exclude_account, assigned=None):
    """Available Claude worker accounts a throttled session can move to, least
    loaded first. opencode accounts are excluded: a Claude transcript can only
    resume under another Claude config dir, not an opencode one.

    ``assigned`` is the per-account count this pass has ALREADY re-homed onto each
    target (account basename -> n). It is folded into the load so a burst of
    throttled sessions spreads across healthy accounts instead of all picking the
    same momentary least-loaded one: the snapshot's live/active counts are static
    within a pass, so without this every caller computes the identical winner and
    stampedes it. An account whose (already-live + just-assigned) load reaches
    ``REHOME_CAP`` drops out of the pool entirely -- better to DEFER_THROTTLED and
    wait for a reset than to pile a session onto an account that will limit-wall."""
    assigned = assigned or {}
    cands = []
    for a in (availability or []):
        acct = a.get("account", "")
        if (not a.get("available")
                or acct == exclude_account
                or not str(acct).startswith(".claude")):
            continue
        base_load = int(a.get("live_sessions") or 0)
        if base_load + assigned.get(acct, 0) >= REHOME_CAP:
            continue                       # already at this pass's admission ceiling
        cands.append(a)
    # Rank by load (live + in-pass assigned) first, but break ties by PROVEN health:
    # an account with a fresh positive verdict sorts ahead of one whose `available`
    # is merely the absence-of-evidence default. account_availability stamps
    # verdict_source as one of probe (a live probe just hit it), passive (a real
    # session row inside the window proves it alive), or carried (a stale verdict
    # carried forward with no fresh evidence). probe/passive are genuine positive
    # evidence; carried/none are not -- so probe/passive sort ahead of carried/none.
    # A target offered purely because nothing bad was recorded is the weakest kind of
    # healthy, so it goes last among equals -- without ever being excluded when it is
    # the only option (the load + cap gate already decides inclusion).
    def _unproven(a):
        return 0 if str(a.get("verdict_source") or "none") in ("probe", "passive") else 1
    cands.sort(key=lambda a: (int(a.get("live_sessions") or 0) + assigned.get(a.get("account", ""), 0),
                              _unproven(a),
                              int(a.get("active_sessions") or 0),
                              str(a.get("tag") or a.get("account") or "")))
    return cands

def decide(rows, throttle, availability=None):
    """Stamp each row with a deterministic action + an account-correct resume command.
    Only AUTONOMOUS, genuinely-DEAD (crashed/killed) sessions are auto-resumable.
    The two look-alikes are held back explicitly so they are never resumed and the
    reason is logged:
      DONE        -> the agent finished; resuming would redo finished work.
      USER_CLOSED -> the user intentionally interrupted/closed it; honor that.

    Rate-limit handling: a resumable autonomous session whose OWNING account is
    throttled is RE-HOMED onto a healthy account (AUTO_RESUME + rehomed=True with
    a resume_config_dir pointing at the target) instead of being parked until the
    owner's limit resets -- which for a weekly cap can be days. Re-home only fires
    when a healthy Claude worker account actually exists; otherwise the session
    falls back to DEFER_THROTTLED and waits, exactly as before.

    Re-home spread: the availability snapshot is static within one pass, so a burst
    of throttled sessions would all pick the same momentary least-loaded target and
    stampede it. ``assigned`` tracks how many this pass has already routed to each
    target and is fed back into ``_rehome_targets`` so the load it sees reflects the
    in-flight decisions -- spreading the burst across healthy accounts and capping
    each at REHOME_CAP so none is pushed over its own session limit."""
    assigned: dict[str, int] = {}
    for r in rows:
        cwd_ok = bool(r["cwd"]) and os.path.isdir(r["cwd"])
        # resume target defaults to the owning account; re-home overrides it below
        r["source_config_dir"] = config_dir(r["account"])
        r["resume_account"] = r["account"]
        r["resume_config_dir"] = config_dir(r["account"])
        r["rehomed"] = False
        if "pytest" in r["project"] or not cwd_ok:
            r["action"] = "SKIP_EPHEMERAL"          # cwd gone / pytest temp
        elif r["disp"] == "LIVE":
            r["action"] = "SKIP_LIVE"
        elif r["disp"] == "DONE":
            r["action"] = "SKIP_DONE"               # finished cleanly — do NOT resume
        elif r["disp"] == "USER_CLOSED":
            r["action"] = "SKIP_USER_CLOSED"        # user stopped it on purpose — honor it
        elif r["disp"] == "PARKED_WAIT":
            r["action"] = "SKIP_PARKED"
        elif r["supervised"]:
            r["action"] = "SUPERVISED"              # run_supervise_loop owns it
        elif r["disp"] == "STOPPED_LIMIT" or r["account"] in throttle:
            # Owning account is rate-limited. Re-home an autonomous, resumable
            # session to a healthy account rather than waiting for the reset.
            resumable = r["autonomous"] and (
                r["disp"] in DEAD or r["disp"] in ("STOPPED_LIMIT", "STOPPED_APIERR"))
            targets = _rehome_targets(availability, r["account"], assigned) if resumable else []
            if targets:
                tgt = targets[0]
                r["action"] = "AUTO_RESUME"         # INFRA: rate limit -> move to healthy acct
                r["rehomed"] = True
                r["resume_account"] = tgt["account"]
                r["resume_config_dir"] = tgt.get("config_dir") or config_dir(tgt["account"])
                assigned[tgt["account"]] = assigned.get(tgt["account"], 0) + 1
            else:
                r["action"] = "DEFER_THROTTLED"     # no healthy account -> wait for reset
        elif r["disp"] == "INFRA_AUTH":
            r["action"] = "BLOCKED_AUTH"            # INFRA: needs human re-login; resume won't help
        elif r["disp"] == "STOPPED_APIERR" and r["autonomous"]:
            r["action"] = "AUTO_RESUME"             # INFRA: transient API error -> retry
        elif r["disp"] in DEAD and r["autonomous"]:
            r["action"] = "AUTO_RESUME"             # AGENT crash, autonomous -> resume
        elif r["disp"] in DEAD or r["disp"] in ("STOPPED_QUIET", "STOPPED_APIERR"):
            r["action"] = "SURFACE"                 # dead/transient but interactive -> human
        else:
            r["action"] = "SKIP"
        r["resume_cmd"] = resume_cmd(r) if r["disp"] in STOPLIKE else None
    return rows

def _log_decisions(rows):
    """Persist WHY each session was treated as completed / dead / user-closed.
    - decisions.log : full human-readable current-state snapshot (overwritten).
    - transitions.log : append-only audit trail of disposition CHANGES across runs."""
    snap = os.path.join(REG_DIR, "decisions.log")
    cat_counts = {}
    for r in rows:
        cat_counts[r["category"]] = cat_counts.get(r["category"], 0) + 1
    with open(snap, "w", encoding="utf-8") as f:
        f.write(f"# fleet session decisions @ {NOW.isoformat()}  ({len(rows)} sessions)\n")
        f.write("# categories: " + "  ".join(f"{k}={v}" for k, v in sorted(cat_counts.items())) + "\n")
        f.write("# age    category project                    disp/action            cause / reason  [sid]\n")
        for r in sorted(rows, key=lambda r: (r["category"], r["age_min"])):
            tag = r["account"].replace(".claude-", "").replace(".claude", "default")
            f.write(f"{r['age_min']:>7}m {r['category']:<8} {r['project']:<26} "
                    f"{r['disp']:<14} {r['action']:<16} {r['cause']:<16} {r['reason']}  "
                    f"[{tag}/{r['session'][:8]}]\n")
    # transitions vs previous snapshot
    prev_path = os.path.join(REG_DIR, "_prev_disp.json")
    prev = {}
    if os.path.exists(prev_path):
        try:
            with open(prev_path, encoding="utf-8") as f:
                prev = json.load(f)
        except (OSError, ValueError):
            prev = {}
    cur = {r["session"]: r["disp"] for r in rows}
    trans = os.path.join(REG_DIR, "transitions.log")
    by_sid = {r["session"]: r for r in rows}
    with open(trans, "a", encoding="utf-8") as f:
        for sid, d in cur.items():
            old = prev.get(sid)
            if old and old != d:
                r = by_sid[sid]
                tag = r["account"].replace(".claude-", "").replace(".claude", "default")
                f.write(f"{NOW.isoformat()}  [{r['category']:<7}] {sid[:8]}  {tag}/{r['project']}  "
                        f"{old} -> {d}  [{r['action']}]  {r['reason']}\n")
    with open(prev_path, "w", encoding="utf-8") as f:
        json.dump(cur, f)

def plan_entry(r):
    """One AUTO_RESUME plan record for the watchdog.

    Carries both the source (where the transcript lives now) and the resume
    target (where it should run). They differ only for a re-homed session;
    the watchdog copies the transcript across before resuming when rehomed."""
    return {"account": r["account"], "config_dir": config_dir(r["account"]),
            "source_config_dir": r["source_config_dir"],
            "resume_account": r["resume_account"],
            "resume_config_dir": r["resume_config_dir"],
            "rehomed": r["rehomed"],
            "session": r["session"], "cwd": r["cwd"], "project": r["project"],
            "disp": r["disp"], "resume_cmd": r["resume_cmd"]}

def write_registry(rows, throttle, auth, probes=None):
    if not os.path.isdir(REG_DIR):
        os.makedirs(REG_DIR, exist_ok=True)
    # synthetic probe rows (project == "_probe") feed the mergers but are NOT real
    # sessions -- keep them out of the sessions list so the operator view stays honest.
    session_rows = [r for r in rows if r.get("project") != "_probe"]
    reg = {"schema": "fleet-sessions/3", "app_version": fleet_version.app_version(), "generated_utc": NOW.isoformat(),
           "window_h": WINDOW_H, "throttle": throttle, "auth": auth,
           "accounts": account_availability(throttle, rows, auth),
            "sessions": [{k: r[k] for k in ("account", "project", "session", "cwd", "git",
                          "category", "cause", "disp", "reason", "action", "autonomous",
                          "supervised", "age_min", "seen_utc", "throttle_reset",
                          "throttle_weekly", "resume_cmd", "rehomed", "resume_account", "last")}
                        for r in session_rows]}
    if probes:
        reg["probes"] = probes  # raw active-probe verdicts (evidence for the operator/UI)
    sessions_path = os.path.join(REG_DIR, "sessions.json")
    plan = [plan_entry(r) for r in rows if r["action"] == "AUTO_RESUME"]
    plan_path = os.path.join(REG_DIR, "resume_plan.json")
    with open(sessions_path, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=1)
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump({"app_version": fleet_version.app_version(), "generated_utc": NOW.isoformat(), "plan": plan}, f, indent=1)
    _log_decisions(rows)
    return sessions_path, plan_path, len(plan)

def print_accounts(throttle, rows, auth):
    """Operator-facing 'which accounts are available right now' block. Shown in
    summary/resume so operator and switcher see the same dynamic blockers."""
    accts = account_availability(throttle, rows, auth)
    avail = [a for a in accts if a["available"]]
    blocked = [a for a in accts if not a["available"]]
    print("ACCOUNTS AVAILABLE NOW (worker, not blocked):")
    print("  " + (", ".join(a["tag"] for a in avail) if avail else "(none - all blocked)"))
    if blocked:
        print("BLOCKED: " + ", ".join(f"{a['tag']} ({a['block_reason']})" for a in blocked))
    print()

def run_probes(rows, selector):
    """Active-probe selected accounts and return (probe_rows, verdicts).

    Probe rows are in classify()'s row shape (via account_probe.verdict_to_row) so the
    existing mergers consume them unchanged. The roster used to pick targets is annotated
    with the throttle/auth derived from THIS scan's rows (not just the stale registry), so
    "blocked" reflects current passive evidence before we spend a probe.
    """
    if selector in ("none", ""):
        return [], []
    try:
        import account_probe  # local import: only paid for when probing
    except ImportError:
        return [], []
    # annotate the roster with the current scan's throttle/auth so target selection is fresh
    throttle = merge_known_throttle({}, rows)
    auth = merge_known_auth(rows)
    registry = {"generated_utc": NOW.isoformat(), "auth": auth, "throttle": throttle}
    annotated = fleet_accounts.annotated_roster(USER, ACCT_POLICY, registry=registry,
                                                throttle=throttle, sessions=rows)
    targets = account_probe.select_targets(annotated, selector=selector,
                                           skip_active_throttle=True,
                                           min_interval_min=PROBE_MIN_INTERVAL,
                                           reg_dir_path=REG_DIR)
    if not targets:
        return [], []
    verdicts = account_probe.probe_accounts(targets)
    # record each probe in the per-account ledger (prev_status -> flip detection), the
    # audit trail the status card reads for RECENT PROBE FLIPS.
    try:
        account_probe.append_probe_ledger(verdicts, REG_DIR)
    except OSError:
        pass
    probe_rows = []
    for v in verdicts:
        pr = account_probe.verdict_to_row(v)
        pr["account"] = v.get("account")
        probe_rows.append(pr)
    return probe_rows, verdicts


def main():
    rows, throttle = scan()
    probe_rows, probe_verdicts = run_probes(rows, PROBE_SELECTOR)
    if probe_rows:
        rows = rows + probe_rows
    throttle = merge_known_throttle(throttle, rows)
    auth = merge_known_auth(rows)
    availability = account_availability(throttle, rows, auth)
    decide(rows, throttle, availability)
    if MODE == "json":
        print(json.dumps({"app_version": fleet_version.app_version(), "now": NOW.isoformat(),
                          "throttle": throttle, "auth": auth,
                          "accounts": account_availability(throttle, rows, auth),
                          "probes": probe_verdicts,
                          "rows": rows}, indent=1))
        return
    if MODE == "registry":
        sp, pp, n = write_registry(rows, throttle, auth, probes=probe_verdicts)
        nsess = len([r for r in rows if r.get("project") != "_probe"])
        probed = f", {len(probe_verdicts)} probed" if probe_verdicts else ""
        print(f"wrote {sp} ({nsess} sessions{probed})")
        print(f"wrote {pp} ({n} AUTO_RESUME)")
        return
    if MODE == "plan":  # machine-readable AUTO_RESUME set for the watchdog
        plan = [plan_entry(r) for r in rows if r["action"] == "AUTO_RESUME"]
        print(json.dumps({"app_version": fleet_version.app_version(), "generated_utc": NOW.isoformat(), "plan": plan}, indent=1))
        return
    if MODE == "resume":
        auto = [r for r in rows if r["action"] == "AUTO_RESUME"]
        surf = [r for r in rows if r["action"] == "SURFACE"]
        print_accounts(throttle, rows, auth)
        print("# AUTO-RESUMABLE (autonomous, dead, account available) — safe to run:\n")
        for r in auto:
            print(f"# [{r['disp']}] {r['project']} ({r['git']})  age={r['age_min']}m")
            print(r["resume_cmd"])
            print()
        if not auto:
            print("# (none right now)\n")
        print("# SURFACE — stopped but interactive; resume only if you mean to:\n")
        for r in surf:
            print(f"# [{r['disp']}] {r['project']} age={r['age_min']}m  -- {r['last'][:70]}")
            print(r["resume_cmd"])
            print()
        return
    # summary (uncapped within window; explicit truncation note)
    print(f"fleet_sessions @ {NOW.strftime('%Y-%m-%d %H:%M')}Z   window={WINDOW_H}h   {len(rows)} sessions\n")
    print_accounts(throttle, rows, auth)
    if throttle:
        print("THROTTLED ACCOUNTS (resume will instantly re-die):")
        for a, t in throttle.items():
            line = f"  {a:<30} resets {t['reset']}"
            weekly = t.get("weekly")
            if weekly:
                line += f"  | weekly {weekly}"
            print(line)
        print()
    order = ["STOPPED_LIMIT", "STOPPED_APIERR", "INFRA_AUTH", "DEAD_MIDTOOL", "DEAD_KILLED",
             "USER_CLOSED", "STOPPED_QUIET", "PARKED_WAIT", "LIVE", "DONE"]
    counts, acts, cats = {}, {}, {}
    for r in rows:
        counts[r["disp"]] = counts.get(r["disp"], 0) + 1
        acts[r["action"]] = acts.get(r["action"], 0) + 1
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    order += [d for d in counts if d not in order]   # any unforeseen disp still shown
    catorder = ["INFRA", "AGENT", "USER", "HANGING", "LIVE"]
    print("category: " + "  ".join(f"{k}={cats[k]}" for k in catorder if cats.get(k)))
    print("disp:     " + "  ".join(f"{k}={counts.get(k,0)}" for k in order if counts.get(k)))
    print("action:   " + "  ".join(f"{k}={acts[k]}" for k in sorted(acts)))
    print()
    CAP = 40
    for disp in order:
        grp = [r for r in rows if r["disp"] == disp]
        if not grp:
            continue
        print(f"== {disp}: {len(grp)} ==")
        for r in grp[:CAP]:
            thr = "  [THROTTLED]" if r["account"] in throttle else ""
            mark = {"AUTO_RESUME": " *AUTO", "SURFACE": " surface", "SUPERVISED": " (sup)",
                    "DEFER_THROTTLED": " defer", "SKIP_DONE": " done",
                    "SKIP_USER_CLOSED": " user-closed"}.get(r["action"], "")
            if r.get("rehomed"):
                rtag = r["resume_account"].replace(".claude-", "").replace(".claude", "default")
                mark += f" -> {rtag}"
            tag = r["account"].replace(".claude-", "").replace(".claude", "default")
            print(f"  {r['age_min']:>6}m  {tag:<18} {r['project']:<26} {r['session'][:8]}{mark}{thr}")
        if len(grp) > CAP:
            print(f"  ... +{len(grp)-CAP} more")
        print()

if __name__ == "__main__":
    main()
